import { createHmac } from 'node:crypto'
import { describe, expect, test } from 'bun:test'
import { createMemoryState, type MemoryStateAdapter } from '@chat-adapter/state-memory'
import { createSlackbotV2 } from '../src/index'
import { openSessionEventStream } from '../src/session-api'
import {
  containsSecretAnswer,
  createSlackUserInputBridge,
  observeUserInputEvents,
  pinSlackSessionRequester
} from '../src/user-input'
import type { SlackbotV2ApiMessage, SlackbotV2Options, SlackbotV2RendererSource } from '../src/types'

const TEAM = 'T1'
const CHANNEL = 'C1'
const USER = 'U1'
const THREAD = '1700000000.000100'
const MESSAGE_TS = '1700000001.000200'
const THREAD_ID = `slack:${CHANNEL}:${THREAD}`
const SIGNING_SECRET = 'fixture-signing-secret'
const QUESTION = {
  id: 'choice', header: 'Deployment', question: 'Which environment?',
  options: [{ label: 'Staging', description: 'Use the staging environment.' }], free_text: true
}
const ORIGIN: SlackbotV2ApiMessage = {
  attachments: [], author: { fullName: 'Test User', isBot: false, isMe: false, userId: USER, userName: 'tester' },
  id: THREAD, isMention: true, raw: {}, teamId: TEAM, text: 'Do the task', threadId: THREAD_ID,
  timestamp: '2026-10-15T00:00:00Z'
}

type Call = { path: string; body: Record<string, unknown> }
type Button = { action_id: string; value: string; text: { text: string } }

function body(init?: RequestInit): Record<string, unknown> {
  if (!init?.body) return {}
  if (new Headers(init.headers).get('content-type')?.includes('application/json')) {
    return JSON.parse(String(init.body))
  }
  const values: Record<string, unknown> = Object.fromEntries(new URLSearchParams(String(init.body)))
  for (const key of ['blocks', 'view']) {
    if (typeof values[key] === 'string') values[key] = JSON.parse(values[key])
  }
  return values
}

function buttons(call: Call): Button[] {
  return ((call.body.blocks ?? []) as Array<{ elements?: Button[] }>)
    .flatMap(block => block.elements ?? [])
    .filter(element => element.action_id)
}

function signed(payload: Record<string, unknown>): RequestInit {
  const timestamp = String(Math.floor(Date.now() / 1000))
  const body = new URLSearchParams({ payload: JSON.stringify(payload) }).toString()
  const signature = createHmac('sha256', SIGNING_SECRET).update(`v0:${timestamp}:${body}`).digest('hex')
  return {
    method: 'POST', body,
    headers: {
      'content-type': 'application/x-www-form-urlencoded',
      'x-slack-request-timestamp': timestamp,
      'x-slack-signature': `v0=${signature}`
    }
  }
}

async function fixture(input: {
  answerStatus?: number
  answerHold?: Promise<unknown>
  answerError?: boolean
  state?: MemoryStateAdapter
  legacy?: boolean
} = {}) {
  const calls: Call[] = []
  const state = input.state ?? createMemoryState()
  await state.connect()
  if (!input.legacy) await pinSlackSessionRequester(state, THREAD_ID, ORIGIN)
  const options: SlackbotV2Options = {
    apiUrl: 'http://api.test', slackApiUrl: 'http://slack.test/api/', botToken: 'fixture-bot-token',
    botUserId: 'UBOT', signingSecret: SIGNING_SECRET, state, recoverRenderObligationsOnStart: false,
    fetch: async (url, init) => {
      const path = new URL(url instanceof Request ? url.url : String(url)).pathname
      calls.push({ path, body: body(init) })
      if (path.includes('/user-input/')) {
        await input.answerHold
        if (input.answerError) throw new TypeError('fixture connection closed')
        return new Response(null, { status: input.answerStatus ?? 204 })
      }
      if (path.endsWith('/chat.postMessage')) return Response.json({ ok: true, ts: MESSAGE_TS })
      if (path.endsWith('/views.open')) return Response.json({ ok: true, view: { id: 'V1' } })
      return Response.json({ ok: true })
    }
  }
  const bot = createSlackbotV2(options)
  const bridge = createSlackUserInputBridge(options, state)
  const context = { executionId: 'exec-1', message: ORIGIN, threadId: THREAD_ID }
  const request = async (questions = [QUESTION]) => {
    await bridge.observe(context, {
      event: 'session.user_input_requested',
      data: { execution_id: 'exec-1', request_id: 'request/1', questions }
    })
    return calls.find(call => call.path.endsWith('/chat.postMessage'))!
  }
  const deliver = async (payload: Record<string, unknown>, invalidSignature = false) => {
    const waits: Promise<unknown>[] = []
    const init = signed(payload)
    if (invalidSignature) (init.headers as Record<string, string>)['x-slack-signature'] = 'v0=invalid'
    const response = await bot.app.request('/api/slack/actions', init, {}, {
      waitUntil(task: Promise<unknown>) { waits.push(task) }, passThroughOnException() {}, props: {}
    })
    return { response, finished: () => Promise.all(waits) }
  }
  const click = (button: Button, overrides: Record<string, unknown> = {}) => ({
    type: 'block_actions', team: { id: TEAM }, user: { id: USER, team_id: TEAM }, channel: { id: CHANNEL },
    message: { ts: MESSAGE_TS, thread_ts: THREAD }, trigger_id: 'fixture-trigger',
    actions: [{ ...button, type: 'button', action_ts: '1700000002.000300' }], ...overrides
  })
  const modalPayload = (value: string, overrides: Record<string, unknown> = {}) => {
    const opened = calls.find(call => call.path.endsWith('/views.open'))!
    const view = opened.body.view as Record<string, unknown>
    return {
      type: 'view_submission', team: { id: TEAM }, user: { id: USER, team_id: TEAM }, trigger_id: 'fixture-submit',
      view: { ...view, id: 'V1', state: { values: { answer: { answer: { type: 'plain_text_input', value } } } } },
      ...overrides
    }
  }
  return { bridge, calls, state, options, context, request, deliver, click, modalPayload }
}

function answerCalls(calls: Call[]): Call[] {
  return calls.filter(call => call.path.includes('/user-input/'))
}

describe('agent questions in Slack', () => {
  test('pins the original requester rather than a later execution participant', async () => {
    const f = await fixture()
    const laterMessage = { ...ORIGIN, author: { ...ORIGIN.author, userId: 'U2' } }
    await pinSlackSessionRequester(f.state, THREAD_ID, laterMessage)
    await f.bridge.observe({ ...f.context, message: laterMessage }, {
      event: 'session.user_input_requested',
      data: { execution_id: 'exec-1', request_id: 'request/1', questions: [QUESTION] }
    })
    const posted = f.calls.find(call => call.path.endsWith('/chat.postMessage'))!
    await (await f.deliver(f.click(buttons(posted)[0]!, { user: { id: 'U2', team_id: TEAM } }))).finished()
    expect(answerCalls(f.calls)).toHaveLength(0)
    await (await f.deliver(f.click(buttons(posted)[0]!))).finished()
    expect(answerCalls(f.calls)).toHaveLength(1)
  })

  test('keeps legacy sessions without a requester pin noninteractive', async () => {
    const f = await fixture({ legacy: true })
    const posted = await f.request()
    expect(buttons(posted)).toHaveLength(0)
    expect(JSON.stringify(posted.body.blocks)).toContain('Start a fresh Slack thread')
    expect(answerCalls(f.calls)).toHaveLength(0)
  })

  test('renders model text only as escaped data, with opaque button values', async () => {
    const f = await fixture()
    const posted = await f.request([{
      ...QUESTION, header: '<!channel>', question: '*act now* <@U2> <https://evil.test|click> & /danger',
      options: [{ label: '<!here>', description: '<@U3> `command`' }]
    }])
    expect(posted.body.thread_ts).toBe(THREAD)
    expect(posted.body.mrkdwn).toBe('false')
    expect(posted.body.parse).toBe('none')
    const rendered = JSON.stringify(posted.body.blocks)
    expect(rendered).not.toContain('<!channel>')
    expect(rendered).not.toContain('<@U2>')
    expect(rendered).toContain('&lt;!channel&gt;')
    expect(rendered).toContain('&lt;https://evil.test|click&gt;')
    expect(rendered).not.toContain('"mrkdwn"')
    expect(buttons(posted).map(button => button.text.text)).toEqual(['&lt;!here&gt;', 'Answer…'])
    for (const button of buttons(posted)) {
      expect(button.value).toMatch(/^[a-f0-9-]{36}$/)
      expect(button.action_id).not.toContain('evil')
    }
    const correlation = await f.state.get<Record<string, unknown>>(`slackbotv2:user-input:card:${buttons(posted)[0]!.value}`)
    expect(correlation).toMatchObject({
      session: THREAD_ID, execution: 'exec-1', request: 'request/1', team: TEAM,
      channel: CHANNEL, threadTs: THREAD, messageTs: MESSAGE_TS, permittedUser: USER
    })
  })

  test('verifies the Slack signature before accepting either interaction type', async () => {
    const f = await fixture()
    const posted = await f.request()
    const denied = await f.deliver(f.click(buttons(posted)[0]!), true)
    expect(denied.response.status).toBe(401)
    await denied.finished()
    const open = await f.deliver(f.click(buttons(posted)[1]!))
    await open.finished()
    const deniedModal = await f.deliver(f.modalPayload('Use staging'), true)
    expect(deniedModal.response.status).toBe(401)
    await deniedModal.finished()
    expect(answerCalls(f.calls)).toHaveLength(0)
  })

  test('refuses a different user without consuming the permitted user click', async () => {
    const f = await fixture()
    const posted = await f.request()
    const button = buttons(posted)[0]!
    const denied = await f.deliver(f.click(button, { user: { id: 'U2', team_id: TEAM } }))
    await denied.finished()
    expect(answerCalls(f.calls)).toHaveLength(0)
    expect(f.calls.at(-1)?.body).toMatchObject({ user: 'U2', thread_ts: THREAD })
    expect(f.calls.at(-1)?.body.text).toContain('Only the user who requested')
    const accepted = await f.deliver(f.click(button))
    await accepted.finished()
    expect(answerCalls(f.calls)).toEqual([{
      path: '/api/session/slack%3AC1%3A1700000000.000100/executions/exec-1/user-input/request%2F1',
      body: { answers: { choice: { answers: ['Staging'] } } }
    }])
    const updated = f.calls.at(-1)!
    expect(updated.path).toEndWith('/chat.update')
    expect(JSON.stringify(updated.body.blocks)).toContain('Chosen answer: Deployment: Staging')
    expect(buttons(updated)).toHaveLength(0)
  })

  test.each([
    { team: { id: 'T2' }, user: { id: USER, team_id: 'T2' } },
    { channel: { id: 'C2' } },
    { message: { ts: MESSAGE_TS, thread_ts: '1700000000.999999' } },
    { message: { ts: '1700000001.999999', thread_ts: THREAD } }
  ])('refuses a mismatched workspace/channel/thread/message binding: %j', async overrides => {
    const f = await fixture()
    const posted = await f.request()
    const delivery = await f.deliver(f.click(buttons(posted)[0]!, overrides))
    await delivery.finished()
    expect(answerCalls(f.calls)).toHaveLength(0)
    expect(f.calls.at(-1)?.path).toEndWith('/chat.postEphemeral')
    expect(f.calls.at(-1)?.body.text).toContain('does not belong')
  })

  test('applies concurrent clicks once and keeps the claim across process redelivery', async () => {
    const hold = Promise.withResolvers<void>()
    const f = await fixture({ answerHold: hold.promise })
    const posted = await f.request()
    const payload = f.click(buttons(posted)[0]!)
    const first = await f.deliver(payload)
    // The webhook acknowledges before the answer HTTP operation completes.
    expect(first.response.status).toBe(200)
    const second = await f.deliver(payload)
    await second.finished()
    hold.resolve()
    await first.finished()
    expect(answerCalls(f.calls)).toHaveLength(1)
    const resumed = await fixture({ state: f.state })
    const redelivery = await resumed.deliver(payload)
    await redelivery.finished()
    expect(answerCalls(resumed.calls)).toHaveLength(0)
    expect(resumed.calls.at(-1)?.body.text).toContain('already answered')
    await f.request()
    expect(f.calls.filter(call => call.path.endsWith('/chat.postMessage'))).toHaveLength(1)
  })

  test.each([404, 409])('marks stale API responses terminal without retrying (%i)', async answerStatus => {
    const f = await fixture({ answerStatus })
    const posted = await f.request()
    const payload = f.click(buttons(posted)[0]!)
    await (await f.deliver(payload)).finished()
    await (await f.deliver(payload)).finished()
    expect(answerCalls(f.calls)).toHaveLength(1)
    const update = f.calls.find(call => call.path.endsWith('/chat.update'))!
    expect(update.body.text).toContain('already answered / expired')
    expect(buttons(update)).toHaveLength(0)
  })

  test('does not retry an ambiguous answer delivery', async () => {
    const f = await fixture({ answerError: true })
    const posted = await f.request()
    const payload = f.click(buttons(posted)[0]!)
    await (await f.deliver(payload)).finished()
    await (await f.deliver(payload)).finished()
    expect(answerCalls(f.calls)).toHaveLength(1)
    const update = f.calls.find(call => call.path.endsWith('/chat.update'))!
    expect(update.body.text).toContain('could not be confirmed')
    expect(buttons(update)).toHaveLength(0)
  })

  test.each(['answered', 'cancelled', 'expired', 'resolved', 'delivery_failed'])('resolves the card before a late click (%s)', async status => {
    const f = await fixture()
    const posted = await f.request()
    await f.bridge.observe(f.context, {
      event: 'session.user_input_resolved', data: { execution_id: 'exec-1', request_id: 'request/1', status, reason: null }
    })
    const terminal = f.calls.at(-1)!
    expect(buttons(terminal)).toHaveLength(0)
    if (status === 'delivery_failed') expect(terminal.body.text).toContain('fresh agent question')
    await (await f.deliver(f.click(buttons(posted)[0]!))).finished()
    expect(answerCalls(f.calls)).toHaveLength(0)
    expect(f.calls.at(-1)?.body.text).toContain('already answered')
  })

  test.each(['session.execution_completed', 'session.execution_cancelled', 'session.execution_failed'])('expires pending cards on execution termination (%s)', async event => {
    const f = await fixture()
    const posted = await f.request()
    await f.bridge.observe(f.context, { event, data: { execution_id: 'exec-1' } })
    expect(buttons(f.calls.at(-1)!)).toHaveLength(0)
    await (await f.deliver(f.click(buttons(posted)[0]!))).finished()
    expect(answerCalls(f.calls)).toHaveLength(0)
  })

  test('opens one free-text modal on redelivery and submits through its saved binding', async () => {
    const f = await fixture()
    const posted = await f.request()
    const click = f.click(buttons(posted)[1]!)
    await (await f.deliver(click)).finished()
    await (await f.deliver(click)).finished()
    expect(f.calls.filter(call => call.path.endsWith('/views.open'))).toHaveLength(1)
    const payload = f.modalPayload('Use the preconfigured staging environment.')
    const submission = await f.deliver(payload)
    expect(submission.response.status).toBe(200)
    await submission.finished()
    await (await f.deliver(payload)).finished()
    expect(answerCalls(f.calls)).toHaveLength(1)
    expect(answerCalls(f.calls)[0]?.body).toEqual({
      answers: { choice: { answers: ['Use the preconfigured staging environment.'] } }
    })
    expect(buttons(f.calls.find(call => call.path.endsWith('/chat.update'))!)).toHaveLength(0)
  })

  test('refuses a modal submitted by a different user or from a different view/workspace', async () => {
    const f = await fixture()
    const posted = await f.request()
    await (await f.deliver(f.click(buttons(posted)[1]!))).finished()
    for (const overrides of [
      { user: { id: 'U2', team_id: TEAM } },
      { team: { id: 'T2' }, user: { id: USER, team_id: 'T2' } },
      { view: { ...(f.modalPayload('ok').view as object), id: 'V2' } }
    ]) {
      const delivery = await f.deliver(f.modalPayload('Use staging', overrides))
      expect(await delivery.response.json()).toMatchObject({ response_action: 'errors' })
      await delivery.finished()
    }
    expect(answerCalls(f.calls)).toHaveLength(0)
  })

  test('a modal submitted after resolution is a no-op with an ephemeral note', async () => {
    const f = await fixture()
    const posted = await f.request()
    await (await f.deliver(f.click(buttons(posted)[1]!))).finished()
    await f.bridge.observe(f.context, {
      event: 'session.user_input_resolved', data: { execution_id: 'exec-1', request_id: 'request/1', status: 'expired' }
    })
    const delivery = await f.deliver(f.modalPayload('Too late'))
    expect(delivery.response.status).toBe(200)
    await delivery.finished()
    expect(answerCalls(f.calls)).toHaveLength(0)
    expect(f.calls.at(-1)?.path).toEndWith('/chat.postEphemeral')
  })

  test('refuses secret-pattern options and modal input without collecting or sending an answer', async () => {
    const f = await fixture()
    const tokenLike = 'sk-' + 'a'.repeat(40)
    const posted = await f.request([{ ...QUESTION, options: [{ label: tokenLike, description: 'Synthetic unsafe option.' }] }])
    await (await f.deliver(f.click(buttons(posted)[0]!))).finished()
    expect(f.calls.at(-1)?.body.text).toContain('Do not send tokens')
    await (await f.deliver(f.click(buttons(posted)[1]!))).finished()
    const delivery = await f.deliver(f.modalPayload(tokenLike))
    expect(await delivery.response.json()).toMatchObject({ response_action: 'errors', errors: { answer: expect.stringContaining('Do not send tokens') } })
    await delivery.finished()
    expect(answerCalls(f.calls)).toHaveLength(0)
    const card = await f.state.get<{ answers: unknown }>(`slackbotv2:user-input:card:${buttons(posted)[0]!.value}`)
    expect(card?.answers).toEqual({})
  })

  test.each([
    'xoxb-' + 'a'.repeat(32), 'ghp_' + 'a'.repeat(36), 'github_pat_' + 'a'.repeat(40),
    'AKIA' + 'A'.repeat(16), 'eyJhbGci.eyJzdWIi.signature', '-----BEGIN PRIVATE KEY-----',
    'Bearer ' + 'a'.repeat(32), 'password=' + 'a'.repeat(12), 'a'.repeat(64),
    'AbCdEfGh1234567890AbCdEfGh1234567890'
  ])('recognizes synthetic token-like text without logging its value', value => {
    expect(containsSecretAnswer(value)).toBe(true)
    expect(containsSecretAnswer('Use the staging environment.')).toBe(false)
  })

  test('pages options in one card and collects all questions before one answer POST', async () => {
    const f = await fixture()
    const choices = Array.from({ length: 23 }, (_, index) => ({ label: `Option ${index}`, description: 'A possible choice.' }))
    const posted = await f.request([
      { ...QUESTION, options: choices, free_text: false },
      { ...QUESTION, id: 'confirm', header: 'Confirmation', options: [{ label: 'Proceed', description: 'Continue.' }] }
    ])
    expect(buttons(posted).filter(button => button.action_id.includes(':option:'))).toHaveLength(20)
    const more = buttons(posted).find(button => button.text.text === 'More options')!
    await (await f.deliver(f.click(more))).finished()
    const page = f.calls.at(-1)!
    expect(buttons(page).map(button => button.text.text)).toContain('Option 22')
    await (await f.deliver(f.click(buttons(page).find(button => button.text.text === 'Option 22')!))).finished()
    expect(answerCalls(f.calls)).toHaveLength(0)
    const nextQuestion = f.calls.at(-1)!
    expect(JSON.stringify(nextQuestion.body.blocks)).toContain('Question 2 of 2')
    await (await f.deliver(f.click(buttons(nextQuestion)[0]!))).finished()
    expect(answerCalls(f.calls)).toHaveLength(1)
    expect(answerCalls(f.calls)[0]?.body).toEqual({
      answers: { choice: { answers: ['Option 22'] }, confirm: { answers: ['Proceed'] } }
    })
    expect(f.calls.filter(call => call.path.endsWith('/chat.postMessage'))).toHaveLength(1)
  })

  test('pages long question data within Slack limits instead of dropping its tail', async () => {
    const f = await fixture()
    const posted = await f.request([{ ...QUESTION, question: '<'.repeat(4000) + 'Question tail.' }])
    const rendered = (posted.body.blocks as Array<{ text?: { text: string } }>)
    for (const block of rendered) {
      if (block.text) expect(block.text.text.length).toBeLessThanOrEqual(3000)
    }
    expect(JSON.stringify(rendered)).not.toContain('Question tail.')
    const more = buttons(posted).find(button => button.text.text === 'More question text')!
    await (await f.deliver(f.click(more))).finished()
    expect(JSON.stringify(f.calls.at(-1)?.body.blocks)).toContain('Question tail.')
    expect((f.calls.at(-1)?.body.blocks as unknown[]).length).toBeLessThanOrEqual(50)
    expect(answerCalls(f.calls)).toHaveLength(0)
  })

  test('passes replayed request/resolution SSE through the service consumer, not the shared text renderer', async () => {
    const f = await fixture()
    const events = [
      { event: 'session.user_input_requested', data: { execution_id: 'exec-1', request_id: 'request/1', questions: [QUESTION] } },
      { event: 'session.user_input_resolved', data: { execution_id: 'exec-1', request_id: 'request/1', status: 'cancelled', reason: 'execution_cancelled' } },
      { event: 'session.execution_completed', data: { execution_id: 'exec-1' } }
    ]
    const options = { ...f.options, fetch: async (url: RequestInfo | URL, init?: RequestInit) => {
      if (String(url).includes('/events?')) {
        return new Response(events.map((event, index) => `id: ${index + 1}\nevent: ${event.event}\ndata: ${JSON.stringify(event.data)}\n\n`).join(''), {
          headers: { 'content-type': 'text/event-stream' }
        })
      }
      return f.options.fetch!(url, init)
    } }
    const ids: number[] = []
    const stream = await openSessionEventStream(options, {
      afterEventId: 0, executionId: 'exec-1', onEventId: id => { ids.push(id) }, threadId: THREAD_ID
    })
    const visible: SlackbotV2RendererSource[] = []
    for await (const event of observeUserInputEvents(options, f.context, stream)) visible.push(event)
    expect(ids).toEqual([1, 2, 3])
    expect(visible.map(event => event.event)).toEqual(['session.execution_completed'])
    expect(f.calls.filter(call => call.path.endsWith('/chat.postMessage'))).toHaveLength(1)
    expect(f.calls.find(call => call.path.endsWith('/chat.update'))?.body.text).toBe('Question cancelled.')
  })

  test('closes a request on a terminal harness output line before the SSE consumer stops', async () => {
    const f = await fixture()
    await f.request()
    await f.bridge.observe(f.context, { event: 'session.output.line', data: JSON.stringify({ method: 'turn/completed' }) })
    expect(f.calls.at(-1)?.body.text).toContain('expired')
    expect(buttons(f.calls.at(-1)!)).toHaveLength(0)
  })
})
