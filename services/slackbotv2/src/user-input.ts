import { createHash, randomUUID } from 'node:crypto'
import { assertSlackOk, callSlackApi } from '@chat-adapter/slack/api'
import type { StateAdapter } from 'chat'
import { answerSessionUserInput, slackApiTimeoutMs, withSlackApiTimeout } from './session-api'
import type { SlackbotV2ApiMessage, SlackbotV2Options, SlackbotV2RendererSource } from './types'
import { isJsonObject, stringValue } from './utils'

const ACTION_PREFIX = 'centaur.user-input:'
const MODAL_CALLBACK = 'centaur.user-input.answer'
const STATE_TTL_MS = 30 * 24 * 60 * 60 * 1000
const CLICK_TTL_MS = 24 * 60 * 60 * 1000
const OPTIONS_PER_PAGE = 20
const SECRET_HINT = 'Do not send tokens, passwords, or private keys. Use a placeholder or a configured credential instead.'
const CLOSED_NOTE = 'This question was already answered or has expired.'

type Question = {
  id: string
  header: string
  question: string
  options: Array<{ label: string; description: string }>
  free_text: boolean
}

type CardStatus = 'pending' | 'submitting' | 'answered' | 'cancelled' | 'resolved'
  | 'expired' | 'delivery_failed' | 'unconfirmed' | 'unavailable'

type RequesterBinding = { team: string; user: string }

type Card = {
  correlationId: string
  session: string
  execution: string
  request: string
  team: string
  channel: string
  threadTs: string
  messageTs?: string
  permittedUser: string
  questions: Question[]
  answers: Record<string, { answers: string[] }>
  page: number
  textPage: number
  status: CardStatus
  lastAnswer?: string
}

type ModalBinding = {
  correlationId: string
  questionIndex: number
  viewId?: string
}

type StreamContext = {
  executionId?: string
  message?: SlackbotV2ApiMessage
  threadId: string
}

type Schedule = (task: Promise<unknown>) => void

type SlackBody = Record<string, unknown>

function digest(parts: string[]): string {
  return createHash('sha256').update(JSON.stringify(parts)).digest('hex')
}

function requestKey(session: string, execution: string, request: string): string {
  return `slackbotv2:user-input:request:${digest([session, execution, request])}`
}

function cardKey(id: string): string {
  return `slackbotv2:user-input:card:${id}`
}

function executionKey(session: string, execution: string): string {
  return `slackbotv2:user-input:execution:${digest([session, execution])}`
}

export async function pinSlackSessionRequester(
  state: StateAdapter,
  session: string,
  message: SlackbotV2ApiMessage
): Promise<void> {
  if (!message.teamId || !/^[UW][A-Z0-9]+$/.test(message.author.userId)
    || message.author.isBot === true) return
  await state.setIfNotExists(`slackbotv2:user-input:requester:${digest([session])}`, {
    team: message.teamId, user: message.author.userId
  } satisfies RequesterBinding)
}

function object(value: unknown): Record<string, unknown> {
  return isJsonObject(value) ? value : {}
}

// Model-supplied text is data, never Slack mrkdwn, mentions, or link syntax.
function plain(text: string, max = 3000): SlackBody {
  return {
    type: 'plain_text',
    text: text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').slice(0, max) || ' ',
    emoji: false
  }
}

function section(text: string): SlackBody {
  return { type: 'section', text: plain(text) }
}

function button(label: string, action: string, correlationId: string): SlackBody {
  return { type: 'button', text: plain(label, 75), action_id: ACTION_PREFIX + action, value: correlationId }
}

function currentQuestion(card: Card): number {
  return card.questions.findIndex(question => !Object.hasOwn(card.answers, question.id))
}

function statusText(card: Card): string {
  switch (card.status) {
    case 'pending': return `Answers collected: ${Object.keys(card.answers).length} of ${card.questions.length}.`
    case 'answered': return 'Answered.'
    case 'submitting': return 'Answer submitted; awaiting confirmation.'
    case 'cancelled': return 'Question cancelled.'
    case 'resolved': return 'Question already resolved.'
    case 'expired': return 'Question already answered / expired.'
    case 'delivery_failed': return 'Answer could not reach the agent. A fresh agent question is needed.'
    case 'unconfirmed': return 'Answer delivery could not be confirmed. This card cannot be submitted again; a fresh agent question is needed.'
    case 'unavailable': return 'The original requesting user is not recorded for this session. Start a fresh Slack thread to answer agent questions.'
  }
}

function cardBlocks(card: Card): SlackBody[] {
  const blocks = [section(statusText(card))]
  if (card.lastAnswer) blocks.push(section(`Chosen answer: ${card.lastAnswer}`))
  const questionIndex = currentQuestion(card)
  if (card.status !== 'pending' || questionIndex < 0) return blocks
  const question = card.questions[questionIndex]!
  const questionText = `Question ${questionIndex + 1} of ${card.questions.length}\n${question.header}\n${question.question}`
  const textStart = card.textPage * 2500
  const visibleText = questionText.slice(textStart, textStart + 2500)
  // A raw character may expand to six escaped characters. These chunks stay
  // under 3000 even for a question made entirely of markup delimiters.
  for (let start = 0; start < visibleText.length; start += 500) {
    blocks.push(section(visibleText.slice(start, start + 500)))
  }
  const start = card.page * OPTIONS_PER_PAGE
  const options = question.options.slice(start, start + OPTIONS_PER_PAGE)
  // Page one question at a time, and at most 20 options, to stay below Slack's
  // 50-block/25-action limits without dropping any questions or answer values.
  for (let offset = 0; offset < options.length; offset += 5) {
    const group = options.slice(offset, offset + 5)
    blocks.push(section(group.map(option => `${option.label.slice(0, 100)}: ${option.description.slice(0, 400)}`).join('\n')))
    blocks.push({
      type: 'actions',
      elements: group.map((option, index) => button(option.label, `option:${questionIndex}:${start + offset + index}`, card.correlationId))
    })
  }
  const controls: SlackBody[] = []
  if (card.textPage > 0) {
    controls.push(button('Previous question text', `question-page:${questionIndex}:${card.textPage - 1}`, card.correlationId))
  }
  if (textStart + 2500 < questionText.length) {
    controls.push(button('More question text', `question-page:${questionIndex}:${card.textPage + 1}`, card.correlationId))
  }
  if (card.page > 0) controls.push(button('Previous options', `page:${card.page - 1}`, card.correlationId))
  if (start + OPTIONS_PER_PAGE < question.options.length) {
    controls.push(button('More options', `page:${card.page + 1}`, card.correlationId))
  }
  if (question.free_text) controls.push(button('Answer…', `text:${questionIndex}`, card.correlationId))
  if (controls.length) blocks.push({ type: 'actions', elements: controls })
  return blocks
}

function parseQuestions(value: unknown): Question[] {
  if (!Array.isArray(value) || !value.length) throw new Error('Invalid user-input questions')
  const ids = new Set<string>()
  return value.map(value => {
    const question = object(value)
    if (typeof question.id !== 'string' || !question.id || ids.has(question.id)
      || typeof question.header !== 'string' || typeof question.question !== 'string'
      || typeof question.free_text !== 'boolean' || !Array.isArray(question.options)) {
      throw new Error('Invalid user-input question')
    }
    ids.add(question.id)
    const options = question.options.map(value => {
      const option = object(value)
      if (typeof option.label !== 'string' || !option.label || typeof option.description !== 'string') {
        throw new Error('Invalid user-input option')
      }
      return { label: option.label, description: option.description }
    })
    return { id: question.id, header: question.header, question: question.question, free_text: question.free_text, options }
  })
}

export function containsSecretAnswer(value: string): boolean {
  return /\b(?:sk-(?:proj-|ant-)?[a-z0-9_-]{16,}|xox[baprs]-[a-z0-9-]{10,}|gh[pousr]_[a-z0-9]{20,}|github_pat_[a-z0-9_]{20,}|(?:AKIA|ASIA)[A-Z0-9]{16})\b/i.test(value)
    || /\beyJ[a-z0-9_-]+\.[a-z0-9_-]+\.[a-z0-9_-]+\b/i.test(value)
    || /-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----/.test(value)
    || /\bBearer\s+[a-z0-9._~+\/-]{16,}/i.test(value)
    || /\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*["']?[^\s"']{8,}/i.test(value)
    || /\b[a-f0-9]{32,}\b/i.test(value)
    || /\b(?=[A-Za-z0-9_+\/-]{32,}\b)(?=[A-Za-z0-9_+\/-]*[A-Z])(?=[A-Za-z0-9_+\/-]*[a-z])(?=[A-Za-z0-9_+\/-]*[0-9])[A-Za-z0-9_+\/-]+\b/.test(value)
}

export function isUserInputInteraction(payload: Record<string, unknown> | null): boolean {
  if (!payload) return false
  if (payload.type === 'view_submission') return object(payload.view).callback_id === MODAL_CALLBACK
  return payload.type === 'block_actions' && Array.isArray(payload.actions)
    && payload.actions.some(action => stringValue(object(action).action_id)?.startsWith(ACTION_PREFIX))
}

export function createSlackUserInputBridge(options: SlackbotV2Options, state: StateAdapter) {
  const leaseMs = 60_000 + 4 * slackApiTimeoutMs(options) + (options.sessionApiTimeoutMs ?? 30_000)

  async function slack(method: string, body: SlackBody) {
    const response = await withSlackApiTimeout(options, method, () => callSlackApi(method, body, {
      apiUrl: options.slackApiUrl,
      fetch: options.fetch as typeof globalThis.fetch | undefined,
      token: options.botToken
    }))
    assertSlackOk(method, response)
    return response
  }

  async function save(card: Card): Promise<void> {
    await state.set(cardKey(card.correlationId), card, STATE_TTL_MS)
  }

  async function update(card: Card): Promise<void> {
    if (!card.messageTs) return
    await slack('chat.update', {
      channel: card.channel, ts: card.messageTs, text: statusText(card),
      mrkdwn: false, parse: 'none', blocks: cardBlocks(card)
    })
  }

  async function note(channel: string | undefined, user: string | undefined, threadTs: string | undefined, text: string) {
    if (!channel || !user) return
    await slack('chat.postEphemeral', {
      channel, user, ...(threadTs ? { thread_ts: threadTs } : {}), text,
      mrkdwn: false, parse: 'none'
    })
  }

  async function locked<T>(id: string, wait: boolean, fn: () => Promise<T>): Promise<T | undefined> {
    let lock = await state.acquireLock(`slackbotv2:user-input:lock:${id}`, leaseMs)
    while (!lock && wait) {
      const { promise, resolve } = Promise.withResolvers<void>()
      setTimeout(resolve, 100)
      await promise
      lock = await state.acquireLock(`slackbotv2:user-input:lock:${id}`, leaseMs)
    }
    if (!lock) return undefined
    try {
      return await fn()
    } finally {
      await state.releaseLock(lock)
    }
  }

  async function render(context: StreamContext, data: Record<string, unknown>) {
    const execution = stringValue(data.execution_id)
    const request = stringValue(data.request_id)
    const message = context.message
    if (!execution || !request || !message || execution !== context.executionId) return
    const destination = /^slack:([^:]+):([^:]+)$/.exec(context.threadId)
    if (!destination || !message.teamId) return
    const requester = await state.get<RequesterBinding>(
      `slackbotv2:user-input:requester:${digest([context.threadId])}`
    )
    const key = requestKey(context.threadId, execution, request)
    await state.setIfNotExists(key, randomUUID(), STATE_TTL_MS)
    const id = await state.get<string>(key)
    if (!id) throw new Error('User-input correlation could not be persisted')
    await locked(id, true, async () => {
      let card = await state.get<Card>(cardKey(id))
      if (!card) {
        card = {
          correlationId: id, session: context.threadId, execution, request,
          team: message.teamId, channel: destination[1]!, threadTs: destination[2]!,
          permittedUser: requester?.team === message.teamId ? requester.user : '',
          questions: parseQuestions(data.questions), answers: {}, page: 0, textPage: 0,
          status: requester?.team === message.teamId ? 'pending' : 'unavailable'
        }
        await save(card)
      }
      // Re-append on replay to repair a crash between saving and indexing.
      await state.appendToList(executionKey(card.session, card.execution), id, { ttlMs: STATE_TTL_MS })
      if (card.messageTs) {
        if (card.status !== 'pending') await update(card)
        return
      }
      const posted = await slack('chat.postMessage', {
        channel: card.channel, thread_ts: card.threadTs,
        client_msg_id: card.correlationId, text: 'The agent has a question. Only the requesting user can answer.',
        mrkdwn: false, parse: 'none', unfurl_links: false, unfurl_media: false, blocks: cardBlocks(card)
      })
      const ts = stringValue(posted.ts)
      if (!ts) throw new Error('Slack did not return a user-input message timestamp')
      card.messageTs = ts
      await save(card)
    })
  }

  async function resolve(id: string, status: CardStatus) {
    await locked(id, true, async () => {
      const card = await state.get<Card>(cardKey(id))
      if (!card) return
      // Execution completion only closes requests still waiting for an answer;
      // an authoritative request resolution can refine a submitted answer.
      if (status !== 'expired' || card.status === 'pending' || card.status === 'submitting') {
        card.status = status
        await save(card)
      }
      await update(card)
    })
  }

  async function submit(card: Card, questionIndex: number, answer: string) {
    const question = card.questions[questionIndex]!
    if (containsSecretAnswer(answer)) {
      await note(card.channel, card.permittedUser, card.threadTs, SECRET_HINT)
      return
    }
    card.answers = { ...card.answers, [question.id]: { answers: [answer] } }
    card.lastAnswer = `${question.header}: ${answer}`
    card.page = 0
    card.textPage = 0
    if (currentQuestion(card) >= 0) {
      await save(card)
      await update(card)
      return
    }
    // This claim is intentionally not released on an ambiguous HTTP failure.
    // Never issue the answer POST twice, even after a process restart or a
    // different Slack click. Durable api-rs resolution is authoritative.
    const submissionKey = `slackbotv2:user-input:submitted:${digest([card.session, card.execution, card.request])}`
    if (!(await state.setIfNotExists(submissionKey, true))) {
      card.status = 'expired'
      await save(card)
      await update(card)
      await note(card.channel, card.permittedUser, card.threadTs, CLOSED_NOTE)
      return
    }
    card.status = 'submitting'
    await save(card)
    try {
      const status = await answerSessionUserInput(options, card.session, card.execution, card.request, card.answers)
      card.status = status === 204 ? 'answered' : 'expired'
    } catch {
      card.status = 'unconfirmed'
    }
    await save(card)
    await update(card)
  }

  async function action(payload: Record<string, unknown>) {
    const clicked = Array.isArray(payload.actions) ? payload.actions.filter(value => stringValue(object(value).action_id)?.startsWith(ACTION_PREFIX)) : []
    const user = stringValue(object(payload.user).id)
    const channel = stringValue(object(payload.channel).id) ?? stringValue(object(payload.container).channel_id)
    const messageTs = stringValue(object(payload.message).ts) ?? stringValue(object(payload.container).message_ts)
    const threadTs = stringValue(object(payload.message).thread_ts) ?? stringValue(object(payload.container).thread_ts) ?? messageTs
    const team = stringValue(object(payload.team).id)
    if (clicked.length !== 1) return
    const click = object(clicked[0])
    const id = stringValue(click.value)
    const card = id ? await state.get<Card>(cardKey(id)) : null
    if (!card) return note(channel, user, threadTs, CLOSED_NOTE)
    if (card.team !== team || card.channel !== channel || card.threadTs !== threadTs || card.messageTs !== messageTs) {
      return note(channel, user, threadTs, 'This question does not belong to this workspace or thread.')
    }
    if (card.permittedUser !== user) return note(channel, user, threadTs, 'Only the user who requested this session can answer this question.')
    const identity = stringValue(click.action_ts) ?? stringValue(payload.trigger_id)
    const actionId = stringValue(click.action_id)!
    if (!identity) return note(channel, user, threadTs, 'This interaction is missing its Slack click identity.')
    const handled = await locked(card.correlationId, false, async () => {
      const latest = await state.get<Card>(cardKey(card.correlationId))
      if (!latest || latest.status !== 'pending') {
        await note(channel, user, threadTs, CLOSED_NOTE)
        return true
      }
      const key = `slackbotv2:user-input:click:${digest([team!, channel!, messageTs!, user!, actionId, identity])}`
      if (!(await state.setIfNotExists(key, true, CLICK_TTL_MS))) return true
      const command = actionId.slice(ACTION_PREFIX.length)
      const questionIndex = currentQuestion(latest)
      const question = latest.questions[questionIndex]
      if (!question) return true
      const questionPage = /^question-page:(\d+):(\d+)$/.exec(command)
      if (questionPage && Number(questionPage[1]) === questionIndex) {
        const index = Number(questionPage[2])
        const length = `Question ${questionIndex + 1} of ${latest.questions.length}\n${question.header}\n${question.question}`.length
        if (index * 2500 < length) {
          latest.textPage = index
          await save(latest)
          await update(latest)
        }
        return true
      }
      const page = /^page:(\d+)$/.exec(command)
      if (page) {
        const index = Number(page[1])
        if (index * OPTIONS_PER_PAGE < question.options.length) {
          latest.page = index
          await save(latest)
          await update(latest)
        }
        return true
      }
      const option = /^option:(\d+):(\d+)$/.exec(command)
      if (option && Number(option[1]) === questionIndex) {
        const selected = question.options[Number(option[2])]
        if (selected) await submit(latest, questionIndex, selected.label)
        return true
      }
      const text = /^text:(\d+)$/.exec(command)
      if (text && Number(text[1]) === questionIndex && question.free_text && stringValue(payload.trigger_id)) {
        const modalId = randomUUID()
        const binding: ModalBinding = { correlationId: latest.correlationId, questionIndex }
        await state.set(`slackbotv2:user-input:modal:${modalId}`, binding, STATE_TTL_MS)
        const opened = await slack('views.open', {
          trigger_id: payload.trigger_id,
          view: {
            type: 'modal', callback_id: MODAL_CALLBACK, private_metadata: modalId,
            title: plain('Answer agent question', 24), submit: plain('Answer'), close: plain('Close'),
            blocks: [section(`${question.header}\n${question.question}`), {
              type: 'input', block_id: 'answer', label: plain('Answer (no secrets)', 2000),
              element: { type: 'plain_text_input', action_id: 'answer', multiline: true, max_length: 3000 }
            }]
          }
        })
        const viewId = stringValue(object(opened.view).id)
        if (!viewId) throw new Error('Slack did not return a user-input modal identity')
        binding.viewId = viewId
        await state.set(`slackbotv2:user-input:modal:${modalId}`, binding, STATE_TTL_MS)
        return true
      }
      await note(channel, user, threadTs, 'This question has already advanced. Use the current answer buttons.')
      return true
    })
    if (!handled) await note(channel, user, threadTs, 'This question is being updated. Please use the current card.')
  }

  async function modal(payload: Record<string, unknown>, schedule: Schedule): Promise<Response> {
    const view = object(payload.view)
    const modalId = stringValue(view.private_metadata)
    const binding = modalId ? await state.get<ModalBinding>(`slackbotv2:user-input:modal:${modalId}`) : null
    const card = binding ? await state.get<Card>(cardKey(binding.correlationId)) : null
    const user = stringValue(object(payload.user).id)
    const fail = (hint: string) => Response.json({ response_action: 'errors', errors: { answer: hint } })
    if (!binding || !card || !binding.viewId || binding.viewId !== view.id
      || card.team !== stringValue(object(payload.team).id)) return fail('This question does not belong to this workspace or view.')
    if (card.permittedUser !== user) {
      schedule(note(card.channel, user, card.threadTs, 'Only the user who requested this session can answer this question.'))
      return fail('Only the requesting user can answer.')
    }
    if (card.status !== 'pending' || currentQuestion(card) !== binding.questionIndex) {
      schedule(note(card.channel, user, card.threadTs, CLOSED_NOTE))
      return new Response('')
    }
    const values = object(object(view.state).values)
    const answer = stringValue(object(object(values.answer).answer).value)?.trim()
    if (!answer) return fail('Enter an answer.')
    if (containsSecretAnswer(answer)) return fail(SECRET_HINT)
    const trigger = stringValue(payload.trigger_id)
    if (!trigger) return fail('This interaction is missing its Slack submission identity.')
    schedule((async () => {
      const handled = await locked(card.correlationId, false, async () => {
        const latest = await state.get<Card>(cardKey(card.correlationId))
        if (!latest || latest.status !== 'pending' || currentQuestion(latest) !== binding.questionIndex) {
          await note(card.channel, user, card.threadTs, CLOSED_NOTE)
          return true
        }
        const question = latest.questions[binding.questionIndex]
        if (!question?.free_text) return true
        const key = `slackbotv2:user-input:click:${digest([card.team, binding.viewId!, user!, trigger])}`
        if (!(await state.setIfNotExists(key, true, CLICK_TTL_MS))) return true
        await submit(latest, binding.questionIndex, answer)
        return true
      })
      if (!handled) await note(card.channel, user, card.threadTs, 'This question is being updated. Please use the current card.')
    })())
    return new Response('')
  }

  return {
    async handleInteraction(payload: Record<string, unknown>, schedule: Schedule): Promise<Response> {
      if (payload.type === 'view_submission') return modal(payload, schedule)
      schedule(action(payload))
      return new Response('')
    },
    async observe(context: StreamContext, event: SlackbotV2RendererSource): Promise<void> {
      const kind = stringValue(event.eventKind) ?? stringValue(event.event)
      const data = object(event.data)
      if (kind === 'session.user_input_requested') return render(context, data)
      if (kind === 'session.user_input_resolved') {
        const execution = stringValue(data.execution_id)
        const request = stringValue(data.request_id)
        const status = stringValue(data.status)
        if (!execution || execution !== context.executionId || !request
          || !['answered', 'cancelled', 'resolved', 'expired', 'delivery_failed'].includes(status ?? '')) return
        const id = await state.get<string>(requestKey(context.threadId, execution, request))
        if (id) await resolve(id, status as CardStatus)
        return
      }
      if (typeof data.execution_id === 'string' && data.execution_id !== context.executionId) return
      let terminal = ['session.execution_completed', 'session.execution_cancelled', 'session.execution_failed'].includes(kind ?? '')
      if (kind === 'session.output.line' && typeof event.data === 'string') {
        try {
          const line = object(JSON.parse(event.data))
          terminal = ['turn.completed', 'turn.failed', 'turn.done', 'result'].includes(String(line.type))
            || ['turn/completed', 'error'].includes(String(line.method))
        } catch {
          // A non-JSON output line cannot resolve a question.
        }
      }
      if (!terminal || !context.executionId) return
      const ids = await state.getList<string>(executionKey(context.threadId, context.executionId))
      for (const id of new Set(ids)) await resolve(id, 'expired')
    }
  }
}

export async function* observeUserInputEvents(
  options: SlackbotV2Options,
  context: StreamContext,
  stream: AsyncIterable<SlackbotV2RendererSource>
): AsyncIterable<SlackbotV2RendererSource> {
  if (!options.state) throw new Error('Slack user-input bridge requires the service state adapter')
  const bridge = createSlackUserInputBridge(options, options.state)
  for await (const event of stream) {
    await bridge.observe(context, event)
    const kind = stringValue(event.eventKind) ?? stringValue(event.event)
    if (kind !== 'session.user_input_requested' && kind !== 'session.user_input_resolved') yield event
  }
}
