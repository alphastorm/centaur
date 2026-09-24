#!/usr/bin/env python3
"""Provider-free OMP v1 peer; recorded turns are replayed without modification."""
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

session_dir = Path(sys.argv[sys.argv.index("--session-dir") + 1])
session_dir.mkdir(parents=True, exist_ok=True)
previous = sorted(session_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime_ns)
session_file = previous[-1] if "--continue" in sys.argv and previous else session_dir / f"fake-{uuid.uuid4().hex}.jsonl"
history = json.loads(session_file.read_text()) if session_file.exists() else {}
session_file.write_text(json.dumps(history))
model = {"provider": "anthropic", "id": "fake-model"}
if "--model" in sys.argv:
    selected = sys.argv[sys.argv.index("--model") + 1]
    provider, _, name = selected.partition("/")
    model = {"provider": provider, "id": name} if name else {"provider": "anthropic", "id": selected}
lock = threading.Lock()
abort = threading.Event()
steer = threading.Event()
streaming = False
prompt_mode = ""
error_active = False


def raw(data):
    with lock:
        sys.stdout.buffer.write(data + b"\n")
        sys.stdout.buffer.flush()


def emit(frame):
    global streaming
    if frame.get("type") == "agent_start":
        streaming = True
    elif frame.get("type") == "agent_end" and frame.get("isTerminal") is True:
        streaming = False
    raw(json.dumps(frame, separators=(",", ":")).encode())


def response(command, success=True, data=None):
    frame = {"type": "response", "id": command["id"], "command": command["type"], "success": success}
    if success and data is not None:
        frame["data"] = data
    if not success:
        frame["error"] = "fake command failure"
    emit(frame)


def terminal():
    emit({"type": "agent_end", "isTerminal": True, "messages": []})


def answer(text, nonterminal=False):
    emit({"type": "agent_start"})
    emit({"type": "turn_start"})
    message = {"role": "assistant"}
    emit({"type": "message_start", "message": message})
    for delta in (text[:len(text) // 2], text[len(text) // 2:]):
        emit({"type": "message_update", "message": message,
              "assistantMessageEvent": {"type": "text_delta", "delta": delta}})
    emit({"type": "message_end", "message": {**message, "content": [{"type": "text", "text": text}],
          "stopReason": "stop", "usage": {"input": 10, "output": 4}}})
    emit({"type": "turn_end"})
    if nonterminal:
        emit({"type": "agent_end", "isTerminal": False, "messages": []})
        answer(" continued")
    else:
        terminal()


def later(fn, *args):
    def run():
        time.sleep(0.02)
        fn(*args)
    threading.Thread(target=run, daemon=True).start()


def await_abort():
    abort.wait(30)
    terminal()


def replay(name):
    global streaming
    for line in Path(__file__).with_name("omp-18.3.0").joinpath(name + ".jsonl").read_text().splitlines():
        frame = json.loads(line)
        if frame.get("type") == "agent_start":
            streaming = True
        elif frame.get("type") == "agent_end" and frame.get("isTerminal") is True:
            streaming = False
        raw(line.encode())


def prompt(command):
    global prompt_mode, error_active
    text = command["message"]
    prompt_mode = text
    abort.clear()
    steer.clear()
    if text in ("__local__", "__local_result__"):
        emit({"type": "command_output", "text": "local command completed"})
        if text == "__local_result__":
            emit({"type": "prompt_result", "id": command["id"], "agentInvoked": False})
            response(command)
        else:
            response(command, data={"agentInvoked": False})
        return
    if text == "__early__":
        answer("early event retained")
        response(command)
        return
    if text == "__out_of_order__":
        def finish():
            steer.wait(30)
            answer("steered out of order")
            response(command)
        later(finish)
        return
    response(command)
    if text.startswith("__replay:"):
        later(replay, text.split(":", 1)[1])
        return
    emit({"type": "agent_start"})
    if text == "__unknown_event__":
        emit({"type": "unrecognized_turn_event"})
    elif text in ("__missing_terminal__", "__nonboolean_terminal__"):
        frame = {"type": "agent_end", "messages": []}
        if text == "__nonboolean_terminal__":
            frame["isTerminal"] = "true"
        emit(frame)
    elif text == "__unknown_update__":
        emit({"type": "message_start", "message": {"role": "assistant"}})
        emit({"type": "message_update", "message": {"role": "assistant"},
              "assistantMessageEvent": {"type": "unknown_delta"}})
    elif text == "__presentation__":
        for kind in ("available_commands_update", "config_update", "session_info_update", "model_changed",
                     "thinking_level_changed", "config_warnings_changed", "advisor_cost_changed", "advisor_yielded",
                     "tool_stream_update", "auto_compaction_start", "auto_compaction_end", "auto_retry_start",
                     "retry_fallback_applied", "retry_fallback_succeeded", "ttsr_triggered", "todo_reminder",
                     "todo_auto_clear", "irc_message", "goal_updated"):
            emit({"type": kind})
        answer("known presentation accepted")
    elif text == "__duplicate_terminal__":
        answer("first turn")
        terminal()
    elif text == "__stale_terminal_during_turn__":
        raw(b'{"type":"agent_end","isTerminal":true,"messages":[]}')
        later(answer, "current turn after stale terminal")
    elif text == "__command_then_abort__":
        emit({"type": "command_output", "text": "command output before abort"})
        later(await_abort)
    elif text == "__error_keeps_running__":
        error_active = True
        emit({"type": "notice", "level": "error", "message": "fake active turn failure"})
        later(await_abort)
    elif text in ("__notice_error__", "__retry_error__", "__extension_error__"):
        frame = {"__notice_error__": {"type": "notice", "level": "error", "message": "fake notice failure"},
                 "__retry_error__": {"type": "auto_retry_end", "success": False, "finalError": "fake retry failure"},
                 "__extension_error__": {"type": "extension_error", "message": "fake extension failure"}}[text]
        emit(frame)
        terminal()
    elif text == "__persistence_error__":
        emit({"type": "notice", "level": "error", "source": "session-persistence",
              "message": "private-store-path: write failed"})
    elif text == "__late_error__":
        response(command, success=False)
    elif text == "__unknown_control__":
        response({"type": "steer", "id": "never-issued"})
    elif text == "__malformed__":
        raw(b"NOT JSON")
    elif text == "__oversize__":
        raw(b"x" * (2 * 1024 * 1024))
    elif text == "__hang__":
        later(await_abort)
    elif text in ("__ignore_abort__", "__late_abort__", "__late_steer__", "__silent_controls__"):
        pass
    elif text in ("__ui__", "__ui_select__"):
        emit({"type": "extension_ui_request", "method": "cancel", "id": "cancelled"})
        emit({"type": "extension_ui_request", "method": "select" if text == "__ui_select__" else "confirm", "id": "dialog"})
    elif text == "__host_tool__":
        emit({"type": "host_tool_call", "id": "tool", "toolName": "host", "arguments": {}})
    elif text == "__host_uri__":
        emit({"type": "host_uri_request", "id": "uri", "uri": "private://fixture"})
    elif text == "__nonterminal__":
        answer("nonterminal", nonterminal=True)
    elif text == "__retried_provider_error__":
        for line in Path(__file__).with_name("omp-18.3.0").joinpath("provider_error.jsonl").read_text().splitlines():
            frame = json.loads(line)
            if frame["type"] == "agent_end":
                frame["isTerminal"] = False
            emit(frame)
        emit({"type": "auto_retry_start"})
        answer("recovered after retry")
    elif text == "__provider_env__":
        answer("inherited" if os.environ.get("OPENAI_API_KEY") == "dummy" else "missing")
    elif text == "__model__":
        answer(model["provider"] + "/" + model["id"])
    elif text.startswith("__remember:"):
        history["codeword"] = text.split(":", 1)[1]
        session_file.write_text(json.dumps(history))
        answer("remembered")
    elif text == "__recall__":
        answer(history.get("codeword", "forgotten"))
    elif text == "__image__":
        answer("image count=" + str(len(command.get("images", []))))
    elif "Attached file saved to" in text:
        answer("document path accepted")
    else:
        later(answer, f"fake response pid={os.getpid()} session={session_file.stem}")


def handle(command):
    global model, error_active
    kind = command["type"]
    if kind == "get_state":
        if command.get("slowNotifications"):
            for _ in range(20):
                time.sleep(0.04)
                emit({"type": "model_changed"})
        response(command, data={"model": model, "isStreaming": streaming,
                               "sessionFile": str(session_file), "sessionId": session_file.stem})
    elif kind == "set_model":
        model = {"provider": command["provider"], "id": command["modelId"]}
        response(command, data=model)
    elif kind == "prompt":
        prompt(command)
    elif kind == "steer":
        if prompt_mode == "__silent_controls__":
            return
        if prompt_mode == "__late_steer__":
            answer("steered before acknowledgement")
        response(command)
        steer.set()
    elif kind == "abort":
        if prompt_mode == "__ignore_abort__":
            return
        if prompt_mode == "__late_abort__":
            terminal()
        response(command)
        if error_active:
            session_dir.joinpath("error-turn-aborted").touch()
            error_active = False
        abort.set()
    elif kind in ("extension_ui_response", "host_tool_result", "host_uri_result"):
        if command.get("id") == "cancelled":
            raise ValueError("presentation cancellation must not receive a reply")
        assert command.get("cancelled") is True or command.get("isError") is True
        terminal()
    else:
        response(command, success=False)


emit({"type": "ready", "protocolVersion": 1, "supportedProtocolVersions": [1]})
emit({"type": "extension_ui_request", "method": "setWidget", "id": "widget"})
for line in sys.stdin:
    if line.strip():
        handle(json.loads(line))
