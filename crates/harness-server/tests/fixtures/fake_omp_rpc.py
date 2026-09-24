#!/usr/bin/env python3
"""Deterministic, provider-free OMP RPC peer for harness-server tests."""
from __future__ import annotations

import base64
import json
import os
import pathlib
import sys
import threading
import time
import uuid
from typing import Any
if "--version" in sys.argv:
    print("omp/18.3.0")
    raise SystemExit(0)


MAX_PHYSICAL = 1_048_576
MAX_REASSEMBLED = 67_108_864
SAFE_TOOLS = ["bash", "edit", "glob", "grep", "lsp", "read", "task", "todo", "write"]

write_lock = threading.Lock()
state_lock = threading.Lock()
protocol_version = 1
chunk_seq = 0
last_assistant_text = ""
session_root = pathlib.Path(os.environ.get("PI_CODING_AGENT_DIR", "/tmp/fake-omp"))
session_root.mkdir(parents=True, exist_ok=True)
session_id = f"fake-session-{uuid.uuid4().hex}"
session_file = str(session_root / f"{session_id}.jsonl")
pathlib.Path(session_file).touch()
model = {"provider": "anthropic", "id": "fake-model"}
abort_event = threading.Event()
pending_ui: str | None = None
pending_host_tool: str | None = None
pending_host_uri: str | None = None
steer_event = threading.Event()
ignore_abort = False
late_control = False
silent_controls = False
error_turn_active = False
streaming = False


def raw_line(data: bytes) -> None:
    with write_lock:
        sys.stdout.buffer.write(data + b"\n")
        sys.stdout.buffer.flush()


def emit(obj: dict[str, Any], *, force_chunk: bool = False, chunk_delay: float = 0) -> None:
    global chunk_seq, streaming
    if obj.get("type") == "agent_start":
        streaming = True
    elif obj.get("type") == "agent_end" and obj.get("isTerminal") is True:
        streaming = False
    raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    with state_lock:
        use_v2 = protocol_version == 2
    if use_v2 and (force_chunk or len(raw) > 512):
        with state_lock:
            chunk_seq += 1
            chunk_id = f"rpc-{chunk_seq}"
        pieces = [raw[index:index + 220] for index in range(0, len(raw), 220)]
        for index, piece in enumerate(pieces):
            if chunk_delay:
                time.sleep(chunk_delay)
            raw_line(json.dumps({
                "type": "rpc_chunk",
                "chunkId": chunk_id,
                "index": index,
                "count": len(pieces),
                "byteLength": len(raw),
                "data": base64.b64encode(piece).decode("ascii"),
            }, separators=(",", ":")).encode("utf-8"))
        return
    raw_line(raw)


def response(command: str, request_id: str | None, success: bool = True,
             data: Any = None, error: str | None = None) -> None:
    frame: dict[str, Any] = {"type": "response", "command": command, "success": success}
    if request_id is not None:
        frame["id"] = request_id
    if success and data is not None:
        frame["data"] = data
    if not success:
        frame["error"] = error or "fake error"
    emit(frame)


def current_state() -> dict[str, Any]:
    return {
        "model": model.copy(),
        "thinkingLevel": "medium",
        "isStreaming": streaming,
        "isCompacting": False,
        "steeringMode": "one-at-a-time",
        "followUpMode": "one-at-a-time",
        "interruptMode": "immediate",
        "sessionFile": session_file,
        "sessionId": session_id,
        "sessionName": "Fake Centaur session",
        "autoCompactionEnabled": True,
        "fastModeEnabled": False,
        "fastModeActive": False,
        "tokensPerSecond": None,
        "messageCount": 0,
        "queuedMessageCount": 0,
        "todoPhases": [],
        "systemPrompt": ["Deterministic fake OMP peer."],
        "dumpTools": [
            {"name": name, "description": f"Fake {name}", "parameters": {}}
            for name in SAFE_TOOLS
        ],
        "contextUsage": {"tokens": 42, "contextWindow": 200000, "percent": 0.00021},
    }

def assistant_events(text: str, *, nonterminal: bool = False,
                     include_tool: bool = False, include_reasoning: bool = False,
                     force_chunk: bool = False, wait_for_interrupt: bool = False) -> None:
    global last_assistant_text
    message_id = f"msg-{uuid.uuid4().hex[:8]}"
    emit({"type": "agent_start"})
    emit({"type": "turn_start"})
    emit({"type": "message_start", "message": {"role": "assistant", "id": message_id}})
    if include_reasoning:
        emit({
            "type": "message_update",
            "message": {"role": "assistant", "id": message_id},
            "assistantMessageEvent": {"type": "thinking_delta", "delta": "fake reasoning"},
        })
    emit({
        "type": "message_update",
        "message": {"role": "assistant", "id": message_id},
        "assistantMessageEvent": {"type": "text_delta", "delta": text},
    }, force_chunk=force_chunk)
    if include_tool:
        call_id = f"tool-{uuid.uuid4().hex[:8]}"
        emit({"type": "tool_execution_start", "toolCallId": call_id,
              "toolName": "read", "args": {"path": "fixture.txt"}})
        emit({"type": "tool_execution_update", "toolCallId": call_id,
              "toolName": "read", "args": {"path": "fixture.txt"},
              "partialResult": "fixture"})
        emit({"type": "tool_execution_end", "toolCallId": call_id,
              "toolName": "read", "result": {"content": "fixture"}, "isError": False})
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if include_reasoning:
        content.insert(0, {"type": "thinking", "thinking": "fake reasoning"})
    emit({"type": "message_end", "message": {
        "role": "assistant", "id": message_id,
        "content": content,
        "usage": {"input": 10, "output": 4, "cacheRead": 2, "cacheWrite": 0},
    }})
    last_assistant_text = text
    emit({"type": "turn_end", "message": {}, "toolResults": []})
    if wait_for_interrupt:
        abort_event.wait(timeout=30)
    if nonterminal:
        emit({"type": "agent_end", "isTerminal": False,
              "messages": [], "reason": "async-delivery-pending"})
        emit({"type": "message_start", "message": {"role": "assistant", "id": message_id + "-2"}})
        emit({
            "type": "message_update",
            "message": {"role": "assistant", "id": message_id + "-2"},
            "assistantMessageEvent": {"type": "text_delta", "delta": " continued"},
        })
        last_assistant_text = text + " continued"
    emit({"type": "agent_end", "isTerminal": True,
          "messages": [], "reason": "end_turn"})


def delayed_normal(text: str, *, delay: float = 0.01, **kwargs: Any) -> None:
    time.sleep(delay)
    assistant_events(text, **kwargs)


def wait_for_abort() -> None:
    global last_assistant_text
    emit({"type": "agent_start"})
    abort_event.wait(timeout=30)
    last_assistant_text = ""
    emit({"type": "agent_end", "isTerminal": True, "messages": [], "reason": "aborted"})
def wait_for_steer(request_id: str | None) -> None:
    steer_event.wait(timeout=30)
    assistant_events("steered out of order")
    response("prompt", request_id, data={"agentInvoked": True})




def handle_prompt(cmd: dict[str, Any]) -> None:
    global pending_ui, pending_host_tool, pending_host_uri, last_assistant_text
    global ignore_abort, late_control, silent_controls, error_turn_active
    request_id = cmd.get("id")
    message = str(cmd.get("message", ""))
    ignore_abort = "__ignore_abort__" in message
    late_control = "__late_abort__" in message or "__late_steer__" in message
    silent_controls = "__silent_controls__" in message
    if "__local__" in message:
        emit({"type": "command_output", "id": request_id, "command": "/fake",
              "text": "local command completed"})
        response("prompt", request_id, data={"agentInvoked": False})
        return
    if "__local_result__" in message:
        emit({"type": "command_output", "text": "local prompt result"})
        emit({"type": "prompt_result", "id": request_id, "agentInvoked": False})
        response("prompt", request_id)
        return
    if "__early__" in message:
        emit({"type": "agent_start"})
        emit({"type": "message_start", "message": {"role": "assistant", "id": "early-msg"}})
        emit({"type": "message_update", "message": {"role": "assistant", "id": "early-msg"},
              "assistantMessageEvent": {"type": "text_delta", "delta": "early "}})
        response("prompt", request_id, data={"agentInvoked": True})
        emit({"type": "message_end", "message": {"role": "assistant", "id": "early-msg",
              "content": [{"type": "text", "text": "early event retained"}]}})
        last_assistant_text = "early event retained"
        emit({"type": "agent_end", "isTerminal": True, "messages": [], "reason": "end_turn"})
        return
    if "__out_of_order__" in message:
        steer_event.clear()
        threading.Thread(target=wait_for_steer, args=(request_id,), daemon=True).start()
        return
    response("prompt", request_id, data={"agentInvoked": True})
    emit({"type": "agent_start"})
    if "__presentation__" in message:
        for kind in ("available_commands_update", "config_update", "session_info_update",
                     "thinking_level_changed", "model_changed", "config_warnings_changed",
                     "advisor_cost_changed", "advisor_yielded", "tool_stream_update",
                     "auto_compaction_start", "auto_compaction_end", "auto_retry_start",
                     "retry_fallback_applied", "retry_fallback_succeeded", "ttsr_triggered",
                     "todo_reminder", "todo_auto_clear", "irc_message", "goal_updated"):
            emit({"type": kind})
        for kind in ("start", "text_start", "text_end", "thinking_start", "thinking_end",
                     "image_end", "toolcall_start", "toolcall_delta", "toolcall_end", "done", "error"):
            emit({"type": "message_update", "message": {"role": "assistant", "id": "presentation"},
                  "assistantMessageEvent": {"type": kind}})
        assistant_events("known presentation accepted")
    elif "__provider_env__" in message:
        names = sorted(name for name in os.environ if name.startswith(("OPENAI_", "GEMINI_", "AWS_", "GOOGLE_")))
        assistant_events("non-anthropic env:" + (",".join(names) or "none"))
    elif "__duplicate_terminal__" in message:
        assistant_events("first turn")
        emit({"type": "agent_end", "isTerminal": True, "messages": []})
    elif "__stale_terminal_during_turn__" in message:
        emit({"type": "agent_start"})
        raw_line(b'{"type":"agent_end","isTerminal":true,"messages":[]}')
        threading.Thread(target=delayed_normal, args=("current turn after stale terminal",),
                         kwargs={"delay": 0.2}, daemon=True).start()
    elif "__unknown_event__" in message:
        emit({"type": "unrecognized_turn_event"})
        emit({"type": "agent_end", "isTerminal": True, "messages": []})
    elif "__missing_terminal__" in message:
        emit({"type": "agent_end", "messages": []})
    elif "__nonboolean_terminal__" in message:
        emit({"type": "agent_end", "isTerminal": "true", "messages": []})
    elif "__unknown_update__" in message:
        emit({"type": "message_update", "message": {"role": "assistant", "id": "unknown"},
              "assistantMessageEvent": {"type": "unknown_delta"}})
        emit({"type": "agent_end", "isTerminal": True, "messages": []})
    elif "__command_then_abort__" in message:
        abort_event.clear()
        emit({"type": "command_output", "id": request_id, "command": "/fake",
              "text": "command output before abort"})
        threading.Thread(target=wait_for_abort, daemon=True).start()
    elif "__error_keeps_running__" in message:
        abort_event.clear()
        error_turn_active = True
        emit({"type": "notice", "level": "error", "message": "fake active turn failure"})
        threading.Thread(target=wait_for_abort, daemon=True).start()
    elif "__late_error__" in message:
        emit({"type": "agent_start"})
        time.sleep(0.01)
        response("prompt", request_id, success=False, error="fake async scheduling failure")
    elif "__persistence_error__" in message:
        emit({"type": "notice", "level": "error", "source": "session-persistence",
              "message": "private-store-path: write failed"})
    elif "__notice_error__" in message or "__retry_error__" in message or "__extension_error__" in message:
        if "__notice_error__" in message:
            emit({"type": "notice", "level": "error", "message": "fake notice failure"})
        elif "__retry_error__" in message:
            emit({"type": "auto_retry_end", "success": False, "finalError": "fake retry failure"})
        else:
            emit({"type": "extension_error", "message": "fake extension failure"})
        emit({"type": "agent_end", "isTerminal": True, "messages": []})
    elif "__answer_then_abort__" in message:
        abort_event.clear()
        threading.Thread(target=delayed_normal, args=("answer before abort",),
                         kwargs={"wait_for_interrupt": True}, daemon=True).start()
    elif "__ui_select__" in message:
        pending_ui = f"ui-{uuid.uuid4().hex[:8]}"
        emit({"type": "extension_ui_request", "id": "cancel-notification",
              "method": "cancel", "targetId": "already-cancelled"})
        emit({"type": "extension_ui_request", "id": pending_ui,
              "method": "select", "title": "Choose", "options": ["Allow"],
              "optionDetails": [{"description": "Must remain denied"}]})
    elif "__reasoning__" in message:
        threading.Thread(target=delayed_normal, args=("reasoned",),
                         kwargs={"include_reasoning": True}, daemon=True).start()
    elif "__nonterminal__" in message:
        threading.Thread(target=delayed_normal, args=("nonterminal",),
                         kwargs={"nonterminal": True}, daemon=True).start()
    elif "__chunk__" in message:
        threading.Thread(target=delayed_normal, args=("x" * 4096,),
                         kwargs={"force_chunk": True}, daemon=True).start()
    elif "__tool__" in message:
        threading.Thread(target=delayed_normal, args=("tool complete",),
                         kwargs={"include_tool": True}, daemon=True).start()
    elif "__image__" in message:
        images = cmd.get("images")
        count = len(images) if isinstance(images, list) else 0
        mime = images[0].get("mimeType") if count and isinstance(images[0], dict) else None
        threading.Thread(target=delayed_normal,
                         args=(f"image count={count} mime={mime}",), daemon=True).start()
    elif "Attached file saved to" in message:
        threading.Thread(target=delayed_normal,
                         args=("document path accepted",), daemon=True).start()
    elif "__unknown_control__" in message:
        response("steer", "centaur-steer-never-issued", data={"queued": True})
    elif ignore_abort or late_control or silent_controls:
        emit({"type": "agent_start"})
    elif "__hang__" in message:
        abort_event.clear()
        threading.Thread(target=wait_for_abort, daemon=True).start()
    elif "__ui__" in message:
        pending_ui = f"ui-{uuid.uuid4().hex[:8]}"
        emit({"type": "extension_ui_request", "id": pending_ui,
              "method": "confirm", "title": "Fake privileged request", "message": "Allow?"})
    elif "__host_tool__" in message:
        pending_host_tool = f"host-{uuid.uuid4().hex[:8]}"
        emit({"type": "host_tool_call", "id": pending_host_tool,
              "toolCallId": "fake-call", "toolName": "fake_host", "arguments": {"value": 1}})
    elif "__host_uri__" in message:
        pending_host_uri = f"uri-{uuid.uuid4().hex[:8]}"
        emit({"type": "host_uri_request", "id": pending_host_uri, "uri": "private://fixture"})
    elif "__malformed__" in message:
        raw_line(b"THIS IS NOT JSON")
    else:
        threading.Thread(target=delayed_normal,
                         args=(f"fake response pid={os.getpid()} session={session_id}",),
                         daemon=True).start()


def handle(cmd: dict[str, Any]) -> None:
    global protocol_version, session_id, session_file, model, pending_ui, pending_host_tool
    global pending_host_uri, error_turn_active
    typ = cmd.get("type")
    request_id = cmd.get("id")
    if typ == "negotiate_protocol":
        if cmd.get("protocolVersion") != 2:
            response("negotiate_protocol", request_id, success=False, error="only v2 is supported")
            return
        response("negotiate_protocol", request_id, data={"protocolVersion": 2})
        with state_lock:
            protocol_version = 2
    elif typ == "get_state":
        if cmd.get("slowChunks"):
            emit({"type": "response", "id": request_id, "command": "get_state",
                  "success": True, "data": {"padding": "x" * 8000}},
                 force_chunk=True, chunk_delay=0.04)
            return
        for kind in ("model_changed", "config_warnings_changed", "advisor_cost_changed"):
            emit({"type": kind})
        response("get_state", request_id, data=current_state())
    elif typ == "get_available_models":
        response("get_available_models", request_id,
                 data={"models": [{"provider": "anthropic", "id": name}
                                  for name in ("fake-model", "fake-model-2")]})
    elif typ == "set_model":
        model = {"provider": str(cmd.get("provider")), "id": str(cmd.get("modelId"))}
        response("set_model", request_id, data=model.copy())
    elif typ in {"set_subagent_subscription", "set_steering_mode", "set_follow_up_mode",
                 "set_interrupt_mode", "set_auto_compaction", "set_auto_retry",
                 "set_host_tools", "set_host_uri_schemes"}:
        response(str(typ), request_id, data={})
    elif typ == "switch_session":
        session_file = str(cmd.get("sessionPath"))
        session_id = pathlib.Path(session_file).stem
        pathlib.Path(session_file).touch(exist_ok=True)
        response("switch_session", request_id, data={"cancelled": False})
    elif typ == "prompt":
        handle_prompt(cmd)
    elif typ == "steer":
        if silent_controls:
            return
        if late_control:
            assistant_events("steered before acknowledgement")
        response("steer", request_id, data={"queued": True})
        steer_event.set()
    elif typ == "abort":
        if ignore_abort:
            return
        if late_control:
            emit({"type": "agent_end", "isTerminal": True,
                  "messages": [], "reason": "aborted"})
        response("abort", request_id, data={"aborted": True})
        if error_turn_active:
            (session_root / "error-turn-aborted").touch()
            error_turn_active = False
        abort_event.set()
    elif typ == "extension_ui_response":
        if cmd.get("id") == "cancel-notification":
            response("extension_ui_response", "cancel-notification", success=False,
                     error="notification must not receive a response")
        if cmd.get("id") == pending_ui:
            pending_ui = None
            emit({"type": "agent_end", "isTerminal": True, "messages": [], "reason": "ui-denied"})
    elif typ == "host_tool_result":
        if cmd.get("id") == pending_host_tool:
            pending_host_tool = None
            emit({"type": "agent_end", "isTerminal": True, "messages": [], "reason": "host-tool-rejected"})
    elif typ == "host_uri_result":
        if cmd.get("id") == pending_host_uri and cmd.get("isError") is True:
            pending_host_uri = None
            emit({"type": "agent_end", "isTerminal": True, "messages": [], "reason": "host-uri-rejected"})
    else:
        response(str(typ or "parse"), request_id, success=False, error="unknown fake command")


def main() -> int:
    emit({
        "type": "ready",
        "protocolVersion": 1,
        "supportedProtocolVersions": [1, 2],
        "maxFrameBytes": MAX_PHYSICAL,
        "maxReassembledFrameBytes": MAX_REASSEMBLED,
    })
    emit({"type": "extension_ui_request", "id": "startup-widget",
          "method": "setWidget", "widgetKey": "safe-profile-probe"})
    for line in sys.stdin.buffer:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("command must be an object")
            handle(value)
        except Exception as exc:
            response("parse", None, success=False, error=f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
