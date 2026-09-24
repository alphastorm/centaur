use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use codex_app_server_protocol::UserInput;
use serde_json::{Value, json};

use crate::omp::normalize::{MAX_RENDERED_TEXT_BYTES, OmpEventNormalizer};
use crate::omp::persistence::{self, SessionMapping};
use crate::omp::profile::{OmpProfile, validate_contained_session_path};
use crate::omp::protocol::{
    Decoder, MAX_PHYSICAL_FRAME_BYTES, frame_type, is_state_notification, parse_ready,
    protocol_error,
};
use crate::stateful::{ProcessEvent, StatefulProcess};
use crate::traits::{HarnessKind, NormalizedEvent, NormalizedTokenUsage};
use crate::{HarnessServerError, Result};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(15);
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);
const TURN_IDLE_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const ABORT_GRACE: Duration = Duration::from_secs(5);
const POLL_INTERVAL: Duration = Duration::from_millis(50);
const MAX_IMAGES: usize = 8;
const MAX_IMAGE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_PENDING_CONTROLS: usize = 1024;

#[derive(Debug)]
pub(crate) enum TurnControl {
    Steer(Vec<UserInput>),
    Interrupt,
}

#[derive(Debug, Clone)]
pub(crate) struct TurnOutcome {
    pub(crate) interrupted: bool,
    pub(crate) usage: Option<NormalizedTokenUsage>,
}

pub(crate) struct OmpSession {
    profile: OmpProfile,
    process: StatefulProcess,
    decoder: Decoder,
    next_id: u64,
    pending_controls: HashMap<String, (&'static str, u64)>,
    thread_id: String,
    state: Value,
    normalizer: OmpEventNormalizer,
}

impl OmpSession {
    pub(crate) fn start(
        profile: OmpProfile,
        thread_id: &str,
        cwd: &Path,
        requested_provider: &str,
        requested_model: &str,
        resume: Option<&SessionMapping>,
    ) -> Result<Self> {
        let command = profile.command(cwd);
        let process = StatefulProcess::spawn(command, MAX_PHYSICAL_FRAME_BYTES)?;
        let mut session = Self {
            profile,
            process,
            decoder: Decoder::default(),
            next_id: 1,
            pending_controls: HashMap::new(),
            thread_id: thread_id.to_string(),
            state: Value::Null,
            normalizer: OmpEventNormalizer::default(),
        };
        session.initialize(requested_provider, requested_model, resume)?;
        Ok(session)
    }

    pub(crate) fn state(&self) -> &Value {
        &self.state
    }

    pub(crate) fn has_exited(&mut self) -> Result<bool> {
        Ok(self.process.try_wait()?.is_some())
    }

    pub(crate) fn run_turn<F, C>(
        &mut self,
        input: &[UserInput],
        mut poll_control: C,
        mut emit: F,
    ) -> Result<TurnOutcome>
    where
        F: FnMut(NormalizedEvent) -> Result<()>,
        C: FnMut() -> Result<Option<TurnControl>>,
    {
        let (message, images) = prompt_content(input)?;
        let turn_sequence = self.next_id;
        let prompt_id = self.command_id("prompt");
        let mut prompt = json!({
            "id": prompt_id.clone(),
            "type": "prompt",
            "message": message,
        });
        if !images.is_empty() {
            prompt["images"] = Value::Array(images);
        }
        self.process.write_json(&prompt)?;

        let started = Instant::now();
        let mut last_progress = started;
        let mut prompt_acknowledged = false;
        let mut agent_started = false;
        let mut terminal_seen = false;
        let mut terminal_check: Option<(String, Instant)> = None;
        let mut interrupted = false;
        let mut abort_deadline = None;
        let mut turn_error = None;
        let mut pending_command_output = String::new();
        let mut usage = None;

        loop {
            while let Some(control) = poll_control()? {
                if !interrupted && self.pending_controls.len() >= MAX_PENDING_CONTROLS {
                    return Err(protocol_error(
                        "OMP pending control response limit exceeded",
                    ));
                }
                match control {
                    TurnControl::Steer(input) if !interrupted => {
                        let (message, images) = prompt_content(&input)?;
                        let id = self.command_id("steer");
                        let mut command = json!({"id": id, "type": "steer", "message": message});
                        if !images.is_empty() {
                            command["images"] = Value::Array(images);
                        }
                        self.process.write_json(&command)?;
                        self.pending_controls.insert(id, ("steer", turn_sequence));
                    }
                    TurnControl::Interrupt if !interrupted => {
                        interrupted = true;
                        let id = self.command_id("abort");
                        self.process
                            .write_json(&json!({"id": id, "type": "abort"}))?;
                        self.pending_controls.insert(id, ("abort", turn_sequence));
                        abort_deadline = Some(Instant::now() + ABORT_GRACE);
                    }
                    TurnControl::Steer(_) | TurnControl::Interrupt => {}
                }
            }

            if terminal_seen && (prompt_acknowledged || interrupted) {
                if let Some(error) = turn_error {
                    return Err(protocol_error(error));
                }
                emit(NormalizedEvent::Result { error: None })?;
                return Ok(TurnOutcome { interrupted, usage });
            }
            if interrupted && abort_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                let _ = self.process.kill_and_wait();
                if let Some(error) = turn_error {
                    return Err(protocol_error(error));
                }
                return Ok(TurnOutcome {
                    interrupted: true,
                    usage,
                });
            }
            if !prompt_acknowledged && started.elapsed() >= COMMAND_TIMEOUT {
                return Err(protocol_error("OMP prompt acknowledgement timed out"));
            }
            if terminal_check
                .as_ref()
                .is_some_and(|(_, deadline)| Instant::now() >= *deadline)
            {
                return Err(protocol_error("OMP terminal state check timed out"));
            }
            if last_progress.elapsed() >= TURN_IDLE_TIMEOUT {
                return Err(protocol_error(
                    "OMP turn made no transport progress before watchdog expiry",
                ));
            }

            let frame = match self.recv_frame(POLL_INTERVAL) {
                Ok(Some(frame)) => frame,
                Ok(None) => continue,
                Err(HarnessServerError::Protocol(message)) if message.contains("timed out") => {
                    continue;
                }
                Err(error) => return Err(error),
            };
            let kind = frame_type(&frame)?;

            if kind == "response" {
                let id = frame
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| protocol_error("OMP response omitted string id"))?;
                let command = frame
                    .get("command")
                    .and_then(Value::as_str)
                    .ok_or_else(|| protocol_error("OMP response omitted command"))?;
                let success = frame
                    .get("success")
                    .and_then(Value::as_bool)
                    .ok_or_else(|| protocol_error("OMP response omitted success"))?;
                if id == prompt_id {
                    if command != "prompt" {
                        return Err(protocol_error("OMP prompt response command mismatch"));
                    }
                    if !success {
                        return Err(protocol_error(format!(
                            "OMP prompt failed: {}",
                            bounded_error(frame.get("error").and_then(Value::as_str))
                        )));
                    }
                    prompt_acknowledged = true;
                    if frame
                        .get("data")
                        .and_then(|data| data.get("agentInvoked"))
                        .and_then(Value::as_bool)
                        == Some(false)
                    {
                        for event in self
                            .normalizer
                            .command_output(&pending_command_output, true)
                        {
                            emit(event)?;
                        }
                        pending_command_output.clear();
                        terminal_seen = true;
                    }
                    last_progress = Instant::now();
                    continue;
                }
                if terminal_check
                    .as_ref()
                    .is_some_and(|(expected, _)| expected == id)
                {
                    if command != "get_state" || !success {
                        return Err(protocol_error(
                            "OMP terminal state check failed or mismatched",
                        ));
                    }
                    let streaming = frame
                        .pointer("/data/isStreaming")
                        .and_then(Value::as_bool)
                        .ok_or_else(|| protocol_error("OMP terminal state omitted isStreaming"))?;
                    terminal_check = None;
                    // OMP 18.3 terminals carry no prompt ID. A response issued
                    // for this prompt confirms quiescence; a stale terminal
                    // while the current agent is streaming cannot settle it.
                    terminal_seen = !streaming;
                    if terminal_seen {
                        last_progress = Instant::now();
                    }
                    continue;
                }
                let Some((expected, issued_turn)) = self.pending_controls.remove(id) else {
                    return Err(protocol_error(format!(
                        "OMP emitted response for unknown active-turn id {id}"
                    )));
                };
                // A terminal event can precede its control acknowledgement. Keep
                // correlation across turns, but never fail a later turn for a
                // control that belonged to an already completed turn.
                if command != expected || (issued_turn == turn_sequence && !success) {
                    return Err(protocol_error(format!(
                        "OMP {expected} response failed or mismatched"
                    )));
                }
                if issued_turn == turn_sequence {
                    last_progress = Instant::now();
                }
                continue;
            }

            if kind == "prompt_result" {
                if frame.get("id").and_then(Value::as_str) == Some(prompt_id.as_str())
                    && frame.get("agentInvoked").and_then(Value::as_bool) == Some(false)
                {
                    for event in self
                        .normalizer
                        .command_output(&pending_command_output, true)
                    {
                        emit(event)?;
                    }
                    pending_command_output.clear();
                    terminal_seen = true;
                    last_progress = Instant::now();
                }
                continue;
            }

            if is_privileged_callback(kind) {
                self.deny_callback(&frame)?;
                last_progress = Instant::now();
                continue;
            }

            if kind == "agent_end" {
                let terminal = frame
                    .get("isTerminal")
                    .and_then(Value::as_bool)
                    .ok_or_else(|| protocol_error("OMP agent_end omitted boolean isTerminal"))?;
                if !agent_started {
                    // Consume late terminals from the previous prompt.
                    continue;
                }
                if terminal && terminal_check.is_none() {
                    let id = self.command_id("get_state");
                    self.process
                        .write_json(&json!({"id": id, "type": "get_state"}))?;
                    terminal_check = Some((id, Instant::now() + COMMAND_TIMEOUT));
                } else if !terminal {
                    last_progress = Instant::now();
                }
                continue;
            }

            if kind == "agent_start" {
                agent_started = true;
                for event in self
                    .normalizer
                    .command_output(&pending_command_output, false)
                {
                    emit(event)?;
                }
                pending_command_output.clear();
            }
            if kind == "command_output" && !agent_started {
                let text = frame
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| protocol_error("OMP command_output omitted text"))?;
                if pending_command_output.len().saturating_add(text.len()) > MAX_RENDERED_TEXT_BYTES
                {
                    return Err(protocol_error(
                        "OMP pending command output exceeds its size limit",
                    ));
                }
                pending_command_output.push_str(text);
                last_progress = Instant::now();
                continue;
            }

            for event in self.normalizer.normalize(&frame)? {
                if let NormalizedEvent::TokenUsage { usage: event_usage } = &event {
                    usage = Some(event_usage.clone());
                }
                if let NormalizedEvent::Error { message } = event {
                    turn_error.get_or_insert(message);
                    if !interrupted {
                        interrupted = true;
                        let id = self.command_id("abort");
                        self.process
                            .write_json(&json!({"id": id, "type": "abort"}))?;
                        self.pending_controls.insert(id, ("abort", turn_sequence));
                        abort_deadline = Some(Instant::now() + ABORT_GRACE);
                    }
                    continue;
                }
                emit(event)?;
            }
            last_progress = Instant::now();
        }
    }

    pub(crate) fn ensure_model(
        &mut self,
        requested_provider: &str,
        requested_model: &str,
    ) -> Result<()> {
        let requested_pair = match (requested_provider, requested_model) {
            ("", "") => None,
            ("", combined) => {
                let (provider, model) = combined.split_once('/').ok_or_else(|| {
                    protocol_error("OMP model overrides without a provider must use provider/model")
                })?;
                if provider.is_empty() || model.is_empty() {
                    return Err(protocol_error(
                        "OMP model overrides without a provider must use provider/model",
                    ));
                }
                Some((provider.to_owned(), model.to_owned()))
            }
            (provider, model) if !provider.is_empty() && !model.is_empty() => {
                Some((provider.to_owned(), model.to_owned()))
            }
            _ => {
                return Err(protocol_error(
                    "OMP provider and model must be selected together",
                ));
            }
        };
        let (observed_provider, observed_model) = state_model(&self.state)?;
        let (desired_provider, desired_model) =
            requested_pair.unwrap_or_else(|| (observed_provider.clone(), observed_model.clone()));
        self.profile
            .verify_allowed_model(&desired_provider, &desired_model)?;

        if desired_provider != observed_provider || desired_model != observed_model {
            let available =
                self.send_command_wait("get_available_models", json!({}), COMMAND_TIMEOUT)?;
            let available_from_runtime = available
                .get("data")
                .and_then(|data| data.get("models"))
                .and_then(Value::as_array)
                .is_some_and(|models| {
                    models.iter().any(|model| {
                        model.get("provider").and_then(Value::as_str)
                            == Some(desired_provider.as_str())
                            && model.get("id").and_then(Value::as_str)
                                == Some(desired_model.as_str())
                    })
                });
            if !available_from_runtime {
                return Err(protocol_error(
                    "requested OMP provider/model is unavailable in the pinned runtime",
                ));
            }
            self.send_command_wait(
                "set_model",
                json!({"provider": desired_provider, "modelId": desired_model}),
                COMMAND_TIMEOUT,
            )?;
            self.state = self
                .send_command_wait("get_state", json!({}), COMMAND_TIMEOUT)?
                .get("data")
                .cloned()
                .ok_or_else(|| protocol_error("OMP get_state response omitted data"))?;
        }

        let (effective_provider, effective_model) = state_model(&self.state)?;
        if effective_provider != desired_provider || effective_model != desired_model {
            return Err(protocol_error(
                "OMP effective provider/model differs from the selected allowlisted model",
            ));
        }
        self.profile
            .verify_allowed_model(&effective_provider, &effective_model)?;
        self.profile.verify_state(&self.state)
    }

    fn initialize(
        &mut self,
        requested_provider: &str,
        requested_model: &str,
        resume: Option<&SessionMapping>,
    ) -> Result<()> {
        let ready_value = self.recv_required_frame(Instant::now() + STARTUP_TIMEOUT)?;
        let ready = parse_ready(&ready_value)?;
        self.decoder.set_limits_from_ready(&ready);
        if ready.supported_protocol_versions.contains(&2) {
            let response = self.send_command_wait(
                "negotiate_protocol",
                json!({"protocolVersion": 2}),
                COMMAND_TIMEOUT,
            )?;
            if response
                .get("data")
                .and_then(|data| data.get("protocolVersion"))
                .and_then(Value::as_u64)
                != Some(2)
            {
                return Err(protocol_error(
                    "OMP v2 negotiation returned the wrong version",
                ));
            }
            self.decoder.enable_v2();
        } else {
            return Err(protocol_error("centaur-safe requires OMP protocol v2"));
        }

        if let Some(mapping) = resume {
            self.send_command_wait(
                "switch_session",
                json!({"sessionPath": mapping.session_file}),
                COMMAND_TIMEOUT,
            )?;
        }
        self.send_command_wait("set_host_tools", json!({"tools": []}), COMMAND_TIMEOUT)?;
        self.send_command_wait(
            "set_host_uri_schemes",
            json!({"schemes": []}),
            COMMAND_TIMEOUT,
        )?;
        self.send_command_wait(
            "set_subagent_subscription",
            json!({"level": "off"}),
            COMMAND_TIMEOUT,
        )?;
        self.send_command_wait(
            "set_steering_mode",
            json!({"mode": "one-at-a-time"}),
            COMMAND_TIMEOUT,
        )?;
        self.send_command_wait(
            "set_follow_up_mode",
            json!({"mode": "one-at-a-time"}),
            COMMAND_TIMEOUT,
        )?;
        self.send_command_wait(
            "set_interrupt_mode",
            json!({"mode": "immediate"}),
            COMMAND_TIMEOUT,
        )?;

        self.state = self
            .send_command_wait("get_state", json!({}), COMMAND_TIMEOUT)?
            .get("data")
            .cloned()
            .ok_or_else(|| protocol_error("OMP get_state response omitted data"))?;

        self.ensure_model(requested_provider, requested_model)?;

        self.profile.verify_state(&self.state)?;
        if let Some(mapping) = resume {
            let resumed_session = self
                .state
                .get("sessionId")
                .and_then(Value::as_str)
                .ok_or_else(|| protocol_error("OMP resumed state omitted sessionId"))?;
            let resumed_file = self
                .state
                .get("sessionFile")
                .and_then(Value::as_str)
                .ok_or_else(|| protocol_error("OMP resumed state omitted sessionFile"))?;
            if resumed_session != mapping.session_id
                || Path::new(resumed_file) != mapping.session_file
            {
                return Err(protocol_error("OMP resumed a different persisted session"));
            }
        }
        self.persist_mapping()
    }

    fn persist_mapping(&self) -> Result<()> {
        let session_id = self
            .state
            .get("sessionId")
            .and_then(Value::as_str)
            .ok_or_else(|| protocol_error("OMP state omitted sessionId"))?
            .to_string();
        let session_file = self
            .state
            .get("sessionFile")
            .and_then(Value::as_str)
            .ok_or_else(|| protocol_error("OMP state omitted sessionFile"))?;
        let session_file =
            validate_contained_session_path(self.profile.session_root(), Path::new(session_file))?;
        persistence::save(
            &self.profile,
            &SessionMapping::new(&self.thread_id, session_id, session_file),
        )
    }

    fn send_command_wait(
        &mut self,
        command: &'static str,
        fields: Value,
        timeout: Duration,
    ) -> Result<Value> {
        let id = self.command_id(command);
        let mut value = fields.as_object().cloned().unwrap_or_default();
        value.insert("id".to_string(), Value::String(id.clone()));
        value.insert("type".to_string(), Value::String(command.to_string()));
        self.process.write_json(&Value::Object(value))?;
        let deadline = Instant::now() + timeout;
        loop {
            let frame = self.recv_required_frame(deadline)?;
            let kind = frame_type(&frame)?;
            if kind == "response" {
                let response_id = frame.get("id").and_then(Value::as_str);
                if let Some((expected, _)) =
                    response_id.and_then(|id| self.pending_controls.remove(id))
                {
                    if frame.get("command").and_then(Value::as_str) != Some(expected)
                        || frame.get("success").and_then(Value::as_bool).is_none()
                    {
                        return Err(protocol_error("OMP late control response mismatched"));
                    }
                    continue;
                }
                if response_id != Some(id.as_str()) {
                    return Err(protocol_error(format!(
                        "OMP {command} received response for unexpected id"
                    )));
                }
                if frame.get("command").and_then(Value::as_str) != Some(command) {
                    return Err(protocol_error(format!(
                        "OMP {command} response command mismatch"
                    )));
                }
                if frame.get("success").and_then(Value::as_bool) != Some(true) {
                    return Err(protocol_error(format!(
                        "OMP {command} failed: {}",
                        bounded_error(frame.get("error").and_then(Value::as_str))
                    )));
                }
                return Ok(frame);
            }
            if is_privileged_callback(kind) {
                self.deny_callback(&frame)?;
                continue;
            }
            if kind == "agent_end" {
                if frame.get("isTerminal").and_then(Value::as_bool).is_none() {
                    return Err(protocol_error("OMP agent_end omitted boolean isTerminal"));
                }
                continue;
            }
            if !is_state_notification(kind) {
                return Err(protocol_error(format!(
                    "OMP emitted semantic frame {kind} while awaiting {command}"
                )));
            }
        }
    }

    fn deny_callback(&mut self, frame: &Value) -> Result<()> {
        let kind = frame_type(frame)?;
        if kind == "extension_ui_request"
            && matches!(
                frame.get("method").and_then(Value::as_str),
                Some(
                    "cancel"
                        | "notify"
                        | "setStatus"
                        | "setWidget"
                        | "setTitle"
                        | "set_editor_text"
                        | "open_url"
                )
            )
        {
            // Presentation updates and 18.3 dialog cancellation have no reply.
            return Ok(());
        }
        let id = frame
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| protocol_error("OMP privileged callback omitted string id"))?;
        let response = match kind {
            "extension_ui_request" => {
                if !matches!(
                    frame.get("method").and_then(Value::as_str),
                    Some("select" | "confirm" | "input" | "editor")
                ) {
                    return Err(protocol_error("unknown OMP extension UI request"));
                }
                json!({"type": "extension_ui_response", "id": id, "cancelled": true})
            }
            "host_tool_call" => json!({
                "type": "host_tool_result",
                "id": id,
                "result": {"content": [{"type": "text", "text": "Host tools are disabled by centaur-safe"}]},
                "isError": true
            }),
            "host_uri_request" => json!({
                "type": "host_uri_result",
                "id": id,
                "isError": true,
                "error": "Host URIs are disabled by centaur-safe"
            }),
            "host_tool_cancel" | "host_uri_cancel" => return Ok(()),
            _ => return Err(protocol_error("unknown privileged OMP callback")),
        };
        self.process.write_json(&response)
    }

    fn recv_required_frame(&mut self, deadline: Instant) -> Result<Value> {
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(protocol_error("OMP frame wait timed out"));
            }
            match self.recv_frame(remaining)? {
                Some(frame) => return Ok(frame),
                None => continue,
            }
        }
    }

    fn recv_frame(&mut self, timeout: Duration) -> Result<Option<Value>> {
        match self.process.recv_timeout(timeout) {
            Ok(ProcessEvent::Frame(line)) => {
                let frame = self.decoder.decode_line(&line)?;
                if frame.as_ref().is_some_and(|frame| {
                    frame.get("type").and_then(Value::as_str) == Some("notice")
                        && frame.get("level").and_then(Value::as_str) == Some("error")
                        && frame.get("source").and_then(Value::as_str)
                            == Some("session-persistence")
                }) {
                    return Err(protocol_error("OMP session persistence failed"));
                }
                Ok(frame)
            }
            Ok(ProcessEvent::StdoutError(error)) => Err(HarnessServerError::Io(error)),
            Ok(ProcessEvent::Eof) => {
                if let Some(status) = self.process.try_wait()? {
                    Err(HarnessServerError::HarnessExited {
                        kind: HarnessKind::Omp,
                        status,
                        stderr: self.process.redacted_stderr_tail(),
                    })
                } else {
                    Err(protocol_error(format!(
                        "OMP stdout closed unexpectedly{}",
                        self.process.redacted_stderr_tail()
                    )))
                }
            }
            Err(RecvTimeoutError::Timeout) => Err(protocol_error("OMP frame wait timed out")),
            Err(RecvTimeoutError::Disconnected) => Err(protocol_error(format!(
                "OMP stdout reader disconnected{}",
                self.process.redacted_stderr_tail()
            ))),
        }
    }

    fn command_id(&mut self, command: &str) -> String {
        let id = format!("centaur-{command}-{}", self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        id
    }
}

pub(crate) fn load_resume_mapping(profile: &OmpProfile, thread_id: &str) -> Result<SessionMapping> {
    persistence::load(profile, thread_id)?.ok_or_else(|| {
        protocol_error(format!(
            "OMP resume mapping for thread {thread_id} is unavailable"
        ))
    })
}

fn state_model(state: &Value) -> Result<(String, String)> {
    let provider = state
        .get("model")
        .and_then(|model| model.get("provider"))
        .and_then(Value::as_str)
        .filter(|provider| !provider.is_empty())
        .ok_or_else(|| protocol_error("OMP state omitted effective model provider"))?;
    let model = state
        .get("model")
        .and_then(|model| model.get("id"))
        .and_then(Value::as_str)
        .filter(|model| !model.is_empty())
        .ok_or_else(|| protocol_error("OMP state omitted effective model id"))?;
    Ok((provider.to_owned(), model.to_owned()))
}

fn is_privileged_callback(kind: &str) -> bool {
    matches!(
        kind,
        "extension_ui_request"
            | "host_tool_call"
            | "host_tool_cancel"
            | "host_uri_request"
            | "host_uri_cancel"
    )
}

fn prompt_content(input: &[UserInput]) -> Result<(String, Vec<Value>)> {
    let mut messages = Vec::new();
    let mut images = Vec::new();
    for item in input {
        match item {
            UserInput::Text { text, .. } => messages.push(text.clone()),
            UserInput::Image { url, .. } => {
                images.push(image_from_data_url(url)?);
            }
            UserInput::LocalImage { path, .. } => {
                images.push(image_from_path(path)?);
            }
            UserInput::Skill { name, path } => {
                messages.push(format!("[skill: {name} at {}]", path.display()));
            }
            UserInput::Mention { name, path } => {
                messages.push(format!("[mention: {name} at {path}]"));
            }
        }
        if images.len() > MAX_IMAGES {
            return Err(protocol_error(format!(
                "OMP input exceeds the {MAX_IMAGES}-image limit"
            )));
        }
    }
    if messages.is_empty() {
        messages.push("Review the attached image.".to_string());
    }
    Ok((messages.join("\n\n"), images))
}

fn image_from_path(path: &Path) -> Result<Value> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() > MAX_IMAGE_BYTES {
        return Err(protocol_error(format!(
            "OMP image must be a regular file no larger than {MAX_IMAGE_BYTES} bytes"
        )));
    }
    let bytes = fs::read(path)?;
    let mime_type = image_mime(path)?;
    Ok(json!({
        "type": "image",
        "data": BASE64_STANDARD.encode(bytes),
        "mimeType": mime_type
    }))
}

fn image_from_data_url(url: &str) -> Result<Value> {
    let Some(rest) = url.strip_prefix("data:") else {
        return Err(protocol_error(
            "OMP remote image URLs are disabled; stage the image in the sandbox",
        ));
    };
    let Some((header, data)) = rest.split_once(',') else {
        return Err(protocol_error("OMP image data URL is malformed"));
    };
    let Some(mime_type) = header.strip_suffix(";base64") else {
        return Err(protocol_error("OMP image data URL must use base64"));
    };
    if !matches!(mime_type, "image/png" | "image/jpeg" | "image/webp") {
        return Err(protocol_error("OMP image MIME type is not allowlisted"));
    }
    let bytes = BASE64_STANDARD
        .decode(data)
        .map_err(|error| protocol_error(format!("OMP image base64 is invalid: {error}")))?;
    if bytes.len() as u64 > MAX_IMAGE_BYTES {
        return Err(protocol_error("OMP image data exceeds its byte limit"));
    }
    Ok(json!({"type": "image", "data": data, "mimeType": mime_type}))
}

fn image_mime(path: &Path) -> Result<&'static str> {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => Ok("image/png"),
        Some("jpg" | "jpeg") => Ok("image/jpeg"),
        Some("webp") => Ok("image/webp"),
        _ => Err(protocol_error("OMP local image type is not allowlisted")),
    }
}

fn bounded_error(error: Option<&str>) -> String {
    let error = error.unwrap_or("unspecified error");
    const LIMIT: usize = 4096;
    if error.len() <= LIMIT {
        return error.to_string();
    }
    let mut end = LIMIT;
    while !error.is_char_boundary(end) {
        end -= 1;
    }
    format!("{} [truncated]", &error[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slow_chunked_command_response_obeys_one_deadline() {
        let root = std::env::temp_dir().join(format!("omp-command-{}", uuid::Uuid::new_v4()));
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_omp_rpc.py");
        let profile = OmpProfile::for_test(fixture, &root);
        let mut session = OmpSession::start(
            profile,
            "deadline-test",
            &root,
            "anthropic",
            "fake-model",
            None,
        )
        .unwrap();
        let started = Instant::now();
        let result = session.send_command_wait(
            "get_state",
            json!({"slowChunks": true}),
            Duration::from_millis(200),
        );
        let elapsed = started.elapsed();
        drop(session);
        fs::remove_dir_all(root).unwrap();
        assert!(
            result.is_err(),
            "slow chunks exceeded deadline but returned success"
        );
        assert!(result.unwrap_err().to_string().contains("timed out"));
        assert!(elapsed < Duration::from_secs(1), "elapsed: {elapsed:?}");
    }
}
