mod normalize;
mod persistence;
mod profile;
mod protocol;
mod session;

use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver, TryRecvError},
};

use codex_app_server_protocol::{
    ApprovalsReviewer, AskForApproval, ClientResponse, InitializeResponse, JSONRPCError,
    JSONRPCErrorError, JSONRPCMessage, JSONRPCRequest, JSONRPCResponse, RequestId, SandboxPolicy,
    ServerNotification, ThreadResumeParams, ThreadResumeResponse, ThreadStartParams,
    ThreadStartResponse, TurnInterruptParams, TurnInterruptResponse, TurnStartParams,
    TurnStartResponse, TurnStatus, TurnSteerParams, TurnSteerResponse, UserInput,
};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::otel::{TraceContext, TurnStatus as TelemetryTurnStatus, TurnTelemetry};
use crate::server::{
    BlocksCommand, BlocksState, parse_blocks_line_with_state, usage_span_input_value,
    write_blocks_error,
};
use crate::traits::{AppServerRuntime, HarnessKind, NormalizedEvent};
use crate::turn::{BridgeConfig, CodexTurnNormalizer};
use crate::util::{absolute_path, default_codex_home, write_value};
use crate::wire::notification_to_wire_value;
use crate::{HarnessServerError, Result};

use self::persistence::SessionMapping;
use self::profile::{OMP_VERSION, OmpProfile};
use self::session::{OmpSession, TurnControl, TurnOutcome, load_resume_mapping};

#[derive(Debug, Default)]
pub struct OmpRuntime;

impl AppServerRuntime for OmpRuntime {
    fn run_stdio(&self) -> Result<()> {
        run_app_server()
    }
}

struct OmpThreadState {
    id: String,
    cwd: PathBuf,
    model: String,
    model_provider: String,
    service_tier: Option<String>,
    completed_turns: Vec<codex_app_server_protocol::Turn>,
    session: Option<OmpSession>,
    resume: Option<SessionMapping>,
    thread_started_sent: bool,
}

enum RuntimeInput {
    JsonRpc(JSONRPCRequest),
    BlocksInterrupt,
}

enum BlocksReaderInput {
    Command(BlocksCommand),
    Error(String),
}

pub(crate) fn run_blocks_server() -> Result<()> {
    let profile = OmpProfile::load()?;
    let cwd = env::current_dir()?;
    let mut state = OmpThreadState {
        id: Uuid::new_v4().to_string(),
        cwd,
        model: String::new(),
        model_provider: String::new(),
        service_tier: None,
        completed_turns: Vec::new(),
        session: None,
        resume: None,
        thread_started_sent: false,
    };
    let mut stdout = io::stdout().lock();
    let (command_tx, command_rx) = mpsc::channel();
    let (control_tx, control_rx) = mpsc::channel();
    let turn_active = Arc::new(AtomicBool::new(false));

    {
        let turn_active = Arc::clone(&turn_active);
        std::thread::spawn(move || {
            let stdin = io::stdin();
            let mut blocks_state = BlocksState::default();
            for raw in stdin.lock().lines() {
                let Ok(line) = raw else {
                    break;
                };
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                match parse_blocks_line_with_state(trimmed, &mut blocks_state) {
                    Ok(BlocksCommand::Interrupt) if turn_active.load(Ordering::SeqCst) => {
                        if control_tx.send(RuntimeInput::BlocksInterrupt).is_err() {
                            break;
                        }
                    }
                    Ok(command) => {
                        if command_tx
                            .send(BlocksReaderInput::Command(command))
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(error) => {
                        if command_tx
                            .send(BlocksReaderInput::Error(error.to_string()))
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
        });
    }

    while let Ok(input) = command_rx.recv() {
        match input {
            BlocksReaderInput::Command(BlocksCommand::User {
                input,
                client_user_message_id,
                model,
                provider,
                reasoning: _,
                trace_context,
            }) => {
                if let Some(model) = model {
                    state.model = model;
                }
                if let Some(provider) = provider {
                    state.model_provider = provider;
                }
                turn_active.store(true, Ordering::SeqCst);
                let result = run_normalized_turn(
                    &profile,
                    &mut state,
                    &input,
                    OmpTurnRequest {
                        client_user_message_id,
                        trace_context: Some(&trace_context),
                        turn_id: None,
                    },
                    &control_rx,
                    &mut stdout,
                );
                turn_active.store(false, Ordering::SeqCst);
                while control_rx.try_recv().is_ok() {}
                if let Err(error) = result {
                    write_blocks_error(&mut stdout, &state.id, "turn", error.to_string())?;
                }
            }
            BlocksReaderInput::Command(BlocksCommand::Interrupt) => {}
            BlocksReaderInput::Command(BlocksCommand::AttachmentChunk) => {}
            BlocksReaderInput::Error(error) => {
                write_blocks_error(&mut stdout, &state.id, "input", error)?;
            }
        }
    }
    Ok(())
}

fn run_app_server() -> Result<()> {
    let profile = OmpProfile::load()?;
    let (request_tx, request_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let stdin = io::stdin();
        for raw in stdin.lock().lines() {
            let Ok(line) = raw else {
                break;
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let message = match serde_json::from_str::<JSONRPCMessage>(trimmed) {
                Ok(message) => message,
                Err(error) => {
                    eprintln!("invalid JSON-RPC message: {error}");
                    continue;
                }
            };
            let JSONRPCMessage::Request(request) = message else {
                continue;
            };
            if request_tx.send(RuntimeInput::JsonRpc(request)).is_err() {
                break;
            }
        }
    });

    let mut stdout = io::stdout().lock();
    let mut threads = HashMap::new();
    while let Ok(input) = request_rx.recv() {
        let RuntimeInput::JsonRpc(request) = input else {
            continue;
        };
        let request_id = request.id.clone();
        if handle_request(&profile, request, &request_rx, &mut threads, &mut stdout).is_err() {
            eprintln!("OMP request failed");
            write_error(
                &mut stdout,
                request_id,
                -32000,
                "OMP request failed".to_owned(),
            )?;
        }
    }
    Ok(())
}

fn handle_request<W: Write>(
    profile: &OmpProfile,
    request: JSONRPCRequest,
    request_rx: &Receiver<RuntimeInput>,
    threads: &mut HashMap<String, OmpThreadState>,
    stdout: &mut W,
) -> Result<()> {
    match request.method.as_str() {
        "initialize" => write_client_response(
            stdout,
            ClientResponse::Initialize {
                request_id: request.id,
                response: InitializeResponse {
                    user_agent: format!("harness-server omp/{OMP_VERSION}"),
                    codex_home: absolute_path(
                        env::var_os("CODEX_HOME")
                            .map(PathBuf::from)
                            .unwrap_or_else(default_codex_home),
                    )?,
                    platform_family: env::consts::FAMILY.to_string(),
                    platform_os: env::consts::OS.to_string(),
                },
            },
        ),
        "thread/start" => {
            let params: ThreadStartParams = request_params(request.params)?;
            let cwd = request_cwd(params.cwd.as_deref())?;
            let mut state = OmpThreadState {
                id: Uuid::new_v4().to_string(),
                cwd,
                model: params.model.clone().unwrap_or_default(),
                model_provider: params.model_provider.clone().unwrap_or_default(),
                service_tier: params.service_tier.clone().flatten(),
                completed_turns: Vec::new(),
                session: None,
                resume: None,
                thread_started_sent: false,
            };
            ensure_session(profile, &mut state)?;
            sync_model_from_session(&mut state);
            let thread_id = state.id.clone();
            let normalizer = normalizer_for(&state, "turn-placeholder");
            let response = ThreadStartResponse {
                thread: normalizer.thread_snapshot()?,
                model: state.model.clone(),
                model_provider: state.model_provider.clone(),
                service_tier: state.service_tier.clone(),
                cwd: absolute_path(state.cwd.clone())?,
                runtime_workspace_roots: Vec::new(),
                instruction_sources: Vec::new(),
                approval_policy: AskForApproval::Never,
                approvals_reviewer: ApprovalsReviewer::User,
                sandbox: SandboxPolicy::DangerFullAccess,
                active_permission_profile: None,
                reasoning_effort: None,
            };
            threads.insert(thread_id, state);
            write_client_response(
                stdout,
                ClientResponse::ThreadStart {
                    request_id: request.id,
                    response,
                },
            )
        }
        "thread/resume" => {
            let params: ThreadResumeParams = request_params(request.params)?;
            let thread_id = params.thread_id.clone();
            if !threads.contains_key(&thread_id) {
                let resume = load_resume_mapping(profile, &thread_id)?;
                let mut state = OmpThreadState {
                    id: thread_id.clone(),
                    cwd: request_cwd(params.cwd.as_deref())?,
                    model: params.model.clone().unwrap_or_default(),
                    model_provider: params.model_provider.clone().unwrap_or_default(),
                    service_tier: params.service_tier.clone().flatten(),
                    completed_turns: Vec::new(),
                    session: None,
                    resume: Some(resume),
                    thread_started_sent: false,
                };
                ensure_session(profile, &mut state)?;
                sync_model_from_session(&mut state);
                threads.insert(thread_id.clone(), state);
            }
            let state = threads
                .get_mut(&thread_id)
                .expect("OMP resume state inserted or existed");
            if let Some(model) = params.model.filter(|model| !model.is_empty()) {
                state.model = model;
            }
            if let Some(provider) = params
                .model_provider
                .filter(|provider| !provider.is_empty())
            {
                state.model_provider = provider;
            }
            let requested_provider = state.model_provider.clone();
            let requested_model = state.model.clone();
            state
                .session
                .as_mut()
                .expect("OMP resume session initialized")
                .ensure_model(&requested_provider, &requested_model)?;
            sync_model_from_session(state);
            let normalizer = normalizer_for(state, "turn-placeholder");
            let mut thread = normalizer.thread_snapshot()?;
            if !params.exclude_turns {
                thread.turns = state.completed_turns.clone();
            }
            write_client_response(
                stdout,
                ClientResponse::ThreadResume {
                    request_id: request.id,
                    response: ThreadResumeResponse {
                        thread,
                        model: state.model.clone(),
                        model_provider: state.model_provider.clone(),
                        service_tier: state.service_tier.clone(),
                        cwd: absolute_path(state.cwd.clone())?,
                        runtime_workspace_roots: Vec::new(),
                        instruction_sources: Vec::new(),
                        approval_policy: AskForApproval::Never,
                        approvals_reviewer: ApprovalsReviewer::User,
                        sandbox: SandboxPolicy::DangerFullAccess,
                        active_permission_profile: None,
                        reasoning_effort: None,
                        initial_turns_page: None,
                    },
                },
            )
        }
        "turn/start" => {
            let params: TurnStartParams = request_params(request.params)?;
            let state = threads.get_mut(&params.thread_id).ok_or_else(|| {
                HarnessServerError::UnknownThread {
                    thread_id: params.thread_id.clone(),
                }
            })?;
            let turn_id = format!("turn-{}", Uuid::new_v4().simple());
            let normalizer = normalizer_for(state, &turn_id);
            write_client_response(
                stdout,
                ClientResponse::TurnStart {
                    request_id: request.id,
                    response: TurnStartResponse {
                        turn: normalizer.turn_snapshot(TurnStatus::InProgress),
                    },
                },
            )?;
            run_normalized_turn(
                profile,
                state,
                &params.input,
                OmpTurnRequest {
                    client_user_message_id: params.client_user_message_id,
                    trace_context: None,
                    turn_id: Some(turn_id),
                },
                request_rx,
                stdout,
            )
        }
        "turn/interrupt" => {
            let _params: TurnInterruptParams = request_params(request.params)?;
            write_client_response(
                stdout,
                ClientResponse::TurnInterrupt {
                    request_id: request.id,
                    response: TurnInterruptResponse {},
                },
            )
        }
        "turn/steer" => write_error(
            stdout,
            request.id,
            -32600,
            "no active OMP turn to steer".to_string(),
        ),
        _ => write_error(
            stdout,
            request.id,
            -32601,
            format!("method not found: {}", request.method),
        ),
    }
}

struct OmpTurnRequest<'a> {
    client_user_message_id: Option<String>,
    trace_context: Option<&'a TraceContext>,
    turn_id: Option<String>,
}

fn run_normalized_turn<W: Write>(
    profile: &OmpProfile,
    state: &mut OmpThreadState,
    input: &[UserInput],
    request: OmpTurnRequest<'_>,
    request_rx: &Receiver<RuntimeInput>,
    stdout: &mut W,
) -> Result<()> {
    ensure_session(profile, state)?;
    let requested_provider = state.model_provider.clone();
    let requested_model = state.model.clone();
    state
        .session
        .as_mut()
        .expect("OMP session ensured")
        .ensure_model(&requested_provider, &requested_model)?;
    sync_model_from_session(state);
    let turn_id = request
        .turn_id
        .unwrap_or_else(|| format!("turn-{}", Uuid::new_v4().simple()));
    let normalizer = RefCell::new(normalizer_for(state, &turn_id));
    let mut telemetry = TurnTelemetry::new(
        request.trace_context,
        HarnessKind::Omp,
        state.model.clone(),
        state.model_provider.clone(),
        &turn_id,
        usage_span_input_value(input),
    );
    let output = RefCell::new(stdout);

    for notification in normalizer
        .borrow_mut()
        .start_notifications(!state.thread_started_sent)?
    {
        if matches!(notification, ServerNotification::ThreadStarted(_)) {
            state.thread_started_sent = true;
        }
        write_notification(&mut **output.borrow_mut(), &notification)?;
    }
    for notification in normalizer
        .borrow_mut()
        .emit_user_message(request.client_user_message_id, input.to_vec())?
    {
        write_notification(&mut **output.borrow_mut(), &notification)?;
    }

    let thread_id = state.id.clone();
    let active_turn_id = turn_id.clone();
    let session = state.session.as_mut().expect("OMP session ensured");
    let result = session.run_turn(
        input,
        || {
            loop {
                match request_rx.try_recv() {
                    Ok(RuntimeInput::BlocksInterrupt) => return Ok(Some(TurnControl::Interrupt)),
                    Ok(RuntimeInput::JsonRpc(request)) => match request.method.as_str() {
                        "turn/steer" => {
                            let params: TurnSteerParams = request_params(request.params)?;
                            if params.thread_id != thread_id
                                || params.expected_turn_id != active_turn_id
                            {
                                write_error(
                                    &mut **output.borrow_mut(),
                                    request.id,
                                    -32600,
                                    "OMP steer does not match the active thread/turn".to_string(),
                                )?;
                                continue;
                            }
                            for notification in normalizer.borrow_mut().emit_user_message(
                                params.client_user_message_id,
                                params.input.clone(),
                            )? {
                                write_notification(&mut **output.borrow_mut(), &notification)?;
                            }
                            write_client_response(
                                &mut **output.borrow_mut(),
                                ClientResponse::TurnSteer {
                                    request_id: request.id,
                                    response: TurnSteerResponse {
                                        turn_id: active_turn_id.clone(),
                                    },
                                },
                            )?;
                            return Ok(Some(TurnControl::Steer(params.input)));
                        }
                        "turn/interrupt" => {
                            let params: TurnInterruptParams = request_params(request.params)?;
                            if params.thread_id != thread_id || params.turn_id != active_turn_id {
                                write_error(
                                    &mut **output.borrow_mut(),
                                    request.id,
                                    -32600,
                                    "OMP interrupt does not match the active thread/turn"
                                        .to_string(),
                                )?;
                                continue;
                            }
                            write_client_response(
                                &mut **output.borrow_mut(),
                                ClientResponse::TurnInterrupt {
                                    request_id: request.id,
                                    response: TurnInterruptResponse {},
                                },
                            )?;
                            return Ok(Some(TurnControl::Interrupt));
                        }
                        _ => {
                            write_error(
                                &mut **output.borrow_mut(),
                                request.id,
                                -32600,
                                format!(
                                    "cannot handle {} while an OMP turn is active",
                                    request.method
                                ),
                            )?;
                        }
                    },
                    Err(TryRecvError::Empty) => return Ok(None),
                    Err(TryRecvError::Disconnected) => return Ok(None),
                }
            }
        },
        |event| {
            telemetry.observe_normalized(&event);
            for notification in normalizer.borrow_mut().process_event(&event)? {
                telemetry.observe_notification(&notification);
                write_notification(&mut **output.borrow_mut(), &notification)?;
            }
            Ok(())
        },
    );

    let outcome = match result {
        Ok(outcome) => outcome,
        Err(error) => {
            telemetry.finish(TelemetryTurnStatus::Failed);
            state.session = None;
            let message = error.to_string();
            for notification in normalizer
                .borrow_mut()
                .process_event(&NormalizedEvent::Error {
                    message: message.clone(),
                })?
            {
                write_notification(&mut **output.borrow_mut(), &notification)?;
            }
            finish_turn(
                state,
                &mut normalizer.borrow_mut(),
                &mut **output.borrow_mut(),
                Some(message),
                false,
            )?;
            return Ok(());
        }
    };
    telemetry.finish(if outcome.interrupted {
        TelemetryTurnStatus::Cancelled
    } else {
        TelemetryTurnStatus::Completed
    });
    finish_outcome(
        state,
        &mut normalizer.borrow_mut(),
        &mut **output.borrow_mut(),
        outcome,
    )
}

fn finish_outcome<W: Write>(
    state: &mut OmpThreadState,
    normalizer: &mut CodexTurnNormalizer,
    stdout: &mut W,
    outcome: TurnOutcome,
) -> Result<()> {
    let _usage = outcome.usage;
    finish_turn(state, normalizer, stdout, None, outcome.interrupted)
}

fn finish_turn<W: Write>(
    state: &mut OmpThreadState,
    normalizer: &mut CodexTurnNormalizer,
    stdout: &mut W,
    error: Option<String>,
    interrupted: bool,
) -> Result<()> {
    let notification = if interrupted {
        normalizer.finish_turn_interrupted()?
    } else {
        normalizer.finish_turn(error)?
    };
    if let Some(notification) = notification {
        if let ServerNotification::TurnCompleted(completed) = &notification {
            state.completed_turns.push(completed.turn.clone());
        }
        write_notification(stdout, &notification)?;
    }
    Ok(())
}

fn ensure_session(profile: &OmpProfile, state: &mut OmpThreadState) -> Result<()> {
    if state.session.is_none() {
        state.session = Some(OmpSession::start(
            profile.clone(),
            &state.id,
            &state.cwd,
            &state.model_provider,
            &state.model,
            state.resume.as_ref(),
        )?);
    }
    Ok(())
}

fn sync_model_from_session(state: &mut OmpThreadState) {
    let Some(session) = &state.session else {
        return;
    };
    if let Some(model) = session.state().get("model") {
        if let Some(provider) = model.get("provider").and_then(Value::as_str) {
            state.model_provider = provider.to_string();
        }
        if let Some(id) = model.get("id").and_then(Value::as_str) {
            state.model = id.to_string();
        }
    }
}

fn normalizer_for(state: &OmpThreadState, turn_id: &str) -> CodexTurnNormalizer {
    let mut config = BridgeConfig::new(state.id.clone(), turn_id.to_string());
    config.cwd = state.cwd.clone();
    config.cli_version = format!("omp-{OMP_VERSION}");
    config.model_provider = state.model_provider.clone();
    CodexTurnNormalizer::new(config)
}

fn request_cwd(cwd: Option<&str>) -> Result<PathBuf> {
    let cwd = cwd.map(PathBuf::from).unwrap_or(env::current_dir()?);
    if cwd.is_absolute() {
        Ok(cwd)
    } else {
        Ok(env::current_dir()?.join(cwd))
    }
}

fn request_params<T: serde::de::DeserializeOwned>(params: Option<Value>) -> Result<T> {
    serde_json::from_value(params.unwrap_or_else(|| json!({})))
        .map_err(|source| HarnessServerError::InvalidParams { source })
}

fn write_notification<W: Write>(stdout: &mut W, notification: &ServerNotification) -> Result<()> {
    write_value(stdout, &notification_to_wire_value(notification)?)
}

fn write_client_response<W: Write>(stdout: &mut W, response: ClientResponse) -> Result<()> {
    let (id, result) = response.into_jsonrpc_parts()?;
    write_value(
        stdout,
        &serde_json::to_value(JSONRPCMessage::Response(JSONRPCResponse { id, result }))?,
    )
}

fn write_error<W: Write>(stdout: &mut W, id: RequestId, code: i64, message: String) -> Result<()> {
    write_value(
        stdout,
        &serde_json::to_value(JSONRPCMessage::Error(JSONRPCError {
            id,
            error: JSONRPCErrorError {
                code,
                message,
                data: None,
            },
        }))?,
    )
}
