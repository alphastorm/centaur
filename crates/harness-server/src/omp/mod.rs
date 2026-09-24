mod normalize;
mod process;
mod protocol;
mod session;

use std::cell::RefCell;
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};

use codex_app_server_protocol::{ServerNotification, UserInput};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::otel::{TraceContext, TurnStatus as TelemetryTurnStatus, TurnTelemetry};
use crate::server::{
    BlocksCommand, BlocksState, parse_blocks_line_with_state, usage_span_input_value,
    write_blocks_error,
};
use crate::traits::{HarnessKind, NormalizedEvent};
use crate::turn::{BridgeConfig, CodexTurnNormalizer};
use crate::util::write_value;
use crate::wire::notification_to_wire_value;
use crate::{HarnessServerError, Result};

use self::session::{OmpSession, TurnControl};

struct OmpThread {
    id: String,
    cwd: PathBuf,
    session_dir: PathBuf,
    session: Option<OmpSession>,
    started: bool,
}

/// One blocks stream owns one OMP process and native session directory.
pub fn run_omp_blocks_server() -> Result<()> {
    let id = env::var("CENTAUR_THREAD_KEY")
        .ok()
        .filter(|id| !id.trim().is_empty())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let root = env::var_os("CENTAUR_OMP_SESSION_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env::var_os("HOME").unwrap_or_else(|| ".".into()))
                .join(".omp/centaur-sessions")
        });
    let session_dir = root.join(format!("{:x}", Sha256::digest(id.as_bytes())));
    fs::create_dir_all(&session_dir)?;
    let mut state = OmpThread {
        id,
        cwd: env::current_dir()?,
        session_dir,
        session: None,
        started: false,
    };
    let (command_tx, command_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let stdin = io::stdin();
        let mut blocks_state = BlocksState::default();
        for raw in stdin.lock().lines() {
            let Ok(line) = raw else { break };
            if line.trim().is_empty() {
                continue;
            }
            let command = parse_blocks_line_with_state(line.trim(), &mut blocks_state)
                .map_err(|error| error.to_string());
            if command_tx.send(command).is_err() {
                break;
            }
        }
    });

    let mut stdout = io::stdout().lock();
    while let Ok(command) = command_rx.recv() {
        match command {
            Ok(BlocksCommand::User {
                input,
                client_user_message_id,
                model,
                provider,
                trace_context,
                ..
            }) => {
                if let Err(error) = run_turn(
                    &mut state,
                    &input,
                    client_user_message_id,
                    provider.as_deref().unwrap_or_default(),
                    model.as_deref().unwrap_or_default(),
                    &trace_context,
                    &command_rx,
                    &mut stdout,
                ) {
                    write_blocks_error(&mut stdout, &state.id, "turn", error.to_string())?;
                }
            }
            Ok(BlocksCommand::Interrupt | BlocksCommand::AttachmentChunk) => {}
            Err(error) => write_blocks_error(&mut stdout, &state.id, "input", error)?,
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_turn<W: Write>(
    state: &mut OmpThread,
    input: &[UserInput],
    client_user_message_id: Option<String>,
    requested_provider: &str,
    requested_model: &str,
    trace_context: &TraceContext,
    commands: &Receiver<std::result::Result<BlocksCommand, String>>,
    stdout: &mut W,
) -> Result<()> {
    let setup = (|| {
        if let Some(session) = state.session.as_mut()
            && session.has_exited()?
        {
            state.session = None;
        }
        if let Some(session) = state.session.as_mut() {
            session.ensure_model(requested_provider, requested_model)?;
        } else {
            state.session = Some(OmpSession::start(
                &state.session_dir,
                &state.cwd,
                requested_provider,
                requested_model,
            )?);
        }
        Ok(())
    })();
    let turn_id = format!("turn-{}", Uuid::new_v4().simple());
    let model = state.session.as_ref().and_then(OmpSession::model);
    let mut config = BridgeConfig::new(state.id.clone(), turn_id.clone());
    config.cwd = state.cwd.clone();
    config.cli_version = "omp".to_string();
    config.model_provider = model
        .map(|model| model.provider.clone())
        .unwrap_or_default();
    let mut telemetry = TurnTelemetry::new(
        Some(trace_context),
        HarnessKind::Omp,
        model.map(|model| model.id.clone()).unwrap_or_default(),
        config.model_provider.clone(),
        &turn_id,
        usage_span_input_value(input),
    );
    let normalizer = RefCell::new(CodexTurnNormalizer::new(config));
    let output = RefCell::new(stdout);
    for notification in normalizer
        .borrow_mut()
        .start_notifications(!state.started)?
    {
        if matches!(notification, ServerNotification::ThreadStarted(_)) {
            state.started = true;
        }
        write_notification(&mut **output.borrow_mut(), &notification)?;
    }
    for notification in normalizer
        .borrow_mut()
        .emit_user_message(client_user_message_id, input.to_vec())?
    {
        write_notification(&mut **output.borrow_mut(), &notification)?;
    }
    let result = setup.and_then(|()| {
        state
            .session
            .as_mut()
            .expect("OMP session started")
            .run_turn(
                input,
                || loop {
                    match commands.try_recv() {
                        Ok(Ok(BlocksCommand::Interrupt)) => {
                            return Ok(Some(TurnControl::Interrupt));
                        }
                        Ok(Ok(BlocksCommand::User {
                            input,
                            client_user_message_id,
                            ..
                        })) => {
                            for notification in normalizer
                                .borrow_mut()
                                .emit_user_message(client_user_message_id, input.clone())?
                            {
                                write_notification(&mut **output.borrow_mut(), &notification)?;
                            }
                            return Ok(Some(TurnControl::Steer(input)));
                        }
                        Ok(Ok(BlocksCommand::AttachmentChunk)) => continue,
                        Ok(Err(error)) => return Err(HarnessServerError::Protocol(error)),
                        Err(TryRecvError::Empty | TryRecvError::Disconnected) => return Ok(None),
                    }
                },
                |event| {
                    telemetry.observe_normalized(&event);
                    for notification in normalizer.borrow_mut().process_event(&event)? {
                        let value = notification_to_wire_value(&notification)?;
                        telemetry.observe_tool_notification(&value);
                        write_value(&mut **output.borrow_mut(), &value)?;
                    }
                    Ok(())
                },
            )
    });
    let terminal = match result {
        Ok(outcome) if outcome.interrupted => {
            telemetry.finish(TelemetryTurnStatus::Cancelled);
            normalizer.borrow_mut().finish_turn_interrupted()?
        }
        Ok(_) => {
            telemetry.finish(TelemetryTurnStatus::Completed);
            normalizer.borrow_mut().finish_turn(None)?
        }
        Err(error) => {
            telemetry.finish(TelemetryTurnStatus::Failed);
            // Drop waits for the child before publishing a non-retry failure.
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
            normalizer.borrow_mut().finish_turn(Some(message))?
        }
    };
    if let Some(notification) = terminal {
        write_notification(&mut **output.borrow_mut(), &notification)?;
    }
    Ok(())
}

fn write_notification<W: Write>(stdout: &mut W, notification: &ServerNotification) -> Result<()> {
    write_value(stdout, &notification_to_wire_value(notification)?)
}
