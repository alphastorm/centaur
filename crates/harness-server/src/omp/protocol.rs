use serde::Deserialize;
use serde_json::Value;

use crate::{HarnessServerError, Result};

// OMP's v1 encoder limits JSON frames to 1 MiB; leave room for line endings.
pub(crate) const MAX_FRAME_BYTES: usize = 1024 * 1024 + 1024;

#[derive(Debug, Deserialize)]
pub(crate) struct Response {
    pub(crate) id: String,
    pub(crate) command: String,
    pub(crate) success: bool,
    #[serde(default)]
    pub(crate) data: Value,
    pub(crate) error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Model {
    pub(crate) provider: String,
    pub(crate) id: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct State {
    pub(crate) model: Option<Model>,
    pub(crate) is_streaming: bool,
}

pub(crate) fn parse_ready(frame: Value) -> Result<()> {
    #[derive(Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    enum Greeting {
        Ready {
            #[serde(rename = "protocolVersion")]
            protocol_version: u64,
        },
    }
    let Greeting::Ready { protocol_version } = serde_json::from_value(frame)?;
    if protocol_version != 1 {
        return Err(protocol_error("expected OMP RPC protocol v1"));
    }
    Ok(())
}

pub(crate) fn frame_type(value: &Value) -> Result<&str> {
    value
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| protocol_error("OMP frame is missing string type"))
}

pub(crate) fn is_state_notification(kind: &str) -> bool {
    matches!(
        kind,
        "available_commands_update"
            | "config_update"
            | "session_info_update"
            | "thinking_level_changed"
            | "model_changed"
            | "config_warnings_changed"
            | "advisor_cost_changed"
    )
}

pub(crate) fn protocol_error(message: impl Into<String>) -> HarnessServerError {
    HarnessServerError::Protocol(format!("OMP protocol error: {}", message.into()))
}
