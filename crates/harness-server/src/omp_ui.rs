//! Tool dialogs are process-local waits, not approvals or resumable tool calls.
//! OMP 18.11 does not put a toolCallId on UI frames. Bind only while exactly one
//! tool is active; guessing during concurrent tools would answer the wrong call.

use std::collections::{HashMap, HashSet};

use codex_app_server_protocol::{
    RequestId, ToolRequestUserInputOption, ToolRequestUserInputQuestion,
    ToolRequestUserInputResponse,
};
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::omp::protocol_error;
use crate::pi::command_line;
use crate::{NormalizedEvent, Result};

#[derive(Debug, Default)]
pub(super) struct ToolUi {
    active_tools: HashSet<String>,
    pending: HashMap<String, PendingDialog>,
    seen: HashSet<String>,
}

#[derive(Debug)]
struct PendingDialog {
    native_id: String,
    tool_call_id: String,
    method: String,
    options: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Dialog {
    id: String,
    method: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    message: String,
    #[serde(default)]
    options: Vec<String>,
    #[serde(default)]
    option_details: Vec<OptionDetail>,
}

#[derive(Deserialize)]
struct OptionDetail {
    #[serde(default)]
    description: String,
}

impl ToolUi {
    pub(super) fn tool_started(&mut self, id: &str) {
        self.active_tools.insert(id.to_owned());
    }

    pub(super) fn tool_ended(&mut self, id: &str) -> Vec<NormalizedEvent> {
        self.active_tools.remove(id);
        let mut expired = Vec::new();
        self.pending.retain(|request_id, dialog| {
            if dialog.tool_call_id != id {
                return true;
            }
            expired.push(NormalizedEvent::UserInputResolved {
                request_id: request_id.clone(),
            });
            false
        });
        expired
    }

    pub(super) fn request(&mut self, frame: Value) -> Result<Vec<NormalizedEvent>> {
        if frame["method"] == "cancel" {
            let target = frame["targetId"].as_str().unwrap_or_default();
            let id = self
                .pending
                .iter()
                .find_map(|(id, dialog)| (dialog.native_id == target).then(|| id.clone()));
            return Ok(id
                .into_iter()
                .map(|request_id| {
                    self.pending.remove(&request_id);
                    NormalizedEvent::UserInputResolved { request_id }
                })
                .collect());
        }
        // Presentation events are data, not requests and never commands.
        if matches!(
            frame["method"].as_str(),
            Some(
                "notify" | "setStatus" | "setWidget" | "setTitle" | "set_editor_text" | "open_url"
            )
        ) {
            return Ok(Vec::new());
        }
        let mut dialog: Dialog = serde_json::from_value(frame)?;
        if !matches!(
            dialog.method.as_str(),
            "select" | "input" | "editor" | "confirm"
        ) {
            return Err(protocol_error("unsupported tool UI dialog"));
        }
        if dialog.id.is_empty() || !self.seen.insert(dialog.id.clone()) {
            return Err(protocol_error("duplicate or empty tool UI request id"));
        }
        if self.active_tools.len() != 1 {
            return Err(protocol_error(format!(
                "tool UI request {} expired: ambiguous tool-call binding ({} active tools)",
                dialog.id,
                self.active_tools.len()
            )));
        }
        let tool_call_id = self
            .active_tools
            .iter()
            .next()
            .expect("one active tool")
            .clone();
        if dialog.method == "confirm" {
            dialog.options = vec!["Yes".into(), "No".into()];
        }
        let options = if dialog.options.is_empty() {
            None
        } else {
            Some(
                dialog
                    .options
                    .iter()
                    .enumerate()
                    .map(|(index, label)| ToolRequestUserInputOption {
                        label: label.clone(),
                        description: dialog
                            .option_details
                            .get(index)
                            .map(|detail| detail.description.clone())
                            .unwrap_or_default(),
                    })
                    .collect(),
            )
        };
        let question = ToolRequestUserInputQuestion {
            id: dialog.id.clone(),
            header: dialog.title.clone(),
            question: if dialog.message.is_empty() {
                dialog.title
            } else {
                dialog.message
            },
            // Select's explicit Other option leads to OMP's subsequent editor
            // dialog; never substitute arbitrary text for a native option.
            is_other: matches!(dialog.method.as_str(), "input" | "editor"),
            is_secret: false,
            options,
        };
        // Scope the host id to this wait, not OMP's restartable native id.
        let request_id = format!("omp-ui-{}", Uuid::new_v4());
        self.pending.insert(
            request_id.clone(),
            PendingDialog {
                native_id: dialog.id,
                tool_call_id: tool_call_id.clone(),
                method: dialog.method,
                options: dialog.options,
            },
        );
        Ok(vec![NormalizedEvent::UserInputRequest {
            request_id,
            tool_call_id,
            questions: vec![question],
        }])
    }

    pub(super) fn answer(&mut self, id: &RequestId, result: Option<Value>) -> Result<Vec<u8>> {
        let RequestId::String(id) = id else {
            return Err(protocol_error("unknown or expired tool UI request"));
        };
        let dialog = self
            .pending
            .get(id)
            .ok_or_else(|| protocol_error("unknown or expired tool UI request"))?;
        let frame = match result {
            None => {
                json!({"type": "extension_ui_response", "id": dialog.native_id, "cancelled": true})
            }
            Some(result) => {
                let mut response: ToolRequestUserInputResponse = serde_json::from_value(result)?;
                if response.answers.len() != 1 {
                    return Err(protocol_error("expected exactly one tool UI answer"));
                }
                let answer = response
                    .answers
                    .remove(&dialog.native_id)
                    .ok_or_else(|| protocol_error("tool UI question id mismatch"))?;
                if answer.answers.len() != 1 {
                    return Err(protocol_error("expected exactly one tool UI value"));
                }
                let value = &answer.answers[0];
                if matches!(dialog.method.as_str(), "select" | "confirm")
                    && !dialog.options.contains(value)
                {
                    return Err(protocol_error("unknown tool UI option"));
                }
                if dialog.method == "confirm" {
                    json!({"type": "extension_ui_response", "id": dialog.native_id, "confirmed": value == "Yes"})
                } else {
                    json!({"type": "extension_ui_response", "id": dialog.native_id, "value": value})
                }
            }
        };
        let bytes = command_line(frame)?;
        // Consume before writing: an uncertain write must never be replayed.
        self.pending.remove(id);
        Ok(bytes)
    }

    pub(super) fn expire(&mut self) -> Vec<String> {
        self.pending.drain().map(|(id, _)| id).collect()
    }
}
