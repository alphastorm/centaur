//! Tool dialogs are process-local waits, not approvals or resumable tool calls.
//! OMP 18.11 does not put a toolCallId on UI frames. Its stock ask can emit
//! message_end(toolCall), extension_ui_request, then tool_execution_start.
//! Bind only when exactly one announced/active tool is outstanding, deferring
//! the host request until that tool starts; never guess during concurrent tools.
//! Recorded stock ask cancellation ends with prompt_result status "aborted";
//! it follows the existing prompt-error path, not a successful empty answer.
//! Stock ask's custom input uses editor; input/confirm remain protocol support.

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
    announced_tools: HashSet<String>,
    active_tools: HashSet<String>,
    deferred: Vec<Dialog>,
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

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
struct OptionDetail {
    #[serde(default)]
    description: String,
}

impl ToolUi {
    pub(super) fn tool_announced(&mut self, id: &str) {
        if !self.active_tools.contains(id) {
            self.announced_tools.insert(id.to_owned());
        }
    }

    pub(super) fn tool_started(&mut self, id: &str) -> Result<Vec<NormalizedEvent>> {
        self.announced_tools.remove(id);
        self.active_tools.insert(id.to_owned());
        let mut events = Vec::new();
        if !self.deferred.is_empty() {
            self.check_binding()?;
            for dialog in std::mem::take(&mut self.deferred) {
                events.extend(self.bind(dialog, id.to_owned())?);
            }
        }
        Ok(events)
    }

    pub(super) fn tool_ended(&mut self, id: &str) -> Vec<NormalizedEvent> {
        if self.announced_tools.remove(id) {
            self.deferred.clear();
        }
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
            self.deferred.retain(|dialog| dialog.id != target);
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
        let dialog: Dialog = serde_json::from_value(frame)?;
        if !matches!(
            dialog.method.as_str(),
            "select" | "input" | "editor" | "confirm"
        ) {
            return Err(protocol_error("unsupported tool UI dialog"));
        }
        if dialog.id.is_empty() || !self.seen.insert(dialog.id.clone()) {
            return Err(protocol_error("duplicate or empty tool UI request id"));
        }
        self.check_binding()?;
        if self.active_tools.is_empty() {
            self.deferred.push(dialog);
            return Ok(Vec::new());
        }
        let tool_call_id = self
            .active_tools
            .iter()
            .next()
            .expect("one active tool")
            .clone();
        self.bind(dialog, tool_call_id)
    }

    fn check_binding(&self) -> Result<()> {
        let count = self.active_tools.len() + self.announced_tools.len();
        if count != 1 {
            return Err(protocol_error(format!(
                "tool UI request expired: ambiguous tool-call binding ({count} candidate tools)"
            )));
        }
        Ok(())
    }

    fn bind(&mut self, mut dialog: Dialog, tool_call_id: String) -> Result<Vec<NormalizedEvent>> {
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
        self.deferred.clear();
        self.pending.drain().map(|(id, _)| id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recorded_select() -> Value {
        include_str!("../tests/fixtures/omp/ask_select.jsonl")
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .filter(|frame| frame["stream"] == "stdout")
            .map(|frame| serde_json::from_str::<Value>(frame["line"].as_str().unwrap()).unwrap())
            .find(|frame| frame["type"] == "extension_ui_request")
            .unwrap()
    }

    #[test]
    fn announced_tools_do_not_make_ambiguous_dialogs_safe() {
        for started in [false, true] {
            let mut ui = ToolUi::default();
            ui.tool_announced("call_ask");
            ui.tool_announced("second");
            if started {
                assert!(ui.tool_started("call_ask").unwrap().is_empty());
            }
            assert!(ui.request(recorded_select()).is_err());
            assert!(ui.expire().is_empty());
        }
        let mut ui = ToolUi::default();
        ui.tool_announced("call_ask");
        assert!(ui.request(recorded_select()).unwrap().is_empty());
        assert!(ui.tool_started("unexpected").is_err());
        assert!(ui.expire().is_empty());
    }

    #[test]
    fn deferred_dialogs_reject_duplicates_and_expire_before_start() {
        for cancel in [true, false] {
            let mut ui = ToolUi::default();
            ui.tool_announced("call_ask");
            let select = recorded_select();
            assert!(ui.request(select.clone()).unwrap().is_empty());
            assert!(ui.request(select.clone()).is_err());
            if cancel {
                assert!(
                    ui.request(json!({"method": "cancel", "targetId": select["id"]}))
                        .unwrap()
                        .is_empty()
                );
            } else {
                assert!(ui.expire().is_empty());
            }
            assert!(ui.tool_started("call_ask").unwrap().is_empty());
        }
    }
}
