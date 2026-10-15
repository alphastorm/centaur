use centaur_session_core::{UserInputAnswers, UserInputRequest};
use centaur_session_sqlx::NewUserInputRequest;

use super::*;

impl SessionRuntime {
    pub async fn user_input_requests(
        &self,
        thread_key: &ThreadKey,
        execution_id: &str,
    ) -> Result<Vec<UserInputRequest>, SessionRuntimeError> {
        let execution = self.store.get_execution(execution_id).await?;
        if execution.thread_key != *thread_key {
            return Err(SessionStoreError::UserInputNotFound.into());
        }
        Ok(self.store.user_input_requests(execution_id).await?)
    }

    pub async fn answer_user_input_request(
        &self,
        execution_id: &str,
        request_id: &str,
        answers: UserInputAnswers,
        actor: &str,
    ) -> Result<(), SessionRuntimeError> {
        self.respond_to_user_input_request(execution_id, request_id, Some(answers), actor)
            .await
    }

    pub async fn cancel_user_input_request(
        &self,
        execution_id: &str,
        request_id: &str,
        actor: &str,
    ) -> Result<(), SessionRuntimeError> {
        self.respond_to_user_input_request(execution_id, request_id, None, actor)
            .await
    }

    async fn respond_to_user_input_request(
        &self,
        execution_id: &str,
        request_id: &str,
        answers: Option<UserInputAnswers>,
        actor: &str,
    ) -> Result<(), SessionRuntimeError> {
        let conflict = || SessionRuntimeError::Store(SessionStoreError::UserInputConflict);
        let execution = self.store.get_execution(execution_id).await?;
        let session = self.store.get_session(&execution.thread_key).await?;
        let sandbox_id = session.sandbox_id.as_deref().ok_or_else(conflict)?;
        // Unlike steering, answering must not ensure/open/resume a pipe.
        let pipe = self
            .sandbox_pipes
            .get(sandbox_id)
            .map(|entry| entry.clone())
            .ok_or_else(conflict)?;
        let mut live = pipe.response_gate.lock().await;
        if !*live || execution.status != ExecutionStatus::Running {
            return Err(conflict());
        }
        let observed = self
            .sandbox_runtime
            .manager
            .observe(&SandboxId::new(sandbox_id))
            .await;
        if !matches!(observed, Ok(ref observed)
            if observed.status.can_open_io() && observed.instance_id == pipe.instance_id)
        {
            *live = false;
            self.store
                .expire_pipe_user_input_requests(&execution.thread_key, &pipe.pipe_id, false)
                .await?;
            return Err(conflict());
        }
        let requests = self.store.user_input_requests(execution_id).await?;
        if let (Some(answers), Some(request)) = (
            answers.as_ref(),
            requests.iter().find(|r| r.request_id == request_id),
        ) {
            if request.status != "pending" {
                return Err(conflict());
            }
            let questions = request.questions.as_array().ok_or_else(conflict)?;
            if answers.len() != questions.len()
                || questions.iter().any(|question| {
                    question["id"]
                        .as_str()
                        .is_none_or(|id| !answers.contains_key(id))
                })
            {
                return Err(SessionRuntimeError::BadRequest(
                    "answers must contain exactly the requested question ids".to_owned(),
                ));
            }
        }
        let answer_digest = answers.as_ref().map(|answers| {
            // BTreeMap gives the digest a deterministic key order.
            let encoded = serde_json::to_vec(answers).expect("answers serialize");
            hex::encode(Sha256::digest(encoded))
        });
        let rpc_id = self
            .store
            .consume_user_input_request(
                execution_id,
                request_id,
                &self.stdout_owner_id,
                &pipe.pipe_id,
                answer_digest.as_deref(),
                actor,
            )
            .await?;
        let response = match answers {
            Some(answers) => json!({"id": rpc_id, "result": {"answers": answers}}),
            None => json!({"id": rpc_id, "error": {"code": -32000, "message": "User declined"}}),
        };
        // No user-message wrapper, session context, or trace fields in a JSON-RPC response.
        if let Err(error) = write_input_lines(
            &pipe,
            &[response.to_string()],
            &execution.thread_key,
            execution_id,
            Some(sandbox_id),
        )
        .await
        {
            self.store
                .fail_user_input_delivery(execution_id, request_id)
                .await?;
            return Err(error);
        }
        Ok(())
    }
}

fn request_id(value: &Value) -> Option<String> {
    match value {
        Value::String(id) => Some(id.clone()),
        Value::Number(id) if id.is_i64() || id.is_u64() => Some(id.to_string()),
        _ => None,
    }
}

pub(super) fn output_request_id(value: &Value) -> Option<String> {
    match value["method"].as_str() {
        Some("item/tool/requestUserInput") => request_id(&value["id"]),
        Some("serverRequest/resolved") => request_id(&value["params"]["requestId"]),
        _ => None,
    }
}

pub(super) async fn record_request_output(
    ctx: &RuntimeContext,
    thread_key: &ThreadKey,
    execution_id: &str,
    pipe_id: &str,
    line: &str,
) -> Result<(), SessionRuntimeError> {
    let Ok(value) = serde_json::from_str::<Value>(line) else {
        return Ok(());
    };
    match value["method"].as_str() {
        Some("item/tool/requestUserInput") => {
            let params = &value["params"];
            let (Some(id), Some(thread_id), Some(turn_id), Some(item_id), Some(questions)) = (
                request_id(&value["id"]),
                params["threadId"].as_str(),
                params["turnId"].as_str(),
                params["itemId"].as_str(),
                params["questions"].as_array(),
            ) else {
                return Ok(());
            };
            let mut questions = Value::Array(questions.clone());
            redact_question_strings(&mut questions);
            ctx.store
                .record_user_input_request(
                    thread_key,
                    execution_id,
                    &ctx.stdout_owner_id,
                    pipe_id,
                    &NewUserInputRequest {
                        request_id: id,
                        rpc_id: value["id"].clone(),
                        thread_id: thread_id.to_owned(),
                        turn_id: turn_id.to_owned(),
                        item_id: item_id.to_owned(),
                        questions,
                    },
                )
                .await?;
        }
        Some("serverRequest/resolved") => {
            if let Some(id) = request_id(&value["params"]["requestId"]) {
                ctx.store
                    .resolve_user_input_request(execution_id, &id, pipe_id)
                    .await?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn redact_question_strings(value: &mut Value) {
    match value {
        Value::String(text) => *text = redact_sensitive_text(text),
        Value::Array(values) => values.iter_mut().for_each(redact_question_strings),
        Value::Object(values) => {
            for (key, value) in values {
                // Question ids are routing keys, not display text. Redacting
                // them would make the answer target a different question.
                if key != "id" {
                    redact_question_strings(value);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn late_resolution_stays_bound_to_the_original_execution() {
        let mut state = StdoutPumpState::default();
        let request = json!({"id":"request-1","method":"item/tool/requestUserInput",
            "params":{"turnId":"turn-1","itemId":"item-1"}})
        .to_string();
        assert_eq!(
            state.execution_for_line(Some("first"), &request).as_deref(),
            Some("first")
        );
        state.forget("first");
        let resolved = json!({"method":"serverRequest/resolved",
            "params":{"requestId":"request-1","threadId":"thread-1"}})
        .to_string();
        assert_eq!(state.execution_for_line(Some("second"), &resolved), None);
        assert_eq!(state.execution_for_line(Some("second"), &request), None);
    }

    #[test]
    fn question_redaction_preserves_json_structure() {
        let mut questions = json!([{"id":"sk-question-id","question":"bearer secret-token","options":[
            {"label":"safe", "description":"API_TOKEN=secret-token"}
        ]}]);
        redact_question_strings(&mut questions);
        assert_eq!(questions[0]["id"], "sk-question-id");
        assert!(!questions.to_string().contains("secret-token"));
        assert_eq!(questions[0]["options"][0]["label"], "safe");
    }
}
