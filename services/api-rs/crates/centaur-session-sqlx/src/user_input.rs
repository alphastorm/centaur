use centaur_session_core::{ThreadKey, UserInputRequest};
use serde_json::Value;
use sqlx::FromRow;
use time::OffsetDateTime;

use crate::{PgSessionStore, SessionStoreError};

pub struct NewUserInputRequest {
    pub request_id: String,
    pub rpc_id: Value,
    pub thread_id: String,
    pub turn_id: String,
    pub item_id: String,
    pub questions: Value,
}

#[derive(FromRow)]
struct RequestRow {
    execution_id: String,
    request_id: String,
    thread_id: String,
    turn_id: String,
    item_id: String,
    questions: Value,
    status: String,
    reason: Option<String>,
    created_at: OffsetDateTime,
    resolved_at: Option<OffsetDateTime>,
}

impl From<RequestRow> for UserInputRequest {
    fn from(row: RequestRow) -> Self {
        Self {
            execution_id: row.execution_id,
            request_id: row.request_id,
            thread_id: row.thread_id,
            turn_id: row.turn_id,
            item_id: row.item_id,
            questions: row.questions,
            status: row.status,
            reason: row.reason,
            created_at: row.created_at,
            resolved_at: row.resolved_at,
        }
    }
}

impl PgSessionStore {
    pub async fn record_user_input_request(
        &self,
        thread_key: &ThreadKey,
        execution_id: &str,
        owner_id: &str,
        pipe_id: &str,
        request: &NewUserInputRequest,
    ) -> Result<(), SessionStoreError> {
        let mut tx = self.pool().begin().await?;
        // Lock the execution before the request, matching terminal transitions.
        let active = sqlx::query_scalar::<_, String>(
            "select execution_id from session_executions
             where execution_id = $1 and thread_key = $2 and status = 'running'
               and stdout_owner_id = $3 for update",
        )
        .bind(execution_id)
        .bind(thread_key.as_str())
        .bind(owner_id)
        .fetch_optional(&mut *tx)
        .await?;
        if active.is_some() {
            sqlx::query(
                "insert into session_user_input_requests
                 (execution_id, request_id, rpc_id, thread_key, thread_id, turn_id, item_id, pipe_id, questions)
                 values ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                 on conflict (execution_id, request_id) do nothing",
            )
            .bind(execution_id)
            .bind(&request.request_id)
            .bind(&request.rpc_id)
            .bind(thread_key.as_str())
            .bind(&request.thread_id)
            .bind(&request.turn_id)
            .bind(&request.item_id)
            .bind(pipe_id)
            .bind(&request.questions)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn user_input_requests(
        &self,
        execution_id: &str,
    ) -> Result<Vec<UserInputRequest>, SessionStoreError> {
        let rows = sqlx::query_as::<_, RequestRow>(
            "select execution_id, request_id, thread_id, turn_id, item_id, questions,
                    status, reason, created_at, resolved_at
             from session_user_input_requests where execution_id = $1
             order by created_at, request_id",
        )
        .bind(execution_id)
        .fetch_all(self.pool())
        .await?;
        Ok(rows.into_iter().map(Into::into).collect())
    }

    /// Consume before attempting stdin delivery. A transaction cannot atomically
    /// commit with a pipe write; never roll this back after a possibly delivered
    /// response, and never replay an answer into a replacement pipe.
    pub async fn consume_user_input_request(
        &self,
        execution_id: &str,
        request_id: &str,
        owner_id: &str,
        pipe_id: &str,
        answer_digest: Option<&str>,
        actor: &str,
    ) -> Result<Value, SessionStoreError> {
        let mut tx = self.pool().begin().await?;
        let active = sqlx::query_scalar::<_, String>(
            "select execution_id from session_executions
             where execution_id = $1 and status = 'running' and stdout_owner_id = $2
               and stdout_owner_lease_expires_at > now() for update",
        )
        .bind(execution_id)
        .bind(owner_id)
        .fetch_optional(&mut *tx)
        .await?;
        if active.is_none() {
            return Err(SessionStoreError::UserInputConflict);
        }
        let row = sqlx::query_as::<_, (Value, String, String)>(
            "select rpc_id, status, pipe_id from session_user_input_requests
             where execution_id = $1 and request_id = $2 for update",
        )
        .bind(execution_id)
        .bind(request_id)
        .fetch_optional(&mut *tx)
        .await?;
        let Some((rpc_id, status, original_pipe)) = row else {
            // Foreign execution requests are conflicts, without disclosing the
            // execution to which the id belongs.
            let exists: bool = sqlx::query_scalar(
                "select exists(select 1 from session_user_input_requests where request_id = $1)",
            )
            .bind(request_id)
            .fetch_one(&mut *tx)
            .await?;
            return Err(if exists {
                SessionStoreError::UserInputConflict
            } else {
                SessionStoreError::UserInputNotFound
            });
        };
        if status != "pending" || original_pipe != pipe_id {
            return Err(SessionStoreError::UserInputConflict);
        }
        sqlx::query(
            "update session_user_input_requests
             set status = $3, reason = $4, answer_digest = $5, actor = $6, resolved_at = now()
             where execution_id = $1 and request_id = $2",
        )
        .bind(execution_id)
        .bind(request_id)
        .bind(if answer_digest.is_some() {
            "answered"
        } else {
            "cancelled"
        })
        .bind(if answer_digest.is_some() {
            None
        } else {
            Some("user_declined")
        })
        .bind(answer_digest)
        .bind(actor)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(rpc_id)
    }

    pub async fn resolve_user_input_request(
        &self,
        execution_id: &str,
        request_id: &str,
        pipe_id: &str,
    ) -> Result<(), SessionStoreError> {
        let mut tx = self.pool().begin().await?;
        lock_execution(&mut tx, execution_id).await?;
        sqlx::query(
            "update session_user_input_requests
             set status = 'resolved', reason = 'server_resolved', resolved_at = now()
             where execution_id = $1 and request_id = $2 and pipe_id = $3 and status = 'pending'",
        )
        .bind(execution_id)
        .bind(request_id)
        .bind(pipe_id)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn expire_pipe_user_input_requests(
        &self,
        thread_key: &ThreadKey,
        pipe_id: &str,
        keep_pipe: bool,
    ) -> Result<(), SessionStoreError> {
        let mut tx = self.pool().begin().await?;
        // Every transition locks execution -> request -> event. In particular,
        // multi-request expiration must not deadlock with a single resolution
        // holding another request while waiting for the per-thread event lock.
        sqlx::query(
            "select e.execution_id from session_executions e
             where e.thread_key = $1 and (e.status in ('queued', 'running') or exists (
                 select 1 from session_user_input_requests r
                 where r.execution_id = e.execution_id and r.status = 'pending'))
             order by e.execution_id for update of e",
        )
        .bind(thread_key.as_str())
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "update session_user_input_requests
             set status = 'expired', reason = 'pipe_lost', resolved_at = now()
             where thread_key = $1 and status = 'pending'
               and (($3 and pipe_id <> $2) or (not $3 and pipe_id = $2))",
        )
        .bind(thread_key.as_str())
        .bind(pipe_id)
        .bind(keep_pipe)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn fail_user_input_delivery(
        &self,
        execution_id: &str,
        request_id: &str,
    ) -> Result<(), SessionStoreError> {
        let mut tx = self.pool().begin().await?;
        lock_execution(&mut tx, execution_id).await?;
        sqlx::query(
            "update session_user_input_requests
             set status = 'delivery_failed', reason = 'response_write_failed'
             where execution_id = $1 and request_id = $2 and status in ('answered', 'cancelled')",
        )
        .bind(execution_id)
        .bind(request_id)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }
}

async fn lock_execution(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    execution_id: &str,
) -> Result<(), sqlx::Error> {
    sqlx::query("select execution_id from session_executions where execution_id = $1 for update")
        .bind(execution_id)
        .execute(&mut **tx)
        .await?;
    Ok(())
}
