create table session_user_input_requests (
    execution_id text not null references session_executions(execution_id) on delete cascade,
    request_id text not null,
    rpc_id jsonb not null,
    thread_key text not null references sessions(thread_key) on delete cascade,
    thread_id text not null,
    turn_id text not null,
    item_id text not null,
    pipe_id text not null,
    questions jsonb not null,
    status text not null default 'pending'
        check (status in ('pending', 'answered', 'cancelled', 'resolved', 'expired', 'delivery_failed')),
    reason text,
    answer_digest text,
    actor text,
    created_at timestamptz not null default now(),
    resolved_at timestamptz,
    primary key (execution_id, request_id)
);

create index session_user_input_requests_pending_idx
    on session_user_input_requests (thread_key, pipe_id) where status = 'pending';

-- Use the same per-thread lock as insert_session_event: event ids must commit
-- in order for after_event_id replay. No response values enter the event log.
create function emit_session_user_input_event() returns trigger language plpgsql as $$
begin
    perform pg_advisory_xact_lock(
        hashtextextended('centaur:session-events:' || new.thread_key, 0)
    );
    if tg_op = 'INSERT' then
        insert into session_events (thread_key, execution_id, event_type, payload)
        values (new.thread_key, new.execution_id, 'session.user_input_requested',
            jsonb_build_object('execution_id', new.execution_id, 'request_id', new.request_id,
                'thread_id', new.thread_id, 'turn_id', new.turn_id, 'item_id', new.item_id,
                'questions', new.questions));
    elsif old.status is distinct from new.status then
        insert into session_events (thread_key, execution_id, event_type, payload)
        values (new.thread_key, new.execution_id, 'session.user_input_resolved',
            jsonb_build_object('execution_id', new.execution_id, 'request_id', new.request_id,
                'status', new.status, 'reason', new.reason));
    end if;
    return new;
end;
$$;

create trigger session_user_input_event
after insert or update of status on session_user_input_requests
for each row execute function emit_session_user_input_event();

-- Covers every terminal path, including recovery and max-duration failure,
-- in the execution transition's transaction rather than a best-effort cleanup.
create function expire_execution_user_input_requests() returns trigger language plpgsql as $$
begin
    if new.status not in ('queued', 'running') and old.status in ('queued', 'running') then
        update session_user_input_requests
        set status = 'expired', reason = 'execution_' || new.status, resolved_at = now()
        where execution_id = new.execution_id and status = 'pending';
    end if;
    return new;
end;
$$;

create trigger session_execution_expire_user_input
after update of status on session_executions
for each row execute function expire_execution_user_input_requests();
