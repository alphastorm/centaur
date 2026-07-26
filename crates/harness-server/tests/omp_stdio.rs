use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use uuid::Uuid;

const TIMEOUT: Duration = Duration::from_secs(10);

struct Bridge {
    child: Child,
    stdin: ChildStdin,
    output: Receiver<Value>,
    session_root: PathBuf,
}

impl Bridge {
    fn spawn(mode: &str, session_root: PathBuf) -> Self {
        let binary = env!("CARGO_BIN_EXE_harness-server");
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/fake_omp_rpc.py")
            .canonicalize()
            .unwrap();
        let mut child = Command::new(binary)
            .args(if mode == "blocks" {
                vec!["omp"]
            } else {
                vec!["omp", "--mode", "jsonrpc"]
            })
            .env("CENTAUR_OMP_BIN", fixture)
            .env("CENTAUR_OMP_SESSION_ROOT", &session_root)
            .env("CENTAUR_OMP_ENABLED", "1")
            .env("CENTAUR_OMP_ALLOWED_MODELS", "fake/fake-model")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, output) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if let Ok(value) = serde_json::from_str(&line) {
                    if tx.send(value).is_err() {
                        break;
                    }
                }
            }
        });
        Self {
            child,
            stdin,
            output,
            session_root,
        }
    }

    fn send(&mut self, value: Value) {
        serde_json::to_writer(&mut self.stdin, &value).unwrap();
        self.stdin.write_all(b"\n").unwrap();
        self.stdin.flush().unwrap();
    }

    fn recv_until(&self, mut predicate: impl FnMut(&Value) -> bool) -> Vec<Value> {
        let deadline = Instant::now() + TIMEOUT;
        let mut values = Vec::new();
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "timed out; values={values:#?}");
            let value = self.output.recv_timeout(remaining).unwrap_or_else(|error| {
                panic!("bridge output ended before expected value: {error}; values={values:#?}")
            });
            let done = predicate(&value);
            values.push(value);
            if done {
                return values;
            }
        }
    }

    fn initialize_and_start(&mut self) -> String {
        self.send(json!({"id": 1, "method": "initialize", "params": {}}));
        self.recv_until(|value| value.get("id") == Some(&json!(1)));
        self.send(json!({
            "id": 2,
            "method": "thread/start",
            "params": {
                "cwd": env!("CARGO_MANIFEST_DIR"),
                "model": "fake/fake-model",
                "approvalPolicy": "never",
                "sandbox": "danger-full-access"
            }
        }));
        let values = self.recv_until(|value| value.get("id") == Some(&json!(2)));
        values
            .last()
            .and_then(|value| value.pointer("/result/thread/id"))
            .and_then(Value::as_str)
            .unwrap()
            .to_string()
    }

    fn start_turn(&mut self, request_id: u64, thread_id: &str, text: &str) -> (String, Vec<Value>) {
        self.send(json!({
            "id": request_id,
            "method": "turn/start",
            "params": {
                "threadId": thread_id,
                "input": [{"type": "text", "text": text, "text_elements": []}]
            }
        }));
        let mut values = self.recv_until(|value| value.get("id") == Some(&json!(request_id)));
        let turn_id = values
            .last()
            .and_then(|value| value.pointer("/result/turn/id"))
            .and_then(Value::as_str)
            .unwrap()
            .to_string();
        values.extend(self.recv_until(|value| {
            value.get("method").and_then(Value::as_str) == Some("turn/completed")
        }));
        (turn_id, values)
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.session_root);
    }
}

fn temp_session_root() -> PathBuf {
    std::env::temp_dir().join(format!("centaur-omp-test-{}", Uuid::new_v4()))
}

fn completed(values: &[Value]) -> &Value {
    values
        .iter()
        .find(|value| value.get("method").and_then(Value::as_str) == Some("turn/completed"))
        .expect("turn/completed notification")
}

fn deltas(values: &[Value]) -> Vec<&str> {
    values
        .iter()
        .filter(|value| {
            value.get("method").and_then(Value::as_str) == Some("item/agentMessage/delta")
        })
        .filter_map(|value| value.pointer("/params/delta").and_then(Value::as_str))
        .collect()
}

#[test]
fn fake_omp_jsonrpc_errors_are_correlated_and_nonfatal() {
    let mut bridge = Bridge::spawn("jsonrpc", temp_session_root());
    bridge.send(json!({"id": 1, "method": "initialize", "params": {}}));
    bridge.recv_until(|value| value.get("id") == Some(&json!(1)));
    bridge.send(json!({
        "id": 2,
        "method": "thread/start",
        "params": {"cwd": 42}
    }));
    let failure = bridge.recv_until(|value| value.get("id") == Some(&json!(2)));
    assert_eq!(
        failure
            .last()
            .unwrap()
            .pointer("/error/code")
            .and_then(Value::as_i64),
        Some(-32000)
    );
    assert_eq!(
        failure
            .last()
            .unwrap()
            .pointer("/error/message")
            .and_then(Value::as_str),
        Some("OMP request failed")
    );

    bridge.send(json!({"id": 3, "method": "initialize", "params": {}}));
    let recovered = bridge.recv_until(|value| value.get("id") == Some(&json!(3)));
    assert_eq!(
        recovered
            .last()
            .unwrap()
            .pointer("/result/userAgent")
            .and_then(Value::as_str),
        Some("harness-server omp/17.1.3")
    );
}

#[test]
fn fake_omp_preserves_early_events_reuses_session_and_handles_local_and_chunks() {
    let mut bridge = Bridge::spawn("jsonrpc", temp_session_root());
    let thread_id = bridge.initialize_and_start();

    let (turn_id, early) = bridge.start_turn(3, &thread_id, "__early__");
    assert_eq!(
        completed(&early)
            .pointer("/params/turn/id")
            .and_then(Value::as_str),
        Some(turn_id.as_str())
    );
    assert_eq!(deltas(&early), vec!["early ", "event retained"]);

    let (_, first) = bridge.start_turn(4, &thread_id, "normal one");
    let first_pid = deltas(&first)
        .into_iter()
        .find(|delta| delta.starts_with("fake response pid="))
        .unwrap()
        .to_string();
    let (_, second) = bridge.start_turn(5, &thread_id, "normal two");
    assert!(deltas(&second).contains(&first_pid.as_str()));

    let (_, local) = bridge.start_turn(6, &thread_id, "__local__");
    assert!(deltas(&local).contains(&"local command completed"));
    assert_eq!(
        completed(&local)
            .pointer("/params/turn/status")
            .and_then(Value::as_str),
        Some("completed")
    );

    let (_, chunked) = bridge.start_turn(7, &thread_id, "__chunk__");
    assert!(deltas(&chunked).iter().any(|delta| delta.len() == 4096));
    let (_, reasoning) = bridge.start_turn(8, &thread_id, "__reasoning__");
    assert!(reasoning.iter().any(|value| {
        value
            .get("method")
            .and_then(Value::as_str)
            .is_some_and(|method| method.starts_with("item/reasoning/"))
    }));

    let (_, tool) = bridge.start_turn(9, &thread_id, "__tool__");
    assert!(tool.iter().any(|value| {
        value.get("method").and_then(Value::as_str) == Some("item/started")
            && value.pointer("/params/item/tool").and_then(Value::as_str) == Some("read")
    }));
    assert!(tool.iter().any(|value| {
        value.get("method").and_then(Value::as_str) == Some("item/completed")
            && value.pointer("/params/item/tool").and_then(Value::as_str) == Some("read")
    }));
    bridge.send(json!({
        "id": 10,
        "method": "turn/start",
        "params": {
            "threadId": thread_id,
            "input": [
                {"type": "text", "text": "__image__", "text_elements": []},
                {"type": "image", "url": "data:image/png;base64,iVBORw0KGgo=", "detail": null}
            ]
        }
    }));
    bridge.recv_until(|value| value.get("id") == Some(&json!(10)));
    let image = bridge
        .recv_until(|value| value.get("method").and_then(Value::as_str) == Some("turn/completed"));
    assert!(deltas(&image).contains(&"image count=1 mime=image/png"));
}

#[test]
fn fake_omp_enforces_terminal_error_callback_and_abort_semantics() {
    let mut bridge = Bridge::spawn("jsonrpc", temp_session_root());
    let thread_id = bridge.initialize_and_start();

    let (_, nonterminal) = bridge.start_turn(3, &thread_id, "__nonterminal__");
    assert!(deltas(&nonterminal).contains(&" continued"));
    bridge.send(json!({
        "id": 20,
        "method": "turn/start",
        "params": {
            "threadId": thread_id,
            "input": [{"type": "text", "text": "__out_of_order__", "text_elements": []}]
        }
    }));
    let turn_started = bridge.recv_until(|value| value.get("id") == Some(&json!(20)));
    let active_turn_id = turn_started
        .last()
        .and_then(|value| value.pointer("/result/turn/id"))
        .and_then(Value::as_str)
        .unwrap()
        .to_owned();
    bridge.send(json!({
        "id": 21,
        "method": "turn/steer",
        "params": {
            "threadId": thread_id,
            "expectedTurnId": active_turn_id,
            "input": [{"type": "text", "text": "steer now", "text_elements": []}]
        }
    }));
    let out_of_order = bridge
        .recv_until(|value| value.get("method").and_then(Value::as_str) == Some("turn/completed"));
    assert!(
        out_of_order
            .iter()
            .any(|value| value.get("id") == Some(&json!(21)))
    );
    assert!(deltas(&out_of_order).contains(&"steered out of order"));

    let (_, late_error) = bridge.start_turn(4, &thread_id, "__late_error__");
    assert_eq!(
        completed(&late_error)
            .pointer("/params/turn/status")
            .and_then(Value::as_str),
        Some("failed")
    );
    assert!(
        completed(&late_error)
            .pointer("/params/turn/error/message")
            .and_then(Value::as_str)
            .unwrap()
            .contains("scheduling failure")
    );

    let (_, ui) = bridge.start_turn(5, &thread_id, "__ui__");
    assert_eq!(
        completed(&ui)
            .pointer("/params/turn/status")
            .and_then(Value::as_str),
        Some("completed")
    );

    let (_, host_tool) = bridge.start_turn(6, &thread_id, "__host_tool__");
    assert_eq!(
        completed(&host_tool)
            .pointer("/params/turn/status")
            .and_then(Value::as_str),
        Some("completed")
    );

    bridge.send(json!({
        "id": 7,
        "method": "turn/start",
        "params": {
            "threadId": thread_id,
            "input": [{"type": "text", "text": "__hang__", "text_elements": []}]
        }
    }));
    let response = bridge.recv_until(|value| value.get("id") == Some(&json!(7)));
    let turn_id = response
        .last()
        .unwrap()
        .pointer("/result/turn/id")
        .and_then(Value::as_str)
        .unwrap()
        .to_string();
    bridge.send(json!({
        "id": 8,
        "method": "turn/interrupt",
        "params": {"threadId": thread_id, "turnId": turn_id}
    }));
    let interrupted = bridge
        .recv_until(|value| value.get("method").and_then(Value::as_str) == Some("turn/completed"));
    assert!(
        interrupted
            .iter()
            .any(|value| value.get("id") == Some(&json!(8)))
    );
    assert_eq!(
        completed(&interrupted)
            .pointer("/params/turn/status")
            .and_then(Value::as_str),
        Some("interrupted")
    );

    let (_, after_abort) = bridge.start_turn(9, &thread_id, "after abort");
    assert_eq!(
        completed(&after_abort)
            .pointer("/params/turn/status")
            .and_then(Value::as_str),
        Some("completed")
    );
}

#[test]
fn fake_omp_fails_closed_on_malformed_stdout_then_lazily_restarts_and_resumes() {
    let root = temp_session_root();
    let mut bridge = Bridge::spawn("jsonrpc", root.clone());
    let thread_id = bridge.initialize_and_start();

    let (_, malformed) = bridge.start_turn(3, &thread_id, "__malformed__");
    assert_eq!(
        completed(&malformed)
            .pointer("/params/turn/status")
            .and_then(Value::as_str),
        Some("failed")
    );
    let (_, restarted) = bridge.start_turn(4, &thread_id, "after malformed");
    assert_eq!(
        completed(&restarted)
            .pointer("/params/turn/status")
            .and_then(Value::as_str),
        Some("completed")
    );

    let mut child = Bridge::spawn("jsonrpc", root.clone());
    child.send(json!({"id": 1, "method": "initialize", "params": {}}));
    child.recv_until(|value| value.get("id") == Some(&json!(1)));
    child.send(json!({
        "id": 2,
        "method": "thread/resume",
        "params": {
            "threadId": thread_id,
            "cwd": env!("CARGO_MANIFEST_DIR"),
            "model": "fake-model",
            "modelProvider": "fake"
        }
    }));
    let resumed = child.recv_until(|value| value.get("id") == Some(&json!(2)));
    assert_eq!(
        resumed
            .last()
            .unwrap()
            .pointer("/result/thread/id")
            .and_then(Value::as_str),
        Some(thread_id.as_str())
    );
    let (_, turn) = child.start_turn(3, &thread_id, "resumed");
    assert_eq!(
        completed(&turn)
            .pointer("/params/turn/status")
            .and_then(Value::as_str),
        Some("completed")
    );
}

#[test]
fn fake_omp_blocks_mode_uses_same_stateful_runtime() {
    let mut bridge = Bridge::spawn("blocks", temp_session_root());
    bridge.send(json!({
        "type": "user",
        "text": "blocks turn",
        "model": "fake-model",
        "provider": "fake"
    }));
    let values = bridge
        .recv_until(|value| value.get("method").and_then(Value::as_str) == Some("turn/completed"));
    assert_eq!(
        completed(&values)
            .pointer("/params/turn/status")
            .and_then(Value::as_str),
        Some("completed")
    );
    bridge.send(json!({
        "type": "user",
        "content": [{
            "type": "attachment",
            "name": "__document__.txt",
            "mimeType": "text/plain",
            "attachment_type": "document",
            "dataBase64": "aGVsbG8="
        }],
        "model": "fake-model",
        "provider": "fake"
    }));
    let document = bridge
        .recv_until(|value| value.get("method").and_then(Value::as_str) == Some("turn/completed"));
    assert!(deltas(&document).contains(&"document path accepted"));
}
