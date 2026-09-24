use std::ffi::OsStr;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

const TIMEOUT: Duration = Duration::from_secs(10);
const KEY: &str = "test-thread";

struct TestRoot(PathBuf);

impl TestRoot {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("centaur-omp-{}", Uuid::new_v4()));
        std::fs::create_dir_all(root.join("home")).unwrap();
        Self(root)
    }

    fn session_dir(&self, key: &str) -> PathBuf {
        self.0.join(format!("{:x}", Sha256::digest(key.as_bytes())))
    }
}

impl Drop for TestRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Bridge {
    child: Child,
    stdin: Option<ChildStdin>,
    output: Receiver<Value>,
    timeout: Duration,
}

impl Bridge {
    fn spawn(root: &TestRoot, key: Option<&str>, env: &[(&str, &str)]) -> Self {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_omp_rpc.py");
        Self::with_binary(root, key, fixture.as_os_str(), env, TIMEOUT)
    }

    fn with_binary(
        root: &TestRoot,
        key: Option<&str>,
        binary: &OsStr,
        env: &[(&str, &str)],
        timeout: Duration,
    ) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_harness-server"));
        command
            .arg("omp")
            .current_dir(&root.0)
            .env("CENTAUR_OMP_BIN", binary)
            .env("CENTAUR_OMP_SESSION_ROOT", &root.0)
            .env("HOME", root.0.join("home"))
            .env_remove("CENTAUR_OMP_ENABLED")
            .env_remove("CENTAUR_OMP_ALLOWED_MODELS")
            .env_remove("CENTAUR_THREAD_KEY")
            .envs(env.iter().copied())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(key) = key {
            command.env("CENTAUR_THREAD_KEY", key);
        }
        let mut child = command.spawn().unwrap();
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().unwrap();
        let (tx, output) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let value = serde_json::from_str(&line.unwrap()).expect("JSON-only harness stdout");
                if tx.send(value).is_err() {
                    break;
                }
            }
        });
        Self {
            child,
            stdin,
            output,
            timeout,
        }
    }

    fn send(&mut self, value: Value) {
        let stdin = self.stdin.as_mut().unwrap();
        serde_json::to_writer(&mut *stdin, &value).unwrap();
        stdin.write_all(b"\n").unwrap();
        stdin.flush().unwrap();
    }

    fn until(&self, mut predicate: impl FnMut(&Value) -> bool) -> Vec<Value> {
        let deadline = Instant::now() + self.timeout;
        let mut values = Vec::new();
        loop {
            let value = self
                .output
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .unwrap_or_else(|error| panic!("bridge ended: {error}; values={values:#?}"));
            let done = predicate(&value);
            values.push(value);
            if done {
                return values;
            }
        }
    }

    fn begin(&mut self, text: &str) -> Vec<Value> {
        self.send(json!({"type": "user", "text": text}));
        self.until(|value| method(value) == "turn/started")
    }

    fn turn(&mut self, text: &str) -> Vec<Value> {
        self.send(json!({"type": "user", "text": text}));
        self.finish()
    }

    fn finish(&self) -> Vec<Value> {
        self.until(|value| method(value) == "turn/completed")
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(2);
        while matches!(self.child.try_wait(), Ok(None)) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn method(value: &Value) -> &str {
    value
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default()
}

fn completed(values: &[Value]) -> &Value {
    values
        .iter()
        .find(|value| method(value) == "turn/completed")
        .expect("turn/completed")
}

fn status(values: &[Value]) -> &str {
    completed(values)
        .pointer("/params/turn/status")
        .and_then(Value::as_str)
        .unwrap()
}

fn deltas(values: &[Value]) -> String {
    values
        .iter()
        .filter(|value| method(value) == "item/agentMessage/delta")
        .filter_map(|value| value.pointer("/params/delta").and_then(Value::as_str))
        .collect()
}

fn identity(values: &[Value]) -> (String, String) {
    let text = deltas(values);
    let (pid, session) = text
        .strip_prefix("fake response pid=")
        .unwrap()
        .split_once(" session=")
        .unwrap();
    (pid.to_string(), session.to_string())
}

#[test]
fn stock_omp_uses_native_per_thread_session_directory() {
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(&root, Some(KEY), &[]);
    let values = bridge.turn("native directory");
    assert_eq!(status(&values), "completed");
    assert!(root.session_dir(KEY).is_dir());
    let initial = identity(&values);
    drop(bridge);
    let mut resumed = Bridge::spawn(&root, Some(KEY), &[]);
    let next = identity(&resumed.turn("after restart"));
    assert_ne!(initial.0, next.0);
    assert_eq!(initial.1, next.1);
}

#[test]
fn stock_omp_needs_no_enablement_or_model_allowlist() {
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(
        &root,
        Some(KEY),
        &[
            ("CENTAUR_OMP_ENABLED", "0"),
            ("CENTAUR_OMP_ALLOWED_MODELS", ""),
        ],
    );
    assert_eq!(status(&bridge.turn("stock configuration")), "completed");
}

#[test]
fn stock_omp_accepts_non_anthropic_models() {
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(&root, Some(KEY), &[]);
    bridge.send(json!({"type": "user", "text": "__model__", "model": "openai/fake-model"}));
    assert_eq!(deltas(&bridge.finish()), "openai/fake-model");
    bridge.send(
        json!({"type": "user", "text": "__model__", "model": "second-model", "provider": "google"}),
    );
    assert_eq!(deltas(&bridge.finish()), "google/second-model");
    assert_eq!(deltas(&bridge.turn("__model__")), "google/second-model");
}

#[test]
fn stock_omp_cli_has_no_mode_selector() {
    let output = Command::new(env!("CARGO_BIN_EXE_harness-server"))
        .args(["omp", "--mode", "jsonrpc", "--help"])
        .output()
        .unwrap();
    assert!(!output.status.success());
}

#[test]
fn stock_omp_runs_with_protocol_v1_peer() {
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(&root, Some(KEY), &[]);
    assert_eq!(status(&bridge.turn("protocol v1")), "completed");
}

#[test]
fn stock_omp_inherits_provider_environment() {
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(&root, Some(KEY), &[("OPENAI_API_KEY", "dummy")]);
    assert_eq!(deltas(&bridge.turn("__provider_env__")), "inherited");
}

#[test]
fn native_resume_separates_long_keys_and_preserves_context() {
    let root = TestRoot::new();
    let a = format!("{}{}", "x".repeat(256), "a".repeat(256));
    let b = format!("{}{}", "x".repeat(256), "b".repeat(256));
    for (key, word) in [(&a, "ALPHA"), (&b, "BETA")] {
        let mut bridge = Bridge::spawn(&root, Some(key), &[]);
        assert_eq!(
            deltas(&bridge.turn(&format!("__remember:{word}"))),
            "remembered"
        );
    }
    for (key, word) in [(&a, "ALPHA"), (&b, "BETA")] {
        let mut bridge = Bridge::spawn(&root, Some(key), &[]);
        assert_eq!(deltas(&bridge.turn("__recall__")), word);
    }
    let mut first = Bridge::spawn(&root, None, &[]);
    first.turn("__remember:UNKEYED");
    drop(first);
    let mut second = Bridge::spawn(&root, None, &[]);
    assert_eq!(deltas(&second.turn("__recall__")), "forgotten");
}

#[test]
fn fake_omp_rejects_malformed_frames_then_resumes() {
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(&root, Some(KEY), &[]);
    bridge.turn("__remember:CONTEXT");
    for prompt in [
        "__unknown_event__",
        "__unknown_update__",
        "__missing_terminal__",
        "__nonboolean_terminal__",
        "__unknown_control__",
        "__late_error__",
        "__malformed__",
        "__oversize__",
    ] {
        assert_eq!(status(&bridge.turn(prompt)), "failed", "{prompt}");
        assert_eq!(deltas(&bridge.turn("__recall__")), "CONTEXT", "{prompt}");
    }
}

#[test]
fn fake_omp_ignores_stale_terminals_and_accepts_known_presentations() {
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(&root, Some(KEY), &[]);
    bridge.turn("__duplicate_terminal__");
    assert_eq!(status(&bridge.turn("after duplicate")), "completed");
    assert_eq!(
        deltas(&bridge.turn("__stale_terminal_during_turn__")),
        "current turn after stale terminal"
    );
    assert_eq!(
        deltas(&bridge.turn("__presentation__")),
        "known presentation accepted"
    );
    assert_eq!(
        deltas(&bridge.turn("__nonterminal__")),
        "nonterminal continued"
    );
    assert_eq!(deltas(&bridge.turn("__early__")), "early event retained");
}

#[test]
fn fake_omp_local_results_do_not_poison_the_next_turn() {
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(&root, Some(KEY), &[]);
    for prompt in ["__local__", "__local_result__"] {
        assert_eq!(deltas(&bridge.turn(prompt)), "local command completed");
        let next = bridge.turn("after local result");
        assert_eq!(status(&next), "completed");
        identity(&next);
    }
}

#[test]
fn fake_omp_command_output_before_interrupt_is_not_a_final_answer() {
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(&root, Some(KEY), &[]);
    let mut values = bridge.begin("__command_then_abort__");
    values.extend(bridge.until(|value| {
        method(value) == "item/completed"
            && value.pointer("/params/item/type") == Some(&json!("dynamicToolCall"))
    }));
    bridge.send(json!({"type": "interrupt"}));
    values.extend(bridge.finish());
    assert_eq!(status(&values), "interrupted");
    assert_eq!(deltas(&values), "");
    assert!(
        values
            .iter()
            .any(|value| value.pointer("/params/item/contentItems/0/text")
                == Some(&json!("command output before abort")))
    );
}

#[test]
fn fake_omp_aborts_before_failure_and_denies_host_callbacks() {
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(&root, Some(KEY), &[]);
    bridge.begin("__error_keeps_running__");
    bridge.until(|value| method(value) == "error");
    assert!(root.session_dir(KEY).join("error-turn-aborted").exists());
    assert_eq!(status(&bridge.finish()), "failed");
    for prompt in [
        "__notice_error__",
        "__retry_error__",
        "__extension_error__",
        "__persistence_error__",
    ] {
        let values = bridge.turn(prompt);
        assert_eq!(status(&values), "failed");
        assert!(
            !completed(&values)
                .to_string()
                .contains("private-store-path")
        );
    }
    for prompt in ["__ui__", "__ui_select__", "__host_tool__", "__host_uri__"] {
        assert_eq!(status(&bridge.turn(prompt)), "completed", "{prompt}");
    }
}

#[test]
fn fake_omp_steers_and_consumes_late_control_responses() {
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(&root, Some(KEY), &[]);
    let initial = identity(&bridge.turn("before controls"));
    for prompt in ["__out_of_order__", "__late_steer__", "__late_abort__"] {
        let mut values = bridge.begin(prompt);
        if prompt == "__late_abort__" {
            bridge.send(json!({"type": "interrupt"}));
        } else {
            bridge.send(json!({"type": "user", "text": "steering update", "client_user_message_id": "steer-item"}));
        }
        values.extend(bridge.finish());
        if prompt == "__late_abort__" {
            assert_eq!(status(&values), "interrupted");
        } else {
            assert_eq!(status(&values), "completed");
            assert!(deltas(&values).starts_with("steered"));
            assert!(
                completed(&values)
                    .pointer("/params/turn/items")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|item| item.get("clientId") == Some(&json!("steer-item")))
            );
        }
        bridge.send(json!({"type": "user", "text": "after control", "provider": "anthropic", "model": "fake-model-2"}));
        assert_eq!(identity(&bridge.finish()), initial);
    }
}

#[test]
fn fake_omp_forced_abort_and_child_exit_resume_native_context() {
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(&root, Some(KEY), &[]);
    bridge.turn("__remember:SURVIVES");
    let initial = identity(&bridge.turn("before abort"));
    bridge.begin("__ignore_abort__");
    bridge.send(json!({"type": "interrupt"}));
    assert_eq!(status(&bridge.finish()), "interrupted");
    assert_eq!(deltas(&bridge.turn("__recall__")), "SURVIVES");
    let after = identity(&bridge.turn("after abort"));
    assert_ne!(initial.0, after.0);
    assert_eq!(initial.1, after.1);
    let status = Command::new("kill")
        .args(["-KILL", &after.0])
        .status()
        .unwrap();
    assert!(status.success());
    let deadline = Instant::now() + TIMEOUT;
    loop {
        let output = Command::new("ps")
            .args(["-o", "stat=", "-p", &after.0])
            .output()
            .unwrap();
        let state = String::from_utf8(output.stdout).unwrap();
        if state.trim().is_empty() || state.trim().starts_with('Z') {
            break;
        }
        assert!(Instant::now() < deadline, "fake child did not exit");
        thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(deltas(&bridge.turn("__recall__")), "SURVIVES");
}

#[test]
fn fake_omp_bounds_unacknowledged_controls() {
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(&root, Some(KEY), &[]);
    bridge.begin("__silent_controls__");
    for _ in 0..1025 {
        bridge.send(json!({"type": "user", "text": "steer"}));
    }
    let failed = bridge.finish();
    assert_eq!(status(&failed), "failed");
    assert!(
        completed(&failed)
            .to_string()
            .contains("pending control response limit")
    );
}

#[test]
fn fake_omp_replays_captured_omp_18_3_turns() {
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(&root, Some(KEY), &[]);
    let text = bridge.turn("__replay:text");
    assert_eq!(status(&text), "completed");
    assert_eq!(deltas(&text), "PONG-1");
    let tool = bridge.turn("__replay:tool");
    assert_eq!(status(&tool), "completed");
    for event in ["item/started", "item/completed"] {
        assert!(tool.iter().any(|value| method(value) == event
            && value.pointer("/params/item/tool") == Some(&json!("bash"))));
    }
    assert_eq!(deltas(&tool), "PONG-2");
    for (name, error) in [
        ("provider_error", "Tracer non-retryable error"),
        ("truncated_stream", "stream ended before message_stop"),
    ] {
        let values = bridge.turn(&format!("__replay:{name}"));
        assert_eq!(status(&values), "failed");
        let message = completed(&values)
            .pointer("/params/turn/error/message")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(message.contains(error), "{message}");
        assert!(!message.contains("raw-http-request"), "{message}");
    }
    assert_eq!(deltas(&bridge.turn("__replay:text")), "PONG-1");
    let retry = bridge.turn("__retried_provider_error__");
    assert_eq!(status(&retry), "completed");
    assert_eq!(deltas(&retry), "recovered after retry");
}

#[test]
fn fake_omp_accepts_blocks_attachments() {
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(&root, Some(KEY), &[]);
    bridge.send(json!({"type": "user", "content": [{"type": "attachment", "name": "note.txt", "mimeType": "text/plain", "attachment_type": "document", "dataBase64": "aGVsbG8="}]}));
    assert_eq!(deltas(&bridge.finish()), "document path accepted");
}

#[test]
fn stock_omp_image_count_uses_native_provider_limits() {
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(&root, Some(KEY), &[]);
    let images: Vec<_> = (0..9)
        .map(|_| json!({"type": "image", "url": "data:image/png;base64,aA=="}))
        .collect();
    let mut content = vec![json!({"type": "text", "text": "__image__", "text_elements": []})];
    content.extend(images);
    bridge.send(json!({"type": "user", "content": content}));
    assert_eq!(deltas(&bridge.finish()), "image count=9");
}

#[test]
fn stock_omp_image_bytes_use_native_provider_limits() {
    use base64::Engine;
    let root = TestRoot::new();
    let mut bridge = Bridge::spawn(&root, Some(KEY), &[]);
    let data = base64::engine::general_purpose::STANDARD.encode(vec![0; 4 * 1024 * 1024 + 1]);
    bridge.send(json!({"type": "user", "content": [
        {"type": "text", "text": "__image__", "text_elements": []},
        {"type": "image", "url": format!("data:image/png;base64,{data}")}
    ]}));
    assert_eq!(deltas(&bridge.finish()), "image count=1");
}

#[test]
#[ignore = "requires ANTHROPIC_API_KEY and real OMP; makes provider calls"]
fn real_omp_streaming_steer_and_resume() {
    assert!(
        std::env::var("ANTHROPIC_API_KEY").is_ok_and(|key| !key.is_empty()),
        "set ANTHROPIC_API_KEY before running real OMP tests"
    );
    let binary = std::env::var_os("CENTAUR_OMP_BIN").unwrap_or_else(|| "omp".into());
    let version = Command::new(&binary)
        .arg("--version")
        .output()
        .expect("real omp on PATH or CENTAUR_OMP_BIN");
    assert!(version.status.success(), "OMP --version failed");
    let model = std::env::var("CENTAUR_REAL_OMP_MODEL")
        .unwrap_or_else(|_| "anthropic/claude-sonnet-4-5".to_string());
    let root = TestRoot::new();
    let codeword = format!("OMP_MEMORY_{}", Uuid::new_v4().simple());
    let acknowledgement = format!("STEER_{}", Uuid::new_v4().simple());
    let timeout = Duration::from_secs(300);
    let mut bridge = Bridge::with_binary(&root, Some(KEY), &binary, &[], timeout);
    bridge.send(json!({
        "type": "user", "model": model,
        "text": format!("Remember the codeword {codeword} for later. Do not use tools. Produce 300 lines numbered 001 through 300, each followed by the words: streaming output remains visible while a steering update arrives. If an update arrives, follow it.")
    }));
    let mut values = bridge.until(|value| method(value) == "item/agentMessage/delta");
    bridge.send(json!({
        "type": "user", "client_user_message_id": "real-omp-steer",
        "text": format!("Stop the numbered listing. Remember the codeword {codeword}. Reply exactly {acknowledgement} and nothing else.")
    }));
    values.extend(bridge.finish());
    assert_eq!(status(&values), "completed");
    let delta_count = values
        .iter()
        .filter(|value| method(value) == "item/agentMessage/delta")
        .count();
    assert!(delta_count > 1, "OMP must preserve streaming deltas");
    assert!(
        deltas(&values).contains(&acknowledgement),
        "OMP did not apply the steering update"
    );
    assert!(
        completed(&values)
            .pointer("/params/turn/items")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item.get("clientId") == Some(&json!("real-omp-steer")))
    );
    eprintln!(
        "real OMP {}: streaming_deltas={delta_count}, steer=completed",
        String::from_utf8_lossy(&version.stdout).trim()
    );
    drop(bridge);

    let mut resumed = Bridge::with_binary(&root, Some(KEY), &binary, &[], timeout);
    resumed.send(json!({
        "type": "user", "model": model,
        "text": "Without using tools, what exact codeword did I ask you to remember earlier? Reply with only that codeword."
    }));
    let recalled = resumed.finish();
    assert_eq!(status(&recalled), "completed");
    assert_eq!(deltas(&recalled).trim(), codeword);
    eprintln!("real OMP: native resume recalled the codeword after child restart");
}
