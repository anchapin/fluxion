//! Workspace E2E integration tests for `fluxion-mcp`.
//!
//! Issue #2913 — `fluxion-mcp/` is a separate package (path = "..",
//! `default-features = false`, `multi-zone` gated by this crate's own
//! feature) and currently has no JSON-RPC E2E coverage against the
//! compiled binary. This file exercises the wire protocol the MCP
//! transports rely on:
//!
//! 1. **JSON-RPC round-trip** — spawn the binary, drive the protocol
//!    over newline-delimited JSON on stdin/stdout, parse the response.
//! 2. **Schema validation** — assert `initialize` and `tools/list`
//!    responses match the JSON-RPC 2.0 + MCP tool schema shape.
//! 3. **Error propagation** — verify JSON-RPC error envelope (code +
//!    message) plus the `request_id` correlation token survives on
//!    error paths.
//! 4. **Performance budget** — measure p50/p95/p99 per-call latency for
//!    a trivial request and assert p50 < 10 ms (the criterion in the
//!    issue) on a long-running server. Each test spawns its own
//!    process so the perf number reflects cold + warm behaviour
//!    rather than suite-wide state.
//!
//! Implementation notes
//! --------------------
//! * We use `env!("CARGO_BIN_EXE_fluxion-mcp")` (set by Cargo for
//!   integration tests in the same package as a binary target) instead
//!   of `cargo run -p fluxion-mcp --bin fluxion-mcp` — the latter
//!   would pay a ~5–15 s cargo-launch tax on every perf measurement
//!   and make the 10 ms budget unverifiable.
//! * Stdout is read on a dedicated thread because the main test thread
//!   owns the stdin pipe; without concurrent draining the server can
//!   block once its stdout pipe buffer fills. Stderr is drained on a
//!   second thread for the same reason (`tracing_subscriber::fmt()`
//!   writes to stderr by default).
//! * Each test's `McpProcess` instance is `Drop`-safe: when a test
//!   panics in the middle of an exchange, the child is killed and
//!   reaped so we don't leak processes across `cargo test` runs.

#![allow(clippy::expect_used)] // Tests assert hard invariants; panics on miss.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

/// End-to-end driver for one `fluxion-mcp` child process. Owns the
/// stdin pipe (write side) and a background reader thread that
/// decodes one JSON line per response into the channel.
struct McpProcess {
    child: Child,
    stdin: Option<std::process::ChildStdin>,
    responses_rx: std::sync::mpsc::Receiver<Value>,
    /// Held to keep the join handles alive; lets the reader thread run
    /// for the lifetime of the connection.
    _stdout_thread: thread::JoinHandle<()>,
    _stderr_thread: thread::JoinHandle<()>,
}

impl McpProcess {
    fn spawn() -> Self {
        // Silence tracing logs on the child so they don't bleed into
        // stderr in CI output. Tracing goes to stderr, never stdout —
        // the wire format is unaffected either way.
        let bin = env!("CARGO_BIN_EXE_fluxion-mcp");
        let mut child = Command::new(bin)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("RUST_LOG", "error")
            .spawn()
            .unwrap_or_else(|e| panic!("failed to spawn {bin}: {e}"));

        let stdin = child
            .stdin
            .take()
            .expect("child stdin was configured as piped");
        let stdout = child
            .stdout
            .take()
            .expect("child stdout was configured as piped");
        let stderr = child
            .stderr
            .take()
            .expect("child stderr was configured as piped");

        let (tx, rx) = channel::<Value>();

        let stdout_thread = thread::Builder::new()
            .name("mcp-stdout-reader".into())
            .spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines() {
                    let line = match line {
                        Ok(l) => l,
                        Err(_) => break, // Broken pipe: child closed stdout.
                    };
                    if line.trim().is_empty() {
                        continue;
                    }
                    // Anything that isn't a JSON object on stdout is a
                    // protocol violation — stop draining so the test
                    // can fail loudly with the timeout path.
                    match serde_json::from_str::<Value>(&line) {
                        Ok(v) => {
                            if tx.send(v).is_err() {
                                break; // Receiver was dropped (test ended).
                            }
                        }
                        Err(_) => break,
                    }
                }
            })
            .expect("spawn stdout reader thread");

        let stderr_thread = thread::Builder::new()
            .name("mcp-stderr-drain".into())
            .spawn(move || {
                // Drain stderr so the child never blocks on a full pipe.
                let reader = BufReader::new(stderr);
                for _line in reader.lines() {
                    // Discard.
                }
            })
            .expect("spawn stderr drain thread");

        Self {
            child,
            stdin: Some(stdin),
            responses_rx: rx,
            _stdout_thread: stdout_thread,
            _stderr_thread: stderr_thread,
        }
    }

    /// Send one JSON-RPC request and block until the matching
    /// response arrives or the 5 s budget expires.
    fn request(&mut self, request: Value) -> Value {
        let line = serde_json::to_string(&request).expect("serialize JSON-RPC request");
        {
            let stdin = self.stdin.as_mut().expect("stdin still open");
            writeln!(stdin, "{line}").expect("write JSON-RPC request to stdin");
            stdin.flush().expect("flush stdin");
        }

        match self.responses_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(v) => v,
            Err(RecvTimeoutError::Timeout) => {
                panic!(
                    "timed out after 5s waiting for response to method {:?}",
                    request.get("method")
                )
            }
            Err(RecvTimeoutError::Disconnected) => {
                panic!(
                    "server closed stdout before responding to method {:?}",
                    request.get("method")
                )
            }
        }
    }

    /// Best-effort graceful shutdown: close stdin (signals EOF to the
    /// server's read loop), wait briefly for the process to exit on
    /// its own, then kill it if it hasn't.
    fn shutdown(&mut self) {
        // Drop stdin to send EOF on the pipe.
        self.stdin.take();
        let _ = self.child.try_wait(); // Cheap non-blocking poll.

        // If still alive, send a polite SIGKILL after a short grace.
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(500) {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            thread::sleep(Duration::from_millis(20));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for McpProcess {
    fn drop(&mut self) {
        // Safety net: even if a test panics before calling
        // `shutdown()`, kill the child so we don't leak it.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Asserts a string looks like a UUIDv4 (36 chars, 4 hyphens, version
/// digit `4` at position 14). MCP responses always carry a fresh
/// UUIDv4 per request — see Issue #2515.
fn assert_uuid_v4(s: &str, context: &str) {
    assert!(s.len() == 36, "{context}: expected 36-char UUID, got '{s}'");
    let hyphens = s.chars().filter(|c| *c == '-').count();
    assert_eq!(
        hyphens, 4,
        "{context}: expected 4 hyphens in UUID, got '{s}'"
    );
    let chars: Vec<char> = s.chars().collect();
    assert_eq!(
        chars[14], '4',
        "{context}: expected UUIDv4 version digit '4' at index 14, got '{s}'"
    );
}

/// AC #1: JSON-RPC round-trip — `initialize` flows through the
/// newline-delimited wire format on stdin/stdout and produces a
/// JSON-RPC 2.0 response envelope with a `result` payload.
#[test]
fn jsonrpc_round_trip_initialize() {
    let mut mcp = McpProcess::spawn();
    let resp = mcp.request(json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": Value::Null,
    }));

    assert_eq!(resp["jsonrpc"], "2.0", "envelope missing jsonrpc field");
    assert_eq!(resp["id"], 1, "response id must echo request id");
    assert!(
        resp.get("error").is_none(),
        "initialize returned an error: {resp}"
    );
    assert!(
        resp["result"].is_object(),
        "initialize missing result object: {resp}"
    );
    let result = &resp["result"];
    assert_eq!(
        result["protocolVersion"], "2024-11-05",
        "unexpected protocolVersion"
    );
    assert_eq!(
        result["serverInfo"]["name"], "fluxion-mcp",
        "unexpected serverInfo.name"
    );
    assert_eq!(
        result["capabilities"]["tools"],
        json!({}),
        "capabilities.tools must be an empty object"
    );

    // Issue #2515 — correlation token, even for non-tool responses.
    let rid = resp["request_id"]
        .as_str()
        .expect("initialize response missing request_id");
    assert_uuid_v4(rid, "initialize request_id");

    mcp.shutdown();
}

/// AC #2: schema validation — every entry in `tools/list` must match
/// the MCP tool schema (string `name`, string `description`, and an
/// `inputSchema` with `type: "object"`). Spot-check the canonical
/// tools exposed by `fluxion-mcp/src/tools.rs::list_tools`.
#[test]
fn schema_validation_tools_list() {
    let mut mcp = McpProcess::spawn();

    // Initialize first (not strictly required by the server, but a
    // realistic client sends it before `tools/list`).
    let init = mcp.request(json!({
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": Value::Null,
    }));
    assert_eq!(init["result"]["protocolVersion"], "2024-11-05");

    let resp = mcp.request(json!({
        "jsonrpc": "2.0",
        "id": "list-1",
        "method": "tools/list",
    }));

    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], "list-1");
    assert!(
        resp.get("error").is_none(),
        "tools/list returned an error: {resp}"
    );

    let tools = resp["result"]["tools"]
        .as_array()
        .expect("tools/list result.tools must be an array");
    assert!(
        !tools.is_empty(),
        "tools/list returned an empty array (expected at least one tool)"
    );

    let mut names: Vec<String> = Vec::with_capacity(tools.len());
    for (idx, tool) in tools.iter().enumerate() {
        let obj = tool
            .as_object()
            .unwrap_or_else(|| panic!("tools[{idx}] is not a JSON object"));
        let name = obj
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("tools[{idx}] missing string `name`"));
        assert!(
            obj.get("description").and_then(Value::as_str).is_some(),
            "tools[{idx}] (`{name}`) missing string `description`"
        );
        let schema = obj
            .get("inputSchema")
            .unwrap_or_else(|| panic!("tools[{idx}] (`{name}`) missing `inputSchema`"));
        assert_eq!(
            schema["type"], "object",
            "tools[{idx}] (`{name}`).inputSchema.type must be \"object\""
        );
        names.push(name.to_string());
    }

    for required in [
        "load_building_model",
        "run_simulation",
        "describe_model",
        "list_construction_assemblies",
        "inspect_fluid_loop",
    ] {
        assert!(
            names.iter().any(|n| n == required),
            "tools/list missing required tool `{required}` (got {names:?})"
        );
    }

    mcp.shutdown();
}

/// AC #3: error propagation — an unknown JSON-RPC method must produce
/// a proper `-32601` error envelope (`Method not found`) with no
/// `result` and a valid correlation token.
#[test]
fn error_propagation_unknown_method() {
    let mut mcp = McpProcess::spawn();
    let resp = mcp.request(json!({
        "jsonrpc": "2.0",
        "id": 99,
        "method": "totally/bogus/method",
        "params": Value::Null,
    }));

    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 99, "error response id must echo request id");

    // The server emits `result: None` (omitted via
    // `skip_serializing_if`) on error paths. Either `result` absent or
    // `result` JSON-null is acceptable per the JSON-RPC 2.0 spec.
    if let Some(r) = resp.get("result") {
        assert!(
            r.is_null(),
            "unknown method should not return a result payload, got {r}"
        );
    }

    let err = resp["error"]
        .as_object()
        .expect("unknown method must return a JSON-RPC error envelope");
    assert_eq!(
        err["code"].as_i64(),
        Some(-32601),
        "expected JSON-RPC -32601 (Method not found), got {:?}",
        err["code"]
    );
    let msg = err["message"]
        .as_str()
        .expect("error.message must be a string");
    assert!(
        msg.to_lowercase().contains("unknown") || msg.to_lowercase().contains("method"),
        "error.message should describe the unknown method, got '{msg}'"
    );

    // Correlation token must survive on error paths (Issue #2515).
    let rid = resp["request_id"]
        .as_str()
        .expect("error response missing request_id");
    assert_uuid_v4(rid, "unknown-method request_id");

    mcp.shutdown();
}

/// AC #3 (extended): tool-level error envelope — calling a tool
/// without first loading a model must return a JSON-shaped error in
/// the `result`, not a JSON-RPC error. This is the "schema-level"
/// error path the MCP layer never sees, since the protocol envelope
/// succeeds but the tool itself reports a domain error.
#[test]
fn error_propagation_tool_without_state() {
    let mut mcp = McpProcess::spawn();
    let resp = mcp.request(json!({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "describe_model",
            "arguments": {}
        }
    }));

    // No JSON-RPC error — the protocol envelope is intact.
    assert!(
        resp.get("error").is_none(),
        "tools/call must not return a JSON-RPC error envelope for tool-level errors, got {resp}"
    );

    // But the tool itself must report "no model loaded" inside the result.
    let result = resp["result"].as_object().expect(
        "tools/call result must be an object (the tool returned JSON re-parsed into a Value)",
    );
    let err_text = result
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        err_text.to_lowercase().contains("model"),
        "expected the tool to report a model-loading error, got {result:?}"
    );

    mcp.shutdown();
}

/// Bonus (not strictly in AC but completes the E2E story): a
/// multi-request sequence on a single connection — proves the server
/// keeps state across requests and that the wire framing is
/// re-entrant on long-lived connections.
#[test]
fn multi_request_stateful_session() {
    let mut mcp = McpProcess::spawn();

    // 1. initialize
    let init = mcp.request(json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": Value::Null,
    }));
    assert_eq!(init["result"]["protocolVersion"], "2024-11-05");

    // 2. load_building_model
    let load = mcp.request(json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "load_building_model",
            "arguments": {
                "num_zones": 3,
                "zone_area": 75.0,
                "window_u_value": 1.4,
                "heating_setpoint": 21.0,
                "cooling_setpoint": 25.0
            }
        }
    }));
    let load_result = load["result"]
        .as_object()
        .expect("load_building_model result must be an object");
    assert_eq!(
        load_result["success"],
        Value::Bool(true),
        "load_building_model did not succeed: {load_result:?}"
    );
    assert_eq!(load_result["model"]["num_zones"], 3);

    // 3. describe_model on the same connection — must observe the
    //    loaded model from step 2.
    let describe = mcp.request(json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "describe_model",
            "arguments": {}
        }
    }));
    let describe_result = describe["result"]
        .as_object()
        .expect("describe_model result must be an object");
    assert_eq!(
        describe_result["num_zones"], 3,
        "describe_model did not observe the model loaded in step 2: {describe_result:?}"
    );

    // All three responses must carry distinct request_ids.
    let mut ids = std::collections::HashSet::new();
    for resp in [&init, &load, &describe] {
        let rid = resp["request_id"]
            .as_str()
            .expect("every response carries request_id");
        assert!(
            ids.insert(rid),
            "duplicate request_id '{rid}' across responses on a single session"
        );
    }

    mcp.shutdown();
}

/// AC #4: performance budget — every JSON-RPC call must come back in
/// under 10 ms on the median. Issue acceptance criterion. We measure
/// `initialize` because it has zero physics work and so isolates the
/// pure wire-format overhead.
///
/// Methodology
/// -----------
/// * 5 unmeasured warmup exchanges to flush any first-call setup
///   (tracing-subscriber initialisation, OS scheduler ramp-up).
/// * N=200 measured exchanges per the long-tail sample size needed
///   for a stable p99 estimate.
/// * Reports p50 / p95 / p99 in test output (visible with
///   `cargo test -- --nocapture`) so a slow run is debuggable.
/// * Two assertions: p50 < 10 ms (the issue acceptance criterion);
///   and p99 < 50 ms (a generous soft ceiling for scheduling jitter
///   on contended CI runners — surfaces gross regressions without
///   flaking on incidental pauses).
#[test]
fn performance_budget_under_10ms_per_call() {
    let mut mcp = McpProcess::spawn();

    // Warmup — not measured.
    for i in 0..5 {
        let resp = mcp.request(json!({
            "jsonrpc": "2.0",
            "id": -1 - i,
            "method": "initialize",
            "params": Value::Null,
        }));
        assert_eq!(resp["result"]["protocolVersion"], "2024-11-05");
    }

    const N: usize = 200;
    let mut samples: Vec<Duration> = Vec::with_capacity(N);
    for i in 0..N {
        let t = Instant::now();
        let resp = mcp.request(json!({
            "jsonrpc": "2.0",
            "id": i,
            "method": "initialize",
            "params": Value::Null,
        }));
        let dt = t.elapsed();
        assert_eq!(resp["result"]["protocolVersion"], "2024-11-05");
        samples.push(dt);
    }

    samples.sort();
    let p50 = samples[N / 2];
    let p95 = samples[(N * 95) / 100];
    let p99 = samples[(N * 99) / 100];
    eprintln!(
        "mcp_e2e perf: N={N} p50={p50:?} p95={p95:?} p99={p99:?} max={:?}",
        samples.last()
    );

    // Hard acceptance criterion from the issue body.
    assert!(
        p50 < Duration::from_millis(10),
        "p50 {p50:?} exceeded the 10 ms/call acceptance criterion"
    );
    // Soft ceiling — must hold with reasonable CI jitter. Failure
    // here indicates a wire-format regression (e.g. extra serialization
    // round-trips, dropped async batching) rather than a single slow
    // scheduling slice.
    assert!(
        p99 < Duration::from_millis(50),
        "p99 {p99:?} exceeded the 50 ms/call soft ceiling (likely a wire regression)"
    );

    mcp.shutdown();
}
