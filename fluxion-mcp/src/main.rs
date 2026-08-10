//! fluxion-mcp — Model Context Protocol server for the Fluxion building energy model.
//!
//! Threading model (Issue #2562)
//! -----------------------------
//! The server is built on a Tokio multi-threaded runtime (`#[tokio::main]`).
//! Mutable session state is held behind [`Arc<tokio::sync::Mutex<McpState>>`],
//! which makes `McpState` usable from any task. Because `tokio::sync::Mutex`
//! is async-aware, a contended lock suspends the awaiting task instead of
//! blocking the runtime worker thread, which preserves the goal-5
//! production-artifact promise of being able to extend the server to
//! HTTP/WebSocket transports without re-architecting the state layer.
//!
//! Stdin/stdout use `tokio::io::stdin()` + `tokio::io::stdout()` with an
//! explicit per-line flush so the wire format stays byte-identical to the
//! pre-async implementation: one JSON-RPC request per line on stdin, one
//! JSON-RPC response per line on stdout.
//!
//! `McpState` itself is `Send` (every field is `Send`) and is held behind
//! a `tokio::sync::Mutex` so no additional `Sync` bound is required on
//! `McpState` itself. The `Arc` clone is what makes the state shareable
//! between the request loop and any future concurrent sub-tasks.

mod metrics;
mod state;
mod tools;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::state::McpState;

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Value,
    pub method: String,
    #[serde(rename = "params")]
    pub params: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
    /// Correlation token (UUIDv4) generated per request so AI agents calling
    /// `set_hvac_control_sequence` etc. get a traceable handle alongside the
    /// `"success": true` envelope (Issue #2515). Echoed from every response;
    /// omitted from the wire when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Register metric descriptions (no-op without a recorder installed).
    // Issue #2515 — per-tool latency / error counters.
    metrics::describe_metrics();

    tracing::info!("Starting fluxion-mcp server");

    let state = Arc::new(Mutex::new(McpState::default()));
    let stdin = BufReader::new(tokio::io::stdin());
    let result = run_server(stdin, tokio::io::stdout(), state).await;
    if let Err(e) = &result {
        tracing::error!("fluxion-mcp server exited with error: {}", e);
    }
    result
}

/// Run the JSON-RPC request loop on the supplied async reader/writer pair.
///
/// Splitting the loop from `main` keeps it testable: tests can drive the
/// server over an in-memory `tokio::io::duplex` pipe without spawning a child
/// process. Wire format is preserved (one JSON object per `\n`-terminated
/// line) so the production binary remains drop-in compatible with existing
/// MCP clients.
///
/// The loop terminates when the reader yields EOF (returns `Ok(None)` from
/// `next_line`). Any I/O error on the reader is propagated; write errors are
/// logged but do not terminate the loop, matching the pre-async behaviour
/// where `println!` failures were silently swallowed by the standard library.
pub async fn run_server<R, W>(
    reader: R,
    mut writer: W,
    state: Arc<Mutex<McpState>>,
) -> anyhow::Result<()>
where
    R: tokio::io::AsyncBufRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        if line.is_empty() {
            continue;
        }

        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(e) => {
                tracing::error!("Failed to parse JSON-RPC request: {}", e);
                continue;
            }
        };

        let response = process_request(request, &state).await;

        let response_json = serde_json::to_string(&response).unwrap_or_else(|e| {
            tracing::error!("Failed to serialize response: {}", e);
            r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32603,"message":"Internal error"}}"#
                .to_string()
        });

        if let Err(e) = async {
            writer.write_all(response_json.as_bytes()).await?;
            writer.write_all(b"\n").await?;
            writer.flush().await?;
            Ok::<_, std::io::Error>(())
        }
        .await
        {
            tracing::error!("Failed to write JSON-RPC response: {}", e);
        }
    }

    Ok(())
}

async fn process_request(request: JsonRpcRequest, state: &Arc<Mutex<McpState>>) -> JsonRpcResponse {
    let id = request.id.clone();
    // Per-request correlation token (Issue #2515). Generated unconditionally
    // so every response — not just tool calls — carries a traceable handle.
    let request_id = uuid::Uuid::new_v4();
    let request_id_str = request_id.to_string();

    match request.method.as_str() {
        "initialize" => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "fluxion-mcp",
                    "version": "0.1.0"
                }
            })),
            error: None,
            request_id: Some(request_id_str),
        },
        "tools/list" => {
            let tools = tools::list_tools();
            JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(serde_json::json!({ "tools": tools })),
                error: None,
                request_id: Some(request_id_str),
            }
        }
        "tools/call" => {
            let params = request.params.unwrap_or(serde_json::Value::Null);
            // Extract the tool name BEFORE dispatch so the span + metric labels
            // are populated even when the dispatch itself fails (e.g. unknown
            // tool name, missing `params.name`).
            let tool_name = params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            // Issue #2515 — wrap the dispatch in a tracing span carrying the
            // tool name and per-request id so distributed traces correlate
            // with the `request_id` echoed in the response envelope.
            let span = tracing::info_span!(
                "mcp_tool_call",
                tool = %tool_name,
                request_id = %request_id,
            );
            let start = std::time::Instant::now();

            let result_value: serde_json::Value = {
                let _enter = span.enter();
                // Acquire the async mutex; this yields `&mut McpState` so the
                // existing synchronous `handle_tool_call` signature is preserved
                // and the entire tool-call dispatch remains single-writer
                // (no aliasing of mutable state).
                let mut state_guard = state.lock().await;
                let result_str = tools::handle_tool_call(&mut state_guard, params);
                // The result string is already formatted (JSON or TOON)
                // Wrap it as a JSON Value (String for TOON, Object for JSON parsed back)
                if result_str.starts_with("toon:v1") {
                    serde_json::json!({ "_toon": result_str })
                } else {
                    // Parse JSON string back to Value for consistent structure
                    serde_json::from_str(&result_str)
                        .unwrap_or_else(|_| serde_json::json!({ "raw": result_str }))
                }
            };

            let elapsed = start.elapsed().as_secs_f64();
            // Record per-tool histogram + (on in-band error) error counter.
            // Kept synchronous + outside the span so it is testable under a
            // thread-local DebuggingRecorder without crossing an await point.
            metrics::record_tool_outcome(&tool_name, elapsed, &result_value);

            JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(result_value),
                error: None,
                request_id: Some(request_id_str),
            }
        }
        _ => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: format!("Unknown method: {}", request.method),
                data: None,
            }),
            request_id: Some(request_id_str),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{duplex, AsyncReadExt, AsyncWriteExt, BufReader};

    /// Helper that drives one full JSON-RPC exchange over a fresh duplex
    /// pipe. Returns the parsed response so individual test cases can
    /// assert on the shape without re-implementing the framing.
    async fn exchange(request: serde_json::Value) -> serde_json::Value {
        let (mut client_write, server_read) = duplex(4096);
        let (server_write, mut client_read) = duplex(4096);
        let state = Arc::new(Mutex::new(McpState::default()));
        let server = tokio::spawn(run_server(BufReader::new(server_read), server_write, state));

        let request_line = format!("{}\n", serde_json::to_string(&request).unwrap());
        client_write
            .write_all(request_line.as_bytes())
            .await
            .unwrap();
        client_write.shutdown().await.unwrap();

        let mut buf = Vec::new();
        client_read.read_to_end(&mut buf).await.unwrap();
        // Server side drops both halves when the read end returns EOF after
        // shutdown; the spawned task completes cleanly.
        server.await.unwrap().unwrap();

        serde_json::from_slice(&buf).unwrap()
    }

    #[tokio::test]
    async fn initialize_returns_protocol_version() {
        let resp = exchange(serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": null
        }))
        .await;

        assert_eq!(resp["jsonrpc"], "2.0");
        assert_eq!(resp["id"], 1);
        assert_eq!(resp["result"]["protocolVersion"], "2024-11-05");
        assert_eq!(resp["result"]["serverInfo"]["name"], "fluxion-mcp");
        assert!(resp.get("error").is_none());
    }

    #[tokio::test]
    async fn tools_list_returns_non_empty_tools() {
        let resp = exchange(serde_json::json!({
            "jsonrpc": "2.0",
            "id": "list",
            "method": "tools/list"
        }))
        .await;

        let tools = resp["result"]["tools"].as_array().expect("tools array");
        assert!(!tools.is_empty(), "expected at least one tool");
        assert!(
            tools.iter().any(|t| t["name"] == "load_building_model"),
            "expected load_building_model tool to be advertised"
        );
    }

    #[tokio::test]
    async fn tools_call_routes_through_shared_state() {
        // First request: load a model.
        let resp = exchange(serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "load_building_model",
                "arguments": {
                    "num_zones": 2,
                    "zone_area": 50.0,
                    "window_u_value": 1.5,
                    "heating_setpoint": 21.0,
                    "cooling_setpoint": 26.0
                }
            }
        }))
        .await;

        // handle_tool_call returns JSON which the server re-parses back into
        // a Value, so `result` is an object — not a raw string.
        let result = resp["result"].as_object().expect("result as object");
        assert_eq!(result["success"], serde_json::json!(true));
        assert_eq!(result["model"]["num_zones"], serde_json::json!(2));
    }

    #[tokio::test]
    async fn unknown_method_returns_jsonrpc_error() {
        let resp = exchange(serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "nonsense/method",
            "params": null
        }))
        .await;

        assert_eq!(resp["error"]["code"], -32601);
        assert!(resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Unknown"));
    }

    /// Issue #2515 — every response carries a non-empty `request_id`
    /// correlation token. Asserted here for `initialize` and a tool call so
    /// both the non-tool and tool code paths echo the field.
    #[tokio::test]
    async fn response_carries_request_id_correlation_token() {
        let resp = exchange(serde_json::json!({
            "jsonrpc": "2.0",
            "id": "rid-1",
            "method": "initialize",
            "params": null
        }))
        .await;
        let rid = resp["request_id"].as_str().expect("request_id present");
        assert!(!rid.is_empty(), "request_id must be non-empty");

        let resp = exchange(serde_json::json!({
            "jsonrpc": "2.0",
            "id": "rid-2",
            "method": "tools/call",
            "params": {
                "name": "describe_model",
                "arguments": {}
            }
        }))
        .await;
        let rid = resp["request_id"]
            .as_str()
            .expect("request_id present on tool call");
        // Must look like a UUIDv4 (36 chars, hyphenated).
        assert_eq!(rid.len(), 36, "request_id '{rid}' is not a UUID");
        assert_eq!(rid.chars().filter(|c| *c == '-').count(), 4);
    }

    #[tokio::test]
    async fn shared_state_survives_across_requests() {
        // Two sequential requests served from the same `Arc<Mutex<McpState>>`:
        // the first loads a model, the second queries `describe_model` which
        // must observe the state mutation from the first call. This proves
        // the `Arc<tokio::sync::Mutex<_>>` actually shares state across
        // requests (as opposed to the per-request local `McpState` the
        // pre-#2562 `RefCell` had).
        let state = Arc::new(Mutex::new(McpState::default()));

        // First request: load a model.
        let (mut w1, r1) = duplex(4096);
        let (server_write1, mut r1_client) = duplex(4096);
        let state_clone = Arc::clone(&state);
        let s1 = tokio::spawn(run_server(BufReader::new(r1), server_write1, state_clone));
        w1.write_all(
            serde_json::to_string(&serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "load_building_model",
                    "arguments": {
                        "num_zones": 2,
                        "zone_area": 50.0,
                        "window_u_value": 1.5,
                        "heating_setpoint": 21.0,
                        "cooling_setpoint": 26.0
                    }
                }
            }))
            .unwrap()
            .as_bytes(),
        )
        .await
        .unwrap();
        w1.write_all(b"\n").await.unwrap();
        w1.shutdown().await.unwrap();
        let mut buf1 = Vec::new();
        r1_client.read_to_end(&mut buf1).await.unwrap();
        s1.await.unwrap().unwrap();

        // Second request on the SAME shared state: describe_model must
        // observe the model loaded in the first request.
        let (mut w2, r2) = duplex(4096);
        let (server_write2, mut r2_client) = duplex(4096);
        let state_clone = Arc::clone(&state);
        let s2 = tokio::spawn(run_server(BufReader::new(r2), server_write2, state_clone));
        w2.write_all(
            serde_json::to_string(&serde_json::json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "describe_model",
                    "arguments": {}
                }
            }))
            .unwrap()
            .as_bytes(),
        )
        .await
        .unwrap();
        w2.write_all(b"\n").await.unwrap();
        w2.shutdown().await.unwrap();
        let mut buf2 = Vec::new();
        r2_client.read_to_end(&mut buf2).await.unwrap();
        s2.await.unwrap().unwrap();

        // Req 1: a successful load. handle_tool_call returns JSON which the
        // server re-parses, so `result` is an object.
        let resp1: serde_json::Value = serde_json::from_slice(&buf1).unwrap();
        let result1 = resp1["result"].as_object().expect("result as object");
        assert_eq!(result1["success"], serde_json::json!(true));
        assert_eq!(result1["model"]["num_zones"], serde_json::json!(2));

        // Req 2: describe_model must reflect the loaded model (not the empty
        // default McpState) — proof that the Arc<Mutex<...>> is actually shared.
        let resp2: serde_json::Value = serde_json::from_slice(&buf2).unwrap();
        assert_eq!(resp2["id"], 2);
        let result2 = resp2["result"].as_object().expect("result as object");
        assert_eq!(
            result2["num_zones"],
            serde_json::json!(2),
            "describe_model did not observe the loaded model: {result2:?}"
        );
    }
}
