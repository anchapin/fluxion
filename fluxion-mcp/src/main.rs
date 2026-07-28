mod state;
mod tools;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cell::RefCell;
use std::io::Read;
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
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting fluxion-mcp server");

    let state = RefCell::new(McpState::default());

    loop {
        let mut input = String::new();
        match std::io::stdin().read_line(&mut input) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                tracing::error!("Failed to read stdin: {}", e);
                break;
            }
        }

        let request: JsonRpcRequest = match serde_json::from_str(&input) {
            Ok(req) => req,
            Err(e) => {
                tracing::error!("Failed to parse JSON-RPC request: {}", e);
                continue;
            }
        };

        let response = process_request(request, &state);

        let response_json = serde_json::to_string(&response).unwrap_or_else(|e| {
            tracing::error!("Failed to serialize response: {}", e);
            r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32603,"message":"Internal error"}}"#
                .to_string()
        });

        println!("{}", response_json);
    }

    Ok(())
}

fn process_request(request: JsonRpcRequest, state: &RefCell<McpState>) -> JsonRpcResponse {
    let id = request.id.clone();

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
        },
        "tools/list" => {
            let tools = tools::list_tools();
            JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(serde_json::json!({ "tools": tools })),
                error: None,
            }
        }
        "tools/call" => {
            let params = request.params.unwrap_or(serde_json::Value::Null);
            let mut state_guard = state.borrow_mut();
            let result_str = tools::handle_tool_call(&mut state_guard, params);
            // The result string is already formatted (JSON or TOON)
            // Wrap it as a JSON Value (String for TOON, Object for JSON parsed back)
            let result_value: serde_json::Value = if result_str.starts_with("toon:v1") {
                serde_json::json!({ "_toon": result_str })
            } else {
                // Parse JSON string back to Value for consistent structure
                serde_json::from_str(&result_str)
                    .unwrap_or_else(|_| serde_json::json!({ "raw": result_str }))
            };
            JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(result_value),
                error: None,
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
        },
    }
}
