//! Per-tool observability for the fluxion-mcp JSON-RPC handler (Issue #2515).
//!
//! Mirrors the pattern in `src/api/metrics.rs` (REST API): use only the public
//! `metrics` macros so the dispatch path stays free of recorder boilerplate.
//! fluxion-mcp is a stdio transport with no HTTP `/metrics` endpoint, so the
//! macros are no-ops until a recorder is installed (e.g. by a future HTTP
//! transport or by embedding fluxion-mcp in a host process). Tests use
//! `metrics_util::debugging::DebuggingRecorder` via `with_local_recorder` so
//! they can assert on emitted metrics without touching a global recorder.

use metrics::{counter, describe_counter, describe_histogram, histogram};
use serde_json::Value;

/// Histogram: per-tool dispatch wall-clock latency in seconds. Labeled `tool`.
pub const TOOL_DURATION_SECONDS: &str = "fluxion_mcp_tool_duration_seconds";

/// Counter: per-tool dispatch failures, labeled `tool` and `kind`.
pub const TOOL_ERRORS_TOTAL: &str = "fluxion_mcp_tool_errors_total";

/// Histogram buckets spanning a sub-millisecond tool lookup to a multi-second
/// annual `run_simulation`. Boundaries extend the REST API's HTTP buckets
/// (1 ms … 10 s) with a 25 ms step for the typical sub-100 ms tool calls:
///
///  1 ms · 5 ms · 10 ms · 25 ms · 50 ms · 100 ms · 500 ms · 1 s · 5 s · 10 s
///
/// Reserved for a future HTTP `/metrics` transport (fluxion-mcp currently uses
/// a stdio transport with no Prometheus exporter). The default recorder uses
/// these boundaries once installed via `PrometheusBuilder::set_buckets`.
#[allow(dead_code)] // reserved for a future metrics endpoint transport
pub const TOOL_DURATION_BUCKETS_SECONDS: &[f64] =
    &[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0];

/// Stable label value emitted for an empty / missing tool name so the
/// `fluxion_mcp_tool_*` series never carries an empty-string label (which is
/// legal Prometheus but confusing in dashboards).
pub const UNKNOWN_TOOL_LABEL: &str = "unknown";

/// Register `# HELP` / unit metadata for the MCP tool metrics. Safe to call
/// repeatedly; `describe_*` only records metadata with the global recorder and
/// is a no-op when no recorder is installed.
pub fn describe_metrics() {
    describe_histogram!(
        TOOL_DURATION_SECONDS,
        metrics::Unit::Seconds,
        "Wall-clock duration of a fluxion-mcp tool dispatch (tools/call)"
    );
    describe_counter!(
        TOOL_ERRORS_TOTAL,
        "Number of failed fluxion-mcp tool dispatches, labeled by tool name \
         and error kind (dispatch|invalid_params|tool_error)"
    );
}

/// Categorize a tool-call failure into a stable `kind` label for the
/// `fluxion_mcp_tool_errors_total` counter. Returns `None` when `result` is
/// not an in-band tool error.
///
/// Classification inspects only the parsed `Value` returned by
/// `handle_tool_call`. Three buckets cover every current error path in
/// `tools.rs`:
///
/// - `dispatch`: unknown tool name (or empty name), i.e. the
///   `_ => { "error": "Unknown tool: ..." }` branch.
/// - `invalid_params`: missing/empty required argument, e.g. "loop_id is
///   required", "changes object is required", "Explicit AI agent confirmation
///   required...".
/// - `tool_error`: any other in-band tool error (rate limit, value out of
///   range, no model loaded, unknown parameter, …).
pub fn categorize_error(tool_name: &str, result: &Value) -> Option<&'static str> {
    let err_str = result.get("error")?.as_str()?;
    if tool_name.is_empty() || err_str.starts_with("Unknown tool") {
        Some("dispatch")
    } else if err_str.contains("required") {
        Some("invalid_params")
    } else {
        Some("tool_error")
    }
}

/// Record the per-tool histogram and (on error) counter for one completed
/// dispatch. Synchronous so it can be exercised under a thread-local
/// `DebuggingRecorder` in tests (`metrics::with_local_recorder`).
///
/// `tool_name` is the raw `params.name` value from the JSON-RPC request (may
/// be empty); `elapsed_seconds` is the measured dispatch wall-clock time;
/// `result` is the parsed `Value` returned by `handle_tool_call`.
pub fn record_tool_outcome(tool_name: &str, elapsed_seconds: f64, result: &Value) {
    // The `metrics` macro label values require `'static` lifetime, so build an
    // owned `String` (mirrors the REST API in `src/api/metrics.rs`, which also
    // passes owned labels).
    let label = if tool_name.is_empty() {
        UNKNOWN_TOOL_LABEL.to_string()
    } else {
        tool_name.to_string()
    };

    histogram!(TOOL_DURATION_SECONDS, "tool" => label.clone()).record(elapsed_seconds);

    if let Some(kind) = categorize_error(tool_name, result) {
        counter!(TOOL_ERRORS_TOTAL, "tool" => label, "kind" => kind).increment(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(msg: &str) -> Value {
        serde_json::json!({ "error": msg })
    }

    #[test]
    fn categorize_unknown_tool_is_dispatch() {
        assert_eq!(
            categorize_error("", &err("Unknown tool: foo")),
            Some("dispatch")
        );
        assert_eq!(
            categorize_error("foo", &err("Unknown tool: foo")),
            Some("dispatch")
        );
        // Empty tool name with any error is still treated as a dispatch miss.
        assert_eq!(categorize_error("", &err("anything")), Some("dispatch"));
    }

    #[test]
    fn categorize_required_is_invalid_params() {
        assert_eq!(
            categorize_error("set_hvac_control_sequence", &err("loop_id is required")),
            Some("invalid_params")
        );
        assert_eq!(
            categorize_error(
                "set_hvac_control_sequence",
                &err("changes object is required")
            ),
            Some("invalid_params")
        );
        assert_eq!(
            categorize_error(
                "set_hvac_control_sequence",
                &err("Explicit AI agent confirmation required. Set confirm: true.")
            ),
            Some("invalid_params")
        );
    }

    #[test]
    fn categorize_other_errors_are_tool_error() {
        assert_eq!(
            categorize_error(
                "run_simulation",
                &err("No model loaded. Call load_building_model first.")
            ),
            Some("tool_error")
        );
        assert_eq!(
            categorize_error(
                "set_hvac_control_sequence",
                &err("Rate limit exceeded: maximum 5 control changes per minute")
            ),
            Some("tool_error")
        );
        assert_eq!(
            categorize_error("set_parameter", &err("Unknown parameter: foo")),
            Some("tool_error")
        );
    }

    #[test]
    fn categorize_success_is_none() {
        assert_eq!(
            categorize_error("load_building_model", &serde_json::json!({"success": true})),
            None
        );
        // A success result with rejected_changes (but no top-level "error") is
        // not counted as a dispatch failure.
        assert_eq!(
            categorize_error(
                "set_hvac_control_sequence",
                &serde_json::json!({"success": true, "rejected_changes": [{"error": "out of range"}]})
            ),
            None
        );
    }

    /// Assert the histogram is emitted with the correct `tool` label on a
    /// successful dispatch.
    #[test]
    fn record_outcome_emits_duration_histogram() {
        use metrics_util::debugging::DebuggingRecorder;

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            record_tool_outcome(
                "describe_model",
                0.0123,
                &serde_json::json!({"num_zones": 2}),
            );
        });

        let map = snapshotter.snapshot().into_hashmap();
        let found = map.keys().any(|ck| {
            ck.key().name() == TOOL_DURATION_SECONDS
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "tool" && l.value() == "describe_model")
        });
        assert!(
            found,
            "expected fluxion_mcp_tool_duration_seconds{{tool=\"describe_model\"}}"
        );
    }

    /// An empty tool name maps to the `unknown` label, not an empty string.
    #[test]
    fn record_outcome_uses_unknown_label_for_empty_name() {
        use metrics_util::debugging::DebuggingRecorder;

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            record_tool_outcome("", 0.001, &err("Unknown tool: "));
        });

        let map = snapshotter.snapshot().into_hashmap();
        let found = map.keys().any(|ck| {
            ck.key().name() == TOOL_DURATION_SECONDS
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "tool" && l.value() == UNKNOWN_TOOL_LABEL)
        });
        assert!(
            found,
            "expected empty tool name to map to '{UNKNOWN_TOOL_LABEL}' label"
        );
    }

    /// A dispatch failure increments the error counter with the right `kind`.
    #[test]
    fn record_outcome_emits_error_counter_on_failure() {
        use metrics_util::debugging::DebuggingRecorder;

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            record_tool_outcome(
                "set_hvac_control_sequence",
                0.0005,
                &err("loop_id is required"),
            );
        });

        let map = snapshotter.snapshot().into_hashmap();
        let found = map.keys().any(|ck| {
            ck.key().name() == TOOL_ERRORS_TOTAL
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "tool" && l.value() == "set_hvac_control_sequence")
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "kind" && l.value() == "invalid_params")
        });
        assert!(
            found,
            "expected fluxion_mcp_tool_errors_total{{tool=\"set_hvac_control_sequence\",kind=\"invalid_params\"}}"
        );
    }

    /// A successful dispatch must NOT emit the error counter.
    #[test]
    fn record_outcome_does_not_emit_error_counter_on_success() {
        use metrics_util::debugging::DebuggingRecorder;

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            record_tool_outcome("describe_model", 0.01, &serde_json::json!({"num_zones": 2}));
        });

        let map = snapshotter.snapshot().into_hashmap();
        let any_error = map.keys().any(|ck| ck.key().name() == TOOL_ERRORS_TOTAL);
        assert!(!any_error, "success must not emit the error counter");
    }
}
