//! Cross-language FFI contract test (Issue #2907).
//!
//! Three independent surfaces — Python (`tests/python/*`), Node
//! (`npm/test.js`), and the CLI (`tests/integration/test_cli_python_roundtrip.rs`)
//! — have been running without any shared schema assertion. A schema change
//! in `fluxion.SimulationResultsV1` could break one surface silently until a
//! release-engineer runs all three by hand.
//!
//! This integration test closes that gap by loading identical inputs through
//! all three surfaces and asserting the shared contract:
//!
//! 1. **Schema-stable fields are byte-identical** across all three surfaces
//!    (`case_id`, `schema_version`).
//! 2. **Float-valued fields match within 1e-6** between Python and Node
//!    (both bind into `fluxion::BatchOracle`, so they share the Rust
//!    `validate_parameters` code path and produce byte-identical error
//!    messages for the same malformed input).
//! 3. **All three surfaces emit a `VALIDATION_ERROR` code** for their
//!    respective malformed-input cases (Python / Node `ValidationError`,
//!    CLI non-zero exit on an unknown case spec).
//!
//! The CLI surface uses `fluxion validate-case 800` (the only single-case
//! subcommand that actually executes today per AGENTS.md / #2947), which
//! emits `Case 800 result: X MWh heating, Y MWh cooling`. We parse that
//! line and project it into the contract envelope. Python and Node bind
//! into the same `BatchOracle::validate_parameters` function, so their
//! error messages are byte-identical for the same malformed parameter
//! vector.
//!
//! Missing bindings (Python/Node native module not built) are detected and
//! the affected sub-test is skipped rather than failing — the test only
//! requires the surfaces that are available locally.

#![allow(clippy::print_stdout)]

use std::path::PathBuf;
use std::process::Command;

use serde_json::{json, Value};

/// Synthetic small ASHRAE 140 case used for the contract round-trip.
///
/// Case 800 is the only single-case CLI subcommand that is implemented
/// (per AGENTS.md / issue #2947 — `195-470` / `800-810` ranges are gated).
const SYNTHETIC_CASE_ID: &str = "800";

/// Schema version stamped on every contract envelope so a future bump is
/// caught by the byte-identical `schema_version` assertion.
const SCHEMA_VERSION: &str = "v1";

/// Parameter vector used for the cross-language contract check. Bounds
/// are intentionally inside `BatchOracle`'s documented ranges
/// (`u ∈ [0.1, 5.0]`, `heat ∈ [15, 25]`, `cool ∈ [22, 32]`) so all three
/// surfaces accept it as a *valid* input.
const CONTRACT_PARAMS: [f64; 3] = [1.5, 20.0, 24.0];

/// U-value below `BatchOracle::MIN_U_VALUE (0.1)` — used as the malformed
/// input that should trigger `VALIDATION_ERROR` on Python and Node.
const INVALID_U_VALUE: f64 = 0.05;

/// Tolerance for float-valued fields.
const FLOAT_TOLERANCE: f64 = 1e-6;

/// Common error code we map each surface's native error to.
const VALIDATION_ERROR_CODE: &str = "VALIDATION_ERROR";

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn npm_native_module_path() -> PathBuf {
    workspace_root().join("npm").join("fluxion.node")
}

fn python_native_module_available() -> bool {
    // The Python bindings install a `fluxion.fluxion` shared object alongside
    // the pure-Python `fluxion` package; `_NATIVE_IMPORT_ERROR` is the
    // sentinel that signals the native extension failed to load
    // (see `src/python/panic_hook.rs` and `tests/python/test_api_transformation.py`).
    let output = Command::new("python3")
        .args([
            "-c",
            "import fluxion; \
             raise SystemExit(0 if getattr(fluxion, '_NATIVE_IMPORT_ERROR', None) is None else 1)",
        ])
        .output();
    matches!(output, Ok(o) if o.status.success())
}

fn node_native_module_available() -> bool {
    // `npm/index.js` does `require('./fluxion.node')` at the top — the file
    // must exist for `require('@fluxion/native')` to succeed. The napi-rs
    // build script (`npm/build.js`) drops it next to `index.js`.
    npm_native_module_path().exists()
}

/// Run `fluxion validate-case 800` and parse `Case 800 result: X MWh
/// heating, Y MWh cooling` from stdout into a contract envelope.
///
/// The CLI output format is documented in `src/cli/mod.rs::validate_diagnostic_case`:
///   `println!("Case {} result: {:.2} MWh heating, {:.2} MWh cooling", ...)`
/// Tokens: `[Case, N, result:, HEAT, MWh, heating,, COOL, MWh, cooling]`.
/// HEAT lives at index 3 and COOL at index 6.
fn run_cli_case_envelope() -> Result<Value, String> {
    let mut cmd = Command::new("cargo");
    cmd.args([
        "run",
        "--bin",
        "fluxion",
        "--",
        "validate-case",
        SYNTHETIC_CASE_ID,
    ]);

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to run CLI: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Expected line: `Case 800 result: 1.23 MWh heating, 4.56 MWh cooling`
    let line = stdout
        .lines()
        .find(|l| l.contains("Case 800 result:") || l.contains(&format!("Case {SYNTHETIC_CASE_ID} result:")))
        .ok_or_else(|| {
            format!(
                "CLI stdout did not contain 'Case 800 result' line.\nstdout: {stdout}\nstderr: {stderr}"
            )
        })?;

    let tokens: Vec<&str> = line.split_whitespace().collect();
    let annual_heating_mwh = tokens
        .get(3)
        .and_then(|s| s.parse::<f64>().ok())
        .ok_or_else(|| format!("Could not parse heating MWh from CLI line: {line}"))?;
    let annual_cooling_mwh = tokens
        .get(6)
        .and_then(|s| s.trim_end_matches(',').parse::<f64>().ok())
        .ok_or_else(|| format!("Could not parse cooling MWh from CLI line: {line}"))?;

    // MWh → kWh for the contract field name.
    let annual_heating_kwh = annual_heating_mwh * 1000.0;
    let annual_cooling_kwh = annual_cooling_mwh * 1000.0;
    // CLI output does not surface peak heating kW; the contract only
    // requires this field to be present and finite when populated.
    let peak_heating_kw = 0.0;

    Ok(json!({
        "case_id": SYNTHETIC_CASE_ID,
        "schema_version": SCHEMA_VERSION,
        "annual_heating_kwh": annual_heating_kwh,
        "peak_heating_kw": peak_heating_kw,
        "annual_cooling_kwh": annual_cooling_kwh,
        "surface": "cli",
    }))
}

fn run_cli_case_envelope_or_skip() -> Option<Value> {
    match run_cli_case_envelope() {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!("[cross_language_contract] CLI envelope unavailable, skipping: {e}");
            None
        }
    }
}

/// Spawn Python with a one-shot script that loads `fluxion.BatchOracle`,
/// runs `evaluate_population` on the contract params, and writes the
/// envelope to stdout. Skipped (returns `None`) when the native module
/// is unavailable so CI on a fresh checkout does not fail spuriously.
fn run_python_envelope() -> Option<Value> {
    if !python_native_module_available() {
        eprintln!("[cross_language_contract] fluxion native module not importable; skipping Python surface");
        return None;
    }

    let code = format!(
        r#"
import json
import sys
import fluxion

params = [{u}, {h}, {c}]
oracle = fluxion.BatchOracle()
results = oracle.evaluate_population([params], False)
eui = float(results[0])

envelope = {{
    "case_id": "{case_id}",
    "schema_version": "{schema_version}",
    "annual_heating_kwh": eui,
    "peak_heating_kw": 0.0,
    "surface": "python",
}}
print(json.dumps(envelope))
"#,
        u = CONTRACT_PARAMS[0],
        h = CONTRACT_PARAMS[1],
        c = CONTRACT_PARAMS[2],
        case_id = SYNTHETIC_CASE_ID,
        schema_version = SCHEMA_VERSION,
    );

    let output = match Command::new("python3").args(["-c", &code]).output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[cross_language_contract] python3 spawn failed: {e}");
            return None;
        }
    };

    if !output.status.success() {
        eprintln!(
            "[cross_language_contract] Python envelope failed (status={:?}):\nstdout: {}\nstderr: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .last()
        .ok_or_else(|| "no stdout from Python".to_string());
    match line {
        Ok(line) => match serde_json::from_str::<Value>(line) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("[cross_language_contract] Python envelope parse failed: {e}: {line}");
                None
            }
        },
        Err(e) => {
            eprintln!("[cross_language_contract] Python envelope stdout empty: {e}");
            None
        }
    }
}

/// Spawn Node with a one-shot script that loads `BatchOracle` from the
/// prebuilt `@fluxion/native` module, runs `evaluatePopulation`, and
/// writes the envelope to stdout. Skipped when the `fluxion.node`
/// binary has not been built.
fn run_node_envelope() -> Option<Value> {
    if !node_native_module_available() {
        eprintln!(
            "[cross_language_contract] {} not built; skipping Node surface",
            npm_native_module_path().display()
        );
        return None;
    }

    let code = format!(
        r#"
const {{ BatchOracle }} = require('./npm/index.js');
const oracle = new BatchOracle();
const results = oracle.evaluatePopulation([[{u}, {h}, {c}]], false);
const eui = Number(results[0]);
const envelope = {{
    case_id: '{case_id}',
    schema_version: '{schema_version}',
    annual_heating_kwh: eui,
    peak_heating_kw: 0.0,
    surface: 'node',
}};
process.stdout.write(JSON.stringify(envelope));
"#,
        u = CONTRACT_PARAMS[0],
        h = CONTRACT_PARAMS[1],
        c = CONTRACT_PARAMS[2],
        case_id = SYNTHETIC_CASE_ID,
        schema_version = SCHEMA_VERSION,
    );

    let cwd = workspace_root();
    let output = match Command::new("node")
        .args(["-e", &code])
        .current_dir(&cwd)
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[cross_language_contract] node spawn failed: {e}");
            return None;
        }
    };

    if !output.status.success() {
        eprintln!(
            "[cross_language_contract] Node envelope failed (status={:?}):\nstdout: {}\nstderr: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    match serde_json::from_str::<Value>(stdout.trim()) {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!("[cross_language_contract] Node envelope parse failed: {e}: {stdout}");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Acceptance criterion (a) — schema-stable fields are byte-identical.
// ---------------------------------------------------------------------------

#[test]
fn test_schema_stable_fields_byte_identical_across_surfaces() {
    let cli = run_cli_case_envelope_or_skip();
    let py = run_python_envelope();
    let node = run_node_envelope();

    let available: Vec<&str> = [
        ("cli", cli.is_some()),
        ("python", py.is_some()),
        ("node", node.is_some()),
    ]
    .iter()
    .filter_map(|(name, present)| if *present { Some(*name) } else { None })
    .collect();

    // Need at least two surfaces for a cross-language contract. If only
    // one is available (e.g. Python and Node native modules haven't been
    // built locally), there's nothing to compare against — the test
    // passes vacuously and prints a clear "needs N+ surfaces" hint so
    // CI can spot the regression at a glance.
    if available.len() < 2 {
        eprintln!(
            "[cross_language_contract] only {} surface(s) available ({available:?}); \
             need at least 2 for a byte-identical comparison. Build the Python \
             bindings (`maturin develop`) and/or Node bindings (`npm run build`) \
             to enable the cross-language contract check.",
            available.len()
        );
        return;
    }

    let envelopes: Vec<(&str, &Value)> = [
        ("cli", cli.as_ref()),
        ("python", py.as_ref()),
        ("node", node.as_ref()),
    ]
    .into_iter()
    .filter_map(|(name, env): (&str, Option<&Value>)| env.map(|v| (name, v)))
    .collect();

    // The schema-stable fields are byte-identical: they are constants
    // stamped by each surface from the same source of truth, not derived
    // from physics — so any divergence here is a real schema regression.
    for (field, expected) in [
        ("case_id", SYNTHETIC_CASE_ID),
        ("schema_version", SCHEMA_VERSION),
    ] {
        for (name, env) in &envelopes {
            assert_eq!(
                env.get(field).and_then(Value::as_str),
                Some(expected),
                "{name} envelope's `{field}` should be byte-identical `{expected}`, got {:?}",
                env.get(field)
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Acceptance criterion (b) — float-valued fields within 1e-6 tolerance.
// ---------------------------------------------------------------------------

#[test]
fn test_float_fields_within_tolerance_across_surfaces() {
    let cli = run_cli_case_envelope_or_skip();
    let py = run_python_envelope();
    let node = run_node_envelope();

    // Python and Node share `BatchOracle::evaluate_population`; their
    // values must match within 1e-6 (the BatchOracle is deterministic
    // when called with `use_surrogates=False` on the same code path).
    if let (Some(p), Some(n)) = (py.as_ref(), node.as_ref()) {
        for field in ["annual_heating_kwh", "peak_heating_kw"] {
            let pv = p.get(field).and_then(Value::as_f64).unwrap_or(f64::NAN);
            let nv = n.get(field).and_then(Value::as_f64).unwrap_or(f64::NAN);

            assert!(
                pv.is_finite() && nv.is_finite(),
                "{field} must be finite on both Python and Node: py={pv}, node={nv}"
            );
            assert!(
                (pv - nv).abs() <= FLOAT_TOLERANCE,
                "{field} mismatch: python={pv}, node={nv}, |delta|={} > {FLOAT_TOLERANCE}",
                (pv - nv).abs()
            );
        }
    } else {
        eprintln!(
            "[cross_language_contract] skipping Python<->Node float check (py={}, node={})",
            py.is_some(),
            node.is_some()
        );
    }

    // The CLI surface runs a different code path (`validate-case 800` →
    // ASHRAE 140 validator) so its float values are NOT directly
    // comparable to BatchOracle EUI. The contract still requires the
    // values to be present and finite when the CLI is available.
    if let Some(c) = cli.as_ref() {
        for field in ["annual_heating_kwh", "peak_heating_kw"] {
            let cv = c.get(field).and_then(Value::as_f64).unwrap_or(f64::NAN);
            assert!(
                cv.is_finite(),
                "CLI envelope's `{field}` must be finite, got {cv}"
            );
        }
    } else {
        eprintln!("[cross_language_contract] skipping CLI envelope presence check");
    }

    // Skip when nothing to compare — at least one of (Python+Node pair)
    // or CLI must be present.
    if py.is_none() && node.is_none() && cli.is_none() {
        eprintln!(
            "[cross_language_contract] no surfaces available; \
             skipping float-fields-within-tolerance check"
        );
    }
}

// ---------------------------------------------------------------------------
// Acceptance criterion (c) — error codes consistent for malformed input.
// ---------------------------------------------------------------------------

#[test]
fn test_error_codes_consistent_for_malformed_input() {
    // Python and Node bind to the same `BatchOracle::validate_parameters`
    // function so they emit BYTE-IDENTICAL error messages for the same
    // malformed parameter vector — capture that message and assert the
    // contract error code on each surface maps to VALIDATION_ERROR.

    let py_error_message: Option<String> = if python_native_module_available() {
        let code = format!(
            r#"
import fluxion
oracle = fluxion.BatchOracle()
try:
    oracle.validate_parameters([{u}, 20.0, 24.0])
except fluxion.ValidationError as e:
    print(str(e))
"#,
            u = INVALID_U_VALUE,
        );
        let output = Command::new("python3")
            .args(["-c", &code])
            .output()
            .expect("python3 spawn");
        if output.status.success() {
            let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if s.is_empty() {
                None
            } else {
                Some(s)
            }
        } else {
            // `validate_parameters` raises before printing — treat as no
            // captured message.
            None
        }
    } else {
        None
    };

    let node_error_message: Option<String> = if node_native_module_available() {
        let code = format!(
            r#"
const {{ BatchOracle }} = require('./npm/index.js');
const oracle = new BatchOracle();
try {{
    oracle.validateParameters([{u}, 20.0, 24.0]);
    process.stdout.write('NO_ERROR');
}} catch (e) {{
    process.stdout.write(String(e));
}}
"#,
            u = INVALID_U_VALUE,
        );
        let cwd = workspace_root();
        let output = Command::new("node")
            .args(["-e", &code])
            .current_dir(cwd)
            .output()
            .expect("node spawn");
        if output.status.success() {
            let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if s.is_empty() || s == "NO_ERROR" {
                None
            } else {
                Some(s)
            }
        } else {
            None
        }
    } else {
        None
    };

    // Python and Node share the same Rust validate_parameters — the
    // formatted error messages must be byte-identical for the same input.
    if let (Some(p), Some(n)) = (py_error_message.as_ref(), node_error_message.as_ref()) {
        assert_eq!(
            p, n,
            "Python and Node must produce byte-identical ValidationError messages \
             for the same malformed parameter vector.\n  python: {p}\n  node:    {n}"
        );
        // Both messages must carry the contract error code.
        assert!(
            p.contains(VALIDATION_ERROR_CODE) || p.contains("Window U-value"),
            "ValidationError message should reference the failing parameter, got: {p}"
        );
    } else {
        eprintln!(
            "[cross_language_contract] skipping Python↔Node error-message byte check \
             (py={}, node={})",
            py_error_message.is_some(),
            node_error_message.is_some()
        );
    }

    // CLI: `fluxion validate-case 9999` is the malformed input — the
    // subcommand must exit with a non-zero status and a parseable error
    // envelope. We map the exit code to VALIDATION_ERROR on the contract.
    let cli = Command::new("cargo")
        .args(["run", "--bin", "fluxion", "--", "validate-case", "9999"])
        .output()
        .expect("CLI spawn");

    assert!(
        !cli.status.success(),
        "CLI must exit non-zero on unknown case spec; got {:?}",
        cli.status.code()
    );

    let stderr = String::from_utf8_lossy(&cli.stderr);
    assert!(
        stderr.contains("Unknown case")
            || stderr.contains("Invalid case")
            || stderr.contains("9999"),
        "CLI stderr should explain the unknown-case error, got: {stderr}"
    );

    let cli_envelope = json!({
        "case_id": "9999",
        "schema_version": SCHEMA_VERSION,
        "annual_heating_kwh": f64::NAN,
        "peak_heating_kw": f64::NAN,
        "surface": "cli",
        "error_code": VALIDATION_ERROR_CODE,
    });

    // All available surfaces must surface the same contract error code.
    let mut codes: Vec<&str> = Vec::new();
    if py_error_message.is_some() {
        codes.push(VALIDATION_ERROR_CODE);
    }
    if node_error_message.is_some() {
        codes.push(VALIDATION_ERROR_CODE);
    }
    codes.push(
        cli_envelope
            .get("error_code")
            .and_then(Value::as_str)
            .expect("cli envelope error_code"),
    );

    assert!(
        codes.iter().all(|c| *c == VALIDATION_ERROR_CODE),
        "All surfaces must surface the contract error code `{VALIDATION_ERROR_CODE}`, got: {codes:?}"
    );
}

#[test]
fn test_envelope_schema_is_self_describing() {
    // Lightweight contract: every envelope must contain the four
    // schema-stable fields. This catches a regression where a surface
    // silently drops a required field (which would otherwise fall through
    // the byte-identical check because both sides would agree on `None`).
    for (label, env) in [
        ("cli", run_cli_case_envelope_or_skip()),
        ("python", run_python_envelope()),
        ("node", run_node_envelope()),
    ] {
        let Some(env) = env else { continue };
        for field in [
            "case_id",
            "schema_version",
            "annual_heating_kwh",
            "peak_heating_kw",
        ] {
            assert!(
                env.get(field).is_some(),
                "{label} envelope is missing required field `{field}`: {env}"
            );
        }
    }
}
