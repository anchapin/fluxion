//! Internal cross-module smoke tests.
//!
//! Each `#[test]` here composes multiple public modules to verify
//! they integrate cleanly. Larger integration tests live under
//! `tests/`.

use crate::invariant::{DefaultInvariantCheck, InvariantCheck};
use crate::kernel::{EdgeCase, Kernel, KernelInput, KernelOutput, ReferenceOutput};
use crate::latency::{time_kernel, LatencyAggregate, TimingConfig};
use crate::summary::{EvaluationOutcome, Summary, SummaryBuilder, CURRENT_SCHEMA_VERSION};

/// A concrete kernel that intentionally violates energy closure —
/// used to drive the failure-shape test below.
struct FailingKernel;
impl Default for FailingKernel {
    fn default() -> Self {
        Self
    }
}
impl Kernel for FailingKernel {
    fn evaluate(&self, input: &KernelInput) -> Result<KernelOutput, crate::kernel::KernelError> {
        // Returns 10× the input — guaranteed energy-closure violation.
        let payload = match &input.params {
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    serde_json::json!(f * 10.0)
                } else {
                    input.params.clone()
                }
            }
            serde_json::Value::Object(map) => {
                let mut new_map = serde_json::Map::new();
                for (k, v) in map {
                    if let Some(f) = v.as_f64() {
                        new_map.insert(k.clone(), serde_json::json!(f * 10.0));
                    } else {
                        new_map.insert(k.clone(), v.clone());
                    }
                }
                serde_json::Value::Object(new_map)
            }
            other => other.clone(),
        };
        Ok(KernelOutput { payload })
    }
}

#[test]
fn schema_version_is_one() {
    assert_eq!(CURRENT_SCHEMA_VERSION, 1);
}

#[test]
fn summary_round_trips_for_invariance() {
    let summary = Summary::new(
        SummaryBuilder::new("c", 0.5)
            .with_generation(1)
            .with_max_error(0.1)
            .with_eval_latency_ns(100)
            .with_eval_latency_spread_ns(10)
            .with_min_invariant_margin(0.1),
    );
    let json = serde_json::to_string(&summary).unwrap();
    let parsed: Summary = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, summary);
}

#[test]
fn default_invariant_checker_handles_pass_and_fail() {
    let check = DefaultInvariantCheck::new();
    let edge = EdgeCase {
        name: "demo".to_string(),
        input: KernelInput {
            case_name: "demo".to_string(),
            params: serde_json::json!({"x": 1.0}),
        },
        reference: ReferenceOutput {
            payload: serde_json::json!({"x": 1.0}),
        },
    };
    // Pass: candidate matches reference exactly.
    let ok_payload = KernelOutput {
        payload: serde_json::json!({"x": 1.0}),
    };
    let ok = check.check(&edge, &ok_payload, &edge.reference).unwrap();
    assert!(ok.min_margin > 0.0);

    // Fail: 10× offset breaches the 1e-6 tolerance.
    let bad_kernel = FailingKernel;
    let bad_output = bad_kernel.evaluate(&edge.input).unwrap();
    let err = check
        .check(&edge, &bad_output, &edge.reference)
        .unwrap_err();
    assert_eq!(err.invariant_kind, "energy_closure");
}

#[test]
fn timing_aggregator_produces_median_and_spread() {
    let cfg = TimingConfig::new().with_n(5).with_warmup(1);
    let agg = time_kernel(cfg, || ());
    assert_eq!(agg.samples, 5);
    // Aggregating a no-op loop yields zero-median and zero spread.
    let no_op_agg =
        LatencyAggregate::from_measurements(&[crate::latency::LatencyMeasurement(0); 5]);
    assert_eq!(no_op_agg.median_ns, 0);
    assert_eq!(no_op_agg.spread_ns, 0);
    let _ = agg;
}

#[test]
fn invariant_hard_fail_summary_has_zero_fitness() {
    let summary = Summary::invariant_hard_fail(
        "ctf-seed-0042",
        Some(137),
        Some(-0.5),
        vec![crate::invariant::InvariantViolation {
            case_name: "demo".to_string(),
            invariant_kind: "energy_closure".to_string(),
            observed: 1.0,
            threshold: Some(1.0e-6),
            message: "delta".to_string(),
        }],
    );
    assert_eq!(summary.fitness, 0.0);
    assert!(!summary.invariants_passed);
    assert_eq!(summary.outcome, EvaluationOutcome::InvariantHardFail);
    assert_eq!(summary.invariant_violations.len(), 1);
}
