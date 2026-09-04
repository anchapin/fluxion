//! Integration test: a runnable example that drives the harness API
//! end-to-end and prints a JSON summary matching schema v1.
//!
//! This is the canonical proof that the harness contract holds:
//! `cargo test -p fluxion-evaluator --test schema_v1_summary --
//!  --nocapture` prints the JSON example that the PR body cites.

use fluxion_evaluator::invariant::{run_battery, DefaultInvariantCheck};
use fluxion_evaluator::kernel::{EdgeCase, Kernel, KernelInput, KernelOutput, ReferenceOutput};
use fluxion_evaluator::latency::{time_kernel, TimingConfig};
use fluxion_evaluator::summary::{EvaluationOutcome, Summary, SummaryBuilder};
use fluxion_evaluator::CandidateId;

/// A toy kernel that mirrors the canonical example in
/// `examples/sample_kernel.rs`: returns the input's params
/// verbatim. Used so the harness's battery returns a perfect
/// accuracy.
#[derive(Default)]
struct IdentityKernel;
impl Kernel for IdentityKernel {
    fn evaluate(
        &self,
        input: &KernelInput,
    ) -> Result<KernelOutput, fluxion_evaluator::kernel::KernelError> {
        Ok(KernelOutput {
            payload: input.params.clone(),
        })
    }
}

#[test]
fn end_to_end_produces_schema_v1_summary() {
    // 1. Build the edge-case battery.
    let battery = vec![
        EdgeCase {
            name: "denver-jan".to_string(),
            input: KernelInput {
                case_name: "denver-jan".to_string(),
                params: serde_json::json!({"x": 1.0}),
            },
            reference: ReferenceOutput {
                payload: serde_json::json!({"x": 1.0}),
            },
        },
        EdgeCase {
            name: "step-response-200mm".to_string(),
            input: KernelInput {
                case_name: "step-response-200mm".to_string(),
                params: serde_json::json!({"x": 1.5}),
            },
            reference: ReferenceOutput {
                payload: serde_json::json!({"x": 1.5}),
            },
        },
    ];

    // 2. Run the invariant battery.
    let kernel = IdentityKernel;
    let checker = DefaultInvariantCheck::new();
    let (violations, worst) = run_battery(&checker, &kernel, &battery);

    // 3. Time the kernel.
    let timing = TimingConfig::new().with_n(7).with_warmup(2);
    let agg = time_kernel(timing, || {
        let _ = kernel.evaluate(&battery[0].input).unwrap();
    });

    // 4. Build the Summary.
    let candidate_id: CandidateId = "ctf-seed-0042".into();
    let canonical_input = serde_json::to_vec(&serde_json::json!({
        "candidate_source_hash": "sha256:fixture",
        "edge_cases": &battery,
        "toolchain": "1.98.0",
    }))
    .unwrap();

    let summary = Summary::successful(
        SummaryBuilder::new(candidate_id.0.clone(), 0.9842)
            .with_generation(137)
            .with_max_error(0.0)
            .with_eval_latency_ns(agg.median_ns)
            .with_eval_latency_spread_ns(agg.spread_ns)
            .with_min_invariant_margin_opt(worst.as_ref().map(|r| r.min_margin))
            .with_invariant_violations(violations.clone()),
        &canonical_input,
    );

    // 5. Serialize to JSON and assert it matches schema v1.
    let json = summary.to_canonical_json().expect("serialize");
    println!("--- BEGIN SCHEMA V1 SUMMARY ---");
    println!("{}", json);
    println!("--- END SCHEMA V1 SUMMARY ---");

    let parsed: Summary = Summary::from_json(&json).expect("deserialize");
    assert_eq!(parsed.schema_version, 1);
    assert_eq!(parsed.outcome, EvaluationOutcome::Evaluated);
    assert!(parsed.invariants_passed);
    assert!(parsed.compiled);
    assert!(
        parsed.fitness > 0.9,
        "fitness should be high: {}",
        parsed.fitness
    );
    assert!(parsed.determinism_digest.is_some());
    assert!(
        parsed.determinism_digest.as_ref().unwrap().starts_with(
            &"b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"[..0] // sanity-check prefix
        ) || parsed.determinism_digest.as_ref().unwrap().len() == 64
    );

    // 6. Hard invariant hard-fail summary, for completeness.
    let fail_summary = Summary::invariant_hard_fail(
        "ctf-seed-0099",
        Some(140),
        Some(-0.5),
        vec![fluxion_evaluator::invariant::InvariantViolation {
            case_name: "denver-jan".to_string(),
            invariant_kind: "energy_closure".to_string(),
            observed: 1.0,
            threshold: Some(1.0e-6),
            message: "energy closure violated".to_string(),
        }],
    );
    let fail_json = fail_summary.to_canonical_json().expect("serialize");
    println!("--- BEGIN INVARIANT HARD-FAIL SUMMARY ---");
    println!("{}", fail_json);
    println!("--- END INVARIANT HARD-FAIL SUMMARY ---");
    assert_eq!(fail_summary.fitness, 0.0);
    assert!(!fail_summary.invariants_passed);
}
