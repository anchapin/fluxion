//! ASHRAE Standard 140 Integration Test Suite
//!
//! Runs the implemented ASHRAE 140 test cases via the Informed
//! `ASHRAE140Validator::validate_analytical_engine()` path and asserts
//! meaningful, regression-catching bounds on Case 600.
//!
//! This replaces the previous `fluxion_value > 0.0` placeholder (issue #2683),
//! which was trivially true for almost any numeric output and gave false
//! confidence that validation was working. The assertion strategy mirrors
//! the sibling `tests/zone_balance_eplus_isolation.rs` pattern of an ACTIVE
//! infrastructure test plus an IGNORED strict-tolerance test:
//!
//!   1. `test_case_600_baseline` (ACTIVE) — asserts a physical-range floor
//!      on every Case 600 metric (catches NaN / negative / zombie outputs
//!      that `> 0.0` silently passed) PLUS a real reference-range assertion
//!      on the metric currently within band (annual heating).
//!   2. `test_case_600_full_reference_tolerance` (IGNORED) — the honest full
//!      bar: every Case 600 metric within its ASHRAE 140 reference range.
//!      Ignored because three of the four metrics are a known structural
//!      failure of the single-node 5R1C model (SOLAR-02 / LIMIT-05 in
//!      `docs/KNOWN_ISSUES.md`); un-ignoring now would fail CI on every PR
//!      for a documented gap. The annual cooling gap is additionally
//!      regression-gated by the strict-energy-gate workflow against a
//!      recorded baseline, so a WORSENING is still caught.

use fluxion::validation::{ASHRAE140Validator, MetricType, ValidationResult};

/// Find a single metric within the Case 600 results. Panics if absent —
/// callers must have already asserted the metric exists.
fn find_metric<'a>(results: &'a [ValidationResult], metric: &MetricType) -> &'a ValidationResult {
    results
        .iter()
        .find(|r| r.metric == *metric)
        .unwrap_or_else(|| panic!("Case 600 metric {metric:?} not found in results"))
}

#[test]
fn test_ashrae_140_comprehensive() {
    let validator = ASHRAE140Validator::new();
    let report = validator.validate_analytical_engine();

    report.print_summary();

    // Check overall pass rate
    let pass_rate = report.pass_rate();
    println!("Overall ASHRAE 140 Pass Rate: {:.1}%", pass_rate * 100.0);

    // In CI, we might want to assert a minimum pass rate
    // assert!(pass_rate >= 0.8, "ASHRAE 140 pass rate too low");
}

/// Case 600 baseline (ACTIVE infrastructure test, passes today).
///
/// Replaces the trivial `fluxion_value > 0.0` (issue #2683) with:
///   - a physical-range floor on every Case 600 metric, and
///   - a real reference-range assertion on annual heating (within band today).
///
/// The remaining three metrics (annual cooling, peak heating, peak cooling)
/// are outside band as a known structural failure; their full reference-
/// tolerance assertions live in `test_case_600_full_reference_tolerance`
/// below, kept `#[ignore]`'d so they do not block every PR on a documented
/// gap (see `docs/KNOWN_ISSUES.md` SOLAR-02 / LIMIT-05).
#[test]
fn test_case_600_baseline() {
    let validator = ASHRAE140Validator::new();
    let report = validator.validate_analytical_engine();

    let case_600: Vec<ValidationResult> = report
        .results
        .into_iter()
        .filter(|r| r.case_id == "600")
        .collect();
    assert!(
        !case_600.is_empty(),
        "Case 600 must produce validation results"
    );

    // The Case 600 HVAC path must emit all four expected metrics.
    for metric in [
        MetricType::AnnualHeating,
        MetricType::AnnualCooling,
        MetricType::PeakHeating,
        MetricType::PeakCooling,
    ] {
        assert!(
            case_600.iter().any(|r| r.metric == metric),
            "Case 600 missing metric {metric:?}"
        );
    }

    let heating = find_metric(&case_600, &MetricType::AnnualHeating);
    let cooling = find_metric(&case_600, &MetricType::AnnualCooling);
    let peak_heating = find_metric(&case_600, &MetricType::PeakHeating);
    let peak_cooling = find_metric(&case_600, &MetricType::PeakCooling);

    // Report observed values + reference bands for transparency so the
    // known cooling/peak gaps are visible under `--nocapture`.
    println!(
        "Case 600 (Informed validate_analytical_engine): \
         heating={:.3} MWh (ref {:.2}-{:.2}, within={}), \
         cooling={:.3} MWh (ref {:.2}-{:.2}, within={}), \
         peak_heating={:.3} kW (ref {:.2}-{:.2}, within={}), \
         peak_cooling={:.3} kW (ref {:.2}-{:.2}, within={})",
        heating.fluxion_value,
        heating.ref_min,
        heating.ref_max,
        heating.is_within_range(),
        cooling.fluxion_value,
        cooling.ref_min,
        cooling.ref_max,
        cooling.is_within_range(),
        peak_heating.fluxion_value,
        peak_heating.ref_min,
        peak_heating.ref_max,
        peak_heating.is_within_range(),
        peak_cooling.fluxion_value,
        peak_cooling.ref_min,
        peak_cooling.ref_max,
        peak_cooling.is_within_range(),
    );

    // Physical-range floor (the meaningful replacement for `> 0.0`).
    // Catches NaN, negative, and zombie outputs — e.g. a regression to the
    // historical 100 kW HVAC-capacity bug (LIMIT-03) — that `> 0.0` silently
    // passed. Bounds are generous: a 96 m² low-mass zone will not
    // legitimately use >50 MWh/year or peak >50 kW.
    for r in [heating, cooling] {
        assert!(
            r.fluxion_value.is_finite(),
            "Case 600 {:?} energy is non-finite ({})",
            r.metric,
            r.fluxion_value
        );
        assert!(
            r.fluxion_value >= 0.0,
            "Case 600 {:?} energy is negative ({})",
            r.metric,
            r.fluxion_value
        );
        assert!(
            r.fluxion_value < 50.0,
            "Case 600 {:?} energy {} MWh exceeds physical ceiling",
            r.metric,
            r.fluxion_value
        );
    }
    for r in [peak_heating, peak_cooling] {
        assert!(
            r.fluxion_value.is_finite(),
            "Case 600 {:?} peak is non-finite ({})",
            r.metric,
            r.fluxion_value
        );
        assert!(
            r.fluxion_value >= 0.0,
            "Case 600 {:?} peak is negative ({})",
            r.metric,
            r.fluxion_value
        );
        assert!(
            r.fluxion_value < 50.0,
            "Case 600 {:?} peak {} kW exceeds physical ceiling",
            r.metric,
            r.fluxion_value
        );
    }

    // Active reference-range assertion on the metric WITHIN band today:
    // annual heating (~4.6 MWh vs ref [4.0, 7.5]). This is a genuine
    // regression guard — a drift that pushes heating out of band now fails
    // CI, which the old `> 0.0` could never catch.
    assert!(
        heating.is_within_range(),
        "Case 600 annual heating {:.3} MWh drifted outside reference band [{:.2}, {:.2}]",
        heating.fluxion_value,
        heating.ref_min,
        heating.ref_max
    );
}

/// Case 600 STRICT full-reference tolerance — IGNORED pending physics fix.
///
/// This is the honest full bar: every Case 600 metric within its ASHRAE 140
/// reference range. It is `#[ignore]`'d because three of the four metrics
/// (annual cooling, peak heating, peak cooling) are a known structural
/// failure of the single-node 5R1C model — the discrete-node solar-injection
/// pathology documented in `docs/KNOWN_ISSUES.md` (SOLAR-02: annual cooling
/// under-predicted; LIMIT-05: simultaneous peak-cooling UNDER + peak-heating
/// OVER is the textbook signature of one lumped mass node on a 1-hour
/// timestep). Un-ignoring now would fail CI on every PR for a documented,
/// tracked gap — exactly the false-confidence anti-pattern of issue #2683's
/// inverse (a test that always fails on a known issue blocks every PR).
///
/// The annual COOLING gap is additionally regression-gated by the strict-
/// energy-gate workflow (`ashrae_140_strict_energy_gate.yml`) against a
/// recorded baseline (`tests/reference_data/zone_balance/strict_energy_gate_
/// baseline.json`, run via `tests/zone_balance_eplus_isolation.rs::
/// test_case_600_annual_energy_ashrae140_tolerance` with `--include-ignored`),
/// so a WORSENING of the cooling gap is still caught even while this test
/// stays ignored.
///
/// Engine values observed on the Informed `validate_analytical_engine()`
/// path (this test file, develop baseline 2026-08-11):
///   AnnualHeating 4.611 MWh  ref [4.00,  7.50]  within band  (PASS)
///   AnnualCooling 3.204 MWh  ref [7.00, 10.00]  outside      (~62% below midpoint)
///   PeakHeating   4.349 kW   ref [2.60,  4.00]  outside      (~+32% above midpoint)
///   PeakCooling   3.677 kW   ref [4.60,  6.00]  outside      (~-31% below midpoint)
///
/// Per AGENTS.md / RULES.md ("no parameter tuning, fix the math"), the fix
/// is owned by the physics layer (GaugeSolver #1465 / #1462, which treats
/// solar as geometric curvature rather than per-timestep energy injection,
/// per the post-#1323 / #1213 / #1328 cooling-load chain), not by loosening
/// the reference bands. When all four metrics re-enter their reference
/// bands, remove this `#[ignore]`.
#[ignore = "Issue #2683 / SOLAR-02 / LIMIT-05: three of four Case 600 \
            metrics (annual cooling, peak heating, peak cooling) are \
            outside the ASHRAE 140 reference band as a known structural \
            failure of the single-node 5R1C model (discrete-node solar-\
            injection pathology, docs/KNOWN_ISSUES.md). Un-ignoring now \
            would fail CI on every PR for a documented gap; the annual \
            cooling gap is regression-gated by the strict-energy-gate \
            workflow against tests/reference_data/zone_balance/\
            strict_energy_gate_baseline.json. Remove the #[ignore] when \
            the GaugeSolver (#1465/#1462) / post-#1323/#1213/#1328 cooling \
            fix closes all four gaps."]
#[test]
fn test_case_600_full_reference_tolerance() {
    let validator = ASHRAE140Validator::new();
    let report = validator.validate_analytical_engine();
    let case_600: Vec<ValidationResult> = report
        .results
        .into_iter()
        .filter(|r| r.case_id == "600")
        .collect();
    assert!(
        !case_600.is_empty(),
        "Case 600 must produce validation results"
    );

    for metric in [
        MetricType::AnnualHeating,
        MetricType::AnnualCooling,
        MetricType::PeakHeating,
        MetricType::PeakCooling,
    ] {
        let r = find_metric(&case_600, &metric);
        assert!(
            r.is_within_range(),
            "Case 600 {:?} = {:.4} outside reference band [{:.4}, {:.4}]",
            metric,
            r.fluxion_value,
            r.ref_min,
            r.ref_max
        );
    }
}

mod phase2 {
    #[test]
    #[ignore = "Issue #62 pending merge"]
    fn test_case_610_shading() {
        // ...
    }
}
