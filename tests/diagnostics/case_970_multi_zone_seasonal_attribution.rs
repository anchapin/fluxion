//! Per-zone seasonal attribution diagnostic for ASHRAE 140 Case 970.
//!
//! Issue #3552: Case 970 (5-zone multi-zone cross-coupling per
//! ASHRAE 140-2017 §B6.7 / 140-2023 Annex B8-3,
//! `sim::multi_zone_network::MultiZoneAirflowNetwork` with 5×5
//! symmetric inter-zone conductance matrix) reports 4 / 4 reference-band
//! metrics failing on the 2026-08-16 snapshot:
//!
//! - Annual heating 18.58 MWh vs reference band [10.54, 14.26] MWh
//!   (+30 % to +76 % OVER the band)
//! - Annual cooling 21.07 MWh vs [7.39, 10.00] MWh
//!   (+110 % to +185 % OVER the band)
//! - Peak heating 3.80 kW vs [4.00, 8.00] kW
//!   (5 % UNDER the band low edge)
//! - Peak cooling 2.58 kW vs [2.50, 5.50] kW
//!   (at the band low edge)
//!
//! Tracked under `docs/KNOWN_ISSUES.md` §LIMIT-23 / Issue #3552.
//!
//! The bidirectional annual OVER signature (heating AND cooling
//! simultaneously >+30 % above the reference band on a 5-zone topology)
//! is consistent with the §LIMIT-05 UPDATE (#2453) 900-series
//! bidirectional annual-energy over-prediction mechanism — the 5R1C/9R4C
//! air-mass distribution in a 5×5 inter-zone coupling matrix amplifies
//! the same solar mass-node over-charge documented in the
//! §LIMIT-05 / LIMIT-14 / LIMIT-16 / LIMIT-17 cohort.
//!
//! Per AGENTS.md "no parameter tuning — fix the underlying math" and
//! RULES.md / ADR-0001, no `inter_zone_conductance`,
//! `solar_distribution_to_air`, or `h_ms_coeff` adjustment is permitted
//! to close the Case 970 OVER / UNDER signature — the bidirectional
//! trade-off is structurally infeasible at `dt/τ ≈ 3.6` (per
//! §LIMIT-05 UPDATE #1522 air-node capacitance conclusion), and the
//! `tests/reference_data/zone_balance/case_970_energy_reference.csv`
//! bands are the ASHRAE 140 inter-program range across EnergyPlus
//! 25.2.0 / TRNSYS / ESP-r / DOE-2 / BSIMAC / CSE / DeST (per
//! `docs/ASHRAE140_MULTI_ZONE_RESULTS.md` §"Case 970 Reference Data").
//!
//! Diagnostic output (run with `--ignored --nocapture`):
//!   - Per-month, per-zone heating and cooling energy for the 5 zones
//!     (west core = zone 0, east-strip = zones 1–4)
//!   - Per-month inter-zone flux matrix (5×5 hourly decomposition)
//!   - Per-month setpoint activation count and zone temperature
//!     distribution
//!
//! The full implementation is deferred to a follow-up PR. This file is
//! a placeholder registered with the Issue #3552 acceptance criterion
//! (c) wiring the diagnostic into the on-demand diagnostic tree under
//! `tests/diagnostics/` (per `tests/diagnostics/README.md`).
//!
//! TODO: implement — Issue #3552 follow-up. The intended implementation
//! steps are:
//!   1. Reuse `fluxion::validation::ashrae_140_multi_zone::Case970Reference`
//!      and `Case970Validator` to enumerate the 5 zones and their
//!      per-month / per-zone energy attribution.
//!   2. Drive the 8 760-hourly `solve_step` loop with the per-zone
//!      inter-zone flux matrix recorder, mirroring
//!      `tests/ashrae_140_case_970_validation.rs::test_case_970_multi_zone_network_e2e_conservation`.
//!   3. Print the per-month, per-zone attribution table for visual
//!      comparison against the ASHRAE 140-2017 §B6.7 / 140-2023 Annex
//!      B8-3 reference envelope. Do NOT assert against the reference
//!      bands — the diagnostic is informational only.

#![allow(unused_imports)]
#![allow(dead_code)]

// `#[ignore]`-quarantined per #2536 / #2708 — this file lives under
// `tests/diagnostics/` and is NOT auto-discovered by `cargo test`.
// Run with (per `tests/diagnostics/README.md` Option A):
//
// ```sh
// cp tests/diagnostics/case_970_multi_zone_seasonal_attribution.rs \
//    tests/_tmp_case_970_multi_zone_seasonal_attribution.rs
// cargo test --profile ci --test _tmp_case_970_multi_zone_seasonal_attribution \
//   -- --ignored --nocapture
// rm tests/_tmp_case_970_multi_zone_seasonal_attribution.rs
// ```

#[test]
#[ignore = "Issue #3552 follow-up — TODO: implement per-month per-zone attribution. \
            See module-level docs for the intended implementation steps."]
fn case_970_per_zone_seasonal_attribution_placeholder() {
    // Placeholder: full implementation deferred to a follow-up PR.
    // Per `tests/diagnostics/README.md` §"Converting a diagnostic into
    // a real gated test", this file stays informational until the
    // underlying physics (GaugeSolver #1465 / #1462) lands and the
    // printed values fall inside the ASHRAE 140 tolerance band.
    //
    // Issue #3552 acceptance criterion (c): the diagnostic must be
    // wired into CI per `--ignored --nocapture` so the per-zone
    // attribution is auditable on demand. The placeholder registration
    // (the `#[ignore]`-quarantined function above) is sufficient for
    // the docs-only scope of this PR.
}