//! Per-month attribution diagnostic for ASHRAE 140 Case 950 HVAC-mode annual cooling.
//!
//! Issue #3551 — Case 950 (HVAC mode) annual cooling measures 33.08 kWh vs the
//! ASHRAE 140 reference band 390–920 kWh (~14× UNDER, ~91 % below the lower
//! bound). The §LIMIT-17 / #3058 companion Case 950FF night-vent free-floating
//! min temperature gap is −23.92 °C vs [−20.20, −17.80] °C band (3.72 °C
//! outside). The two signatures are **bidirectionally coupled** — no parameter
//! adjustment to `h_ve_night`, `MAX_CONVECTIVE_TO_AIR_MULTIPLIER`, or
//! `solar_distribution_to_air` can close both at once without violating
//! AGENTS.md / RULES.md / ADR-0001 ("no parameter tuning", "fix the underlying
//! math"). Per-case parameter tuning is explicitly out of scope.
//!
//! This diagnostic is the placeholder wiring for the Issue #3551 acceptance
//! criterion: "Per-month attribution test (`#[ignore]`-quarantined, runs
//! `--ignored --nocapture`) wired into CI that prints the hourly contribution
//! split (HVAC condensation vs mass release vs infiltration)". Full
//! implementation is a follow-up PR — the architectural fix is routed to the
//! GaugeSolver rework #1465 / #1462 (production-path switchover staged via
//! #3291 / PR #3482 — Phase A8 default flip, gated on the `gauge-solver`
//! cargo feature and §LIMIT-21 β-soak closure).
//!
//! Diagnostic output (run with `--ignored --nocapture`, full implementation):
//!   - Per-month HVAC-mode annual cooling attribution (kWh / month)
//!   - Hourly contribution split: HVAC condensation vs mass release vs
//!     infiltration for the 90-day cooling season (June–August)
//!   - Annual H / C / peak H / peak C band delta vs the ASHRAE 140
//!     reference bands (annual_cooling: 390–920 kWh; peak_cooling: 0.70–0.90 kW)
//!   - Cross-check against Case 950FF free-floating trajectory (must remain
//!     in [−20.20, −17.80] °C band per §LIMIT-17 regression-avoidance clause)
//!
//! Path forward (out of scope for #3551):
//! 1. Implement the per-month attribution walker (HVAC condensation / mass
//!    release / infiltration decomposition, hour-by-hour) using the existing
//!    `MultiNodeThermalSolver` instrumentation hooks.
//! 2. Add the `Case 950 (HVAC) annual cooling within [390, 920] kWh AND
//!    Case 950FF min free-floating within [−20.20, −17.80] °C simultaneously`
//!    assertion — the Issue #3551 acceptance criterion requires the
//!    **same single solver change** to close both signatures.
//! 3. Promote to a real gated test once the GaugeSolver #1465 / #1462
//!    production-path switchover lands.

// TODO: implement — see `docs/KNOWN_ISSUES.md` §LIMIT-24 (Issue #3551) for
// the per-metric engine-vs-reference table and the §LIMIT-17
// regression-avoidance clause that the future PR must satisfy.

#[test]
#[ignore = "Placeholder diagnostic for Issue #3551 — full implementation is a follow-up PR routed to GaugeSolver #1465 / #1462"]
fn test_case_950_hvac_mode_seasonal_attribution() {
    // TODO: implement — walk Case 950 (HVAC mode) hourly for the 90-day
    // cooling season and print the contribution split (HVAC condensation
    // vs mass release vs infiltration). Compare against the ASHRAE 140
    // reference bands (annual_cooling 390–920 kWh, peak_cooling
    // 0.70–0.90 kW) and the §LIMIT-17 Case 950FF free-floating
    // regression-avoidance clause.
    //
    // See `docs/KNOWN_ISSUES.md` §LIMIT-24 for the per-metric engine
    // output table and the closure-criterion statement.
}
