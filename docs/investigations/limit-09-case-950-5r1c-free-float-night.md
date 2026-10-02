# LIMIT-09 — investigation history

Narrative history for **LIMIT-09**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-09` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-09: Case 950 5R1C free-float night-vent override — pre-existing test failure (Issue #3071)

- **Description:** The companion integration test
  `tests/ashrae_140_blind_validation.rs::test_case_950_5r1c_free_float_uses_night_vent_overrides_issue_1422`
  has been observed failing identically on unmodified `develop` across
  multiple wave-orches­tration PRs (verified by sub-agents on #2871, #2898,
  #2903, and others). The empirical 5-day July average ΔT(07:00 − 06:00)
  measured by the test is ~+0.57 °C (range: +0.50 °C … +0.64 °C day-by-day),
  far below the >+1.0 °C threshold the test asserts. The diagnostic block
  prints:
  ```
  [#1422 Case 950FF free-float] 5-day July average ΔT(07-06) = +0.57°C
  thread '...' panicked at tests/ashrae_140_blind_validation.rs:2344:5:
  Case 950FF free-float zone T must rise > 1.0°C from 06:00 to 07:00 (night vent turns off) on average over 5 July days, got +0.57°C — structural fix to step_physics_9r4c may be reverted
  ```
  The ΔT collapse means the cached `derived_h_ext` / `derived_den` in
  `step_physics_9r4c` do not pick up the `h_ve_night` contribution that the
  test exercises — the 5R1C free-floating temperature (`t_i_free_5r1c`) is
  biased warm relative to the night-fan-off state the test simulates by
  turning off the fan at 07:00.

- **Affected Tests:**
  `tests/ashrae_140_blind_validation.rs::test_case_950_5r1c_free_float_uses_night_vent_overrides_issue_1422`
  (the integration test; now `#[ignore]`-quarantined with the reason
  `"Pre-existing failure tracked in #3071; blocked by #1422 + GaugeSolver
  #1465/#1462; once structural fix lands, re-test"`).
  The sibling diagnostic `test_case_950_mass_temperature_precooled_issue_1422`
  still passes and remains an enabled regression check.

- **Affected Metrics:** Case 950FF free-float 06:00 → 07:00 zone ΔT (°C)
  — a structural coupling-block diagnostic, not an ASHRAE 140 band metric.

- **Severity:** Low (no ASHRAE 140 reference band is gated on this test;
  the strict ±15 % annual-energy gate is already covered by
  `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`,
  and Cases 600 / 900 / 950 are NOT in that baseline per `release_gates.yaml`
  known structural failures).

- **GitHub Issue:** #3071 (this entry); root cause is tracked by #1422
  (night-vent override), #3059 (5R1C structural GaugeSolver work),
  and #3058 (Case 950FF night-vent mass coupling — same limitation).
  Long-term fix routed to GaugeSolver rework **#1465 / #1462**, which
  treats solar as geometric curvature rather than per-timestep energy
  injection (per AGENTS.md / RULES.md "fix the underlying math"; per-case
  parameter tuning to close this gap is explicitly out of scope).

- **Status:** 🔄 **Known pre-existing failure, quarantined pending GaugeSolver**.
  Re-enable once #1465 (or equivalent structural fix) lands and the
  ΔT(07-06) signal moves above the >+1.0 °C threshold on the standard
  `cargo test --test all_tests ashrae_140_blind_validation:: -- --ignored` run.
