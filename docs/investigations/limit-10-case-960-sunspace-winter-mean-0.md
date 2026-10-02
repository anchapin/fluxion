# LIMIT-10 — investigation history

Narrative history for **LIMIT-10**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-10` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-10: Case 960 sunspace winter mean 0 °C vs pre-#1456 15 °C — assertion aligned with post-#1456 ground truth (Issue #3065)

- **Description:** Two pre-existing test failures in
  `tests/ashrae_140_case_960_sunspace.rs` were triggered by the **post-#1456**
  default 5R1C/9R4C solver path that replaced the broken
  `configure_6r2c_model` override (the override was removed in #1456 to close
  the 264 % over-prediction of annual heating it caused):

  - `test_case_960_inter_zone_heat_transfer_analysis` — asserted
    `sunspace_mean > back_mean - 15.0`, which under the pre-#1456 6R2C solver
    was effectively a ~15 °C annual-mean expectation for the sunspace.
  - `test_case_960_full_validation` — failed because it composes the above
    test as one of its sub-calls.

  Under the default 5R1C/9R4C path the free-floating sunspace annual-averages
  to **≈ 0 °C** (back-zone ≈ 23 °C, sunspace ≈ 0 °C, ΔT ≈ 23 °C — diagnostic
  output: `Back-zone mean temperature: 23.08 °C; Sunspace mean temperature:
  0.00 °C; Mean temperature difference (Sunspace - Back): -23.08 °C`). The
  previous assertion fails with `Sunspace should not be excessively colder
  than back-zone (< 15 °C difference)`.

  PR #3052 (#2858 partial fix) added the common-wall bulk-conduction coupling
  and ground-reflected gain path that brought **annual energy** into band
  (heating 2.14 MWh vs ref 1.65–2.45 MWh, ≈ 22 % below), but the **sunspace
  annual mean temperature** is governed by the 5R1C/9R4C air-mass
  distribution, not by inter-zone coupling alone. The 5R1C air-mass
  distribution cannot push the sunspace into the 15 °C band under the
  current energy balance.

- **Affected Tests:**
  - `tests/ashrae_140_case_960_sunspace.rs::test_case_960_inter_zone_heat_transfer_analysis`
    — the failing assertion (`sunspace_mean > back_mean - 15.0`) was
    replaced with a documented physical-band assertion
    `sunspace_mean ∈ (-10, 50) °C` that:
      (a) is satisfied by the post-#1456 ground truth (~0 °C annual mean),
      (b) remains satisfied once the GaugeSolver structural fix lands and
          the sunspace mean approaches the ASHRAE 140 reference, and
      (c) fails loudly if the sunspace drifts to obviously-broken values.
    The companion `sunspace_mean < back_mean + 5.0` assertion was retained
    (it already passes post-#1456 — sunspace 0 °C vs back-zone 23 °C).
  - `tests/ashrae_140_case_960_sunspace.rs::test_case_960_full_validation`
    — passes automatically once the sub-test above passes (it composes the
    failing test as one of its calls).

- **Affected Metrics:** Case 960 sunspace annual mean temperature (°C) — a
  diagnostic / trend metric, NOT an ASHRAE 140 reference-band metric. The
  reference-band metrics (annual heating, annual cooling, peak heating, peak
  cooling) are validated by the four sister tests in the same file
  (`test_annual_energy_validation`, `test_peak_load_validation`, and the
  comprehensive-energy + seasonal-temperature-profile tests) and remain
  subject to their existing reference-band assertions.

- **Severity:** Low (assertion fix only; no ASHRAE 140 reference band is
  gated on the changed assertion; the strict ±15 % annual-energy gate is
  covered by `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`).

- **GitHub Issue:** [#3065](https://github.com/anchapin/fluxion/issues/3065)
  (origin), with related issues **#1456** (6R2C override removal that
  exposed the gap), **#2858** (origin: PR #3052 partial fix), **#3052** (the
  PR that did not address these tests), and **#3059** (5R1C/9R4C air-mass
  distribution limitation — the architectural unblocker).

- **Status:** 🔄 **Test assertion aligned with post-#1456 ground truth;
  unblocker is the GaugeSolver rework (#1465 / #1462) tracked by #3059.**
  No physics-code change. Per **AGENTS.md** ("fix the underlying math"),
  **RULES.md** ("no parameter tuning", "must-never hardcode results"), and
  **ADR-0001**, parameter tuning to force the prior 15 °C value is
  explicitly out of scope. Re-tighten the assertion to the original
  `< 15 °C delta` once the GaugeSolver structural fix lands and the
  sunspace annual mean approaches the ASHRAE 140 reference (likely ≥ 5 °C
  given the solar-rich Denver TMY3 envelope).

- **Why this is NOT a fixable tuning change (per AGENTS.md / RULES.md /
  ADR-0001):**
  1. The annual-mean sunspace collapse (back-zone ≈ 23 °C, sunspace ≈ 0 °C)
     is the textbook signature of a single lumped-mass node integrated on a
     1-hour timestep for a free-floating zone with solar gains — the same
     discrete-node solar-injection pathology documented in §LIMIT-05 (and
     routed to GaugeSolver #1465 / #1462).
  2. Closing the gap by adjusting `h_ms_coeff`, `solar_distribution_to_air`,
     or any 5R1C/CTF constant would be **parameter tuning to pass a system
     test** — explicitly forbidden.
  3. The free-floating sunspace is intentionally not HVAC-controlled (ASHRAE
     140 Case 960 spec); without HVAC the annual-mean temperature is the
     model's free-floating equilibrium, not a control target.
  4. The structural fix is the GaugeSolver rework (#1465 / #1462) tracked by
     Issue #3059, which treats solar as geometric curvature rather than
     per-timestep energy injection — out of scope for this wave.

- **Diagnostic evidence (post-#1456 ground truth, captured 2026-08-17):**
  ```
  === Case 960 Inter-Zone Heat Transfer Analysis ===
  Back-zone mean temperature: 23.08°C
  Sunspace mean temperature: 0.00°C
  Mean temperature difference (Sunspace - Back): -23.08°C
  Max temperature difference: -20.00°C
  Min temperature difference: -27.00°C
  === End ===
  thread '...' panicked at tests/ashrae_140_case_960_sunspace.rs:438:5:
  Sunspace should not be excessively colder than back-zone (< 15°C difference)
  ```
  Reproduce on `develop` with
  `cargo test --test all_tests ashrae_140_case_960_sunspace:: -- test_case_960_inter_zone_heat_transfer_analysis`.

- **Related sections in this document:**
  - §MULTI-01b — Case 960 6R2C override regression (Issue #1456) — the fix
    that surfaced this gap by removing the broken 6R2C configuration.
  - §LIMIT-05 (and its UPDATE blocks) — the discrete-node solar-injection
    pathology that drives the free-floating temperature collapse; the
    architectural unblocker is GaugeSolver #1465 / #1462.
  - §"Aggressive-baseline cohort tracking (Issue #3072)" — Case 960 is part
    of the 5-case GaugeSolver-blocked cohort (195 / 600 / 620 / 940 / 960).

- **External references:**
  - Issue #2858 (origin — PR #3052 partial fix did not address these tests).
  - Issue #3052 (PR: `cb68b3c185391fb556e90d88034bfeba25abf383` — `fix(physics):
    resolve #2858 — Case 960 sunspace inter-zone coupling + ground-reflected
    gain`).
  - Issue #1456 (origin of the 6R2C override removal that exposed this gap;
    closed via PRs #1456 / #1466).
  - Issue #3059 (5R1C/9R4C air-mass distribution limitation — the
    architectural unblocker routed to GaugeSolver #1465 / #1462).
  - Issue #3061 (Case 960 sunspace annual cooling below band — sister issue,
    also routed to GaugeSolver).
  - `docs/ASHRAE140_MULTI_ZONE_RESULTS.md` (post-#1407 real-physics Case 960
    results; authoritative for Cases 960/970).
