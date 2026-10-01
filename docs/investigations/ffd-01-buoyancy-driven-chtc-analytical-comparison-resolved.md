# FFD-01 — investigation history

Narrative history for **FFD-01**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `FFD-01` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### FFD-01: buoyancy-driven CHTC analytical comparison — RESOLVED (test-side Ra miscalculation)

- **Issue:** [#2612](https://github.com/anchapin/fluxion/issues/2612) (test 1,
  `test_buoyancy_driven_chtc_analytical`) — CHTC error ~161 % vs the ±15 %
  tolerance for the Chen & Griffith (1963) buoyancy-driven natural-convection
  benchmark.
- **Affected Tests:** `tests/ffd_cosimulation_validation.rs::
  test_buoyancy_driven_chtc_analytical` (was `#[ignore]`).
- **Severity:** N/A (was quarantined; now fixed).
- **Status:** ✅ **Fixed** — root cause was a test-side reference
  miscalculation, not an FFD solver bug.
- **Resolution Notes (Python-verified):** The test hard-coded the analytical
  reference Rayleigh number as `ra = 1.6e9`. For the stated configuration
  (L = 3 m, ΔT = 10 K, ν = 1.5e-5 m²/s, α = 2.1e-5 m²/s, air at 20 °C) the
  correct Rayleigh number is `Ra = g·β·ΔT·L³/(ν·α) ≈ 2.87e10` (Python:
  `(1/293.15)·9.81·10·27 / (1.5e-5·2.1e-5) = 2.8684e10`). The `1.6e9` value is
  an arithmetic mistake — it would require L ≈ 1.15 m, not 3 m. The
  `BuoyancyDrivenFfdSolver` stub computes Ra correctly from first principles, so
  it produced CHTC = 3.32 W/(m²·K) while the miscalculated reference gave 1.27
  W/(m²·K) → 161.7 % error. The fix computes the reference Ra from first
  principles (independent code path) rather than loosening the tolerance —
  required by RULES.md ("no parameter tuning / hardcoding to match"). After the
  fix the test validates the solver's Ra → Nu → CHTC pipeline produces a
  physically-sensible CHTC (3.32 W/(m²·K), within the ASHRAE natural-convection
  band 0.5–10 W/(m²·K)) and that Ra is in the turbulent regime (> 1e9). The
  `#[ignore]` is removed; the test now passes in the normal `cargo test` run.
