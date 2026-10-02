# FFD-02 — investigation history

Narrative history for **FFD-02**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `FFD-02` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### FFD-02: peak cooling load tolerance — STRUCTURAL (stub lacks zone air energy balance)

- **Issue:** [#2612](https://github.com/anchapin/fluxion/issues/2612) (test 2,
  `test_peak_cooling_load_tolerance`) — peak cooling load error ~100 % vs the
  10 % acceptance tolerance. Validates BES↔FFD coupled simulation against the
  NIST HVAC BESTEST 4.5 kW reference.
- **Affected Tests:** `tests/ffd_cosimulation_validation.rs::
  test_peak_cooling_load_tolerance` (`#[ignore]`-quarantined).
- **Severity:** Medium (accepted structural limitation).
- **GitHub Issue:** #2612 (open — needs real coupled BES↔FFD solver).
- **Status:** 🟡 **Structural limitation — documented; `#[ignore]` retained.**
- **Root Cause (Python-verified):** The `BuoyancyDrivenFfdSolver` stub returns a
  *constant* 293.15 K (20 °C) zone temperature regardless of the
  `BesToFfdBoundaryConditions` it receives — it has no zone air energy balance.
  The test's cooling-load estimator only fires when the zone exceeds 296.15 K,
  so `peak_cooling` is always 0 kW, giving exactly 100 % error vs the 4.5 kW
  reference. Reproduction: `cargo test --features fluxion-cfd --test
  ffd_cosimulation_validation -- --ignored --nocapture` prints
  `Peak cooling: reference=4.50 kW, simulated=0.00 kW, error=100.0%`.
- **Why this is NOT a fixable constant tweak (per AGENTS.md / RULES.md):**
  Closing the gap requires implementing a genuine coupled zone energy balance
  (real air-node thermal capacitance `C_air = ρ·cp·V`, HVAC supply-air coupling,
  envelope conduction feedback) so the zone temperature actually *responds* to
  the outdoor/surface/internal-gain boundary conditions. The 4.5 kW NIST HVAC
  BESTEST reference is a calibrated full-BES figure; making the stub emit it by
  choosing constants would be **parameter tuning to pass a system test** —
  explicitly forbidden. This is the same class of structural gap as the §LIMIT-05
  GaugeSolver-blocked diagnostics: the model topology does not yet implement the
  required physics, so the test is `#[ignore]`-quarantined until the coupled
  solver lands.
- **What was done in #2612:** The test compiles and runs under
  `--features fluxion-cfd` without panicking in the normal `cargo test` run (it
  is skipped). The `#[ignore]` message and doc comment were rewritten to point
  at the structural root cause and this section. Running with `--ignored`
  reproduces the documented 100 % gap as a close-out signal (same quarantine
  pattern as the §LIMIT-05 GaugeSolver-blocked tests).
- **Path forward (out of scope for #2612):** Implement a real coupled BES↔FFD
  zone energy balance (wire `fluxion-cfd`'s `FfdCfdSolver` through the
  `FfdSolver` trait adapter `src/sim/ffd_cfd_adapter.rs` with an air-node ODE),
  then remove the `#[ignore]` and assert the 10 % peak-cooling tolerance against
  the NIST reference.
