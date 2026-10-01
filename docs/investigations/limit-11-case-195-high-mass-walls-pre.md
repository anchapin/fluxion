# LIMIT-11 — investigation history

Narrative history for **LIMIT-11**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-11` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-11: Case 195 high-mass walls — pre-existing zero-energy assertion (Issue #3064)

- **Description:** The integration test
  `tests/ashrae_140_solid_conduction_variants.rs::test_case_195_high_mass_walls`
  has been observed failing identically on unmodified `develop` across
  multiple wave-orchestration PRs (verified by sub-agents on the originating
  #2868 wave and PR #3044 sub-agent report). The failure is on the
  zero-energy assertion:

  ```
  === ASHRAE 140 Case 195: High-Mass Walls Variant ===
  Baseline Case 195: 195
  High-mass variant: 195-HM
  Construction: HighMass

  Energy Results:
    Baseline energy: -18.21 kWh
    High-mass energy: 0.00 kWh

  thread 'test_case_195_high_mass_walls' (1740812) panicked at tests/ashrae_140_solid_conduction_variants.rs:73:5:
  High-mass model should produce non-zero energy consumption
  ```

  Reproduce on `develop` with
  `cargo test --test all_tests ashrae_140_solid_conduction_variants:: test_case_195_high_mass_walls -- --nocapture`.

  The low-mass baseline produces a small negative residual (`-18.21 kWh` —
  the no-loads / no-solar envelope with ε_ext = 0.1 from the #2868 fix), but
  the high-mass variant returns an **exactly** zero energy. This is the
  classic signature of a mass-node initial temperature that was already
  perfectly matched to the zone setpoint at t=0, leaving no driving
  temperature difference for the no-loads / no-solar envelope over the
  8760 h horizon — i.e. the high-mass construction is being initialized at
  thermal equilibrium with the steady-state no-loads / no-solar boundary
  conditions rather than at a perturbed starting state that would let the
  envelope drive heat flow.

  The low-mass `Case195HighMass.spec()` configuration that drives this
  includes the same zero-loads / zero-solar envelope as the no-loads /
  no-solar variants; the only structural change is the multi-node wall
  capacitance and the HighMass `ConstructionType` flag. PR #3044 fixed
  three coupled bugs in the **low-mass** Case 195 path (t_i_act divisor,
  H_tr,3 degenerate-to-0, hard-coded ε_ext=0.9) that dropped annual
  heating from 6810 kWh to 3238 kWh, but did not address the additional
  mass-node initialization needed by the high-mass variant. Per
  AGENTS.md "fix the underlying math" and RULES.md "no parameter tuning"
  / "must-never hardcode results", adding a settle loop or adjusting the
  initial mass temperature is **out of scope** for a parallel sub-agent —
  the structural fix is the GaugeSolver rework routed to Issue #3059.

- **Affected Tests:**
  `tests/ashrae_140_solid_conduction_variants.rs::test_case_195_high_mass_walls`
  (the failing test; now `#[ignore]`-quarantined with the reason
  `"Pre-existing zero-energy assertion failure; tracked in #3064,
  blocked by GaugeSolver structural rework #1465/#1462; once #3059
  lands, re-test"`).
  The companion integration test
  `tests/ashrae_140_solid_conduction_variants.rs::test_solid_conduction_variants_integration`
  is **also failing on unmodified `develop`** for the same root cause
  (the integration assertion is `pass_rate > 80.0`, and the high-mass
  variant returning 0.00 kWh drops the rate to 75.0% which is below 80.0%).
  This integration failure is a **pre-existing** wave-orchestration known
  issue, OUT OF SCOPE per the Issue #3064 PR scope ("Mark the failing
  test as `#[ignore]`" — singular) and explicitly excluded from this
  quarantine PR. Tracked separately; the orchestrator should treat the
  integration test failure as a follow-up issue (recommended: a parallel
  LIMIT-12 quarantine PR or a follow-up fix to the >80% threshold).

- **Affected Metrics:** Case 195 high-mass annual energy (kWh) — a
  diagnostic / trend metric, NOT an ASHRAE 140 reference-band metric. The
  low-mass Case 195 reference-band metrics (annual heating, annual
  cooling, peak heating, peak cooling) are validated by the eight tests
  in `tests/ashrae_140_case_195_solid_conduction.rs` and remain subject
  to their existing assertions.

- **Severity:** Low (no ASHRAE 140 reference band is gated on this test;
  the strict ±15% annual-energy gate is covered by
  `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`,
  and Case 195 is NOT in that baseline per `release_gates.yaml` known
  structural failures).

- **GitHub Issue:** [#3064](https://github.com/anchapin/fluxion/issues/3064)
  (this entry), with related issues **#2868** (origin — Case 195 annual
  heating over-prediction; PR #3044 fixed the low-mass variant),
  **#3044** (the PR that did not address high-mass variant),
  **#3059** (5R1C/9R4C air-mass distribution limitation; unblocker).
  Long-term fix routed to GaugeSolver rework **#1465 / #1462**, which
  treats solar as geometric curvature rather than per-timestep energy
  injection (per AGENTS.md / RULES.md "fix the underlying math";
  per-case parameter tuning to close this gap is explicitly out of scope).

- **Status:** 🔄 **Known pre-existing failure, quarantined pending GaugeSolver**.
  Re-enable once #1465 (or equivalent structural fix) lands and the
  high-mass energy moves off the zero floor on the standard
  `cargo test --test all_tests ashrae_140_solid_conduction_variants:: -- --ignored` run.

- **Why this is NOT a fixable tuning change (per AGENTS.md / RULES.md /
  ADR-0001):**
  1. The exactly-zero annual energy is the textbook signature of an
     initial-condition equilibrium trap: the mass node is at the same
     temperature as the steady-state no-loads / no-solar envelope, so
     the envelope has no thermal driving force over the 8760 h horizon.
     Closing this by warming the initial mass temperature, adding a
     settle loop, or perturbing the initial state would be **parameter
     tuning to pass a system test** — explicitly forbidden by RULES.md
     "no parameter tuning" and "must-never hardcode results".
  2. The no-loads / no-solar envelope is intentionally not HVAC-controlled
     (ASHRAE 140 Case 195 spec); without HVAC and without solar / internal
     gains, the annual-mean energy is the model's free-floating
     equilibrium, not a control target.
  3. The structural fix is the GaugeSolver rework (#1465 / #1462) tracked
     by Issue #3059, which treats mass-node initialisation consistently
     across the 5R1C / 9R4C / multi-node paths — out of scope for this
     wave.

- **Diagnostic evidence (post-#3044 ground truth, captured 2026-08-17):**
  ```
  === ASHRAE 140 Case 195: High-Mass Walls Variant ===
  Baseline Case 195: 195
  High-mass variant: 195-HM
  Construction: HighMass

  Energy Results:
    Baseline energy: -18.21 kWh
    High-mass energy: 0.00 kWh

  thread 'test_case_195_high_mass_walls' (1740812) panicked at tests/ashrae_140_solid_conduction_variants.rs:73:5:
  High-mass model should produce non-zero energy consumption
  ```
  Reproduce on `develop` with
  `cargo test --test all_tests ashrae_140_solid_conduction_variants:: test_case_195_high_mass_walls -- --nocapture`.

- **Related sections in this document:**
  - §LIMIT-08 — Case 195 (no-loads) peak heating weather-file gap (Issue
    #2868 — partially resolved by PR #3044).
  - §LIMIT-05 (and its UPDATE blocks) — the discrete-node solar-injection
    pathology that drives free-floating temperature collapse; the
    architectural unblocker is GaugeSolver #1465 / #1462.
  - §LIMIT-09 — Case 950 5R1C free-float night-vent override (Issue
    #3071) — same quarantine template, same unblocker.
  - §LIMIT-10 — Case 960 sunspace winter mean (Issue #3065) — same
    quarantine template, same unblocker.
  - §"Aggressive-baseline cohort tracking (Issue #3072)" — Case 195 is
    part of the 5-case GaugeSolver-blocked cohort (195 / 600 / 620 /
    940 / 960).

- **External references:**
  - Issue #2868 (origin — Case 195 annual heating over-prediction;
    closed via PR #3044 for the low-mass variant).
  - PR #3044 (the PR that did not address the high-mass variant;
    `fix(ashrae140): resolve #2868 — Case 195 t_i_act / H_tr,3 / ε_ext`).
  - Issue #3059 (5R1C/9R4C air-mass distribution limitation — the
    architectural unblocker routed to GaugeSolver #1465 / #1462).
- `tests/ashrae_140_solid_conduction_variants.rs` (line 35 — the
     quarantined `test_case_195_high_mass_walls` test).
  - `tests/ashrae_140_case_195_solid_conduction.rs` (sibling tests
     for the low-mass variant; all currently passing post-#3044).
