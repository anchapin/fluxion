# LIMIT-33 — investigation history

Narrative history for **LIMIT-33**. The current state of this limitation is the
`LIMIT-33` row in `docs/KNOWN_ISSUES.md`; this file is the provenance behind it.

---

#### LIMIT-33: #4241 ideal-HVAC discrete storage residual inflates Case 600 annual loads ~50% — Case 600 heating leaves the published ±15% band (Issue #4314)

- **Discovered:** 2026-10-06, while triaging the three persistent red checks on
  `develop` @ `e93ba790` (Node/NAPI Bindings ubuntu-24.04 + windows-latest,
  Surrogate Drift Tolerance Gate #1784). All three fail deterministically with
  the same numbers on every run since PR #4258 merged (2026-09-29 21:50 UTC).
- **Symptom:** `npm/test.js` Case 600 heating assertion:
  `ASHRAE 600 annual heating 7181.51 kWh outside ±15% published band [4314, 5836]`.
  Surrogate fallback drift gate: peak drift 50.0000% vs the 2665.03 kWh fallback
  baseline recorded 2026-08-15.
- **Root cause (verified experimentally, 2026-10-06):** PR #4258 replaced the
  Norton-product ideal load with
  `Q = C_air*(T_sp − T_prev)/dt + h_coeff*(T_sp − T_free)`
  (`src/sim/thermal_model_physics/hvac.rs::compute_zone_hvac_load`). Split
  experiment (annual Case 600 engine loop, same tree, storage term disabled in
  place then reverted):
  - develop full #4241: heating **7181.507 kWh** (reproduces CI exactly)
  - storage term disabled, h_ve conductance correction kept: **5594.264 kWh**
    (inside the published band)
  - pre-#4241 recorded (napi, 2026-09-28): 4777.08 kWh
  The storage term therefore contributes **+1587 kWh/yr** and single-handedly
  pushes Case 600 out of the band; the h_ve conductance unification is
  band-compatible alone. Mechanism: the ideal load is a residual of the
  free-floating trajectory and is never applied back to the air-node state, so
  the node re-floats to `T_free` every timestep and the storage term re-charges
  `C_air*(T_sp − T_prev)/dt` on every heating hour rather than once per genuine
  recovery transient. PR #4258's own body flagged this term for double-counting
  review under a "Do not auto-merge" heading; the flagged review and the Case 900
  baseline question (gate widened to ±25% in the same PR) were not resolved
  before merge.
- **Surrogate drift gate:** same physics change, not a surrogate regression. The
  fallback baseline (2665.03 kWh, recorded 2026-08-15) predates #4241; the 50%
  drift is the #4241 load shift propagating through the fallback dispatch path
  (`set_loads` + `step_physics`).
- **Why not re-record:** re-recording the drift baseline and widening the npm
  heating assertion would entrench an unreviewed residual formulation while the
  engine sits 23% above the ASHRAE 140 published heating band. RULES.md forbids
  tuning validation outputs; the gates are the last published-band checks still
  holding (the Rust strict-energy ratchet baselines were re-recorded inside
  #4258 itself).
- **Intended fix:** decide the discrete ideal-loads semantics and implement them
  in `compute_zone_hvac_load` — either (a) feed the ideal load back into the
  air-node state so the zone is actually held at setpoint and the storage term
  charges once per recovery transient, or (b) justify and document the
  residual-only formulation and remove the storage term. Then re-record the
  recorded-value baselines (strict energy, fabric parity, Case 600 January grid
  baseline, surrogate fallback baseline) from the corrected engine. A physics
  change of this size goes through Alex's review before merge.
- **Owns:** Issue #4314.

### UPDATE 2026-10-07 — Option A (state feedback) implemented, pending review

Alex chose Option A ("pursue option A", 2026-10-06). Branch
`fix/limit33-state-feedback` off `origin/develop` @ `58d9af3b`.

**Engine change** (`step_5r1c.rs`, `t_i_act` update + `T_prev` source): when the
`#4241` residual demand is active and unclamped, `t_i_act` is set to the active
setpoint (heating sp if `Q > 0`, cooling sp if `Q < 0`); when capacity-clamped the
residual partial update `t_free + Q/den_true` is kept (bounds the feedback loop).
The controlled state is persisted where the next step already reads it:
`setpoints.temperatures` (assigned `t_i_act` at the end of every step) becomes the
`air_node_t_prev` source, so `T_prev` follows the controlled trajectory. One-step
fixed point, no intra-step iteration.

**Why not the brief's literal sketch** (copying `t_i_act` into
`mass.air_temperatures`): that variant reproduces the brief's numbers exactly
(5285.12 / 4433.17 kWh) but it overwrites the free-float air state, which is a
repo-wide convention: the #1295 strict energy-conservation invariant gate and the
gauge mass-state proxy both read `mass.air_temperatures` as the FREE state. With
the free state overwritten, the Case 600 conservation audit fails on all 168
steps (~937 W residuals). The implemented variant keeps the free state intact —
the audit stays exact — while still holding the zone at setpoint and charging the
storage term once per recovery transient.

**Case 600 (NAPI/WD600 drive, npm binding, measured this branch):** heating
7181.51 → **5590.48 kWh** (published [4314, 5836], in band), cooling
6026.39 → **4675.48 kWh** (published [4275, 5784], in band). **Case 900
unchanged** (9R4C-insensitive), fabric 9r4c rows and strict-energy
900/920/950/960/970 metrics reproduce develop exactly.

**Re-records (physics first, then baselines):**
- npm cooling recorded value 4010.89 → 4675.48 (`npm/test.js`; 54/54 pass).
- Strict-energy baseline: 600 H 7.707→6.000 MWh (gap 36.87→3.23), 600 C
  5.187→4.023 MWh (gap 0.00→5.01, moves just below band), 800 H 8.091→6.300
  (42.09→7.32), 800 C 4.270→3.314 (10.75→27.37). The 600 C and 800 C moves below
  band are honest consequences of the state-feedback semantics and are flagged
  in the PR for reviewer attention.
- Fabric parity baseline: only `case_600_5r1c` H/C and the Case 600 parity
  ratios move; gate green.
- Case 600 January grid baseline: thermal 178.318→139.351 kWh, electrical
  59.439→46.450 kWh, grid import 46.683→33.710 kWh, peak 1.570→1.222 kW.
- Surrogate fallback drift baseline 2665.03 → 4603.62 kWh: the measured value
  reproduces identically on develop and this branch (Case 900 9R4C-insensitive),
  so this clears the pre-existing #4241-era drift rather than a LIMIT-33 effect.

**Follow-up flagged:** exact-exponential hold metering (`Q_exact` with A's state
consistency) — at Case 600's dt/τ ≈ 7.5 the exact-exponential hold converges
toward the steady-state form, so it is a genuine candidate for a second study
(brief §6.5). Not implemented here.

**Status:** physics corrected, baselines re-recorded, gates locally green.
PR opened against develop; merge awaits Alex's review.
