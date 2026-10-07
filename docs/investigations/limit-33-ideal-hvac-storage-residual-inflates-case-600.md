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
