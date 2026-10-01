# LIMIT-31 — investigation history

Narrative history for **LIMIT-31**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-31` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-31: CTF↔zone coupling defect — conduction backends bypassed for conditioned high-mass cases (Issue #3979)

- **Description:** The validator's `simulate_case` pipeline (feeding
  `validate_analytical_engine` → `docs/ASHRAE140_RESULTS.md`) wired CTF
  backends into high-mass models that `from_spec_with_selector` had already
  promoted to 9R4C. `step_physics_9r4c` folds the backend wall flux into the
  air-node balance as an *additive* correction
  (`phi_ia_with_iz = phi_ia + Σ(ctf_flux_w) + Σ(fd_flux_w)`) on top of the
  9R4C network's own multi-node FD envelope conduction — a full-magnitude
  double count of wall transmission. Result: every conditioned 900-series
  metric sat at ~2.2–3.3× reference mid (900 H = 5.05 MWh vs [1.17, 2.04];
  940 H = 6.97 vs [0.79, 1.41]) while direct-construction tests and the
  diagnostics-collector path (both backend-free) reported in-band values.
  **Fix (Issue #3979):** `build_case_model` no longer wires any conduction
  backend — conditioned high-mass cases run the 9R4C FD multi-node path
  backend-free; `enable_advanced_solver` was removed with its last caller.
  Case 900 heating recovers 5.05 → ~1.165 MWh (inside the ±15% widened
  annual-energy gate, ~0.5% below the raw published lower edge; the residual
  under-prediction is the §LIMIT-05-family regime, root-caused under #3980).
  CTF remains the single-zone 5R1C fast cross-check per ADR-0017
  (`ConductionSolverKind::Ctf` / `enable_ctf` are unchanged public API).
- **Status:** 🔄 Open — bypass landed (Issue #3979); substitutive-coupling
  re-derivation (and the frozen-coefficient / sol-air-boundary / sign-convention
  suspects) deferred to the fine-grid FD teacher work (#3980, #3981). Full
  analysis: `docs/investigations/issue-3979-ctf-zone-coupling.md`.
- **Impact:** ASHRAE 140 900-series conditioned annual energies (~10–14
  metrics) recover from catastrophic FAIL to the widened-gate / known-LIMIT
  regime.
