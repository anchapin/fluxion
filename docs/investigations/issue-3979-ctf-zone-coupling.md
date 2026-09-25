---
summary-lines: 7
---

# Investigation: CTF↔Zone Coupling Defect and the 900-Series Bypass (Issue #3979)

<!--
Required 7-line summary block (docs gate, lines 2–8):
-->
Summary: Root-cause analysis of the ASHRAE 140 900-series energy catastrophe
(~10–14 metrics at +111% to +371% over reference mid). Verified mechanism: the
validator wired CTF backends into 9R4C-promoted high-mass models whose step path
adds the backend wall flux on top of the network's own conduction — a
full-magnitude double count. Fix: bypass (no conduction backend for conditioned
high-mass cases); CTF remains a 5R1C cross-check pending the #3980 FD teacher.
Affected: src/validation/ashrae_140_validator/mod.rs (build_case_model).
Status: bypass landed via #3979; residual ~0.5% under-prediction documented.

## 1. Symptom

`docs/ASHRAE140_RESULTS.md` reported catastrophic annual energies for every
conditioned high-mass case, stable across months of commits:

| Case | Metric | Value | Published range (MWh) |
|------|--------|-------|-----------------------|
| 900 | Heating | 5.053 | [1.17, 2.04] |
| 900 | Cooling | 7.754 | [2.13, 3.67] |
| 910 | Heating | 5.429 | [1.51, 2.28] |
| 920 | Heating | 5.354 | [3.26, 4.30] |
| 930 | Heating | 5.532 | [4.14, 5.34] |
| 940 | Heating | 6.967 | [0.79, 1.41] |

All ~2.2–3.3× the reference mid — while the *direct-construction* integration
tests (`tests/all_tests/ashrae_140_case_900.rs` etc.) and the
diagnostics-collector path reported in-band values for the same cases. Only the
`validate_analytical_engine` → `simulate_case` pipeline produced the blown
numbers (this is the path that feeds the results document).

## 2. Verified mechanism — additive-correction double counting

1. `simulate_case` built the model via `from_spec_with_selector`; for high-mass
   specs that constructor auto-promotes 5R1C → 9R4C (`SolverKind::NineRFourC`).
2. The (now removed) `enable_advanced_solver` then called
   `enable_ctf_with_fd_fallback(&wall_layers, 3600.0, 50, 5)`, wiring per-zone
   CTF solvers (`ConductionBackend::ctf_solvers`, `ctf_enabled = true`) into
   that 9R4C model — despite `enable_ctf`'s documented contract targeting
   *single-zone 5R1C* models (`src/sim/thermal_model_solvers.rs`).
3. Each step, `prepare_solvers_and_sol_air` computed the CTF wall flux and
   `step_physics_9r4c` (`src/sim/thermal_model_physics/physics_impl/step_9r4c.rs`,
   former lines ~297/~335) folded it into the air-node balance as an **additive**
   correction:
   `phi_ia_with_iz = phi_ia + Σ(ctf_flux_w) + Σ(fd_flux_w)`
   — on top of the 9R4C network's own envelope transmission already provided by
   `multi_node_solvers` (`step_with_gains`, the implicit FD multi-node
   discretization built at construction).
4. Net effect: full-magnitude double counting of wall conduction in the zone
   air heat balance → the ~2–3× energies above. The CTF 50-term convolution
   itself is mathematically sound (module-isolation tests pass); the defect is
   the *coupling*, not the method — consistent with the issue's evidence that
   9R4C-without-CTF reports Case 900 heating at ~1.29 MWh in-band (#2453).

Supporting evidence:
- Case 950 was already bypassed (#2363) with the comment "CTF may cause massive
  energy over-prediction; the blind path (which produces correct results)
  doesn't enable CTF" — the same defect observed in isolation.
- Free-floating high-mass cases never reached the mutator (explicit
  CTF↔zone feedback diverges without HVAC damping), and their metrics were not
  part of the catastrophe.

## 3. The bypass (landed)

`build_case_model` (extracted from `simulate_case` as the testable seam) now
builds every case via `from_spec_with_selector` with **no conduction backend**
wired. Guarded by
`test_case_900_model_has_no_conduction_backend_wired` (build path) and
`test_case_900_validator_pipeline_heating_in_widened_band` (annual pipeline).

Post-bypass, both validator paths agree (they differ only in warmup, which
shifts Case 900 heating by ~0.003 MWh):

| Path | Case 900 heating | Case 900 cooling |
|------|------------------|------------------|
| `simulate_case` (14-day warmup) | 1.1653 MWh | — |
| `simulate_case` (warmup disabled) | 1.1624 MWh | — |
| `simulate_case_with_diagnostics_collector` | 1.1624 MWh | 1.3093 MWh |

Case 900 heating moves from 5.05 → 1.165 MWh (from +215% over reference mid to
~0.5% below the published lower edge — inside the repo's ±15% widened
annual-energy gate; see §4).

## 4. Residual under-prediction (documented, not fixed here)

The honest post-bypass value sits ~0.5% below the RAW published lower edge
(1.165 vs 1.17 MWh) and cooling remains under-predicted (1.31 vs [2.13, 3.67]
MWh) — the pre-existing LIMIT-05-family under-prediction regime (#2453,
#3799). Per RULES.md no constants were tuned to close the gap. Root-causing
that regime belongs to the fine-grid implicit FD teacher work (#3980) and the
regression-test series (#3981).

## 5. Suspects for the CTF coupling itself (hypotheses for #3980)

Ordered by suspicion; none verified — the bypass makes them non-blocking:

1. **Substitutive-vs-additive semantics.** A backend flux is only valid in the
   9R4C/5R1C step paths if it *replaces* the network's own envelope term for
   the surfaces it covers. The 5R1C path consumes CTF flux alongside its own
   `h_tr_em` term as well — worth re-deriving both couplings from the zone
   balance before re-enabling CTF anywhere (see also
   `docs/investigations/issue-1280-ctf-peak-load.md`).
2. **Boundary-temperature mismatch.** `prepare_solvers_and_sol_air` fed the
   per-zone *roof* sol-air temperature to the wall CTF solvers as `t_exterior`
   (surface-orientation mismatch), and CTF coefficients were frozen at
   construction (3600 s, wall layers only) while the film coefficients they
   embed are time-varying.
3. **Sign convention at the interior node.** `CTFSolver::step` returns flux
   with a sign convention that must oppose the network's own `H_tr` direction;
   an inverted sign doubles instead of cancels.

## 6. Decision record

- CTF remains the fast cross-check for single-zone 5R1C (ADR-0017 posture) and
  is reachable via `ConductionSolverKind::Ctf` / `enable_ctf`.
- Conditioned high-mass cases run the 9R4C FD multi-node path backend-free
  until the equation-based DAE teacher (#3978, #3980) provides an authoritative
  envelope solver; the unconditional backend-wiring decision then gets revisited.
- The deprecated `enable_ctf_with_fd_fallback` mutator and its validator caller
  are decoupled; the mutator itself is untouched (public API stability).
