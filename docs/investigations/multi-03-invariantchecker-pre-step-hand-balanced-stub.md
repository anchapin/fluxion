# MULTI-03 — investigation history

Narrative history for **MULTI-03**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `MULTI-03` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### MULTI-03: `InvariantChecker` Pre-Step Hand-Balanced Stub Residual (Issue #3066)

- **Description:** `tests/cli_multi_zone_energy_conservation.rs::test_two_zone_balanced_stub_passes` previously invoked `InvariantChecker::check_invariant` on a hand-balanced stub (T_air = T_mass = T_prev_mass = T_outdoor = 20 °C, all loads = 0). On the high-mass Case 960 (9R4C) branch the check returned ~88.7 W instead of zero. The residual is *structural*, not a bug:
  - The 9R4C `InvariantChecker` branch evaluates the BE-implicit algebraic identity `denom · T_m_new = numer` where `numer = cm/dt·T_m_prev + h_tr_em·t_sol_air + h_tr_3·T_s + φ_m` and `T_s = (h_tr_ms·T_m_prev + h_tr_is·T_air + φ_st) / (h_tr_ms + h_tr_is + h_tr_me)`.
  - At the hand-balanced stub `φ_st = 0`, so `T_s = T_air·(h_tr_ms + h_tr_is)/(h_tr_ms + h_tr_is + h_tr_me) < T_air` whenever `h_tr_me > 0` (always true for high-mass construction).
  - Substituting `T_m_prev = T_air` and `T_m_new = T_air` gives a residual of `h_tr_3 · T_air · h_tr_me / (h_tr_ms + h_tr_is + h_tr_me)` per zone. For Case 960 this is ~62 W (back-zone) + ~27 W (sunspace) ≈ 88.7 W total, matching the reported failure.
  - The integrated-flux `EnergyBalanceValidator` (the product surface for Issue #1344) is unaffected because it uses the `q_*` formulation which vanishes at `T_air = T_mass = T_outdoor` regardless of `h_tr_me` and `φ_st`.
- **Affected Cases:** Any high-mass multi-zone stub tested with the 9R4C `InvariantChecker` (Case 960, 970, and all 9R4C-routed cases).
- **Affected Metrics:** Test-only. No production validation impact.
- **Severity:** Low (test artefacts only; no effect on ASHRAE 140 pass rate, energy balance, or `EnergyBalanceValidator` output).
- **GitHub Issue:** [#3066](https://github.com/anchapin/fluxion/issues/3066)
- **Status:** ✅ Resolved (#3066, test-only). The `InvariantChecker` assertion has been removed from `test_two_zone_balanced_stub_passes`; the test now exercises only the `EnergyBalanceValidator`, which IS zero for the hand-balanced stub and is the Issue #1344 product surface.
- **Phase Addressed:** Phase Wave post-#1323
- **Resolution Notes:** The fix is a 23-line test-only change (`tests/cli_multi_zone_energy_conservation.rs` lines 119-152). No solver code modified. The `InvariantChecker` contract is preserved — it remains the correct diagnostic for *post-step* states where the integrator produced `T_m_new` (see its module-level docs at `src/sim/invariant_checker.rs:1-132`). For *pre-step* balanced stubs on the 9R4C path, it does not, by design, evaluate to zero unless `h_tr_me = 0` (single-lumped-mass-only construction).
