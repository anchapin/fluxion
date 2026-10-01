# LIMIT-22 — investigation history

Narrative history for **LIMIT-22**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-22` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-22: Gauge-build-only test failures exposed by the exact Crank-Nicolson mass-state proxy (Issue #3297 aftermath)

- **Description:** `fd7ef13` (Issue #3297) replaced the PR2.5 trivial
  mass-state proxy
  (`t_mass = (h_tr_em·T_air + h_tr_3·T_air)/(h_tr_em + h_tr_3)`, which
  left the ~50–170 kWh non-zero strict-gate residual quoted in the
  #3297 issue body) with the exact Crank-Nicolson mirror of
  `InvariantChecker::zone_balance_for`
  (`step_dispatcher.rs::write_gauge_mass_state_proxy`). The strict-gate
  residual is now exactly 0, but three tests that passed before
  `fd7ef13` (verified passing at `fd7ef13^` = `832b0fe`) fail on the
  gauge build — each was passing for a physically-wrong reason:
  1. `test_case_950_mass_temperature_precooled_issue_1422` — the
     > 2 °C overnight mass pre-cool band (Issue #1422 regression
     guard) was satisfied by the trivial proxy writing
     `t_mass = t_air` (the air swing; no mass time constant). The
     exact CN node at Case 950's τ_mass ≈ 61 h attenuates a 12-h
     overnight air swing by 1/√(1 + (2π·61/12)²) ≈ 0.031 and swings
     +1.09 °C on the gauge air trajectory (legacy 5R1C: +2.41 °C at
     T_mass ≈ +41 °C July vs gauge ≈ −27.6 °C). The underlying gap is
     gauge air-trajectory fidelity for night-flush high-mass cases
     (§LIMIT-21 cohort).
  2. `test_case_960_inter_zone_heat_transfer_analysis` — passed
     pre-#3297 on the pure-legacy fall-through (the multi-zone hook
     was disabled); with the multi-zone arm re-enabled, the gauge
     multi-zone integration is oscillatory-unstable for the Case 960
     sunspace: step-level sunspace−back ΔT spikes reach ±140 °C
     (max +142.10 °C / min −135.88 °C) around the #3297 fail-closed
     [−50, 100] °C guard, while the annual means stay in-band
     (sunspace ≈ 13.8 °C, back ≈ 19.8 °C). The 60 °C inter-zone ΔT
     band is calibrated to the legacy 5R1C/9R4C trajectory (the
     un-guarded −162 °C sunspace divergence was the PR2.6 Case 960
     regression the fail-closed guard contains).
  3. `test_different_zones_respond_differently_to_targeted_gain` —
     the checker's 5R1C residual routes an artificial load gain
     through `φm · m_air_frac` only; with
     `m_air_frac = rad_frac · solar_distribution_to_air = 0` (this
     Case900 spec) the gain leverage is structurally zero, and the
     exact-CN proxy makes every zone imbalance exactly 0 (verified:
     `phi_m = 0.0000`, `residual = 0.000000`). The pre-#3297 pass was
     vacuous on the trivial proxy's non-zero baseline residual —
     sibling of §LIMIT-19 / #3103 and §MULTI-03 / #3066 (the
     `InvariantChecker` artificial-gain confusion family).
- **Affected Tests:** the three above, each quarantined
  **gauge-build-only** via
  `#[cfg_attr(feature = "gauge-solver", ignore = "...")]` with
  cross-references to this entry. The default-build assertions remain
  fully live and pass (2026-09-03 verification: all three green;
  zone_balance 19/0/2).
- **Affected Metrics:** gauge build only. Strict-gate residuals,
  energy numbers, and the legacy-path trajectory are unchanged (the
  §LIMIT-21 pre-existing set is the only remaining gauge-build
  failure cohort outside these quarantines).
- **Severity:** Low–Medium (test-side, feature-gated quarantines; no
  production code, threshold, baseline, or checker-formula change).
- **Un-quarantine criteria:** (1) gauge air trajectory matches the
  legacy night-flush pre-cool physics (or #1422 re-derives the band
  against a gauge-calibrated trajectory with maintainer sign-off);
  (2) gauge multi-zone integration stability lands for Case
  960-class configurations; (3) the §LIMIT-19 / #1344
  `EnergyBalanceValidator` investigation resolves the checker's
  zero-leverage artificial-gain formula. Unblocker: #3291 / #1465 /
  #1462.
- **GitHub Issue:** #3297 (this entry completes the issue's
  documentation of the remaining gauge-build regressions; the issue's
  "no regressions on the gauge baseline" criterion is met only in the
  sense that the regressions are now registered, cross-linked, and
  feature-gated — the physics gaps remain open).
- **Status:** 🔄 **Known structural gaps; quarantined gauge-build-only
  and routed to the GaugeSolver program.** Per AGENTS.md / RULES.md /
  ADR-0001, no test threshold was relaxed and no constant tuned — the
  assertions are retained verbatim under the quarantine markers.
