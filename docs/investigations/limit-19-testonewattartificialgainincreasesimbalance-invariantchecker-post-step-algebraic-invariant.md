# LIMIT-19 — investigation history

Narrative history for **LIMIT-19**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-19` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-19: `test_one_watt_artificial_gain_increases_imbalance` — InvariantChecker post-step algebraic-invariant confusion (Issue #3103)

- **Description:** The unit test
  `tests/invariant_checker_test.rs::test_one_watt_artificial_gain_increases_imbalance`
  fails on unmodified `develop` HEAD. The test calls
  `InvariantChecker::check_invariant_with_artificial_gain(&model, 3600.0, T_out, 1.0, 0)`
  and asserts `|balance_with_gain| > |balance_without_gain|` (and that
  the increase is ≈ 1.0 W within 0.1 W tolerance). Instead the residual
  *shrinks* in magnitude — the captured `Balance with 1W artificial gain: 225.9317696247872`
  is below the no-gain baseline (assertion at `tests/invariant_checker_test.rs:137`).
  The failure is the **unit-level analogue** of the pre-existing
  `InvariantChecker` post-step algebraic-invariant confusion
  characterised by **§MULTI-03 / Issue #3066** (the ~88.7 W hand-balanced
  stub residual on the 9R4C BE-implicit identity, resolved test-only by
  removing the over-strict `InvariantChecker` assertion and retaining
  only the `EnergyBalanceValidator` check):
  - The `InvariantChecker` evaluates the **post-step algebraic identity**
    of the 9R4C BE-implicit update (`denom · T_m_new − numer` where
    `T_s = (h_tr_ms·T_m_prev + h_tr_is·T_air + φ_st) / (h_tr_ms + h_tr_is + h_tr_me)`).
  - At hand-balanced states with `φ_st = 0`,
    `T_s = T_air · (h_tr_ms + h_tr_is) / (h_tr_ms + h_tr_is + h_tr_me) < T_air`
    whenever `h_tr_me > 0` (always true for high-mass construction).
  - When 1 W of gain shifts the post-step surface temperatures into this
    `T_s < T_air` regime, the algebraic identity can **decrease** in
    magnitude even though the integrator produced a `T_m_new` value —
    the test's `|balance_with_gain| > |balance_without_gain|` assertion
    is therefore not a robust invariant under the current solver
    topology (same mechanism as the §MULTI-03 88.7 W hand-balanced
    residual; see `tests/cli_multi_zone_energy_conservation.rs` lines
    119-152 for the #3066 fix that removed the over-strict
    `InvariantChecker` assertion).
  - The **integrated-flux `EnergyBalanceValidator`** (Issue #1344) is
    unaffected because it uses the `q_*` formulation which vanishes at
    `T_air = T_mass = T_outdoor` regardless of `h_tr_me` and `φ_st`. It
    is the documented product-surface diagnostic. The §MULTI-03 / #3066
    sub-agent explicitly noted: *"Pre-existing, unrelated failure
    confirmed via `git stash` round-trip: `invariant_checker_test::
    test_one_watt_artificial_gain_increases_imbalance` fails identically
    on unmodified `develop` and is outside the scope of #3066."*

- **Affected Tests:**
  `tests/invariant_checker_test.rs::test_one_watt_artificial_gain_increases_imbalance`
  (the unit test; now `#[ignore]`-quarantined with the reason
  `"Artificial gain should increase energy imbalance magnitude — LIMIT-19
  (Issue #3103, sibling-of-LIMIT-MULTI-03 #3066) — same InvariantChecker
  post-step algebraic-invariant confusion; the test asserts
  |balance_with_gain| > |balance_without_gain| but the algebraic
  identity shrinks in magnitude when gain shifts post-step surface
  temperatures. Tracked for follow-up alongside the #3066 /
  EnergyBalanceValidator (Issue #1344) investigation."`). The
  assertion body (both `gain_balance_abs > normal_balance_abs` and
  `(increase - 1.0).abs() < 0.1`) is retained below the `#[ignore]`
  marker for documentation; per AGENTS.md / RULES.md / ADR-0001, no
  parameter tuning is permitted on the `InvariantChecker` balance
  values to absorb the magnitude shrink.

- **Affected Metrics:** Test-only. No production validation impact —
  the `EnergyBalanceValidator` (Issue #1344) product surface is
  unaffected, and the `InvariantChecker` remains a valid diagnostic
  for *post-step* states where the integrator has produced
  `T_m_new` (see its module-level docs at
  `src/sim/invariant_checker.rs:1-132`).

- **Severity:** Low (test artefacts only; no effect on ASHRAE 140 pass
  rate, energy balance, or `EnergyBalanceValidator` output). For
  comparison, §MULTI-03 / #3066 was also Low severity at the
  integration-test layer — both are quarantined at the test layer with
  no production solver-code change.

- **GitHub Issue:** [#3103](https://github.com/anchapin/fluxion/issues/3103)
  (this entry). Sibling issue is **#3066 / §MULTI-03** (the
  `InvariantChecker` pre-step hand-balanced stub residual — same
  post-step algebraic-invariant confusion; resolved by removing the
  over-strict `InvariantChecker` assertion in
  `tests/cli_multi_zone_energy_conservation.rs`). Long-term resolution
  is the **`EnergyBalanceValidator` (Issue #1344)** follow-up
  investigation, which exposes the integrated-flux `q_*` form as the
  product-surface diagnostic. Per AGENTS.md / RULES.md "fix the
  underlying math" / "no parameter tuning" / "must-never hardcode
  results", per-case tuning of the `InvariantChecker` balance values
  is explicitly out of scope. Recommended direction from Issue #3103
  body: Option A (re-state the assertion in the integrated-flux
  `EnergyBalanceValidator` form, Issue #1344) or Option B
  (`#[ignore]` with linkage to #3066 — **this entry implements Option
  B**).

- **Status:** 🔄 **Known pre-existing failure, quarantined pending
  `EnergyBalanceValidator` investigation.** Re-enable once #1344 (or
  equivalent structural fix) lands and either (a) the test is
  re-stated in the integrated-flux form (`EnergyBalanceValidator`) per
  Option A of the #3103 issue body, or (b) the `InvariantChecker`
  post-step semantics are aligned so that magnitude-comparison
  assertions hold. Acceptance is dual: (a) the assertion body retained
  below the `#[ignore]` marker holds without any solver constant,
  balance, or assertion relaxation; (b) no `InvariantChecker` constant
  is tuned to absorb the magnitude shrink. Cohort-level tracking owned
  by Issue #3103; sibling tracking owned by Issue #3066.
