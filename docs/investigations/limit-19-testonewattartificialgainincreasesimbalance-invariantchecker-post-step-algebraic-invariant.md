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

---

## 2026-11-13 addendum (loop round 140) — the re-derivation Alex routed: verdict

The round-140 primary item re-derived the one-watt finding at the current tip
(`a2a7397d`, develop after the #4804/#4812 wave) with a full closure
composition, per the standing re-derive/re-scope note. Method: the probe's
operating point (Case600 via `ThermalSelector::default()`, one step,
T_out 20 °C, dt 3600 s) was dumped record-and-state-complete (per-zone R77
mass record 12×`[a,b,d,bl]`, air-node record stride 9 live, hour-start mass
20.0, committed mass 19.868879893598628), the checker's composite replay was
replicated in Python from the dump, and the counterfactual was checked against
a REAL gained engine run (+1 W on zone 0 before the step, the exact
modification `check_invariant_with_artificial_gain` applies to `loads`).

**Verdict: accounted non-closure — records-only convention plus a
test-semantics gap. NOT an engine energy-balance break.** The measured facts:

1. **The engine closes bit-exactly.** `check_invariant` at the committed load
   reads **exactly 0.0** (the #3297 residual-zero property, live at this tip
   — the r138 InteriorSurfaceNetwork routing did not disturb it). The gained
   engine run's own committed-load identity **also reads exactly 0.0**. No
   imbalance exists at either load; the sacrosanct energy balance is intact.
2. **The −20.77657448921011 W is the checker's own probe output**, not an
   engine residual: `(t_committed − t_composite(load+Δ)) · d_last` with
   d_last = 36820.66871596761 W/K (the record's final sub-step CN
   denominator). The Python replication of the arm's delta-routed recurrence
   reproduces the checker's value **to the last digit** from the dumped
   record alone.
3. **The probe equals the true counterfactual.** The real gained engine run
   commits mass 19.868879893598628 → **19.86944415750836** (ΔT = +5.64e-4 K
   under +1 W over the hour); `(t_base − t_gained) · d_last` =
   **−20.776574488686855 W** vs the probe's −20.77657448921011 W — gap
   **5.23e-10 W** (2.5e-11 relative), which is the replay's different
   floating-point fold (replay-vs-engine at temperature level: 1.42e-14 K).
   The recorded affine algebra reproduces the engine's own response; the
   advance is affine in load and every load-dependent channel is recorded.
4. **Channel decomposition** (what the response is made of): the mass
   record's load-linear `bl` coefficients carry −8.6876162346903 W; the
   air-node record's `h3od` slot-4 coupling carries −12.08895825451981 W;
   together −20.77657448921011 W. Dropping the air channel misses 58% of the
   engine's response (bl-only replay lands 3.28e-4 K from the true gained
   advance) — the round-113 supplement's "composition misses ~1.5× until the
   channels are modeled" class, quantified here for the mass composite.
5. **The old assertion's premise is the bug.** "Increase ≈ 1 W" presumes
   pre-step first-law semantics: gain injected into a frozen state shows up
   as ~1 W of unbalanced input. The post-step algebraic identity evaluates at
   the committed state, so the probe measures how much the gain WOULD have
   moved the state — the −20.78 W reading is that response in identity units,
   and the engine balanced both runs exactly. This is the same
   hand-balance confusion §MULTI-03/#3066 named; the r140 attribution turns
   it from a hypothesis into a measurement.

**Endpoint taken (Option B-plus): the quarantine stands, now attributed.** The
`#[ignore]` reason carries the dated update; the assertion body is retained
untouched below it (per the original acceptance clause). The correct-semantics
contracts are pinned positive in `tests/one_watt_closure_contract.rs`:
committed-load closure bit-exact (C1), the probe value bit-exact so route
movement is visible (C2), and probe-equals-true-counterfactual at the
measured 1e-9 W bound (C3; the measured gap is 5.23e-10 W — a bound, not a
tolerance widened from anything: nothing pre-existing was relaxed). Teeth
measured at the tip: T1 delta-branch reduced to bl-only → {C2,C3};
T2 gains-partition env default flipped → {C2,C3} (closure is
route-independent by construction — the mirror tracks whatever route ran —
so only the value pins bite a route move; measured, correcting the predicted
{C1,C2,C3}); T3 record write disabled → {C1,C2,C3} (the fall-through arm
loses the bit-exact composite closure — ulp-class cancellation — so the
closure contract bites the record-write seam directly, which also makes
these contracts the accepted-step guard for the round-140 seam relocation). The #1344 EnergyBalanceValidator follow-up remains the long-term
owner of the product-surface diagnostic story; this addendum owns the unit
question.

Machine-record note: the standing note carried "one-watt
−20.77657448921011 W both trees (re-derive/re-scope)". The re-measurement at
this tip confirms the value to the last digit (the r138 routing did not move
it) and re-scopes its meaning: it was never the no-gain residual — the no-gain
residual is exactly 0 — it is the probe's state-response reading, stable
across tips because the recorded coefficients and drivers are stable.
