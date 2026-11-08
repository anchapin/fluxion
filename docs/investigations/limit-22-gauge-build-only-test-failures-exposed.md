# LIMIT-22 — investigation history

Narrative history for **LIMIT-22**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-22` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 2. No wording was changed, softened or deleted.

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


---

#### 2026-10-26 (physics loop round 102): scoping the φ_st-through-t_s arm — the 5R1C checker mirror is probe-gain-blind by construction post-#4582 (r101)

- **Premise verified in code (this entry's item 3; the #3103/#3066
  family).** Since round 101 the engine routes internal-gain radiative
  to the surface node in full (`step_5r1c.rs` φ_st/φ_st_int partition
  at the §C.3 site), and the checker's 9R4C arm mirrors that: it
  rebuilds `t_s = (h_tr_ms·T_m_prev + h_tr_is·T_air + φ_st) /
  (h_tr_ms + h_tr_is + h_tr_me)` with the full radiative term
  (`invariant_checker.rs`, `NineRFourC` branch), so the artificial-gain
  probe has leverage there. The 5R1C arms do not: the R77
  composite-coefficient arm replays persisted `(a_k, b_k)` sub-step
  coefficients that encode the engine's own committed-hour drive (the
  checker records that the load delta's φ_m share is structurally zero
  since round 101); the per-class t_s-first mirror's mass-drive numer
  carries only φ_m; the legacy lumped identity likewise.
  `check_invariant_with_artificial_gain` on a 5R1C state therefore
  compares two identical residuals — zero leverage by construction.
  The engine reaches the mass through the surface layer (the composite
  coefficients encode the `h_tr_ms·t_s` path); the mirror simply does
  not rebuild that path from a perturbed load. This is an instrument
  blind spot, not an engine gain-routing defect.
- **Should the mirror carry φ_st through t_s? Yes, in principle.** For
  the probe to have 5R1C leverage the mirror must rebuild the per-class
  surface algebra from the perturbed load exactly as the engine's
  t_s-first layer does (per-class φ_st shares, the R77 sub-step φ
  ramping, then the surface→mass path), keeping the without-gain
  residual bit-zero — the 9R4C arm is the worked pattern (rebuild `t_s`
  analytically from the same scalars the integrator consumed).
- **Why scoped, not attempted this round:** the 5R1C t_s-first layer is
  sub-stepped (R77 coupled-N, default 12 sub-steps, per-class sol-air
  and ramped drives), and the composite-coefficient arm exists
  precisely because a closed hourly form stopped matching the composite
  advance (#3983 §64). A correct rebuild re-derives the composite
  sub-step algebra inside the checker (checker-lags-integrator class,
  #4504 precedent) with bit-exactness against the persisted-coefficient
  route, and the strict-gate residual-exactly-zero property (`fd7ef13`,
  #3297) must survive it. That is a multi-day instrumentation project,
  not a drive-by; this block records the verdict and the acceptance
  shape.
- **Un-quarantine implication (criterion 3 of this entry):** once the
  mirror carries the load delta through a rebuilt surface path,
  `test_different_zones_respond_differently_to_targeted_gain`'s
  structural zero-leverage cause is removed; the gauge-build quarantine
  on that test can be re-evaluated then (its assertion may still fail
  on trajectory grounds — separate gaps, criteria 1-2).
- **Interim product surface:** unchanged — the integrated-flux
  `EnergyBalanceValidator` (#1344) remains the documented diagnostic;
  this stays in the §LIMIT-19 / #3103 family pending that investigation.

### 2026-10-26 (loop round 103) — rebuilt: the R77 composite arm carries the load delta

**What landed.** The R77 composite-coefficient arm's persisted record gains a fourth
entry per sub-step: the load-linear coefficient of `b` (`db/d(load_w)`), computed
engine-side on the same branch `b` itself comes from — the R94 per-class partition
(`s_c·w` share of `phi_st_int = load·rad_frac`), the R88 lumped share (`s`), their
flag-gated archaeology variants, and the R90 midpoint-ramp scaling when the harness
flag is on. The checker's replay rebuilds `b` at the caller's load
(`b_eff = b + bl·Δ` when `Δ ≠ 0`, `b` verbatim at `Δ = 0`), so the composite mirror
now routes the artificial-gain probe's load delta through the engine's own
per-class φ_st share algebra and sub-step affine composition
(checker-lags-integrator, #4504 class).

**Scope, stated honestly.** This is a single-site drive rebuild: the mass-drive
site of the committed composite operator. The air-node feedback term (a perturbed
load also moves `t_i_sub` through the air ODE, feeding back into the mass advance)
is not mirrored — the same single-site scope as the 9R4C arm, which evaluates its
rebuilt `t_s` at the committed air state. The per-class t_s-first mirror (round 73)
and the legacy lumped identity (gauge-proxy states without an R77 record) remain
load-blind; they stay owned by the §LIMIT-19/#1344 family.

**Evidence.**
- TDD red-first: `test_r77_composite_gain_probe_leverages_5r1c_state` (new, explicit
  `ThermalSelector::legacy()` so both builds take the 5R1C path) failed pre-rebuild
  with probe response exactly 0 W (rc=101); post-rebuild the 1 kW probe response is
  8687.616234633788 W with the no-gain residual exactly 0, and the response is
  affine in the delta (2 kW → 17375.232469267445 W). Bit-identical on the gauge
  build (`--features gauge-solver`).
- Python replication (round workspace, `r103_composite_replication.py`): zero-delta
  replay exactly reproduces the committed mass state; the response matches the
  closed form `d_last·Σ_k (Π_{j=k+1..N-1} a_j)·bl_k·Δ`; affine scaling exact on
  synthetic N=12 coefficient sets.
- No-tuning check: full battery at the changed tree — strict 13 PASS / 29 KNOWN-FAIL (tracked) / 0 REGRESSION with unchanged per-case values (600/H 5.566, 970/H 1.594, 950FF/Tmax 39.010 — identical to the pre-change run); output-diff 42/42 bit-identical (all +0.0000%, the two from-zero rows 0 → 0.000000); §6 2/2, §7 emit 2/2 + conformance 1/1, hotloop 6, grid 3, GC10a 3, §11 anchors 5 — all rc=0 (run 2026-10-26T22:45:20Z–22:48:06Z at the changed tree).

**Un-quarantine (criterion 3).** `test_different_zones_respond_differently_to_targeted_gain`'s
gauge-build `cfg_attr` quarantine is lifted. The quarantine's stated cause no longer
holds for this test state: since the #3817 heavyweight-spec promotion, Case 900
auto-promotes to the 9R4C FD network under the Gauge selector too, so both builds
dispatch it to the leveraged 9R4C arm (verified 2026-10-26 at b8b14e06: bit-identical
pass on both builds, zone-0 response −0.10620833931898233 W to the 10 W gain, zones
1–2 residual exactly 0 — their identity is load-free by construction). For
record-bearing 5R1C states the leverage is restored by this round's rebuild.

**β-soak correlation (same night).** The deferred post-merge develop dispatch
(run 40408376011) failed criterion 3 on an unrelated #4640 follow-up: the
`validate-case` stub guard (`test_cli_validate_case_range_returns_not_implemented`,
tests/integration/test_cli.rs:164) still expects "not implemented" after #4640
implemented the diagnostic ranges (the command now exits 0 with a range summary).
Also recorded: the run's state-validation step reported `beta-soak-state.json`
missing, and the workflow auto-posted a "streak resets to 0" verdict to #3286 from
this non-scheduled dispatch. Not physics defects; recorded for the soak verdict.

### 2026-10-27 (loop round 104) — the per-class t_s-first mirror carries the load delta; air-borne arms scoped multi-day

**What landed.** The round-73 per-class boundary record
(`Applied5R1CClassBoundary`) gains `phi_st_zone_load_coef` — the
load-linear coefficient of `phi_st_zone` (`dphi_st_zone/d(load_w)`, W per
W), recorded engine-side on the same branch the value came from: the gains
site `phi_st = load_w·rad_frac + remaining_sol·st_sol_frac`, so the
coefficient is `rad_frac`, written at the applied-boundary record site. The
checker's per-class mirror rebuilds `phi_st_c` at the caller's load
(`phi_st_c = phi_st_zone·w + coef·Δ·w` when Δ ≠ 0, verbatim at Δ = 0), so
the artificial-gain probe's delta routes through the engine's own gains
algebra on the arm's fall-through states (checker-lags-integrator, #4504
class; the round-103 `bload` pattern one record over).

**Evidence.**
- TDD red-first:
  `test_perclass_ts_mirror_gain_probe_leverages_5r1c_state` (explicit
  `ThermalSelector::legacy()`, Case 600; the zone's R77 sub-step record
  surgically cleared so the checker falls through to the per-class arm
  with the boundary + t_s records live from the real step) failed
  pre-rebuild with probe response exactly 0 W (rc=101; both balances
  186.66586081708164 — identical). Post-rebuild the 1 kW probe response is
  400.0000000000003 W — the closed form `rad_frac·Δ` = 400 W (`rad_frac`
  0.4, Case 600 `conv_frac` 0.6) — affine in the delta (2 kW →
  799.9999999999995 W). The no-gain residual is bit-identical to the red
  run (186.66586081708164): the Δ = 0 branch consumes the committed value
  verbatim.
- Baseline note (honesty): on the surgical state the absolute per-class
  residual is non-zero (186.67 W) because the committed mass state came
  from the coupled sub-step advance while the per-class mirror reproduces
  the hourly decomposition — the response contract cancels that offset,
  and the Python replication confirms the response is exact under
  arbitrary stored-`t_s` baseline mismatches. On a genuine fall-through
  state (boundary record without a composite record from the same step)
  the aggregation identity matches the advance it mirrors.
- Round-103 contract regression: `test_r77_composite_gain_probe_leverages_5r1c_state`
  still green with unchanged values (0 → 8687.616234633788 W per 1 kW;
  2 kW → 17375.232469267445 W). Bit-identical on the gauge build
  (`--features gauge-solver`): per-class 400.0000000000003 W, R77
  8687.616234633788 W — 2 passed on both builds.
- Python replication (loop scratch, `r104_perclass_replication.py`): the
  mirror algebra's response is exactly `coef·Δ` for arbitrary per-class
  configs and deltas (the sub-step denominator cancels through the surface
  identity; the aggregation identity is load-free — it recomputes from the
  stored per-class `t_s`), zero-delta replay verbatim, affine scaling
  exact.
- No-tuning battery at the changed tree: strict 13 PASS / 29 KNOWN-FAIL
  (tracked) / 0 REGRESSION with unchanged per-case values; output-diff
  42/42 all +0.0000% (from-zero rows 0 → 0.000000); §6 2/2, §7 emit 2/2 +
  conformance 1/1, hotloop, grid, GC10a 3, §11 anchors 5 — all rc=0
  (2026-10-27 r104b gate log, tree = this block's commit).

**Scoping verdict for the remaining blind arms — both multi-day, same root
cause.**
- *Air-node feedback term* (R77 arm; also the 9R4C arm's committed-air
  basis): a perturbed load also moves `t_i_sub` through the air ODE,
  feeding back into the mass advance through the sub-step maps. Mirroring
  it requires rebuilding the sub-step air-node ODE inside the checker (CN
  and forward-Euler forms, per-class variants, vent and inter-zone
  couplings) with a bit-exact zero-delta replay — a multi-day
  instrumentation project, not a drive-by. The committed-air single-site
  convention is deliberate and shared with the 9R4C arm (round-103 scope
  note).
- *Gauge-proxy legacy arm*: the legacy lumped identity's only route for
  the load is `h_tr_3·t_i` (the committed air blend — the identity's
  `phi_st` local is computed from `load_w` but consumed by nothing except
  the debug print). Its load leverage IS the air-node feedback; with
  committed air states the zero leverage is structural, not a recording
  gap. Same multi-day verdict, same project.
- Both stay owned by the §LIMIT-19/#1344 family; the unblocking project is
  the checker air-node mirror (adjacent to the GaugeSolver program's
  un-blockers #3291/#1465/#1462).

**Reachability, stated honestly.** The R77 record write is unconditional
per sub-step under the always-on coupled scheme, so live fall-through
entry states are: per-zone records whose layout predates the round-103
fourth entry (`len % 4 != 0`), zones the mass-advance block skipped, and
surgical/test isolations like this round's contract test. The fix
completes the mirror contract for every state the arm can be entered on.

## 2026-10-28 — loop round 105, increment A: the air-node affine record lands

First bounded increment of the checker air-node mirror project (plan:
`docs/implementation-plans/checker-air-node-mirror-plan.md`, PR #4648).
`MassState::r77_air_substep_coeffs` records, per zone and sub-step, the
five coefficients the coupled air advance actually consumed —
`[D_k, M_k, W_k, den_k, h3od_k]` — pushed in lockstep with the mass
4-tuple, write-only (zero consumers; the Δ=0 replay is untouched by
construction), and only on the default integrator route: the R74
ramped-forcing and R90 filtered-lag archaeology arms have map shapes the
tuple cannot express and keep the committed-air convention.

**The contract caught three plan errors before they became checker
bugs.** The red-first contract
(`test_r77_air_record_composes_engine_counterfactual`) re-runs the engine
with a +1 kW zone-0 load delta (free-float fixture, HVAC fully disabled)
and demands the records compose to the committed response. N=1 passes
under several candidate forms; N=2 discriminates them:
1. The `phi_ia` load coefficient enters the `term_rest_1`-scaled basis
   (#1585 `mul_assign`), so the raw `conv_frac` under-records by
   `term_rest_1` — while the lag route's own `term_rest_1` cancels
   against `lag_input`'s division and must NOT reappear (a bare
   `lag_dload · term_rest_1` over-records ~57×; the first N=12 run
   predicted 287 K where the engine moved 4.9 K).
2. The mass advance reads `t_i_sub` BEFORE the fresh air value is
   swapped into `air_node_t_air`, so both maps consume the previous
   sub-step's state pair — the air→mass route carries `r_air` from the
   prior sub-step (which starts at 0, hour-start committed), not the
   same-step fresh value.
3. Per-class zones couple through per-class `h3_c/den_c`, not the lumped
   `h_tr3_arg_i/denom_m` (N=2 needs gain 0.0504, lumped gives 0.0414) —
   the record's fifth slot carries the consumed composite
   `Σ_c w_cap·h3_c/den_c` itself.

**Exactness evidence:** the composed recurrence equals the engine's
committed response on both states to the 1e-9-relative contract bar at
N=2, N=12 (default) and N=96 (`FLX_R77_COUPLED_N`), Δt_air ≈ 4.9 K and
Δt_m ≈ 0.56 K per 1 kW. Red demonstrated by stubbing the record-write
site (test fails on the record precondition); green on the restored
site. Zero-delta bit-identity is proven by the round battery (record is
write-only), not by this test.

Next: increment B — the checker's R77-arm air-route mirror consuming
this record (recurrence as verified here), then the gauge-proxy legacy
arm re-anchor (increment D).

## 2026-10-28 — loop round 106, increment B: the R77 composite arm consumes the air record

Second bounded increment of the checker air-node mirror project
(`docs/implementation-plans/checker-air-node-mirror-plan.md` §5 B). The
R77 composite arm's replay now advances the plan-§2 staggered response
pair alongside the committed-path composite: per sub-step, the mass
replay carries `h3od_k·r_air_prev·Δ` (the counterfactual air state the
mass block actually reads — the PREVIOUS sub-step's, per the engine data
flow the round-105 contract pinned), and the per-W response pair
`(r_air, r_m)` advances on the same recurrence the record contract
proved against the engine's own counterfactual. The probe response
re-pins from the round-103 direct-share-only value to the composed
closed form; the mass-only replay below the paired branch keeps the
committed-air fall-through for absent/stale-layout air records
(archaeology routes).

**Python replication first** (dev-time evidence, 2000 synthetic configs,
N ∈ {1, 2, 3, 12}): the composed recurrence equals the explicit
two-trajectory counterfactual to 1e-9 relative; the in-loop `t_comp`
form equals `committed_composite + r_m_N·Δ`; the balance response is
`−(r_m_N·Δ·d_last)`; and the Δ = 0 in-loop form is BIT-exact to the
committed replay (the h3od term never enters — explicit zero-delta
branch).

**Red-first contract** (`test_r77_composite_gain_probe_leverages_5r1c_state`,
round-106 re-pin): on the pre-B tree the arm replays the mass record
alone — probe response −8687.616234633788 W (the round-103 direct share,
verbatim) against the composed closed form −20776.574489044437 W, rc=101.
Post-change the measured response is −20776.57448904463 W against the
closed form −20776.574489044437 W (1.9e-10 absolute, ≈9.3e-15 relative —
the 1e-9 contract bar), no-gain residual exactly 0, response exactly affine
(2 kW → 2×, 41553.148978088604 W). The composed per-W responses read off
the records — r_air_N 0.004909700, r_m_N 0.000564264 (d_last
36820.66871596761 W/K) — give Δt_air ≈ 4.91 K and Δt_m ≈ 0.564 K per
1 kW, matching the round-105 record contract's engine counterfactual on
the same fixture family.

**Zero-delta bit-identity** is proven by the round battery at the
changed tree, not by this test: strict 13 PASS / 29 KNOWN-FAIL /
0 REGRESSION with unchanged per-case values, output-diff 42/42 all
+0.0000%, §6/§7 pins + GC10a/§11 anchors all rc=0 (2026-10-28 gate log
at the changed tree). Sibling contracts green on the same build:
`test_r77_air_record_composes_engine_counterfactual` (round 105) and
`test_perclass_ts_mirror_gain_probe_leverages_5r1c_state` (round 104).

**Scope, stated honestly:** this restores leverage on the R77 composite
arm only. The per-class t_s-first mirror's `t_i` route (plan increment
C) and the gauge-proxy legacy identity (increment D) keep their
committed-air fall-throughs; D stays owned by §LIMIT-19/#1344.
Inter-zone cross-responses remain explicitly out of scope (plan §5).

## 2026-10-28 — loop round 107, increment C: the per-class t_i route consumes the air record

Third bounded increment of the checker air-node mirror project
(`docs/implementation-plans/checker-air-node-mirror-plan.md` §5 C). The
per-class t_s-first mirror's surface identity now consumes the air-node
record's composed air response under the artificial-gain probe: the
identity's air input moves by `r_air_N·Δ` (plan §2 staggered recurrence,
both updates consuming the previous sub-step's state pair), on top of the
round-104 direct share `rad_frac·Δ`. At Δ = 0 the route never engages
(the #3297 residual-zero property); an absent or stale-layout record keeps
the committed-air fall-through.

**Layout extension (the increment-C contract forced it):** the per-class
arm's entry condition is precisely that the zone's R77 mass record is
absent — the composite arm takes precedence otherwise — so the response
pair must compose from the AIR record alone. The round-105 five-tuple is
completed to seven entries with the mass advance's own consumed `(a, bl)`
(same lockstep push, same values the mass 4-tuple carries, including the
R89 composed form). Python replication first (2000+2000 synthetic
configs): the 7-slot composition equals the explicit two-trajectory
counterfactual to 1e-9 relative and equals the round-106 paired 5+4
composition slot-for-slot; the per-class surface-identity response
closes to exactly `(rad_frac + h_is_zone·r_air_N)·Δ`, affine in Δ, with
the Δ = 0 route record-free.

**Red-first** (`test_perclass_ti_route_consumes_air_record`): on the
pre-C tree the shape precondition fails (`got len 60` — five-slot record,
no mass coefficients to compose from) and the re-pinned r104/r105/r106
contracts fail on the same five-slot layouts; rc=101 with the
`--include-ignored` module run also sweeping the quarantined #3103
known-fail (expected in that mode, not a regression).

**Zero-delta bit-identity** is proven by the round battery at the changed
tree, not by this test: strict 13 PASS / 29 KNOWN-FAIL / 0 REGRESSION
with unchanged per-case values, output-diff 42/42 (40 rows +0.0000% + 2
from-zero rows), §6/§7 pins + GC10a/§11 anchors all rc=0 (2026-10-28 gate
log at the changed tree).

**Scope, stated honestly:** this restores leverage on the per-class
t_s-first arm only. The gauge-proxy legacy identity (increment D) keeps
its committed-air fall-through; D stays owned by §LIMIT-19/#1344 and its
composition now has the self-contained record it needs. Inter-zone
cross-responses remain explicitly out of scope (plan §5).

## 2026-10-29 — loop round 108, increment D: the gauge-proxy legacy identity consumes the air record

Fourth bounded increment of the checker air-node mirror project
(`docs/implementation-plans/checker-air-node-mirror-plan.md` §5 D). The
legacy (gauge-proxy) lumped identity's only load route `h_tr_3·t_i`
consumes the air-node record: under the artificial-gain probe the
identity's committed air blend responds with the composed air response
`r_air_N·Δ` (plan §2 staggered recurrence), so the residual response
closes to `−h_tr_3·r_air_N·Δ` — α-independent, because the blend weights
sum to 1 (both inputs are committed views of the same air node). At Δ = 0
the route never engages (the #3297 residual-zero property); an absent or
stale-layout record keeps the committed-air fall-through. The re-anchor
lands on BOTH legacy sites (the identical-construction fallback and the
no-per-class-state else branch) through one shared helper
(`composed_air_route_shift`), with the arm-level load delta hoisted so all
three 5R1C consumers share one computation.

**Python replication first** (dev-time evidence, 4000 synthetic configs,
0 failures): the re-anchored response equals the explicit two-trajectory
counterfactual to 8.3e-12 relative, α-independent (five random blend
weights per config, same spread), Δ = 0 residual verbatim, affine 2×
exact, absent-record fall-through response exactly 0.

**Red-first** (`test_legacy_identity_air_route_consumes_air_record`): on
the pre-D tree rc=101 with the probe response exactly **0 W** (both
balances 35.19047278083417 — the round-104 blind verdict verbatim)
against the composed closed form −210.5495525168603 W.

**Post-change:** measured **−210.54955251686079 W** vs composed closed
form **−210.5495525168603 W** per 1 kW (≈2.3e-15 relative; h_tr_3
42.88440451366867 W/K, r_air_N 0.004909700 per W — the same composed
response rounds 105–107 consumed), affine (2 kW → −421.09910503372157 W),
and the fall-through discriminator returns the pre-D blind value **0 W**
verbatim when the air record is cleared. Sibling contracts byte-identical
on the same build: r104 direct share 400.0000000000003 W, r106
−20776.57448904463 W, r107 1213.0462878569015 W.

**Blind-arm list status.** The round-104 scoping verdict's two blind arms
are both closed where the record exists: the R77 composite arm (round
106) and the gauge-proxy legacy identity (this increment). On gauge-route
states the identity still keeps the committed-air fall-through — the
gauge dispatch's mass-state proxy persists no sub-step record (the gauge
solver computes no air-node response to record), so the fall-through
there is the documented absent-record convention, and the gauge-route
response itself stays owned by §LIMIT-19/#1344. Inter-zone
cross-responses remain explicitly out of scope (plan §5).

## 2026-11-05 — loop round 113: the inter-zone cross-response (plan §5
## increment E; the record's eighth slot)

The plan §5 out-of-scope list's first item — "inter-zone cross-responses
(Case 960 family — a single-zone probe's neighbor-side response needs a
per-zone response vector, not a scalar)" — scoped and landed on the
NEIGHBOR side. The engine's iz block adds
`q_iz_net[i] = h_tr_iz[i]·(Σ_j T[j] − N·T[i])` to `phi_ia` from the
COMMITTED hour-start air states, and `num_rest` is mul_assign'd by
`term_rest_1` (#1585), so a neighbor's committed state enters the num
with slope exactly `ramp_f·term_rest_1[i]·h_tr_iz[i]` — never recorded
before this round, hence invisible to every probe arm.

**Landed:** the air-node record's EIGHTH entry per sub-step,
`C = (1−D)·ramp_f·term_rest_1[i]·h_tr_iz[i]/den`, pushed in the same
lockstep (gated by the iz block's own `num_zones > 1` and
`h_tr_iz[i] > 0.0` conditions — a single-zone model's slot is exactly
0.0), plus the checker's `composed_cross_response_vector` composing
`r_air' = D·r_air + M·r_m + W·Δ_j + C·(Σ_i prev_i − N·prev_j)` per
sub-step over the previously committed air-response vector with zero
hour-boundary seeds.

**Python replication first** (`/tmp/r113_replication.py`, dev-time
evidence, all scenarios PASS): the neighbor's first-crossed-hour
response composes exactly (≤1.5e-11 relative across symmetric,
asymmetric-iz, reversed-probe, N=1 and N=96 fixtures); the hour-1 self
response stays exact (1.4e-13, the round-105 property); the
iz-source-only perturbation reproduces the full numeric neighbor
response (the other channels are provably inert for the neighbor); and
the boundary is quantified — the probe zone's own hour-2 response
misses by the carry channels (records-only −0.0022 K/W vs numeric
+0.209 K/W on the synthetic twin), closing to 8.7e-14 with the
hour-boundary seeds + the R71 ramp-source response channel added.

**Red-first** (round-105 stub precedent): the C-slot push stubbed →
contracts {1,2,3,4} fail for the recorded reasons (stale 7-wide layout
"got 84"; response exactly 0) with contract 5 green, rc=101; restored →
green ×3. **Teeth ×3:** the iz-sign flip fails exactly {3} (prediction
{2,3} missed — within a probe hour the iz term is probe-independent, so
the sign only manifests at the crossed hour; disclosed), the
ramp-factor drop fails exactly {1,3}.

**Post-change:** the fixture's flagship contract asserts the composed
hour-2 neighbor response against the engine counterfactual
(single-hour probe, load reverted) at ≤1e-9 relative; the probe zone's
own hour-2 divergence is pinned, not tuned (composed −0.056 K vs
measured +0.275 K per 1 kW — the bounded-later carry channels, named in
the plan §5 amendment: the R71 ramp-source persistence response, the
hour-boundary air/mass state seeds, the R90 lag state on non-record
routes). Battery bit-identity proven at the changed tree — the slot is
write-only-pooled and every Δ=0 path is untouched, so no re-record is
owed. The held r109–r112 suites pin the seven-entry layout in their
helpers; updating them is a disclosed landing-time rebase step for the
merge train (#4672 → #4673 → #4674 → #4675, then this PR).

## Round 115 — increment F, the self-response carry channels (2026-11-07)

The r113 re-bounded remainder landed: the air record's NINTH per-sub-step
entry records the R71 ramp-source persistence response
(`NP_k = (1−relax)·(1−f_k)·num_prev_load_slope/den`, exactly 0.0 on seed
hours), fed by a new per-zone freeze-site field
(`MassState::r71_num_prev_load_slope = conv_frac·term_rest_1` — the frozen
`num_tm + num_rest`'s whole load slope on the recorded route: the lag route
is post-ramp and the FLX_R90_PHIST_RAMP prev-ramp branch is off where the
record exists). The checker gains `composed_cross_response_vector_seeded`
(seed-taking API: hour-boundary air/mass seeds + the previous hour's probe
delta); the zero-seed round-113 form delegates bit-for-bit and its
discrimination gap stays pinned. Python replication: seeds+np closes to
5.0e-14 air / 1.8e-12 mass on the synthetic twin. The real fixture's seeded
pair matches the engine counterfactual ≤1e-9 relative on both zones and
both state pairs (contract 3 re-pin, duck note N1 honored); contract 6
closes the continuation with zero iz coupling (seeds + persistence alone).
Red-first stub → {1,3,4,6} rc=101; teeth: (1−f) drop → {1,3,6}, un-gated
routing → {3,6}. Battery bit-identity at the changed tree — the slot is
write-only and the freeze field is record-only, so no re-record is owed.

## Round 117 — the R90 lag channel's closure (2026-11-08)

The r113 amendment's third carry channel — "the R90 lag state on
non-record routes", the last un-disposed item after round 115 landed the
ramp-source persistence and the hour-boundary seeds — is bounded and
closed as contracts + docs only (no engine change). The channel splits by
route:

- **Recorded route — structurally absent, now a measured contract.** The
  committed lag state CHARGES under a sustained probe (the input
  `h_tr_is·phi_st/term_rest_1·(1−s88)` is load-linear): its response is
  exactly the affine decay chain `rl ← decay·rl + dlag_in·(1−decay)`
  (2.1e-16 / 1.0e-15 across the two hours; zone 0: 0.5413 → 1.0347 lag
  units under 1 kW; the neighbor exactly 0). Nothing consumes it: the
  seeded continuation (nine-slot record + air/mass seeds + NP + the iz
  committed source, no lag term) closes against the engine counterfactual
  at ≤1e-9 on both zones, both state pairs, both hours (2.7e-16 …
  8.2e-14). Any future `solar_lag` consumer on this route breaks the
  contract.
- **Filtered route — live, exactly characterized.** With
  `FLX_R90_NO_LAG_FILTER=0` the num consumes the MIDPOINT lag value, and
  the probe response is the recorded-form recurrence with the W slot
  replaced per sub-step by the lag-channel form (state term
  `term_rest_1·decay_half·rl_unit`, coef_rest recovered from the
  recorded twin's slot with the seed-hour provenance
  `coef_rest = conv·term_rest_1` pinned): closes at ≤1e-9 against the
  engine (5.5e-16 / 3.8e-16), hour 2 ONLY with the committed lag seed
  (no-seed miss 1.02e-3, pinned); the recorded-form composition
  mis-mirrors by 5.1e-3 / 3.7e-3 — the round-116 suppression is
  load-bearing. The route keeps its documented fall-through.

Scope findings: the channel is probed-zone-local (the neighbor's
iz-committed response is route-W-insensitive, 5.3e-15); at `c_air = 0`
the channel vanishes (decay 0 — fixtures need physical air capacitance);
the lag STATE is byte-identical across routes at the same hour (contract
18's parity, re-observed). Python replication `/tmp/r117_replication.py`
(ALL CLAIMS OK) against a 4-run × 2-hour engine probe dump. Teeth ×3 on
the clean tree, each failing exactly its predicted set (fake state
consumer → contract 1's continuation; midpoint corruption → {2,3};
charge clamp → the charge pin + {2,3}). Battery bit-identity at the
changed tree — contracts + a comment only, no re-record owed. The
round-74 "seed steps stay bit-identical" comment is corrected in place
(the R76 refresh postdates it), closing duck note N2-r116.

## Round 118 — the HVAC-masked probe fixtures (2026-11-08)

The §5 out-of-scope list's last un-disposed item after round 117 —
"HVAC-masked zones (probe fixtures use free-float setpoints)" — is bounded
and closed as contracts + docs only (no engine change). Every probe
fixture the plan landed through H pins `free_float = true` plus ±999
setpoints, so the probe surface had no measured statement about zones
under ACTIVE setpoint control. Round 118 bounds it: the controlled
commit (LIMIT-33 Option A, #4314) plus the capacity-clamped ideal loads
(#4241/#2826) create exactly three routes, pinned by
`tests/hvac_masked_probe_contract.rs` (five contracts, per-model branch
preconditions recomputed in-test):

- **Unclamped ideal — the mask.** The committed controlled state
  (`setpoints.temperatures`, the array the checker's probe reads) is the
  setpoint VERBATIM in both probe arms: the controlled response is
  exactly 0 while the free advance composes from the record
  (r_air/r_m, ≤1e-9) and stays BIT-IDENTICAL to a free-float twin. Any
  future probe fixture with active HVAC and headroom therefore measures
  the controlled-state routes, not the free advance — the committed-air
  basis itself stops moving (checker-blind by construction, the
  convention the LIMIT-33 comment documents).
- **Capacity-clamped — pass-through.** The clamp pins Q at ±cap
  (load-independent), so `t_i_act = t_free + Q/den_true` passes the free
  response through UNSCALED and the level closes from the record's final
  den slot (`den_true = den/term_rest_1`, the #2868 form).
- **Deadband — parity.** Demand 0 commits the free state verbatim.

The coupled advance is response- and HVAC-blind on every route: the
committed air AND mass records are bit-identical across the three routes
and the free-float control, and the record persists with the full
nine-slot/twelve-sub-step r113/r115 layout under active control — the
composition contracts from rounds 105–117 compose the same values under
setpoint control. Python replication `/tmp/r118_replicate.py` (ALL CLAIMS
OK 19/19) against a six-run engine dump; teeth ×3 each failing exactly
its predicted set (mask-site commit swap → the mask contract only;
clamped divisor → the clamped route only, pass-through
divisor-independent; clamp removal → the clamped route only, Q
load-dependent). Battery bit-identity at the changed tree — contracts +
docs only, no re-record owed.
