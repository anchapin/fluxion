# Checker Air-Node Mirror — Implementation Plan

**Status:** ACTIVE (round 105; increment A landed round 105 — PR #4649; increment B landed round 106 — PR #4653, the R77 arm consumes the air record, response re-pinned to the composed closed form; increment C landed round 107 — the per-class t_s-first mirror's `t_i` route consumes the record via the self-contained seven-entry layout; increment D landed round 108 — the gauge-proxy legacy identity's `h_tr_3·t_i` route consumes the record through the shared `composed_air_route_shift` helper on both legacy sites; the §5 B–D sequence is complete, closing the round-104 scoping verdict's blind-arm list on record-carrying states — the gauge-route response itself stays owned by §LIMIT-19/#1344; the §7 surface-pool blitz queue is next; rounds 110–112 closed the blitz's remaining slices (lag-carry #4673, solar-split #4674, multi-zone vent #4675) and round 113 landed increment E — the inter-zone cross-response, PR #4676; round 115 landed increment F — the sustained-probe self-response continuation (the record's ninth np slot + the seeded checker API); round 116 pinned the record gate's archaeology-route contracts — R74/R90-filtered suppression + consumer fall-throughs, R75 form-invariance precision (the scoping correction of the §5 increment-F parenthetical); round 117 closed the r113 amendment's last carry channel — the R90 lag state on non-record routes (structurally absent on the recorded route, exactly characterized on the filtered one; contracts + docs only)). Scoping verdict: §LIMIT-22 history, round-104 dated block (both blind arms air-borne, multi-day; B restores the composite arm's share via the record); the §5 out-of-scope list's inter-zone item is disposed by the round-113 dated block — the neighbor side landed, the self-response continuation landed round 115 (increment F: the np persistence slot + the seeded API — the R90-lag channel stays structurally absent on the recorded route). Ownership: §LIMIT-19 / #1344 family; adjacent to GaugeSolver un-blockers #3291/#1465/#1462.
**Goal:** restore the artificial-gain probe's leverage on the two arms that lost it to the committed-air convention (the R77 composite arm's air route, the gauge-proxy legacy arm's `h_tr_3·t_i` route), by recording the sub-stepped air-node composite's affine coefficients engine-side and recomputing the probe response checker-side.
**Ceiling:** unchanged — every identity this plan touches stays in the 1e-5 W residual class.
**Method:** TDD with Python replication of each closed form before any code; red-first contract tests; full battery bit-identity at zero delta on every increment; no option plumbing (records are unconditional on the coupled default route, exactly like the round-78 `[a, b, d, bl]` record).

---

## 1. How the sub-stepped air-node composite enters the checker identities

The engine default advances each zone through the R77 coupled single-N scheme (N=12, round-78 flip): per sub-step k, the air node advances first, then the mass CN advance consumes that sub-step's air temperature `t_i_sub` as its interior boundary.

Air advance per sub-step (all sites in `step_5r1c.rs`):

- `num_i = derived_h_ms_is_prod[i]·r76_mass[i] + num_rest[i]` (mass-refresh term at the R76 refresh site; `num_rest` carries `phi_ia` plus exterior/inter-zone corrections).
- R71 midpoint ramp `num_i = num_prev + f·(num_i − num_prev)` when the previous hour seeded (`num_prev != 0.0`), `f = (k+0.5)/N`.
- Night-vent numerator term (load-independent), R77-coupled R75 ramp.
- Solar-lag driver: under the R93/R94 defaults the lag input on a T5138-live zone is `h_tr_is·phi_st_int/term_rest_1` with `phi_st_int = load_w·rad_frac` (internal-gain §C.3 air share), scaled `(1 − s_share)` under the R88 default; the R90 default adds the **unfiltered** input times `term_rest_1` to `num_i`. Lumped zones carry the same load-linear part through `phi_st`.
- Exact-exponential relax: `t_air' = steady + (t_air − steady)·exp(−dt_sub/τ_air)`, `steady = num_i/den_i`, `den_i` ramped like `num_i` (conductances + vent; load-independent), `τ_air = C_air·term_rest_1/den_i`.

Mass advance per sub-step: `t_m' = a_k·t_m + b_k` (CN, or its forward-Euler fallback), where `b_k` contains `h_tr_3·t_i_sub` and the mass-side gains share; the record `[a_k, b_k, d_k, bl_k]` persists per zone (round-78/103/104 work, `MassState::r77_mass_substep_coeffs`).

Checker consumers (`invariant_checker.rs`): the R77 composite arm (round-103 rebuild) replays `t_comp = a·t_comp + (b + bl·Δ)` per sub-step and returns `(t_mass − t_composite)·d_last`; the per-class t_s-first mirror (round-104 rebuild) consumes the committed `rec.t_i` in its surface identities; the legacy lumped identity consumes the committed `t_air` through `h_tr_3·t_i`. All three therefore see **only the direct surface/mass share** of a probe delta. The air-borne share — the response of `t_i_sub` itself, through `phi_ia` (convective share × R71 ramp), through the lag route (`rad_frac` share), and through the `num_tm` mass-refresh feedback — is structurally invisible. That is exactly the round-104 scoping verdict: the arms are not deaf by accident; the record simply does not exist yet.

## 2. The residual-zero property (#3297 constraint)

Issue #3297's lesson is that a single proxy value cannot satisfy the strict-gate Crank–Nicolson identity simultaneously in `t_sol_air`, `t_i` and `t_m_prev`. The mirror never writes a proxy: the engine's committed trajectory is untouched, and the checker rebuilds only the **delta response** at the identity's evaluation site. Four properties define correctness:

- **Z1 — zero-delta bit-exactness.** At `Δ = 0` every record this plan adds is written but consumed by nothing; all three checker arms consume committed states verbatim. The strict battery, output-diff (42/42 from-zero rows) and §6/§7 pins must be bit-identical per increment; a no-re-record claim is made only on that proven bit-identity, never by skipping the run.
- **Z2 — no new committed states.** The air record is a pooled per-step buffer like `r77_mass_substep_coeffs` (taken, cleared, capacity kept); nothing it carries feeds back into any advance.
- **Z3 — exact affine response.** For fixed committed states and fixed schedules, every map in the sub-step network is affine in the load (conductances, vent terms, decay constants and ramp fractions are load-independent), so the hour's composite response to a probe delta is **exactly** affine — not first-order. The checker's recomputed response must equal the composed closed form of the recurrence below, Python-replicated for arbitrary synthetic configs before implementation.
- **Z4 — probe response equals engine share algebra.** The response coefficients are recorded on the same branch that produced the consumed value (round-103 `bl_k` discipline); the checker never re-derives a share the engine did not use.

Response recurrence per sub-step k (default route; `r_lag ≡ 0` because the unfiltered lag input is state-free):

```text
r_air' = D_k·r_air + M_k·r_m + W_k
r_m'   = a_k·r_m + h3od_k·r_air + bl_k
```

with `D_k = exp(−dt_sub/τ_air)` (or the consumed branch's relax factor), `M_k = (1−D_k)·coef_tm/den_k` where `coef_tm = derived_h_ms_is_prod[i]` on the refresh route (0 under the R76-no-refresh harness), and `W_k = (1−D_k)·c_load/den_k` with `c_load = ramp_f·term_rest_1·conv_frac + h_tr_is·(1 − s_share)·rad_frac` — the num works in the `term_rest_1`-scaled basis (#1585: `num_rest` is `mul_assign`'d by `term_rest_1`, so the `phi_ia` load coefficient enters scaled; the lag add's own `term_rest_1` cancels against `lag_input`'s division and must not reappear), the ramp factor only on the R71-ramped branch (seed hours drop it), the lag-route term computed on the branch that produced it (T5138-live via `phi_st_int`, lumped via `phi_st`; the R88-off archaeology variant drops `(1 − s_share)`). `h3od_k` is the mass advance's air-route response coefficient — `Σ_c w_cap·h3_c/den_c` on the per-class branch (each class couples through its own `h3_c = h_am_c + h_vent_i·w` over its own `den_c`; the lumped `h_tr3_arg_i/denom_m` is NOT the per-class route), `h_tr3_arg_i/denom_m` lumped. Both updates consume the PREVIOUS sub-step's state pair: the fresh air value is swapped into `air_node_t_air` only after the mass block reads `t_i_sub` (engine data flow), so the air→mass route carries `r_air` from the prior sub-step, which starts at 0 (hour-start committed). Branches not covered here keep the committed-air fall-through, documented per increment.

## 3. TDD approach

1. Python replication first, per increment: compose the recurrence over N sub-steps for synthetic single-zone configs (free-float setpoints −50/100 so HVAC cannot mask the air node) and assert the composed response equals the closed form to ≤1e-9 relative. The committed contract test embeds the same first-principles algebra; the scratch replication script is dev-time evidence only.
2. Red-first per increment: the new contract test must fail on the pre-increment tree (`rc=101`) for the recorded reason (record absent / response exactly 0 W / stale closed form), then green after the increment. Red `got` values are asserted against the prediction, per loop discipline.
3. Battery per increment at the changed tree: strict + output-diff + §6/§7 pins + GC10a/§11 anchors; §LIMIT-22 history dated block per increment; `_doc_roundNN` provenance on every re-pinned value; delta_config/KNOWN_ISSUES/doc inventory regenerated, never hand-edited.
4. Duck review before merge; `[AI-assisted]` prefix + attribution; serial PRs, each rebased on the previous tip.

## 4. Record design (increment A)

New pooled buffer `MassState::r77_air_substep_coeffs: Option<Vec<Vec<f64>>>` (clone-alongside `r77_mass_substep_coeffs`), written only on the coupled default route, on the same branches the air advance executes. Per zone, SEVEN entries per sub-step, layout `[D_0, M_0, W_0, den_0, h3od_0, a_0, bl_0, …]`, guarded `len % 7 == 0` [round 107 — the round-105 five-tuple was completed by the mass advance's own consumed `(a, bl)` — the same values the mass 4-tuple carries, pushed in the same lockstep — because the consumers that need the response pair are precisely the arms that engage when the mass record is absent (the per-class t_s-first mirror; the gauge-proxy legacy identity); the increment-C contract forced this extension the same way the increment-A contract forced the h3od slot]:

- `D_k` — the relax factor the air advance consumed (exact-exponential decay; the `c_air = 0`/degenerate branch records the consumed steady-relax form).
- `M_k` — mass-refresh response coefficient `(1−D_k)·coef_tm/den_k`.
- `W_k` — load response coefficient `(1−D_k)·c_load/den_k` on the consumed ramp/branch combination.
- `den_k` — the ramped denominator actually consumed (pairs with the next entry).
- `h3od_k` — the mass advance's air-route response coefficient on the consumed branch: `Σ_c w_cap·h3_c/den_c` per-class, `h_tr3_arg_i/denom_m` lumped (see §2 — the plan's original raw-conductance slot was corrected by the increment-A contract, which showed the per-class composite is the only coefficient that composes the engine's true response; the N=1 fixture passes under both forms, the N=2 fixture discriminates them).
- `a_k`, `bl_k` — the mass advance's own consumed affine coefficients (round 107; identical to the mass 4-tuple's first/fourth entries for the same sub-step, including the R89 composed form), so the §2 recurrence composes from this record alone.

The existing 4-tuple record and its `len % 4` guard are untouched; the air record pairs with it by sub-step count (equal lengths asserted checker-side when both are present). [Round 107: the air record is self-contained — the pairing assert remains as a shape check where the mass record is present.] [Round 113 (increment E): the record's EIGHTH entry per sub-step — `C`, the inter-zone cross-response coefficient `(1−D)·ramp_f·term_rest_1[i]·h_tr_iz[i]/den`, pushed in the same lockstep as the seven-entry tuple. It records the num-side slope a NEIGHBOR's committed hour-start air temperature gains through the iz block (`q_iz_net[i] = h_tr_iz[i]·(Σ_j T[j] − N·T[i])` added to `phi_ia`, then term_rest_1-scaled by the #1585 mul_assign), gated by the same conditions the block applies (`num_zones > 1`, `h_tr_iz[i] > 0.0` — zero otherwise, so a single-zone model's slot is exactly 0.0), with the R71 ramp factor riding on it (the iz term is inside `num_rest`, one of the two pre-ramp num components) and the same `(1−D)/den` normalization as the `M`/`W` slots. The `len % 7 == 0` guard and every `7·k` stride become `8·k`; a seven-entry buffer is stale layout and keeps the documented committed-air fall-through. The held r109–r112 suites pin the seven-entry layout in their helpers and assert it at landing rebase — one bounded landing-time update disclosed in the PR.]

[Round 115 (increment F): the record's NINTH entry per sub-step — `NP`, the
ramp-source persistence response `(1−relax)·(1−f_k)·num_prev_load_slope/den`,
pushed in the same lockstep, exactly 0.0 on seed hours (`num_prev == 0`: the
R71 interpolation consumes no `num_prev` term there). `num_prev_load_slope`
is a new per-zone persisted field (`MassState::r71_num_prev_load_slope`),
written at the R71 freeze site next to the drivers:
`conv_frac·term_rest_1[i]` — the freeze (`num_tm + num_rest`, both hour-start
arrays) carries the hour's load ONLY through `num_rest`'s convective share:
the lag route is post-ramp and the `FLX_R90_PHIST_RAMP` prev-ramp branch is
off on the recorded route, so this is the freeze's whole load slope (Z4).
The `len % 8 == 0` guard and every `8·k` stride become `9·k`; an
eight-entry buffer is stale layout and keeps the documented committed-air
fall-through.]

## 5. Increment boundaries

- **A — engine record (this round's landable increment):** add the buffer + write sites + the record-shape/value contract test (Python-replicated coefficients on a synthetic coupled step). Battery-inert at Δ=0; no re-record, bit-identity proven by the run.
- **B — R77 composite arm consumes the air record:** the replay gains the response recurrence; the arm's gain-probe contract re-pins from the direct-share-only value (8687.616… W per 1 kW, round-103 record) to the composed closed form. Red-first; provenance to the code sha.
- **C — per-class t_s-first mirror consumes (landed round 107):** the surface identities' `t_i` route picks up the air response — `t_i + r_air_N·Δ` under the probe, committed air verbatim at Δ = 0 and on absent/stale-layout records; per-class contract values re-pinned with the same discipline (the round-104 closed form `rad_frac·Δ` gains `h_is_zone·r_air_N·Δ`).
- **D — gauge-proxy legacy arm re-anchor (landed round 108):** its only load route `h_tr_3·t_i` becomes live through the same recurrence; the arm's blind-probe verdict (§LIMIT-22) closes when its response equals the composed closed form. Landed as `−h_tr_3·r_air_N·Δ` (α-independent — the blend weights sum to 1) on both legacy sites via the shared `composed_air_route_shift` helper; gauge-route proxy states persist no record and keep the documented committed-air fall-through.
- **E — inter-zone cross-response (landed round 113):** the plan §7 blitz's closure left this §5 item as the named next slice; the round-113 scoping pass bounded the NEIGHBOR side and landed it: the record's eighth entry `C` (§4 amendment) + the checker's `composed_cross_response_vector`, composing `r_air' = D·r_air + M·r_m + W·Δ_j + C·(Σ_i prev_i − N·prev_j)` per sub-step over the previously committed air-response vector with zero hour-boundary seeds. Exact for a zone whose own committed states did not respond in the previous hour — the neighbor of a first probe hour — because its own load, lag state and mass are probe-independent there (each inertness pinned byte-exact by `tests/multizone_cross_response_contract.rs` contract 5; the flagship contract asserts the engine-counterfactual agreement at ≤1e-9 relative). Red-first evidence per the round-105 precedent: the C-slot push stubbed → contracts {1,2,3,4} fail for the recorded reasons (stale 7-wide layout "got 84"; response exactly 0) with contract 5 green, rc=101; restored → green ×3. Teeth ×3: the iz-sign flip fails exactly {3} (prediction {2,3} — C2 stays green because within a probe hour the iz term is probe-independent; disclosed), the ramp-factor drop fails exactly {1,3}, the stub round above. Battery bit-identity proven at the changed tree (write-only slot, Δ=0 paths untouched) — test-only, no re-record owed.
- **F — sustained-probe self-response continuation (landed round 115):** the
  r113 amendment's re-bounded remainder. The engine records the persistence
  response (the ninth slot `NP`, §4 amendment, fed by the new
  `r71_num_prev_load_slope` freeze field), and the checker gains
  `composed_cross_response_vector_seeded` — the seed-taking API extension
  (`seed_air` / `seed_mass` = the committed hour-boundary shifts, i.e. the
  previous hour's own composed end pair; `prev_load_delta_w` = the previous
  hour's probe that the frozen `num_prev` carries). The zero-seed round-113
  form keeps its semantics bit-for-bit (it delegates with zero seeds and no
  persistence delta; contract 3 pins both its remaining discrimination gap
  and the seeded match — the r113 duck note N1 tripwire re-pinned in the
  same landing). Python replication `/tmp/r115_replication.py` (dev-time
  evidence): the seeds+np composition closes to 5.0e-14 air / 1.8e-12 mass
  on the synthetic two-zone twin (the np channel carries 2.19 of the 5.00
  K/W response; seeds-only misses at 2.81); the real fixture's seeded pair
  matches the engine counterfactual ≤1e-9 relative on both zones and both
  state pairs (contract 3), and with zero iz coupling the seeds+np pair
  alone closes it (contract 6). Red-first: the slot stubbed to 0.0 →
  contracts {1, 3, 4, 6} fail for the recorded reasons (NP closed-form got
  0.0; seeded composed 0.1190 vs measured 0.2745 K — the seeds-only value),
  {2, 5} green, rc=101; restored → green ×3. Teeth ×3, each failing
  exactly its prediction: the `(1−f)` weight dropped → {1, 3, 6}; the
  persistence delta routed un-gated to every zone → {3, 6}; the stub above
  → {1, 3, 4, 6}. Still out of scope: the archaeology-route record-gate
  contracts — landed round 116 as increment G below; the R90 lag channel
  on non-record routes (structurally absent where the record exists —
  `r77_air_record` requires `r90_no_lag_filter`); the 9R4C arm's
  committed-air basis (§LIMIT-19 adjacency; interacts with ADR-0011, which
  stays Alex's call); HVAC-masked zones (probe fixtures use free-float
  setpoints).
- **G — the record gate's route contracts (landed round 116):** the §5
  remainder after F — the archaeology-route item — landed as contracts +
  docs only (no engine change; held on the ASHRAE weather wall as the
  merge train's seventh car). `tests/air_record_route_contract.rs` (new
  binary, ENV_LOCK convention, env restored by a drop guard) pins the
  round-105 dated gate `r77_air_record = r77_coupled && !r74_ramp_exact
  && r90_no_lag_filter` and its consumers' documented committed-air
  fall-through: (1) the default-route nine-slot record premise; (2) R74
  suppresses the record (the mass record's gate is `r77_coupled` alone
  and stays untouched) and all three consumers fall through — the
  composite arm to the mass-only record composition (the round-106
  expected-value pattern, in-test replay of the arm's own arithmetic),
  the per-class t_i route to `rad_frac·Δ`, the legacy identity to exactly
  0 W — plus the presence-based truthiness pin (`"0"` still selects the
  route; the ad-hoc `.is_ok()` convention, the round-111 precedent); (3)
  the R90-filtered route likewise, plus the canonical env_bool token set
  (`0`/`false`/`no`/`off`/empty falsy; explicit truthy and a typo token
  behave as unset — fail-closed to the record-carrying default); (4) R75
  stays record-carrying and exact — **the scoping precision**: the
  increment-F parenthetical ("R74/R75/R90-filtered … fall through") was
  loose about R75; the gate's own round-105 comment already states R75
  vent-ramp is form-invariant ("load-independent constant") and the
  interpolation is the coupled route's own default (`r77_coupled ||
  r75_vent_ramp`), so the flag is inert where the record lives — pinned
  by hour-by-hour byte-identity of the records flag on/off (the
  real-amplitude hour: nv_prev −4690 W → nv_cur +2345 W across the
  outdoor step) and by the seeded composition matching the engine
  counterfactual ≤1e-9 relative with the vent live (all four entries,
  both zones, sustained probe). Python replication
  `/tmp/r116_replication.py` (dev-time, ALL CLAIMS OK): the vent term is
  probe-independent and additive — the composition closes to 5.3e-15
  (hour 1) / 6.4e-14 (hour 2) on the twin and the recorded slots are
  identical across vent shapes; the exclusions are load-bearing —
  feeding the default-form slots under the R74 exact-exponential
  mis-mirrors the ramped-hour response by a 0.80 relative gap (zone 0),
  the R90-filtered lag state by 0.031. Measured scoping finding recorded
  in-contract: the R74 site freezes the hour-start mass refresh
  (`num_tm_ref + num_rest_ref`) and integrates the ramp exactly from f0
  to f1, so R74 differs from the default coupled route even on seed
  hours (the round-74 comment's "seed steps stay bit-identical" predates
  the round-76 refresh) — no cross-route state identity is claimed
  anywhere. Teeth ×3, each failing exactly its own contract:
  `!r74_ramp_exact` dropped → test 2 at the gate assert;
  `r90_no_lag_filter` dropped → test 3 in the token set;
  `&& !r75_vent_ramp` added → test 4 at the hour-0 byte-identity;
  restored → green ×3. Measurement note: the fall-through pins measure
  DIFFERENTIALLY per arm — the per-class mirror's aggregation identity
  carries a structural residual on the route models (the class rebuild
  does not reproduce the lumped committed mass) that appears in both of
  the arm's runs and cancels only against its own arm's baseline; the
  no-gain baseline is re-measured after each surgery (the round-107
  contract's own order). Battery at the changed tree (CI-verbatim recipe,
  detached, DONE_0): weather provenance rc=0; strict gate **13 PASS / 29
  KNOWN-FAIL / 0 REGRESSION** unchanged; **output-diff 42/42 all
  +0.0000% — bit-identical, no re-record owed**; fabric self-test +
  selector-parity verdict rc=0 (all deltas 0.00%); §6 (2 passed), §7-GSR
  (2), §7-conformance (1), GC10a (5), fluxion-core lib **476 passed**
  (the QAS + GC anchors' home).
  Still out of scope after G: the R90 lag channel on non-record routes;
  the 9R4C arm's committed-air basis (§LIMIT-19/ADR-0011, Alex's);
  HVAC-masked zones.
- **H — the R90 lag channel's closure pins (landed round 117):** the §5
  remainder named above — the r113 amendment's last carry channel —
  bounded and closed as contracts + docs only (no engine change; held on
  the ASHRAE weather wall in the merge train). The channel splits by
  route. On the RECORDED route the lag state is consumed by nothing (the
  num takes the unfiltered input; the mass advance never reads it), so
  the round-113 amendment's "structurally absent on the recorded route"
  becomes a measured contract: `tests/r90_lag_channel_contract.rs`
  contract 1 pins the committed lag-state charge EXACTLY — the affine
  decay chain `rl ← decay·rl + dlag_in·(1−decay)` with
  `dlag_in = h_tr_is·rad_frac·(1−s88)/term_rest_1` (2.1e-16 hour 1,
  1.0e-15 hour 2 on the 2-zone lumped fixture; zone 0 charging 0.5413
  → 1.0347 lag units under a sustained 1 kW probe, the neighbor exactly
  0) — while the seeded continuation (slots + air/mass seeds + NP + the
  iz committed source, NO lag term) still closes against the engine
  counterfactual at ≤1e-9 (2.7e-16 … 8.2e-14 across both zones, both
  state pairs, both hours). Any future `solar_lag` consumer on this
  route breaks the closure. On the FILTERED route (record suppressed)
  the channel is live and now exactly characterized: the response is the
  recorded-form recurrence with the W slot replaced per sub-step by
  `(1−D)·(coef_rest + lag_dload·(1−decay_half) +
  term_rest_1·decay_half·rl_unit)/den`, coef_rest recovered from the
  recorded twin's slot with the seed-hour provenance pinned
  (`coef_rest = conv·term_rest_1`), closing against the engine at
  ≤1e-9 (5.5e-16 hour 1, 3.8e-16 hour 2) — hour 2 closes ONLY with the
  committed lag seed (the no-seed form misses by 1.02e-3 relative,
  pinned), and the recorded-form composition mis-mirrors the filtered
  engine by 5.1e-3 / 3.7e-3 (the round-116 exclusion is load-bearing,
  both hours pinned). Scope findings recorded in-suite: the channel is
  probed-zone-local (the neighbor's iz-committed response is
  route-W-insensitive — the iz term enters only from hour-start
  committed states, measured 5.3e-15); at `c_air = 0` the channel
  vanishes entirely (decay 0, `lag_mid → input` — the fixture carries
  physical air capacitance V·ρ·cp); the lag STATE itself is
  byte-identical across routes at the same hour (contract 18's parity,
  re-observed). Python replication `/tmp/r117_replication.py` (dev-time
  evidence, ALL CLAIMS OK) verified against the engine dump (a 4-run
  × 2-hour probe harness). Characterization, not red-first — no engine
  change (the r109–r116 contracts precedent); teeth ×3 on the clean
  tree, each failing exactly its predicted set: a fake `+ lag_old_r77`
  consumer added to the default num → contract 1 fails at the hour-1
  continuation (8.9e-4; the within-hour charge makes the state
  consumable from sub-step 1 — predicted hour-2, disclosed), {2,3}
  green; the midpoint form corrupted (decay_half → decay on the
  lag_old term) → {2,3} fail, 1 green; the state charge clamped (the
  input term dropped from the advance) → {1 at the charge pin
  (5.41e-1 vs 0.0), 2, 3} fail — proving the charge pin observes the
  real charge and the continuation pin observes consumption. Rider
  landed: the round-74 note's "seed steps stay bit-identical" corrected
  in place (the R76 refresh postdates it; the round-116 measurement
  cited), closing duck note N2-r116. Still out of scope after H: the
  9R4C arm's committed-air basis (§LIMIT-19/ADR-0011, Alex's);
  HVAC-masked zones.

## 6. Verification per increment

Every increment runs the full battery at its own tree (strict 13 PASS / 29 KNOWN-FAIL / 0 REGRESSION expected unchanged at A; B–D move only the probe-response contract values, never the strict per-case numbers), plus the doc gates when `docs/` changes. A verdict of "exact" is claimed only with the Python replication attached and the red-first evidence recorded. The checker-lags-integrator rule (#4504 class) applies to every mirror: same branch, same values, no algebra the engine did not execute.

## 7. What this project feeds next (blitz queue, FREE_FLOAT_BRAINSTORM_R85.md rank-1 remainder)

The round-105 increment-A contract work sharpened the blitz's remaining slices; the surface-pool slice is next after B–D:

- **Surface pools' remaining contracts** (next blitz slice): pin per class `c` — (i) the sol-air composition (`t_sol_air_c` = t_ext + the α/ε₀ radiative correction from the class's absorbed fluxes), (ii) the transmittance/absorption split per surface orientation into `phi_st_sol`'s per-class fractions (Table 5-138 renormalization 0.9664), (iii) the `t_s` advance contract per class (the r70 surface-node workspace's input→output map: absorbed flux + conduction coupling → next `t_s`), (iv) the longwave exchange term between pools. Each as a red-first contract against Python-replicated closed forms, mirroring the round-85 mass-node suite's shape.
- **Lag filter**: the R90/R93/R94 site algebra is now fully recorded (rounds 103–105); its blitz slice shrinks to pinning the persisted-state transitions (the lag state's own carry contract), which increment B's mirror makes observable for free.
- **Solar partition** (LIMIT-30 0.30/0.30/0.30 split sites) and **vent** contracts stay as originally scoped — unaffected by this project except that the h3od record now documents the exact air↔mass coupling the vent terms ride on.
