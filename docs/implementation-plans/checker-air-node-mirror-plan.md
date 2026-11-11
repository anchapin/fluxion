# Checker Air-Node Mirror — Implementation Plan

**Status:** ACTIVE (round 105; increment A landed round 105 — PR #4649; increment B landed round 106 — PR #4653, the R77 arm consumes the air record, response re-pinned to the composed closed form; increment C landed round 107 — the per-class t_s-first mirror's `t_i` route consumes the record via the self-contained seven-entry layout; increment D landed round 108 — the gauge-proxy legacy identity's `h_tr_3·t_i` route consumes the record through the shared `composed_air_route_shift` helper on both legacy sites; the §5 B–D sequence is complete, closing the round-104 scoping verdict's blind-arm list on record-carrying states — the gauge-route response itself stays owned by §LIMIT-19/#1344; the §7 surface-pool blitz queue is next; rounds 110–112 closed the blitz's remaining slices (lag-carry #4673, solar-split #4674, multi-zone vent #4675) and round 113 landed increment E — the inter-zone cross-response, PR #4676; round 115 landed increment F — the sustained-probe self-response continuation (the record's ninth np slot + the seeded checker API); round 116 pinned the record gate's archaeology-route contracts — R74/R90-filtered suppression + consumer fall-throughs, R75 form-invariance precision (the scoping correction of the §5 increment-F parenthetical); round 117 closed the r113 amendment's last carry channel — the R90 lag state on non-record routes (structurally absent on the recorded route, exactly characterized on the filtered one; contracts + docs only); round 118 disposed the §5 list's last item — the HVAC-masked probe fixtures (the three setpoint-control routes bounded as contracts + docs only, held on the ASHRAE weather wall in the merge train)). Scoping verdict: §LIMIT-22 history, round-104 dated block (both blind arms air-borne, multi-day; B restores the composite arm's share via the record); the §5 out-of-scope list's inter-zone item is disposed by the round-113 dated block — the neighbor side landed, the self-response continuation landed round 115 (increment F: the np persistence slot + the seeded API — the R90-lag channel stays structurally absent on the recorded route) — and the HVAC-masked item is disposed by the round-118 block (increment I). Ownership: §LIMIT-19 / #1344 family; adjacent to GaugeSolver un-blockers #3291/#1465/#1462.
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
- **I — the HVAC-masked probe fixtures (landed round 118):** the §5
  out-of-scope list's last un-disposed item — every probe fixture the plan
  landed through H pins `free_float = true` plus ±999 setpoints (the
  round-103/104 "nothing through HVAC" convention) — bounded and closed as
  contracts + docs only (no engine change; held on the ASHRAE weather wall
  in the merge train as the ninth car). `tests/hvac_masked_probe_contract.rs`
  (new binary, five contracts) pins the three setpoint-control routes the
  LIMIT-33 Option-A commit (#4314, the `t_i_act` site) and the
  capacity-clamped ideal loads (#4241/#2826) create, each with its
  per-model branch precondition recomputed from committed values in-test
  (the #4156 unified conductance + the #4241 residual form). UNCLAMPED
  IDEAL: the committed controlled state is the setpoint VERBATIM in both
  probe arms, so the controlled response is exactly 0 (the mask) while the
  free advance is untouched — the free-air/mass deltas compose from the
  committed nine-slot record (r_air/r_m, ≤1e-9) and are BIT-IDENTICAL to a
  free-float twin under the same load. CAPACITY-CLAMPED: the clamp pins Q
  at ±cap (load-independent), so the residual commit passes the free
  response through UNSCALED (Δctrl = r_air·Δ) and the level closes as
  `t_free + cap/den_true` with `den_true = den/term_rest_1` recomputed
  from the record's final den slot and `derived_term_rest_1` (the #2868
  form); the free advance stays bit-identical to the unclamped route's.
  DEADBAND: demand 0 by the load-sign rule commits the free state
  verbatim (ctrl == free, exactly). Under every route the coupled advance
  — and therefore the committed air AND mass records — is HVAC-blind
  (bit-identical across all three routes and the free-float control), and
  the record persists with the full nine-slot/twelve-sub-step layout (the
  r113/r115 stride), pinning the write's HVAC-independence. Python
  replication `/tmp/r118_replicate.py` (dev-time evidence, ALL CLAIMS OK
  19/19) against a six-run engine dump. Teeth ×3 on the clean tree, each
  failing exactly its predicted set: the unclamped commit returning
  `t_free` → exactly the mask contract; the clamped divisor switched to
  `h_tr_is` (the pre-#2868 form) → exactly the clamped route (its
  pass-through is divisor-independent and stays green); the capacity clamp
  removed → exactly the clamped route (the `t_i_act` site's own
  demand ≥ capacity check keeps the route selected, but Q goes
  load-dependent — `r_air·Δ·(1 − h_coeff/den_true)` — failing the
  pass-through and level pins). Battery bit-identity at the changed tree —
  contracts + docs only, no re-record owed. Still out of scope after I:
  the 9R4C arm's committed-air basis (§LIMIT-19/ADR-0011, Alex's).

## 6. Verification per increment

Every increment runs the full battery at its own tree (strict 13 PASS / 29 KNOWN-FAIL / 0 REGRESSION expected unchanged at A; B–D move only the probe-response contract values, never the strict per-case numbers), plus the doc gates when `docs/` changes. A verdict of "exact" is claimed only with the Python replication attached and the red-first evidence recorded. The checker-lags-integrator rule (#4504 class) applies to every mirror: same branch, same values, no algebra the engine did not execute.

## 7. What this project feeds next (blitz queue, FREE_FLOAT_BRAINSTORM_R85.md rank-1 remainder)

The round-105 increment-A contract work sharpened the blitz's remaining slices; the surface-pool slice is next after B–D:

- **Surface pools' remaining contracts** (next blitz slice): pin per class `c` — (i) the sol-air composition (`t_sol_air_c` = t_ext + the α/ε₀ radiative correction from the class's absorbed fluxes), (ii) the transmittance/absorption split per surface orientation into `phi_st_sol`'s per-class fractions (Table 5-138 renormalization 0.9664), (iii) the `t_s` advance contract per class (the r70 surface-node workspace's input→output map: absorbed flux + conduction coupling → next `t_s`), (iv) the longwave exchange term between pools. Each as a red-first contract against Python-replicated closed forms, mirroring the round-85 mass-node suite's shape. UPDATE 2026-11-10 (loop round 125): item (iii) landed with #4672 (`tests/surface_pool_contract.rs`, the r109 suite: the t_s advance identity + lockstep, the class partition, the sol-air zone aggregate, route precedence, the degenerate/identical-construction arms, and the #2890 interior-LW steady estimates). The class-drive half of item (ii) + the round-94 dual-component partition landed as `tests/surface_pool_class_drive_contract.rs` (loop r125): the Table 5-138 fraction constants (0.642/0.1678/0.1566 over the 0.9664 renormalization) and the §C.3 gains share pinned bit-exactly at the persisted load-coefficient seam — the numeric content round-92's slice 5 explicitly deferred — with both-route teeth; the r92 seam deferral is closed. Still open: (i)'s per-surface sol-air composition closed form, (ii)'s per-orientation transmittance/absorption split upstream of the class fractions, (iv)'s full longwave pool-exchange term (the #2890 view-factor network follow-up).
  - **[Round 126 — LANDED.** `tests/sol_air_composition_contract.rs` (new standalone suite, 11 contracts, Python replication at `/tmp/r126_solair_replication.py` dev-time evidence): item (i)'s per-surface sol-air composition closed form pinned at the Issue #4166 exterior-boundary seam — the per-surface sol-air temperature (ASHRAE ΔR direction per #4224, the `h_c ≥ 1` and `[0, 1]` clamps, the `(T_out − T_sky)⁴` mis-read discriminator), `h_tr_em = A/(R + 1/h_c)`, the ISO 13790 §12.3.2 `(1 + cos tilt)/2` sky view factor, the ASHRAE 140 §5.2.6 `(a, b)` convection pairs with the power-law 10 m → 2.7 m conversion, the `is_windward` wrap semantics, the `aggregate_zone_boundary` composition with its 1e-12 degenerate gate, the fluxion-core sky-temperature derivation, and the persisted per-class exterior-boundary record recomposed from the spec surfaces under a controlled weather hour (zeroed solar premise-pinned by `cls_beam == 0`; north/south wind discrimination arms). Bit-exactness: engine-vs-transcription where the transcription shares the engine's `.powi(4)` intrinsic (a hand-written fold rounds differently on some arms — the r126 night-arm 1-ULP lesson); Python literals at 1e-9 relative on those arms and bit-exact where fold-invariant (zone aggregate, sky temperature); fold-free record values (`cls_g`, `cls_area`, zone `h_tr_em`) bit-exact; sol-air-bearing record values (`cls_t`, zone `t_sol_air`) at 1e-12 relative — the class loop holds an inlined copy of the sol-air arithmetic whose `powi(4)` lowering can differ from the exported method's by 1 ULP. Teeth ×3 with predicted failing sets: ΔR operand swap fails {roof/wall arms, night/clamp arms, zone-aggregate composition, north-wind engine arm}; `f_sky` dropped fails {roof/wall arms (wall), zone-aggregate composition}; §5.2.6 wall-windward `b` drift fails {§5.2.6 pins, power-law pins}. Still open after this slice: (ii)'s per-orientation transmittance/absorption split upstream of the class fractions and (iv)'s full longwave pool-exchange term (the #2890 view-factor network follow-up).]
  - **[Round 129 — LANDED.** `tests/window_solar_split_contract.rs` (new standalone suite, 7 contracts, Python replication at `/tmp/r129_window_split_replication.py` dev-time evidence): item (ii)'s per-orientation transmittance/absorption split pinned at the `src/sim/solar.rs` window seam — the ASHRAE HF Ch. 15 SHGC incidence-ratio table (knots, linear interpolation, both clamps, and the `θ ≤ 0` raw-SHGC branch via the Up face at zenith), the beam/diffuse/ground composition closed forms with the `algebraic_add` total chain (diffuse `×0.9×V_sky` per loop r120, ground `×0.85` per #1271), the night `total ≤ 0` and below-horizon (`θ = 90` — beam exactly 0 with diffuse/ground live) edge routes, the four-orientation split at a fixed hour (south beam 7101.888 W at ratio knot 0.94; north/east/west beam exactly 0.0; east/west bit-equal), the shaded-beam route (`(1−f_sh)` on the beam only — the r119/r120 record; the fins/overhang diffuse composition stays the r128 scoping item, Alex's call), and the ENGINE-TRUTH zone pool: the Case-600 model's `_dhat_calculate_zone_solar_gain` window component recomposed bit-exactly through the same exported path (`cached_solar_position` mid-hour convention, combined-orientation area, the #4269 ground-reflection 0.2 fallback, the per-surface fold, the #4166 irradiance stash) — with the spec-path premise disclosed (Case 600 carries `WindowSpec::double_clear_glass()` SHGC 0.77, not the analytic constructor's #741-corrected 0.787). Pin teeth ×3: ratio-knot drift 0.970→0.969 fails {table contract}; diffuse factor 0.9→0.88 fails {composition, factors, edge routes, orientation split, shaded route} (the engine-truth arm correctly stays green — it recomposes through the same exported path, so a shared-factor mutation moves both sides; the r129 prediction named it red and was corrected on observation); dropping `(1−f_sh)` fails {shaded route} alone. Still open after this slice: (iv)'s full longwave pool-exchange term (the #2890 view-factor network follow-up).]
  - **[Round 109 — LANDED.** `tests/surface_pool_contract.rs` (new standalone suite, 8 contracts, Python replication at `/tmp/r109_ts_replication.py` dev-time evidence): (iii) pinned by the t_s-first BE surface-advance identity + prev-snapshot lockstep (bit-exact against the persisted applied-boundary record), the committed-route precedence pin (the coupled composite replay from the persisted r77 record vs the hourly shadow advance, discriminated), the degenerate `h_air_mass·w` route (committed composite + shadow t_s under the same closed form) and the identical-construction fallback (shadow t_s untouched, committed composite unaffected); (i) pinned by the applied-record class-partition + sol-air h-weighted aggregate identities; (iv) pinned at its wired scope — the steady per-surface estimates the step persists today (`InteriorSurfaceNetwork` view factors stay unit-tested at their own site; the network is deliberately not routed into the 5R1C air balance, Issue #2890-followup); (ii) already pinned by the rounds-91–105 contract family (`mass_advance_contract.rs` contracts 15–22), not duplicated. All 8 pins pass bit-exact against the tip `7ca76496`; a deliberate coefficient mutation fails the flagship identity, so the pins discriminate.]
- **Lag filter**: the R90/R93/R94 site algebra is now fully recorded (rounds 103–105); its blitz slice shrinks to pinning the persisted-state transitions (the lag state's own carry contract), which increment B's mirror makes observable for free.
- **Solar partition** (LIMIT-30 0.30/0.30/0.30 split sites) and **vent** contracts stay as originally scoped — unaffected by this project except that the h3od record now documents the exact air↔mass coupling the vent terms ride on.
  - **[Round 110 — LANDED.** `tests/solar_lag_carry_contract.rs` (new standalone suite, 6 contracts, Python replication at `/tmp/r110_lag_replication.py` dev-time evidence): the carry contract pinned bit-exactly — the persisted `solar_lag` state equals the per-sub-step low-pass recurrence (`lag := lag·decay + input·(1−decay)`, `τ_lag = sqrt(τ_air·τ_mass)`) composed over each hour's own record denominators (the R71 ramp riding in) and seeded from the previous hour's persisted state, on both input branches (T5138-live: the internal-gain §C.3 air share at the per-class share; lumped: the full pool at the zone-level share); the state's fixed point equals the closed form at 1e-9; the release side decays bit-exactly at input 0 with the record `D_k` pairing the lag route's `τ_air` at 1e-12. Pin teeth: decay-exponent and share-scale mutations fail the predicted contracts. The vent record identities (the §7 vent remainder): night vent rides only the air node's consumed denominator (`Δden = term_rest_1·h_ve_night` on every sub-step) and never enters the mass record's `(a, bl, h3od)` — the air↔mass coupling documented by h3od stays vent-free under the R86 air-side-only default — with `h_vent_mass_zone = 0` pinned and the `FLX_R86_VENT_AIR_ONLY=0` archaeology token restoring it on record. Solar-partition disposition: the default-route partition stays owned by the rounds-91–105 contract family (contracts 15–22) and r109's item-(ii) pin; the retired `FLX_R91_SOL2ST` falsy archaeology token (a construction-time flag on the density-driven 0.30·(1−w_solar) split) remains unpinned and is the queue's remaining slice.]
  - **[Round 111 — LANDED.** `tests/sol2st_split_archaeology_contract.rs` (new standalone suite, 5 contracts, Python replication at `/tmp/r111_sol2st_replication.py` dev-time evidence): the retired `FLX_R91_SOL2ST` falsy-token archaeology route pinned — the density-driven split restored bit-exactly as `to_air = 0.30·(1−w_solar)`, `beam = 0.30`, with `w_solar = smoothstep01(κ_wall, 80000, 100000)` exercised at exact-f64 κ fixtures {60000, 90000, 120000} (w_solar exactly {0.0, 0.5, 1.0}, to_air exactly {0.30, 0.15, 0.00}); the real Case 600 spec path (public `fluxion_core::ashrae_case_series::series_600::case_600_baseline`, premise-asserted lightweight κ) carries both arms; truthiness fails CLOSED to the retired default (unset, every truthy token, every unrecognized value read (0.0, 0.0) — only the canonical falsy set {0, false, no, off, empty} restores); construction-time capture pinned (fields fixed at `from_spec`, not re-read per step) with step-time consumption teeth (24 h solar driving diverges 22.4 K air / 7.1 K mass). Pin teeth: three mutations fail exactly their predicted contracts — archaeology-branch erasure (pins 1/2/4/5 red, truthiness green), 0.30→0.25 coefficient drift (same signature), fail-open default=false (pins 2/3/5 red, falsy-arm pins green). With this pin the §7 solar-partition queue is closed at the sweep's honest endpoint: every route, knob and retired token on the partition site carries a record pin. The default route itself stays owned by mass_advance_contract contract 15 (round 92), untouched here.]
- **[Round 112 — LANDED.** `tests/multizone_vent_routing_contract.rs` (new standalone suite, 7 contracts, Python replication at `/tmp/r112_multizone_replication.py` dev-time evidence): the zone-0-only multi-zone vent routing — the residual disclosed rounds 110/111 — pinned at ALL its sites in a hand-rolled 2-zone lumped 5R1C fixture (Case 960 is 9R4C-promoted; asymmetric zones discriminate the vent conductance's `.first()` property read). (a) The den-side routing identity per sub-step: the vented zone's record den shifts by exactly `term_rest_1[0]·h_ve_night` while the neighbor's delta is bit-exactly zero — the vent-active-hour rebuild reproduces the cached `derived_den` bit-exactly (same operand order), with absolute first-sub-step dens pinned bit-exactly. (b) Neighbor record truth: hour 1 fully inert (whole air + mass records byte-identical vent-on/off); hour 2 the air-record coefficient slots stay byte-identical while the mass record's live-air `b` slot moves now that the neighbor's own air route diverges through the coupling. (c) The vented zone's slot-level truth: the load-independent mass slots (`a`, `denom_m`, `b_load`) and the air record's `h3od/a/bl` trio byte-identical vent-on/off; the mass `b` moves only through its own live air route (first differing sub-step, not the first). (d) The coupling discrimination and its one-step lag: the inter-zone load reads the committed hour-start temperatures, so the neighbor responds one hour later while its record coefficients never move; zero coupling leaves it byte-identical for both hours. (e) The cached multi-zone den composition (`Issue #351` arm) bit-exactly against the closed form, with single-zone discriminators. (f) The num-side routing — the discovery of this round: the default route gates the vent into the numerator at TWO sites (the `h_ext·T_out·tr1` shadow and the LIMIT-17 `nv` add, both `i == 0`), and the committed air decomposes as `(C + h_ms_is_prod·tm + (tr1+1)·h_ve_night·T_out)/den_on` at 0.01 K (the mass recurrence composes the committed mass state bit-exactly from the record's own a/b coefficients; −0.0035 K unmodeled lag residual in −16.25 K). (g) Zero-capacity inertness byte-exact (every vent term is proportional to fan capacity). Teeth: full-route move to zone 1 fails contracts {1,2,3,4,6}; conductance drift fails {1} alone (the bit-exact premise constants); cached-den inter-zone drop fails {1,2,4,5,7} — the zero-fan pin catches the cached≠rebuilt divergence (the prediction set for that one contract was incomplete; disclosed); 3× green runs at the clean tree.]
- **Solar partition / vent**: the vent record identities landed round 110 (see the lag-filter LANDED block); the solar-partition remainder is the retired `FLX_R91_SOL2ST` falsy archaeology token — the default-route partition is already pinned by the rounds-91–105 family.
