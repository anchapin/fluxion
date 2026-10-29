# Checker Air-Node Mirror — Implementation Plan

**Status:** ACTIVE (round 105; increment A landed round 105 — PR #4649; increment B landed round 106 — PR #4653, the R77 arm consumes the air record, response re-pinned to the composed closed form; increment C landed round 107 — the per-class t_s-first mirror's `t_i` route consumes the record via the self-contained seven-entry layout; increment D landed round 108 — the gauge-proxy legacy identity's `h_tr_3·t_i` route consumes the record through the shared `composed_air_route_shift` helper on both legacy sites; the §5 B–D sequence is complete, closing the round-104 scoping verdict's blind-arm list on record-carrying states — the gauge-route response itself stays owned by §LIMIT-19/#1344; the §7 surface-pool blitz queue is next). Scoping verdict: §LIMIT-22 history, round-104 dated block (both blind arms air-borne, multi-day; B restores the composite arm's share via the record). Ownership: §LIMIT-19 / #1344 family; adjacent to GaugeSolver un-blockers #3291/#1465/#1462.
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

The existing 4-tuple record and its `len % 4` guard are untouched; the air record pairs with it by sub-step count (equal lengths asserted checker-side when both are present). [Round 107: the air record is self-contained — the pairing assert remains as a shape check where the mass record is present.]

## 5. Increment boundaries

- **A — engine record (this round's landable increment):** add the buffer + write sites + the record-shape/value contract test (Python-replicated coefficients on a synthetic coupled step). Battery-inert at Δ=0; no re-record, bit-identity proven by the run.
- **B — R77 composite arm consumes the air record:** the replay gains the response recurrence; the arm's gain-probe contract re-pins from the direct-share-only value (8687.616… W per 1 kW, round-103 record) to the composed closed form. Red-first; provenance to the code sha.
- **C — per-class t_s-first mirror consumes (landed round 107):** the surface identities' `t_i` route picks up the air response — `t_i + r_air_N·Δ` under the probe, committed air verbatim at Δ = 0 and on absent/stale-layout records; per-class contract values re-pinned with the same discipline (the round-104 closed form `rad_frac·Δ` gains `h_is_zone·r_air_N·Δ`).
- **D — gauge-proxy legacy arm re-anchor (landed round 108):** its only load route `h_tr_3·t_i` becomes live through the same recurrence; the arm's blind-probe verdict (§LIMIT-22) closes when its response equals the composed closed form. Landed as `−h_tr_3·r_air_N·Δ` (α-independent — the blend weights sum to 1) on both legacy sites via the shared `composed_air_route_shift` helper; gauge-route proxy states persist no record and keep the documented committed-air fall-through.
- **Explicitly out of scope (bounded later, not silently):** inter-zone cross-responses (Case 960 family — a single-zone probe's neighbor-side response needs a per-zone response vector, not a scalar); the R74/R75/R90-filtered archaeology routes (they fall through to the committed-air convention, same as today); the 9R4C arm's committed-air basis (§LIMIT-19 adjacency; interacts with ADR-0011, which stays Alex's call); HVAC-masked zones (probe fixtures use free-float setpoints).

## 6. Verification per increment

Every increment runs the full battery at its own tree (strict 13 PASS / 29 KNOWN-FAIL / 0 REGRESSION expected unchanged at A; B–D move only the probe-response contract values, never the strict per-case numbers), plus the doc gates when `docs/` changes. A verdict of "exact" is claimed only with the Python replication attached and the red-first evidence recorded. The checker-lags-integrator rule (#4504 class) applies to every mirror: same branch, same values, no algebra the engine did not execute.

## 7. What this project feeds next (blitz queue, FREE_FLOAT_BRAINSTORM_R85.md rank-1 remainder)

The round-105 increment-A contract work sharpened the blitz's remaining slices; the surface-pool slice is next after B–D:

- **Surface pools' remaining contracts** (next blitz slice): pin per class `c` — (i) the sol-air composition (`t_sol_air_c` = t_ext + the α/ε₀ radiative correction from the class's absorbed fluxes), (ii) the transmittance/absorption split per surface orientation into `phi_st_sol`'s per-class fractions (Table 5-138 renormalization 0.9664), (iii) the `t_s` advance contract per class (the r70 surface-node workspace's input→output map: absorbed flux + conduction coupling → next `t_s`), (iv) the longwave exchange term between pools. Each as a red-first contract against Python-replicated closed forms, mirroring the round-85 mass-node suite's shape.
- **Lag filter**: the R90/R93/R94 site algebra is now fully recorded (rounds 103–105); its blitz slice shrinks to pinning the persisted-state transitions (the lag state's own carry contract), which increment B's mirror makes observable for free.
- **Solar partition** (LIMIT-30 0.30/0.30/0.30 split sites) and **vent** contracts stay as originally scoped — unaffected by this project except that the h3od record now documents the exact air↔mass coupling the vent terms ride on.
