# Checker Air-Node Mirror — Implementation Plan

**Status:** ACTIVE (round 105). Scoping verdict: §LIMIT-22 history, round-104 dated block (both blind arms air-borne, multi-day). Ownership: §LIMIT-19 / #1344 family; adjacent to GaugeSolver un-blockers #3291/#1465/#1462.
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
r_m'   = a_k·r_m + (h_tr_3,k/d_k)·r_air' + bl_k
```

with `D_k = exp(−dt_sub/τ_air)` (or the consumed branch's relax factor), `M_k = (1−D_k)·coef_tm/den_k` where `coef_tm = derived_h_ms_is_prod[i]` on the refresh route (0 under the R76-no-refresh harness), and `W_k = (1−D_k)·c_load/den_k` with `c_load = ramp_f·conv_frac + h_tr_is·(1 − s_share)·rad_frac` — the ramp factor only on the R71-ramped branch (seed hours drop it), the lag-route term computed on the branch that produced it (T5138-live via `phi_st_int`, lumped via `phi_st`; the R88-off archaeology variant drops `(1 − s_share)`). Branches not covered here keep the committed-air fall-through, documented per increment.

## 3. TDD approach

1. Python replication first, per increment: compose the recurrence over N sub-steps for synthetic single-zone configs (free-float setpoints −50/100 so HVAC cannot mask the air node) and assert the composed response equals the closed form to ≤1e-9 relative. The committed contract test embeds the same first-principles algebra; the scratch replication script is dev-time evidence only.
2. Red-first per increment: the new contract test must fail on the pre-increment tree (`rc=101`) for the recorded reason (record absent / response exactly 0 W / stale closed form), then green after the increment. Red `got` values are asserted against the prediction, per loop discipline.
3. Battery per increment at the changed tree: strict + output-diff + §6/§7 pins + GC10a/§11 anchors; §LIMIT-22 history dated block per increment; `_doc_roundNN` provenance on every re-pinned value; delta_config/KNOWN_ISSUES/doc inventory regenerated, never hand-edited.
4. Duck review before merge; `[AI-assisted]` prefix + attribution; serial PRs, each rebased on the previous tip.

## 4. Record design (increment A)

New pooled buffer `MassState::r77_air_substep_coeffs: Option<Vec<Vec<f64>>>` (clone-alongside `r77_mass_substep_coeffs`), written only on the coupled default route, on the same branches the air advance executes. Per zone, five entries per sub-step, layout `[D_0, M_0, W_0, den_0, h_tr_3_0, …]`, guarded `len % 5 == 0`:

- `D_k` — the relax factor the air advance consumed (exact-exponential decay; the `c_air = 0`/degenerate branch records the consumed steady-relax form).
- `M_k` — mass-refresh response coefficient `(1−D_k)·coef_tm/den_k`.
- `W_k` — load response coefficient `(1−D_k)·c_load/den_k` on the consumed ramp/branch combination.
- `den_k` — the ramped denominator actually consumed (pairs with the next entry).
- `h_tr_3,k` — the conductance the mass CN consumed toward `t_i_sub` (the argument at the `crank_nicolson_iso13790` call site; no new algebra).

The existing 4-tuple record and its `len % 4` guard are untouched; the air record pairs with it by sub-step count (equal lengths asserted checker-side when both are present).

## 5. Increment boundaries

- **A — engine record (this round's landable increment):** add the buffer + write sites + the record-shape/value contract test (Python-replicated coefficients on a synthetic coupled step). Battery-inert at Δ=0; no re-record, bit-identity proven by the run.
- **B — R77 composite arm consumes the air record:** the replay gains the response recurrence; the arm's gain-probe contract re-pins from the direct-share-only value (8687.616… W per 1 kW, round-103 record) to the composed closed form. Red-first; provenance to the code sha.
- **C — per-class t_s-first mirror consumes:** the surface identities' `t_i` route picks up the air response; per-class contract values re-pin with the same discipline.
- **D — gauge-proxy legacy arm re-anchor:** its only load route `h_tr_3·t_i` becomes live through the same recurrence; the arm's blind-probe verdict (§LIMIT-22) closes when its response equals the composed closed form.
- **Explicitly out of scope (bounded later, not silently):** inter-zone cross-responses (Case 960 family — a single-zone probe's neighbor-side response needs a per-zone response vector, not a scalar); the R74/R75/R90-filtered archaeology routes (they fall through to the committed-air convention, same as today); the 9R4C arm's committed-air basis (§LIMIT-19 adjacency; interacts with ADR-0011, which stays Alex's call); HVAC-masked zones (probe fixtures use free-float setpoints).

## 6. Verification per increment

Every increment runs the full battery at its own tree (strict 13 PASS / 29 KNOWN-FAIL / 0 REGRESSION expected unchanged at A; B–D move only the probe-response contract values, never the strict per-case numbers), plus the doc gates when `docs/` changes. A verdict of "exact" is claimed only with the Python replication attached and the red-first evidence recorded. The checker-lags-integrator rule (#4504 class) applies to every mirror: same branch, same values, no algebra the engine did not execute.
