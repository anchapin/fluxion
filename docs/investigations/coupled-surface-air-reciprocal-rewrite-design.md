# Design: conservation-correct reciprocal surface↔air coupling for the r144 t_s sub-state configuration

> **Doc type:** design document (decision pending — no engine change made or proposed in this PR)
> **Author:** onyx (OS3 physics loop, round 145, item 2)
> **Date:** 2026-11-14
> **Status:** DRAFT-FOR-ALEX — item (A) carries the go/no-go decision; items (B)–(E) carry the remaining calls
> **Engine anchor:** `src/sim/thermal_model_physics/zone_balance.rs` at develop `e95b1725` (post-r144)
> **Governing direction:** fidelity-first (2026-10-09, extended 2026-10-22): known, verifiable, tested physics is the only adoption signal; bands are recorded, never weighted; energy balance is sacrosanct per RULES.md

## Summary

Round 144 measured the env-gated `FLX_R144_TS_SUBSTATE` arm — the D5 hybrid surface-node sub-state, transplanted from the hourly route into the r85 per-class coupled sub-step at `dt_m` — and returned two dated findings: the arm is **refuted** as the free-float Tmax lever (both low-mass FF Tmax rows moved ~9.5 K further below band), and the committed configuration is **not zone-energy-conserving**, because the D5 hybrid couples surface→air one-way (the surface balance consumes the air temperature as a driver; the air node never receives the reciprocal `h_is·(t_s − t_i)` flux). Alex's 2026-11-14 decision (option C) commissioned this document: the design for the conservation-correct reciprocal rewrite, scoped to a concrete plan, with **no engine change** in this round. The default build is untouched and conserving as it stands; the defect exists only in the OFF arm's committed form, and the rewrite is what would make a future armed configuration fit to commit.

## 1. Background — the three routes and where each stands

The interior film/surface coupling exists in three configurations at the r144 tip:

1. **The r85 per-class sub-step (default build, coupled-N default).** Each class advances at `dt_m = dt_sub/mass_sub` (25 s at N=12) by Crank–Nicolson on the mass with the **algebraic-series** interior coupling `h_am_c = h_ms_c·h_is_c/(h_ms_c + h_is_c)` (`zone_balance.rs`, the R85 HARNESS note, ~line 2440). A massless surface node (`C_s → 0`) partitions energy exactly: whatever leaves the air lane for the class arrives at the class mass, conserving by construction. This is the production path.
2. **The hourly D5 route (default since round 71, #3983 §57).** The per-class surface node advances first by backward Euler with the disclosed structural capacitance `C_s = 0.1·cm_c`, consuming `(t_i, tm, phi_s_c)`, and the mass advance consumes the advanced `t_s`. Since round 83 the hourly route's committed **mass** is overridden by the R76/R77 sub-stepped lumped state — the D5 mass advance is records-only, so the one-way form has never been load-bearing on the hourly route either; the committed `t_s` pair is telemetry the engine carries in the r70 pair.
3. **The r144 arm (`FLX_R144_TS_SUBSTATE`, off by default).** Mirrors the D5 hybrid INTO the sub-step at `dt_m`: BE `t_s` consuming `(t_i_sub, tm, phi_s_c)` with the sub-step's own R91/R94 surface-pool partition re-destined to the surface node, then BE mass consuming `h_am_c·t_s_eff`, the r89 inner-step composition generalized to the 4-d affine system `x = [tm, ts_0..2]` committed from the composed slots (22-slot mass-record layout; the air record keeps its 9-slot stride with the arm's composed air gain in `h3od`). Flag off, byte-identity proven on the canonical battery (strict 14/28/0, output-diff 42/42 +0.0000%).

The arm is the first configuration in the program's history where a `t_s`-carrying, **committed** state advances on the one-way coupling — hence the conservation finding.

## 2. The defect, concretely

Per class `c` and sub-step, the arm's surface balance books the film exchange as

```
(C_s/dt_m)·(t_s⁺ − t_s) = φ_s_c − h_ms_c·(t_s⁺ − tm) − h_is_c·(t_s⁺ − t_i_drv)
```

The term `−h_is_c·(t_s⁺ − t_i_drv)` is heat the surface draws from (or dumps to) the zone air. In a conserving network the air node's own advance books the equal-and-opposite `+h_is_c·(t_s⁺ − t_i_drv)`. The arm's air lane keeps its pre-r144 form — the R77 folded air-side term and mass-series coupling, no surface term — so the exchange is booked **once**. Over an hour (N=12 sub-steps, 3 classes) the imbalance integrates into the committed air and surface states; the r144 A/B measured its signature on the conditioned family (600/640/800 C −73 %, 600/800 H +64 %). The same form on the hourly route was benign only because round 83 made that route's mass advance records-only; nothing about the form itself conserves.

## 3. The conservation-correct formulation

### 3.1 Anchoring

ISO 13790 Annex C (the simplified hourly network the engine's §C.3 sites implement) couples its three nodes — air `θ_i`, surface `θ_s`, mass `θ_m` — through `H_tr_is` (surface↔air) and `H_tr_ms` (mass↔surface), with `H_tr_is(θ_s − θ_air)` appearing in the air-node balance and the equal-and-opposite term in the surface-node balance: the standard's network is reciprocal by construction. The engine's own r90 record reads the §C.3 reference surface node as algebraic — no separate surface capacitance — which makes the default r85 algebraic series the `C_s → 0` image of that network, and makes the D5 `C_s = 0.1·cm_c` node a **disclosed structural extension** toward the dynamic order #1522 demands, not a §C.3 deviation. The rewrite below preserves exactly that framing: `C_s → 0` collapses the armed configuration onto the default's conserving algebraic series, and any finite `C_s` is a declared structural assumption carried in one place (`cs_c = 0.1 * cls_cm[c]`, as today).

The §C.3 **site pattern** (the established r88/r94 convention: gains partitions decompose at the §C.3 conductance-share site, `s_c = h_ms_c/(h_ms_c + h_is_c)`) is reused unchanged — the rewrite touches the **coupling terms**, not the partition sites. One limit statement made precise here because §3.4 contracts it: as `C_s → 0`, Scheme 2 reduces to the **§C.3 two-node system** (air + mass, surface algebraic) — which the engine's own r90 record identifies as the default build's **continuous limit**, not step-for-step with the default's folded discrete operator (the r85 series fold is itself an elimination of the algebraic surface, and the two forms agree in the continuous limit, not bit-for-bit at finite dt_m). The contract pins the continuous limit, not discrete identity with the folded form.

### 3.2 Recommended form: Scheme 2 — simultaneous per-zone implicit solve (exact conservation)

Per zone and sub-step, solve the 5-node linear system once per sub-step (air `t_i`, three class surfaces `t_s,0..2`, and the class masses advanced by their own CN lanes against the solved surfaces):

```
air:     (C_ia/dt_m)·(t_i⁺ − t_i) = φ_ia + h_ve·(t_out − t_i⁺) + Σ_c h_is_c·(t_s,c⁺ − t_i⁺) − h_am_eff·(t_i⁺ − tm_eff)
surface: (C_s/dt_m)·(t_s⁺ − t_s) = φ_s_c − h_ms_c·(t_s⁺ − tm⁺) − h_is_c·(t_s⁺ − t_i⁺)
mass:    (cm_c/dt_m)·(tm⁺ − tm)  = φ_m_c − h_ms_c·(tm⁺ − t_s⁺)
```

- **The folded aggregate is replaced, not added to.** Under the arm, the air lane's R77 folded mass-series aggregate (`h_am_c = h_ms_c·h_is_c/(h_ms_c + h_is_c)` with its effective drive — the algebraic-surface elimination) is **subsumed by the explicit chain** above: the air↔surface (`h_is_c`) and surface↔mass (`h_ms_c`) terms ARE that coupling, now carried nodally. Keeping both would double-count the same conductance. This is the r144 re-destination precedent (the drive moved off the folded path when the surface node took it) generalized to the coupling path. The sketch's `h_am_eff·(t_i⁺ − tm_eff)` line only stands for whatever air↔exterior-adjacent terms the air lane already carries (ventilation, envelope) — those are untouched.
- **Exact zone-energy conservation for the modeled pools**: every coupling flux (`h_is_c·(t_s⁺ − t_i⁺)`, `h_ms_c·(tm⁺ − t_s⁺)`) is evaluated at the **same shared solved states in both balances it appears in** — one flux evaluation per pair, booked twice with opposite signs over the same `dt_m`. The pairing principle is independent of the storage-integrator choice: if a lane keeps its CN storage (the r89 composed-mass precedent), the coupling fluxes are still written as a single shared evaluation (all-BE coupling with CN storage terms is the workable combination — CN-paired fluxes with half-old/half-new evaluation would need the same half-form in the partner balance, which is a valid alternative, not a requirement). No evaluation-point mismatch exists by construction, which is what the conservation claim needs.
- **Affine composition survives**: the system is linear in `(t_i, tm, ts_0..2)`; one BE/CN step is an affine map `x⁺ = M·x + s`, and the r89 composed form `x_N = M^N·x + (Σ M^k)·s` generalizes verbatim to the 5-d state. The record channel composes exactly as r144's did, with the matrix grown 4→5 and the air lane's coefficients absorbed into `M` — the checker mirror stays exact by construction.
- **Stability**: implicit in every coupling — no `h·dt/C` explicit criterion anywhere. The r144-measured surface-lane stiffness (`sig ≈ 0.86` per inner step) is carried by the implicit solve rather than fought.
- **Diff shape**: the r144 arm's per-class BE block (`den_s = cs_dt + h_is_c + h_ms_c + g_lw_c`, then the mass BE) becomes a per-zone 5×5 solve; the mass advance composes per class as today; the air lane's coefficients join the matrix instead of remaining a lumped aggregate.

### 3.3 Alternative: Scheme 1 — frozen-exchange Gauss–Seidel (smaller diff, stability caveat)

Keep the r144 sequential shape (BE `t_s` → BE mass → air lane) and add the reciprocal source into the air lane consuming the just-solved `t_s⁺` with the **same frozen `t_i` driver** the surface balance consumed (`t_i` at sub-step entry on both sides). The air↔surface pair then books `±h_is_c·(t_s⁺ − t_i)` — conserved. The mass↔surface pair, however, evaluates its coupling at mismatched states (surface balance at `tm`, mass balance at `t_s⁺` with the series fold), so exact conservation needs the mass balance re-drawn to the direct `−h_ms_c·(tm⁺ − t_s⁺)` pairing — at which point the remaining difference from Scheme 2 is only the air lane's implicitness. **Caveat to measure before adoption**: the air node's capacitance is small; an explicit-in-`t_s⁺` source at `dt_m` must be checked for overshoot on the stiff low-mass cases (the r144 stability hunt already showed this lane is the fragile one). Scheme 2 prices this risk out for a ~1.5× larger matrix-solve diff. The author's recommendation is **Scheme 2**; Scheme 1 is recorded as the fallback if the 5-d solve is judged too invasive for the value.

### 3.4 The `C_s → 0` consistency contract

Whichever scheme: a contract pins the armed configuration's limit — as `cs_c → 0`, the committed sub-step converges to the §C.3 two-node system (air + mass with the algebraic-surface elimination), which the r90 record establishes as the default build's **continuous limit**; the contract checks the limit in the continuous sense (the algebraic elimination recovered from the armed solve at `C_s → 0`, `t_s = (φ_s + h_ms_c·tm + h_is_c·t_i)/(h_ms_c + h_is_c)` substituted through), **not** bit-identity with the default's folded discrete operator at finite `dt_m`. This is the structural guarantee that the armed form contains the §C.3 reference as a limit case rather than diverging from it, and it doubles as the archaeology bridge (falsy `FLX_R144_TS_SUBSTATE` keeps the pre-r144 advance bit-identical, as today — the r144-form gate is the bit-exact bridge, the `C_s` limit is the continuous one).

### 3.5 What the conservation claim covers — and deliberately omits

The sub-step `t_s` balance carries **no inter-pool longwave terms**, unchanged from r144. The routed `InteriorSurfaceNetwork` on the hourly route carries the r130-documented non-conserving floor↔ceiling pair (`F = 1.0` both directions, no reciprocity division — pinned against its exact closed form), and at `dt_m` the surface lane's inertia turns a frozen `T⁴` term into a runaway (measured ±160 K by step 3 hour-frozen; a 1-hour explosion with per-sub-step refresh). That slice is scoped to the **#2890-followup reciprocity fix** and stays out of this rewrite's claim: the rewrite makes the air/surface/mass **pool exchanges** conserving; inter-surface LW redistribution remains a separate, already-documented defect on both routes until the reciprocity fix lands. Recommended ordering: the reciprocity fix first (it is the smaller, better-preconditioned item), then the LW-under-the-arm slice on top of the reciprocal form; but the reciprocal rewrite is **not blocked** by it — the omission is disclosed and bounded, exactly as today.

## 4. Env-gate plan

- Successor gate `FLX_R145_RECIPROCAL` (naming follows the round convention), implying the r144 sub-state base: engaged only with `r91_t5138 && r94_gains_c3 && r77_air_record`, the same recorded-route conjunction the r144 arm gates on, so the checker mirror stays exact where the records exist.
- `FLX_R144_TS_SUBSTATE` retains its meaning (the r144 hybrid form) for archaeology; the falsy-token rules are unchanged (unset or any falsy token = legacy advance bit-identical).
- Contracts, red-first at the unfixed tip per the loop discipline: (a) the conservation contract — the per-zone/sub-step energy ledger closes to round-off under the gate (the per-class FFAUDIT emission, which today lives only in the legacy loop, is part of this item's scope — the r144 record names it as remainder); (b) the `C_s → 0` limit contract of §3.4; (c) the 5-d composed-map mirror contract (checker agreement, the ≤1e-9 relative flagship precedent); teeth ×3 channel-discriminating as usual.
- Python replication first (composed 5-d map vs sequential; the reciprocal channel vs finite difference), the r144 precedent (4.0e-15 / 3.5e-15 over 400 configs).

## 5. Record channel — the dated plan amendment

The air record's 9-slot stride grows: under Scheme 2 the air lane's composed coefficients change (the surface coupling rides in `M`), so the air record gains the reciprocal-coupling slot(s) and the mass record's 22-slot layout is superseded by the 5-d composed layout. Per the r113/r144 lockstep precedent, the layout extends **by a dated plan amendment in the same PR, never by a parallel record**: entries extend in the same push, the checker mirror composes the 5-d recurrence, bit-exact at Δ=0, and the r144-era slots remain readable for the hybrid form via the retained archaeology gate.

## 6. The r144 A/B evidence, weighed honestly — and what the rewrite is worth

The r144 A/B answered the free-float question the arm was built to probe: **the storage-release ordering through a `C_s` surface node is not the FF Tmax lever** — both low-mass FF Tmax rows moved ~9.5 K *further* below band (600FF 49.42 → 39.95; 650FF 48.71 → 39.61), the wrong direction for the §LIMIT-28 undershoot the §Round-80 record bounded at ~1.5 K for this family. Under the pre-2026-10-22 discipline that would have buried the sub-state program. Under the standing direction it is simply the recorded answer: bands are not a decision signal.

What survives, on the physics merits alone:

1. **Conservation is a first-principles property, independent of band direction.** RULES.md makes zone energy balance sacrosanct; the r144 committed form books its film exchange once. A configuration known to be non-conserving is not a fit default-build candidate at any `C_s`, and the §C.3-anchored reciprocal form is the known, verifiable correction — "known, verifiable, tested" is exactly the standing adoption criterion.
2. **The dynamic-order substrate is still live.** #1522's bidirectional peak signature remains undemonstrated on the legacy path; the r67/r70 program identified the per-surface interior network as the carrier of the missing dynamic order. The r144 refutation removed the *ordering* hypothesis; it did not remove the *capacitated surface node* as the structural substrate for that work. The reciprocal form is the only conserving way to carry it.
3. **What the refutation honestly removes**: any band-side urgency, and the claim that flipping the arm would improve §LIMIT-28 rows. If Alex judges the sub-state program not worth continuing, the default build is already conserving and nothing needs to land — the defect lives in an OFF arm, disclosed in code and record. That is decision point (A) below, and "do not land" is a defensible answer under the same fidelity-first rule: the rule obligates *correct physics adopted on its own merits*, not the adoption of any particular arm.

## 7. Re-record scope attached to a default-on flip

Landing the rewrite as default (a flip Alex would direct separately, per the two-step discipline: arm lands env-gated first, byte-identity proven, then the flip decision) carries the full train the r143/r144 precedents define:

- Strict battery re-record `_doc_round145`-class from the flipped default build (canonical DRYCOLD, env unset, CI-verbatim single-`--` invocation; 14 PASS / 28 KNOWN-FAIL / 0 REGRESSION partition re-measured, moves recorded not tuned).
- Output-diff CSV 42/42 old→new provenance; fabric 900/950 rows; §6/§7 pins + CSV re-emits; hotloop/grid goldens re-adjudicated; blind.
- Zone_balance module-size ratchet re-anchor after ALL comment edits (the r138/r144 fixture-catch precedent — measure last).
- Test inventory: `generate_test_inventory.py --verify`, baselines in lockstep, all four inventory surfaces, AGENTS/README citations; doc inventory if docs land in the same train.
- Duck review posted pre-merge; six green required contexts; `[AI-assisted]` prefix + attribution; develop via PR route only.
- The re-record rides the same PR as the flip (the r133 premise-break precedent): a conservation-form change on the committed route is not battery-inert — **no byte-identity claim is available or attempted**; the whole point is that the committed state moves.

## 8. Decision points for Alex

- **(A) Go/no-go on the rewrite itself.** Land `FLX_R145_RECIPROCAL` (Scheme 2) as an env-gated arm with the §4 contract suite, or close the sub-state program with the r144 disclosure standing (the default build is conserving today; the defect stays confined to an OFF arm). The author's recommendation: land the arm — the conservation defect in a committed-capable configuration is a known defect in the program's own substrate, and the fix is the documented, tested kind the loop exists to make; but the band-side motivation is genuinely gone and "close it" is a defensible fidelity-first answer.
- **(B) Scheme choice.** Scheme 2 (5-d simultaneous implicit; recommended; exact conservation, no stability caveat, larger diff) vs Scheme 1 (frozen-exchange sequential; smaller diff; requires the §3.3 stability measurement before adoption).
- **(C) Ordering vs the #2890-followup LW reciprocity fix.** Recommended: reciprocity fix first, LW-under-the-arm second. Both orderings are workable; the rewrite itself is not blocked.
- **(D) The hourly D5 shadow.** Keep records-only (recommended — its mass advance stays discarded, so the one-way form remains benign there) or apply the same reciprocal form on the hourly route (larger blast radius; only worth it if the hourly route ever commits its per-class mass).
- **(E) The flip decision, if the arm lands green.** Default-on carries the §7 train and a moved strict partition; per the standing two-step discipline this is Alex's explicit call at that future point, informed by the arm's own A/B.

## 9. Inputs this document rests on (all read at develop `e95b1725`)

- The r144 arm and its conservation disclosure: `zone_balance.rs:1777-1830` (arm block, disclosure verbatim), the per-class sub-step site (~2440-2530), the arm's per-class balance and the LW omission note (~2610-2700).
- The D5 default flips and their history: `zone_balance.rs:428-437` (round-71 #3983 §57), KNOWN_ISSUES §LIMIT-28 rows for rounds 83/85/88/89b/90 (the records-only shadow, the per-class default, the §C.3 mass share, the full dt_m coverage, the lag-filter retirement).
- The r130 InteriorSurfaceNetwork floor↔ceiling non-conservation and the ±160 K sub-step LW runaway: the arm's LW note (`zone_balance.rs:2623-2642`) and the r144 record.
- The r144 A/B verdict and the ~9.5 K FF Tmax movement: LOOP144.md, KNOWN_ISSUES §LIMIT-28 (2026-11-14 UPDATE), `docs/investigations/limit-28-ashrae-140-free-floating-cohort-600ff.md` §Round-144.
- ISO 13790 §C anchoring as read by the engine's own records: the r90 §C.3 continuous-limit note, the r88/r94 §C.3 site pattern, the r101 internal-gain partition (`ae77fcad`).
