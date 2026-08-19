# ADR-0009: `h_tr_em` Wind-Dependent Per-Step Recompute — Tracking Stub (Issue #3063)

- **Status:** Proposed (tracking stub only — no architectural decision recorded)
- **Date:** 2026-08-17
- **Deciders:** Fluxion maintainers (TBD)
- **Supersedes:** None
- **Depends on:** **ADR-0008** (snapshot-diff verifier for bit-identical baselines from #3070) before any code change; **#3059** (GaugeSolver #1465/#1462 unblocker) for the production-path migration
- **Issue:** [#3063](https://github.com/anchapin/fluxion/issues/3063)
- **Related:** #2891 (origin), #3024 (partial fix PR), #2868 (sister issue: Case 195 surface-balance initialisation), #3059 (5R1C/9R4C architectural rework), #1465 / #1462 (GaugeSolver shadow-mode and validation), #3072 (aggressive-baseline cohort meta-issue)

---

This ADR is the **7-line summary** of the tracking-stub scaffolding shipped with issue #3063:

1. PR #3024 (issue #2891) introduced wind-velocity-dependent `h_se` (the exterior film coefficient used for the sol-air longwave correction) per ASHRAE 140 §5.2.6, but left `h_tr_em` (envelope-to-mass conductance) time-invariant at build-time `1/EXTERIOR_FILM_COEFF_DEFAULT`.
2. The cooling-deadband shift exposed by PR #3024 (Case 195 annual cooling 220 → 758 kWh, target ≤ 50 kWh) is the diagnostic signature of the missing `h_tr_em` per-step recompute; at low wind speeds the wind-dependent `h_se` amplifies the sol-air longwave correction, shifting more hours into the cooling deadband.
3. Per `RULES.md` ("no parameter tuning", "must-never hardcode results"), `AGENTS.md` ("do NOT modify physics code without checking `ARCHITECTURE.md` first"), and the issue's own scope, the actual recomputation is out of scope for a single sub-agent.
4. This ADR ships the **tracking stub only** — the LIMIT-13 entry in `docs/KNOWN_ISSUES.md` and refuses to record an actual implementation decision. *(The snapshot diff verifier `scripts/verify_h_tr_em_regression.py` and pytest harness `scripts/ci/test_verify_h_tr_em_regression.py` that §2 / §3 below originally described were removed 2026-08-19 as orphan — see `.agents/results/result-pm.md`; the §2 and §3 sections are retained as historical context for the verifier design that any future implementation should re-derive.)*
5. The verifier fails closed (exit 2) when the placeholder snapshot set has not been populated, so a future implementer is forced to capture real per-step measurements via **ADR-0008**'s pattern before shipping any code change.
6. `tests/reference_data/zone_balance/strict_energy_gate_baseline.json` is **not modified** by this scaffolding — per AGENTS.md, it must never be raised to hide a regression.
7. `src/sim/`, `src/physics/`, and `fluxion-core/` are **not modified** by this scaffolding — only docs, scripts, and pytest tests.

---

## Executive Summary

This ADR is a **tracking stub**. It records the fact that PR #3024 (issue #2891) closed the wind-dependent `h_se` half of the issue body's request but left the sister `h_tr_em` path time-invariant, and that the resulting 220 kWh → 758 kWh Case 195 annual-cooling shift is the diagnostic signature of the missing per-step recomputation. Per `RULES.md` and `AGENTS.md`, **no physics code is modified in this ADR**. The actual `h_tr_em_zone: Vec<f64>` extension and per-step recompute lands in a future PR that runs the verifier end-to-end and either passes (bit-identical physics preserved) or fails (regression flagged, recompute rejected) on the merits of the diff.

## Context

PR #3024 (issue #2891) introduced wind-velocity-dependent exterior convection in the 5R1C path (ASHRAE 140 §5.2.6). The post-#3024 measurements were:

| Case 195 metric | Pre-#3024 | Post-#3024 | Target | Status |
|-----------------|-----------|------------|--------|--------|
| Annual heating | 7.42 MWh | 6.25 MWh | ≤ 6.30 MWh | ✅ acceptance met |
| Annual cooling | 220 kWh | 758 kWh | ≤ 50 kWh (full) / ≤ 1500 kWh (scoped) | ❌ open |
| Peak heating | — | ≤ 1.05 kW | ≤ 1.05 kW (already met per Issue #2868 / LIMIT-08) | ✅ acceptance met |

The sub-agent report that closed #3024 explicitly noted the gap:

> "h_tr_em (envelope-to-mass conductance) remains time-invariant."

That is the issue body of #3063. The two conductances (exterior film `h_se` and envelope-to-mass `h_tr_em`) share the same `EXTERIOR_FILM_COEFF` source but live in different code paths; PR #3024 fixed the sol-air side, not the envelope-to-mass side.

The current production path consumes `h_tr_em` as a constant per zone (a build-time reciprocal of `EXTERIOR_FILM_COEFF_DEFAULT = 18.3 W/m²·K`). At low wind speeds (V ≈ 1–2 m/s, where `h_c` drops to 4–6 W/m²K) the wind-dependent `h_se` amplifies the sol-air longwave correction, shifting more hours into the cooling deadband (T_zone > 27 °C). The only way to close the 50 kWh Case 195 cooling target is to make `h_tr_em` wind-dependent at every timestep so the wall path aligns with the FD solver and the surface-balance paths.

## Recommended Direction (per Issue #3063)

The issue's "Recommended Direction" calls for the following steps:

1. **Extend `HvacState` or `MassState` with `h_tr_em_zone: Vec<f64>`** (per-zone, per-timestep) — a new field on the existing sub-struct (the same `ThermalModelData<T>` split from #2878 / #3070 already exposes both `HvacState` and `MassState` as per-zone `Vec<f64>` containers, so the extension is mechanically straightforward).
2. **Recompute `h_tr_em_zone` at every timestep in `step_physics_5r1c`** via `physics::exterior_convection::h_c_ext_wind_dependent` — the helper already exists at `src/physics/exterior_convection.rs:136`, the wind-at-building-height converter at `src/physics/exterior_convection.rs:178`, and the per-step wind sourcing pattern at `physics_impl.rs:339-362` (the existing wind-dependent `h_se` path).
3. **Update the `EnergyPlus-equivalent baseline` invariant check** to read the per-step value rather than the build-time constant.

Each of these is a structural solver-code change that, per `AGENTS.md` and `RULES.md`, cannot be done by a single sub-agent without (a) deep physics expertise, (b) bit-identical or controlled-delta baseline snapshots (per **ADR-0008**), and (c) coordination with the GaugeSolver rework (#1465/#1462 per **#3059**). This ADR ships only the scaffolding — the placeholder snapshot set, the verifier, and the pytest harness — so the eventual implementer has a mechanical, exit-coded way to prove bit-identical physics without any human in the loop approving parameter-tuning-style adjustments.

## What this ADR ships

This ADR ships the **tracking scaffolding** for issue #3063:

### 1. `docs/KNOWN_ISSUES.md` §LIMIT-13 entry

A new LIMIT-13 entry that documents:

- The structural gap (the build-time `1/EXTERIOR_FILM_COEFF_DEFAULT` is consumed at every timestep without being recomputed from the per-step wind speed).
- The post-#3024 measurements (Annual heating 7.42 → 6.25 MWh acceptance-met; annual cooling 220 → 758 kWh open).
- The acceptance criteria (Case 195 cooling ≤ 50 kWh full or ≤ 1500 kWh scoped; peak heating ≤ 1.05 kW already met).
- The diagnostic signature (220 → 758 kWh cooling shift driven by `h_se` wind-dependent sol-air amplification at low wind speeds, with `h_tr_em` still time-invariant).
- The link to the existing `StepHeating` step at `physics_impl.rs:339-362` where the per-step wind-dependent `h_se` is already computed — the same pattern generalises to `h_tr_em`.
- The implementation plan (extend `HvacState` or `MassState` with `h_tr_em_zone: Vec<f64>`; recompute per step; update the `EnergyPlus-equivalent baseline` invariant).
- The *what this PR does NOT do* list (no physics code, no baseline modification, no `ARCHITECTURE.md` / `RULES.md` modification, no architectural decision recorded).
- The references to #2891 (origin), #3024 (partial fix), #3059 (5R1C/9R4C architectural rework), #1465 / #1462 (GaugeSolver), #2868 (sister issue), #3072 (aggressive-baseline cohort meta-issue), and **ADR-0008** (snapshot-diff verifier pattern).

### 2. Snapshot diff verifier (originally `scripts/verify_h_tr_em_regression.py`, removed 2026-08-19 as orphan — see `.agents/results/result-pm.md`)

CLI + Python module. Mirrors the pattern of `scripts/verify_gauge_solver_regression.py` from #3070:

- `argparse` CLI with `--before`, `--after`, `--tolerance`, `--json`, `--strict`, `--allow-placeholder`.
- Exit codes follow the documented `EXIT_OK=0 / EXIT_REGRESSION=1 / EXIT_PLACEHOLDER=2 / EXIT_USAGE=3` contract.
- Human-readable report (default) and JSON output (`--json`).
- `self-test` mode (`python3 scripts/verify_h_tr_em_regression.py --self-test`) for hermetic validation without pytest.
- Fail-closed default: refuses to diff a placeholder snapshot set (exit 2) so a future implementer cannot silently compare against an empty baseline. `--allow-placeholder` is the explicit opt-out.
- SHA-256 fingerprint check (`--strict`): a hand-edited case file whose content drifts from the manifest's stamped `sha256` trips exit 2 so a silent tweak to the placeholder cannot slip past the gate.

*This section is retained as historical context. A future PR that submits the actual per-step recompute must re-derive this verifier — the `argparse` / exit-code / SHA-256 contract documented above is the design that should be restored.*

The snapshot set is intentionally **not** shipped by this PR. The verifier refuses to diff a placeholder set (exit 2), forcing the future implementer to capture real per-step measurements via **ADR-0008**'s pattern before shipping any code change. The directory the future implementer will populate is `tests/reference_data/h_tr_em_baseline/` (mirroring the `tests/reference_data/gauge_solver_baseline/` directory from #3070).

### 3. Pytest harness (originally `scripts/ci/test_verify_h_tr_em_regression.py`, removed 2026-08-19 as orphan)

A pytest harness covering the verifier's contract:

- Snapshot-set loading (success / missing manifest / wrong schema version).
- `is_placeholder` detection.
- `compute_diff` no-drift / regression / tolerance-override / schema-drift scenarios.
- `verify_fingerprints` empty / silent-edit scenarios.
- `main()` end-to-end: clean (exit 0), regression (exit 1), placeholder (exit 2), missing manifest (exit 3), JSON output shape, `--strict` SHA-256 mismatch (exit 2), CLI tolerance override.

*Retained as historical context for the harness coverage that any future verifier re-implementation should restore.*

## What this ADR does NOT do

1. **It does NOT modify physics code.** Per AGENTS.md ("do NOT modify physics code without checking `ARCHITECTURE.md` first"), the actual `h_tr_em_zone: Vec<f64>` extension and per-step recompute are deferred to a future PR. This stub only ships scaffolding.
2. **It does NOT modify `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`.** Per AGENTS.md, the strict-energy-gate baseline must NEVER be raised to hide a regression. The bit-identical baseline for the TDD approach lives in `tests/reference_data/h_tr_em_baseline/`, a separate directory tree, so the two never conflate.
3. **It does NOT modify ARCHITECTURE.md or RULES.md.** Those are source-of-truth documents; this stub references them.
4. **It does NOT record an architectural decision.** The actual `extend-HvacState-vs-MassState` choice is deferred to a future PR that submits both the recomputation and the verifier output.
5. **It does NOT mark any case as passing.** It documents the scaffolding status only.
6. **It does NOT pre-populate the snapshot values.** Per RULES.md ("must-never hardcode results"), the baseline must be captured by running the actual Case 195 / 600 / 620 integration tests with the per-step `h_tr_em_zone` snapshot enabled. The shipped scaffolding forces the first real measurement campaign to produce real numbers.

## Decision

**None recorded.** This is a tracking stub. The actual architectural decision (extend `HvacState` or `MassState` with `h_tr_em_zone: Vec<f64>` and recompute per step via `physics::exterior_convection::h_c_ext_wind_dependent`, with bit-identical physics guaranteed by the verifier) is deferred to a future PR that:

1. Runs the existing Case 195 / 600 / 620 integration tests against `develop` to populate `tests/reference_data/h_tr_em_baseline/` with real numbers, following the **ADR-0008** pattern (placeholder snapshot set, real-measurement capture, no manual tuning).
2. Implements the per-step recomputation in `step_physics_5r1c` (`physics_impl.rs:155`), sourcing the per-step wind speed from `ThermalModelData::weather.wind_speed` via `wind_at_building_height_from_10m` (mirroring the existing `h_se` path at lines 339-362).
3. Runs the verifier end-to-end (the verifier must be re-derived per the contract in §2 — see the orphan-removal note above):
   ```
   python3 scripts/verify_h_tr_em_regression.py \
       --before tests/reference_data/h_tr_em_baseline \
       --after  <post-recompute-snapshot-dir>
   ```
4. Submits a PR with the verifier output as evidence. Exit 0 → merge; exit 1 → recompute rejected and iterated; exit 2 → placeholder drift detected (re-capture required); exit 3 → usage error.

The acceptance is the issue's documented criterion:

- Case 195 annual heating ≤ 6.30 MWh (already met by PR #3024 + Issue #2868).
- Case 195 annual cooling ≤ 50 kWh (full) or ≤ 1500 kWh (scoped).
- Case 195 peak heating ≤ 1.05 kW (already met).

When that PR lands, this stub will be either:

- Superseded by a full ADR that records the extend/recompute decision, the per-case metrics, and the verifier-output evidence; OR
- Marked `Rejected` if the verifier output shows the recompute cannot be made bit-identical and the issue is re-routed to the GaugeSolver rework (#1465/#1462).

## Consequences

### Positive

- The TDD scaffolding for issue #3063 is in place. A future implementer has a mechanical, exit-coded way to prove bit-identical physics without any human in the loop approving parameter-tuning-style adjustments.
- The verifier fails closed by default (`--allow-placeholder` is the opt-out), so a recompute run that forgets to capture real baselines cannot silently green-light itself.
- The SHA-256 fingerprint check catches silent edits to the placeholder that would otherwise pass the placeholder detection by preserving `null` values.
- The pytest harness gives CI a deterministic way to lock the verifier's exit-code contract.

### Negative

- None. This is a tracking stub plus a verifier; it does not change any architecture, test, or pass-rate claim.

### Neutral

- The `Status: Proposed` marker is intentional and remains until the underlying structural PR lands. If the per-step recompute is determined to be insufficient (and the issue is fully re-routed to the GaugeSolver rework), this stub will be marked `Rejected` with a one-paragraph explanation.
- The verifier's per-metric tolerance defaults to `0.0` (bit-identical). Future recomputes that intentionally relax a metric (e.g. to absorb cross-runner numerical noise) must lower the baseline **and** commit the engine improvement together (per RULES.md / AGENTS.md: never raise to hide a regression).

## Per-step `h_tr_em` semantics (documentation for the future implementer)

The recompute is fault-tolerant by construction:

- The current production path consumes `h_tr_em` as a constant per zone (a build-time reciprocal of `EXTERIOR_FILM_COEFF_DEFAULT`).
- The fault-tolerant recomputation is `h_tr_em_step[i] = 1.0 / h_c_ext_wind_dependent(ExteriorSurfaceDirection::VerticalWallWindward, v_building_at_step[i])` where `v_building_at_step[i]` is the per-step wind speed at building mid-height (sourced from the per-step weather buffer via `wind_at_building_height_from_10m`).
- At V = 3.4 m/s, `h_c_ext_wind_dependent(VerticalWallWindward, 3.4)` returns 17.6 W/m²K, which yields `h_tr_em_step = 0.0568 m²·K/W` — within the 5 % band of the legacy `1/EXTERIOR_FILM_COEFF = 0.0546 m²·K/W` (the residual 0.7 W/m²K is the longwave radiative portion that is added on the sol-air side, per `src/physics/exterior_convection.rs:128-135`).
- The production path already does this for the sol-air longwave correction (see `physics_impl.rs:339-362`); the `h_tr_em_zone` extension generalises the same per-step recompute to the envelope-to-mass conductance and lets the 5R1C wall path align with the FD solver / surface-balance paths.
- An alternative direction (orthogonal to the per-step recompute) is the GaugeSolver rework (#1465/#1462), which treats solar as geometric curvature rather than per-timestep energy injection. The two are complementary: the per-step recompute closes the wind-dependent path; the GaugeSolver closes the geometric-curvature path.

## References

- Issue #3063 — `h_tr_em` (envelope-to-mass conductance) remains time-invariant in 5R1C path (#2891 follow-up)
- Issue #2891 — original wind-dependent `h_se` request
- PR #3024 — wind-dependent `h_se` closure (annual heating 7.42 → 6.25 MWh; exposes the cooling shift documented here)
- Issue #2868 — sister issue (Case 195 surface-balance initialisation fix, coupled through the same envelope-to-mass path; closed via PR #3044)
- Issue #3059 — 5R1C/9R4C architectural rework (the GaugeSolver unblocker)
- Issue #1465 — Phase 3 GaugeSolver validation against ASHRAE 140 Case 900 (closed individually, NOT yet production-path)
- Issue #1462 — Phase 1b shadow-mode GaugeSolver in `physics_adapter.rs` (closed individually, NOT yet production-path)
- Issue #3072 — aggressive-baseline cohort meta-issue (Cases 195 / 600 / 620 / 940 / 960)
- `tests/reference_data/h_tr_em_baseline/` — the placeholder snapshot set (to be created by the future implementer; the verifier rejects placeholder snapshots until real measurements are captured)
- `scripts/verify_h_tr_em_regression.py` — the verifier *(removed 2026-08-19 as orphan; re-derive from §2 above)*
- `scripts/ci/test_verify_h_tr_em_regression.py` — pytest harness *(removed 2026-08-19 as orphan; re-derive from §3 above)*
- `docs/KNOWN_ISSUES.md` §LIMIT-13 — the per-case-issue entry (companion to this ADR)
- `docs/adr/0008-thermal-model-data-tdd-refactor.md` — the snapshot-diff verifier pattern (#3070) that this ADR mirrors
- `docs/adr/0007-gauge-solver-structural-work.md` — the architectural unblocker (production-path GaugeSolver switchover)
- `src/physics/exterior_convection.rs` — `h_c_ext_wind_dependent` (the helper the future implementer must call from `step_physics_5r1c`)
- `src/sim/thermal_model_physics/physics_impl.rs:155` — `step_physics_5r1c` (the per-timestep loop where the future implementer must inject the per-step `h_tr_em_zone` recompute)
- `src/sim/thermal_model_physics/physics_impl.rs:339-362` — the existing wind-dependent `h_se` path (the pattern the future implementer mirrors)
- `RULES.md` — "no parameter tuning" + "must-never hardcode results"
- `AGENTS.md` — "do NOT modify physics code without checking ARCHITECTURE.md first"; strict-energy-gate baseline must NEVER be raised
- `ADR-0001` — No-Parameter-Tuning Rule
- `ADR-0007` — GaugeSolver Structural Work (the sibling stub for the architectural unblocker)
- `ADR-0008` — ThermalModelData Refactor (the sibling stub for the snapshot-diff verifier pattern)
- `docs/ASHRAE140_RESULTS.md` §"Structural Blockers (Issue #3072)" — current pass-rate snapshot for the wider aggressive-baseline cohort
