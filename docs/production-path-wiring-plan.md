# Production-Path Wiring Plan: GaugeSolver as Default Thermal Solver

> Status: SUPERSEDED (Phase A8 / #3291 / PR #3482 / 2026-09-07).
> Owner: @anchapin.
> Source-of-truth after Phase A8: `AGENTS.md:12` + `ARCHITECTURE.md:716-735` + `KNOWN_ISSUES.md` §LIMIT-21.
> Cross-references: #3291, #3482, #3297, #3286, #3290, #3511.
> Archived plan: `docs/archive/production-path-wiring-plan.md`.

> **⚠ SUPERSEDED — see Issue #3291 / PR #3482 (Phase A8, 2026-09-07).**
>
> The implementation strategy described in this document was
> **substantially** realised in Phase A8 (#3291, PR #3482, 2026-09-07)
> via a different route than the one this plan envisioned:
>
> - **Selector-driven dispatch** via `ThermalSelector` (the `Default`
>   impl on `ZoneSolverKind` resolves to `Gauge` per #3291) instead of
>   `enable_gauge_solver()` + `step_physics` fall-through.
> - **`from_spec_with_selector` initialisation** (single-zone for
>   `num_zones == 1`, multi-zone for `num_zones >= 2`) instead of a
>   generic `enable_*()` method paralleling `enable_ctf()` / `enable_fd()`.
> - **Panic-on-missing-backend** as the §LIMIT-21 "fail-loud" design
>   instead of "Implement fallback to legacy solvers if GaugeSolver
>   fails" (Phase 3 checkbox below).
> - **`gauge-solver` cfg gate intentionally retained** as the
>   production-path gate pending §LIMIT-21 closure (Issue #3297) /
>   β-soak (#3286) — the cfg-gate removal is the work tracked by
>   #3290 (PR4 of the design tree).
>
> The full original plan is preserved at
> `docs/archive/production-path-wiring-plan.md` for historical reference
> (it is referenced by #3213, the original tracking issue).

## Cross-references to the post-#3291 reality

- `AGENTS.md:12` (Phase A8 note) and `AGENTS.md` Module 5 (swap-point update)
- `ARCHITECTURE.md:716-735` (Thermal selector Phase A8 note)
- `ARCHITECTURE.md:1267-1269` (Phase A8 status note)
- `docs/adr/0007-gauge-solver-structural-work.md` (ADR-0007 status updated
  per #3511 to reflect the Phase A8 landing)
- `docs/KNOWN_ISSUES.md` §LIMIT-21 (Issue #3297) + §LIMIT-22
- `docs/agents/beta-soak-criterion-2-tracker.md`
- `src/sim/thermal_selector.rs` (module docs + `ThermalSelector::legacy`
  helper for `ThermalModel::new`, Issue #3508)
- `src/sim/thermal_model_physics/step_dispatcher.rs:99-160` (the
  selector-driven dispatch + the unconditional-panic guard)

## Resolved by Phase A8 — checkbox reconciliation

| Original plan item (this doc) | Phase A8 outcome | Cross-reference |
| ----------------------------- | ---------------- | --------------- |
| Phase 3 — "Remove `#[cfg(feature = \"gauge-solver\")]` gate from step_dispatcher gauge block" | **NOT removed** — the cfg gate is intentionally retained as the production-path gate pending §LIMIT-21 closure. | `AGENTS.md:12`; `src/sim/thermal_model_physics/step_dispatcher.rs:99` |
| Phase 3 — "Make `gauge_zone_solver: Option<GaugeZoneSolver>` available in default build" | **NOT changed** — field is still cfg-gated; the `Gauge` selector is reached via `from_spec_with_selector` which initialises the matching backend. | `src/sim/thermal_model_core.rs::from_spec_with_selector`; `AGENTS.md:12` |
| Phase 3 — "Route `step_physics_5r1c` / `step_physics_9r4c` to GaugeSolver" | **Resolved** via selector-driven dispatch, not the plan's `enable_*` + fall-through approach. | `src/sim/thermal_model_physics/step_dispatcher.rs:99-160` |
| Phase 3 — "Implement fallback to legacy solvers if GaugeSolver fails" | **REVERSED** — Phase A8 explicitly removed the fallback (the §LIMIT-21 "fail-loud" design). A missing gauge backend PANICS. | `src/sim/thermal_model_physics/step_dispatcher.rs:126-133`; `KNOWN_ISSUES.md` §LIMIT-21 |
| Phase 2 — "Create `enable_gauge_solver()` method following `enable_ctf()` / `enable_fd()` pattern" | **Partially resolved** — `enable_gauge_solver` / `enable_gauge_solver_multi_zone` were added in #3275 PR2.1, but the dispatcher never consults them the way this plan envisioned. The actual initialisation is via `from_spec_with_selector`. | `src/sim/thermal_model_core.rs::from_spec_with_selector` |
| Phase 1 — "Store `WallSpec` in `WallSurface` or parallel structure" | **Not resolved** in Phase A8 — the gauge backend still requires `WallSpec` for `add_opaque_surface`, and the REST/CLI paths cannot supply it (fail-closed per #3305). This is the §LIMIT-21 residual work. | `KNOWN_ISSUES.md` §LIMIT-21 (Issue #3297) |

## Remaining work (NOT in this plan; tracked separately)

- **#3286 (β-soak gate, 30 consecutive nightly green):** production path
  for the gauge rollout.
- **#3290 (PR4 cfg-gate removal):** once #3286 trips, remove the
  `#[cfg(feature = "gauge-solver")]` gate in `step_dispatcher.rs:99`
  and the corresponding field declarations.
- **#3297 / §LIMIT-21:** the wall-spec-required gauge init for
  REST/CLI bindings; gated on #3286.

---

*Issue #3512: this redirect doc replaces the live `docs/production-path-wiring-plan.md`
which is now archived at `docs/archive/production-path-wiring-plan.md` (commit
$(git log -1 --format=%H docs/archive/production-path-wiring-plan.md)). The
archived copy preserves the pre-Phase A8 plan verbatim for diff / historical
reference.*
