# Result — Backend (Issue #2806)

## Status: ✅ COMPLETE

Physics simulation divergence in `tests/python/test_model_mutations.py` is **resolved**. All 48 tests in that file pass; no divergence remains.

## Summary

Issue #2806 reported `fluxion.SimulationError: simulation diverged at timestep 0 in zone zone_0` in `tests/python/test_model_mutations.py`, surfaced once PR #2805 un-broke the `python-bindings.yml` workflow YAML. The suspected root cause — a `C_m = 1.0 J/K` placeholder from a code path bypassing `ThermalModel::from_spec` (same family as the REST `/v1/simulate` bug fixed in #2803/#2747) — **was confirmed and fixed**.

The fix mirrors PR #2803's `build_model_from_schema` resolution, applied to the two PyO3 entry points that built the model via the placeholder path. Two further latent wiring defects (also in the Python bindings, and likewise only visible once divergence was lifted) were fixed in the same pass; one additional pre-existing, out-of-scope bug was documented.

## Root cause (confirmed)

`ThermalModel::new(num_zones)` (src/sim/thermal_model_core.rs:2600) deliberately leaves `thermal_capacitance = 1.0 J/K` (line 2687) and `air_thermal_capacitance = 0.0` (line 2693) — placeholders meant to be overwritten by `from_spec`, which the Python bindings never call. With `C_m = 1.0`, `select_integration_method` picks conditionally-stable Explicit-Euler and the update `Tm += (q_net/C_m)·dt` with `dt = 3600 s` blows up at timestep 0. This is identical to LIMIT-07 (#2747), previously fixed only for the REST path.

Both Python constructors hit this path:
- `Model(num_zones)` → `src/python/model_bindings.rs::Model::new`
- `MultiZoneThermalModel(num_zones)` → `src/python/bindings.rs::PyMultiZoneThermalModel::new`

(The `from_case_spec` / `from_case` paths were already correct — they use `from_spec`.)

## Files changed

| File | Change |
|------|--------|
| `src/python/model_bindings.rs` | New `populate_default_model_physics(&mut ThermalModel)` helper mirroring #2803's `build_model_from_schema` (ISO 13790 §7.2 capacitance `C_m = wall+roof+floor cap`, `C_air = ρ·cp·V`, `h_tr_ms`/`h_tr_em`/`h_tr_me` from ISO 13790 Eq. 64 / §7.2.2.2, using real ASHRAE 140 Case 600 `Assemblies::low_mass_*` constructions — no invented capacitance numbers). Called from `Model::new`. `Model::simulate` now passes `Some(&empty_lighting)` (envelope-only, matching #2747) so the auto-loaded Office profile's internal-gains accumulation quirk does not drive EUI negative. Added 5 Rust regression tests. |
| `src/python/bindings.rs` | `PyMultiZoneThermalModel::new` calls `populate_default_model_physics`. `simulate_multi_zone` passes `Some(&empty_lighting)` and **resets cumulative energy trackers** (`reset_heating_cooling_energy()`) before each run — the method returns `annual_heating+annual_cooling` trackers that were never reset, so sequential calls previously returned N× the per-call energy (10429 → 20861 → 31294 …), breaking determinism. |

No engine-core (`src/sim/**`, `src/physics/**`) changes — the fix is confined to the binding/wiring layer, per scope. `ThermalModel::new` itself is untouched (its existing Rust callers use surrogates or single steps, or overwrite via `from_spec`).

## The fix (3 wiring defects, all in the Python bindings)

1. **C_m placeholder (the issue's subject).** `populate_default_model_physics` populates real envelope physics from the default geometry + a real low-mass construction. Lifts `C_m` from 1.0 J/K to a real envelope value (→ Crank-Nicolson, unconditionally stable). Divergence eliminated.
2. **Negative EUI from auto-loaded Office profile.** With all-`None` loads, `solve_timesteps_with_dt` auto-loads the bundled Office profile (solver_core.rs:206), whose per-step `loads[i] += internal_gains` accumulation quirk overheats the small default zone and drives EUI negative (net cooling). `simulate`/`simulate_multi_zone` now pass `Some(&empty_lighting)` → envelope-only baseline, exactly as #2803 did for the REST path. (`simulate_with_loads` remains the documented auto-load path.)
3. **Non-deterministic `simulate_multi_zone`.** It ignored `solve_timesteps`' per-call EUI and returned never-reset cumulative `annual_heating_energy + annual_cooling_energy` trackers → linear accumulation across calls. Now resets them before each run → deterministic, per-call energy.

## Acceptance criteria checklist

- [x] `tests/python/test_model_mutations.py` runs with **no divergence** — 48/48 pass.
- [x] Root cause = `C_m = 1.0` placeholder (confirmed: `new()` leaves 1.0; guard test `new_leaves_cm_one_placeholder_before_fix` pins this).
- [x] Fix wires real physics (ISO 13790 §7.2 capacitance + conductances from real ASHRAE 140 Case 600 constructions) — no parameter tuning, no hardcoded results, energy-conserving stable solver selected.
- [x] Mirrors the #2803/#2747 resolution pattern.
- [x] Changes scoped to binding/wiring + tests (no engine-core edits).
- [x] `cargo fmt -- --check` clean · `cargo clippy --lib --features python-bindings -- -D warnings` clean.

## Verification performed

```
# Rust-level (C_m no longer 1.0; analytical path stable; office-profile quirk pinned)
cargo test --features python-bindings --profile ci --lib python::
  → 126 passed; 0 failed   (incl. 5 new #2806 regression tests)

# CI gates
cargo fmt -- --check                                        → clean
CARGO_BUILD_JOBS=1 cargo clippy --lib --features python-bindings -- -D warnings  → clean
cargo build --profile ci                                    → clean (default build unaffected)

# Actual failing pytest from the issue (maturin wheel built locally)
maturin develop --features python-bindings
python3 -m pytest tests/python/test_model_mutations.py -q   → 48 passed in 9.5s
```

New Rust regression tests: `new_leaves_cm_one_placeholder_before_fix`, `populate_default_physics_replaces_cm_placeholder_single_zone`, `populate_default_physics_replaces_cm_placeholder_multi_zone`, `default_model_analytical_simulation_does_not_diverge`, `default_multi_zone_model_analytical_simulation_does_not_diverge`, `office_profile_autoload_is_the_negative_eui_cause`.

maturin + pytest were available locally and WERE run (strongest verification). Per-call EUI is physically sane: single-zone ≈ 174 kWh/m²/yr; multi-zone ≈ 10 429 kWh ÷ 60 m² ≈ 174 kWh/m²/yr (heating-dominated, consistent with #2803's ≈ 112 for a different geometry).

## Out-of-scope findings (NOT fixed — separate pre-existing bugs)

Running the full `tests/python/` suite surfaces other failures that are **pre-existing and unrelated** to the #2806 divergence. They were masked previously (the bindings diverged before reaching them):

- **`test_api_transformation.py` thermostat tests** (`test_change_heating_setpoint`, `test_change_cooling_setpoint`): `MultiZoneThermalModel.set_zone_setpoints(zone, h, c)` has **zero effect** on simulated energy — fresh models with setpoints (20,24) / (22,24) / (15,30) all return *identical* 10429.05. This is a separate multi-zone solver wiring bug (setpoints don't propagate to the multi-zone HVAC energy). It was previously hidden by the #3 accumulation artifact (2X ≠ X made the tests appear to pass); the correct reset fix unmasked it. Out of scope for #2806 — recommend a separate issue.
- `test_api_transformation.py::test_construction_u_value` (0.269 ≠ expected), `test_hvac_bindings.py::test_heat_pump_configuration_and_cop` (COP derating 3.5 ≮ 3.5), `test_state_store.py` (`KPIResult.heating_mae` renamed → `heating_mae_max`), `test_generate_scorecard.py` (AttributeError): all unrelated to physics divergence.

## Commit

`fix(bindings): route Python model construction through real physics init — closes #2806`
