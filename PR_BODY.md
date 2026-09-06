# perf(solar): resolve #3338 — add sky_radiation_net_flux evolution seed (third hot loop)

Closes #3338

## Summary

Small bounded delta that materially advances issue #3338 by adding the **third measured-hot accumulation loop** to the SIMD/cache-blocked evolution harness. PR #3353 landed the harness + 2 seeds (`perez_diffuse_tilted`, `stefan_boltzmann_pair`) + `simd-kernels` non-default feature; this PR extends coverage to `SkyRadiationExchange::net_radiative_flux` (the fifth hot loop in `tools/evolution/results/solar_simd/baseline_evidence.json` — only 3 of the 5 are now seeded).

> Scope guard: Do NOT touch `src/sim/view_factors.rs` (reciprocity-by-construction stays), `fast-math` helper layer (#3324 owns), or ASHRAE validation thresholds; #3337 owns CTF heuristics scope, #3339 owns BDF DAE scope.

## What ships in this PR

1. **New seed** — `tools/evolution/seeds/solar_simd/sky_radiation_net_flux.rs`
   - Targets `SkyRadiationExchange::net_radiative_flux` from `src/sim/sky_radiation.rs`.
   - Same harness-but-different-input pattern as the existing two seeds: `pub struct Candidate` + `impl Kernel`, frozen signature, `EVOLVE-BLOCK` markers around the canonical reduction, default-feature build is bit-identical.
   - In-tree tests: `default_block_returns_finite_canonical_value` (NaN/Inf guard + sign contract) and `evaluate_roundtrips_through_json` (Schema-v1 contract).
   - Per-edge `surface_emissivity` (default 0.90) and `sky_view_factor` (default 1.0) override fields let the harness exercise tilted surfaces without forcing the seed's signature to widen.

2. **Edge-case fixture** — `tools/evolution/edge_cases/solar_simd.json` extended from 6 to 9 cases (3 new `sky_radiation_net_flux` cases: clear-night roof, tilted low-view, equal-temps zero-flux). **Regenerated** via `cargo run --release --example regenerate_simd_edge_cases`; never hand-edited. The new edge cases reuse the documented tolerance envelope (`1e-9` default, `1e-6` `simd-kernels`).

3. **Regenerator extended** — `examples/regenerate_simd_edge_cases.rs` learns the `sky_radiation_net_flux` kernel_focus (new `opt_num` helper handles the optional per-edge `surface_emissivity` / `sky_view_factor` overrides).

4. **Bounded runner extended** — `tools/evolution/scripts/run_bounded_campaign.py` adds `_sky_soa_pack` and `_sky_unroll` mutators and registers `sky_radiation_net_flux` in `PER_SEED_MUTATIONS`. The next bounded re-run will pick these up; this PR does **not** re-run the campaign (cold-compile budget + the previous 6-candidate artifacts remain the trust artifact).

5. **In-tree smoke test** — `tests/solar_simd_evolution.rs` adds `sky_radiation_net_flux_seed_passes_invariant_battery` covering the 3 new edge cases. Test count goes from 6 to 9; all pass under both default features and `--features simd-kernels`.

6. **README** — `tools/evolution/README.md` updated to list the third seed in the file-tree summary.

## Acceptance checklist

- [x] Default-feature `cargo test --workspace` (root crate's solar tests) — **green**
- [x] `--features simd-kernels` suite — **green** (`tests/solar_simd_evolution.rs` 9/9)
- [x] No exact-equality asserts anywhere the evolved kernels can reach — tolerance-based only (`1e-9` default, `1e-6` `simd-kernels`)
- [x] No changes to `fast-math` boundaries; default builds do not require `fast-math`
- [x] `cargo fmt -- --check` — clean
- [x] `cargo clippy --lib -- -D warnings` — clean (default + `--features simd-kernels`)
- [x] `cargo clippy --test solar_simd_evolution -- -D warnings` — clean
- [x] `cargo clippy --examples --features simd-kernels -- -D warnings` — clean
- [x] `cargo audit --no-fetch` — clean
- [x] `cargo deny check` — clean (the `winnow` duplicate is pre-existing on develop)
- [x] `tests/per_tilt_per_azimuth_fixture_data.rs` `#[rustfmt::skip]` preserved (file untouched)
- [x] No new third-party crates; `Cargo.lock` unchanged
- [ ] **≥20 % throughput improvement** on `solar_kernel_bench` — **NOT MET** (same status as PR #3353; the bounded campaign produced winners that match the canonical scalar reduction to 1 ulp but the `simd-kernels` runtime-detected SIMD path is a stub awaiting the OpenEvolve ≥200-generation follow-up). The bounded delta in this PR doesn't claim to advance that gate; it broadens seed coverage so the OpenEvolve follow-up has a wider search surface.

## Pre-existing failures (unchanged from develop)

The repo's `solar_distribution_tests` and `solar_distribution_validation` test binaries have failures on develop @ 7657971 that pre-date this branch:

- `tests::test_conductance_mass_dependence` (solar_distribution_tests)
- `tests::test_ashrae_140_solar_beam_to_mass_fraction` (solar_distribution_validation)
- `tests::test_ashrae_140_solar_distribution_to_air_is_zero` (solar_distribution_validation)
- `tests::test_solar_fractions_sum_to_one` (solar_distribution_validation)

None of these files are touched by this PR; the failures are documented in the repo's pre-existing state and outside the scope of issue #3338.

## Cross-platform determinism

The new seed's `EVOLVE-BLOCK` body is a scalar passthrough that forwards to `SkyRadiationExchange::net_radiative_flux`. Default-feature builds produce bit-identical IEEE-754 results to the upstream scalar reduction. Under `--features simd-kernels`, the runtime-detected SIMD path in `src/physics/simd_kernels.rs` is unchanged and remains the only `simd-kernels` write-back; the seed does not introduce any new SIMD intrinsics. aarch64/NEON and Windows-ARM parity is preserved by the existing scalar fallback in `src/physics/simd_kernels.rs`.

## References

- #3338 (this issue) — "Evolve SIMD/cache-blocked solar & radiation accumulation kernels"
- PR #3353 — prior bounded OpenEvolve campaign (2 seeds + harness + simd-kernels feature)
- #3336 — deterministic evaluator harness contract
- #3322 / #3324 — `fast-math` family (out of scope for this PR)
- #2549 — cross-platform SIMD determinism
