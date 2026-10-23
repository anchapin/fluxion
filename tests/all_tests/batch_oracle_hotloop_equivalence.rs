//! Numerical-equivalence regression guard for Issue #2687.
//!
//! Issue #2687 reworked the BatchOracle per-timestep hot loop purely for
//! allocation reduction (VectorField → SmallVec backing, scratch fields →
//! SmallVec, and zero-alloc `get_temperatures_into` / `predict_loads_into` /
//! `set_loads`-via-`from_slice` on the CPU surrogate path). None of those
//! changes may alter the *numerical* output — they are storage/ownership
//! refactors that preserve identical f64 values in identical order.
//!
//! This test snapshots the EUI output of a small, deterministic
//! `evaluate_population` run on **both** code paths exercised by the
//! allocation work:
//!
//! 1. The analytical path (`use_surrogates = false`) — the same path the
//!    `dhat_alloc_budget` gate measures; it drives the 5R1C solver's
//!    SmallVec-backed VectorField arithmetic and the SmallVec scratch
//!    buffers.
//! 2. The CPU surrogate / mock path (`use_surrogates = true`, no ONNX model
//!    loaded) — this is the `run_cpu_surrogate` orchestrator hot loop where
//!    `get_temperatures_into` / `predict_loads_into` / `set_loads` were made
//!    allocation-free.
//!
//! The golden EUI arrays below were captured from a known-good run. Any future
//! change that perturbs the physics output (e.g. an accidental reassociation
//! of floating-point ops, a scratch-buffer aliasing bug, or a SmallVec
//! ordering regression) will flip one of these `assert_eq!`s and fail the
//! gate. When the allocation optimization in #2687 landed, these values were
//! verified bit-identical to the pre-change baseline.
//!
//! Run: `cargo test --profile ci -p fluxion --test batch_oracle_hotloop_equivalence`

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::BatchOracle;

/// Same base model as the shared FFI default (`BatchOracle::default_model`):
/// the ASHRAE 600 baseline via the default selector, so this equivalence
/// guard measures a well-posed single-zone analytical model.
///
/// Issue #4496 root cause (diagnosed 2026-10-19, round 70): the previous
/// fixture built a bare `ThermalModel::<VectorField>::new(1)` — a bare
/// low-level constructor whose derived envelope conductances leave the
/// 5R1C mass-node advance numerically unstable for this population. With
/// per-step energy tracing: the state amplified ~×2.5/step to ~1e154 by
/// step 69, saturated the ideal-HVAC ±100 kW cap (±100 kWh steps), and
/// hit inf/NaN at step 72 — the remaining ~8 688 hours of the "annual"
/// run were a dead NaN state. The old goldens
/// (`[0.00735449732369986, 0.00895211063575374, 5.010550650878273]`
/// analytical / `[0.013806125988018891, 0.017001003523629522,
/// 5.020197761109098]` surrogate) were therefore the energy of a
/// numerically dead simulation, not physics. The #4495 input-envelope
/// guards (#4497) reject the first non-finite advance (step 72), which
/// surfaced as the mysterious "exact ±5.0 golden flip" of #4496 — a real
/// signal that the fixture model was blowing up, NOT a compile-layout
/// sensitivity (the round-69 "guards never fire, 204/205 calls" counter
/// was mis-measured; the guards fire from step 72 of member 1 on this
/// population). The fixture is fixed to the well-posed ASHRAE 600
/// baseline; the goldens below are re-derived from the corrected fixture
/// on develop `76dcabc9` (in-envelope engine behavior bit-identical to
/// the #4497 engine — verified by running the same fixture on both).
fn create_single_zone_model() -> ThermalModel<VectorField> {
    BatchOracle::default_model().expect("ASHRAE 600 baseline default model must initialize")
}

/// Deterministic, always-valid population: a window-U sweep with a fixed
/// heating/cooling setpoint pair, mirroring `dhat_alloc_budget.rs`.
fn deterministic_population(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| vec![0.5 + (i as f64) * 0.4, 20.0, 26.0])
        .collect()
}

#[test]
fn analytical_path_eui_is_bit_identical_to_baseline() {
    let oracle = BatchOracle::from_model(create_single_zone_model()).unwrap();
    let population = deterministic_population(3);

    // Analytical physics path (use_surrogates = false). Sequential per-config,
    // deterministic regardless of thread scheduling.
    let results: Vec<f64> = oracle
        .evaluate_population(population, false)
        .expect("analytical evaluate_population must succeed");

    // Sanity: every config is valid (heating < cooling), so all EUIs finite.
    assert!(
        results.iter().all(|r| r.is_finite()),
        "analytical EUIs must all be finite: {results:?}"
    );

    // Golden snapshot captured from the known-good (post-#2687) run and
    // verified — by stashing the refactor and re-running on pristine
    // `develop` physics — to be IEEE-754 bit-identical to the pre-change
    // output (`struct.pack('>d', x)` bit patterns matched for every element).
    // Bit-identical comparison; if any element differs the physics output
    // changed. The three configs use window U-values 0.5 / 0.9 / 1.3 with
    // heating=20°C, cooling=26°C.
    //
    // Updated 2026-08-28: The ThermalModelData refactoring in commit
    // 0545b03 (issue #2878) changed `model.zone_area` to
    // `model.setpoints.zone_area`. This altered the EUI output order due to
    // the parallel collection ordering in the analytical evaluation path,
    // and changed the EUI magnitudes (the moved field has a different
    // default): [0.10639163960830303, 0.0, 2.3658276261447644] → the values
    // below. See issue #3232.
    //
    // Corrected 2026-09-12 (issue #3708): PR #3243 (commit 8600e22) pasted
    // these constants 3-decimal-truncated (`2.099`, `2.959`) instead of the
    // full-precision f64 values, so this gate has been red since that PR
    // merged — its test-suite CI jobs were in the cancelled-probe /
    // Hetzner-overflow fallback state and the binary never executed there.
    // This is a transcription fix, not a physics change: the analytical-path
    // output was verified bit-identical (same
    // [2.099433588906497, 2.9595394892940883, 0.0]) when running this exact
    // test at 8600e22 (the #3232 golden-set commit) and at develop tip
    // e8691c0, so no commit in that window — including c20e78a's
    // deterministic occupancy seeding (#3707) and #3635's BatchOracle error
    // propagation — perturbed the physics. Transcription deltas were
    // +4.33588906497e-4 (+2.07e-4 relative) and +5.394892940883e-4
    // (+1.82e-4 relative), exactly the 3-decimal truncation error of the
    // committed literals.
    //
    // Re-recorded 2026-10-07: the #4241 discrete-residual physics change
    // (PR #4258, merged as 63eb02e9) altered ideal-HVAC load computation and
    // deliberately re-recorded the engine baselines (strict-energy and fabric
    // baselines re-recorded in 7739ea4a/742539cf), but missed this gate.
    // Bisected: values are bit-identical at aedb0d2a (pre-#4258) and change
    // exactly at 63eb02e9; identical across develop tip and repeat runs
    // (deterministic). So this is a legitimate physics-baseline refresh, not
    // a regression: goldens re-derived from current engine behavior.
    // Re-recorded 2026-10-19 (loop round 73): the R71 sub-hour air-driver
    // default flip (#1522; env-gated `FLX_R71_AIR_CAP` harness in round 72,
    // PR #4505) changed the 5R1C air-node advance, so Case 600 (single-zone,
    // on the modified path) moves again; the strict-gate, fabric, output-diff
    // and this hotloop golden were re-recorded together from the corrected
    // engine on loop73/air-cap-default-flip (same precedent as the
    // 2026-10-07 #4258 refresh: legitimate physics-baseline refresh, not a
    // regression). Previous goldens (D5 flip, round 71):
    // [59.46714897354147, 110.3238524931423, 136.48280523784027].
    // Re-recorded 2026-10-20 (loop round 78): the coupled single-N default
    // flip (`FLX_R77_COUPLED_N` promoted to the engine default at fixed
    // N=12, Alex-directed 2026-10-20 17:12 ET, #3983 §63/§64) changed the
    // 5R1C mass/air advance again; same corrected-engine refresh precedent
    // (strict-gate, fabric, output-diff and this golden re-recorded
    // together from the flipped default build). Previous goldens (R71
    // air-driver flip, round 73):
    // [58.99791056935924, 110.51711796412485, 136.651947557101].
    // Re-recorded 2026-10-21 (loop round 85): the per-class coupled
    // sub-step mass advance promoted to the engine default
    // (`FLX_R84_PERCLASS`, Alex-directed 2026-10-21 21:18 ET, #3983 §65)
    // changed the 5R1C mass/air advance on the spec-built Case 600
    // baseline again; same corrected-engine refresh precedent as rounds
    // 73/78 (strict-gate, fabric, output-diff, §6/§7 pins, grid adapter
    // and this golden re-recorded together from the flipped default
    // build). The analytical path's EUI is highly sensitive to the mass
    // advance (round 78 moved it +67 %); this round moved it
    // [98.47883488604658, 131.92702641312698, 152.40195691921105] ->
    // [74.00399446463592, 115.20549183194221, 140.4909718771871]
    // (−25 %/−13 %/−8 %), the largest swing since round 78 — flagged for
    // review, deterministic across runs. Nothing tuned.
    // Re-recorded 2026-10-22 (loop round 89): the ISO 13790 §C.3
    // surface-pool mass share promoted to the engine default
    // (`FLX_R88_PHIST_MASS_SHARE`, Alex-directed 2026-10-22 14:57 ET,
    // round-88 arm #4543) changed the 5R1C mass/air advance again; same
    // corrected-engine refresh precedent as rounds 73/78/85
    // (strict-gate, fabric, output-diff, §6/§7 pins and this golden
    // re-recorded together from the flipped default build). This round
    // moved the analytical EUI
    // [74.00399446463592, 115.20549183194221, 140.4909718771871] ->
    // [60.698137054765, 109.47558035782295, 139.72333408786784]
    // (−18 %/−5 %/−0.5 %). Nothing tuned.
    // Re-recorded 2026-10-22 (loop round 89b, item 2a): the mass-node
    // full time-coverage fix promoted to the engine default
    // (`FLX_R89_MASS_DT_ALIGN`, Alex-directed standing rule; flip commit
    // ceb19b0b). The analytical EUI moved
    // [60.698137054765, 109.47558035782295, 139.72333408786784] ->
    // [78.39195778347752, 128.81993774427215, 156.4015468093156]
    // (+29 %/+18 %/+12 %) — the mass node now integrates the full hour
    // (coverage 300 s -> 3600 s/hour); deterministic across runs.
    // Nothing tuned.
    // Re-recorded 2026-10-22 (loop round 90): the R77 solar-lag filter
    // removed from the default route (`FLX_R90_NO_LAG_FILTER` default-on;
    // ISO 13790 §C.3 algebraic surface node, §LIMIT-28 round-90 record).
    // The analytical EUI moved
    // [78.39195778347752, 128.81993774427215, 156.4015468093156] ->
    // [80.56768592230117, 133.08315456372387, 160.64670886138663]
    // (+2.8 %/+3.3 %/+2.7 %). Nothing tuned.
    let golden = [80.56768592230117, 133.08315456372387, 160.64670886138663];
    assert_eq!(
        results.as_slice(),
        golden.as_slice(),
        "analytical-path EUI drifted from #2687 baseline (allocation refactor \
         must be numerically inert): got {results:?}"
    );
}

#[test]
fn cpu_surrogate_mock_path_eui_is_bit_identical_to_baseline() {
    let oracle = BatchOracle::from_model(create_single_zone_model()).unwrap();
    let population = deterministic_population(3);

    // CPU surrogate / mock path (use_surrogates = true, no ONNX model loaded).
    // This is the `run_cpu_surrogate` orchestrator hot loop whose
    // `get_temperatures_into` / `predict_loads_into` / `set_loads` calls were
    // made allocation-free in #2687. Result placement is by population index,
    // so the output is deterministic across rayon chunk-completion ordering.
    let results: Vec<f64> = oracle
        .evaluate_population(population, true)
        .expect("surrogate evaluate_population must succeed");

    assert!(
        results.iter().all(|r| r.is_finite()),
        "surrogate/mock EUIs must all be finite: {results:?}"
    );

    // Golden snapshot captured from the known-good (post-#2687) run and
    // verified bit-identical to the pre-change baseline (same stash-and-
    // re-run procedure as the analytical path above).
    // Re-recorded 2026-10-19 with the analytical path above (R71 air-driver
    // default flip, loop round 73, #1522): same corrected-engine refresh.
    // Previous goldens (D5 flip, round 71):
    // [238.26898220394858, 252.92568913230716, 264.3030402695419].
    // Re-recorded 2026-10-20 with the analytical path above (coupled
    // single-N default flip, loop round 78, #3983 §63/§64): same
    // corrected-engine refresh. Previous goldens (R71 air-driver flip,
    // round 73):
    // [238.26081821570014, 252.9180153722391, 264.2953715294398].
    // Re-recorded 2026-10-21 (loop round 85) with the analytical path
    // above: per-class coupled sub-step mass advance default
    // (`FLX_R84_PERCLASS`, #3983 §65). Previous goldens (round 78
    // coupled-N flip):
    // [244.9999907038609, 256.9802090722031, 266.941327636793].
    // Re-recorded 2026-10-22 (loop round 89) with the analytical path
    // above: ISO §C.3 surface-pool mass share default
    // (`FLX_R88_PHIST_MASS_SHARE`, #3983 round 89). Previous goldens
    // (round 85 per-class flip):
    // [237.20858313258188, 252.24156042826417, 263.7616804309977].
    // Re-recorded 2026-10-22 (loop round 89b, item 2a mass-node
    // time-coverage default flip):
    // [236.6238139068431, 252.05392915614644, 263.75261896509267] ->
    // [238.29500516989185, 253.05445617909788, 264.43994131835].
    // Re-recorded 2026-10-22 (loop round 90, R77 lag-filter removal
    // default flip, `FLX_R90_NO_LAG_FILTER`, §LIMIT-28 round-90 record):
    // surrogate/mock EUIs moved
    // [238.29500516989185, 253.05445617909788, 264.43994131835] ->
    // [238.2946250962913, 253.05417431579534, 264.43970946521875]
    // (<= 0.0002 %). Nothing tuned.
    let golden = [238.2946250962913, 253.05417431579534, 264.43970946521875];
    assert_eq!(
        results.as_slice(),
        golden.as_slice(),
        "CPU-surrogate/mock-path EUI drifted from #2687 baseline (allocation \
         refactor must be numerically inert): got {results:?}"
    );
}

#[test]
fn hot_loop_is_deterministic_across_runs() {
    // Allocation refactors that reuse buffers can introduce non-determinism
    // if a buffer is ever read before being fully rewritten. Run the
    // analytical path twice and assert byte-for-byte identical output — this
    // catches any stale-buffer aliasing introduced by the #2687 reuse work.
    let oracle = BatchOracle::from_model(create_single_zone_model()).unwrap();
    let population = deterministic_population(4);

    let a = oracle
        .evaluate_population(population.clone(), false)
        .unwrap();
    let b = oracle.evaluate_population(population, false).unwrap();

    assert_eq!(
        a, b,
        "analytical hot loop must be deterministic across runs"
    );
}
