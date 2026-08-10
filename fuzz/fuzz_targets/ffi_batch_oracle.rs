//! Fuzz target: `BatchOracle::evaluate_population` (FFI surface).
//!
//! This is the Rust function the PyO3 `BatchOracle.evaluate_population_py`
//! entrypoint (src/lib.rs) and the NAPI `BatchOracle.evaluate_population`
//! (src/napi/batch_oracle.rs) both delegate to. It is the single hottest
//! cross-boundary call for optimisation loops, so it must never panic on
//! attacker- or user-supplied parameter vectors.
//!
//! The fuzzer feeds arbitrary parameter populations, including:
//!   * NaN / Inf / subnormal floats (validated up-front by `validate_parameters`),
//!   * extreme window U-values (1e-6 .. 1e6 W/m^2K — far outside the documented
//!     0.1..5.0 physical range),
//!   * heating >= cooling setpoint swaps (rejected by the validation logic),
//!   * zero- and very-large populations (exercises rayon `par_iter` over empty
//!     and large slices, guarding against the thread-pool exhaustion documented
//!     in the BatchOracle pre-commit hook),
//!   * parameter vectors with extra trailing elements (apply_parameters must
//!     ignore them).
//!
//! **Invariant:** the call must never panic / abort. Invalid inputs must be
//! rejected with `Err(FluxionError::Validation)`; valid inputs must produce a
//! finite, non-negative EUI per candidate (NaN marks a filtered-out candidate).

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

/// Cap the zone count so a single fuzz iteration stays cheap (the physics
/// solver allocates O(num_zones) state per candidate). The real FFI allows
/// any `usize`, but values > 64 do not exercise new code paths.
const MAX_ZONES: usize = 32;
/// Cap population size per iteration — `evaluate_population` already handles
/// arbitrary sizes via rayon chunks; we only need enough to cover the empty,
/// single, small-batch, and parallel-chunk branches.
const MAX_POPULATION: usize = 16;

/// One design candidate. Raw `u64` bit-patterns are reinterpreted as `f64`
/// so the fuzzer can explore the full IEEE-754 space (NaN, Inf, subnormals,
/// and extreme magnitudes) rather than just the 0..1 float range.
#[derive(Arbitrary, Debug)]
struct Candidate {
    u_value_bits: u64,
    heating_bits: u64,
    cooling_bits: u64,
    /// Extra trailing parameters — `apply_parameters` must tolerate these
    /// without indexing out of bounds.
    trailing: Vec<u8>,
}

impl Candidate {
    fn to_params(&self) -> Vec<f64> {
        let mut params = vec![
            f64::from_bits(self.u_value_bits),
            f64::from_bits(self.heating_bits),
            f64::from_bits(self.cooling_bits),
        ];
        // Reinterpret each full 8-byte chunk as an extra f64 parameter.
        for chunk in self.trailing.chunks_exact(8) {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(chunk);
            params.push(f64::from_bits(u64::from_le_bytes(arr)));
        }
        params
    }
}

#[derive(Arbitrary, Debug)]
struct FfiInput {
    /// Number of zones in the base model template.
    num_zones_raw: u8,
    /// Whether to use AI surrogates (CPU analytical path vs. surrogate path).
    use_surrogates: bool,
    /// The population of design candidates to evaluate.
    population: Vec<Candidate>,
}

fuzz_target!(|input: FfiInput| {
    // Clamp the zone count into a reasonable fuzzing range. `ThermalModel::new`
    // accepts any usize but allocates O(n) state, so we bound it to keep each
    // iteration fast.
    let num_zones = (input.num_zones_raw as usize).clamp(1, MAX_ZONES);

    let base_model =
        fluxion::sim::engine::ThermalModel::<fluxion::physics::cta::VectorField>::new(num_zones);
    let oracle = fluxion::BatchOracle::from_model(base_model);

    // Bound the population so a single iteration cannot OOM on huge inputs.
    let population: Vec<Vec<f64>> = input
        .population
        .into_iter()
        .take(MAX_POPULATION)
        .map(|c| c.to_params())
        .collect();

    // === Invariant: must never panic regardless of input. ===
    // Invalid parameter vectors (NaN, out-of-range, swapped setpoints) are
    // filtered out by `validate_parameters` and surface as `f64::NAN` entries
    // in the results vector rather than a panic.
    let result = oracle.evaluate_population(population.clone(), input.use_surrogates);

    let results = match result {
        Ok(r) => r,
        Err(_) => return,
    };

    // Length contract: one EUI per input candidate (NaN for filtered ones).
    assert_eq!(
        results.len(),
        population.len(),
        "evaluate_population must return one result per candidate"
    );

    // Finite-or-NaN only — never +/-Inf, which would indicate an unguarded
    // overflow in the physics or surrogate path.
    for (i, &eui) in results.iter().enumerate() {
        assert!(
            eui.is_nan() || eui.is_finite(),
            "non-finite EUI at index {}: {}",
            i,
            eui
        );
        // Valid (non-filtered) EUIs must be non-negative — energy consumption
        // is clamped via `.max(0.0)` in the analytical path; assert the
        // contract holds for the surrogate path too.
        if !eui.is_nan() {
            assert!(
                eui >= 0.0 || eui.is_nan(),
                "negative EUI at index {}: {}",
                i,
                eui
            );
        }
    }
});
