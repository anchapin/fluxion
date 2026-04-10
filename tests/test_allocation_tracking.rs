//! Allocation tracking tests for hot loop performance (BUG-05, PERF-01).
//!
//! These tests use the `dhat` heap profiler to measure allocations during
//! thermal simulation. They establish baseline allocation counts that should
//! improve after optimization.
//!
//! Run with: cargo test test_allocation_tracking --release -- --nocapture

// Set dhat as the global allocator for this test binary
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Helper: create a simple 1-zone model with default parameters.
fn create_single_zone_model() -> ThermalModel<VectorField> {
    let mut model = ThermalModel::<VectorField>::new(1);
    // Set reasonable defaults
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    // Initialize temperatures
    model.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass_temperatures = VectorField::from_scalar(20.0, 1);
    model
}

/// Test: Measure allocation count for a single 1-year simulation.
///
/// This test runs the hot loop (8760 timesteps) with analytical physics.
/// It reports total allocation count and allocated bytes.
///
/// Baseline target (before optimization): < 10,000 allocations per config per year
/// (to be determined by actual measurement)
#[test]
fn test_allocation_count_single_model() {
    let _profiler = dhat::Profiler::new_heap();

    let mut model = create_single_zone_model();
    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Run 1 year (8760 steps)
    let _energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);

    // dhat will automatically print allocation summary when _profiler drops.
    println!("Completed single model allocation tracking test.");
}

/// Test: Measure allocation count for batch evaluation of 1000 configs.
///
/// This uses BatchOracle to evaluate a population of 1000 configurations in parallel.
/// It measures total allocations and computes average per config.
///
/// Baseline target: Should scale linearly and not have pathological allocations.
#[test]
fn test_allocation_count_batch_1000() {
    let _profiler = dhat::Profiler::new_heap();

    // Build BatchOracle
    let base_model = create_single_zone_model();
    let oracle = fluxion::BatchOracle::from_model(base_model);

    // Generate synthetic population: [window_u_value, heating_setpoint, cooling_setpoint]
    let mut rng = StdRng::seed_from_u64(42);
    let population: Vec<Vec<f64>> = (0..1000)
        .map(|_| {
            vec![
                rng.gen_range(0.1..5.0),        // U-value
                18.0 + rng.gen_range(0.0..7.0), // heating setpoint (18-25)
                22.0 + rng.gen_range(0.0..8.0), // cooling setpoint (22-30)
            ]
        })
        .collect();

    // Evaluate with analytical physics (no surrogates)
    let results = oracle
        .evaluate_population(population, false)
        .expect("Batch evaluation failed");

    assert_eq!(results.len(), 1000);

    // dhat will automatically print allocation summary when _profiler drops.
    println!("Completed batch allocation tracking test (1000 configs).");
}
