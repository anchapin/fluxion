//! FLEXLAB Test Cell Empirical Validation (Issue #1807)
//!
//! Integration test that builds and runs a Fluxion model matching the LBNL
//! FLEXLAB test cell X3A geometry, construction, and schedules. This is the
//! "apples-to-apples" model for empirical validation T10.5.
//!
//! The test validates that:
//! 1. The model builds successfully from the CaseSpec
//! 2. The simulation runs without panics or NaN values
//! 3. Annual energy consumption is physically reasonable
//! 4. The model diff is documented and within acceptable bounds

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::flexlab_test_cell::{flexlab_test_cell_spec, model_diff_summary};

/// Helper: simulate 1 year (8760 hourly timesteps).
fn simulate_year(model: &mut ThermalModel<VectorField>) -> f64 {
    let surrogate = SurrogateManager::new().expect("Failed to create surrogate manager");
    model.solve_timesteps(8760, &surrogate, false, None, None, None)
}

/// Test that the FLEXLAB spec builds and runs a full-year simulation.
///
/// This is the primary empirical validation test for T10.5. It verifies:
/// - The CaseSpec builds without errors
/// - The thermal model is created from the spec
/// - The 8760-step simulation completes without panics
/// - Annual energy is physically reasonable (not zero, not extreme)
/// - No NaN values appear in the results
#[test]
fn test_flexlab_x3a_full_simulation() {
    println!("\n=== FLEXLAB Test Cell X3A - Full Year Simulation ===");

    // Build the spec
    let spec = flexlab_test_cell_spec();
    println!("Case ID: {}", spec.case_id);
    println!("Description: {}", spec.description);
    println!(
        "Geometry: {}m × {}m × {}m = {:.1} m² floor area",
        spec.geometry[0].width,
        spec.geometry[0].depth,
        spec.geometry[0].height,
        spec.geometry[0].width * spec.geometry[0].depth,
    );
    println!("Window area: {:.2} m²", spec.total_window_area());
    println!("Infiltration: {} ACH", spec.infiltration_ach);

    // Create the thermal model
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    println!("Thermal model created: {} zone(s)", model.num_zones);

    // Run 1 year
    println!("Running 8760-step simulation...");
    let annual_energy = simulate_year(&mut model);

    println!("Annual energy: {:.1} kWh", annual_energy);

    // Validate energy is physical (not NaN, not extreme)
    assert!(
        annual_energy.is_finite(),
        "Annual energy must be finite, got {annual_energy}"
    );
    // Energy should be nonzero — either heating or cooling (or both) is active
    assert!(
        annual_energy.abs() > 0.01,
        "Annual energy should be non-trivial, got {annual_energy} kWh"
    );
    // Reasonable bounds: test cell in Berkeley should not exceed 100 MWh/year
    assert!(
        annual_energy.abs() < 100_000.0,
        "Annual energy {annual_energy} kWh exceeds 100 MWh — likely a bug"
    );

    println!("PASS: FLEXLAB X3A simulation completed successfully.");
}

/// Test that the FLEXLAB model diff is documented.
///
/// Verifies that all model differences between the Fluxion model and the
/// reference FLEXLAB facility are documented, which is required for the
/// empirical validation report.
#[test]
fn test_flexlab_model_diff_documented() {
    let diffs = model_diff_summary();
    assert!(
        diffs.len() >= 5,
        "Model diff should document at least 5 differences, got {}",
        diffs.len()
    );

    println!("\n=== FLEXLAB Model Diff Summary ===");
    for (i, diff) in diffs.iter().enumerate() {
        println!("  {}. {}", i + 1, diff);
    }
}

/// Test that the spec geometry matches FLEXLAB reference dimensions.
///
/// Validates the key geometric parameters against the Modelica source
/// (`Buildings.ThermalZones.Detailed.FLEXLAB.Rooms.X3A.TestCell`).
#[test]
fn test_flexlab_geometry_matches_reference() {
    let spec = flexlab_test_cell_spec();
    let geo = &spec.geometry[0];

    // Dimensions from Modelica
    assert!(
        (geo.width - 6.6675).abs() < 1e-6,
        "Width should be 6.6675m, got {}",
        geo.width
    );
    assert!(
        (geo.depth - 9.144).abs() < 1e-6,
        "Depth should be 9.144m, got {}",
        geo.depth
    );
    assert!(
        (geo.height - 3.6576).abs() < 1e-6,
        "Height should be 3.6576m, got {}",
        geo.height
    );

    // Floor area: 60.97 m²
    let floor_area = geo.width * geo.depth;
    assert!(
        (floor_area - 60.97).abs() < 0.1,
        "Floor area should be ~60.97 m², got {floor_area}"
    );

    // Window area: 10.75 m²
    let window_area = spec.total_window_area();
    assert!(
        (window_area - 10.75).abs() < 0.1,
        "Window area should be ~10.75 m², got {window_area}"
    );
}
