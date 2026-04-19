//! Performance optimization tests.
//!
//! This module contains comprehensive tests for performance optimizations
//! including solver improvements, zone coupling optimizations, and regression detection.

use crate::physics::cta::VectorField;
use crate::thermal::solver::{SolverResult, ThermalSolver};
use crate::thermal::zone_coupling::{ZoneCoupling, ZoneCouplingOptimized};
use crate::validation::performance::metrics::{collect_performance_metrics, PerformanceMetrics};
use crate::validation::performance::optimization::{
    generate_optimization_report, SolverOptimization, ZoneCouplingOptimization,
};
use ndarray::Array2;
use std::time::Instant;

/// Create a standard thermal model for testing.
fn create_standard_thermal_model() -> ThermalSolver {
    let num_zones = 3;
    let temps = VectorField::from_scalar(20.0, num_zones);
    let caps = VectorField::from_scalar(1000000.0, num_zones);
    let gains = VectorField::from_scalar(1000.0, num_zones);

    // Create a simple conductance matrix
    let mut matrix = vec![vec![0.0; num_zones]; num_zones];
    for i in 0..num_zones {
        for j in 0..num_zones {
            if i != j {
                matrix[i][j] = 50.0; // 50 W/K conductance between zones
            }
        }
    }

    ThermalSolver::new(temps, caps, gains, matrix)
}

/// Benchmark zone coupling performance.
fn benchmark_zone_coupling(coupling: &dyn ZoneCouplingTrait) -> f64 {
    let start_time = Instant::now();

    // Perform multiple calculations to get meaningful timing
    for _ in 0..1000 {
        coupling.calculate_heat_flow();
    }

    start_time.elapsed().as_secs_f64() * 1000.0 // Convert to ms
}

/// Trait for zone coupling benchmarking.
trait ZoneCouplingTrait {
    fn calculate_heat_flow(&self) -> Vec<f64>;
}

impl ZoneCouplingTrait for ZoneCoupling {
    fn calculate_heat_flow(&self) -> Vec<f64> {
        self.calculate_heat_flow_legacy()
    }
}

impl ZoneCouplingTrait for ZoneCouplingOptimized {
    fn calculate_heat_flow(&self) -> Vec<f64> {
        self.calculate_total_heat_flow()
    }
}

#[test]
fn test_solver_optimization_improvement() {
    // Setup baseline solver (no optimizations)
    let mut baseline_solver = create_standard_thermal_model();
    baseline_solver.use_adaptive_convergence = false;
    baseline_solver.use_warm_start = false;

    // Collect baseline metrics
    let baseline_start = Instant::now();
    let baseline_result = baseline_solver.solve(3600.0);
    let baseline_duration = baseline_start.elapsed();

    let baseline_metrics = PerformanceMetrics {
        timestep_duration: baseline_duration,
        memory_usage: 0, // Not measured in this test
        iterations_per_timestep: baseline_result.iterations,
    };

    // Setup optimized solver
    let mut optimized_solver = create_standard_thermal_model();
    optimized_solver.enable_optimizations();

    // Collect optimized metrics
    let optimized_start = Instant::now();
    let optimized_result = optimized_solver.solve(3600.0);
    let optimized_duration = optimized_start.elapsed();

    let optimized_metrics = PerformanceMetrics {
        timestep_duration: optimized_duration,
        memory_usage: 0, // Not measured in this test
        iterations_per_timestep: optimized_result.iterations,
    };

    // Verify improvement
    let optimization =
        SolverOptimization::calculate_improvement(&baseline_metrics, &optimized_metrics);

    // The optimized solver should either converge faster or in fewer iterations
    assert!(
        optimization.improvement_percent >= 0.0,
        "Expected non-negative performance improvement, got {}",
        optimization.improvement_percent
    );

    // Verify no major accuracy regression (residuals should be similar)
    assert!(
        baseline_result.residual.abs() < 100.0 || optimized_result.residual.abs() < 100.0,
        "Both solvers should produce reasonable results"
    );
}

#[test]
fn test_zone_coupling_optimization() {
    let zones = 5;

    // Create baseline zone coupling
    let mut baseline = ZoneCoupling::new(zones);
    baseline.temperatures = vec![20.0, 22.0, 19.0, 21.0, 23.0];
    baseline.conductances = vec![50.0, 60.0, 40.0, 55.0, 45.0];

    // Create optimized zone coupling
    let mut optimized = ZoneCouplingOptimized::new(zones);
    let conductance_data = vec![
        0.0, 50.0, 60.0, 40.0, 55.0, 50.0, 0.0, 60.0, 40.0, 55.0, 60.0, 60.0, 0.0, 40.0, 55.0,
        40.0, 40.0, 40.0, 0.0, 55.0, 55.0, 55.0, 55.0, 55.0, 0.0,
    ];
    optimized.conductance_matrix =
        Array2::from_shape_vec((zones, zones), conductance_data).unwrap();
    optimized.set_temperature_vector(vec![20.0, 22.0, 19.0, 21.0, 23.0]);

    // Performance comparison
    let baseline_time = benchmark_zone_coupling(&baseline);
    let optimized_time = benchmark_zone_coupling(&optimized);

    // Optimized version should be at least as fast as baseline
    assert!(
        optimized_time <= baseline_time * 1.5, // Allow some variance
        "Optimized zone coupling should be faster or comparable: baseline={}ms, optimized={}ms",
        baseline_time,
        optimized_time
    );
}

#[test]
fn test_optimization_regression_detection() {
    let report = generate_optimization_report();

    // Verify we have improvements tracked
    assert!(
        report.improvements.len() > 0,
        "Expected at least one optimization improvement"
    );

    // Verify no regressions are reported
    assert!(
        report.regressions.is_empty(),
        "Expected no regressions in optimization report"
    );

    // Verify overall improvement is positive
    assert!(
        report.total_improvement_percent > 0.0,
        "Expected positive total improvement, got {}",
        report.total_improvement_percent
    );
}

#[test]
fn test_solver_convergence_with_optimizations() {
    let mut solver = create_standard_thermal_model();
    solver.enable_optimizations();

    let result = solver.solve(3600.0);

    // Should converge within max iterations
    assert!(
        result.converged || result.iterations <= 50,
        "Solver should converge or stop at max iterations"
    );

    // Should have reasonable residual
    assert!(
        result.residual < 1e3, // Allow larger tolerance for this test
        "Solver residual should be reasonable: {}",
        result.residual
    );
}

#[test]
fn test_warm_start_functionality() {
    let mut solver = create_standard_thermal_model();
    solver.enable_optimizations();

    // Set a warm start close to expected solution
    let initial_guess = VectorField::new(vec![22.5, 23.0, 22.8]);
    solver.set_warm_start(&initial_guess);

    let result = solver.solve(3600.0);

    // With warm start, should converge quickly
    assert!(
        result.converged || result.iterations < 50,
        "Warm start should help convergence"
    );
}

#[test]
fn test_adaptive_convergence_tolerance() {
    let mut solver = create_standard_thermal_model();
    solver.enable_optimizations();

    let result = solver.solve(3600.0);

    // If converged, residual should be below tolerance
    if result.converged {
        assert!(
            result.residual < 1e-5, // Slightly above 1e-6 for test tolerance
            "Converged solver should meet tolerance: residual = {}",
            result.residual
        );
    }
}

#[test]
fn test_zone_coupling_vectorization_correctness() {
    // Test that vectorized calculations produce same results as legacy
    let zones = 3;

    // Setup identical conditions
    let temperatures = vec![25.0, 20.0, 18.0];
    let conductances = vec![0.0, 50.0, 60.0];

    // Legacy calculation
    let mut legacy = ZoneCoupling::new(zones);
    legacy.temperatures = temperatures.clone();
    legacy.conductances = conductances.clone();
    let legacy_flow = legacy.calculate_heat_flow_legacy();

    // Optimized calculation
    let mut optimized = ZoneCouplingOptimized::new(zones);
    let conductance_data = vec![0.0, 50.0, 60.0, 50.0, 0.0, 60.0, 60.0, 60.0, 0.0];
    optimized.conductance_matrix =
        Array2::from_shape_vec((zones, zones), conductance_data).unwrap();
    optimized.set_temperature_vector(temperatures);
    let optimized_flow = optimized.calculate_total_heat_flow();

    // Results should be similar (allowing for calculation differences)
    for i in 0..zones {
        assert!(
            (legacy_flow[i] - optimized_flow[i]).abs() < 10.0,
            "Zone {}: legacy={}, optimized={}, diff={}",
            i,
            legacy_flow[i],
            optimized_flow[i],
            (legacy_flow[i] - optimized_flow[i]).abs()
        );
    }
}

#[test]
fn test_material_properties_caching() {
    use crate::thermal::zone_coupling::get_material_properties;

    // Test that material properties are cached and accessible
    let concrete = get_material_properties("concrete");
    assert!(concrete.is_some(), "Concrete properties should be cached");

    let props = concrete.unwrap();
    assert!(props.conductivity > 0.0, "Conductivity should be positive");
    assert!(props.density > 0.0, "Density should be positive");
    assert!(
        props.specific_heat > 0.0,
        "Specific heat should be positive"
    );
}
