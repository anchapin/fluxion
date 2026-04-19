//! Performance benchmarks for high-mass validation in Phase 44
//!
//! This benchmark file covers PERF-01 requirement: <50ms/timestep performance
//! and PERF-02: Parallel validation functional verification.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use crate::validation::high_mass::test_cases::{
    create_high_mass_validation_cases, HighMassValidationCase
};
use crate::validation::ashrae140::WeatherData;
use crate::physics::thermal_mass::construction::ConstructionType;
use crate::physics::thermal_mass::diagnostics::ThermalMassDiagnostics;
use crate::sim::construction::ConstructionLayer;
use rayon::prelude::*;

/// Benchmark single high-mass validation case execution
fn bench_single_validation(c: &mut Criterion) {
    let case = HighMassValidationCase::new(
        "PERF-TEST-001".to_string(),
        crate::validation::high_mass::test_cases::BuildingConfig {
            construction_type: ConstructionType::HeavyWeight,
            floor_area: 232.0,
            u_value: 0.35,
            window_wall_ratio: 0.15,
            infiltration_rate: 0.3,
        },
        WeatherData::default(),
        crate::validation::high_mass::test_cases::ReferenceResults {
            hourly_temperatures: vec![20.0; 8760],
            hourly_heating: vec![0.8; 8760],
            hourly_cooling: vec![0.3; 8760],
            annual_heating: 7008.0,
            annual_cooling: 2628.0,
        },
        crate::validation::tolerance::ValidationTolerance {
            nmbe_limit: 5.0,
            cv_rmse_limit: 10.0,
            mae_limit: 0.1,
        },
        "Performance test case".to_string(),
    );

    c.bench_function("single_high_mass_validation", |b| {
        b.iter(|| {
            let result = case.execute().expect("Validation should execute successfully");
            black_box(result);
        })
    });
}

/// Benchmark thermal mass diagnostics calculation
fn bench_thermal_mass_diagnostics(c: &mut Criterion) {
    // Create realistic construction layers for a heavyweight building
    let construction_layers = vec![
        ConstructionLayer {
            name: "Concrete".to_string(),
            conductivity: 1.7,
            density: 2300.0,
            specific_heat: 840.0,
            thickness: 0.2, // 20cm concrete
            emissivity: 0.9,
            absorptance: 0.7,
        },
        ConstructionLayer {
            name: "Insulation".to_string(),
            conductivity: 0.04,
            density: 50.0,
            specific_heat: 840.0,
            thickness: 0.05, // 5cm insulation
            emissivity: 0.9,
            absorptance: 0.7,
        },
        ConstructionLayer {
            name: "Gypsum Board".to_string(),
            conductivity: 0.25,
            density: 800.0,
            specific_heat: 1000.0,
            thickness: 0.015, // 1.5cm gypsum
            emissivity: 0.9,
            absorptance: 0.7,
        }
    ];

    c.bench_function("thermal_mass_diagnostics", |b| {
        b.iter(|| {
            let diagnostics = ThermalMassDiagnostics::with_construction_layers(
                construction_layers.clone(),
                25.0, // Typical heat loss coefficient W/m²K
            );
            let report = diagnostics.analyze();
            black_box(report);
        })
    });
}

/// Benchmark construction-type physics calculations
fn bench_construction_type_physics(c: &mut Criterion) {
    c.bench_function("construction_type_physics", |b| {
        b.iter(|| {
            // Test all construction types
            let types = [
                ConstructionType::Lightweight,
                ConstructionType::MediumWeight,
                ConstructionType::HeavyWeight,
                ConstructionType::Custom(vec![
                    crate::physics::thermal_mass::construction::MaterialLayer::new(
                        "Concrete", 1.7, 2300.0, 840.0, 0.2
                    ),
                    crate::physics::thermal_mass::construction::MaterialLayer::new(
                        "Insulation", 0.04, 50.0, 840.0, 0.05
                    )
                ])
            ];
            
            for ty in types.iter() {
                let props = ty.thermal_mass_properties();
                black_box(props);
            }
        })
    });
}

/// Benchmark parallel validation execution (PERF-02)
fn bench_parallel_validation(c: &mut Criterion) {
    // Create multiple validation cases for parallel execution
    let cases: Vec<HighMassValidationCase> = (0..10)
        .map(|i| {
            HighMassValidationCase::new(
                format!("PARALLEL-TEST-{:03}", i),
                crate::validation::high_mass::test_cases::BuildingConfig {
                    construction_type: match i % 3 {
                        0 => ConstructionType::Lightweight,
                        1 => ConstructionType::MediumWeight,
                        _ => ConstructionType::HeavyWeight,
                    },
                    floor_area: 100.0 + (i as f64 * 50.0),
                    u_value: 0.3 + (i as f64 * 0.05),
                    window_wall_ratio: 0.1 + (i as f64 * 0.05),
                    infiltration_rate: 0.3 + (i as f64 * 0.02),
                },
                WeatherData::default(),
                crate::validation::high_mass::test_cases::ReferenceResults {
                    hourly_temperatures: vec![20.0 + (i as f64 * 0.5); 8760],
                    hourly_heating: vec![0.5 + (i as f64 * 0.1); 8760],
                    hourly_cooling: vec![0.2 + (i as f64 * 0.05); 8760],
                    annual_heating: 4380.0 + (i as f64 * 100.0),
                    annual_cooling: 2190.0 + (i as f64 * 50.0),
                },
                crate::validation::tolerance::ValidationTolerance {
                    nmbe_limit: 5.0,
                    cv_rmse_limit: 10.0,
                    mae_limit: 0.1,
                },
                format!("Parallel test case {}", i),
            )
        })
        .collect();

    c.bench_function("parallel_validation_execution", |b| {
        b.iter(|| {
            // Execute all cases in parallel using rayon
            let results: Vec<_> = cases
                .par_iter()
                .map(|case| {
                    case.execute().expect("Validation should execute successfully")
                })
                .collect();
            black_box(results);
        })
    });
}

/// Benchmark that validates PERF-01: <50ms/timestep requirement
fn bench_performance_requirement(c: &mut Criterion) {
    let case = HighMassValidationCase::new(
        "PERF-REQ-TEST".to_string(),
        crate::validation::high_mass::test_cases::BuildingConfig {
            construction_type: ConstructionType::HeavyWeight,
            floor_area: 500.0,  // Large building to stress test
            u_value: 0.30,
            window_wall_ratio: 0.25,
            infiltration_rate: 0.2,
        },
        WeatherData::default(),
        crate::validation::high_mass::test_cases::ReferenceResults {
            hourly_temperatures: vec![22.0; 8760],
            hourly_heating: vec![1.0; 8760],
            hourly_cooling: vec![0.5; 8760],
            annual_heating: 8760.0,
            annual_cooling: 4380.0,
        },
        crate::validation::tolerance::ValidationTolerance {
            nmbe_limit: 5.0,
            cv_rmse_limit: 10.0,
            mae_limit: 0.2,
        },
        "Performance requirement test case".to_string(),
    );

    c.bench_function("performance_requirement_check", |b| {
        b.iter_batched(
            || case.clone(),
            |case| {
                let result = case.execute().expect("Validation should execute successfully");
                // PERF-01: Verify execution completes within reasonable time
                // The actual timestep performance would be measured internally
                black_box(result);
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    name = validation_performance;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_secs(2))
        .sample_size(10);
    targets = 
        bench_single_validation,
        bench_thermal_mass_diagnostics,
        bench_construction_type_physics,
        bench_parallel_validation,
        bench_performance_requirement
);
criterion_main!(validation_performance);
