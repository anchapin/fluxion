//! High-mass validation module.
//!
//! This module provides comprehensive validation capabilities
//! for high-mass building energy simulations.

pub mod metrics;
pub mod reports;
pub mod test_cases;

use crate::physics::thermal_mass::construction::ConstructionType;

pub use metrics::HighMassMetrics;
pub use reports::{CombinedHighMassReport, HighMassSummary, HighMassValidationReport};
pub use test_cases::HighMassValidationCase;

/// Run all high-mass validation cases and return results
pub fn run_all_high_mass_cases() -> Vec<crate::validation::report::ValidationResult> {
    use crate::validation::high_mass::test_cases::create_high_mass_validation_cases;

    let cases = create_high_mass_validation_cases();
    let mut results = Vec::new();

    for case in cases {
        match case.execute() {
            Ok(result) => results.push(result),
            Err(e) => {
                eprintln!("Error executing case {}: {}", case.case_id, e);
                // Create a failed result
                results.push(crate::validation::report::ValidationResult {
                    case_id: case.case_id,
                    metric: crate::validation::report::MetricType::AnnualHeating,
                    fluxion_value: 0.0,
                    ref_min: 0.0,
                    ref_max: 0.0,
                    percent_error: 0.0,
                    status: crate::validation::report::ValidationStatus::Fail,
                    per_program: None,
                });
            }
        }
    }

    results
}

/// Generate a combined validation report from results
pub fn generate_combined_report(
    results: &[crate::validation::report::ValidationResult],
) -> CombinedHighMassReport {
    // This is a simplified version - in a real implementation, we would
    // convert ValidationResult to HighMassValidationReport with full metrics
    // For now, we'll create a basic combined report

    let case_reports = Vec::new(); // Would be populated with actual reports
    let summary = HighMassSummary::from_reports(&case_reports);

    CombinedHighMassReport {
        case_reports,
        summary,
    }
}

/// Validate construction type and return appropriate construction properties
pub fn validate_construction_type(
    construction: &crate::sim::construction::Construction,
) -> ConstructionType {
    // Analyze construction layers to determine construction type
    // This is a simplified version - real implementation would analyze
    // material properties and thicknesses

    let total_thickness: f64 = construction
        .layers
        .iter()
        .map(|layer| layer.thickness)
        .sum();
    let avg_density: f64 = construction
        .layers
        .iter()
        .map(|layer| layer.density)
        .sum::<f64>()
        / construction.layers.len() as f64;

    // Simple heuristic based on thickness and density
    if total_thickness > 0.3 && avg_density > 1500.0 {
        ConstructionType::HeavyWeight
    } else if total_thickness > 0.2 || avg_density > 1000.0 {
        ConstructionType::MediumWeight
    } else {
        ConstructionType::Lightweight
    }
}
