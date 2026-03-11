use crate::validation::diagnostic::EnergyBreakdown;
use anyhow::Result;
use csv::Writer;
use serde::Serialize;
use std::path::Path;

/// Component entry for aggregated energy breakdown.
#[derive(Debug, Clone, Serialize)]
pub struct ComponentEntry {
    pub case_id: String,
    pub component: String,
    pub energy_mwh: f64,
}

/// Aggregate energy breakdowns from an iterator over (case_id, EnergyBreakdown).
pub fn aggregate_from_validator<I>(iter: I) -> Vec<ComponentEntry>
where
    I: Iterator<Item = (String, EnergyBreakdown)>,
{
    let mut entries = Vec::new();
    for (case_id, breakdown) in iter {
        entries.push(ComponentEntry {
            case_id: case_id.clone(),
            component: "envelope_conduction".to_string(),
            energy_mwh: breakdown.envelope_conduction_mwh,
        });
        entries.push(ComponentEntry {
            case_id: case_id.clone(),
            component: "infiltration".to_string(),
            energy_mwh: breakdown.infiltration_mwh,
        });
        entries.push(ComponentEntry {
            case_id: case_id.clone(),
            component: "solar_gains".to_string(),
            energy_mwh: breakdown.solar_gains_mwh,
        });
        entries.push(ComponentEntry {
            case_id: case_id.clone(),
            component: "internal_gains".to_string(),
            energy_mwh: breakdown.internal_gains_mwh,
        });
        entries.push(ComponentEntry {
            case_id: case_id.clone(),
            component: "heating".to_string(),
            energy_mwh: breakdown.heating_mwh,
        });
        entries.push(ComponentEntry {
            case_id,
            component: "cooling".to_string(),
            energy_mwh: breakdown.cooling_mwh,
        });
    }
    entries
}

/// Export component entries to a CSV file.
pub fn export_component_csv(
    entries: &[ComponentEntry],
    path: &Path,
) -> Result<()> {
    let mut wtr = Writer::from_path(path)?;
    wtr.write_record(&["Case", "Component", "Energy_MWh"])?;
    for entry in entries {
        wtr.write_record(&[
            &entry.case_id,
            &entry.component,
            &format!("{:.4}", entry.energy_mwh),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

/// Conservation check result.
#[derive(Debug, Clone, Serialize)]
pub struct ConservationResult {
    pub net_balance_mwh: f64,
    pub tolerance_pct: f64,
    pub is_valid: bool,
}

/// Check energy conservation: net balance should be near zero within tolerance.
pub fn check_conservation(breakdown: &EnergyBreakdown, tolerance_pct: f64) -> ConservationResult {
    let net = breakdown.solar_gains_mwh + breakdown.internal_gains_mwh
        - breakdown.heating_mwh
        - breakdown.cooling_mwh;
    let total_input = breakdown.solar_gains_mwh + breakdown.internal_gains_mwh;
    let tolerance = if total_input > 0.0 {
        total_input * (tolerance_pct / 100.0)
    } else {
        // If no input, tolerance absolute (e.g., 0.01 MWh)
        0.01
    };
    let is_valid = net.abs() <= tolerance;
    ConservationResult {
        net_balance_mwh: net,
        tolerance_pct,
        is_valid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_and_export() {
        let breakdown = EnergyBreakdown {
            envelope_conduction_mwh: 2.5,
            infiltration_mwh: 1.0,
            solar_gains_mwh: 3.0,
            internal_gains_mwh: 1.5,
            heating_mwh: 5.0,
            cooling_mwh: 2.0,
            net_balance_mwh: 0.0,
        };
        let entries = aggregate_from_validator(vec![("600".to_string(), breakdown)].into_iter());
        assert_eq!(entries.len(), 6);
        // Find heating entry
        let heating_entry = entries
            .iter()
            .find(|e| e.component == "heating" && e.case_id == "600")
            .unwrap();
        assert_eq!(heating_entry.energy_mwh, 5.0);
    }

    #[test]
    fn test_conservation() {
        let balanced = EnergyBreakdown {
            solar_gains_mwh: 4.0,
            internal_gains_mwh: 1.0,
            heating_mwh: 3.0,
            cooling_mwh: 2.0,
            envelope_conduction_mwh: 0.0,
            infiltration_mwh: 0.0,
            net_balance_mwh: 0.0,
        };
        let result = check_conservation(&balanced, 1.0);
        // net = 4+1-3-2 = 0, should be valid
        assert!(result.is_valid);
    }

    #[test]
    fn test_conservation_fail() {
        let unbalanced = EnergyBreakdown {
            solar_gains_mwh: 4.0,
            internal_gains_mwh: 1.0,
            heating_mwh: 3.5,
            cooling_mwh: 2.0,
            envelope_conduction_mwh: 0.0,
            infiltration_mwh: 0.0,
            net_balance_mwh: 0.0,
        };
        let result = check_conservation(&unbalanced, 1.0);
        // net = -0.5, tolerance ~0.05, so invalid
        assert!(!result.is_valid);
    }
}
