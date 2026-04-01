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
pub fn export_component_csv(entries: &[ComponentEntry], path: &Path) -> Result<()> {
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
        assert!(!result.is_valid);
    }

    #[test]
    fn test_conservation_zero_input() {
        let breakdown = EnergyBreakdown {
            solar_gains_mwh: 0.0,
            internal_gains_mwh: 0.0,
            heating_mwh: 0.0,
            cooling_mwh: 0.0,
            envelope_conduction_mwh: 0.0,
            infiltration_mwh: 0.0,
            net_balance_mwh: 0.0,
        };
        let result = check_conservation(&breakdown, 1.0);
        assert!(result.is_valid);
        assert!((result.net_balance_mwh - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_multiple_cases() {
        let b1 = EnergyBreakdown {
            envelope_conduction_mwh: 1.0,
            infiltration_mwh: 0.5,
            solar_gains_mwh: 2.0,
            internal_gains_mwh: 1.0,
            heating_mwh: 3.0,
            cooling_mwh: 1.5,
            net_balance_mwh: 0.0,
        };
        let b2 = EnergyBreakdown {
            envelope_conduction_mwh: 2.0,
            infiltration_mwh: 1.0,
            solar_gains_mwh: 3.0,
            internal_gains_mwh: 1.5,
            heating_mwh: 4.0,
            cooling_mwh: 2.5,
            net_balance_mwh: 0.0,
        };
        let entries = aggregate_from_validator(
            vec![("600".to_string(), b1), ("900".to_string(), b2)].into_iter(),
        );
        assert_eq!(entries.len(), 12);
        let cooling_900 = entries
            .iter()
            .find(|e| e.component == "cooling" && e.case_id == "900")
            .unwrap();
        assert_eq!(cooling_900.energy_mwh, 2.5);
    }

    #[test]
    fn test_component_entry_clone() {
        let entry = ComponentEntry {
            case_id: "600".to_string(),
            component: "heating".to_string(),
            energy_mwh: 5.0,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.case_id, entry.case_id);
        assert_eq!(cloned.component, entry.component);
        assert_eq!(cloned.energy_mwh, entry.energy_mwh);
    }

    #[test]
    fn test_conservation_result_fields() {
        let breakdown = EnergyBreakdown {
            solar_gains_mwh: 10.0,
            internal_gains_mwh: 5.0,
            heating_mwh: 8.0,
            cooling_mwh: 7.0,
            envelope_conduction_mwh: 0.0,
            infiltration_mwh: 0.0,
            net_balance_mwh: 0.0,
        };
        let result = check_conservation(&breakdown, 5.0);
        assert!((result.net_balance_mwh - 0.0).abs() < 1e-6);
        assert!((result.tolerance_pct - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_export_csv_to_temp_file() {
        let entries = vec![
            ComponentEntry {
                case_id: "600".to_string(),
                component: "heating".to_string(),
                energy_mwh: 5.0,
            },
            ComponentEntry {
                case_id: "600".to_string(),
                component: "cooling".to_string(),
                energy_mwh: 3.0,
            },
        ];

        let temp_path = std::env::temp_dir().join("fluxion_test_components.csv");
        let result = export_component_csv(&entries, &temp_path);
        assert!(result.is_ok());

        let content = std::fs::read_to_string(&temp_path).unwrap();
        assert!(content.contains("Case,Component,Energy_MWh"));
        assert!(content.contains("600,heating,5.0000"));
        assert!(content.contains("600,cooling,3.0000"));

        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_conservation_result_clone() {
        let result = ConservationResult {
            net_balance_mwh: 0.5,
            tolerance_pct: 1.0,
            is_valid: true,
        };
        let cloned = result.clone();
        assert_eq!(cloned.net_balance_mwh, result.net_balance_mwh);
        assert!(cloned.is_valid);
    }
}
