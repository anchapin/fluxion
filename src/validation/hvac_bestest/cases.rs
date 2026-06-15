//! HVAC BESTEST Test Case Definitions
//!
//! ASHRAE RP-865 HVAC BESTEST cases for validating airside equipment,
//! controls, and part-load performance.

use serde::{Deserialize, Serialize};

/// HVAC BESTEST case identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum HVACBestestCase {
    /// Equipment part-load performance test
    #[default]
    Case600,
    /// Equipment cycling losses test
    Case610,
    /// Control strategy validation test
    Case620,
    /// Zone temperature maintenance test
    Case630,
    /// Economizer performance test
    Case640,
}

/// Operating mode for HVAC BESTEST cases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatingMode {
    Heating,
    Cooling,
    Mixed,
}

/// Equipment type under test
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EquipmentType {
    Chiller,
    Boiler,
    HeatPump,
    VAV,
    CAV,
}

/// Test case definition for HVAC BESTEST
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HVACBestestCaseDefinition {
    /// Case identifier
    pub case_id: HVACBestestCase,
    /// Descriptive name
    pub name: String,
    /// Equipment type under test
    pub equipment_type: EquipmentType,
    /// Operating mode
    pub mode: OperatingMode,
    /// Rated capacity (W)
    pub rated_capacity: f64,
    /// Rated efficiency (COP or EER)
    pub rated_efficiency: f64,
    /// Design outdoor temperature (°C)
    pub design_outdoor_temp: f64,
    /// Minimum outdoor temperature for heating (°C)
    pub min_outdoor_temp_heating: f64,
    /// Maximum outdoor temperature for cooling (°C)
    pub max_outdoor_temp_cooling: f64,
    /// Reference energy consumption range (kWh)
    pub ref_energy_min: f64,
    pub ref_energy_max: f64,
    /// Reference peak demand range (W)
    pub ref_demand_min: f64,
    pub ref_demand_max: f64,
    /// Acceptable tolerance (%)
    pub tolerance_percent: f64,
}

impl Default for HVACBestestCaseDefinition {
    fn default() -> Self {
        Self {
            case_id: HVACBestestCase::Case600,
            name: String::new(),
            equipment_type: EquipmentType::Chiller,
            mode: OperatingMode::Cooling,
            rated_capacity: 100000.0,
            rated_efficiency: 4.5,
            design_outdoor_temp: 35.0,
            min_outdoor_temp_heating: -5.0,
            max_outdoor_temp_cooling: 45.0,
            ref_energy_min: 0.0,
            ref_energy_max: 0.0,
            ref_demand_min: 0.0,
            ref_demand_max: 0.0,
            tolerance_percent: 10.0,
        }
    }
}

/// Pre-defined ASHRAE RP-865 HVAC BESTEST cases
pub fn get_bestest_cases() -> Vec<HVACBestestCaseDefinition> {
    vec![
        // Case 600: Chiller part-load performance
        HVACBestestCaseDefinition {
            case_id: HVACBestestCase::Case600,
            name: "Chiller Part-Load Performance".to_string(),
            equipment_type: EquipmentType::Chiller,
            mode: OperatingMode::Cooling,
            rated_capacity: 100000.0, // 100 kW
            rated_efficiency: 4.5,    // COP 4.5
            design_outdoor_temp: 35.0,
            min_outdoor_temp_heating: -5.0,
            max_outdoor_temp_cooling: 45.0,
            // Reference data for annual energy (kWh)
            ref_energy_min: 25000.0,
            ref_energy_max: 35000.0,
            // Reference peak demand (W)
            ref_demand_min: 90000.0,
            ref_demand_max: 110000.0,
            tolerance_percent: 10.0,
        },
        // Case 610: Boiler part-load performance
        HVACBestestCaseDefinition {
            case_id: HVACBestestCase::Case610,
            name: "Boiler Part-Load Performance".to_string(),
            equipment_type: EquipmentType::Boiler,
            mode: OperatingMode::Heating,
            rated_capacity: 80000.0, // 80 kW
            rated_efficiency: 0.85,  // 85% thermal efficiency
            design_outdoor_temp: -5.0,
            min_outdoor_temp_heating: -20.0,
            max_outdoor_temp_cooling: 20.0,
            ref_energy_min: 30000.0,
            ref_energy_max: 45000.0,
            ref_demand_min: 70000.0,
            ref_demand_max: 90000.0,
            tolerance_percent: 10.0,
        },
        // Case 620: Heat pump performance
        HVACBestestCaseDefinition {
            case_id: HVACBestestCase::Case620,
            name: "Heat Pump Performance".to_string(),
            equipment_type: EquipmentType::HeatPump,
            mode: OperatingMode::Mixed,
            rated_capacity: 12000.0, // 12 kW heating
            rated_efficiency: 3.5,   // COP 3.5 heating
            design_outdoor_temp: -5.0,
            min_outdoor_temp_heating: -15.0,
            max_outdoor_temp_cooling: 40.0,
            ref_energy_min: 8000.0,
            ref_energy_max: 15000.0,
            ref_demand_min: 10000.0,
            ref_demand_max: 14000.0,
            tolerance_percent: 12.0,
        },
        // Case 630: VAV system performance
        HVACBestestCaseDefinition {
            case_id: HVACBestestCase::Case630,
            name: "VAV System Performance".to_string(),
            equipment_type: EquipmentType::VAV,
            mode: OperatingMode::Mixed,
            rated_capacity: 5000.0, // 5 kW
            rated_efficiency: 3.0,  // COP 3.0
            design_outdoor_temp: 35.0,
            min_outdoor_temp_heating: -5.0,
            max_outdoor_temp_cooling: 45.0,
            ref_energy_min: 5000.0,
            ref_energy_max: 8000.0,
            ref_demand_min: 4000.0,
            ref_demand_max: 6000.0,
            tolerance_percent: 15.0,
        },
        // Case 640: CAV system performance
        HVACBestestCaseDefinition {
            case_id: HVACBestestCase::Case640,
            name: "CAV System Performance".to_string(),
            equipment_type: EquipmentType::CAV,
            mode: OperatingMode::Mixed,
            rated_capacity: 10000.0, // 10 kW
            rated_efficiency: 3.2,   // COP 3.2
            design_outdoor_temp: 35.0,
            min_outdoor_temp_heating: -5.0,
            max_outdoor_temp_cooling: 45.0,
            ref_energy_min: 8000.0,
            ref_energy_max: 12000.0,
            ref_demand_min: 8000.0,
            ref_demand_max: 12000.0,
            tolerance_percent: 15.0,
        },
    ]
}

/// Reference data for ASHRAE RP-865 HVAC BESTEST
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HVACBestestReferenceData {
    /// Case identifier
    pub case_id: HVACBestestCase,
    /// Annual energy consumption (kWh)
    pub annual_energy_kwh: f64,
    /// Peak demand (W)
    pub peak_demand_w: f64,
    /// Part-load efficiency at 50% (COP)
    pub plr_50_cop: f64,
    /// Part-load efficiency at 75% (COP)
    pub plr_75_cop: f64,
    /// Part-load efficiency at 100% (COP)
    pub plr_100_cop: f64,
}

/// Get reference data for a specific case
pub fn get_reference_data(case_id: HVACBestestCase) -> Option<HVACBestestReferenceData> {
    match case_id {
        HVACBestestCase::Case600 => Some(HVACBestestReferenceData {
            case_id,
            annual_energy_kwh: 30000.0,
            peak_demand_w: 100000.0,
            plr_50_cop: 4.2,
            plr_75_cop: 4.4,
            plr_100_cop: 4.5,
        }),
        HVACBestestCase::Case610 => Some(HVACBestestReferenceData {
            case_id,
            annual_energy_kwh: 37500.0,
            peak_demand_w: 80000.0,
            plr_50_cop: 0.82,
            plr_75_cop: 0.84,
            plr_100_cop: 0.85,
        }),
        HVACBestestCase::Case620 => Some(HVACBestestReferenceData {
            case_id,
            annual_energy_kwh: 11500.0,
            peak_demand_w: 12000.0,
            plr_50_cop: 3.3,
            plr_75_cop: 3.4,
            plr_100_cop: 3.5,
        }),
        HVACBestestCase::Case630 => Some(HVACBestestReferenceData {
            case_id,
            annual_energy_kwh: 6500.0,
            peak_demand_w: 5000.0,
            plr_50_cop: 2.8,
            plr_75_cop: 2.9,
            plr_100_cop: 3.0,
        }),
        HVACBestestCase::Case640 => Some(HVACBestestReferenceData {
            case_id,
            annual_energy_kwh: 10000.0,
            peak_demand_w: 10000.0,
            plr_50_cop: 3.0,
            plr_75_cop: 3.1,
            plr_100_cop: 3.2,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_bestest_cases() {
        let cases = get_bestest_cases();
        assert_eq!(cases.len(), 5);
        assert_eq!(cases[0].case_id, HVACBestestCase::Case600);
        assert_eq!(cases[1].case_id, HVACBestestCase::Case610);
    }

    #[test]
    fn test_get_reference_data() {
        let ref_data = get_reference_data(HVACBestestCase::Case600);
        assert!(ref_data.is_some());
        let data = ref_data.unwrap();
        assert_eq!(data.plr_100_cop, 4.5);
    }

    #[test]
    fn test_case_ranges() {
        let cases = get_bestest_cases();
        for case in cases {
            assert!(
                case.ref_energy_min <= case.ref_energy_max,
                "Case {:?}: min > max",
                case.case_id
            );
            assert!(
                case.ref_demand_min <= case.ref_demand_max,
                "Case {:?}: min > max",
                case.case_id
            );
            assert!(
                case.tolerance_percent > 0.0 && case.tolerance_percent <= 100.0,
                "Case {:?}: invalid tolerance",
                case.case_id
            );
        }
    }
}
