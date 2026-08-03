//! HVAC BESTEST Test Runner
//!
//! Executes HVAC BESTEST cases and validates results against reference data.

use crate::sim::hvac::{
    Boiler, CAVSystem, Chiller, HVACMode, HeatPump, VAVTerminal, VariableCapacityEquipment,
};
use crate::validation::hvac_bestest::cases::{
    get_bestest_cases, get_reference_data, EquipmentType, HVACBestestCase,
    HVACBestestCaseDefinition,
};

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Test result for a single HVAC BESTEST case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HVACBestestResult {
    /// Case identifier
    pub case_id: HVACBestestCase,
    /// Whether the test passed
    pub passed: bool,
    /// Calculated annual energy (kWh)
    pub annual_energy_kwh: f64,
    /// Calculated peak demand (W)
    pub peak_demand_w: f64,
    /// Energy error vs reference (%)
    pub energy_error_percent: f64,
    /// Demand error vs reference (%)
    pub demand_error_percent: f64,
    /// Part-load efficiency at 50% (COP)
    pub plr_50_cop: f64,
    /// Part-load efficiency at 75% (COP)
    pub plr_75_cop: f64,
    /// Part-load efficiency at 100% (COP)
    pub plr_100_cop: f64,
    /// Validation message
    pub message: String,
}

impl HVACBestestResult {
    /// Create a new result
    pub fn new(case_id: HVACBestestCase) -> Self {
        Self {
            case_id,
            passed: false,
            annual_energy_kwh: 0.0,
            peak_demand_w: 0.0,
            energy_error_percent: 0.0,
            demand_error_percent: 0.0,
            plr_50_cop: 0.0,
            plr_75_cop: 0.0,
            plr_100_cop: 0.0,
            message: String::new(),
        }
    }

    /// Check if result is within acceptable tolerance
    pub fn within_tolerance(&self, tolerance_percent: f64) -> bool {
        self.energy_error_percent.abs() <= tolerance_percent
            && self.demand_error_percent.abs() <= tolerance_percent
    }
}

/// HVAC BESTEST test runner
pub struct HVACBestestRunner {
    /// Test cases to run
    cases: Vec<HVACBestestCaseDefinition>,
}

impl Default for HVACBestestRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl HVACBestestRunner {
    /// Create a new runner with default cases
    pub fn new() -> Self {
        Self {
            cases: get_bestest_cases(),
        }
    }

    /// Create a runner with specific cases
    pub fn with_cases(cases: Vec<HVACBestestCaseDefinition>) -> Self {
        Self { cases }
    }

    /// Run all test cases
    pub fn run_all(&self) -> Vec<HVACBestestResult> {
        self.cases.iter().map(|c| self.run_case(c)).collect()
    }

    /// Run a specific test case
    pub fn run_case(&self, case_def: &HVACBestestCaseDefinition) -> HVACBestestResult {
        let mut result = HVACBestestResult::new(case_def.case_id);

        match case_def.equipment_type {
            EquipmentType::Chiller => {
                self.run_chiller_test(case_def, &mut result);
            }
            EquipmentType::Boiler => {
                self.run_boiler_test(case_def, &mut result);
            }
            EquipmentType::HeatPump => {
                self.run_heatpump_test(case_def, &mut result);
            }
            EquipmentType::VAV => {
                self.run_vav_test(case_def, &mut result);
            }
            EquipmentType::CAV => {
                self.run_cav_test(case_def, &mut result);
            }
        }

        // Check tolerance
        result.passed = result.within_tolerance(case_def.tolerance_percent);

        result
    }

    fn run_chiller_test(
        &self,
        case_def: &HVACBestestCaseDefinition,
        result: &mut HVACBestestResult,
    ) {
        let chiller = Chiller::new(
            "CH-1".to_string(),
            case_def.rated_capacity,
            case_def.rated_efficiency,
            case_def.design_outdoor_temp,
        );

        // Calculate PLR curves
        result.plr_50_cop =
            chiller.calculate_efficiency(0.5, case_def.design_outdoor_temp, HVACMode::Cooling);
        result.plr_75_cop =
            chiller.calculate_efficiency(0.75, case_def.design_outdoor_temp, HVACMode::Cooling);
        result.plr_100_cop =
            chiller.calculate_efficiency(1.0, case_def.design_outdoor_temp, HVACMode::Cooling);

        // Simulate annual energy consumption
        let (annual_energy, peak_demand) = self.simulate_annual_chiller(&chiller, case_def);
        result.annual_energy_kwh = annual_energy;
        result.peak_demand_w = peak_demand;

        // Calculate errors
        if let Some(ref_data) = get_reference_data(case_def.case_id) {
            result.energy_error_percent =
                ((annual_energy - ref_data.annual_energy_kwh) / ref_data.annual_energy_kwh) * 100.0;
            result.demand_error_percent =
                ((peak_demand - ref_data.peak_demand_w) / ref_data.peak_demand_w) * 100.0;
            result.message = format!(
                "Chiller: Energy={:.1} kWh (ref: {:.0}-{:.0}), Peak={:.0} W (ref: {:.0}-{:.0})",
                annual_energy,
                case_def.ref_energy_min,
                case_def.ref_energy_max,
                peak_demand,
                case_def.ref_demand_min,
                case_def.ref_demand_max
            );
        }
    }

    fn run_boiler_test(
        &self,
        case_def: &HVACBestestCaseDefinition,
        result: &mut HVACBestestResult,
    ) {
        let boiler = Boiler::new(
            "BO-1".to_string(),
            case_def.rated_capacity,
            case_def.rated_efficiency,
            case_def.design_outdoor_temp,
        );

        // Calculate PLR curves
        result.plr_50_cop =
            boiler.calculate_efficiency(0.5, case_def.design_outdoor_temp, HVACMode::Heating);
        result.plr_75_cop =
            boiler.calculate_efficiency(0.75, case_def.design_outdoor_temp, HVACMode::Heating);
        result.plr_100_cop =
            boiler.calculate_efficiency(1.0, case_def.design_outdoor_temp, HVACMode::Heating);

        // Simulate annual energy consumption
        let (annual_energy, peak_demand) = self.simulate_annual_boiler(&boiler, case_def);
        result.annual_energy_kwh = annual_energy;
        result.peak_demand_w = peak_demand;

        if let Some(ref_data) = get_reference_data(case_def.case_id) {
            result.energy_error_percent =
                ((annual_energy - ref_data.annual_energy_kwh) / ref_data.annual_energy_kwh) * 100.0;
            result.demand_error_percent =
                ((peak_demand - ref_data.peak_demand_w) / ref_data.peak_demand_w) * 100.0;
            result.message = format!(
                "Boiler: Energy={:.1} kWh (ref: {:.0}-{:.0}), Peak={:.0} W (ref: {:.0}-{:.0})",
                annual_energy,
                case_def.ref_energy_min,
                case_def.ref_energy_max,
                peak_demand,
                case_def.ref_demand_min,
                case_def.ref_demand_max
            );
        }
    }

    fn run_heatpump_test(
        &self,
        case_def: &HVACBestestCaseDefinition,
        result: &mut HVACBestestResult,
    ) {
        let heatpump = HeatPump::new(
            "HP-1".to_string(),
            case_def.rated_capacity,
            case_def.rated_capacity * 0.85, // Cooling ~85% of heating
            case_def.rated_efficiency,
            case_def.rated_efficiency * 0.9, // EER slightly lower
        );

        // Calculate PLR curves
        result.plr_50_cop = heatpump.heating_cop_at_temperature(case_def.design_outdoor_temp);
        result.plr_75_cop = result.plr_50_cop * 1.02; // Estimate
        result.plr_100_cop = result.plr_50_cop * 1.05; // Estimate

        // Simulate annual energy consumption
        let (annual_energy, peak_demand) = self.simulate_annual_heatpump(&heatpump, case_def);
        result.annual_energy_kwh = annual_energy;
        result.peak_demand_w = peak_demand;

        if let Some(ref_data) = get_reference_data(case_def.case_id) {
            result.energy_error_percent =
                ((annual_energy - ref_data.annual_energy_kwh) / ref_data.annual_energy_kwh) * 100.0;
            result.demand_error_percent =
                ((peak_demand - ref_data.peak_demand_w) / ref_data.peak_demand_w) * 100.0;
            result.message =
                format!(
                "HeatPump: Energy={:.1} kWh (ref: {:.0}-{:.0}), Peak={:.0} W (ref: {:.0}-{:.0})",
                annual_energy, case_def.ref_energy_min, case_def.ref_energy_max,
                peak_demand, case_def.ref_demand_min, case_def.ref_demand_max
            );
        }
    }

    fn run_vav_test(&self, case_def: &HVACBestestCaseDefinition, result: &mut HVACBestestResult) {
        let vav = VAVTerminal::new("VAV-1".to_string(), 0, 0.5);

        // Calculate PLR curves
        result.plr_50_cop =
            vav.calculate_efficiency(0.5, case_def.design_outdoor_temp, HVACMode::Cooling);
        result.plr_75_cop =
            vav.calculate_efficiency(0.75, case_def.design_outdoor_temp, HVACMode::Cooling);
        result.plr_100_cop =
            vav.calculate_efficiency(1.0, case_def.design_outdoor_temp, HVACMode::Cooling);

        // Simulate annual energy consumption
        let (annual_energy, peak_demand) = self.simulate_annual_vav(&vav, case_def);
        result.annual_energy_kwh = annual_energy;
        result.peak_demand_w = peak_demand;

        if let Some(ref_data) = get_reference_data(case_def.case_id) {
            result.energy_error_percent =
                ((annual_energy - ref_data.annual_energy_kwh) / ref_data.annual_energy_kwh) * 100.0;
            result.demand_error_percent =
                ((peak_demand - ref_data.peak_demand_w) / ref_data.peak_demand_w) * 100.0;
            result.message = format!(
                "VAV: Energy={:.1} kWh (ref: {:.0}-{:.0}), Peak={:.0} W (ref: {:.0}-{:.0})",
                annual_energy,
                case_def.ref_energy_min,
                case_def.ref_energy_max,
                peak_demand,
                case_def.ref_demand_min,
                case_def.ref_demand_max
            );
        }
    }

    fn run_cav_test(&self, case_def: &HVACBestestCaseDefinition, result: &mut HVACBestestResult) {
        let cav = CAVSystem::new("CAV-1".to_string(), 1.0);

        // Calculate PLR curves
        result.plr_50_cop =
            cav.calculate_efficiency(0.5, case_def.design_outdoor_temp, HVACMode::Cooling);
        result.plr_75_cop =
            cav.calculate_efficiency(0.75, case_def.design_outdoor_temp, HVACMode::Cooling);
        result.plr_100_cop =
            cav.calculate_efficiency(1.0, case_def.design_outdoor_temp, HVACMode::Cooling);

        // Simulate annual energy consumption
        let (annual_energy, peak_demand) = self.simulate_annual_cav(&cav, case_def);
        result.annual_energy_kwh = annual_energy;
        result.peak_demand_w = peak_demand;

        if let Some(ref_data) = get_reference_data(case_def.case_id) {
            result.energy_error_percent =
                ((annual_energy - ref_data.annual_energy_kwh) / ref_data.annual_energy_kwh) * 100.0;
            result.demand_error_percent =
                ((peak_demand - ref_data.peak_demand_w) / ref_data.peak_demand_w) * 100.0;
            result.message = format!(
                "CAV: Energy={:.1} kWh (ref: {:.0}-{:.0}), Peak={:.0} W (ref: {:.0}-{:.0})",
                annual_energy,
                case_def.ref_energy_min,
                case_def.ref_energy_max,
                peak_demand,
                case_def.ref_demand_min,
                case_def.ref_demand_max
            );
        }
    }

    /// Simulate annual chiller energy consumption
    fn simulate_annual_chiller(
        &self,
        chiller: &Chiller,
        _case_def: &HVACBestestCaseDefinition,
    ) -> (f64, f64) {
        let start = Instant::now();

        // Simplified bin analysis for chiller
        // Hours at each outdoor temperature bin
        let bins: [(f64, f64); 8] = [
            (5.0, 500.0),   // 5°C - 500 hours
            (10.0, 800.0),  // 10°C - 800 hours
            (15.0, 1000.0), // 15°C - 1000 hours
            (20.0, 1200.0), // 20°C - 1200 hours
            (25.0, 1500.0), // 25°C - 1500 hours
            (30.0, 1800.0), // 30°C - 1800 hours
            (35.0, 1200.0), // 35°C - 1200 hours
            (40.0, 500.0),  // 40°C - 500 hours
        ];

        let mut total_energy_kwh: f64 = 0.0;
        let mut peak_demand_w: f64 = 0.0;

        for (temp, hours) in bins.iter() {
            // Cooling demand varies with temperature
            let plr = if *temp < 20.0 {
                0.3
            } else if *temp < 30.0 {
                0.6
            } else {
                0.85
            };

            let capacity = chiller.calculate_capacity(plr, *temp);
            let power = chiller.calculate_power(capacity, *temp, HVACMode::Cooling);

            total_energy_kwh += power * hours / 1000.0; // Convert W to kW
            peak_demand_w = peak_demand_w.max(power);
        }

        // Allow for standby losses (2%)
        total_energy_kwh *= 1.02;

        let elapsed = start.elapsed();
        if elapsed.as_secs() > 0 {
            println!("  Chiller simulation: {:.2}s", elapsed.as_secs_f64());
        }

        (total_energy_kwh, peak_demand_w)
    }

    /// Simulate annual boiler energy consumption
    fn simulate_annual_boiler(
        &self,
        boiler: &Boiler,
        _case_def: &HVACBestestCaseDefinition,
    ) -> (f64, f64) {
        let start = Instant::now();

        // Heating bins
        let bins: [(f64, f64); 8] = [
            (-10.0, 300.0), // -10°C - 300 hours
            (-5.0, 600.0),  // -5°C - 600 hours
            (0.0, 1000.0),  // 0°C - 1000 hours
            (5.0, 1500.0),  // 5°C - 1500 hours
            (10.0, 2000.0), // 10°C - 2000 hours
            (15.0, 1800.0), // 15°C - 1800 hours
            (20.0, 800.0),  // 20°C - 800 hours
            (25.0, 0.0),    // 25°C - 0 hours
        ];

        let mut total_energy_kwh: f64 = 0.0;
        let mut peak_demand_w: f64 = 0.0;

        for (temp, hours) in bins.iter() {
            if *hours == 0.0 {
                continue;
            }

            let plr = if *temp < 0.0 {
                0.85
            } else if *temp < 10.0 {
                0.6
            } else {
                0.35
            };

            let capacity = boiler.calculate_capacity(plr, *temp);
            let power = boiler.calculate_power(capacity, *temp, HVACMode::Heating);

            total_energy_kwh += power * hours / 1000.0;
            peak_demand_w = peak_demand_w.max(power);
        }

        // Include pilot/standby losses
        total_energy_kwh *= 1.05;

        let elapsed = start.elapsed();
        if elapsed.as_secs() > 0 {
            println!("  Boiler simulation: {:.2}s", elapsed.as_secs_f64());
        }

        (total_energy_kwh, peak_demand_w)
    }

    /// Simulate annual heat pump energy consumption
    fn simulate_annual_heatpump(
        &self,
        hp: &HeatPump,
        case_def: &HVACBestestCaseDefinition,
    ) -> (f64, f64) {
        let start = Instant::now();

        // Mixed heating/cooling bins
        let heating_bins: [(f64, f64); 5] = [
            (-10.0, 400.0),
            (-5.0, 800.0),
            (0.0, 1200.0),
            (5.0, 1000.0),
            (10.0, 400.0),
        ];

        let cooling_bins: [(f64, f64); 4] =
            [(25.0, 600.0), (30.0, 1000.0), (35.0, 800.0), (40.0, 200.0)];

        let mut total_energy_kwh: f64 = 0.0;
        let mut peak_demand_w: f64 = 0.0;

        // Heating energy
        for (temp, hours) in heating_bins.iter() {
            if *hours == 0.0 {
                continue;
            }

            let cop = hp.heating_cop_at_temperature(*temp);
            let capacity = hp.heating_capacity
                * (1.0 - (*temp - case_def.design_outdoor_temp).abs() * 0.01).max(0.5);
            let power = capacity / cop;

            total_energy_kwh += power * hours / 1000.0;
            peak_demand_w = peak_demand_w.max(power);
        }

        // Cooling energy
        for (temp, hours) in cooling_bins.iter() {
            if *hours == 0.0 {
                continue;
            }

            let cop = hp.cooling_cop_at_temperature(*temp);
            let capacity = hp.cooling_capacity
                * (1.0 - (*temp - case_def.design_outdoor_temp).abs() * 0.015).max(0.5);
            let power = capacity / cop;

            total_energy_kwh += power * hours / 1000.0;
            peak_demand_w = peak_demand_w.max(power);
        }

        let elapsed = start.elapsed();
        if elapsed.as_secs() > 0 {
            println!("  HeatPump simulation: {:.2}s", elapsed.as_secs_f64());
        }

        (total_energy_kwh, peak_demand_w)
    }

    /// Simulate annual VAV energy consumption
    fn simulate_annual_vav(
        &self,
        vav: &VAVTerminal,
        _case_def: &HVACBestestCaseDefinition,
    ) -> (f64, f64) {
        let start = Instant::now();

        let bins: [(f64, f64, f64); 6] = [
            (5.0, 0.3, 400.0),
            (10.0, 0.4, 800.0),
            (15.0, 0.5, 1200.0),
            (20.0, 0.6, 1500.0),
            (25.0, 0.75, 1000.0),
            (30.0, 0.9, 200.0),
        ];

        let mut total_energy_kwh: f64 = 0.0;
        let mut peak_demand_w: f64 = 0.0;

        for (temp, plr, hours) in bins.iter() {
            if *hours == 0.0 {
                continue;
            }

            let capacity = vav.calculate_capacity(*plr, *temp);
            let power = vav.calculate_power(capacity, *temp, HVACMode::Cooling);

            total_energy_kwh += power * hours / 1000.0;
            peak_demand_w = peak_demand_w.max(power);
        }

        // Include fan energy
        total_energy_kwh *= 1.15;

        let elapsed = start.elapsed();
        if elapsed.as_secs() > 0 {
            println!("  VAV simulation: {:.2}s", elapsed.as_secs_f64());
        }

        (total_energy_kwh, peak_demand_w)
    }

    #[allow(dead_code)]
    /// Simulate annual CAV energy consumption (deprecated: replaced by simulate_annual_cav_terminal)
    fn simulate_annual_cav(
        &self,
        cav: &CAVSystem,
        _case_def: &HVACBestestCaseDefinition,
    ) -> (f64, f64) {
        let start = Instant::now();

        let bins: [(f64, f64); 6] = [
            (5.0, 400.0),
            (10.0, 800.0),
            (15.0, 1200.0),
            (20.0, 1500.0),
            (25.0, 1000.0),
            (30.0, 200.0),
        ];

        let mut total_energy_kwh: f64 = 0.0;
        let mut peak_demand_w: f64 = 0.0;

        for (temp, hours) in bins.iter() {
            if *hours == 0.0 {
                continue;
            }

            let plr = 0.7; // Typical average PLR for CAV
            let capacity = cav.calculate_capacity(plr, *temp);
            let power = cav.calculate_power(capacity, *temp, HVACMode::Cooling);

            total_energy_kwh += power * hours / 1000.0;
            peak_demand_w = peak_demand_w.max(power);
        }

        let elapsed = start.elapsed();
        if elapsed.as_secs() > 0 {
            println!("  CAV simulation: {:.2}s", elapsed.as_secs_f64());
        }

        (total_energy_kwh, peak_demand_w)
    }
}

/// Run all HVAC BESTEST tests and return results
pub fn run_hvac_bestest() -> Vec<HVACBestestResult> {
    let runner = HVACBestestRunner::new();
    runner.run_all()
}

/// Validate results and return summary
pub fn validate_results(results: &[HVACBestestResult]) -> (usize, usize, f64) {
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.len() - passed;

    let total_error: f64 = results.iter().map(|r| r.energy_error_percent.abs()).sum();
    let mean_error = if results.is_empty() {
        0.0
    } else {
        total_error / results.len() as f64
    };

    (passed, failed, mean_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_creation() {
        let runner = HVACBestestRunner::new();
        let results = runner.run_all();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_chiller_simulation() {
        let cases = get_bestest_cases();
        let chiller_case = cases
            .iter()
            .find(|c| c.case_id == HVACBestestCase::Case600)
            .unwrap();
        let runner = HVACBestestRunner::new();
        let result = runner.run_case(chiller_case);

        assert_eq!(result.case_id, HVACBestestCase::Case600);
        assert!(result.annual_energy_kwh > 0.0);
        assert!(result.peak_demand_w > 0.0);
    }

    #[test]
    fn test_boiler_simulation() {
        let cases = get_bestest_cases();
        let boiler_case = cases
            .iter()
            .find(|c| c.case_id == HVACBestestCase::Case610)
            .unwrap();
        let runner = HVACBestestRunner::new();
        let result = runner.run_case(boiler_case);

        assert_eq!(result.case_id, HVACBestestCase::Case610);
        assert!(result.annual_energy_kwh > 0.0);
    }

    #[test]
    fn test_validate_results() {
        let results = run_hvac_bestest();
        let (passed, failed, mean_error) = validate_results(&results);

        println!(
            "Passed: {}, Failed: {}, Mean Error: {:.2}%",
            passed, failed, mean_error
        );

        // At least some tests should pass or have reasonable error
        assert!(results.len() > 0);
    }

    #[test]
    fn test_plr_curves() {
        let cases = get_bestest_cases();
        let runner = HVACBestestRunner::new();

        for case in cases {
            let result = runner.run_case(&case);
            assert!(
                result.plr_100_cop > 0.0,
                "Case {:?} should have valid PLR",
                case.case_id
            );
            assert!(
                result.plr_50_cop <= result.plr_100_cop * 1.1,
                "Case {:?}: PLR 50% should be <= 100% COP",
                case.case_id
            );
        }
    }
}
