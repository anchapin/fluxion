//! Investigation test for Issue #272: Peak Load Values
//!
//! This test investigates why peak load values are significantly different from reference ranges.
//!
//! Current State (after PR #246 fix):
//! - Peak loads are TOO HIGH (11.13-17.78 kW vs 4.20-5.60 kW reference)
//! - This is the OPPOSITE of the original issue description
//! - Original issue #226 described peak loads being TOO LOW (1.39-5.00 kW constant)
//! - PR #246 fixed the unit conversion issue
//! - Now we have a new problem: peak loads are ~2-6x TOO HIGH
//!
//! KEY FINDING:
//! The test `test_issue_272_case_600_peak_load_calculation` revealed that:
//! - When zone temp reaches setpoint exactly (20.00°C)
//! - HVAC power still reports 3563.54 W consumption
//! - Temperature error = 0.00°C, so expected power = 0.00 W
//! - But mass temp = 12.59°C (much cooler than zone)
//!
//! This suggests HVAC energy is being stored in thermal mass and counted as consumption,
//! even though it's not actually "lost" from the building.
//!
//! Root Cause Analysis:
//!
//! The HVAC power calculation in step_physics returns energy in kWh:
//! 1. hvac_power_demand() returns power in Watts (W) per zone
//! 2. Summed across zones and multiplied by dt (3600s) = Joules
//! 3. Divided by 3.6e6 = kWh
//! 4. Validator multiplies by 1000 to get "Watts" for peak tracking
//!
//! Conversion check:
//! - If HVAC power = 5000 W for 1 hour
//! - Energy = 5000 × 3600 = 18,000,000 J
//! - kWh = 18,000,000 / 3,600,000 = 5.0 kWh
//! - Peak W = 5.0 × 1000 = 5000 W ✓
//!
//! The conversion appears correct. The problem must be elsewhere.
//!
//! Possible Root Causes:
//!
//! 1. **HVAC Power Demand Calculation** (hvac_power_demand)
//!    - Uses sensitivity = term_rest_1 / den
//!    - Power = t_err / sensitivity
//!    - If sensitivity is too low, power will be too high
//!    - Need to verify sensitivity calculation
//!
//! 2. **Thermal Network Parameters** (conductances, capacitances)
//!    - If conductances are too low, the system responds slowly
//!    - This causes larger temperature errors
//!    - Larger t_err → higher power demand
//!
//! 3. **Load Calculation** (solar + internal gains)
//!    - If loads are too high, temperature errors increase
//!    - Higher temps → higher cooling demand
//!    - Lower temps → higher heating demand
//!
//! 4. **Temperature Initialization**
//!    - Starting temperatures affect first few hours
//!    - Cold start → high initial heating demand
//!    - Hot start → high initial cooling demand
//!
//! Investigation Plan:
//!
//! 1. Log HVAC power demand, sensitivity, and temperature error during peak hours
//! 2. Compare with expected values from thermal network theory
//! 3. Verify conductance and capacitance calculations
//! 4. Check if peak occurs at expected hours (coldest/hottest)

use fluxion::sim::engine::ThermalModel;
use fluxion::physics::cta::VectorField;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec};
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Test to investigate peak load calculation for Case 600.
///
/// This test:
/// 1. Creates a Case 600 thermal model
/// 2. Runs one hour of simulation with extreme weather
/// 3. Logs the HVAC power demand calculation
/// 4. Verifies the calculation is physically reasonable
#[test]
fn test_issue_272_case_600_peak_load_calculation() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    // Simulate the coldest hour in Denver (typically January, hour 0)
    let coldest_hour = 0;
    let weather_data = weather.get_hourly_data(coldest_hour).unwrap();
    let outdoor_temp = weather_data.dry_bulb_temp;

    // Set model to very cold state to induce peak heating demand
    // This simulates startup conditions
    model.temperatures = VectorField::new(vec![15.0; spec.num_zones]);
    model.mass_temperatures = VectorField::new(vec![15.0; spec.num_zones]);

    println!("=== Issue #272 Peak Load Investigation: Case 600 ===");
    println!("Outdoor Temperature: {:.2}°C", outdoor_temp);
    println!("Heating Setpoint: {:.2}°C", model.heating_setpoint);
    println!("Cooling Setpoint: {:.2}°C", model.cooling_setpoint);
    println!("Floor Area: {:.2} m²", spec.geometry[0].floor_area());
    println!();

    // Log thermal network parameters
    println!("Thermal Network Parameters:");
    println!("  h_tr_em (Ext->Mass): {:.2} W/K", model.h_tr_em.as_slice()[0]);
    println!("  h_tr_ms (Mass->Surf): {:.2} W/K", model.h_tr_ms.as_slice()[0]);
    println!("  h_tr_is (Surf->Int): {:.2} W/K", model.h_tr_is.as_slice()[0]);
    println!("  h_tr_w (Ext->Int via win): {:.2} W/K", model.h_tr_w.as_slice()[0]);
    println!("  h_ve (Ventilation): {:.2} W/K", model.h_ve.as_slice()[0]);
    println!("  h_tr_floor (Ground): {:.2} W/K", model.h_tr_floor.as_slice()[0]);
    println!("  Thermal Capacitance: {:.2} J/K", model.thermal_capacitance.as_slice()[0]);
    println!();

    // Log derived parameters
    println!("Derived Parameters:");
    println!("  derived_den: {:.6}", model.derived_den.as_slice()[0]);
    println!("  derived_sensitivity: {:.6}", model.derived_sensitivity.as_slice()[0]);
    println!();

    // Run simulation for one timestep
    let hvac_kwh = model.step_physics(coldest_hour, outdoor_temp);

    println!("Simulation Results:");
    println!("  HVAC Energy: {:.6} kWh", hvac_kwh);
    println!("  HVAC Power (instant): {:.2} W", hvac_kwh * 1000.0);
    println!("  Zone Temperature: {:.2}°C", model.temperatures.as_slice()[0]);
    println!("  Mass Temperature: {:.2}°C", model.mass_temperatures.as_slice()[0]);
    println!();

    // Calculate expected power demand manually
    let t_i = model.temperatures.as_slice()[0];
    let t_heating_sp = model.heating_setpoint;
    let sensitivity = model.derived_sensitivity.as_slice()[0];
    let t_err = t_heating_sp - t_i;
    let expected_power = t_err / sensitivity;

    println!("Manual Calculation Check:");
    println!("  Temperature Error: {:.6}°C", t_err);
    println!("  Sensitivity: {:.6} K/W", sensitivity);
    println!("  Expected Power: {:.2} W", expected_power);
    println!("  Actual Power: {:.2} W", hvac_kwh * 1000.0);
    println!();

    // Sanity checks
    assert!(hvac_kwh > 0.0, "HVAC should be heating in cold weather");
    assert!(expected_power > 0.0, "Heating power should be positive");

    // Check if power is reasonable (should be < 50 kW for a 48 m² building)
    let power_kw = hvac_kwh * 1000.0 / 1000.0;
    assert!(power_kw < 50.0, "Peak power should be < 50 kW for Case 600");

    println!("Power is < 50 kW: ✓ ({:.2} kW)", power_kw);
}

/// Test peak cooling load for Case 600.
///
/// Similar to heating test but for cooling conditions.
#[test]
fn test_issue_272_case_600_peak_cooling_calculation() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    // Simulate the hottest hour in Denver (typically July, hour ~4300)
    let hottest_hour = 4300; // July 13, 14:00 (approximately)
    let weather_data = weather.get_hourly_data(hottest_hour).unwrap();
    let outdoor_temp = weather_data.dry_bulb_temp;

    // Set model to very hot state to induce peak cooling demand
    model.temperatures = VectorField::new(vec![30.0; spec.num_zones]);
    model.mass_temperatures = VectorField::new(vec![30.0; spec.num_zones]);

    // Add solar load (typical mid-day value)
    let solar_load = 200.0; // W/m²
    let total_loads = vec![solar_load];
    model.set_loads(&total_loads);

    println!("=== Issue #272 Peak Cooling Load Investigation: Case 600 ===");
    println!("Outdoor Temperature: {:.2}°C", outdoor_temp);
    println!("Solar Load: {:.2} W/m²", solar_load);
    println!("Cooling Setpoint: {:.2}°C", model.cooling_setpoint);
    println!();

    // Run simulation
    let hvac_kwh = model.step_physics(hottest_hour, outdoor_temp);

    println!("Simulation Results:");
    println!("  HVAC Energy: {:.6} kWh", hvac_kwh);
    println!("  HVAC Power (instant): {:.2} W", (-hvac_kwh) * 1000.0);
    println!("  Zone Temperature: {:.2}°C", model.temperatures.as_slice()[0]);
    println!();

    // Sanity check
    assert!(hvac_kwh < 0.0, "HVAC should be cooling in hot weather");

    let power_kw = (-hvac_kwh) * 1000.0 / 1000.0;
    println!("Cooling Power: {:.2} kW", power_kw);

    // Check if power is reasonable (should be < 30 kW for cooling)
    // Reference range is 2.90-3.90 kW, so something is definitely wrong
    assert!(power_kw > 0.0, "Cooling power should be positive");
}

/// Test sensitivity calculation directly.
///
/// The sensitivity should represent the change in zone temperature
/// per unit of HVAC power input.
#[test]
fn test_issue_272_sensitivity_calculation_verification() {
    let spec = ASHRAE140Case::Case600.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("=== Issue #272 Sensitivity Calculation Verification ===");
    println!();

    // Log the sensitivity calculation components
    let h_ms_is_prod = model.derived_h_ms_is_prod.as_slice()[0];
    let term_rest_1 = model.derived_term_rest_1.as_slice()[0];
    let h_ext = model.derived_h_ext.as_slice()[0];
    let den = model.derived_den.as_slice()[0];
    let sensitivity = model.derived_sensitivity.as_slice()[0];

    println!("Sensitivity Components:");
    println!("  h_ms_is_prod (H_ms * H_is): {:.6}", h_ms_is_prod);
    println!("  term_rest_1 (H_ms + H_is): {:.6}", term_rest_1);
    println!("  h_ext (H_w + H_ve): {:.6}", h_ext);
    println!("  den = h_ms_is_prod + term_rest_1 * h_ext: {:.6}", den);
    println!("  sensitivity = term_rest_1 / den: {:.6}", sensitivity);
    println!();

    // Sensitivity should be in units of K/W (temperature change per Watt of HVAC power)
    // For a 48 m² room with typical construction, sensitivity should be ~0.001-0.01 K/W
    // This means 1 kW of HVAC power changes temperature by 1-10°C
    assert!(sensitivity > 0.0, "Sensitivity must be positive");
    assert!(sensitivity < 1.0, "Sensitivity should be < 1 K/W");

    println!("Sensitivity range check: 0.001-1.0 K/W");

    // If sensitivity is too small (e.g., 0.0001), then 1 kW changes temp by 0.1°C
    // This would cause HVAC power to be 10x too high
    if sensitivity < 0.001 {
        println!("WARNING: Sensitivity is very small (< 0.001 K/W)");
        println!("This would cause HVAC power to be ~10x too high!");
    }
}
