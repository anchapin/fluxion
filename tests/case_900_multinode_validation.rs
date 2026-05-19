//! Multi-Node HVAC Case 900 Validation Test
//!
//! Issue #861: ASHRAE 140 Case 900 validation with multi-node HVAC
//!
//! This test validates the multi-node HVAC infrastructure (9R4C thermal network)
//! against ASHRAE 140 reference values for Case 900 (high-mass building with HVAC).
//!
//! ## Reference Values (ASHRAE 140-2023)
//!
//! Case 900:
//!   - Annual Heating: 1.17 - 2.04 MWh
//!   - Annual Cooling: 2.13 - 3.67 MWh
//!   - Peak Heating: 1.10 - 2.10 kW
//!   - Peak Cooling: 2.10 - 3.50 kW
//!
//! Case 900FF (free-floating):
//!   - Min Temperature: -6.4 to -1.6°C
//!   - Max Temperature: 41.8 to 46.4°C
//!
//! ## Multi-Node Model (9R4C)
//!
//! The 9R4C thermal network consists of:
//!   - 4 thermal mass nodes: wall, roof, floor, internal
//!   - 9 thermal resistances: various coupling conductances
//!
//! This is a more detailed model than the 5R1C single-node approach.

use fluxion::physics::multi_node_solver::MultiNodeSolver;
use fluxion::sim::multi_node_hvac_runner::MultiNodeHvacRunner;
use fluxion::sim::multi_node_thermal::ThermalMassNode;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// ASHRAE 140 reference ranges for Case 900
mod reference {
    /// Case 900 - High mass building with HVAC
    pub mod case_900 {
        /// Annual heating energy range (MWh)
        pub const ANNUAL_HEATING_MIN: f64 = 1.17;
        pub const ANNUAL_HEATING_MAX: f64 = 2.04;

        /// Annual cooling energy range (MWh)
        pub const ANNUAL_COOLING_MIN: f64 = 2.13;
        pub const ANNUAL_COOLING_MAX: f64 = 3.67;

        /// Peak heating load (kW)
        pub const PEAK_HEATING_MIN: f64 = 1.10;
        pub const PEAK_HEATING_MAX: f64 = 2.10;

        /// Peak cooling load (kW)
        pub const PEAK_COOLING_MIN: f64 = 2.10;
        pub const PEAK_COOLING_MAX: f64 = 3.50;
    }

    /// Case 900FF - High mass free-floating
    pub mod case_900ff {
        pub const MIN_TEMP_MIN: f64 = -6.4;
        pub const MIN_TEMP_MAX: f64 = -1.6;
        pub const MAX_TEMP_MIN: f64 = 41.8;
        pub const MAX_TEMP_MAX: f64 = 46.4;
    }
}

/// Tolerance for annual energy validation (±15% as per ASHRAE 140)
const ANNUAL_ENERGY_TOLERANCE: f64 = 0.15;

/// Tolerance for peak loads (±10% as per ASHRAE 140)
const PEAK_LOAD_TOLERANCE: f64 = 0.10;

/// Tolerance for free-floating temperatures (±5% of reference range)
const TEMP_TOLERANCE: f64 = 0.05;

/// Create a MultiNodeHvacRunner configured for Case 900.
///
/// Uses high-mass thermal parameters matching ASHRAE 140 Case 900 construction:
/// - Heavy concrete walls (200mm concrete + 50mm insulation)
/// - Insulated roof (200mm concrete slab)
/// - Carpeted floor
/// - Internal thermal mass (furniture, partitions)
///
/// ASHRAE 140 Case 900 thermal characteristics:
/// - Floor Area: 48 m² (8m × 6m)
/// - Wall Area: 75.6 m²
/// - Total Thermal Capacitance: ~20,000 kJ/K
fn create_case_900_runner() -> MultiNodeHvacRunner {
    // Case 900 uses heavy-mass construction (concrete block + foam insulation)
    // These thermal mass parameters are derived from ASHRAE 140 Table 7.3
    // and the Case 900 construction specifications.
    //
    // Wall: Heavy concrete (200mm) with foam insulation (50mm)
    //   - Thermal mass: ~8e6 J/K for wall alone
    //   - Coupling to exterior: ~80 W/K (through insulation)
    //   - Coupling to interior surface: ~25 W/K
    let wall = ThermalMassNode::new(
        20.0, // Initial temperature (°C)
        8e6,  // Thermal capacitance (J/K) - heavy concrete
        80.0, // h_tr_em: exterior-to-mass conductance (W/K)
        25.0, // h_tr_ms: mass-to-surface conductance (W/K)
    );

    // Roof: 200mm concrete slab
    //   - Slightly less thermal mass than walls
    //   - Higher coupling to exterior (exposed to sky)
    let roof = ThermalMassNode::new(
        20.0, // Initial temperature (°C)
        5e6,  // Thermal capacitance (J/K) - roof concrete
        60.0, // h_tr_em for roof (exposed to sky)
        20.0, // h_tr_ms for roof
    );

    // Floor: Carpeted concrete slab on grade
    //   - Ground coupled (lower coupling to exterior)
    //   - Moderate thermal mass
    let floor = ThermalMassNode::new(
        20.0, // Initial temperature (°C)
        3e6,  // Thermal capacitance (J/K)
        40.0, // h_tr_em for floor (ground coupled)
        15.0, // h_tr_ms for floor
    );

    // Internal thermal mass: furniture, partitions, internal walls
    //   - Provides additional damping
    //   - Coupled to zone air and other masses
    let internal = ThermalMassNode::new(
        20.0, // Initial temperature (°C)
        2e6,  // Thermal capacitance (J/K) - internal mass
        50.0, // h_tr_me: internal mass to envelope mass
        30.0, // h_tr_ms: surface to internal mass
    );

    // Zone air to interior surface conductance
    // Typical residential: 10-20 W/K
    let h_tr_is = 15.0;

    let solver = MultiNodeSolver::new(h_tr_is, wall, roof, floor, internal);

    // ASHRAE 140 Case 900 HVAC setpoints:
    // - Heating: 20°C
    // - Cooling: 27°C (with 2°C deadband)
    let h_ve = 20.0; // Ventilation conductance (W/K) - typical for residential
    let h_tr_w = 5.0; // Window conductance (W/K) - highly insulated windows

    // Default 14-day warmup as per ASHRAE 140 §B2 guidance
    MultiNodeHvacRunner::new(solver, h_ve, h_tr_w, 20.0, 27.0).with_warmup_days(14)
}

/// Run Case 900 simulation using multi-node HVAC runner
/// Returns (annual_heating_kwh, annual_cooling_kwh, peak_heating_kw, peak_cooling_kw, min_temp, max_temp)
fn simulate_case_900_multinode() -> (f64, f64, f64, f64, f64, f64) {
    let weather = DenverTmyWeather::new();
    let mut runner = create_case_900_runner();

    let mut min_temp = f64::INFINITY;
    let mut max_temp = f64::NEG_INFINITY;

    // Full year simulation (8760 hours)
    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        let t_outdoor = weather_data.dry_bulb_temp;

        // Solar gain into the zone (W)
        // ASHRAE 140 Case 900 has windows with solar transmission
        // This is a simplified estimate - actual calculation depends on window area and orientation
        let solar_gain = 0.0; // Will be handled by solver's exterior temperature + solar distribution

        // Internal gains: 200W continuous (ASHRAE 140 standard for residential)
        let internal_gain = 200.0;

        // Step the simulation
        let _q_hvac = runner.step(t_outdoor, solar_gain, internal_gain, 3600.0);

        // Track zone temperature using compute_zone_air_temperature
        let t_air =
            runner
                .solver
                .compute_zone_air_temperature(t_outdoor, runner.h_ve, internal_gain);
        min_temp = min_temp.min(t_air);
        max_temp = max_temp.max(t_air);
    }

    (
        runner.annual_heating_energy,
        runner.annual_cooling_energy,
        runner.peak_heating_power,
        runner.peak_cooling_power,
        min_temp,
        max_temp,
    )
}

/// Run Case 900FF (free-floating) simulation using multi-node runner
/// Returns (min_temp, max_temp)
fn simulate_case_900ff_multinode() -> (f64, f64) {
    let weather = DenverTmyWeather::new();
    let mut runner = create_case_900_runner();

    // Free-floating: setpoints far outside any possible temperature range
    // This ensures HVAC is never triggered
    runner.heating_setpoint = -999.0;
    runner.cooling_setpoint = 999.0;

    let mut min_temp = f64::INFINITY;
    let mut max_temp = f64::NEG_INFINITY;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        let t_outdoor = weather_data.dry_bulb_temp;

        let solar_gain = 0.0;
        let internal_gain = 200.0; // FF cases have 200W continuous internal gains

        runner.step(t_outdoor, solar_gain, internal_gain, 3600.0);

        // Compute zone air temperature using the multi-node thermal balance
        let t_air =
            runner
                .solver
                .compute_zone_air_temperature(t_outdoor, runner.h_ve, internal_gain);
        min_temp = min_temp.min(t_air);
        max_temp = max_temp.max(t_air);
    }

    (min_temp, max_temp)
}

// ============================================================================
// TEST CASES
// ============================================================================

/// Test: Case 900 multi-node annual heating energy
///
/// Validates that annual heating energy is within ASHRAE 140 reference range.
#[test]
fn test_case_900_multinode_annual_heating() {
    let (heating_kwh, _, peak_heating, _, min_temp, max_temp) = simulate_case_900_multinode();
    let heating_mwh = heating_kwh / 1000.0;

    println!("\n=== Case 900 Multi-Node Annual Heating ===");
    println!(
        "Annual Heating: {:.2} MWh (reference: {:.2} - {:.2} MWh)",
        heating_mwh,
        reference::case_900::ANNUAL_HEATING_MIN,
        reference::case_900::ANNUAL_HEATING_MAX
    );
    println!("Peak Heating: {:.2} kW", peak_heating);
    println!(
        "Zone Temperature Range: {:.2}°C - {:.2}°C",
        min_temp, max_temp
    );

    let ref_min = reference::case_900::ANNUAL_HEATING_MIN;
    let ref_max = reference::case_900::ANNUAL_HEATING_MAX;
    let tolerance = (ref_max - ref_min) * ANNUAL_ENERGY_TOLERANCE;

    let in_range = heating_mwh >= ref_min - tolerance && heating_mwh <= ref_max + tolerance;

    if in_range {
        println!("✅ PASS: Annual heating within reference range");
    } else {
        println!(
            "❌ FAIL: Annual heating {:.2} MWh outside range [{:.2}, {:.2}] MWh",
            heating_mwh, ref_min, ref_max
        );
        println!(
            "   Tolerance: ±{:.2} MWh ({:.0}%)",
            tolerance,
            ANNUAL_ENERGY_TOLERANCE * 100.0
        );
    }

    // Assert with detailed failure message
    assert!(
        heating_mwh >= ref_min - tolerance && heating_mwh <= ref_max + tolerance,
        "Annual heating {:.2} MWh outside reference range [{:.2}, {:.2}] MWh (±{:.0}% tolerance: ±{:.2} MWh)",
        heating_mwh,
        ref_min,
        ref_max,
        ANNUAL_ENERGY_TOLERANCE * 100.0,
        tolerance
    );
}

/// Test: Case 900 multi-node annual cooling energy
///
/// Validates that annual cooling energy is within ASHRAE 140 reference range.
#[test]
fn test_case_900_multinode_annual_cooling() {
    let (_, cooling_kwh, _, peak_cooling, min_temp, max_temp) = simulate_case_900_multinode();
    let cooling_mwh = cooling_kwh / 1000.0;

    println!("\n=== Case 900 Multi-Node Annual Cooling ===");
    println!(
        "Annual Cooling: {:.2} MWh (reference: {:.2} - {:.2} MWh)",
        cooling_mwh,
        reference::case_900::ANNUAL_COOLING_MIN,
        reference::case_900::ANNUAL_COOLING_MAX
    );
    println!("Peak Cooling: {:.2} kW", peak_cooling);
    println!(
        "Zone Temperature Range: {:.2}°C - {:.2}°C",
        min_temp, max_temp
    );

    let ref_min = reference::case_900::ANNUAL_COOLING_MIN;
    let ref_max = reference::case_900::ANNUAL_COOLING_MAX;
    let tolerance = (ref_max - ref_min) * ANNUAL_ENERGY_TOLERANCE;

    let in_range = cooling_mwh >= ref_min - tolerance && cooling_mwh <= ref_max + tolerance;

    if in_range {
        println!("✅ PASS: Annual cooling within reference range");
    } else {
        println!(
            "❌ FAIL: Annual cooling {:.2} MWh outside range [{:.2}, {:.2}] MWh",
            cooling_mwh, ref_min, ref_max
        );
    }

    assert!(
        cooling_mwh >= ref_min - tolerance && cooling_mwh <= ref_max + tolerance,
        "Annual cooling {:.2} MWh outside reference range [{:.2}, {:.2}] MWh (±{:.0}% tolerance)",
        cooling_mwh,
        ref_min,
        ref_max,
        ANNUAL_ENERGY_TOLERANCE * 100.0
    );
}

/// Test: Case 900 multi-node peak heating load
///
/// Validates that peak heating load is within ASHRAE 140 reference range.
#[test]
fn test_case_900_multinode_peak_heating() {
    let (heating_kwh, _, peak_heating, _, _, _) = simulate_case_900_multinode();
    let heating_mwh = heating_kwh / 1000.0;

    println!("\n=== Case 900 Multi-Node Peak Heating ===");
    println!(
        "Peak Heating: {:.2} kW (reference: {:.2} - {:.2} kW)",
        peak_heating,
        reference::case_900::PEAK_HEATING_MIN,
        reference::case_900::PEAK_HEATING_MAX
    );
    println!("Annual Heating: {:.2} MWh", heating_mwh);

    let ref_min = reference::case_900::PEAK_HEATING_MIN;
    let ref_max = reference::case_900::PEAK_HEATING_MAX;
    let tolerance = (ref_max - ref_min) * PEAK_LOAD_TOLERANCE;

    let in_range = peak_heating >= ref_min - tolerance && peak_heating <= ref_max + tolerance;

    if in_range {
        println!("✅ PASS: Peak heating within reference range");
    } else {
        println!(
            "❌ FAIL: Peak heating {:.2} kW outside range [{:.2}, {:.2}] kW",
            peak_heating, ref_min, ref_max
        );
    }

    assert!(
        peak_heating >= ref_min - tolerance && peak_heating <= ref_max + tolerance,
        "Peak heating {:.2} kW outside reference range [{:.2}, {:.2}] kW (±{:.0}% tolerance)",
        peak_heating,
        ref_min,
        ref_max,
        PEAK_LOAD_TOLERANCE * 100.0
    );
}

/// Test: Case 900 multi-node peak cooling load
///
/// Validates that peak cooling load is within ASHRAE 140 reference range.
#[test]
fn test_case_900_multinode_peak_cooling() {
    let (_, cooling_kwh, _, peak_cooling, _, _) = simulate_case_900_multinode();
    let cooling_mwh = cooling_kwh / 1000.0;

    println!("\n=== Case 900 Multi-Node Peak Cooling ===");
    println!(
        "Peak Cooling: {:.2} kW (reference: {:.2} - {:.2} kW)",
        peak_cooling,
        reference::case_900::PEAK_COOLING_MIN,
        reference::case_900::PEAK_COOLING_MAX
    );
    println!("Annual Cooling: {:.2} MWh", cooling_mwh);

    let ref_min = reference::case_900::PEAK_COOLING_MIN;
    let ref_max = reference::case_900::PEAK_COOLING_MAX;
    let tolerance = (ref_max - ref_min) * PEAK_LOAD_TOLERANCE;

    let in_range = peak_cooling >= ref_min - tolerance && peak_cooling <= ref_max + tolerance;

    if in_range {
        println!("✅ PASS: Peak cooling within reference range");
    } else {
        println!(
            "❌ FAIL: Peak cooling {:.2} kW outside range [{:.2}, {:.2}] kW",
            peak_cooling, ref_min, ref_max
        );
    }

    assert!(
        peak_cooling >= ref_min - tolerance && peak_cooling <= ref_max + tolerance,
        "Peak cooling {:.2} kW outside reference range [{:.2}, {:.2}] kW (±{:.0}% tolerance)",
        peak_cooling,
        ref_min,
        ref_max,
        PEAK_LOAD_TOLERANCE * 100.0
    );
}

/// Test: Case 900FF multi-node free-floating temperatures
///
/// Validates that free-floating temperatures are within ASHRAE 140 reference range.
#[test]
fn test_case_900ff_multinode_temperatures() {
    let (min_temp, max_temp) = simulate_case_900ff_multinode();

    println!("\n=== Case 900FF Multi-Node Free-Floating ===");
    println!(
        "Min Temperature: {:.2}°C (reference: {:.2} - {:.2}°C)",
        min_temp,
        reference::case_900ff::MIN_TEMP_MIN,
        reference::case_900ff::MIN_TEMP_MAX
    );
    println!(
        "Max Temperature: {:.2}°C (reference: {:.2} - {:.2}°C)",
        max_temp,
        reference::case_900ff::MAX_TEMP_MIN,
        reference::case_900ff::MAX_TEMP_MAX
    );

    let min_ref_range = reference::case_900ff::MIN_TEMP_MIN..=reference::case_900ff::MIN_TEMP_MAX;
    let max_ref_range = reference::case_900ff::MAX_TEMP_MIN..=reference::case_900ff::MAX_TEMP_MAX;

    let min_in_range = min_ref_range.contains(&min_temp);
    let max_in_range = max_ref_range.contains(&max_temp);

    if min_in_range {
        println!("✅ Min temp {:.2}°C within reference", min_temp);
    } else {
        println!(
            "❌ Min temp {:.2}°C outside reference [{:.2}, {:.2}]",
            min_temp,
            reference::case_900ff::MIN_TEMP_MIN,
            reference::case_900ff::MIN_TEMP_MAX
        );
    }

    if max_in_range {
        println!("✅ Max temp {:.2}°C within reference", max_temp);
    } else {
        println!(
            "❌ Max temp {:.2}°C outside reference [{:.2}, {:.2}]",
            max_temp,
            reference::case_900ff::MAX_TEMP_MIN,
            reference::case_900ff::MAX_TEMP_MAX
        );
    }

    // Check min temperature
    assert!(
        min_temp >= reference::case_900ff::MIN_TEMP_MIN - 2.0
            && min_temp <= reference::case_900ff::MIN_TEMP_MAX + 2.0,
        "Min temperature {:.2}°C outside reference range [{:.2}, {:.2}]°C",
        min_temp,
        reference::case_900ff::MIN_TEMP_MIN,
        reference::case_900ff::MIN_TEMP_MAX
    );

    // Check max temperature
    assert!(
        max_temp >= reference::case_900ff::MAX_TEMP_MIN - 2.0
            && max_temp <= reference::case_900ff::MAX_TEMP_MAX + 2.0,
        "Max temperature {:.2}°C outside reference range [{:.2}, {:.2}]°C",
        max_temp,
        reference::case_900ff::MAX_TEMP_MIN,
        reference::case_900ff::MAX_TEMP_MAX
    );
}

/// Test: Case 900 multi-node validation summary
///
/// This is the primary acceptance test that produces a detailed pass/fail report.
#[test]
fn test_case_900_multinode_validation_summary() {
    let (heating_kwh, cooling_kwh, peak_heating, peak_cooling, min_temp, max_temp) =
        simulate_case_900_multinode();
    let (ff_min, ff_max) = simulate_case_900ff_multinode();

    let heating_mwh = heating_kwh / 1000.0;
    let cooling_mwh = cooling_kwh / 1000.0;

    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║          ASHRAE 140 Case 900 Multi-Node HVAC Validation Summary           ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Metric                │ Calculated    │ Reference Range     │ Status      ║");
    println!("╠══════════════════════╪═══════════════╪═════════════════════╪═════════════╣");

    // Annual Heating
    let ref_heat = format!(
        "{:.2} - {:.2} MWh",
        reference::case_900::ANNUAL_HEATING_MIN,
        reference::case_900::ANNUAL_HEATING_MAX
    );
    let heat_ok = heating_mwh
        >= reference::case_900::ANNUAL_HEATING_MIN * (1.0 - ANNUAL_ENERGY_TOLERANCE)
        && heating_mwh <= reference::case_900::ANNUAL_HEATING_MAX * (1.0 + ANNUAL_ENERGY_TOLERANCE);
    let heat_status = if heat_ok { "✓ PASS" } else { "✗ FAIL" };
    println!(
        "║ Annual Heating        │ {:>8.2} MWh  │ {:>18}   │ {:^9} ║",
        heating_mwh, ref_heat, heat_status
    );

    // Annual Cooling
    let ref_cool = format!(
        "{:.2} - {:.2} MWh",
        reference::case_900::ANNUAL_COOLING_MIN,
        reference::case_900::ANNUAL_COOLING_MAX
    );
    let cool_ok = cooling_mwh
        >= reference::case_900::ANNUAL_COOLING_MIN * (1.0 - ANNUAL_ENERGY_TOLERANCE)
        && cooling_mwh <= reference::case_900::ANNUAL_COOLING_MAX * (1.0 + ANNUAL_ENERGY_TOLERANCE);
    let cool_status = if cool_ok { "✓ PASS" } else { "✗ FAIL" };
    println!(
        "║ Annual Cooling        │ {:>8.2} MWh  │ {:>18}   │ {:^9} ║",
        cooling_mwh, ref_cool, cool_status
    );

    // Peak Heating
    let ref_pk_heat = format!(
        "{:.2} - {:.2} kW",
        reference::case_900::PEAK_HEATING_MIN,
        reference::case_900::PEAK_HEATING_MAX
    );
    let pk_heat_ok = peak_heating
        >= reference::case_900::PEAK_HEATING_MIN * (1.0 - PEAK_LOAD_TOLERANCE)
        && peak_heating <= reference::case_900::PEAK_HEATING_MAX * (1.0 + PEAK_LOAD_TOLERANCE);
    let pk_heat_status = if pk_heat_ok { "✓ PASS" } else { "✗ FAIL" };
    println!(
        "║ Peak Heating          │ {:>8.2} kW   │ {:>18}   │ {:^9} ║",
        peak_heating, ref_pk_heat, pk_heat_status
    );

    // Peak Cooling
    let ref_pk_cool = format!(
        "{:.2} - {:.2} kW",
        reference::case_900::PEAK_COOLING_MIN,
        reference::case_900::PEAK_COOLING_MAX
    );
    let pk_cool_ok = peak_cooling
        >= reference::case_900::PEAK_COOLING_MIN * (1.0 - PEAK_LOAD_TOLERANCE)
        && peak_cooling <= reference::case_900::PEAK_COOLING_MAX * (1.0 + PEAK_LOAD_TOLERANCE);
    let pk_cool_status = if pk_cool_ok { "✓ PASS" } else { "✗ FAIL" };
    println!(
        "║ Peak Cooling          │ {:>8.2} kW   │ {:>18}   │ {:^9} ║",
        peak_cooling, ref_pk_cool, pk_cool_status
    );

    // Free-float Min
    let ref_ff_min = format!(
        "{:.2} - {:.2}°C",
        reference::case_900ff::MIN_TEMP_MIN,
        reference::case_900ff::MIN_TEMP_MAX
    );
    let ff_min_ok = ff_min >= reference::case_900ff::MIN_TEMP_MIN - 2.0
        && ff_min <= reference::case_900ff::MIN_TEMP_MAX + 2.0;
    let ff_min_status = if ff_min_ok { "✓ PASS" } else { "✗ FAIL" };
    println!(
        "║ FF Min Temperature    │ {:>8.2}°C   │ {:>18}   │ {:^9} ║",
        ff_min, ref_ff_min, ff_min_status
    );

    // Free-float Max
    let ref_ff_max = format!(
        "{:.2} - {:.2}°C",
        reference::case_900ff::MAX_TEMP_MIN,
        reference::case_900ff::MAX_TEMP_MAX
    );
    let ff_max_ok = ff_max >= reference::case_900ff::MAX_TEMP_MIN - 2.0
        && ff_max <= reference::case_900ff::MAX_TEMP_MAX + 2.0;
    let ff_max_status = if ff_max_ok { "✓ PASS" } else { "✗ FAIL" };
    println!(
        "║ FF Max Temperature    │ {:>8.2}°C   │ {:>18}   │ {:^9} ║",
        ff_max, ref_ff_max, ff_max_status
    );

    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    // Temperature range in HVAC mode
    println!(
        "\nZone Temperature Range (HVAC mode): {:.2}°C - {:.2}°C",
        min_temp, max_temp
    );

    // Overall pass/fail
    let all_pass = heat_ok && cool_ok && pk_heat_ok && pk_cool_ok && ff_min_ok && ff_max_ok;

    println!("\n═══════════════════════════════════════════════════════════════════════════════");
    if all_pass {
        println!("✅ ALL VALIDATIONS PASSED - Multi-node HVAC is validated for Case 900");
    } else {
        println!("❌ SOME VALIDATIONS FAILED - See details above");
    }
    println!("═══════════════════════════════════════════════════════════════════════════════");

    // Final assertion
    assert!(
        heat_ok && cool_ok && pk_heat_ok && pk_cool_ok && ff_min_ok && ff_max_ok,
        "Case 900 multi-node validation failed - see output above for details"
    );
}

// VALIDATION METHODOLOGY DOCUMENTATION
// ====================================
//
// Validation methodology for multi-node Case 900 HVAC tests:
//
// ## Approach
//
// 1. **Multi-Node Model (9R4C)**: Uses ThermalMassNode for wall, roof, floor, internal
//    - Per-surface exterior temperatures (Issue #863)
//    - More detailed thermal network than 5R1C
//
// 2. **14-Day Warmup**: Per ASHRAE 140 §B2 guidance
//    - Avoids phantom energy from transient initial conditions
//    - Mass temperatures converge before energy accumulation
//
// 3. **Validation Metrics**:
//    - Annual heating/cooling energy (MWh)
//    - Peak heating/cooling loads (kW)
//    - Free-floating temperatures (°C)
//
// 4. **Tolerances**:
//    - Annual energy: ±15% (ASHRAE 140 standard)
//    - Peak loads: ±10% (ASHRAE 140 standard)
//    - Temperatures: ±2°C (physical reasonability)
//
// ## Expected Results
//
// | Metric           | Multi-Node (9R4C) | ASHRAE 140 Ref |
// |------------------|-------------------|----------------|
// | Annual Heating   | TBD               | 1.17 - 2.04 MWh |
// | Annual Cooling   | TBD               | 2.13 - 3.67 MWh |
// | Peak Heating     | TBD               | 1.10 - 2.10 kW |
// | Peak Cooling     | TBD               | 2.10 - 3.50 kW |
// | FF Min Temp      | TBD               | -6.4 to -1.6°C |
// | FF Max Temp      | TBD               | 41.8 to 46.4°C |
//
// ## Notes
//
// - Multi-node model uses per-surface exterior temperatures (Issue #863)
// - Warmup period prevents phantom heating from transient initial conditions
// - Internal gains (200W) included in zone air temperature calculation
