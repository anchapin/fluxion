//! Unit tests for temperature-dependent air exchange rate (ACH) using stack effect.
//!
//! Tests validate the formula Q_vent = 0.025·A·√(ΔT/h) for buoyancy-driven ventilation
//! and the air enthalpy method Q = ρ·Cp·ACH·V·ΔT for heat transfer.

/// Stack effect coefficient for buoyancy-driven ventilation
/// Based on ASHRAE 140 natural ventilation model
const STACK_EFFECT_COEFFICIENT: f64 = 0.025;

/// Air density at standard conditions (kg/m³)
const AIR_DENSITY: f64 = 1.2;

/// Air specific heat capacity (J/kgK)
const AIR_SPECIFIC_HEAT: f64 = 1000.0;

/// Tolerance for numerical comparison (1%)
const TOLERANCE_PCT: f64 = 1.0;

/// Calculate stack effect ventilation rate
///
/// Q_vent = C * A_door * sqrt(ΔT / h_door)
///
/// # Arguments
/// * `door_area` - Area of opening (m²)
/// * `door_height` - Height of opening (m)
/// * `delta_t` - Temperature difference between zones (°C)
///
/// # Returns
/// Ventilation rate (m³/hr)
fn calculate_stack_effect_ventilation(door_area: f64, door_height: f64, delta_t: f64) -> f64 {
    // Stack effect formula: Q = 0.025 * A * sqrt(ΔT / h)
    STACK_EFFECT_COEFFICIENT * door_area * (delta_t / door_height).sqrt()
}

/// Calculate air exchange rate (ACH)
///
/// ACH = Q_vent / V_zone
///
/// # Arguments
/// * `ventilation_rate` - Volumetric flow rate (m³/hr)
/// * `zone_volume` - Zone volume (m³)
///
/// # Returns
/// Air exchange rate (1/hr)
fn calculate_ach(ventilation_rate: f64, zone_volume: f64) -> f64 {
    ventilation_rate / zone_volume
}

/// Calculate ventilation heat transfer using air enthalpy method
///
/// Q_vent = ρ * Cp * ACH * V * ΔT
///
/// # Arguments
/// * `ach` - Air exchange rate (1/hr)
/// * `zone_volume` - Zone volume (m³)
/// * `delta_t` - Temperature difference (°C)
///
/// # Returns
/// Heat transfer rate (W)
fn calculate_ventilation_heat_transfer(ach: f64, zone_volume: f64, delta_t: f64) -> f64 {
    // Q = ρ * Cp * ACH * V * ΔT
    // Result in J/hr, convert to W by dividing by 3600
    (AIR_DENSITY * AIR_SPECIFIC_HEAT * ach * zone_volume * delta_t) / 3600.0
}

#[test]
fn test_stack_effect_ach_formula() {
    // Test stack effect ACH formula for winter condition
    // Temperature A = 20°C, Temperature B = 0°C (ΔT = 20°C)

    let temp_a = 20.0; // °C
    let temp_b = 0.0; // °C
    let delta_t = temp_a - temp_b; // 20°C

    let door_height = 2.0; // m
    let door_area = 1.5; // m²
    let zone_volume = 3.0; // m³ (approximate: door_area * door_height)

    // Stack effect: Q_vent = 0.025 * A * sqrt(ΔT / h)
    // Q_vent = 0.025 * 1.5 * sqrt(20 / 2.0) = 0.025 * 1.5 * 3.16 = 0.119 m³/hr
    let expected_ventilation = 0.119; // m³/hr

    let q_vent = calculate_stack_effect_ventilation(door_area, door_height, delta_t);

    println!("Stack effect ventilation: {:.4} m³/hr", q_vent);
    println!("Expected: {:.4} m³/hr", expected_ventilation);
    println!(
        "Difference: {:.2}%",
        ((q_vent - expected_ventilation).abs() / expected_ventilation) * 100.0
    );

    assert!(
        (q_vent - expected_ventilation).abs() / expected_ventilation * 100.0 < TOLERANCE_PCT,
        "Ventilation {:.4} m³/hr differs from expected {:.4} m³/hr by more than {}%",
        q_vent,
        expected_ventilation,
        TOLERANCE_PCT
    );

    // ACH = Q_vent / V_zone = 0.119 / 3.0 = 0.04 /hr (very low, closed door)
    let expected_ach = 0.04; // /hr

    let ach = calculate_ach(q_vent, zone_volume);

    println!("ACH: {:.4} /hr", ach);
    println!("Expected: {:.4} /hr", expected_ach);

    assert!(
        (ach - expected_ach).abs() < 0.005,
        "ACH {:.4} /hr differs from expected {:.4} /hr",
        ach,
        expected_ach
    );
}

#[test]
fn test_air_enthalpy_method() {
    // Test air enthalpy heat transfer calculation
    // ACH = 0.04 /hr, Volume = 3.0 m³, ΔT = 20°C

    let ach = 0.04; // /hr
    let zone_volume = 3.0; // m³
    let delta_t = 20.0; // °C

    // Q_vent = ρ * Cp * ACH * V * ΔT
    // Q_vent = 1.2 * 1000 * 0.04 * 3.0 * 20 = 2880 J/hr = 0.8 W
    let expected_heat_transfer = 0.8; // W

    let q_vent = calculate_ventilation_heat_transfer(ach, zone_volume, delta_t);

    println!("Ventilation heat transfer: {:.2} W", q_vent);
    println!("Expected: {:.2} W", expected_heat_transfer);
    println!(
        "Difference: {:.2}%",
        ((q_vent - expected_heat_transfer).abs() / expected_heat_transfer) * 100.0
    );

    assert!(
        (q_vent - expected_heat_transfer).abs() / expected_heat_transfer * 100.0 < TOLERANCE_PCT,
        "Heat transfer {:.2} W differs from expected {:.2} W by more than {}%",
        q_vent,
        expected_heat_transfer,
        TOLERANCE_PCT
    );
}

#[test]
fn test_temperature_dependence() {
    // Test that ACH increases with temperature difference
    // ACH ∝ √(ΔT)

    let door_area = 1.5; // m²
    let door_height = 2.0; // m
    let zone_volume = 3.0; // m³

    // Test at different ΔT values
    let delta_t_0 = 0.0; // °C
    let delta_t_10 = 10.0; // °C
    let delta_t_20 = 20.0; // °C

    let q_vent_0 = calculate_stack_effect_ventilation(door_area, door_height, delta_t_0);
    let q_vent_10 = calculate_stack_effect_ventilation(door_area, door_height, delta_t_10);
    let q_vent_20 = calculate_stack_effect_ventilation(door_area, door_height, delta_t_20);

    let ach_0 = calculate_ach(q_vent_0, zone_volume);
    let ach_10 = calculate_ach(q_vent_10, zone_volume);
    let ach_20 = calculate_ach(q_vent_20, zone_volume);

    println!("ΔT=0°C:  ACH = {:.4} /hr", ach_0);
    println!("ΔT=10°C: ACH = {:.4} /hr", ach_10);
    println!("ΔT=20°C: ACH = {:.4} /hr", ach_20);

    // ΔT = 0°C → ACH = 0 /hr (no buoyancy, equal temps)
    assert_eq!(ach_0, 0.0, "Zero ΔT should give zero ACH");

    // ACH at 20°C should be √(20/10) = √2 ≈ 1.41× higher than at 10°C
    let ratio = ach_20 / ach_10;
    let expected_ratio = f64::sqrt(20.0 / 10.0);

    println!(
        "Ratio (ACH_20 / ACH_10): {:.2}× (expected {:.2}×)",
        ratio, expected_ratio
    );

    assert!(
        (ratio - expected_ratio).abs() < 0.01,
        "ACH should scale with √(ΔT), expected ratio {:.2}, got {:.2}",
        expected_ratio,
        ratio
    );
}

#[test]
fn test_stack_effect_door_geometry() {
    // Test that ventilation rate scales with door area and inversely with height
    // Q ∝ A / √h

    let delta_t = 20.0; // °C

    // Base geometry
    let door_area_1 = 1.5; // m²
    let door_height_1 = 2.0; // m

    let q_vent_1 = calculate_stack_effect_ventilation(door_area_1, door_height_1, delta_t);

    // Double area (should double ventilation)
    let door_area_2 = 3.0; // m²
    let door_height_2 = 2.0; // m

    let q_vent_2 = calculate_stack_effect_ventilation(door_area_2, door_height_2, delta_t);

    let ratio_area = q_vent_2 / q_vent_1;
    let expected_ratio_area = 2.0;

    println!(
        "Doubling door area: {:.2}× (expected {:.2}×)",
        ratio_area, expected_ratio_area
    );

    assert!(
        (ratio_area - expected_ratio_area).abs() < 0.01,
        "Ventilation should scale linearly with area, expected ratio {:.2}, got {:.2}",
        expected_ratio_area,
        ratio_area
    );

    // Double height (should reduce ventilation by √2)
    let door_area_3 = 1.5; // m²
    let door_height_3 = 4.0; // m

    let q_vent_3 = calculate_stack_effect_ventilation(door_area_3, door_height_3, delta_t);

    let ratio_height = q_vent_3 / q_vent_1;
    let expected_ratio_height = 1.0 / f64::sqrt(2.0);

    println!(
        "Doubling door height: {:.2}× (expected {:.2}×)",
        ratio_height, expected_ratio_height
    );

    assert!(
        (ratio_height - expected_ratio_height).abs() < 0.01,
        "Ventilation should scale inversely with √(height), expected ratio {:.2}, got {:.2}",
        expected_ratio_height,
        ratio_height
    );
}

#[test]
fn test_ventilation_heat_transfer_scaling() {
    // Test that heat transfer scales with all parameters
    // Q = ρ * Cp * ACH * V * ΔT

    let ach = 0.1; // /hr
    let zone_volume = 10.0; // m³
    let delta_t = 10.0; // °C

    let q_base = calculate_ventilation_heat_transfer(ach, zone_volume, delta_t);

    // Double ACH (should double heat transfer)
    let q_2x_ach = calculate_ventilation_heat_transfer(ach * 2.0, zone_volume, delta_t);
    let ratio_ach = q_2x_ach / q_base;

    println!("Doubling ACH: {:.2}× (expected 2.00×)", ratio_ach);

    assert!(
        (ratio_ach - 2.0).abs() < 0.01,
        "Heat transfer should scale linearly with ACH"
    );

    // Double volume (should double heat transfer)
    let q_2x_vol = calculate_ventilation_heat_transfer(ach, zone_volume * 2.0, delta_t);
    let ratio_vol = q_2x_vol / q_base;

    println!("Doubling volume: {:.2}× (expected 2.00×)", ratio_vol);

    assert!(
        (ratio_vol - 2.0).abs() < 0.01,
        "Heat transfer should scale linearly with volume"
    );

    // Double ΔT (should double heat transfer)
    let q_2x_dt = calculate_ventilation_heat_transfer(ach, zone_volume, delta_t * 2.0);
    let ratio_dt = q_2x_dt / q_base;

    println!("Doubling ΔT: {:.2}× (expected 2.00×)", ratio_dt);

    assert!(
        (ratio_dt - 2.0).abs() < 0.01,
        "Heat transfer should scale linearly with ΔT"
    );
}

#[test]
fn test_common_pitfall_missing_rho_cp() {
    // Test common pitfall: omitting ρ·Cp gives 1200x error
    // This is why the air enthalpy method is critical

    let ach = 0.1; // /hr
    let zone_volume = 10.0; // m³
    let delta_t = 10.0; // °C

    // Correct: Q = ρ * Cp * ACH * V * ΔT
    let q_correct = calculate_ventilation_heat_transfer(ach, zone_volume, delta_t);

    // Incorrect: Q = ACH * V * ΔT (omitting ρ·Cp)
    let q_incorrect = (ach * zone_volume * delta_t) / 3600.0;

    println!("Correct (with ρ·Cp): {:.2} W", q_correct);
    println!("Incorrect (without ρ·Cp): {:.2} W", q_incorrect);

    let error_factor = q_correct / q_incorrect;
    let expected_error = AIR_DENSITY * AIR_SPECIFIC_HEAT; // 1200

    println!(
        "Error factor: {:.0}× (expected {:.0}×)",
        error_factor, expected_error
    );

    // Omitting ρ·Cp gives ~1200x error
    assert!(
        (error_factor - expected_error).abs() < 100.0,
        "Expected ~1200× error, got {:.0}×",
        error_factor
    );
}

#[test]
fn test_stack_effect_extreme_delta_t() {
    // Test stack effect with extreme temperature difference
    // Large ΔT in sunspace: 0°C (exterior) to 50°C (sunspace)

    let delta_t = 50.0; // °C
    let door_area = 1.5; // m²
    let door_height = 2.0; // m

    let q_vent = calculate_stack_effect_ventilation(door_area, door_height, delta_t);

    // Q = 0.025 * 1.5 * sqrt(50/2.0) = 0.025 * 1.5 * 5.0 = 0.188 m³/hr
    let expected_ventilation = 0.188; // m³/hr

    println!("Extreme ΔT ventilation: {:.4} m³/hr", q_vent);
    println!("Expected: {:.4} m³/hr", expected_ventilation);

    assert!(
        (q_vent - expected_ventilation).abs() / expected_ventilation * 100.0 < TOLERANCE_PCT,
        "Ventilation {:.4} m³/hr differs from expected {:.4} m³/hr",
        q_vent,
        expected_ventilation
    );

    // Verify √(ΔT) scaling: 50°C should be √(50/20) = √2.5 ≈ 1.58× higher than 20°C
    let q_vent_20 = calculate_stack_effect_ventilation(door_area, door_height, 20.0);
    let ratio_extreme = q_vent / q_vent_20;
    let expected_ratio = f64::sqrt(50.0 / 20.0);

    println!(
        "Ratio (ΔT=50°C / ΔT=20°C): {:.2}× (expected {:.2}×)",
        ratio_extreme, expected_ratio
    );

    assert!(
        (ratio_extreme - expected_ratio).abs() < 0.01,
        "Extreme ΔT should follow √(ΔT) scaling"
    );
}

#[test]
fn test_ach_zero_zone_volume() {
    // Test edge case: zero zone volume should give infinite ACH
    // This is physically invalid but mathematically correct

    let q_vent = calculate_stack_effect_ventilation(1.5, 2.0, 10.0);
    let ach = calculate_ach(q_vent, 0.0);

    // Should be infinity (or very large number)
    assert!(ach.is_infinite(), "Zero volume should give infinite ACH");
}

#[test]
fn test_ach_zero_ventilation() {
    // Test edge case: zero ventilation should give zero ACH

    let ach = calculate_ach(0.0, 10.0);

    assert_eq!(ach, 0.0, "Zero ventilation should give zero ACH");
}

#[test]
fn test_heat_transfer_zero_ach() {
    // Test edge case: zero ACH should give zero heat transfer

    let q = calculate_ventilation_heat_transfer(0.0, 10.0, 10.0);

    assert_eq!(q, 0.0, "Zero ACH should give zero heat transfer");
}

#[test]
fn test_heat_transfer_zero_delta_t() {
    // Test edge case: zero ΔT should give zero heat transfer

    let q = calculate_ventilation_heat_transfer(0.1, 10.0, 0.0);

    assert_eq!(q, 0.0, "Zero ΔT should give zero heat transfer");
}

#[test]
fn test_heat_transfer_negative_delta_t() {
    // Test that negative ΔT gives negative heat transfer (cooling)
    // Heat flows from hot to cold, so negative sign is correct

    let q = calculate_ventilation_heat_transfer(0.1, 10.0, -10.0);

    println!("Heat transfer with negative ΔT: {:.2} W", q);

    assert!(
        q < 0.0,
        "Negative ΔT should give negative heat transfer (cooling)"
    );
}

#[test]
fn test_comprehensive_winter_scenario() {
    // Test comprehensive winter scenario with all components
    // Sunspace at 40°C, back-zone at 20°C, door connecting them

    let temp_sunspace = 40.0; // °C
    let temp_backzone = 20.0; // °C
    let delta_t = temp_sunspace - temp_backzone; // 20°C

    let door_area = 2.0; // m²
    let door_height = 2.5; // m
    let backzone_volume = 50.0; // m³

    // Calculate ventilation
    let q_vent = calculate_stack_effect_ventilation(door_area, door_height, delta_t);
    let ach = calculate_ach(q_vent, backzone_volume);
    let q_heat = calculate_ventilation_heat_transfer(ach, backzone_volume, delta_t);

    println!("Winter scenario:");
    println!(
        "  Sunspace: {:.1}°C, Back-zone: {:.1}°C",
        temp_sunspace, temp_backzone
    );
    println!("  Door: {:.1} m² × {:.1} m", door_area, door_height);
    println!("  Ventilation: {:.4} m³/hr", q_vent);
    println!("  ACH: {:.4} /hr", ach);
    println!("  Heat transfer: {:.2} W", q_heat);

    // Verify physical constraints
    assert!(q_vent > 0.0, "Ventilation should be positive");
    assert!(ach > 0.0, "ACH should be positive");
    assert!(
        q_heat > 0.0,
        "Heat transfer should be positive (sunspace -> back-zone)"
    );
}
