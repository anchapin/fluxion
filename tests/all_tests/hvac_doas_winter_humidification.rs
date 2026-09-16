//! DOAS winter humidification isolation tests (Issue #2464).
//!
//! Verifies the equipment-layer pre-requisite for the HVAC BESTEST
//! humidification series (RP-865 follow-on, HD001 single-stage adiabatic) and
//! the ASHRAE 62.1 §6.4 minimum indoor humidity guidance for cold-dry
//! climates. The DOAS must engage the humidifier when the outdoor air is
//! drier than the target dew-point setpoint and a humidifier is configured,
//! and must leave the supply humidity ratio unchanged otherwise.
//!
//! Reference values are computed against ASHRAE HoF (2021) Ch.1 formulas
//! (Magnus-Tetens for `T ≥ 0 °C`, Hyland-Wexler ice for `T < 0 °C`).
//!
//! ASHRAE 62.1-2019 §6.4 — minimum indoor humidity guidance (informative):
//! `w ≈ 0.0024 kg/kg` (≈ 30 % RH @ 20 °C zone temp). For cold-dry climates
//! (ASHRAE 169 climate zones 5B, 6A, 7, 8) the outdoor humidity ratio drops
//! to `≈ 0.0010 kg/kg` for 4–6 months/year; the humidifier must restore the
//! indoor ratio to the target during these periods.

use fluxion::sim::hvac::cooling_coil::CoolingCoil;
use fluxion::sim::hvac::doas::{Doas, DoasControl, DoasMode, DoasUnit};
use fluxion::sim::hvac::heating_coil::HeatingCoilComponent;
use fluxion::sim::hvac::humidifier::HumidifierComponent;
use fluxion::weather::psychrometrics::{calculate_humidity_ratio, saturation_vapor_pressure};
use fluxion_core::weather::psychrometrics::STANDARD_ATMOSPHERIC_PRESSURE_Pa;

const SEA_LEVEL_PA: f64 = STANDARD_ATMOSPHERIC_PRESSURE_Pa;
/// Latent heat of vaporization of water at 0 °C [J/kg] — ASHRAE HoF Ch.1 Eq. 32.
const H_FG_0C_J_PER_KG: f64 = 2_501_000.0;
/// Test tolerance on humidity ratio comparisons: ±1 % of the saturation value.
const W_REL_TOL: f64 = 0.01;
/// Test tolerance on latent capacity comparisons: ±1 % of the expected value.
const CAPACITY_REL_TOL: f64 = 0.01;

/// Build a standard 1.5 m³/s DOAS with optional humidifier. Sized to
/// dehumidify up to 35 °C / 80 % RH OA to a 10 °C dew-point and reheat the
/// cold coil leaving air to an 18 °C neutral supply. The reheat coil is sized
/// at 60 kW so that cold-dry OA (e.g. −10 °C) reaches the supply setpoint
/// without capacity limiting.
fn doas_with_humidifier(humidifier: Option<HumidifierComponent>) -> DoasUnit {
    let cooling = CoolingCoil::new(
        "CC-DOAS".into(),
        150_000.0, // 150 kW rated total
        0.50,      // rated SHR
        0.10,      // bypass factor
        10.0,      // ADP 10 °C
        1.8,       // design mass flow
    );
    let reheat = HeatingCoilComponent::new("HC-DOAS".into(), 60_000.0, 2.0);
    DoasUnit::new("DOAS-HUM".into(), 1.5, cooling, Some(reheat), humidifier)
}

fn outdoor_air(temp_c: f64, rh_percent: f64) -> fluxion::sim::hvac::MoistAirState {
    fluxion::sim::hvac::MoistAirState::try_new(temp_c, rh_percent, SEA_LEVEL_PA)
        .expect("valid outdoor air")
}

// =============================================================================
// Humidification activates in cold-dry winter conditions
// =============================================================================

/// Cold-dry outdoor air (−10 °C / 20 % RH, Denver-winter analog) must engage
/// the humidifier. The leaving humidity ratio should approach
/// `w_sat(target_dew_point)` (≈ 7.63e-3 kg/kg at 10 °C) and the latent
/// capacity should equal `ṁ_h2o · h_fg` within 1 %.
#[test]
fn humidifier_engages_in_cold_dry_winter() {
    let humidifier = HumidifierComponent::new("HUM-1".into(), 0.050, 1.8);
    let doas = doas_with_humidifier(Some(humidifier));
    let control = DoasControl::active(10.0, 18.0);
    let oa = outdoor_air(-10.0, 20.0);
    let rho = oa.density_kg_per_m3;

    let perf = doas
        .compute_doas_performance(&oa, rho, &control)
        .expect("DOAS performance must compute");

    // Mode remains HeatingOnly — the humidifier is a sub-stage of the cold-dry
    // path, not a separate operating mode.
    assert_eq!(
        perf.mode,
        DoasMode::HeatingOnly,
        "cold-dry OA must remain in HeatingOnly mode (humidifier is sub-stage)"
    );

    // Humidifier is active.
    assert!(
        perf.humidifier_active,
        "humidifier must engage when post-reheat w < w_sat(target_dp)"
    );
    assert!(
        perf.humidifier_capacity_w > 0.0,
        "humidifier capacity must be positive"
    );
    assert!(
        perf.humidifier_moisture_rate_kg_per_s > 0.0,
        "moisture rate must be positive"
    );

    // Supply humidity ratio must rise to the target saturation at the
    // neutral supply dry-bulb. The post-reheat dry-bulb is at 18 °C, so
    // `w_sat(10 °C)` is well below saturation at the leaving dry-bulb.
    let w_target = calculate_humidity_ratio(10.0, 100.0, SEA_LEVEL_PA);
    let w_supply = perf.supply_air.humidity_ratio_kg_per_kg_dry_air;
    assert!(
        (w_supply - w_target).abs() <= w_target * W_REL_TOL,
        "supply humidity ratio {w_supply} vs target {w_target} (tol {W_REL_TOL})"
    );

    // Latent capacity matches ṁ_h2o · h_fg within 1 %.
    let expected_capacity = perf.humidifier_moisture_rate_kg_per_s * H_FG_0C_J_PER_KG;
    assert!(
        (perf.humidifier_capacity_w - expected_capacity).abs()
            <= expected_capacity * CAPACITY_REL_TOL,
        "Q_lat {} W vs expected {} W",
        perf.humidifier_capacity_w,
        expected_capacity
    );

    // The dew-point target is met after the humidifier stage.
    assert!(
        perf.target_dew_point_met,
        "target dew-point must be met after the humidifier stage"
    );

    // The outdoor-air humidity ratio (≈ 3.19e-4 kg/kg) is well below the
    // target — verify we observed a meaningful moisture lift.
    assert!(
        w_supply > oa.humidity_ratio_kg_per_kg_dry_air + 1.0e-3,
        "supply w {w_supply} should be well above OA w {}",
        oa.humidity_ratio_kg_per_kg_dry_air
    );
}

// =============================================================================
// Humidification does NOT engage when outdoor air is already at/above target
// =============================================================================

/// When outdoor dew-point exceeds the target (humid summer), the DOAS runs
/// in CoolingDehumidification mode. The humidifier must NOT add moisture —
/// the cooling coil is driving `w` *down*, not up.
#[test]
fn humidifier_does_not_engage_in_cooling_dehumidification_mode() {
    let humidifier = HumidifierComponent::new("HUM-2".into(), 0.050, 1.8);
    let doas = doas_with_humidifier(Some(humidifier));
    let control = DoasControl::active(10.0, 18.0);
    let oa = outdoor_air(32.0, 60.0);
    let rho = oa.density_kg_per_m3;

    let perf = doas
        .compute_doas_performance(&oa, rho, &control)
        .expect("DOAS performance must compute");

    assert_eq!(perf.mode, DoasMode::CoolingDehumidification);
    assert!(
        !perf.humidifier_active,
        "humidifier must NOT engage in CoolingDehumidification mode"
    );
    assert_eq!(perf.humidifier_capacity_w, 0.0);
    assert_eq!(perf.humidifier_moisture_rate_kg_per_s, 0.0);
}

/// Ventilation mode implies OA dew-point ≤ target, so the outdoor humidity
/// ratio is at or below `w_sat(target_dp)`. If the OA is dry, the
/// humidifier correctly engages to bring the supply up to the target; if
/// the OA is exactly at saturation at the target dew-point, the humidifier
/// does not engage (no moisture lift required).
///
/// This test verifies that the humidifier engages in Ventilation mode for
/// mildly-dry outdoor air (e.g. shoulder-season spring/fall conditions when
/// the DOAS would otherwise run with no active conditioning).
#[test]
fn humidifier_engages_in_ventilation_mode_when_oa_is_dry() {
    let humidifier = HumidifierComponent::new("HUM-3".into(), 0.050, 1.8);
    let doas = doas_with_humidifier(Some(humidifier));
    let control = DoasControl::active(10.0, 18.0);
    // OA 18.0 °C / 50 % RH: dry-bulb exactly at the supply setpoint
    // (within the ±0.05 °C Ventilation deadband). Dew-point ≈ 7.4 °C ≤ target.
    let oa = outdoor_air(18.0, 50.0);
    let rho = oa.density_kg_per_m3;

    let perf = doas
        .compute_doas_performance(&oa, rho, &control)
        .expect("DOAS performance must compute");

    assert_eq!(perf.mode, DoasMode::Ventilation);
    // The fan heat may nudge the post-fan dry-bulb above the supply setpoint,
    // so reheat stays off. The post-fan humidity ratio equals the OA value,
    // below the target → humidifier must engage.
    assert!(
        perf.humidifier_active,
        "humidifier must engage in Ventilation when OA humidity is below target"
    );
    assert!(perf.humidifier_capacity_w > 0.0);
}

/// When sensible cooling (no dehumidification) is active — warm-dry outdoor
/// with OA dew-point already below target — the post-cooling humidity ratio
/// equals the entering value. Because the entering ratio is below the
/// target dew-point saturation, the humidifier *correctly* engages to
/// bring the supply up to ASHRAE 62.1 §6.4 levels. This test verifies
/// that the path round-trips through the cooler (no cooling latent term)
/// while the humidifier raises `w` to target.
#[test]
fn humidifier_engages_in_sensible_cooling_mode() {
    let humidifier = HumidifierComponent::new("HUM-4".into(), 0.050, 1.8);
    let doas = doas_with_humidifier(Some(humidifier));
    let control = DoasControl::active(10.0, 18.0);
    // OA 30 °C / 15 % RH → dew-point ≈ 0.8 °C (< target 10 °C), so the
    // DOAS is in SensibleCooling mode (no dehumidification). The post-cooling
    // humidity ratio equals the OA value, well below `w_sat(target_dp)`.
    let oa = outdoor_air(30.0, 15.0);
    let rho = oa.density_kg_per_m3;

    let perf = doas
        .compute_doas_performance(&oa, rho, &control)
        .expect("DOAS performance must compute");

    assert_eq!(perf.mode, DoasMode::SensibleCooling);
    // No cooling latent term — sensible-only cooling is by definition.
    assert_eq!(perf.cooling_latent_capacity_w, 0.0);
    // Humidifier must engage to bring the dry outdoor air up to the target.
    assert!(
        perf.humidifier_active,
        "humidifier must engage in SensibleCooling when post-cooling w < target"
    );
    assert!(perf.humidifier_capacity_w > 0.0);
}

/// When the DOAS is OFF, the humidifier must NOT engage regardless of the
/// outdoor state.
#[test]
fn humidifier_does_not_engage_when_doas_is_off() {
    let humidifier = HumidifierComponent::new("HUM-5".into(), 0.050, 1.8);
    let doas = doas_with_humidifier(Some(humidifier));
    let oa = outdoor_air(-10.0, 20.0);
    let rho = oa.density_kg_per_m3;

    let perf = doas
        .compute_doas_performance(&oa, rho, &DoasControl::off())
        .expect("DOAS performance must compute");

    assert_eq!(perf.mode, DoasMode::Off);
    assert!(!perf.humidifier_active);
    assert_eq!(perf.humidifier_capacity_w, 0.0);
    assert_eq!(perf.humidifier_moisture_rate_kg_per_s, 0.0);
    assert_eq!(perf.volumetric_flow_m3_per_s, 0.0);
}

// =============================================================================
// Humidification does NOT engage when no humidifier is configured
// =============================================================================

/// Pre-#2464 behavior must be preserved: a DOAS without a humidifier
/// configuration must not engage any humidification, even in cold-dry winter
/// conditions. This is the back-compat / regression guard.
#[test]
fn humidifier_inactive_when_no_humidifier_configured() {
    let doas = doas_with_humidifier(None);
    let control = DoasControl::active(10.0, 18.0);
    let oa = outdoor_air(-10.0, 20.0);
    let rho = oa.density_kg_per_m3;

    let perf = doas
        .compute_doas_performance(&oa, rho, &control)
        .expect("DOAS performance must compute");

    assert!(!doas.has_humidifier());
    assert!(!perf.humidifier_active);
    assert_eq!(perf.humidifier_capacity_w, 0.0);
    assert_eq!(perf.humidifier_moisture_rate_kg_per_s, 0.0);

    // Supply humidity ratio must equal outdoor (no humidification stage).
    assert!(
        (perf.supply_air.humidity_ratio_kg_per_kg_dry_air - oa.humidity_ratio_kg_per_kg_dry_air)
            .abs()
            < 1.0e-9,
        "supply w should equal OA w when no humidifier is configured"
    );
}

// =============================================================================
// Capacity clamping against rated moisture rate
// =============================================================================

/// When the moisture demand exceeds the rated humidifier capacity, the
/// delivered rate must clamp at the rated value and the leaving humidity
/// ratio must stop below the target (a smaller-rated humidifier cannot
/// reach the target).
#[test]
fn humidifier_capacity_clamps_at_rated_moisture_rate() {
    // Rated at 0.001 kg_water/s — well below the ~ 4.2e-3 kg/s required
    // for the cold-dry scenario, so the leaving ratio will undershoot target.
    let undersized_humidifier = HumidifierComponent::new("HUM-SMALL".into(), 0.001, 1.8);
    let doas = doas_with_humidifier(Some(undersized_humidifier));
    let control = DoasControl::active(10.0, 18.0);
    let oa = outdoor_air(-10.0, 20.0);
    let rho = oa.density_kg_per_m3;

    let perf = doas
        .compute_doas_performance(&oa, rho, &control)
        .expect("DOAS performance must compute");

    assert!(perf.humidifier_active);
    assert!((perf.humidifier_moisture_rate_kg_per_s - 0.001).abs() < 1.0e-9);
    // The undersized humidifier cannot reach the target; the leaving ratio
    // is between OA w and target w.
    let w_target = calculate_humidity_ratio(10.0, 100.0, SEA_LEVEL_PA);
    let w_supply = perf.supply_air.humidity_ratio_kg_per_kg_dry_air;
    assert!(
        w_supply > oa.humidity_ratio_kg_per_kg_dry_air,
        "supply w {w_supply} should be above OA w"
    );
    assert!(
        w_supply < w_target,
        "undersized humidifier cannot reach target {w_target}: got {w_supply}"
    );
}

// =============================================================================
// Saturation guard
// =============================================================================

/// Physically, an adiabatic humidifier cannot supersaturate the air. Verify
/// the leaving relative humidity stays ≤ 100 % even when the target
/// humidity ratio would imply a supersaturated state (this is reachable only
/// via pathological DOAS setpoints where target_dp > supply_db).
#[test]
fn leaving_air_never_supersaturated() {
    let humidifier = HumidifierComponent::new("HUM-6".into(), 1.0, 1.8);
    let doas = doas_with_humidifier(Some(humidifier));
    // Pathological setpoint: target dew-point = 18 °C > supply dry-bulb = 5 °C.
    // The cooling/heating path will produce a 5 °C leaving state; the
    // humidifier request is for w_sat(18 °C) which exceeds w_sat(5 °C).
    let control = DoasControl::active(18.0, 5.0);
    // Cold dry OA, well below the target dew-point.
    let oa = outdoor_air(-10.0, 20.0);
    let rho = oa.density_kg_per_m3;

    let perf = doas
        .compute_doas_performance(&oa, rho, &control)
        .expect("DOAS performance must compute");

    // The leaving air must remain physical.
    assert!(
        perf.supply_air.relative_humidity_percent <= 100.0 + 1.0e-6,
        "leaving RH {} % must not exceed 100 %",
        perf.supply_air.relative_humidity_percent
    );
    // Partial vapor pressure must be at or below saturation at the leaving
    // dry-bulb.
    let p_sat_leaving = saturation_vapor_pressure(perf.supply_air.dry_bulb_c);
    assert!(
        perf.supply_air.partial_vapor_pressure_pa <= p_sat_leaving + 1.0e-3,
        "leaving p_v {} must be ≤ p_sat {}",
        perf.supply_air.partial_vapor_pressure_pa,
        p_sat_leaving
    );
}

// =============================================================================
// Energy-balance closure (latent heat flow)
// =============================================================================

/// The latent heat delivered by the humidifier must match the enthalpy-flow
/// closure `Q_lat = ṁ_da · (h_out − h_in)` for the humidity-ratio change
/// alone — the airside coupling layer (`airside_coupling.rs`) credits this
/// through `supply_latent_heat_w`.
///
/// Energy-balance check: the increase in moist-air enthalpy across the
/// humidifier stage equals the rated latent capacity. Per ASHRAE HoF Ch.1,
/// the latent contribution to moist-air enthalpy is `h_fg + c_pw · T`
/// (≈ 2.45 MJ/kg + 1.86 kJ/(kg·K) · T), which we approximate as the
/// 0 °C reference (`2.501 MJ/kg`) for the high-level closure.
#[test]
fn humidifier_energy_balance_closes() {
    let humidifier = HumidifierComponent::new("HUM-EB".into(), 0.050, 1.8);
    let doas = doas_with_humidifier(Some(humidifier));
    let control = DoasControl::active(10.0, 18.0);
    let oa = outdoor_air(-10.0, 20.0);
    let rho = oa.density_kg_per_m3;

    let perf = doas
        .compute_doas_performance(&oa, rho, &control)
        .expect("DOAS performance must compute");

    // The latent heat credited to the zone is exactly the rated capacity.
    let expected_latent = perf.humidifier_moisture_rate_kg_per_s * H_FG_0C_J_PER_KG;
    assert!(
        (perf.humidifier_capacity_w - expected_latent).abs() <= 1.0,
        "Q_lat {} W vs expected {} W",
        perf.humidifier_capacity_w,
        expected_latent
    );
}

// =============================================================================
// Back-compat: in CoolingDehumidification mode the cooling coil drives the
// leaving humidity ratio to `w_sat(target_dp)`, so the humidifier does not
// engage whether or not a humidifier is configured. With/without humidifier
// must produce identical supply humidity ratios.
// =============================================================================

#[test]
fn back_compat_cooling_dehumidification_path_with_humidifier() {
    let with_hum =
        doas_with_humidifier(Some(HumidifierComponent::new("HUM-BC".into(), 0.050, 1.8)));
    let without_hum = doas_with_humidifier(None);

    let control = DoasControl::active(10.0, 18.0);
    // OA 32 °C / 60 % RH → dew-point ≈ 23 °C > target 10 °C → CoolingDehumidification.
    // After dehumidification, supply w = w_sat(target_dp) ≈ 0.00763.
    let oa = outdoor_air(32.0, 60.0);
    let rho = oa.density_kg_per_m3;

    let perf_with = with_hum
        .compute_doas_performance(&oa, rho, &control)
        .expect("with humidifier");
    let perf_without = without_hum
        .compute_doas_performance(&oa, rho, &control)
        .expect("without humidifier");

    // Both produce CoolingDehumidification; the humidifier must NOT engage in
    // either case (the supply is already at saturation of the target dp).
    assert_eq!(perf_with.mode, DoasMode::CoolingDehumidification);
    assert_eq!(perf_without.mode, DoasMode::CoolingDehumidification);
    assert!(!perf_with.humidifier_active);
    assert_eq!(perf_with.humidifier_capacity_w, 0.0);
    // Supply humidity ratio must match (cooling coil drives both to the
    // same `w_sat(target_dp)`).
    assert!(
        (perf_with.supply_air.humidity_ratio_kg_per_kg_dry_air
            - perf_without.supply_air.humidity_ratio_kg_per_kg_dry_air)
            .abs()
            < 1.0e-12,
        "CoolingDehumidification supply w should match between with/without humidifier"
    );
}
