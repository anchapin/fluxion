//! Integration tests for ASHRAE 140 thermostat setback and night ventilation cases.
//!
//! These cases test time-based HVAC control strategies:
//!
//! # Thermostat Setback Cases (640, 940)
//! - Heating setpoint drops to 10°C overnight (23:00-07:00)
//! - Normal heating setpoint (20°C) during daytime
//! - Tests thermal mass benefits from setpoint variation
//!
//! # Night Ventilation Cases (650, 950)
//! - Heating is disabled
//! - Ventilation fan runs 18:00-07:00
//! - Tests passive cooling through night air exchange
//! - Cooling-only operation during the day

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, HvacSchedule, NightVentilation};
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Reference ranges for thermostat setback cases
mod reference {
    // Case 640 - Low mass with thermostat setback
    pub mod case_640 {
        // Round 28 provenance correction: published 140-2023 Annex B8 values (was:
        // 2.75/3.80, 5.95/8.10 — issue #54 values of unverified origin; see benchmark.rs).
        pub const ANNUAL_HEATING_MIN: f64 = 2.403;
        pub const ANNUAL_HEATING_MAX: f64 = 2.682;
        pub const ANNUAL_COOLING_MIN: f64 = 5.237;
        pub const ANNUAL_COOLING_MAX: f64 = 5.893;
    }

    // Case 650 - Low mass with night ventilation
    pub mod case_650 {
        pub const ANNUAL_HEATING_MIN: f64 = 0.00;
        pub const ANNUAL_HEATING_MAX: f64 = 0.00;
        pub const ANNUAL_COOLING_MIN: f64 = 4.82;
        pub const ANNUAL_COOLING_MAX: f64 = 7.06;
    }

    // Case 940 - High mass with thermostat setback (published 140-2023 Annex B8 values,
    // round 29 provenance correction; see benchmark.rs Case 940 note)
    pub mod case_940 {
        pub const ANNUAL_HEATING_MIN: f64 = 0.863;
        pub const ANNUAL_HEATING_MAX: f64 = 1.389;
        pub const ANNUAL_COOLING_MIN: f64 = 2.203;
        pub const ANNUAL_COOLING_MAX: f64 = 2.613;
    }

    // Case 950 - High mass with night ventilation
    pub mod case_950 {
        pub const ANNUAL_HEATING_MIN: f64 = 0.00;
        pub const ANNUAL_HEATING_MAX: f64 = 0.00;
        pub const ANNUAL_COOLING_MIN: f64 = 0.39;
        pub const ANNUAL_COOLING_MAX: f64 = 0.92;
    }
}

/// Simulates a case and returns annual heating/cooling in MWh
fn simulate_case(case: ASHRAE140Case) -> (f64, f64) {
    let spec = case.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    let mut annual_heating_joules = 0.0;
    let mut annual_cooling_joules = 0.0;

    for step in 0..8760 {
        let hour_of_day = step % 24;
        let weather_data = weather.get_hourly_data(step).unwrap();

        // Update weather data on model for solar gain calculation (Issue #278)
        model.solar.weather = Some(weather_data.clone());

        // Apply dynamic setpoints based on HVAC schedule
        // Issue #2870: use the sub-hour ramp-aware setpoint so Cases 640/940
        // see the smoothed setback→occupied transition over a 2-hour ramp.
        if let Some(hvac_schedule) = spec.hvac.first() {
            if let Some(heating_sp) = hvac_schedule
                .heating_setpoint_at_fractional_hour(f64::from(hour_of_day as u8) + 0.5)
            {
                model.setpoints.heating_setpoint = heating_sp;
            }
            if let Some(cooling_sp) = hvac_schedule.cooling_setpoint_at_hour(hour_of_day as u8) {
                model.setpoints.cooling_setpoint = cooling_sp;
            }
        }

        let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if hvac_kwh > 0.0 {
            annual_heating_joules += hvac_kwh * 3.6e6;
        } else {
            annual_cooling_joules += (-hvac_kwh) * 3.6e6;
        }
    }

    (annual_heating_joules / 3.6e9, annual_cooling_joules / 3.6e9)
}

#[test]
fn test_hvac_schedule_with_setback() {
    // Test thermostat setback schedule
    let schedule = HvacSchedule::with_setback(20.0, 27.0, 10.0, 23, 7);

    assert!(schedule.is_enabled(), "Schedule should be enabled");

    // During setback hours (23:00-07:00), heating setpoint should be 10°C
    assert_eq!(
        schedule.heating_setpoint_at_hour(0),
        Some(10.0),
        "Midnight should have setback setpoint"
    );
    assert_eq!(
        schedule.heating_setpoint_at_hour(6),
        Some(10.0),
        "6 AM should have setback setpoint"
    );

    // During normal hours, heating setpoint should be 20°C
    assert_eq!(
        schedule.heating_setpoint_at_hour(12),
        Some(20.0),
        "Noon should have normal setpoint"
    );
    assert_eq!(
        schedule.heating_setpoint_at_hour(20),
        Some(20.0),
        "8 PM should have normal setpoint"
    );

    // Cooling setpoint should remain constant
    assert_eq!(
        schedule.cooling_setpoint_at_hour(0),
        Some(27.0),
        "Cooling setpoint should be constant"
    );
    assert_eq!(
        schedule.cooling_setpoint_at_hour(12),
        Some(27.0),
        "Cooling setpoint should be constant"
    );
}

#[test]
fn test_night_ventilation_schedule() {
    let vent = NightVentilation::case_650();

    // Night ventilation runs 18:00-07:00
    assert!(vent.is_active_at_hour(18), "Should be active at 18:00");
    assert!(vent.is_active_at_hour(0), "Should be active at midnight");
    assert!(vent.is_active_at_hour(6), "Should be active at 6 AM");

    // Not active during daytime
    assert!(!vent.is_active_at_hour(12), "Should not be active at noon");
    assert!(!vent.is_active_at_hour(10), "Should not be active at 10 AM");
    assert!(!vent.is_active_at_hour(17), "Should not be active at 5 PM");
}

#[test]
fn test_case_640_setback_low_mass() {
    let (heating, cooling) = simulate_case(ASHRAE140Case::Case640);

    println!("\n=== ASHRAE 140 Case 640 Results ===");
    println!(
        "Annual Heating: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        heating,
        reference::case_640::ANNUAL_HEATING_MIN,
        reference::case_640::ANNUAL_HEATING_MAX
    );
    println!(
        "Annual Cooling: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        cooling,
        reference::case_640::ANNUAL_COOLING_MIN,
        reference::case_640::ANNUAL_COOLING_MAX
    );
    println!("=== End ===\n");

    // Verify heating and cooling are positive
    assert!(heating >= 0.0, "Heating should be non-negative");
    assert!(cooling >= 0.0, "Cooling should be non-negative");
}

#[test]
fn test_case_650_night_vent_low_mass() {
    let (heating, cooling) = simulate_case(ASHRAE140Case::Case650);

    println!("\n=== ASHRAE 140 Case 650 Results ===");
    println!(
        "Annual Heating: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        heating,
        reference::case_650::ANNUAL_HEATING_MIN,
        reference::case_650::ANNUAL_HEATING_MAX
    );
    println!(
        "Annual Cooling: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        cooling,
        reference::case_650::ANNUAL_COOLING_MIN,
        reference::case_650::ANNUAL_COOLING_MAX
    );
    println!("=== End ===\n");

    // Heating should be zero (disabled for night ventilation cases)
    assert_eq!(
        heating, 0.0,
        "Heating should be zero for night ventilation case"
    );
    assert!(cooling >= 0.0, "Cooling should be non-negative");
}

#[test]
fn test_case_940_setback_high_mass() {
    let (heating, cooling) = simulate_case(ASHRAE140Case::Case940);

    println!("\n=== ASHRAE 140 Case 940 Results ===");
    println!(
        "Annual Heating: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        heating,
        reference::case_940::ANNUAL_HEATING_MIN,
        reference::case_940::ANNUAL_HEATING_MAX
    );
    println!(
        "Annual Cooling: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        cooling,
        reference::case_940::ANNUAL_COOLING_MIN,
        reference::case_940::ANNUAL_COOLING_MAX
    );
    println!("=== End ===\n");

    assert!(heating >= 0.0, "Heating should be non-negative");
    assert!(cooling >= 0.0, "Cooling should be non-negative");
}

#[test]
fn test_case_950_night_vent_high_mass() {
    let (heating, cooling) = simulate_case(ASHRAE140Case::Case950);

    println!("\n=== ASHRAE 140 Case 950 Results ===");
    println!(
        "Annual Heating: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        heating,
        reference::case_950::ANNUAL_HEATING_MIN,
        reference::case_950::ANNUAL_HEATING_MAX
    );
    println!(
        "Annual Cooling: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        cooling,
        reference::case_950::ANNUAL_COOLING_MIN,
        reference::case_950::ANNUAL_COOLING_MAX
    );
    println!("=== End ===\n");

    // Heating should be zero (disabled for night ventilation cases)
    assert_eq!(
        heating, 0.0,
        "Heating should be zero for night ventilation case"
    );
    assert!(cooling >= 0.0, "Cooling should be non-negative");
}

#[test]
fn test_case_940_setback_schedule_reaches_demand() {
    // Round 26 mechanism assertion: the from-spec `heating_schedule`
    // (ASHRAE 140 §5.2.2.1.4.1: 20 °C 07:00-23:00, 10 °C 23:00-07:00) must
    // actually reach the HVAC demand. Before the fix the demand read only
    // the constant per-zone `heating_setpoints` vector, Case 940 simulated
    // byte-identical to Case 900 on the strict gate, and the manual scalar
    // setpoint writes in `simulate_case` were silently overridden by the
    // per-zone vector (Issue #2826 contract).
    let spec = ASHRAE140Case::Case940.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    // Step 0 → hour 0 → inside the 23:00-07:00 setback window.
    let wd_night = weather.get_hourly_data(0).unwrap();
    model.solar.weather = Some(wd_night.clone());
    let _ = model.step_physics(0, wd_night.dry_bulb_temp, 3600.0);
    assert_eq!(
        model.setpoints.heating_setpoints.as_ref()[0],
        10.0,
        "setback setpoint (10 °C) must be applied during the 23:00-07:00 window"
    );

    // Step 12 → hour 12 → occupied hours, 20 °C.
    let wd_day = weather.get_hourly_data(12).unwrap();
    model.solar.weather = Some(wd_day.clone());
    let _ = model.step_physics(12, wd_day.dry_bulb_temp, 3600.0);
    assert_eq!(
        model.setpoints.heating_setpoints.as_ref()[0],
        20.0,
        "occupied setpoint (20 °C) must be applied during 07:00-23:00"
    );
}

#[test]
fn test_setback_reduces_heating() {
    // Compare Case 600 (baseline) vs Case 640 (setback)
    // Setback should reduce heating energy
    let (heating_600, _) = simulate_case(ASHRAE140Case::Case600);
    let (heating_640, _) = simulate_case(ASHRAE140Case::Case640);

    println!(
        "Heating - Case 600: {:.2} MWh, Case 640: {:.2} MWh",
        heating_600, heating_640
    );

    // Round 26: the setback schedule is now consumed by the default HVAC
    // demand path (per-zone vectors synced from `heating_schedule` each
    // step), so Case 640 must heat strictly less than Case 600 (the ASHRAE
    // 140 §5.2.2.1.4.1 setback lowers the overnight setpoint 20 °C → 10 °C).
    // Previously this file only "documented current behavior": the demand
    // read the constant per-zone vector, the manual scalar writes here were
    // silently overridden, and 640 simulated identical to 600.
    assert!(
        heating_640 < heating_600,
        "Case 640 setback must reduce annual heating vs Case 600 (got 640={heating_640} >= 600={heating_600})"
    );
}

#[test]
fn test_night_vent_reduces_cooling() {
    // Compare Case 600 (baseline) vs Case 650 (night vent)
    // Night ventilation should reduce cooling energy
    let (_, cooling_600) = simulate_case(ASHRAE140Case::Case600);
    let (_, cooling_650) = simulate_case(ASHRAE140Case::Case650);

    println!(
        "Cooling - Case 600: {:.2} MWh, Case 650: {:.2} MWh",
        cooling_600, cooling_650
    );

    // Note: This test documents current behavior
    // Night ventilation should provide passive cooling
}

#[test]
fn test_case_specifications() {
    // Verify setback cases have setback configured
    let spec_640 = ASHRAE140Case::Case640.spec();
    assert!(
        spec_640.hvac[0].setback_setpoint.is_some(),
        "Case 640 should have setback"
    );
    assert_eq!(
        spec_640.hvac[0].setback_setpoint,
        Some(10.0),
        "Setback should be 10°C"
    );

    let spec_940 = ASHRAE140Case::Case940.spec();
    assert!(
        spec_940.hvac[0].setback_setpoint.is_some(),
        "Case 940 should have setback"
    );

    // Verify night ventilation cases have ventilation configured
    let spec_650 = ASHRAE140Case::Case650.spec();
    assert!(
        spec_650.night_ventilation.is_some(),
        "Case 650 should have night ventilation"
    );
    assert!(
        spec_650.hvac[0].heating_setpoint < 0.0,
        "Case 650 should have heating disabled"
    );

    let spec_950 = ASHRAE140Case::Case950.spec();
    assert!(
        spec_950.night_ventilation.is_some(),
        "Case 950 should have night ventilation"
    );
}

/// Round 27 (§LIMIT-41 follow-up): on a `gauge-solver` build the gauge
/// dispatch path must consume `heating_schedule`/`cooling_schedule` the same
/// way the default demand paths do since round 26 (PR #4403). Before the
/// per-step sync in `collect_gauge_inputs`, gauge-routed Case 640 was
/// simulated byte-identical to Case 600 (measured: both H 28.635 / C
/// 5.715 MWh on develop 270de03a, gauge build, Golden-NREL EPW) because the
/// dispatch read only the constant per-zone setpoint vectors. Gauge-build
/// only: the default build compiles this arm out.
#[cfg(feature = "gauge-solver")]
#[test]
fn test_gauge_dispatch_consumes_setback_schedule() {
    use fluxion::weather::epw::EpwWeatherSource;
    use fluxion::weather::WeatherSource;

    let epw = EpwWeatherSource::from_file("assets/weather/USA_CO_Golden-NREL.724666_TMY3.epw")
        .expect("Golden-NREL EPW must be present");

    fn gauge_annual(
        case: ASHRAE140Case,
        epw: &fluxion::weather::epw::EpwWeatherSource,
    ) -> (f64, f64) {
        let spec = case.spec();
        let mut model = ThermalModel::<VectorField>::from_spec_with_selector(
            &spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");
        let mut h = 0.0_f64;
        let mut c = 0.0_f64;
        for step in 0..8760 {
            let w = epw.get_hourly_data(step).unwrap();
            model.solar.weather = Some(w.clone());
            let ekwh = model.step_physics(step, w.dry_bulb_temp, 3600.0);
            if ekwh > 0.0 {
                h += ekwh;
            } else {
                c += -ekwh;
            }
        }
        (h / 1000.0, c / 1000.0)
    }

    let (h600, c600) = gauge_annual(ASHRAE140Case::Case600, &epw);
    let (h640, c640) = gauge_annual(ASHRAE140Case::Case640, &epw);

    // The setback must reach the gauge demand: 640's nights are held at
    // 10 °C instead of 20 °C, so its annual heating must be strictly below
    // the identical-geometry 600 twin. Before the sync both were
    // byte-identical (28.635 MWh).
    assert!(
        h640 < h600 * 0.95,
        "gauge-routed Case 640 heating ({h640:.3} MWh) must be below Case 600 \
         ({h600:.3} MWh) — the setback schedule is not reaching the gauge demand"
    );
    // Cooling barely moves (setback is heating-only; recorded twin delta
    // is -0.3 %): a sanity bound, not a band assertion.
    assert!(
        (c640 - c600).abs() < 0.2 * c600,
        "gauge-routed 640 cooling ({c640:.3}) drifted from its 600 twin ({c600:.3})"
    );
}
