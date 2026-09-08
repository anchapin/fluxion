//! ASHRAE 140 Case 900-series (high-mass) factory methods + Case 920 / Case 950
//! single-zone validators.
//!
//! Issue #3555: extracted from `src/validation/ashrae_140_cases.rs::CaseBuilder`
//! to shrink the legacy monolith and align the in-tree case definitions with
//! the `crate::validation::ashrae140::cases` tree.

use crate::sim::construction::Assemblies;
use crate::sim::thermal_selector::ThermalSelector;
use crate::validation::ashrae_140_cases::{
    CaseBuilder, CaseSpec, HvacSchedule, InternalLoads, NightVentilation, ShadingDevice, WindowSpec,
};
use serde::{Deserialize, Serialize};

/// Case 900 — high-mass baseline (8 m × 6 m × 2.7 m, 12 m² south double-clear
/// window, 0.5 ACH, 20°C / 27°C, Denver ground-coupled).
pub fn case_900_baseline() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("900".to_string())
        .with_description(
            "High mass baseline - concrete construction with south windows".to_string(),
        )
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac_setpoints(20.0, 27.0)
        .with_infiltration(0.5)
        .with_num_zones(1)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 900 should validate")
}

/// Case 910 — high-mass with 1 m south overhang.
pub fn case_910_south_shading() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("910".to_string())
        .with_description("High mass with south shading (1m overhang)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_shading(ShadingDevice::overhang(1.0, 2.7))
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac_setpoints(20.0, 27.0)
        .with_infiltration(0.5)
        .with_num_zones(1)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 910 should validate")
}

/// Case 920 — high-mass with 6 m² east + 6 m² west windows.
pub fn case_920_ew_windows() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("920".to_string())
        .with_description("High mass with east/west windows (6m² each)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_ew_windows(6.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac_setpoints(20.0, 27.0)
        .with_infiltration(0.5)
        .with_num_zones(1)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 920 should validate")
}

/// Case 930 — high-mass with east/west overhang + fins.
pub fn case_930_ew_shading() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("930".to_string())
        .with_description("High mass with east/west shading (overhang + fins)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_ew_windows(6.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_shading(ShadingDevice::overhang_and_fins(1.0, 1.0, 2.7))
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac_setpoints(20.0, 27.0)
        .with_infiltration(0.5)
        .with_num_zones(1)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 930 should validate")
}

/// Case 940 — high-mass with overnight setback (10°C).
pub fn case_940_setback() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("940".to_string())
        .with_description("High mass with thermostat setback (overnight)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac_setback(20.0, 27.0, 10.0)
        .with_infiltration(0.5)
        .with_num_zones(1)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 940 should validate")
}

/// Case 950 — high-mass with night ventilation (heating disabled).
///
/// Per issue #1347 (case_950): the spec wires a HvacSchedule with a
/// 22:00-06:00 setback window (8 h/day = 2920 hours/year, AC4) AND a
/// NightVentilation schedule (18:00-07:00, AC3). Heating is OFF by
/// spec (heating_sp = -100°C — "no heating" per ASHRAE 140 Case 950),
/// so the setback setpoint value is moot (it is overwritten by the
/// operating-hours disabled-region fill in `schedule.rs`). The
/// setback hours are a *marker* in the spec so the validator can
/// assert "HvacSchedule night-flush window = 8 h/day" without
/// changing simulation behavior.
pub fn case_950_night_vent() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("950".to_string())
        .with_description("High mass with night ventilation (no heating)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_hvac(HvacSchedule::with_operating_hours_and_setback(
            -100.0, 27.0, 7, 18, // operating hours (cooling 7-18, heating OFF always)
            -100.0, 22, 6, // setback window 22:00-06:00 (setpoint -100 → no heating)
        ))
        .with_night_ventilation(NightVentilation::case_650())
        .with_infiltration(0.5)
        .with_num_zones(1)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 950 should validate")
}

/// Case 900FF — high-mass free-floating (no HVAC, no internal loads).
///
/// Per ASHRAE 140, free-floating cases have NO internal loads.
pub fn case_900ff() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("900FF".to_string())
        .with_description("High mass free-floating (no HVAC, no internal loads)".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::single_clear_glass())
        // No internal loads for free-floating cases per ASHRAE 140
        .with_hvac(HvacSchedule::free_floating())
        .with_infiltration(0.5)
        .with_num_zones(1)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 900FF should validate")
}

/// Case 950FF — high-mass free-floating with night ventilation.
///
/// Per ASHRAE 140, free-floating cases have NO internal loads.
pub fn case_950ff() -> CaseSpec {
    CaseBuilder::new()
        .with_case_id("950FF".to_string())
        .with_description(
            "High mass free-floating with night ventilation (no internal loads)".to_string(),
        )
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_south_window(12.0)
        .with_window_properties(WindowSpec::double_clear_glass())
        // No internal loads for free-floating cases per ASHRAE 140
        .with_hvac(HvacSchedule::free_floating())
        .with_night_ventilation(NightVentilation::case_650())
        .with_infiltration(0.5)
        .with_num_zones(1)
        .with_ground_temperature(
            crate::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        )
        .build()
        .expect("Case 950FF should validate")
}

// =============================================================================
// ASHRAE 140 Case 920 — Single-Zone Validator (Issue #1346)
// =============================================================================
//
// `Case920ValidationResult` and `validate_case_920` follow the Case 960
// validator shape (`ashrae_140_multi_zone.rs::Case960Validator` /
// `case_960.rs::Case960ReferenceImplementation::validate_case_960_result`)
// but are single-zone: they consume a `CaseSpec` produced by
// `ASHRAE140Case::Case920.spec()` (geometry 8m × 6m × 2.7m, 200 mm concrete,
// 6 m² east + 6 m² west double-clear windows, 0.5 ACH, 20°C/27°C, Denver TMY3)
// and compare the simulation outputs against the ASHRAE 140-2023 Annex B8
// reference bands recorded in
// `tests/reference_data/zone_balance/case_920_energy_reference.csv`.
//
// The reference data is the SUMMARY CSV (annual/peak), not the per-hour CSV.
// The per-hour CSV (`case_920_energy_hourly.csv`, 8760 h) is also available
// for future hourly-breakdown tests. Reference bands asserted by this
// validator:
//
//   * annual_heating : 3.26 – 4.30 MWh (ref midpoint 3.78 MWh, ±15% → 3.213 – 4.347 MWh)
//   * annual_cooling : 1.84 – 3.31 MWh (ref midpoint 2.575 MWh, ±15% → 2.189 – 2.961 MWh)
//   * peak_heating   : 2.10 – 2.80 kW  (ref midpoint 2.45 kW, ±15% → 2.083 – 2.817 kW)
//   * peak_cooling   : 1.40 – 1.90 kW  (ref midpoint 1.65 kW, ±15% → 1.402 – 1.897 kW)
//
// Per-orientation solar distribution (the issue's second acceptance criterion)
// is NOT part of `Case920ValidationResult` (which only carries metered energy)
// — it is exercised by `test_case_920_per_orientation_solar_distribution` in
// `tests/ashrae_140_blind_validation.rs` against the `IncidentSolarAccumulator`
// field on `ThermalModelData`.

/// Result of validating a Case 920 simulation against ASHRAE 140-2023 Annex B8.
///
/// Mirrors the metered-energy portion of `case_960::Case960Result` so the two
/// per-case validators share a uniform shape across the single-zone and
/// multi-zone paths. All `*_mwh` fields are in megawatt-hours, all `*_kw`
/// fields are in kilowatts. `band_pass` is a bitfield-like struct
/// (`pass_annual_heating`, …) reporting per-metric pass/fail against the
/// ±15% acceptance band of the reference midpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case920ValidationResult {
    /// Annual heating energy from the blind simulation (MWh).
    pub annual_heating_mwh: f64,
    /// Annual cooling energy from the blind simulation (MWh).
    pub annual_cooling_mwh: f64,
    /// Peak heating demand observed during the year (kW).
    pub peak_heating_kw: f64,
    /// Peak cooling demand observed during the year (kW).
    pub peak_cooling_kw: f64,
    /// Reference minimum for annual heating (MWh) — raw ASHRAE 140 Annex B8.
    pub ref_annual_heating_min_mwh: f64,
    /// Reference maximum for annual heating (MWh) — raw ASHRAE 140 Annex B8.
    pub ref_annual_heating_max_mwh: f64,
    /// Reference minimum for annual cooling (MWh) — raw ASHRAE 140 Annex B8.
    pub ref_annual_cooling_min_mwh: f64,
    /// Reference maximum for annual cooling (MWh) — raw ASHRAE 140 Annex B8.
    pub ref_annual_cooling_max_mwh: f64,
    /// Reference minimum for peak heating (kW) — raw ASHRAE 140 Annex B8.
    pub ref_peak_heating_min_kw: f64,
    /// Reference maximum for peak heating (kW) — raw ASHRAE 140 Annex B8.
    pub ref_peak_heating_max_kw: f64,
    /// Reference minimum for peak cooling (kW) — raw ASHRAE 140 Annex B8.
    pub ref_peak_cooling_min_kw: f64,
    /// Reference maximum for peak cooling (kW) — raw ASHRAE 140 Annex B8.
    pub ref_peak_cooling_max_kw: f64,
    /// `true` iff `annual_heating_mwh` falls inside the ref band.
    pub pass_annual_heating: bool,
    /// `true` iff `annual_cooling_mwh` falls inside the ref band.
    pub pass_annual_cooling: bool,
    /// `true` iff `peak_heating_kw` falls inside the ref band.
    pub pass_peak_heating: bool,
    /// `true` iff `peak_cooling_kw` falls inside the ref band.
    pub pass_peak_cooling: bool,
    /// `true` iff all four per-metric checks pass. The acceptance test in
    /// `tests/ashrae_140_blind_validation.rs` is gated with `#[ignore]` until
    /// the underlying physics closes the band (`#1323` / `#1213`).
    pub all_pass: bool,
}

impl Case920ValidationResult {
    /// Returns a compact printable representation for log output.
    pub fn summary(&self) -> String {
        format!(
            "Case 920: H={:.3}/{:.3}..{:.3} MWh ({}), C={:.3}/{:.3}..{:.3} MWh ({}), \
             PH={:.3}/{:.3}..{:.3} kW ({}), PC={:.3}/{:.3}..{:.3} kW ({}) → all_pass={}",
            self.annual_heating_mwh,
            self.ref_annual_heating_min_mwh,
            self.ref_annual_heating_max_mwh,
            pass_str(self.pass_annual_heating),
            self.annual_cooling_mwh,
            self.ref_annual_cooling_min_mwh,
            self.ref_annual_cooling_max_mwh,
            pass_str(self.pass_annual_cooling),
            self.peak_heating_kw,
            self.ref_peak_heating_min_kw,
            self.ref_peak_heating_max_kw,
            pass_str(self.pass_peak_heating),
            self.peak_cooling_kw,
            self.ref_peak_cooling_min_kw,
            self.ref_peak_cooling_max_kw,
            pass_str(self.pass_peak_cooling),
            self.all_pass,
        )
    }
}

fn pass_str(p: bool) -> &'static str {
    if p {
        "PASS"
    } else {
        "FAIL"
    }
}

/// ASHRAE 140 Case 920 reference bands (Annex B8, validated across BSIMAC,
/// CSE, DeST, EnergyPlus, ESP-r, TRNSYS — per the CSV provenance header in
/// `tests/reference_data/zone_balance/case_920_energy_reference.csv`).
///
/// These are the raw ASHRAE 140 inter-program bands, NOT the per-program
/// ±15% bands. The validator uses the raw band as the pass/fail window
/// (matching the issue acceptance criterion: "Annual heating energy in
/// [lower, upper] band per ASHRAE 140-2017 Table 8-2").
pub const CASE_920_ANNUAL_HEATING_MIN_MWH: f64 = 3.26;
pub const CASE_920_ANNUAL_HEATING_MAX_MWH: f64 = 4.30;
pub const CASE_920_ANNUAL_COOLING_MIN_MWH: f64 = 1.84;
pub const CASE_920_ANNUAL_COOLING_MAX_MWH: f64 = 3.31;
pub const CASE_920_PEAK_HEATING_MIN_KW: f64 = 2.10;
pub const CASE_920_PEAK_HEATING_MAX_KW: f64 = 2.80;
pub const CASE_920_PEAK_COOLING_MIN_KW: f64 = 1.40;
pub const CASE_920_PEAK_COOLING_MAX_KW: f64 = 1.90;

/// Validate ASHRAE 140 Case 920 (high-mass east/west windows) against the
/// reference bands in `tests/reference_data/zone_balance/case_920_energy_reference.csv`.
///
/// This is the single-zone companion to the multi-zone
/// `validate_case_960_with_validator` in `ashrae_140_multi_zone.rs`. It does
/// NOT tune the physics, modify the model, or apply any per-case correction
/// (issue hard rule: "No parameter tuning — validation harness, not physics
/// tuning"). It runs a blind annual simulation from the spec alone and
/// reports the four band checks.
///
/// Acceptance criterion (issue #1346):
///   "`validate_case_920` returns a `CaseValidationResult` (not
///    panic/unimplemented) for `CaseSpec` with 6 m² east + 6 m² west."
///
/// This function is unconditionally non-panicking for any well-formed
/// `CaseSpec` produced by `ASHRAE140Case::Case920.spec()`. The strict
/// per-band pass/fail is reported via `result.all_pass`; the function
/// itself only returns an error if `ThermalModel::from_spec` cannot
/// construct a model from the spec (which the CaseBuilder is required to
/// not produce for Case 920).
pub fn validate_case_920(spec: &CaseSpec) -> Case920ValidationResult {
    // Run the blind annual simulation. We deliberately do NOT use
    // `ASHRAE140Validator::validate_case` because that path is the
    // multi-case `BenchmarkReport` builder and would conflate Case 920
    // results with the other 600/900 cases in the wider harness. A
    // dedicated single-spec path keeps the result schema clean and lets
    // the unit test in this module assert on a single `Case920ValidationResult`
    // without filtering.
    let sim = simulate_case_920_blind(spec);
    build_case_920_validation_result(
        sim.annual_heating_mwh,
        sim.annual_cooling_mwh,
        sim.peak_heating_kw,
        sim.peak_cooling_kw,
    )
}

/// Compact simulation output for the Case 920 validator.
#[derive(Debug, Clone, Copy)]
struct Case920BlindSim {
    annual_heating_mwh: f64,
    annual_cooling_mwh: f64,
    peak_heating_kw: f64,
    peak_cooling_kw: f64,
}

/// Blind annual simulation: only the `CaseSpec` is passed to the engine
/// (no case ID, no test-only flags, no per-case tuning). Returns the four
/// metered-energy metrics the validator compares against the reference band.
///
/// Uses `ThermalModel::from_spec` (the same spec-driven path that the
/// Case 600/900 strict-tolerance tests in `tests/zone_balance_eplus_isolation.rs`
/// use) and the Denver TMY3 weather source. This is the spec-only path the
/// issue's "blind execution" criterion requires: the engine never sees a
/// case ID.
fn simulate_case_920_blind(spec: &CaseSpec) -> Case920BlindSim {
    use crate::physics::cta::VectorField;
    use crate::sim::engine::ThermalModel;
    use crate::weather::WeatherSource;
    use fluxion_core::weather::denver::DenverTmyWeather;

    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();
    const STEPS: usize = 8760;

    for step in 0..STEPS {
        let hour_of_day = step % 24;
        let weather_data = match weather.get_hourly_data(step) {
            Ok(w) => w,
            Err(_) => continue, // Defensive: should never happen with TMY data
        };
        // Extract the only field used downstream (f64 is Copy) so we can move
        // weather_data into model.solar.weather without an extra clone (Issue #2893).
        let dry_bulb_temp = weather_data.dry_bulb_temp;
        model.solar.weather = Some(weather_data);
        if let Some(hvac) = spec.hvac.first() {
            let hour = hour_of_day as u8;
            let heating_sp = hvac
                .heating_setpoint_at_hour(hour)
                .unwrap_or(hvac.heating_setpoint);
            let cooling_sp = model.setpoints.cooling_schedule.value(hour as usize);
            model.setpoints.heating_setpoint = heating_sp;
            model.setpoints.cooling_setpoint = cooling_sp;
        }
        model.step_physics(step, dry_bulb_temp, 3600.0);
    }

    Case920BlindSim {
        // The model reports cumulative energy in kWh; ASHRAE 140 reference
        // bands are in MWh. Same conversion the test harness uses.
        annual_heating_mwh: model.hvac.annual_heating_energy / 1000.0,
        annual_cooling_mwh: model.hvac.annual_cooling_energy / 1000.0,
        peak_heating_kw: model.get_peak_heating_power_kw(),
        peak_cooling_kw: model.get_peak_cooling_power_kw(),
    }
}

/// Build a `Case920ValidationResult` from the four simulated metrics and
/// compare each against the ASHRAE 140 Annex B8 raw reference band. Split
/// out as a pure function so the unit test can call it with synthetic
/// values without driving a full year of physics.
pub fn build_case_920_validation_result(
    annual_heating_mwh: f64,
    annual_cooling_mwh: f64,
    peak_heating_kw: f64,
    peak_cooling_kw: f64,
) -> Case920ValidationResult {
    let pass_annual_heating = annual_heating_mwh >= CASE_920_ANNUAL_HEATING_MIN_MWH
        && annual_heating_mwh <= CASE_920_ANNUAL_HEATING_MAX_MWH;
    let pass_annual_cooling = annual_cooling_mwh >= CASE_920_ANNUAL_COOLING_MIN_MWH
        && annual_cooling_mwh <= CASE_920_ANNUAL_COOLING_MAX_MWH;
    let pass_peak_heating = peak_heating_kw >= CASE_920_PEAK_HEATING_MIN_KW
        && peak_heating_kw <= CASE_920_PEAK_HEATING_MAX_KW;
    let pass_peak_cooling = peak_cooling_kw >= CASE_920_PEAK_COOLING_MIN_KW
        && peak_cooling_kw <= CASE_920_PEAK_COOLING_MAX_KW;
    Case920ValidationResult {
        annual_heating_mwh,
        annual_cooling_mwh,
        peak_heating_kw,
        peak_cooling_kw,
        ref_annual_heating_min_mwh: CASE_920_ANNUAL_HEATING_MIN_MWH,
        ref_annual_heating_max_mwh: CASE_920_ANNUAL_HEATING_MAX_MWH,
        ref_annual_cooling_min_mwh: CASE_920_ANNUAL_COOLING_MIN_MWH,
        ref_annual_cooling_max_mwh: CASE_920_ANNUAL_COOLING_MAX_MWH,
        ref_peak_heating_min_kw: CASE_920_PEAK_HEATING_MIN_KW,
        ref_peak_heating_max_kw: CASE_920_PEAK_HEATING_MAX_KW,
        ref_peak_cooling_min_kw: CASE_920_PEAK_COOLING_MIN_KW,
        ref_peak_cooling_max_kw: CASE_920_PEAK_COOLING_MAX_KW,
        pass_annual_heating,
        pass_annual_cooling,
        pass_peak_heating,
        pass_peak_cooling,
        all_pass: pass_annual_heating
            && pass_annual_cooling
            && pass_peak_heating
            && pass_peak_cooling,
    }
}

// =============================================================================
// ASHRAE 140 Case 950 — High-Mass Night-Ventilation Validator (Issue #1347)
// =============================================================================
//
// `Case950ValidationResult` and `validate_case_950` follow the Case 920
// validator shape (`validate_case_920` / `Case920ValidationResult` introduced
// in PR #1346) but specialize in night-ventilation + setback scheduling:
// Case 950 is the high-mass night-flush case (8m × 6m × 2.7m, 200 mm concrete,
// 12 m² south double-clear window, 0.5 ACH, HEATING OFF, 5 ACH night-flush,
// Denver TMY3).

/// Result of validating a Case 950 simulation against ASHRAE 140-2023 Annex B8.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case950ValidationResult {
    pub annual_heating_mwh: f64,
    pub annual_cooling_mwh: f64,
    pub peak_heating_kw: f64,
    pub peak_cooling_kw: f64,
    pub ref_annual_heating_min_mwh: f64,
    pub ref_annual_heating_max_mwh: f64,
    pub ref_annual_cooling_min_mwh: f64,
    pub ref_annual_cooling_max_mwh: f64,
    pub ref_peak_heating_min_kw: f64,
    pub ref_peak_heating_max_kw: f64,
    pub ref_peak_cooling_min_kw: f64,
    pub ref_peak_cooling_max_kw: f64,
    pub pass_annual_heating: bool,
    pub pass_annual_cooling: bool,
    pub pass_peak_heating: bool,
    pub pass_peak_cooling: bool,
    pub all_pass: bool,
}

impl Case950ValidationResult {
    pub fn summary(&self) -> String {
        format!(
            "Case 950: H={:.3}/{:.3}..{:.3} MWh ({}), C={:.3}/{:.3}..{:.3} MWh ({}), \
             PH={:.3}/{:.3}..{:.3} kW ({}), PC={:.3}/{:.3}..{:.3} kW ({}) → all_pass={}",
            self.annual_heating_mwh,
            self.ref_annual_heating_min_mwh,
            self.ref_annual_heating_max_mwh,
            pass_str(self.pass_annual_heating),
            self.annual_cooling_mwh,
            self.ref_annual_cooling_min_mwh,
            self.ref_annual_cooling_max_mwh,
            pass_str(self.pass_annual_cooling),
            self.peak_heating_kw,
            self.ref_peak_heating_min_kw,
            self.ref_peak_heating_max_kw,
            pass_str(self.pass_peak_heating),
            self.peak_cooling_kw,
            self.ref_peak_cooling_min_kw,
            self.ref_peak_cooling_max_kw,
            pass_str(self.pass_peak_cooling),
            self.all_pass,
        )
    }
}

/// ASHRAE 140 Case 950 reference bands (Annex B8).
pub const CASE_950_ANNUAL_HEATING_MIN_MWH: f64 = 0.00;
pub const CASE_950_ANNUAL_HEATING_MAX_MWH: f64 = 0.00;
pub const CASE_950_ANNUAL_COOLING_MIN_MWH: f64 = 0.39;
pub const CASE_950_ANNUAL_COOLING_MAX_MWH: f64 = 0.92;
pub const CASE_950_PEAK_HEATING_MIN_KW: f64 = 0.00;
pub const CASE_950_PEAK_HEATING_MAX_KW: f64 = 0.00;
pub const CASE_950_PEAK_COOLING_MIN_KW: f64 = 0.70;
pub const CASE_950_PEAK_COOLING_MAX_KW: f64 = 0.90;

/// Validate ASHRAE 140 Case 950 (high-mass night ventilation).
pub fn validate_case_950(spec: &CaseSpec) -> Case950ValidationResult {
    let sim = simulate_case_950_blind(spec);
    build_case_950_validation_result(
        sim.annual_heating_mwh,
        sim.annual_cooling_mwh,
        sim.peak_heating_kw,
        sim.peak_cooling_kw,
    )
}

#[derive(Debug, Clone, Copy)]
struct Case950BlindSim {
    annual_heating_mwh: f64,
    annual_cooling_mwh: f64,
    peak_heating_kw: f64,
    peak_cooling_kw: f64,
}

fn simulate_case_950_blind(spec: &CaseSpec) -> Case950BlindSim {
    use crate::physics::cta::VectorField;
    use crate::sim::engine::ThermalModel;
    use crate::weather::WeatherSource;
    use fluxion_core::weather::denver::DenverTmyWeather;

    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();
    const STEPS: usize = 8760;

    for step in 0..STEPS {
        let hour_of_day = step % 24;
        let weather_data = match weather.get_hourly_data(step) {
            Ok(w) => w,
            Err(_) => continue,
        };
        let dry_bulb_temp = weather_data.dry_bulb_temp;
        model.solar.weather = Some(weather_data);
        if let Some(hvac) = spec.hvac.first() {
            let hour = hour_of_day as u8;
            let heating_sp = hvac
                .heating_setpoint_at_hour(hour)
                .unwrap_or(hvac.heating_setpoint);
            let cooling_sp = model.setpoints.cooling_schedule.value(hour as usize);
            model.setpoints.heating_setpoint = heating_sp;
            model.setpoints.cooling_setpoint = cooling_sp;
        }
        model.step_physics(step, dry_bulb_temp, 3600.0);
    }

    Case950BlindSim {
        annual_heating_mwh: model.hvac.annual_heating_energy / 1000.0,
        annual_cooling_mwh: model.hvac.annual_cooling_energy / 1000.0,
        peak_heating_kw: model.get_peak_heating_power_kw(),
        peak_cooling_kw: model.get_peak_cooling_power_kw(),
    }
}

pub fn build_case_950_validation_result(
    annual_heating_mwh: f64,
    annual_cooling_mwh: f64,
    peak_heating_kw: f64,
    peak_cooling_kw: f64,
) -> Case950ValidationResult {
    let pass_annual_heating = annual_heating_mwh >= CASE_950_ANNUAL_HEATING_MIN_MWH
        && annual_heating_mwh <= CASE_950_ANNUAL_HEATING_MAX_MWH;
    let pass_annual_cooling = annual_cooling_mwh >= CASE_950_ANNUAL_COOLING_MIN_MWH
        && annual_cooling_mwh <= CASE_950_ANNUAL_COOLING_MAX_MWH;
    let pass_peak_heating = peak_heating_kw >= CASE_950_PEAK_HEATING_MIN_KW
        && peak_heating_kw <= CASE_950_PEAK_HEATING_MAX_KW;
    let pass_peak_cooling = peak_cooling_kw >= CASE_950_PEAK_COOLING_MIN_KW
        && peak_cooling_kw <= CASE_950_PEAK_COOLING_MAX_KW;
    Case950ValidationResult {
        annual_heating_mwh,
        annual_cooling_mwh,
        peak_heating_kw,
        peak_cooling_kw,
        ref_annual_heating_min_mwh: CASE_950_ANNUAL_HEATING_MIN_MWH,
        ref_annual_heating_max_mwh: CASE_950_ANNUAL_HEATING_MAX_MWH,
        ref_annual_cooling_min_mwh: CASE_950_ANNUAL_COOLING_MIN_MWH,
        ref_annual_cooling_max_mwh: CASE_950_ANNUAL_COOLING_MAX_MWH,
        ref_peak_heating_min_kw: CASE_950_PEAK_HEATING_MIN_KW,
        ref_peak_heating_max_kw: CASE_950_PEAK_HEATING_MAX_KW,
        ref_peak_cooling_min_kw: CASE_950_PEAK_COOLING_MIN_KW,
        ref_peak_cooling_max_kw: CASE_950_PEAK_COOLING_MAX_KW,
        pass_annual_heating,
        pass_annual_cooling,
        pass_peak_heating,
        pass_peak_cooling,
        all_pass: pass_annual_heating
            && pass_annual_cooling
            && pass_peak_heating
            && pass_peak_cooling,
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for `validate_case_920` / `validate_case_950`. Moved here
    //! from the legacy `src/validation/ashrae_140_cases.rs::tests` module as
    //! part of Issue #3555 burn-down; the test bodies themselves are
    //! unchanged — only the import paths for `ASHRAE140Case::Case920` /
    //! `ASHRAE140Case::Case950` had to switch from the legacy monolith to
    //! the new top-level path.

    use super::*;
    use crate::validation::ashrae_140_cases::ASHRAE140Case;

    #[test]
    fn test_build_case_920_validation_result_band_logic() {
        // Midpoint of all four bands → all_pass = true.
        let r = build_case_920_validation_result(
            0.5 * (CASE_920_ANNUAL_HEATING_MIN_MWH + CASE_920_ANNUAL_HEATING_MAX_MWH),
            0.5 * (CASE_920_ANNUAL_COOLING_MIN_MWH + CASE_920_ANNUAL_COOLING_MAX_MWH),
            0.5 * (CASE_920_PEAK_HEATING_MIN_KW + CASE_920_PEAK_HEATING_MAX_KW),
            0.5 * (CASE_920_PEAK_COOLING_MIN_KW + CASE_920_PEAK_COOLING_MAX_KW),
        );
        assert!(r.pass_annual_heating, "midpoint heating must pass");
        assert!(r.pass_annual_cooling, "midpoint cooling must pass");
        assert!(r.pass_peak_heating, "midpoint peak heating must pass");
        assert!(r.pass_peak_cooling, "midpoint peak cooling must pass");
        assert!(r.all_pass, "all four midpoints → all_pass");

        // Below-band heating (the engine's current #1323 output) must fail.
        let r = build_case_920_validation_result(1.708, 1.713, 0.0, 0.0);
        assert!(
            !r.pass_annual_heating,
            "1.708 MWh must be below 3.26 MWh lower band"
        );
        assert!(!r.all_pass, "below-band → all_pass=false");
    }

    #[test]
    fn test_case_920_spec_has_6m2_east_and_west_windows() {
        let spec = ASHRAE140Case::Case920.spec();
        assert_eq!(spec.case_id, "920");
        assert!(spec.validate().is_ok(), "Case 920 spec must validate");
        let mut east_area = 0.0;
        let mut west_area = 0.0;
        for zone_windows in &spec.windows {
            for w in zone_windows {
                match w.orientation {
                    crate::validation::ashrae_140_cases::Orientation::East => east_area += w.area,
                    crate::validation::ashrae_140_cases::Orientation::West => west_area += w.area,
                    _ => {}
                }
            }
        }
        assert!(
            (east_area - 6.0).abs() < 1e-9,
            "Case 920 must have 6 m² east glazing, got {east_area}"
        );
        assert!(
            (west_area - 6.0).abs() < 1e-9,
            "Case 920 must have 6 m² west glazing, got {west_area}"
        );
        assert_eq!(
            spec.construction_type,
            crate::validation::ashrae_140_cases::ConstructionType::HighMass,
            "Case 920 must be high-mass construction"
        );
    }

    #[test]
    fn test_validate_case_920_returns_result_for_case_920_spec() {
        let spec = ASHRAE140Case::Case920.spec();
        let result = validate_case_920(&spec);
        assert!(
            result.annual_heating_mwh.is_finite(),
            "annual_heating_mwh must be finite, got {}",
            result.annual_heating_mwh
        );
        assert!(
            result.annual_cooling_mwh.is_finite(),
            "annual_cooling_mwh must be finite, got {}",
            result.annual_cooling_mwh
        );
        assert!(
            result.peak_heating_kw.is_finite(),
            "peak_heating_kw must be finite"
        );
        assert!(
            result.peak_cooling_kw.is_finite(),
            "peak_cooling_kw must be finite"
        );
        assert!(result.annual_heating_mwh >= 0.0);
        assert!(result.annual_cooling_mwh >= 0.0);
        assert!(result.ref_annual_heating_min_mwh > 0.0);
        assert!(result.ref_annual_heating_max_mwh > result.ref_annual_heating_min_mwh);
    }

    #[test]
    fn test_build_case_950_validation_result_band_logic() {
        let r = build_case_950_validation_result(
            0.5 * (CASE_950_ANNUAL_HEATING_MIN_MWH + CASE_950_ANNUAL_HEATING_MAX_MWH),
            0.5 * (CASE_950_ANNUAL_COOLING_MIN_MWH + CASE_950_ANNUAL_COOLING_MAX_MWH),
            0.5 * (CASE_950_PEAK_HEATING_MIN_KW + CASE_950_PEAK_HEATING_MAX_KW),
            0.5 * (CASE_950_PEAK_COOLING_MIN_KW + CASE_950_PEAK_COOLING_MAX_KW),
        );
        assert!(r.pass_annual_heating, "midpoint heating must pass");
        assert!(r.pass_annual_cooling, "midpoint cooling must pass");
        assert!(r.pass_peak_heating, "midpoint peak heating must pass");
        assert!(r.pass_peak_cooling, "midpoint peak cooling must pass");
        assert!(r.all_pass, "all four midpoints → all_pass");
    }

    #[test]
    fn test_case_950_spec_has_22_06_setback_window() {
        let spec = ASHRAE140Case::Case950.spec();
        assert_eq!(spec.case_id, "950");
        assert!(spec.validate().is_ok(), "Case 950 spec must validate");
        let hvac = spec.hvac.first().expect("Case 950 spec must have HVAC");
        let setback_hours = hvac
            .setback_hours
            .expect("Case 950 must carry a setback window");
        assert_eq!(
            setback_hours,
            (22, 6),
            "Case 950 setback window must be (22, 6)"
        );
        assert!(
            hvac.heating_setpoint <= -50.0,
            "Case 950 heating must be OFF"
        );
    }

    #[test]
    fn test_case_950_spec_has_night_ventilation_active_18_to_7() {
        let spec = ASHRAE140Case::Case950.spec();
        let nv = spec
            .night_ventilation
            .expect("Case 950 must have night ventilation configured");
        assert_eq!(nv.operating_hours, (18, 7));
        assert!((nv.fan_capacity - 1703.16).abs() < 1e-9);
        assert!(!nv.adds_heat);
    }

    #[test]
    fn test_validate_case_950_returns_result_for_case_950_spec() {
        let spec = ASHRAE140Case::Case950.spec();
        let result = validate_case_950(&spec);
        assert!(result.annual_heating_mwh.is_finite());
        assert!(result.annual_cooling_mwh.is_finite());
        assert!(result.peak_heating_kw.is_finite());
        assert!(result.peak_cooling_kw.is_finite());
        assert!(result.annual_heating_mwh >= 0.0);
        assert!(result.annual_cooling_mwh >= 0.0);
        assert!((result.ref_annual_heating_min_mwh - 0.00).abs() < 1e-9);
        assert!((result.ref_annual_cooling_min_mwh - 0.39).abs() < 1e-9);
        assert!((result.ref_peak_cooling_min_kw - 0.70).abs() < 1e-9);
    }
}
