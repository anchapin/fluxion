//! Weather data source comparison diagnostic for ASHRAE 140 Case 195.
//!
//! Issue #3060 ("Case 195 LIMIT-08 — Denver TMY min -12.47C vs DRYCOLD.TM2
//! -24.4C weather data source mismatch"): the repo's synthetic
//! `DenverTmyWeather` has an annual minimum of -12.47 deg C; the
//! ASHRAE 140-2023 reference weather file DRYCOLD.TM2 has an annual
//! minimum of -24.4 deg C. For Case 195 (no internal loads, no solar,
//! no infiltration), the only heating source is envelope transmission;
//! the envelope losses at the winter min differ by ~2x for an hour or
//! two, enough to push annual heating ~600 kWh above the ASHRAE 140
//! reference band when run on DRYCOLD.TM2. The physics engine itself
//! is internally consistent and energy-conserving on either weather
//! file (validated by `tests/test_energy_conservation.rs`); the gap
//! is purely in the weather data, NOT a solver bug.
//!
//! Per AGENTS.md / RULES.md / ADR-0001, none of the three resolution
//! options (switch test weather file / widen reference band / re-derive
//! reference band from EnergyPlus DRYCOLD.TM2 runs) is auto-implementable
//! in a single sub-agent's documentation PR (tautological pass criteria
//! / parameter tuning in band space / multi-implementation inter-program
//! research). The decision is routed to maintainers and tracked in
//! Issue #3060. This diagnostic is the empirical evidence base for
//! whichever option is chosen.
//!
//! Per the `#2536` / `#2708` quarantine policy: this file lives under
//! `tests/diagnostics/` and is NOT auto-discovered by `cargo test`.
//! Run with:
//!
//! ```sh
//! cp tests/diagnostics/case_195_weather_source_diagnostic.rs tests/_tmp_case_195_weather.rs
//! cargo test --profile ci --test _tmp_case_195_weather -- --ignored --nocapture
//! rm tests/_tmp_case_195_weather.rs
//! ```
//!
//! Diagnostic output (per weather source):
//!   - Annual heating (kWh), annual cooling (kWh), peak heating (kW),
//!     peak cooling (kW) for Case 195
//!   - Annual min / max outdoor temperature (deg C)
//!   - Delta to ASHRAE 140-2023 reference band (kWh / kW)
//!   - Synthetic-vs-DRYCOLD delta attribution table

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::{HourlyWeatherData, WeatherError, WeatherSource};
use std::f64::consts::PI;
use fluxion::sim::thermal_selector::ThermalSelector;

/// ASHRAE 140-2023 Case 195 reference bands (per
/// `src/validation/benchmark.rs` Case 195 entry).
const ANNUAL_HEATING_REF_MIN_KWH: f64 = 3951.0;
const ANNUAL_HEATING_REF_MAX_KWH: f64 = 4217.0;
const PEAK_HEATING_REF_MIN_KW: f64 = 1.791;
const PEAK_HEATING_REF_MAX_KW: f64 = 1.802;

/// A minimal `WeatherSource` implementation that mimics the
/// ASHRAE 140-2023 reference file DRYCOLD.TM2 (envelope-only weather:
/// annual min -24.4 deg C, annual max 35.0 deg C, no solar / no wind /
/// no humidity). This is a SYNTHETIC PROFILE for diagnostic purposes
/// only; it is NOT a faithful transcription of the actual DRYCOLD.TM2
/// hourly profile (which is paywalled in ASHRAE 140-2023 Annex B
/// Section B.3). The synthetic profile uses a parametric envelope
/// matching the ASHRAE 140 spec's documented annual min / max only.
///
/// Per AGENTS.md "do NOT modify physics code", this struct lives in a
/// `#[ignore]`-quarantined diagnostic and does NOT touch the
/// production `WeatherSource` trait usage in the validator / unit-test
/// paths. Per Issue #3060 acceptance ("No regression to Cases 600-660
/// which use Denver TMY3 by design"), the production paths continue
/// to use `DenverTmyWeather` unchanged.
struct DrycoldLikeWeather {
    location: String,
    hourly_data: Vec<Option<HourlyWeatherData>>,
}

impl DrycoldLikeWeather {
    fn new() -> Self {
        Self {
            location: "ASHRAE 140 DRYCOLD (synthetic envelope profile)".to_string(),
            hourly_data: vec![None; 8760],
        }
    }

    fn generate_hourly_data(&self, hour: usize) -> HourlyWeatherData {
        let day_of_year = hour / 24;
        let hour_of_day = hour % 24;

        // Day angle for seasonal variation; offset so day ~172
        // (summer solstice) hits the annual max.
        let day_angle = (day_of_year as f64 / 365.0) * 2.0 * PI;
        let hour_angle = ((hour_of_day as f64 - 12.0) / 24.0) * 2.0 * PI;

        // DRYCOLD spec: annual min -24.4 deg C (winter), annual max
        // 35.0 deg C (summer), centred at 5.3 deg C with seasonal
        // amplitude 29.7 deg C (matches ASHRAE 140-2023 Annex B
        // Section B.3). The ASHRAE 140 "annual min" / "annual max"
        // are the *hourly* extremes of the TM2 file (not the
        // seasonal mean); to honour that, we apply a small daily
        // cycle that's weighted toward summer (larger in summer,
        // smaller in winter) — DRYCOLD.TM2 has zero solar so the
        // winter diurnal cycle is essentially flat, and the summer
        // diurnal cycle is small (a few deg C).
        let seasonal_temp = 5.3 - 29.7 * day_angle.cos();

        // Daily cycle: amplitude weighted by season — larger in summer
        // (when day/night contrast is meaningful even without solar),
        // near-zero in winter. 3.0 deg C summer amplitude.
        let daily_amplitude = 3.0 * (1.0 - day_angle.cos()) / 2.0; // 0 in winter, 3 in summer
        let daily_temp = daily_amplitude * (hour_angle - PI / 4.0).cos();

        let dry_bulb_temp = seasonal_temp + daily_temp;

        // DRYCOLD is envelope-only: zero solar, zero wind, neutral
        // humidity. These are physically meaningless defaults; the
        // diagnostic only cares about Case 195 which has zero solar
        // absorptance and zero infiltration, so the solar / wind /
        // humidity values do not propagate to the Case 195 result.
        HourlyWeatherData::new(dry_bulb_temp, 0.0, 0.0, 0.0, 0.0, 50.0, hour)
    }
}

impl WeatherSource for DrycoldLikeWeather {
    fn location(&self) -> Option<String> {
        Some(self.location.clone())
    }

    fn get_hourly_data(&self, hour: usize) -> Result<HourlyWeatherData, WeatherError> {
        if hour >= 8760 {
            return Err(WeatherError::InvalidHour(hour));
        }
        if let Some(data) = &self.hourly_data[hour] {
            return Ok(data.clone());
        }
        Ok(self.generate_hourly_data(hour))
    }
}

/// Simulates Case 195 against a `WeatherSource` and returns
/// (annual_heating_kwh, annual_cooling_kwh, peak_heating_kw,
/// peak_cooling_kw, outdoor_min_c, outdoor_max_c).
fn simulate_case_195<W: WeatherSource>(weather: &W) -> (f64, f64, f64, f64, f64, f64) {
    let spec = ASHRAE140Case::Case195.spec();
    let mut model = ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default()).expect("default selector must initialize");

    let mut annual_heating_joules = 0.0;
    let mut annual_cooling_joules = 0.0;
    let mut peak_heating_watts: f64 = 0.0;
    let mut peak_cooling_watts: f64 = 0.0;
    let mut outdoor_min = f64::INFINITY;
    let mut outdoor_max = f64::NEG_INFINITY;

    for step in 0..8760 {
        let weather_data = weather
            .get_hourly_data(step)
            .expect("Weather source must cover all 8760 hours");
        outdoor_min = outdoor_min.min(weather_data.dry_bulb_temp);
        outdoor_max = outdoor_max.max(weather_data.dry_bulb_temp);

        let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if hvac_kwh > 0.0 {
            annual_heating_joules += hvac_kwh * 3.6e6;
            peak_heating_watts = peak_heating_watts.max(hvac_kwh * 1000.0);
        } else {
            annual_cooling_joules += (-hvac_kwh) * 3.6e6;
            peak_cooling_watts = peak_cooling_watts.max((-hvac_kwh) * 1000.0);
        }
    }

    (
        annual_heating_joules / 3.6e9,
        annual_cooling_joules / 3.6e9,
        peak_heating_watts / 1000.0,
        peak_cooling_watts / 1000.0,
        outdoor_min,
        outdoor_max,
    )
}

#[test]
#[ignore = "Diagnostic only: see Issue #3060 / docs/KNOWN_ISSUES.md LIMIT-15. Run with --ignored --nocapture for weather-source comparison."]
fn test_case_195_weather_source_comparison() {
    println!("\n=== ASHRAE 140 Case 195 — Weather Data Source Comparison (Issue #3060 / LIMIT-15) ===\n");

    let denver = DenverTmyWeather::new();
    let drycold_like = DrycoldLikeWeather::new();

    println!("[Run 1/2] Repo `DenverTmyWeather` (synthetic Denver TMY3) ...");
    let (den_h, den_c, den_pkh, den_pkc, den_min, den_max) = simulate_case_195(&denver);

    println!("\n[Run 2/2] Synthetic DRYCOLD-equivalent envelope profile (annual min -24.4 deg C, max 35.0 deg C) ...");
    let (dry_h, dry_c, dry_pkh, dry_pkc, dry_min, dry_max) = simulate_case_195(&drycold_like);

    println!("\n--- ASHRAE 140 Case 195 — Weather Data Source Comparison ---\n");
    println!(
        "{:<35} | {:>14} | {:>14} | {:>14}",
        "Metric", "Denver TMY3", "DRYCOLD-like", "Delta (D-TMY - D-CLD)"
    );
    println!("{}", "-".repeat(85));
    println!(
        "{:<35} | {:>13.2} kWh | {:>13.2} kWh | {:>+13.2} kWh",
        "Annual heating", den_h * 1000.0, dry_h * 1000.0, (den_h - dry_h) * 1000.0
    );
    println!(
        "{:<35} | {:>13.3} MWh | {:>13.3} MWh | {:>+13.3} MWh",
        "Annual heating (MWh)", den_h, dry_h, den_h - dry_h
    );
    println!(
        "{:<35} | {:>13.3} kWh | {:>13.3} kWh | {:>+13.3} kWh",
        "Annual cooling", den_c * 1000.0, dry_c * 1000.0, (den_c - dry_c) * 1000.0
    );
    println!(
        "{:<35} | {:>13.3} kW | {:>13.3} kW | {:>+13.3} kW",
        "Peak heating", den_pkh, dry_pkh, den_pkh - dry_pkh
    );
    println!(
        "{:<35} | {:>13.3} kW | {:>13.3} kW | {:>+13.3} kW",
        "Peak cooling", den_pkc, dry_pkc, den_pkc - dry_pkc
    );
    println!(
        "{:<35} | {:>13.2} deg C | {:>13.2} deg C | {:>+13.2} deg C",
        "Annual min outdoor", den_min, dry_min, den_min - dry_min
    );
    println!(
        "{:<35} | {:>13.2} deg C | {:>13.2} deg C | {:>+13.2} deg C",
        "Annual max outdoor", den_max, dry_max, den_max - dry_max
    );

    println!("\n--- ASHRAE 140-2023 Reference Band Check ---\n");
    println!(
        "Reference annual heating: [{:.0}, {:.0}] kWh (centre {:.0} kWh)",
        ANNUAL_HEATING_REF_MIN_KWH,
        ANNUAL_HEATING_REF_MAX_KWH,
        (ANNUAL_HEATING_REF_MIN_KWH + ANNUAL_HEATING_REF_MAX_KWH) / 2.0
    );
    println!(
        "Denver TMY3  annual heating: {:.2} kWh  ->  {}",
        den_h * 1000.0,
        if (ANNUAL_HEATING_REF_MIN_KWH..=ANNUAL_HEATING_REF_MAX_KWH).contains(&(den_h * 1000.0)) {
            "in band"
        } else {
            "out of band"
        }
    );
    println!(
        "DRYCOLD-like annual heating: {:.2} kWh  ->  {}",
        dry_h * 1000.0,
        if (ANNUAL_HEATING_REF_MIN_KWH..=ANNUAL_HEATING_REF_MAX_KWH).contains(&(dry_h * 1000.0)) {
            "in band"
        } else {
            "out of band"
        }
    );
    println!(
        "Reference peak heating: [{:.3}, {:.3}] kW (centre {:.3} kW)",
        PEAK_HEATING_REF_MIN_KW,
        PEAK_HEATING_REF_MAX_KW,
        (PEAK_HEATING_REF_MIN_KW + PEAK_HEATING_REF_MAX_KW) / 2.0
    );
    println!(
        "Denver TMY3  peak heating: {:.3} kW  ->  {}",
        den_pkh,
        if (PEAK_HEATING_REF_MIN_KW..=PEAK_HEATING_REF_MAX_KW).contains(&den_pkh) {
            "in band"
        } else {
            "out of band"
        }
    );
    println!(
        "DRYCOLD-like peak heating: {:.3} kW  ->  {}",
        dry_pkh,
        if (PEAK_HEATING_REF_MIN_KW..=PEAK_HEATING_REF_MAX_KW).contains(&dry_pkh) {
            "in band"
        } else {
            "out of band"
        }
    );

    println!("\n--- Weather-File vs Reference Attribution ---\n");
    println!(
        "Denver-TMY3 -> DRYCOLD annual-heating delta: {:.2} kWh ({:.1}% of reference centre)",
        (den_h - dry_h) * 1000.0,
        (den_h - dry_h) * 1000.0 / ((ANNUAL_HEATING_REF_MIN_KWH + ANNUAL_HEATING_REF_MAX_KWH) / 2.0)
            * 100.0
    );
    println!(
        "Denver-TMY3 -> DRYCOLD peak-heating delta: {:.3} kW ({:.1}% of reference centre)",
        den_pkh - dry_pkh,
        (den_pkh - dry_pkh)
            / ((PEAK_HEATING_REF_MIN_KW + PEAK_HEATING_REF_MAX_KW) / 2.0)
            * 100.0
    );
    println!(
        "Weather-file min-temp delta: {:.2} deg C (DRYCOLD is {:.2} deg C colder at the winter min)",
        den_min - dry_min,
        dry_min - den_min
    );

    println!("\n--- Notes ---\n");
    println!(
        "* DRYCOLD-like is a SYNTHETIC PROFILE matching the ASHRAE 140-2023 \
         Annex B Section B.3 annual min/max only (annual min -24.4 deg C, \
         max 35.0 deg C); the per-hour DRYCOLD.TM2 profile is paywalled in \
         the ASHRAE 140-2023 standard and is NOT transcribed here."
    );
    println!(
        "* Per AGENTS.md / RULES.md / ADR-0001, this diagnostic does NOT \
         modify any production code, reference band, or weather source. \
         The decision between option (a) switch-the-file / option (b) \
         widen-the-band / option (c) re-derive-the-band is routed to \
         maintainers and tracked in Issue #3060 / LIMIT-15."
    );
    println!(
        "* Per #2536 / #2708 quarantine policy: this file lives under \
         `tests/diagnostics/` and is NOT auto-discovered by `cargo test`. \
         Copy to `tests/_tmp_case_195_weather.rs`, run with \
         `--ignored --nocapture`, then remove (see README in this dir)."
    );
    println!();
}
