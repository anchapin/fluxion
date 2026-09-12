// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! ASHRAE 140 weather presets for the WASM surface (Issue #3624).
//!
//! [`crate::FluidSimulation::step`] previously hardcoded
//! `outdoor_temp = 20.0`, which pinned the ASHRAE 600 envelope inside its
//! 20–27 °C deadband and structurally produced zero HVAC energy — the root
//! cause of the zeroed-energy FFI failure tracked by issue #3624. This
//! module wires a real annual outdoor-temperature schedule into the step
//! loop.
//!
//! The schedule is the WD600 (ASHRAE 140 Section B2) synthetic annual
//! weather — the same drive the engine-side Case 600 validation tests use
//! (`tests/ashrae_140_case_600_series.rs` loads `assets/weather/WD600.epw`).
//! The full 8760-hour dry-bulb series is embedded in
//! [`crate::wd600_data::WD600_DRY_BULB_C`] because `fluxion-core` (and the
//! filesystem-backed `EpwWeatherSource`) is deliberately unavailable on
//! `wasm32` targets; the native fidelity test re-parses the EPW fixture and
//! asserts the embedded series matches, so the two cannot drift.

use std::sync::OnceLock;

use crate::wd600_data::WD600_DRY_BULB_C;

/// Named weather preset that selects the WD600 annual schedule.
///
/// Matches the `weather: "ASHRAE_600"` identifier the ASHRAE 140 FFI smoke
/// tests pass in `FluidSimulationConfig`.
pub const ASHRAE_600_WEATHER_KEY: &str = "ASHRAE_600";

/// Parse (once) and return the embedded WD600 hourly dry-bulb schedule.
///
/// 8760 entries, hour 0 = January 1st 00:00, matching the EPW fixture
/// ordering so `schedule[step]` lines up with
/// `EpwWeatherSource::get_hourly_data(step)` on the engine side.
pub fn wd600_dry_bulb() -> &'static Vec<f64> {
    static SCHEDULE: OnceLock<Vec<f64>> = OnceLock::new();
    SCHEDULE.get_or_init(|| {
        WD600_DRY_BULB_C
            .split(',')
            .map(|s| {
                s.parse::<f64>().unwrap_or_else(|e| {
                    panic!("embedded WD600 dry-bulb constant failed to parse: {e}")
                })
            })
            .collect()
    })
}

/// Resolve a named weather preset to its annual dry-bulb schedule.
///
/// Returns `None` for unknown names — callers then fall back to the
/// neutral 20 °C outdoor default (pre-#3624 behavior, kept for
/// backward compatibility with configs that never requested a weather
/// drive).
pub fn preset_schedule(name: &str) -> Option<&'static Vec<f64>> {
    match name {
        ASHRAE_600_WEATHER_KEY | "WD600" => Some(wd600_dry_bulb()),
        _ => None,
    }
}

/// Outdoor temperature for timestep `steps_taken` (0-based) of `schedule`.
///
/// Wraps modulo the schedule length (typical-TMY repetition when a
/// multi-year run outlives the 8760-hour schedule). An empty schedule
/// falls back to the neutral 20 °C default so the helper is total.
pub fn schedule_outdoor_at(steps_taken: usize, schedule: &[f64]) -> f64 {
    if schedule.is_empty() {
        20.0
    } else {
        schedule[steps_taken % schedule.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wd600_schedule_has_8760_finite_entries() {
        let sched = wd600_dry_bulb();
        assert_eq!(sched.len(), 8760, "WD600 must cover one full year");
        assert!(sched.iter().all(|v| v.is_finite()));
        // Physical sanity for the ASHRAE 140 synthetic drive: never below
        // -40 °C or above +45 °C.
        let min = sched.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = sched.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert!(min > -40.0, "WD600 minimum {min} below physical bound");
        assert!(max < 45.0, "WD600 maximum {max} above physical bound");
    }

    #[test]
    fn wd600_first_hour_matches_epw_fixture() {
        // Hour 0 of WD600.epw (Jan 1, 00:00) is -18.0 °C.
        assert!((wd600_dry_bulb()[0] - (-18.0)).abs() < 0.051);
    }

    #[test]
    fn schedule_outdoor_at_wraps_modulo_length() {
        let sched = vec![10.0, 30.0];
        assert_eq!(schedule_outdoor_at(0, &sched), 10.0);
        assert_eq!(schedule_outdoor_at(1, &sched), 30.0);
        assert_eq!(schedule_outdoor_at(2, &sched), 10.0);
        assert_eq!(schedule_outdoor_at(8761, &sched), 30.0);
    }

    #[test]
    fn schedule_outdoor_at_empty_schedule_falls_back_to_neutral() {
        assert_eq!(schedule_outdoor_at(0, &[]), 20.0);
    }

    #[test]
    fn preset_schedule_resolves_ashrae_600_aliases() {
        assert!(preset_schedule("ASHRAE_600").is_some());
        assert!(preset_schedule("WD600").is_some());
        assert!(preset_schedule("TMY3_CHICAGO").is_none());
        assert!(preset_schedule("").is_none());
    }

    /// Native-only fidelity gate: the embedded series must match the
    /// canonical `assets/weather/WD600.epw` fixture within 0.05 °C (the
    /// fixture carries one decimal digit). This is what makes the generated
    /// [`crate::wd600_data::WD600_DRY_BULB_C`] constant trustworthy — a
    /// stale regeneration or hand-edit turns this test red.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn wd600_schedule_matches_epw_fixture() {
        let epw_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/weather/WD600.epw");
        let text = std::fs::read_to_string(epw_path).unwrap_or_else(|e| {
            panic!("WD600.epw fixture must be readable for the parity test: {e}")
        });
        let epw_dry_bulb: Vec<f64> = text
            .lines()
            .skip(8) // EPW header: 8 lines
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                l.split(',')
                    .nth(6)
                    .unwrap_or_else(|| panic!("EPW data line missing dry-bulb column"))
                    .trim()
                    .parse()
                    .expect("EPW dry-bulb column must parse as f64")
            })
            .collect();
        assert_eq!(epw_dry_bulb.len(), 8760, "EPW fixture must cover one year");

        let sched = wd600_dry_bulb();
        let max_delta = epw_dry_bulb
            .iter()
            .zip(sched.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, |a: f64, b| a.max(b));
        assert!(
            max_delta < 0.051,
            "embedded WD600 schedule drifted from assets/weather/WD600.epw: max |delta| = {max_delta}"
        );
    }
}
