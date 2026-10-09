// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT
//
// [AI-assisted] Physics-loop diagnostic (NOT shipped): per-orientation annual
// irradiance harness reproducing the limit-35 §13 east/west asymmetry finding
// and testing candidate intra-hour solar-integration conventions against the
// in-repo E+ 25.2 reference (tests/reference_data/solar/
// ashrae_140_surface_incident_solar.csv, Golden-NREL 724666 TMY3, Timestep=1).

use fluxion::solar::solar_position::calculate_solar_position;
use fluxion::solar::surface_irradiance::{calculate_surface_irradiance, Orientation};
use fluxion::weather::epw::EpwWeatherSource;
use fluxion::weather::WeatherSource;

// Golden-NREL 724666 TMY3 LOCATION header: 39.742 N, -105.178 E, UTC-7.
const LAT: f64 = 39.742;
const LON: f64 = -105.178;
const UTC_OFFSET: Option<f64> = Some(-7.0);

const ORIENTATIONS: [(usize, &str); 5] = [
    (0, "North"),
    (1, "East"),
    (2, "South"),
    (3, "West"),
    (4, "Up"),
];

// E+ 25.2 reference annual totals, beam + sky, Wh/m2·yr (summary CSV).
const REF: [f64; 5] = [278_100.0, 902_100.0, 1_150_700.0, 780_100.0, 1_621_300.0];

fn month_day(doy0: usize) -> (u32, u32) {
    let cum = [0usize, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365];
    let doy = doy0 + 1; // 1-based
    for m in 0..12 {
        if doy <= cum[m + 1] {
            return (m as u32 + 1, (doy - cum[m]) as u32);
        }
    }
    (12, 31)
}

/// Sun-position convention under test.
#[derive(Clone, Copy)]
enum Mode {
    /// Instantaneous position at hour-start + `offset` hours (offset 0 = engine).
    Offset(f64),
    /// Position averaged over N substeps uniformly covering the hour.
    SubAvg(u32),
    /// Linear sweep: beam = DNI * mean of clamped cos(AOI) at N substeps.
    SubBeamAvg(u32),
}

fn annual_sums(mode: Mode, weather: &EpwWeatherSource) -> [f64; 5] {
    let mut sums = [0.0f64; 5];
    for i in 0..8760 {
        let wd = weather.get_hourly_data(i).expect("hourly data");
        let (month, day) = month_day(wd.day_of_year());
        let h0 = wd.hour_of_day() as f64; // hour-start convention: engine evaluates here
        let doy = wd.day_of_year() + 1;

        // Evaluate position(s) per mode.
        let mut positions: Vec<_> = Vec::new();
        match mode {
            Mode::Offset(off) => {
                positions.push(calculate_solar_position(
                    LAT, LON, 2024, month, day, h0 + off, UTC_OFFSET,
                ));
            }
            Mode::SubAvg(n) | Mode::SubBeamAvg(n) => {
                for k in 0..n {
                    let h = h0 + (k as f64 + 0.5) / n as f64;
                    positions.push(calculate_solar_position(
                        LAT, LON, 2024, month, day, h, UTC_OFFSET,
                    ));
                }
            }
        }

        for &(oi, _name) in ORIENTATIONS.iter() {
            let orientation = match oi {
                0 => Orientation::North,
                1 => Orientation::East,
                2 => Orientation::South,
                3 => Orientation::West,
                _ => Orientation::Up,
            };
            let irr = match mode {
                Mode::SubBeamAvg(n) => {
                    // Integrate only the beam geometrically; keep the diffuse
                    // evaluated at the hour center.
                    let center = positions[n as usize / 2];
                    let beam = positions
                        .iter()
                        .map(|p| {
                            if !p.is_above_horizon() {
                                0.0
                            } else {
                                let (tilt_deg, az_deg) =
                                    fluxion::solar::surface_irradiance::orientation_to_angles(
                                        orientation,
                                    );
                                (wd.dni * p.incidence_cosine(tilt_deg, az_deg)).max(0.0)
                            }
                        })
                        .sum::<f64>()
                        / n as f64;
                    // Reuse calculate_surface_irradiance for diffuse/ground at
                    // the center position, then swap in the integrated beam.
                    let base = calculate_surface_irradiance(
                        &center, wd.dni, wd.dhi, None, orientation, 0.2, doy,
                    );
                    fluxion::solar::surface_irradiance::SurfaceIrradiance::new(
                        beam, base.diffuse_wm2, base.ground_reflected_wm2,
                    )
                }
                _ => {
                    let p = &positions[0];
                    calculate_surface_irradiance(p, wd.dni, wd.dhi, None, orientation, 0.2, doy)
                }
            };
            sums[oi] += irr.beam_wm2 + irr.diffuse_wm2;
        }
    }
    sums
}

fn main() {
    let epw_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/weather/USA_CO_Golden-NREL.724666_TMY3.epw".to_string());
    let weather = EpwWeatherSource::from_file(&epw_path).expect("load EPW");

    let modes: Vec<(&str, Mode)> = vec![
        ("hour-start (engine)", Mode::Offset(0.0)),
        ("quarter-past", Mode::Offset(0.25)),
        ("mid-hour", Mode::Offset(0.5)),
        ("three-quarter", Mode::Offset(0.75)),
        ("hour-end", Mode::Offset(1.0)),
        ("sub-avg x2", Mode::SubAvg(2)),
        ("sub-avg x4", Mode::SubAvg(4)),
        ("sub-avg x12", Mode::SubAvg(12)),
        ("beam-avg x12", Mode::SubBeamAvg(12)),
    ];

    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>10} {:>10} | {:>6} {:>6} {:>6} {:>6} {:>6} | {:>7} {:>7} | {}",
        "mode", "N", "E", "S", "W", "Up", "rN", "rE", "rS", "rW", "rUp", "E/W", "E+/W", "sum|dev|"
    );
    let ref_ew = REF[1] / REF[3];
    for (name, mode) in modes {
        let s = annual_sums(mode, &weather);
        let ratios: Vec<f64> = s.iter().zip(REF.iter()).map(|(a, b)| a / b).collect();
        let sum_dev: f64 = ratios.iter().map(|r| (r - 1.0).abs()).sum();
        println!(
            "{:<20} {:>10.0} {:>10.0} {:>10.0} {:>10.0} {:>10.0} | {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3} | {:>7.3} {:>7.3} | {:>7.3}",
            name, s[0], s[1], s[2], s[3], s[4], ratios[0], ratios[1], ratios[2], ratios[3], ratios[4], s[1] / s[3], ref_ew, sum_dev
        );
    }
}
