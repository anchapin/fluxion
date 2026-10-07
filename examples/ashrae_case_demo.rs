//! Live single-case ASHRAE 140 demo.
//!
//! Runs one ASHRAE 140-2023 case (Case 600, the low-mass south-window
//! baseline) end-to-end through fluxion's public Rust API and prints the
//! annual heating/cooling energy against the published acceptance bands.
//!
//! This is the live-physics counterpart to `run_ashrae_check.py` (which
//! compares *recorded* suite values in ~5 s): this binary builds the engine
//! and simulates all 8760 hours, so the first run needs a full release
//! build of the `fluxion` crate.
//!
//! What it does:
//!
//! 1. Builds the Case 600 spec via `ASHRAE140Case::Case600.spec()` — the
//!    same spec-driven, blind path the strict ±15% gate tests use
//!    (tests/all_tests/zone_balance_eplus_isolation.rs). No case ID is
//!    passed to the engine.
//! 2. Steps the `ThermalModel` for 8760 hours driven by the Golden, CO
//!    TMY3 weather file, metering the HVAC load each hour.
//! 3. Reads the published annual-energy bands from the repo's reference
//!    data (tests/reference_data/zone_balance/case_600_energy_reference.csv,
//!    sourced from ASHRAE 140-2023 Annex B) and prints PASS/OUT-OF-BAND
//!    per metric.
//!
//! Run from the repository root:
//!
//!     cargo run --release --manifest-path examples/Cargo.toml --bin ashrae_case_demo
//!
//! Exit codes: 0 both metrics in band; 1 a metric out of band; 2 the run
//! could not start (missing data file). An out-of-band result is reported
//! honestly — per repo rules outputs are never tuned to pass.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::epw::EpwWeatherSource;
use fluxion::weather::WeatherSource;

const EPW_PATH: &str = "assets/weather/USA_CO_Golden-NREL.724666_TMY3.epw";
const REFERENCE_CSV: &str = "tests/reference_data/zone_balance/case_600_energy_reference.csv";
const HOURS_PER_YEAR: usize = 8760;
const SECONDS_PER_HOUR: f64 = 3600.0;
const J_TO_MWH: f64 = 1.0 / 3.6e9;

/// One published band, in MWh.
struct Band {
    low: f64,
    high: f64,
}

/// Parse a metric's ±15% band (the last two numeric columns of the
/// reference row) from the repo's reference CSV.
fn read_band(csv: &str, metric: &str) -> Result<Band, String> {
    for line in csv.lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.first() == Some(&metric) && fields.len() >= 8 {
            let low = fields[6]
                .trim()
                .parse::<f64>()
                .map_err(|e| format!("{metric}: bad low bound {:?}: {e}", fields[6]))?;
            let high = fields[7]
                .trim()
                .parse::<f64>()
                .map_err(|e| format!("{metric}: bad high bound {:?}: {e}", fields[7]))?;
            return Ok(Band { low, high });
        }
    }
    Err(format!("{metric}: row not found in reference CSV"))
}

fn main() -> std::process::ExitCode {
    println!("ASHRAE 140-2023 live demo — Case 600 (low-mass, south window), Golden CO TMY3");

    let csv = match std::fs::read_to_string(REFERENCE_CSV) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cannot read {REFERENCE_CSV}: {e}");
            eprintln!("run this binary from the repository root");
            return std::process::ExitCode::from(2);
        }
    };
    let (heating_band, cooling_band) = match (
        read_band(&csv, "annual_heating"),
        read_band(&csv, "annual_cooling"),
    ) {
        (Ok(h), Ok(c)) => (h, c),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("{e}");
            return std::process::ExitCode::from(2);
        }
    };

    let spec = ASHRAE140Case::Case600.spec();
    let mut model = match ThermalModel::<VectorField>::from_spec_with_selector(
        &spec,
        &ThermalSelector::default(),
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Case 600 spec failed to initialize: {e}");
            return std::process::ExitCode::from(2);
        }
    };
    let weather = match EpwWeatherSource::from_file(EPW_PATH) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("cannot read {EPW_PATH}: {e:?}");
            return std::process::ExitCode::from(2);
        }
    };

    let t0 = std::time::Instant::now();
    let mut total_heating_j = 0.0_f64;
    let mut total_cooling_j = 0.0_f64;
    let mut peak_heating_w = 0.0_f64;
    let mut peak_cooling_w = 0.0_f64;

    for step in 0..HOURS_PER_YEAR {
        let weather_data = match weather.get_hourly_data(step) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("weather source failed for hour {step}: {e:?}");
                return std::process::ExitCode::from(2);
            }
        };
        model.solar.weather = Some(weather_data.clone());
        // Metered load: positive = heating, negative = cooling (kWh).
        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, SECONDS_PER_HOUR);
        let energy_j = energy_kwh * 3.6e6;
        if energy_kwh > 0.0 {
            total_heating_j += energy_j;
            peak_heating_w = peak_heating_w.max(energy_j / SECONDS_PER_HOUR);
        } else if energy_kwh < 0.0 {
            total_cooling_j += -energy_j;
            peak_cooling_w = peak_cooling_w.max(-energy_j / SECONDS_PER_HOUR);
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();

    let heating_mwh = total_heating_j * J_TO_MWH;
    let cooling_mwh = total_cooling_j * J_TO_MWH;

    println!("\nAnnual results (engine, {elapsed:.1} s wall clock):");
    let mut all_in_band = true;
    for (name, value, band, peak_kw) in [
        (
            "heating",
            heating_mwh,
            &heating_band,
            peak_heating_w / 1000.0,
        ),
        (
            "cooling",
            cooling_mwh,
            &cooling_band,
            peak_cooling_w / 1000.0,
        ),
    ] {
        let in_band = value >= band.low && value <= band.high;
        all_in_band &= in_band;
        println!(
            "  annual {name}: {value:.3} MWh  published band [{:.3}, {:.3}] MWh  {}  (peak {peak_kw:.3} kW)",
            band.low,
            band.high,
            if in_band { "PASS" } else { "OUT-OF-BAND" },
        );
    }

    if all_in_band {
        println!("\nCase 600: both annual metrics inside the published bands.");
        std::process::ExitCode::from(0)
    } else {
        println!(
            "\nCase 600: at least one annual metric is outside its published band \
             (see docs/KNOWN_ISSUES.md for the tracked structural gaps)."
        );
        std::process::ExitCode::from(1)
    }
}
