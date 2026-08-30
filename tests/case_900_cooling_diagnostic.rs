//! Diagnostic test for Case 900 cooling energy shortfall
//!
//! Objective: Identify root cause of -33.76% cooling energy underestimation
//! (actual 6.13 MWh vs target 8.00-10.50 MWh)
//!
//! This test:
//! 1. Runs Case 900 with exact Phase 29 configuration
//! 2. Extracts hourly cooling power, zone temperature, solar gains
//! 3. Exports detailed CSV for analysis
//! 4. Reports daily and monthly cooling energy
//! 5. Identifies pattern: is cooling running too much/little? Is zone staying warm?

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

const TARGET_COOLING_MIN_MWH: f64 = 8.00;
const TARGET_COOLING_MAX_MWH: f64 = 10.50;
const REFERENCE_COOLING_MWH: f64 = 6.13;

struct HourlyData {
    step: usize,
    outdoor_temp: f64,
    zone_temp: f64,
    cooling_power_kw: f64,
    solar_gain_w: f64,
    internal_gain_w: f64,
}

struct DailyData {
    day: usize,
    cooling_kwh: f64,
    avg_zone_temp: f64,
    max_zone_temp: f64,
    min_zone_temp: f64,
    avg_outdoor_temp: f64,
}

#[derive(Clone)]
struct MonthlyData {
    month: usize,
    cooling_kwh: f64,
    days: usize,
}

fn run_simulation() -> (Vec<HourlyData>, Vec<DailyData>, Vec<MonthlyData>, f64, f64) {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    let weather = DenverTmyWeather::new();

    let warmup_days = 14;
    let warmup_steps = warmup_days * 24;

    for step in 0..warmup_steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }

    let steps = 8760;
    let mut hourly_data = Vec::with_capacity(steps);
    let mut daily_data: Vec<DailyData> = Vec::with_capacity(365);
    let mut monthly_data: Vec<MonthlyData> = vec![
        MonthlyData {
            month: 1,
            cooling_kwh: 0.0,
            days: 0
        };
        12
    ];

    let mut day_cooling_kwh = 0.0;
    let mut day_zone_temps = Vec::new();
    let mut day_outdoor_temps = Vec::new();
    let mut annual_cooling_joules = 0.0;
    let mut annual_heating_joules = 0.0;

    for step in 0..steps {
        let hour = step % 24;
        let day = step / 24;
        let month = (day / 30).min(11);
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());

        let zone_temp = model
            .setpoints
            .temperatures
            .as_slice()
            .first()
            .copied()
            .unwrap_or(20.0);

        let solar_gain_wm2 = model
            .solar
            .solar_gains
            .as_slice()
            .first()
            .copied()
            .unwrap_or(0.0);
        let zone_area = model
            .setpoints
            .zone_area
            .as_slice()
            .first()
            .copied()
            .unwrap_or(48.0);
        let solar_gain = solar_gain_wm2 * zone_area;

        let internal_gain_wm2 = model
            .setpoints
            .loads
            .as_slice()
            .first()
            .copied()
            .unwrap_or(0.0);
        let internal_gain = internal_gain_wm2 * zone_area;

        let energy_kwh =
            model.step_physics(warmup_steps + step, weather_data.dry_bulb_temp, 3600.0);
        let energy_joules = energy_kwh * 3.6e6;

        let cooling_power_kw = if energy_kwh < 0.0 { -energy_kwh } else { 0.0 };

        if energy_joules > 0.0 {
            annual_heating_joules += energy_joules;
        } else {
            annual_cooling_joules += -energy_joules;
        }

        hourly_data.push(HourlyData {
            step,
            outdoor_temp: weather_data.dry_bulb_temp,
            zone_temp,
            cooling_power_kw,
            solar_gain_w: solar_gain,
            internal_gain_w: internal_gain,
        });

        day_cooling_kwh += cooling_power_kw;
        day_zone_temps.push(zone_temp);
        day_outdoor_temps.push(weather_data.dry_bulb_temp);

        if hour == 23 {
            let avg_zone = day_zone_temps.iter().sum::<f64>() / day_zone_temps.len() as f64;
            daily_data.push(DailyData {
                day,
                cooling_kwh: day_cooling_kwh,
                avg_zone_temp: avg_zone,
                max_zone_temp: day_zone_temps.iter().cloned().fold(f64::MIN, f64::max),
                min_zone_temp: day_zone_temps.iter().cloned().fold(f64::MAX, f64::min),
                avg_outdoor_temp: day_outdoor_temps.iter().sum::<f64>()
                    / day_outdoor_temps.len() as f64,
            });

            monthly_data[month].cooling_kwh += day_cooling_kwh;
            monthly_data[month].days += 1;

            day_cooling_kwh = 0.0;
            day_zone_temps.clear();
            day_outdoor_temps.clear();
        }
    }

    let annual_cooling_mwh = annual_cooling_joules / 3.6e9;
    let annual_heating_mwh = annual_heating_joules / 3.6e9;

    (
        hourly_data,
        daily_data,
        monthly_data,
        annual_cooling_mwh,
        annual_heating_mwh,
    )
}

fn export_csv(hourly_data: &[HourlyData], filename: &str) -> std::io::Result<()> {
    let file = std::fs::File::create(filename)?;
    let mut wtr = csv::Writer::from_writer(file);
    wtr.write_record(&[
        "step",
        "outdoor_temp_c",
        "zone_temp_c",
        "cooling_power_kw",
        "solar_gain_w",
        "internal_gain_w",
    ])?;
    for h in hourly_data {
        wtr.write_record(&[
            h.step.to_string(),
            format!("{:.2}", h.outdoor_temp),
            format!("{:.2}", h.zone_temp),
            format!("{:.4}", h.cooling_power_kw),
            format!("{:.2}", h.solar_gain_w),
            format!("{:.2}", h.internal_gain_w),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

fn analyze_cooling_pattern(daily_data: &[DailyData], monthly_data: &[MonthlyData]) {
    println!("\n=== Case 900 Cooling Diagnostic Analysis ===");
    println!("\n--- Monthly Cooling Energy ---");
    for (i, m) in monthly_data.iter().enumerate() {
        if m.days > 0 {
            println!(
                "Month {:2}: {:8.2} kWh ({:4.1} days avg daily: {:6.2} kWh/day)",
                i + 1,
                m.cooling_kwh,
                m.days,
                m.cooling_kwh / m.days as f64
            );
        }
    }

    let total_cooling = monthly_data.iter().map(|m| m.cooling_kwh).sum::<f64>();
    println!(
        "\nTotal Annual Cooling: {:.2} kWh ({:.2} MWh)",
        total_cooling,
        total_cooling / 1000.0
    );

    let warm_months: Vec<_> = daily_data
        .iter()
        .filter(|d| d.avg_outdoor_temp > 20.0)
        .collect();
    let warm_cooling: f64 = warm_months.iter().map(|d| d.cooling_kwh).sum();
    let cool_cooling: f64 = daily_data.iter().map(|d| d.cooling_kwh).sum::<f64>() - warm_cooling;

    println!("\n--- Cooling by Temperature Regime ---");
    println!("Warm months (>20°C avg): {:.2} kWh", warm_cooling);
    println!("Cool months (≤20°C avg): {:.2} kWh", cool_cooling);

    let hot_days: Vec<_> = daily_data
        .iter()
        .filter(|d| d.max_zone_temp > 26.0)
        .collect();
    let warm_days: Vec<_> = daily_data
        .iter()
        .filter(|d| d.max_zone_temp > 24.0 && d.max_zone_temp <= 26.0)
        .collect();
    let comfort_days: Vec<_> = daily_data
        .iter()
        .filter(|d| d.max_zone_temp <= 24.0)
        .collect();

    println!("\n--- Zone Temperature Distribution ---");
    println!("Hot days (>26°C max):   {:4}", hot_days.len());
    println!("Warm days (24-26°C):   {:4}", warm_days.len());
    println!("Comfort days (≤24°C):  {:4}", comfort_days.len());

    if !hot_days.is_empty() {
        let hot_cooling: f64 = hot_days.iter().map(|d| d.cooling_kwh).sum();
        let avg_hot_cooling = hot_cooling / hot_days.len() as f64;
        println!("  Hot days avg cooling: {:.2} kWh/day", avg_hot_cooling);
    }
}

#[test]
fn test_case_900_cooling_diagnostic() {
    println!("Running Case 900 cooling diagnostic...");
    println!("Reference: 6.13 MWh actual vs target 8.00-10.50 MWh (33.76% underestimation)");

    let (hourly_data, daily_data, monthly_data, annual_cooling_mwh, annual_heating_mwh) =
        run_simulation();

    println!("\n=== Results ===");
    println!(
        "Annual Cooling: {:.2} MWh (target: {:.2}-{:.2} MWh)",
        annual_cooling_mwh, TARGET_COOLING_MIN_MWH, TARGET_COOLING_MAX_MWH
    );
    println!("Annual Heating: {:.2} MWh", annual_heating_mwh);

    let error_pct = ((annual_cooling_mwh - REFERENCE_COOLING_MWH) / REFERENCE_COOLING_MWH) * 100.0;
    println!("Error vs reference: {:.2}%", error_pct);

    let in_range = annual_cooling_mwh >= TARGET_COOLING_MIN_MWH
        && annual_cooling_mwh <= TARGET_COOLING_MAX_MWH;
    println!("In target range: {}", if in_range { "YES" } else { "NO" });

    if let Ok(()) = export_csv(&hourly_data, "output/case_900_cooling_diagnostic.csv") {
        println!("\nHourly CSV exported to: output/case_900_cooling_diagnostic.csv");
    }

    analyze_cooling_pattern(&daily_data, &monthly_data);

    println!("\n=== Diagnostic Summary ===");
    if annual_cooling_mwh < TARGET_COOLING_MIN_MWH {
        println!(
            "WARNING: Cooling energy {:.2} MWh is below target range ({:.2}-{:.2} MWh)",
            annual_cooling_mwh, TARGET_COOLING_MIN_MWH, TARGET_COOLING_MAX_MWH
        );
        println!("This confirms the reported cooling underestimation issue.");
        println!(
            "Zone temperature analysis: {} hot days indicates cooling is insufficient",
            daily_data.iter().filter(|d| d.max_zone_temp > 26.0).count()
        );
    } else if annual_cooling_mwh > TARGET_COOLING_MAX_MWH {
        println!(
            "NOTE: Cooling energy {:.2} MWh exceeds upper target ({:.2} MWh)",
            annual_cooling_mwh, TARGET_COOLING_MAX_MWH
        );
    } else {
        println!(
            "PASS: Cooling energy {:.2} MWh is within target range ({:.2}-{:.2} MWh)",
            annual_cooling_mwh, TARGET_COOLING_MIN_MWH, TARGET_COOLING_MAX_MWH
        );
    }
}
