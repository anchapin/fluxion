//! Peak Load Attribution Test for Case 900
//!
//! This test decomposes peak load mismatches into subsystem contributions:
//! - Solar gains
//! - Internal gains
//! - Envelope conduction
//! - Infiltration
//! - Control effects (HVAC)
//!
//! The attribution helps identify which subsystem is causing peak over/under-prediction.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::validation::diagnostics::SimulationDiagnostics;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct PeakLoadAttribution {
    pub peak_heating_kw: f64,
    pub peak_heating_hour: usize,
    pub peak_cooling_kw: f64,
    pub peak_cooling_hour: usize,
    pub solar_at_peak_heating_w: f64,
    pub internal_at_peak_heating_w: f64,
    pub conduction_at_peak_heating_w: f64,
    pub infiltration_at_peak_heating_w: f64,
    pub hvac_at_peak_heating_w: f64,
    pub solar_at_peak_cooling_w: f64,
    pub internal_at_peak_cooling_w: f64,
    pub conduction_at_peak_cooling_w: f64,
    pub infiltration_at_peak_cooling_w: f64,
    pub hvac_at_peak_cooling_w: f64,
}

impl PeakLoadAttribution {
    pub fn new() -> Self {
        Self {
            peak_heating_kw: 0.0,
            peak_heating_hour: 0,
            peak_cooling_kw: 0.0,
            peak_cooling_hour: 0,
            solar_at_peak_heating_w: 0.0,
            internal_at_peak_heating_w: 0.0,
            conduction_at_peak_heating_w: 0.0,
            infiltration_at_peak_heating_w: 0.0,
            hvac_at_peak_heating_w: 0.0,
            solar_at_peak_cooling_w: 0.0,
            internal_at_peak_cooling_w: 0.0,
            conduction_at_peak_cooling_w: 0.0,
            infiltration_at_peak_cooling_w: 0.0,
            hvac_at_peak_cooling_w: 0.0,
        }
    }

    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            self.peak_heating_hour,
            self.peak_heating_kw,
            self.solar_at_peak_heating_w,
            self.internal_at_peak_heating_w,
            self.conduction_at_peak_heating_w,
            self.infiltration_at_peak_heating_w,
            self.hvac_at_peak_heating_w,
            self.peak_cooling_hour,
            self.peak_cooling_kw,
            self.solar_at_peak_cooling_w,
            self.internal_at_peak_cooling_w,
            self.conduction_at_peak_cooling_w,
            self.infiltration_at_peak_cooling_w,
            self.hvac_at_peak_cooling_w,
        )
    }

    pub fn csv_header() -> String {
        "peak_heating_hour,peak_heating_kw,solar_W,internal_W,conduction_W,infiltration_W,hvac_W,peak_cooling_hour,peak_cooling_kw,solar_W,internal_W,conduction_W,infiltration_W,hvac_W".to_string()
    }
}

impl Default for PeakLoadAttribution {
    fn default() -> Self {
        Self::new()
    }
}

pub fn calculate_peak_attribution(diag: &SimulationDiagnostics) -> PeakLoadAttribution {
    let mut attribution = PeakLoadAttribution::new();

    let mut max_heating = f64::MIN;
    let mut max_cooling = f64::MAX;
    let mut peak_heating_idx = 0;
    let mut peak_cooling_idx = 0;

    for i in 0..diag.hours.len() {
        let hvac = diag.loads.hvac[i][0];
        if hvac > max_heating {
            max_heating = hvac;
            peak_heating_idx = i;
        }
        if hvac < max_cooling {
            max_cooling = hvac;
            peak_cooling_idx = i;
        }
    }

    attribution.peak_heating_hour = diag.hours[peak_heating_idx];
    attribution.peak_heating_kw = max_heating / 1000.0;
    attribution.peak_cooling_hour = diag.hours[peak_cooling_idx];
    attribution.peak_cooling_kw = max_cooling.abs() / 1000.0;

    let h_idx = peak_heating_idx;
    let c_idx = peak_cooling_idx;

    attribution.solar_at_peak_heating_w = diag.loads.solar[h_idx][0];
    attribution.internal_at_peak_heating_w = diag.loads.internal[h_idx][0];
    attribution.conduction_at_peak_heating_w = diag.loads.conduction[h_idx][0];
    attribution.infiltration_at_peak_heating_w = diag.loads.infiltration[h_idx][0];
    attribution.hvac_at_peak_heating_w = diag.loads.hvac[h_idx][0];

    attribution.solar_at_peak_cooling_w = diag.loads.solar[c_idx][0];
    attribution.internal_at_peak_cooling_w = diag.loads.internal[c_idx][0];
    attribution.conduction_at_peak_cooling_w = diag.loads.conduction[c_idx][0];
    attribution.infiltration_at_peak_cooling_w = diag.loads.infiltration[c_idx][0];
    attribution.hvac_at_peak_cooling_w = diag.loads.hvac[c_idx][0];

    attribution
}

pub fn calculate_peak_attribution_for_case(case_spec: &ASHRAE140Case) -> PeakLoadAttribution {
    let spec = case_spec.spec();
    let weather = DenverTmyWeather::new();

    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let diag = SimulationDiagnostics::new(model.hvac.num_zones, 8760);
    model.set_diagnostics(Some(diag));

    let num_zones = model.hvac.num_zones;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();

        model.solar.weather = Some(weather_data.clone());

        if let Some(hvac_schedule) = spec.hvac.first() {
            let hour = (step % 24) as u8;
            let heating_sp = hvac_schedule
                .heating_setpoint_at_hour(hour)
                .unwrap_or(hvac_schedule.heating_setpoint);
            let cooling_sp = model.setpoints.cooling_schedule.value(hour as usize);
            model.setpoints.heating_setpoint = heating_sp;
            model.setpoints.cooling_setpoint = cooling_sp;

            if spec.hvac.len() > 1 {
                let mut heating_sps = vec![heating_sp; num_zones];
                let mut cooling_sps = vec![cooling_sp; num_zones];
                for (zone_idx, hvac) in spec.hvac.iter().enumerate() {
                    if zone_idx < num_zones {
                        let h_sp = hvac
                            .heating_setpoint_at_hour(hour)
                            .unwrap_or(hvac.heating_setpoint);
                        let c_sp = model.setpoints.cooling_schedule.value(hour as usize);
                        heating_sps[zone_idx] = h_sp;
                        cooling_sps[zone_idx] = c_sp;
                    }
                }
                model.setpoints.heating_setpoints = VectorField::new(heating_sps);
                model.setpoints.cooling_setpoints = VectorField::new(cooling_sps);
            }
        }

        let mut internal_loads: Vec<f64> = Vec::with_capacity(num_zones);
        for zone_idx in 0..num_zones {
            let internal_gains = spec
                .internal_loads
                .get(zone_idx)
                .or(spec.internal_loads.first())
                .and_then(|l| l.as_ref())
                .map_or(0.0, |l| l.total_load);

            let floor_area = spec
                .geometry
                .get(zone_idx)
                .or(spec.geometry.first())
                .map_or(20.0, |g| g.floor_area());

            internal_loads.push(internal_gains / floor_area);
        }
        model.set_loads(&internal_loads);

        let _hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }

    let diag = model
        .get_diagnostics()
        .expect("Diagnostics should be attached");
    calculate_peak_attribution(diag)
}

#[test]
fn test_case_900_peak_attribution() {
    println!("\n=== Case 900 Peak Load Attribution ===");

    let attribution = calculate_peak_attribution_for_case(&ASHRAE140Case::Case900);

    println!(
        "Peak Heating: {:.2} kW at hour {}",
        attribution.peak_heating_kw, attribution.peak_heating_hour
    );
    println!("  Solar: {:.2} W", attribution.solar_at_peak_heating_w);
    println!(
        "  Internal: {:.2} W",
        attribution.internal_at_peak_heating_w
    );
    println!(
        "  Conduction: {:.2} W",
        attribution.conduction_at_peak_heating_w
    );
    println!(
        "  Infiltration: {:.2} W",
        attribution.infiltration_at_peak_heating_w
    );
    println!("  HVAC: {:.2} W", attribution.hvac_at_peak_heating_w);

    println!(
        "\nPeak Cooling: {:.2} kW at hour {}",
        attribution.peak_cooling_kw, attribution.peak_cooling_hour
    );
    println!("  Solar: {:.2} W", attribution.solar_at_peak_cooling_w);
    println!(
        "  Internal: {:.2} W",
        attribution.internal_at_peak_cooling_w
    );
    println!(
        "  Conduction: {:.2} W",
        attribution.conduction_at_peak_cooling_w
    );
    println!(
        "  Infiltration: {:.2} W",
        attribution.infiltration_at_peak_cooling_w
    );
    println!("  HVAC: {:.2} W", attribution.hvac_at_peak_cooling_w);

    assert!(
        attribution.peak_heating_kw > 0.0,
        "Peak heating should be positive"
    );
    assert!(
        attribution.peak_cooling_kw > 0.0,
        "Peak cooling should be positive"
    );
    assert!(
        attribution.peak_heating_hour < 8760,
        "Peak heating hour should be valid"
    );
    assert!(
        attribution.peak_cooling_hour < 8760,
        "Peak cooling hour should be valid"
    );

    println!("\n✅ Case 900 peak attribution computed successfully");
}

#[test]
fn test_case_600_peak_attribution() {
    println!("\n=== Case 600 Peak Load Attribution ===");

    let attribution = calculate_peak_attribution_for_case(&ASHRAE140Case::Case600);

    println!(
        "Peak Heating: {:.2} kW at hour {}",
        attribution.peak_heating_kw, attribution.peak_heating_hour
    );
    println!(
        "Peak Cooling: {:.2} kW at hour {}",
        attribution.peak_cooling_kw, attribution.peak_cooling_hour
    );

    assert!(
        attribution.peak_heating_kw > 0.0,
        "Peak heating should be positive"
    );
    assert!(
        attribution.peak_cooling_kw > 0.0,
        "Peak cooling should be positive"
    );

    println!("\n✅ Case 600 peak attribution computed successfully");
}

#[test]
fn test_case_960_peak_attribution() {
    println!("\n=== Case 960 Peak Load Attribution ===");

    let attribution = calculate_peak_attribution_for_case(&ASHRAE140Case::Case960);

    println!(
        "Peak Heating: {:.2} kW at hour {}",
        attribution.peak_heating_kw, attribution.peak_heating_hour
    );
    println!(
        "Peak Cooling: {:.2} kW at hour {}",
        attribution.peak_cooling_kw, attribution.peak_cooling_hour
    );

    assert!(
        attribution.peak_heating_kw > 0.0,
        "Peak heating should be positive"
    );
    assert!(
        attribution.peak_cooling_kw > 0.0,
        "Peak cooling should be positive"
    );

    println!("\n✅ Case 960 peak attribution computed successfully");
}

#[test]
fn test_peak_attribution_csv_export() {
    use std::fs::File;
    use std::io::Write;

    let attribution = calculate_peak_attribution_for_case(&ASHRAE140Case::Case900);

    let csv_path = "case_900_peak_attribution.csv";
    let mut file = File::create(csv_path).expect("Should create CSV file");
    writeln!(file, "{}", PeakLoadAttribution::csv_header()).expect("Should write header");
    writeln!(file, "{}", attribution.to_csv_row()).expect("Should write data");

    assert!(Path::new(csv_path).exists(), "CSV file should exist");

    std::fs::remove_file(csv_path).expect("Should remove test file");

    println!("\n✅ Peak attribution CSV export verified");
}

#[test]
fn test_all_900_series_peak_attribution() {
    println!("\n=== All 900-Series Peak Load Attribution ===");

    let cases = [
        ("900", ASHRAE140Case::Case900),
        ("920", ASHRAE140Case::Case920),
        ("930", ASHRAE140Case::Case930),
        ("940", ASHRAE140Case::Case940),
        ("950", ASHRAE140Case::Case950),
        ("960", ASHRAE140Case::Case960),
    ];

    for (case_id, case_enum) in cases {
        println!("\n  Case {}...", case_id);
        let attribution = calculate_peak_attribution_for_case(&case_enum);

        println!(
            "    Peak Heating: {:.2} kW at hour {}",
            attribution.peak_heating_kw, attribution.peak_heating_hour
        );
        println!(
            "    Peak Cooling: {:.2} kW at hour {}",
            attribution.peak_cooling_kw, attribution.peak_cooling_hour
        );

        let total_heating = attribution.solar_at_peak_heating_w
            + attribution.internal_at_peak_heating_w
            + attribution.conduction_at_peak_heating_w
            + attribution.infiltration_at_peak_heating_w
            + attribution.hvac_at_peak_heating_w;

        if total_heating.abs() > 0.01 {
            println!("    Subsystem contribution at peak heating:");
            println!(
                "      Solar: {:.1}%",
                attribution.solar_at_peak_heating_w / total_heating * 100.0
            );
            println!(
                "      Internal: {:.1}%",
                attribution.internal_at_peak_heating_w / total_heating * 100.0
            );
            println!(
                "      Conduction: {:.1}%",
                attribution.conduction_at_peak_heating_w / total_heating * 100.0
            );
            println!(
                "      Infiltration: {:.1}%",
                attribution.infiltration_at_peak_heating_w / total_heating * 100.0
            );
            println!(
                "      HVAC: {:.1}%",
                attribution.hvac_at_peak_heating_w / total_heating * 100.0
            );
        }
    }

    println!("\n✅ All 900-series peak attribution computed successfully");
}
