use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HourlyComparison {
    pub hour: usize,
    pub fluxion_outdoor_temp: f64,
    pub energyplus_outdoor_temp: f64,
    pub fluxion_zone_temp: f64,
    pub energyplus_zone_temp: f64,
    pub fluxion_heating: f64,
    pub energyplus_heating: f64,
    pub fluxion_cooling: f64,
    pub energyplus_cooling: f64,
    pub outdoor_temp_delta: f64,
    pub zone_temp_delta: f64,
    pub heating_delta: f64,
    pub cooling_delta: f64,
    pub heating_percent_error: f64,
    pub cooling_percent_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaAnalysisReport {
    case_id: String,
    comparisons: Vec<HourlyComparison>,
    summary: DeltaSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaSummary {
    total_hours: usize,
    mean_outdoor_temp_delta: f64,
    mean_zone_temp_delta: f64,
    mean_heating_delta: f64,
    mean_cooling_delta: f64,
    max_heating_error_hour: usize,
    max_heating_error: f64,
    max_cooling_error_hour: usize,
    max_cooling_error: f64,
    hours_with_high_heating_error: usize,
    hours_with_high_cooling_error: usize,
}

impl DeltaAnalysisReport {
    pub fn new(case_id: String) -> Self {
        Self {
            case_id,
            comparisons: Vec::new(),
            summary: DeltaSummary {
                total_hours: 0,
                mean_outdoor_temp_delta: 0.0,
                mean_zone_temp_delta: 0.0,
                mean_heating_delta: 0.0,
                mean_cooling_delta: 0.0,
                max_heating_error_hour: 0,
                max_heating_error: 0.0,
                max_cooling_error_hour: 0,
                max_cooling_error: 0.0,
                hours_with_high_heating_error: 0,
                hours_with_high_cooling_error: 0,
            },
        }
    }

    pub fn add_comparison(&mut self, comp: HourlyComparison) {
        self.comparisons.push(comp);
    }

    pub fn compute_summary(&mut self) {
        let n = self.comparisons.len();
        if n == 0 {
            return;
        }

        self.summary.total_hours = n;

        let mut sum_outdoor = 0.0;
        let mut sum_zone = 0.0;
        let mut sum_heating = 0.0;
        let mut sum_cooling = 0.0;

        for comp in &self.comparisons {
            sum_outdoor += comp.outdoor_temp_delta.abs();
            sum_zone += comp.zone_temp_delta.abs();
            sum_heating += comp.heating_delta.abs();
            sum_cooling += comp.cooling_delta.abs();

            if comp.heating_percent_error.abs() > self.summary.max_heating_error {
                self.summary.max_heating_error = comp.heating_percent_error.abs();
                self.summary.max_heating_error_hour = comp.hour;
            }
            if comp.cooling_percent_error.abs() > self.summary.max_cooling_error {
                self.summary.max_cooling_error = comp.cooling_percent_error.abs();
                self.summary.max_cooling_error_hour = comp.hour;
            }

            if comp.heating_percent_error.abs() > 50.0 {
                self.summary.hours_with_high_heating_error += 1;
            }
            if comp.cooling_percent_error.abs() > 50.0 {
                self.summary.hours_with_high_cooling_error += 1;
            }
        }

        self.summary.mean_outdoor_temp_delta = sum_outdoor / n as f64;
        self.summary.mean_zone_temp_delta = sum_zone / n as f64;
        self.summary.mean_heating_delta = sum_heating / n as f64;
        self.summary.mean_cooling_delta = sum_cooling / n as f64;
    }

    pub fn to_markdown(&self) -> String {
        let mut md = format!(
            r#"# Hourly Delta Analysis Report - {}

## Summary
- **Total Hours Analyzed**: {}
- **Mean Outdoor Temp Delta**: {:.2}°C
- **Mean Zone Temp Delta**: {:.2}°C
- **Mean Heating Delta**: {:.2} kWh
- **Mean Cooling Delta**: {:.2} kWh

## Peak Errors
- **Max Heating Error**: {:.1}% at hour {}
- **Max Cooling Error**: {:.1}% at hour {}

## High Error Hours
- Hours with >50% heating error: {}
- Hours with >50% cooling error: {}

## Hourly Details

| Hour | T_out Flux | T_out EP | ΔT_out | T_zone Flux | T_zone EP | ΔT_zone | Heat Flux | Heat EP | ΔHeat | Cool Flux | Cool EP | ΔCool |
|------|------------|----------|--------|-------------|-----------|---------|-----------|---------|-------|-----------|---------|-------|
"#,
            self.case_id,
            self.summary.total_hours,
            self.summary.mean_outdoor_temp_delta,
            self.summary.mean_zone_temp_delta,
            self.summary.mean_heating_delta,
            self.summary.mean_cooling_delta,
            self.summary.max_heating_error,
            self.summary.max_heating_error_hour,
            self.summary.max_cooling_error,
            self.summary.max_cooling_error_hour,
            self.summary.hours_with_high_heating_error,
            self.summary.hours_with_high_cooling_error
        );

        for comp in &self.comparisons {
            md.push_str(&format!(
                "| {} | {:.1} | {:.1} | {:+.1} | {:.1} | {:.1} | {:+.1} | {:.2} | {:.2} | {:+.2} | {:.2} | {:.2} | {:+.2} |\n",
                comp.hour,
                comp.fluxion_outdoor_temp,
                comp.energyplus_outdoor_temp,
                comp.outdoor_temp_delta,
                comp.fluxion_zone_temp,
                comp.energyplus_zone_temp,
                comp.zone_temp_delta,
                comp.fluxion_heating,
                comp.energyplus_heating,
                comp.heating_delta,
                comp.fluxion_cooling,
                comp.energyplus_cooling,
                comp.cooling_delta
            ));
        }

        md
    }

    pub fn to_csv(&self) -> String {
        let mut csv = String::from("Hour,Fluxion_Outdoor_Temp,EnergyPlus_Outdoor_Temp,Outdoor_Temp_Delta,Fluxion_Zone_Temp,EnergyPlus_Zone_Temp,Zone_Temp_Delta,Fluxion_Heating,EnergyPlus_Heating,Heating_Delta,Fluxion_Cooling,EnergyPlus_Cooling,Cooling_Delta,Heating_Percent_Error,Cooling_Percent_Error\n");

        for comp in &self.comparisons {
            csv.push_str(&format!(
                "{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{:.1}\n",
                comp.hour,
                comp.fluxion_outdoor_temp,
                comp.energyplus_outdoor_temp,
                comp.outdoor_temp_delta,
                comp.fluxion_zone_temp,
                comp.energyplus_zone_temp,
                comp.zone_temp_delta,
                comp.fluxion_heating,
                comp.energyplus_heating,
                comp.heating_delta,
                comp.fluxion_cooling,
                comp.energyplus_cooling,
                comp.cooling_delta,
                comp.heating_percent_error,
                comp.cooling_percent_error
            ));
        }

        csv
    }
}

fn parse_csv_value(s: &str) -> f64 {
    s.trim().parse().unwrap_or(0.0)
}

pub fn compare_hourly_data(
    fluxion_csv: &str,
    energyplus_csv: &str,
    case_id: &str,
) -> Result<DeltaAnalysisReport, String> {
    let fluxion_file =
        File::open(fluxion_csv).map_err(|e| format!("Failed to open Fluxion CSV: {}", e))?;
    let ep_file =
        File::open(energyplus_csv).map_err(|e| format!("Failed to open EnergyPlus CSV: {}", e))?;

    let fluxion_reader = BufReader::new(fluxion_file);
    let ep_reader = BufReader::new(ep_file);

    let fluxion_lines: Vec<String> = fluxion_reader.lines().map_while(Result::ok).collect();
    let ep_lines: Vec<String> = ep_reader.lines().map_while(Result::ok).collect();

    if fluxion_lines.len() < 2 || ep_lines.len() < 2 {
        return Err("CSV files must have header and at least one data row".to_string());
    }

    let fluxion_header = &fluxion_lines[0];
    let ep_header = &ep_lines[0];

    println!("Fluxion header: {}", fluxion_header);
    println!("EnergyPlus header: {}", ep_header);

    let mut report = DeltaAnalysisReport::new(case_id.to_string());

    let fluxion_data: Vec<String> = fluxion_lines[1..].to_vec();
    let ep_data: Vec<String> = ep_lines[1..].to_vec();

    let num_rows = fluxion_data.len().min(ep_data.len());

    for i in 0..num_rows {
        let fcols: Vec<&str> = fluxion_data[i].split(',').collect();
        let ecols: Vec<&str> = ep_data[i].split(',').collect();

        if fcols.len() < 9 || ecols.len() < 9 {
            continue;
        }

        let hour = i;
        let fluxion_outdoor = parse_csv_value(fcols[4]);
        let ep_outdoor = parse_csv_value(ecols[4]);

        let fluxion_zone = parse_csv_value(fcols[5]);
        let ep_zone = parse_csv_value(ecols[5]);

        let fluxion_heating = parse_csv_value(fcols[7]);
        let ep_heating = parse_csv_value(ecols[7]);

        let fluxion_cooling = parse_csv_value(fcols[8]);
        let ep_cooling = parse_csv_value(ecols[8]);

        let outdoor_delta = fluxion_outdoor - ep_outdoor;
        let zone_delta = fluxion_zone - ep_zone;
        let heating_delta = fluxion_heating - ep_heating;
        let cooling_delta = fluxion_cooling - ep_cooling;

        let heating_pct_error = if ep_heating.abs() > 0.01 {
            (heating_delta / ep_heating.abs()) * 100.0
        } else {
            0.0
        };

        let cooling_pct_error = if ep_cooling.abs() > 0.01 {
            (cooling_delta / ep_cooling.abs()) * 100.0
        } else {
            0.0
        };

        let comp = HourlyComparison {
            hour,
            fluxion_outdoor_temp: fluxion_outdoor,
            energyplus_outdoor_temp: ep_outdoor,
            fluxion_zone_temp: fluxion_zone,
            energyplus_zone_temp: ep_zone,
            fluxion_heating,
            energyplus_heating: ep_heating,
            fluxion_cooling,
            energyplus_cooling: ep_cooling,
            outdoor_temp_delta: outdoor_delta,
            zone_temp_delta: zone_delta,
            heating_delta,
            cooling_delta,
            heating_percent_error: heating_pct_error,
            cooling_percent_error: cooling_pct_error,
        };

        report.add_comparison(comp);
    }

    report.compute_summary();

    Ok(report)
}

#[derive(Parser, Debug)]
#[command(name = "fluxion-delta")]
#[command(about = "Hourly delta analysis tool for comparing Fluxion against EnergyPlus", long_about = None)]
struct Args {
    #[arg(short, long)]
    fluxion_csv: PathBuf,

    #[arg(short, long)]
    energyplus_csv: PathBuf,

    #[arg(short, long, default_value = "Unknown")]
    case_id: String,

    #[arg(short, long)]
    output: Option<PathBuf>,

    #[arg(short, long, default_value = "markdown")]
    format: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Fluxion Delta Analysis Tool");
    println!("=============================");
    println!("Fluxion CSV: {:?}", args.fluxion_csv);
    println!("EnergyPlus CSV: {:?}", args.energyplus_csv);
    println!("Case ID: {}", args.case_id);
    println!();

    let report = compare_hourly_data(
        args.fluxion_csv.to_str().unwrap(),
        args.energyplus_csv.to_str().unwrap(),
        &args.case_id,
    )?;

    println!("{}", report.to_markdown());

    if let Some(output_path) = args.output {
        let content = match args.format.as_str() {
            "csv" => report.to_csv(),
            _ => report.to_markdown(),
        };
        std::fs::write(&output_path, content)?;
        println!("\nReport saved to: {:?}", output_path);
    }

    Ok(())
}
