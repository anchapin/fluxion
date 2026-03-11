use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;
use crate::validation::ashrae_140_cases::CaseSpec;
use crate::validation::diagnostic::HourlyData;
use crate::weather::denver::DenverTmyWeather;
use crate::weather::WeatherSource;
use anyhow::Result;
use csv::WriterBuilder;
use serde::{Deserialize, Serialize};
use serde_yaml;
use std::collections::HashMap;
use std::path::Path;

/// Delta configuration: base case and a set of variant modifications.
#[derive(Debug, Deserialize, Serialize)]
pub struct DeltaConfig {
    pub base: CaseSpec,
    pub variants: Vec<Variant>,
}

/// A variant applies a patch (direct field modifications) and/or a sweep (parametric sweep over values).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Variant {
    pub name: String,
    #[serde(default)]
    pub patch: Option<HashMap<String, serde_yaml::Value>>,
    #[serde(default)]
    pub sweep: Option<HashMap<String, Vec<f64>>>,
}

/// Delta report containing comparison between base and each variant.
#[derive(Debug, Serialize)]
pub struct DeltaReport {
    pub base_name: String,
    pub variants: Vec<VariantResult>,
}

/// Result for a single variant.
#[derive(Debug, Serialize)]
pub struct VariantResult {
    pub name: String,
    pub annual_heating_mwh: f64,
    pub annual_cooling_mwh: f64,
    pub peak_heating_kw: f64,
    pub peak_cooling_kw: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hourly_differences: Option<Vec<HourlyDelta>>,
}

/// Hourly difference record.
#[derive(Debug, Serialize)]
pub struct HourlyDelta {
    pub hour: usize,
    pub zone: usize,
    pub component: String,
    pub base_value: f64,
    pub variant_value: f64,
    pub difference: f64,
}

/// Internal simulation output structure.
#[derive(Debug, Clone)]
struct SimulationResult {
    annual_heating_mwh: f64,
    annual_cooling_mwh: f64,
    peak_heating_kw: f64,
    peak_cooling_kw: f64,
    hourly_data: Option<Vec<HourlyData>>,
}

/// Parse a DeltaConfig from a YAML file.
pub fn parse_config(path: &Path) -> Result<DeltaConfig> {
    let file = std::fs::File::open(path)?;
    let config: DeltaConfig = serde_yaml::from_reader(file)?;
    Ok(config)
}

/// Expand all variants into concrete CaseSpec instances.
///
/// For each variant:
/// - Apply the patch (if any) to the base CaseSpec.
/// - If a sweep is defined, generate a Cartesian product of all parameter values.
///   Each combination yields a separate variant with a name like "original: param=value, ...".
pub fn expand_variants(config: &DeltaConfig) -> Result<Vec<(String, CaseSpec)>> {
    let base_yaml = serde_yaml::to_value(&config.base)?;
    let mut results = Vec::new();

    for variant in &config.variants {
        let mut current = base_yaml.clone();

        // Apply patch if present
        if let Some(patch) = &variant.patch {
            current = apply_patch(current, patch)?;
        }

        if let Some(sweep) = &variant.sweep {
            let sweep_items: Vec<(String, Vec<f64>)> =
                sweep.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            let mut combinations = Vec::new();
            generate_sweep_combinations(
                &current,
                &sweep_items,
                0,
                HashMap::new(),
                &mut combinations,
            )?;
            for (name_suffix, spec_value) in combinations {
                let spec: CaseSpec = serde_yaml::from_value(spec_value)?;
                let full_name = if name_suffix.is_empty() {
                    variant.name.clone()
                } else {
                    format!("{}: {}", variant.name, name_suffix)
                };
                results.push((full_name, spec));
            }
        } else {
            let spec: CaseSpec = serde_yaml::from_value(current)?;
            results.push((variant.name.clone(), spec));
        }
    }

    Ok(results)
}

/// Apply a patch hashmap to a YAML value (deep merge).
fn apply_patch(
    mut base: serde_yaml::Value,
    patch: &HashMap<String, serde_yaml::Value>,
) -> Result<serde_yaml::Value> {
    for (key, value) in patch {
        set_nested(&mut base, key, value.clone())?;
    }
    Ok(base)
}

/// Set a nested field in a YAML value using dot notation.
fn set_nested(
    value: &mut serde_yaml::Value,
    path: &str,
    new_value: serde_yaml::Value,
) -> Result<()> {
    let parts: Vec<&str> = path.split('.').collect();
    if parts.is_empty() {
        return Err(anyhow::anyhow!("Empty path"));
    }
    let mut current = value;
    for (i, part) in parts.iter().enumerate() {
        if i == parts.len() - 1 {
            if let Some(obj) = current.as_mapping_mut() {
                obj.insert(
                    serde_yaml::Value::String(part.to_string()),
                    new_value.clone(),
                );
            } else {
                return Err(anyhow::anyhow!("Cannot set '{}': not an object", path));
            }
        } else {
            match current {
                serde_yaml::Value::Mapping(ref mut map) => {
                    let key = serde_yaml::Value::String(part.to_string());
                    let next = map.get_mut(&key).ok_or_else(|| {
                        anyhow::anyhow!("Key '{}' not found in path '{}'", part, path)
                    })?;
                    if !next.is_mapping() {
                        return Err(anyhow::anyhow!("Intermediate '{}' is not an object", part));
                    }
                    current = next;
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Expected object at '{}' in path '{}'",
                        part,
                        path
                    ))
                }
            }
        }
    }
    Ok(())
}

/// Recursively generate all combinations for a sweep.
fn generate_sweep_combinations(
    base: &serde_yaml::Value,
    sweep_items: &[(String, Vec<f64>)],
    idx: usize,
    assigned: HashMap<String, f64>,
    out: &mut Vec<(String, serde_yaml::Value)>,
) -> Result<()> {
    if idx == sweep_items.len() {
        let mut value = base.clone();
        for (param, val) in &assigned {
            set_nested(&mut value, param, serde_yaml::to_value(val)?)?;
        }
        let suffix = assigned
            .iter()
            .map(|(k, v)| format!("{}={:.2}", k, v))
            .collect::<Vec<_>>()
            .join(", ");
        out.push((suffix, value));
        return Ok(());
    }

    let (param, values) = &sweep_items[idx];
    for &val in values {
        let mut new_assigned = assigned.clone();
        new_assigned.insert(param.clone(), val);
        generate_sweep_combinations(base, sweep_items, idx + 1, new_assigned, out)?;
    }
    Ok(())
}

/// Run a simulation for a given CaseSpec and return annual/peak metrics plus optional hourly data.
fn run_simulation(spec: &CaseSpec, collect_hourly: bool) -> Result<SimulationResult> {
    // Build model from spec
    let mut model = ThermalModel::<VectorField>::from_spec(spec);
    model.reset_peak_power();

    let is_free_floating = spec.is_free_floating();
    if is_free_floating {
        model.heating_setpoint = -999.0;
        model.cooling_setpoint = 999.0;
        model.hvac_heating_capacity = 0.0;
        model.hvac_cooling_capacity = 0.0;
    }

    // Set HVAC enabled flags per zone
    let num_zones = spec.num_zones;
    let mut hvac_enabled_vals = vec![1.0; num_zones];
    if !spec.hvac.is_empty() {
        for (zone_idx, hvac) in spec.hvac.iter().enumerate() {
            if zone_idx < num_zones {
                hvac_enabled_vals[zone_idx] = if hvac.is_enabled() { 1.0 } else { 0.0 };
            }
        }
    }
    model.hvac_enabled = VectorField::new(hvac_enabled_vals.clone());

    // Prepare weather (Denver TMY)
    let weather = DenverTmyWeather::new();

    // Accumulators
    let mut annual_heating_joules: f64 = 0.0;
    let mut annual_cooling_joules: f64 = 0.0;

    let mut hourly_data_vec: Option<Vec<HourlyData>> = if collect_hourly {
        Some(Vec::new())
    } else {
        None
    };

    const STEPS: usize = 8760;
    for step in 0..STEPS {
        let hour_of_day = step % 24;
        let weather_data = weather.get_hourly_data(step).unwrap();

        // Update weather data on model for solar gain calculation
        model.weather = Some(weather_data.clone());

        // Apply HVAC schedule: handle single-zone and multi-zone
        if !spec.hvac.is_empty() {
            // Use first schedule as default base
            let base_heating_sp = spec.hvac[0].heating_setpoint;
            let base_cooling_sp = spec.hvac[0].cooling_setpoint;
            let hour_u8 = hour_of_day as u8;
            let heating_sp = spec.hvac[0]
                .heating_setpoint_at_hour(hour_u8)
                .unwrap_or(base_heating_sp);
            let cooling_sp = spec.hvac[0]
                .cooling_setpoint_at_hour(hour_u8)
                .unwrap_or(base_cooling_sp);

            if spec.hvac.len() > 1 {
                // Multi-zone: set per-zone vectors
                let mut heating_sps = vec![heating_sp; num_zones];
                let mut cooling_sps = vec![cooling_sp; num_zones];
                for (zone_idx, hvac) in spec.hvac.iter().enumerate() {
                    if zone_idx < num_zones {
                        let h_sp = hvac
                            .heating_setpoint_at_hour(hour_u8)
                            .unwrap_or(hvac.heating_setpoint);
                        let c_sp = hvac
                            .cooling_setpoint_at_hour(hour_u8)
                            .unwrap_or(hvac.cooling_setpoint);
                        heating_sps[zone_idx] = h_sp;
                        cooling_sps[zone_idx] = c_sp;
                    }
                }
                model.heating_setpoints = VectorField::new(heating_sps);
                model.cooling_setpoints = VectorField::new(cooling_sps);
            } else {
                // Single-zone: use scalars
                model.heating_setpoint = heating_sp;
                model.cooling_setpoint = cooling_sp;
            }
        }

        // Apply night ventilation if active (e.g., Case 650)
        if let Some(vent) = &spec.night_ventilation {
            if vent.is_active_at_hour(hour_of_day as u8) {
                if let Some(hvac_schedule) = spec.hvac.first() {
                    // For cases with heating disabled (negative setpoint), adjust cooling setpoint
                    if hvac_schedule.heating_setpoint < 0.0 {
                        model.cooling_setpoint = -100.0;
                    }
                }
            }
        }

        // Prepare internal loads in W/m²
        let mut internal_loads_density = Vec::with_capacity(num_zones);
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
            internal_loads_density.push(internal_gains / floor_area);
        }
        model.set_loads(&internal_loads_density);

        // Step physics
        let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp);

        // Accumulate energy (convert kWh to Joules)
        if hvac_kwh > 0.0 {
            annual_heating_joules += hvac_kwh * 3.6e6;
        } else {
            annual_cooling_joules += (-hvac_kwh) * 3.6e6;
        }

        // Collect hourly diagnostics if requested
        if let Some(ref mut hourly_vec) = hourly_data_vec {
            let mut hourly = HourlyData::new(step, num_zones);
            hourly.outdoor_temp = weather_data.dry_bulb_temp;
            hourly.zone_temps = model.temperatures.as_slice().to_vec();
            hourly.mass_temps = model.mass_temperatures.as_slice().to_vec();

            for zone_idx in 0..num_zones {
                // Floor area
                let floor_area = spec
                    .geometry
                    .get(zone_idx)
                    .or(spec.geometry.first())
                    .map_or(20.0, |g| g.floor_area());
                // Wall area
                let wall_area = spec
                    .geometry
                    .get(zone_idx)
                    .or(spec.geometry.first())
                    .map_or(0.0, |g| g.wall_area());
                // Window area
                let window_area: f64 = spec
                    .windows
                    .get(zone_idx)
                    .map(|wlist| wlist.iter().map(|w| w.area).sum::<f64>())
                    .unwrap_or(0.0);
                let opaque_area = wall_area - window_area;

                // Zone temperature and delta-T
                let zone_temp = model
                    .temperatures
                    .as_slice()
                    .get(zone_idx)
                    .copied()
                    .unwrap_or(20.0);
                let delta_t = zone_temp - weather_data.dry_bulb_temp;

                // Envelope conduction (walls + roof)
                let wall_u = spec.construction.wall.u_value(None, None);
                let roof_u = spec.construction.roof.u_value(None, None);
                let cond =
                    opaque_area * wall_u * delta_t.abs() + floor_area * roof_u * delta_t.abs();
                hourly.envelope_conduction[zone_idx] = cond;

                // Infiltration (W)
                let volume = floor_area
                    * spec
                        .geometry
                        .get(zone_idx)
                        .or(spec.geometry.first())
                        .map_or(2.7, |g| g.height);
                let infil = spec.infiltration_ach * volume * 1.2 * 1005.0 * delta_t.abs() / 3600.0;
                hourly.infiltration_loss[zone_idx] = infil;

                // Solar gains: model.solar_gains is in W/m², convert to total Watts
                let solar_wm2 = model
                    .solar_gains
                    .as_ref()
                    .get(zone_idx)
                    .copied()
                    .unwrap_or(0.0);
                hourly.solar_gains[zone_idx] = solar_wm2 * floor_area;

                // Internal loads: convert from density back to total Watts
                let internal_total = internal_loads_density[zone_idx] * floor_area;
                hourly.internal_loads[zone_idx] = internal_total;
            }

            // HVAC power: assign to zone 0 (single-zone assumption for totals)
            if hvac_kwh > 0.0 {
                let hvac_watts = hvac_kwh * 1000.0;
                if num_zones > 0 {
                    hourly.hvac_heating[0] = hvac_watts;
                }
            } else {
                let hvac_watts = (-hvac_kwh) * 1000.0;
                if num_zones > 0 {
                    hourly.hvac_cooling[0] = hvac_watts;
                }
            }

            hourly_vec.push(hourly);
        }
    }

    // Retrieve peak power directly from model (Issue #272)
    let peak_heating_kw = model.get_peak_heating_power_kw();
    let peak_cooling_kw = model.get_peak_cooling_power_kw();

    Ok(SimulationResult {
        annual_heating_mwh: annual_heating_joules / 3.6e9,
        annual_cooling_mwh: annual_cooling_joules / 3.6e9,
        peak_heating_kw,
        peak_cooling_kw,
        hourly_data: hourly_data_vec,
    })
}

/// Compute per-hour, per-zone, per-component differences between base and variant.
fn compute_hourly_deltas(
    base_data: &[HourlyData],
    var_data: &[HourlyData],
) -> Result<Vec<HourlyDelta>> {
    if base_data.len() != var_data.len() {
        return Err(anyhow::anyhow!(
            "Mismatched hour counts: base {}, variant {}",
            base_data.len(),
            var_data.len()
        ));
    }
    let num_zones = base_data.first().map(|d| d.zone_temps.len()).unwrap_or(0);
    if num_zones == 0 {
        return Err(anyhow::anyhow!("Zero zones in hourly data"));
    }

    let mut deltas = Vec::new();
    for (hour, (base_h, var_h)) in base_data.iter().zip(var_data.iter()).enumerate() {
        let components: &[(&str, &[f64], &[f64])] = &[
            ("solar_gains", &base_h.solar_gains, &var_h.solar_gains),
            (
                "internal_loads",
                &base_h.internal_loads,
                &var_h.internal_loads,
            ),
            (
                "infiltration_loss",
                &base_h.infiltration_loss,
                &var_h.infiltration_loss,
            ),
            (
                "envelope_conduction",
                &base_h.envelope_conduction,
                &var_h.envelope_conduction,
            ),
            ("hvac_heating", &base_h.hvac_heating, &var_h.hvac_heating),
            ("hvac_cooling", &base_h.hvac_cooling, &var_h.hvac_cooling),
        ];
        for (comp_name, base_vals, var_vals) in components {
            for zone in 0..num_zones {
                let b = base_vals.get(zone).copied().unwrap_or(0.0);
                let v = var_vals.get(zone).copied().unwrap_or(0.0);
                deltas.push(HourlyDelta {
                    hour,
                    zone,
                    component: comp_name.to_string(),
                    base_value: b,
                    variant_value: v,
                    difference: v - b,
                });
            }
        }
    }
    Ok(deltas)
}

/// Generate Markdown report with base row and diffs for each variant, including sweep statistics.
fn generate_markdown_report(report: &DeltaReport, base: &SimulationResult) -> String {
    let mut out = String::new();
    out.push_str("# Delta Test Report\n\n");
    out.push_str(&format!("**Base Case:** {}\n\n", report.base_name));

    out.push_str("| Variant | Annual Heating (MWh) | Δ | %Δ | Annual Cooling (MWh) | Δ | %Δ | Peak Heating (kW) | Δ | %Δ | Peak Cooling (kW) | Δ | %Δ |\n");
    out.push_str("|---------|---------------------|-----|-----|----------------------|-----|-----|-------------------|-----|-----|-------------------|-----|-----|\n");

    // Base row
    out.push_str(&format!(
        "| {} | {:.3} | - | - | {:.3} | - | - | {:.3} | - | - | {:.3} | - | - |\n",
        report.base_name,
        base.annual_heating_mwh,
        base.annual_cooling_mwh,
        base.peak_heating_kw,
        base.peak_cooling_kw
    ));

    // Variant rows
    for v in &report.variants {
        let dh = v.annual_heating_mwh - base.annual_heating_mwh;
        let pct_h = if base.annual_heating_mwh != 0.0 {
            (dh / base.annual_heating_mwh) * 100.0
        } else {
            0.0
        };
        let dc = v.annual_cooling_mwh - base.annual_cooling_mwh;
        let pct_c = if base.annual_cooling_mwh != 0.0 {
            (dc / base.annual_cooling_mwh) * 100.0
        } else {
            0.0
        };
        let dph = v.peak_heating_kw - base.peak_heating_kw;
        let pct_ph = if base.peak_heating_kw != 0.0 {
            (dph / base.peak_heating_kw) * 100.0
        } else {
            0.0
        };
        let dpc = v.peak_cooling_kw - base.peak_cooling_kw;
        let pct_pc = if base.peak_cooling_kw != 0.0 {
            (dpc / base.peak_cooling_kw) * 100.0
        } else {
            0.0
        };

        out.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.1}% | {:.3} | {:.3} | {:.1}% | {:.3} | {:.3} | {:.1}% | {:.3} | {:.3} | {:.1}% |\n",
            v.name,
            v.annual_heating_mwh, dh, pct_h,
            v.annual_cooling_mwh, dc, pct_c,
            v.peak_heating_kw, dph, pct_ph,
            v.peak_cooling_kw, dpc, pct_pc
        ));
    }

    // Add sweep statistics summary
    let sweep_summary = generate_sweep_statistics(&report);
    if !sweep_summary.is_empty() {
        out.push_str("\n");
        out.push_str(&sweep_summary);
    }

    out
}

/// Generate a markdown summary of sweep statistics if any groups have multiple entries.
fn generate_sweep_statistics(report: &DeltaReport) -> String {
    // Group variants by sweep name (before colon)
    let mut groups: HashMap<String, Vec<&VariantResult>> = HashMap::new();
    for variant in &report.variants {
        if let Some(colon) = variant.name.find(':') {
            let base_name = variant.name[..colon].to_string();
            groups.entry(base_name).or_default().push(variant);
        }
    }

    // Filter groups with at least 2 variants (actual sweeps)
    let sweep_groups: HashMap<_, _> = groups
        .into_iter()
        .filter(|(_, variants)| variants.len() >= 2)
        .collect();

    if sweep_groups.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    out.push_str("## Sweep Statistics\n\n");

    for (group_name, variants) in sweep_groups {
        out.push_str(&format!("### {}\n\n", group_name));
        out.push_str("| Metric | Mean | Std Dev |\n");
        out.push_str("|--------|------|---------|\n");

        let heating_vals: Vec<f64> = variants.iter().map(|v| v.annual_heating_mwh).collect();
        let cooling_vals: Vec<f64> = variants.iter().map(|v| v.annual_cooling_mwh).collect();
        let peak_h_vals: Vec<f64> = variants.iter().map(|v| v.peak_heating_kw).collect();
        let peak_c_vals: Vec<f64> = variants.iter().map(|v| v.peak_cooling_kw).collect();

        let (mean_h, std_h) = mean_std(&heating_vals);
        let (mean_c, std_c) = mean_std(&cooling_vals);
        let (mean_ph, std_ph) = mean_std(&peak_h_vals);
        let (mean_pc, std_pc) = mean_std(&peak_c_vals);

        out.push_str(&format!(
            "| Annual Heating (MWh) | {:.3} | {:.3} |\n",
            mean_h, std_h
        ));
        out.push_str(&format!(
            "| Annual Cooling (MWh) | {:.3} | {:.3} |\n",
            mean_c, std_c
        ));
        out.push_str(&format!(
            "| Peak Heating (kW) | {:.3} | {:.3} |\n",
            mean_ph, std_ph
        ));
        out.push_str(&format!(
            "| Peak Cooling (kW) | {:.3} | {:.3} |\n",
            mean_pc, std_pc
        ));
        out.push_str("\n");
    }

    out
}

/// Compute mean and sample standard deviation (n-1 denominator).
fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = variance.sqrt();
    (mean, std)
}

/// Export hourly differences to a long-format CSV.
fn export_hourly_deltas_csv(report: &DeltaReport, path: &Path) -> Result<()> {
    let mut wtr = WriterBuilder::new().has_headers(true).from_path(path)?;
    wtr.write_record(&[
        "Hour",
        "Zone",
        "Component",
        "Base_Value",
        "Variant_Value",
        "Difference",
    ])?;

    for variant in &report.variants {
        if let Some(ref deltas) = variant.hourly_differences {
            for d in deltas {
                wtr.write_record(&[
                    d.hour.to_string(),
                    d.zone.to_string(),
                    d.component.clone(),
                    format!("{:.4}", d.base_value),
                    format!("{:.4}", d.variant_value),
                    format!("{:.4}", d.difference),
                ])?;
            }
        }
    }

    wtr.flush()?;
    Ok(())
}

/// Run the comparison between a base case and multiple variants.
///
/// Returns the DeltaReport and the base SimulationResult (for report generation).
pub fn run_comparison(
    base: &CaseSpec,
    variants: &[(String, CaseSpec)],
    include_hourly: bool,
) -> Result<(DeltaReport, SimulationResult)> {
    let base_result = run_simulation(base, include_hourly)?;
    let mut variant_results = Vec::new();

    for (name, spec) in variants {
        let var_result = run_simulation(spec, include_hourly)?;
        let hourly_differences = if include_hourly {
            let base_hourly = base_result
                .hourly_data
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Base hourly data missing"))?;
            let var_hourly = var_result
                .hourly_data
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Variant hourly data missing"))?;
            Some(compute_hourly_deltas(base_hourly, var_hourly)?)
        } else {
            None
        };

        variant_results.push(VariantResult {
            name: name.clone(),
            annual_heating_mwh: var_result.annual_heating_mwh,
            annual_cooling_mwh: var_result.annual_cooling_mwh,
            peak_heating_kw: var_result.peak_heating_kw,
            peak_cooling_kw: var_result.peak_cooling_kw,
            hourly_differences,
        });
    }

    let report = DeltaReport {
        base_name: base.case_id.clone(),
        variants: variant_results,
    };

    Ok((report, base_result))
}

/// High-level function: parse config, expand variants, simulate, and write report files.
pub fn run_and_report(config: DeltaConfig, output_dir: &Path, include_hourly: bool) -> Result<()> {
    let variants = expand_variants(&config)?;
    if variants.is_empty() {
        anyhow::bail!("No variants defined");
    }
    let (report, base_result) = run_comparison(&config.base, &variants, include_hourly)?;

    // Write markdown report
    let md_path = output_dir.join("delta_report.md");
    let md_content = generate_markdown_report(&report, &base_result);
    std::fs::write(md_path, md_content)?;

    // Write hourly differences CSV if requested
    if include_hourly {
        let csv_path = output_dir.join("hourly_differences.csv");
        export_hourly_deltas_csv(&report, &csv_path)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::ashrae_140_cases::{ASHRAE140Case, HvacSchedule};
    use tempfile::tempdir;

    #[test]
    fn test_config_parsing() {
        // Build a DeltaConfig with a CaseSpec from Case600 and some variants
        let base_spec = ASHRAE140Case::Case600.spec();
        let config = DeltaConfig {
            base: base_spec.clone(),
            variants: vec![
                Variant {
                    name: "patch_test".to_string(),
                    patch: Some({
                        let mut map = HashMap::new();
                        map.insert(
                            "infiltration_ach".to_string(),
                            serde_yaml::to_value(1.5).unwrap(),
                        );
                        map
                    }),
                    sweep: None,
                },
                Variant {
                    name: "sweep_test".to_string(),
                    patch: None,
                    sweep: Some({
                        let mut map = HashMap::new();
                        map.insert("window_u_value".to_string(), vec![2.0, 3.0]);
                        map
                    }),
                },
            ],
        };
        // Serialize to YAML and write to a temporary file
        let yaml = serde_yaml::to_string(&config).expect("Failed to serialize config to YAML");
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().join("test_delta.yaml");
        std::fs::write(&path, &yaml).expect("Failed to write temp YAML file");
        // Parse the file
        let parsed = parse_config(&path).expect("Failed to parse config");
        // Validate the parsed content
        assert_eq!(parsed.base.case_id, base_spec.case_id);
        assert_eq!(parsed.variants.len(), 2);
        assert_eq!(parsed.variants[0].name, "patch_test");
        assert!(parsed.variants[0]
            .patch
            .as_ref()
            .unwrap()
            .contains_key("infiltration_ach"));
        assert_eq!(parsed.variants[1].name, "sweep_test");
        assert!(parsed.variants[1]
            .sweep
            .as_ref()
            .unwrap()
            .contains_key("window_u_value"));
    }

    #[test]
    fn test_patch_application() {
        let base_spec = ASHRAE140Case::Case600.spec();
        let config = DeltaConfig {
            base: base_spec,
            variants: vec![Variant {
                name: "patch_test".to_string(),
                patch: Some({
                    let mut map = HashMap::new();
                    map.insert(
                        "infiltration_ach".to_string(),
                        serde_yaml::to_value(1.5).unwrap(),
                    );
                    map
                }),
                sweep: None,
            }],
        };
        let expanded = expand_variants(&config).unwrap();
        assert_eq!(expanded.len(), 1);
        let (name, spec) = &expanded[0];
        assert_eq!(name, "patch_test");
        assert_eq!(spec.infiltration_ach, 1.5);
    }

    #[test]
    fn test_sweep_expansion() {
        let base_spec = ASHRAE140Case::Case600.spec();
        let config = DeltaConfig {
            base: base_spec,
            variants: vec![Variant {
                name: "sweep_test".to_string(),
                patch: None,
                sweep: Some({
                    let mut map = HashMap::new();
                    map.insert("infiltration_ach".to_string(), vec![0.5, 1.0, 1.5]);
                    map
                }),
            }],
        };
        let expanded = expand_variants(&config).unwrap();
        assert_eq!(expanded.len(), 3);
        // Check naming
        assert!(expanded[0].0.contains("infiltration_ach=0.5"));
        assert!(expanded[1].0.contains("infiltration_ach=1.0"));
        assert!(expanded[2].0.contains("infiltration_ach=1.5"));
    }

    #[test]
    fn test_run_simulation_basic() {
        let spec = ASHRAE140Case::Case600.spec();
        let result = run_simulation(&spec, false).unwrap();
        assert!(result.annual_heating_mwh > 0.0);
        assert!(result.annual_cooling_mwh > 0.0);
        assert!(result.peak_heating_kw > 0.0);
        assert!(result.peak_cooling_kw > 0.0);
        assert!(result.hourly_data.is_none());
    }

    #[test]
    fn test_run_simulation_with_hourly() {
        let spec = ASHRAE140Case::Case600.spec();
        let result = run_simulation(&spec, true).unwrap();
        assert!(result.hourly_data.is_some());
        let hourly = result.hourly_data.as_ref().unwrap();
        assert_eq!(hourly.len(), 8760);
        assert_eq!(hourly[0].zone_temps.len(), spec.num_zones);
    }

    #[test]
    fn test_comparison() {
        let base_spec = ASHRAE140Case::Case600.spec();
        // Create variant with higher window U-value (worse insulation) to change loads
        let mut variant_spec = base_spec.clone();
        // Case600 default window U is around 3.0; change to 5.0
        variant_spec.window_properties.u_value = 5.0;
        let variants = vec![("high_u_window".to_string(), variant_spec)];
        let (report, base_result) = run_comparison(&base_spec, &variants, true).unwrap();
        assert_eq!(report.variants.len(), 1);
        let var_res = &report.variants[0];
        // Check that annual heating differs from base (non-zero difference)
        let heating_abs_diff = (var_res.annual_heating_mwh - base_result.annual_heating_mwh).abs();
        assert!(
            heating_abs_diff > 1e-6,
            "Heating diff should be non-zero: {}",
            heating_abs_diff
        );
        // Cooling may increase or decrease; just check that it's non-zero
        let cooling_abs_diff = (var_res.annual_cooling_mwh - base_result.annual_cooling_mwh).abs();
        assert!(
            cooling_abs_diff > 1e-6,
            "Cooling diff should be non-zero: {}",
            cooling_abs_diff
        );
        // Peaks should be non-zero and finite
        assert!(var_res.peak_heating_kw.is_finite() && var_res.peak_heating_kw > 0.0);
        assert!(var_res.peak_cooling_kw.is_finite() && var_res.peak_cooling_kw > 0.0);
        // Hourly differences should be populated
        assert!(var_res.hourly_differences.is_some());
        let deltas = var_res.hourly_differences.as_ref().unwrap();
        assert!(!deltas.is_empty());
        // Check that we have entries for hvac_heating at hour 0
        assert!(deltas
            .iter()
            .any(|d| d.hour == 0 && d.component == "hvac_heating" && d.zone == 0));
    }

    #[test]
    fn test_markdown_generation() {
        let base_result = SimulationResult {
            annual_heating_mwh: 5.0,
            annual_cooling_mwh: 3.0,
            peak_heating_kw: 10.0,
            peak_cooling_kw: 15.0,
            hourly_data: None,
        };
        let report = DeltaReport {
            base_name: "TestBase".to_string(),
            variants: vec![VariantResult {
                name: "TestVariant".to_string(),
                annual_heating_mwh: 6.0,
                annual_cooling_mwh: 3.5,
                peak_heating_kw: 11.0,
                peak_cooling_kw: 16.0,
                hourly_differences: None,
            }],
        };
        let md = generate_markdown_report(&report, &base_result);
        assert!(md.contains("# Delta Test Report"));
        assert!(md.contains("TestBase"));
        assert!(md.contains("TestVariant"));
        assert!(md.contains("|"));
        // Check for numeric differences
        assert!(md.contains("1.000")); // heating diff = 1.0
        assert!(md.contains("20.0%")); // heating % diff = 20%
    }

    #[test]
    fn test_csv_export() {
        let report = DeltaReport {
            base_name: "Base".to_string(),
            variants: vec![VariantResult {
                name: "Var1".to_string(),
                annual_heating_mwh: 0.0,
                annual_cooling_mwh: 0.0,
                peak_heating_kw: 0.0,
                peak_cooling_kw: 0.0,
                hourly_differences: Some(vec![
                    HourlyDelta {
                        hour: 0,
                        zone: 0,
                        component: "solar_gains".to_string(),
                        base_value: 100.0,
                        variant_value: 110.0,
                        difference: 10.0,
                    },
                    HourlyDelta {
                        hour: 0,
                        zone: 0,
                        component: "hvac_heating".to_string(),
                        base_value: 500.0,
                        variant_value: 550.0,
                        difference: 50.0,
                    },
                    HourlyDelta {
                        hour: 1,
                        zone: 0,
                        component: "solar_gains".to_string(),
                        base_value: 120.0,
                        variant_value: 130.0,
                        difference: 10.0,
                    },
                ]),
            }],
        };
        let temp_dir = tempdir().unwrap();
        let csv_path = temp_dir.path().join("delta_hourly_test.csv");
        export_hourly_deltas_csv(&report, &csv_path).unwrap();
        // Read file
        let content = std::fs::read_to_string(&csv_path).unwrap();
        let mut lines = content.lines();
        // Header
        let header = lines.next().unwrap();
        assert!(header.contains("Hour"));
        assert!(header.contains("Zone"));
        assert!(header.contains("Component"));
        assert!(header.contains("Base_Value"));
        assert!(header.contains("Variant_Value"));
        assert!(header.contains("Difference"));
        // Data rows
        let data_lines: Vec<&str> = lines.collect();
        assert_eq!(data_lines.len(), 3);
        // Verify first row has values
        let first = data_lines[0];
        assert!(first.contains("0")); // hour
        assert!(first.contains("0")); // zone
        assert!(first.contains("solar_gains"));
        assert!(first.contains("100.0000"));
    }

    #[test]
    fn test_sweep_statistics() {
        let base_result = SimulationResult {
            annual_heating_mwh: 5.0,
            annual_cooling_mwh: 3.0,
            peak_heating_kw: 10.0,
            peak_cooling_kw: 15.0,
            hourly_data: None,
        };
        let report = DeltaReport {
            base_name: "Base".to_string(),
            variants: vec![
                VariantResult {
                    name: "sweep_infil: 0.5".to_string(),
                    annual_heating_mwh: 5.5,
                    annual_cooling_mwh: 3.5,
                    peak_heating_kw: 11.0,
                    peak_cooling_kw: 16.0,
                    hourly_differences: None,
                },
                VariantResult {
                    name: "sweep_infil: 1.0".to_string(),
                    annual_heating_mwh: 6.0,
                    annual_cooling_mwh: 4.0,
                    peak_heating_kw: 12.0,
                    peak_cooling_kw: 17.0,
                    hourly_differences: None,
                },
                VariantResult {
                    name: "sweep_infil: 1.5".to_string(),
                    annual_heating_mwh: 6.5,
                    annual_cooling_mwh: 4.5,
                    peak_heating_kw: 13.0,
                    peak_cooling_kw: 18.0,
                    hourly_differences: None,
                },
                VariantResult {
                    name: "patch_test".to_string(),
                    annual_heating_mwh: 5.2,
                    annual_cooling_mwh: 3.2,
                    peak_heating_kw: 10.5,
                    peak_cooling_kw: 15.5,
                    hourly_differences: None,
                },
            ],
        };
        let md = generate_markdown_report(&report, &base_result);
        assert!(md.contains("## Sweep Statistics"));
        assert!(md.contains("sweep_infil"));
        // Check for mean heating value approx 6.0
        assert!(md.contains("6.000"));
        // Std dev for heating: for [5.5,6.0,6.5] sample std dev = 0.5
        assert!(md.contains("0.500"));
    }
}
