//! Typed loaders for EnergyPlus reference CSVs.
//!
//! This module is the single source of truth for parsing the E+ reference data
//! shipped under `tests/reference_data/`. The TDD framework
//! (`crate::testing::tdd_framework`) consumes these loaders instead of
//! asserting against hand-typed constants — every `TestCaseResult` produced
//! by `run_<domain>_tests()` now compares Fluxion output to an actual
//! EnergyPlus row, so the framework can fail on a real physics regression.
//!
//! # CSV format
//!
//! See `tests/reference_data/README.md` for the column layouts. All loaders
//! tolerate the `#`-prefixed comment header lines and the column-header row.

use std::fs;
use std::path::Path;

/// Sub-directory containing the EnergyPlus reference CSVs.
pub const REFERENCE_DATA_DIR: &str = "tests/reference_data";

/// One row of a conduction step-response CSV.
///
/// Mirrors `tests/reference_data/conduction/step_response_*.csv` columns:
///
/// `hour, T_outdoor, T_zone, T_surface_inside, T_surface_outside, q_inside_Wm2, q_outside_Wm2`
#[derive(Debug, Clone, Copy)]
pub struct StepResponseRow {
    /// Elapsed hours since simulation start.
    pub hour: f64,
    /// Outdoor air drybulb (°C).
    pub t_outdoor: f64,
    /// Zone mean air temperature (°C).
    pub t_zone: f64,
    /// Inside face surface temperature of the test surface (°C).
    pub t_surface_inside: f64,
    /// Outside face surface temperature of the test surface (°C).
    pub t_surface_outside: f64,
    /// Inside face conduction heat flux, positive into zone (W/m²).
    pub q_inside_wm2: f64,
    /// Outside face conduction heat flux, positive into wall (W/m²).
    pub q_outside_wm2: f64,
}

/// Parsed conduction step-response CSV.
#[derive(Debug, Clone)]
pub struct StepResponseRef {
    /// Construction name (e.g. `"200mm_concrete"`).
    pub name: String,
    /// Source model label from the CSV comments.
    pub source_model: String,
    /// EnergyPlus version recorded in the CSV header.
    pub energyplus_version: String,
    /// All data rows in chronological order.
    pub rows: Vec<StepResponseRow>,
}

impl StepResponseRef {
    /// Mean inside-face heat flux (W/m²) across the entire run.
    pub fn mean_q_inside(&self) -> f64 {
        if self.rows.is_empty() {
            return 0.0;
        }
        self.rows.iter().map(|r| r.q_inside_wm2).sum::<f64>() / self.rows.len() as f64
    }

    /// Peak inside-face heat flux (W/m²) — the largest magnitude.
    pub fn peak_abs_q_inside(&self) -> f64 {
        self.rows
            .iter()
            .map(|r| r.q_inside_wm2.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Effective R-value (m²·K/W) derived from mean flux and mean temperature
    /// drop across the wall. Returns `None` if the surface is in steady state
    /// (ΔT ≈ 0).
    pub fn effective_r_value(&self) -> Option<f64> {
        let mean_dt: f64 = self
            .rows
            .iter()
            .map(|r| (r.t_surface_inside - r.t_surface_outside).abs())
            .sum::<f64>()
            / self.rows.len() as f64;
        let mean_flux = self.mean_q_inside().abs();
        if mean_dt < 1e-6 || mean_flux < 1e-6 {
            None
        } else {
            Some(mean_dt / mean_flux)
        }
    }
}

/// Load a conduction step-response CSV.
///
/// `name` is the construction suffix used in the file name, e.g.
/// `"200mm_concrete"`, `"composite"`, `"floor"`, `"lightweight"`, `"roof"`.
pub fn load_conduction_step_response(name: &str) -> Result<StepResponseRef, String> {
    let path = Path::new(REFERENCE_DATA_DIR)
        .join("conduction")
        .join(format!("step_response_{}.csv", name));
    let content = fs::read_to_string(&path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    parse_step_response(name, &content)
}

fn parse_step_response(name: &str, content: &str) -> Result<StepResponseRef, String> {
    let mut energyplus_version = String::new();
    let mut source_model = String::new();
    let mut rows: Vec<StepResponseRow> = Vec::new();
    // Auto-detect column layout. Two variants exist in the reference data:
    //   7 cols: hour, T_outdoor, T_zone, T_surface_inside, T_surface_outside,
    //           q_inside_Wm2, q_outside_Wm2
    //   6 cols: hour, T_ext,    T_surface_inside, T_surface_outside,
    //           heat_flux_inside, heat_flux_outside   (free-floating, no T_zone)
    let mut seven_col = false;
    let mut six_col = false;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("# EnergyPlus Version:") {
            energyplus_version = rest.trim().to_string();
            continue;
        }
        if let Some(rest) = line.strip_prefix("# Model:") {
            source_model = rest.trim().to_string();
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        // Detect column layout from the header row.
        if line.starts_with("hour") {
            if line.contains("T_zone") {
                seven_col = true;
            } else {
                six_col = true;
            }
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if seven_col && parts.len() >= 7 {
            rows.push(StepResponseRow {
                hour: parts[0].trim().parse().map_err(|e| {
                    format!("bad hour in row {:?}: {}", parts, e)
                })?,
                t_outdoor: parts[1].trim().parse().unwrap_or(0.0),
                t_zone: parts[2].trim().parse().unwrap_or(0.0),
                t_surface_inside: parts[3].trim().parse().unwrap_or(0.0),
                t_surface_outside: parts[4].trim().parse().unwrap_or(0.0),
                q_inside_wm2: parts[5].trim().parse().unwrap_or(0.0),
                q_outside_wm2: parts[6].trim().parse().unwrap_or(0.0),
            });
        } else if six_col && parts.len() >= 6 {
            rows.push(StepResponseRow {
                hour: parts[0].trim().parse().map_err(|e| {
                    format!("bad hour in row {:?}: {}", parts, e)
                })?,
                t_outdoor: parts[1].trim().parse().unwrap_or(0.0),
                t_zone: 0.0, // not recorded in the 6-column format
                t_surface_inside: parts[2].trim().parse().unwrap_or(0.0),
                t_surface_outside: parts[3].trim().parse().unwrap_or(0.0),
                q_inside_wm2: parts[4].trim().parse().unwrap_or(0.0),
                q_outside_wm2: parts[5].trim().parse().unwrap_or(0.0),
            });
        }
    }

    if rows.is_empty() {
        return Err(format!(
            "no data rows found in step_response_{}.csv",
            name
        ));
    }

    Ok(StepResponseRef {
        name: name.to_string(),
        source_model,
        energyplus_version,
        rows,
    })
}

/// One row of a zone-balance reference CSV (annual/peak metrics table).
#[derive(Debug, Clone, Copy)]
pub struct ZoneBalanceMetric {
    /// Metric name, e.g. `"annual_heating"`.
    pub metric: &'static str,
    /// EnergyPlus reference minimum value (matches `unit`).
    pub ref_min: f64,
    /// EnergyPlus reference maximum value (matches `unit`).
    pub ref_max: f64,
    /// Midpoint of the reference band — the value compared against.
    pub ref_midpoint: f64,
    /// Tolerance band half-width, as a fraction (e.g. `0.15` for ±15%).
    pub tolerance_pct: f64,
}

/// Parsed zone-balance annual reference (`case_XXX_energy_reference.csv`).
#[derive(Debug, Clone)]
pub struct ZoneBalanceRef {
    /// ASHRAE 140 case ID, e.g. `"600"`.
    pub case_id: String,
    /// Parsed metrics.
    pub metrics: Vec<ZoneBalanceMetric>,
}

impl ZoneBalanceRef {
    /// Look up a metric by name.
    pub fn get(&self, metric: &str) -> Option<&ZoneBalanceMetric> {
        self.metrics.iter().find(|m| m.metric == metric)
    }

    /// Annual heating in MWh (midpoint), or `0.0` if missing.
    pub fn annual_heating_mwh(&self) -> f64 {
        self.get("annual_heating").map(|m| m.ref_midpoint).unwrap_or(0.0)
    }

    /// Annual cooling in MWh (midpoint), or `0.0` if missing.
    pub fn annual_cooling_mwh(&self) -> f64 {
        self.get("annual_cooling").map(|m| m.ref_midpoint).unwrap_or(0.0)
    }

    /// Peak heating in kW (midpoint), or `0.0` if missing.
    pub fn peak_heating_kw(&self) -> f64 {
        self.get("peak_heating").map(|m| m.ref_midpoint).unwrap_or(0.0)
    }

    /// Peak cooling in kW (midpoint), or `0.0` if missing.
    pub fn peak_cooling_kw(&self) -> f64 {
        self.get("peak_cooling").map(|m| m.ref_midpoint).unwrap_or(0.0)
    }
}

/// Load a zone-balance annual energy reference CSV (`case_XXX_energy_reference.csv`).
pub fn load_zone_balance_case(case_id: &str) -> Result<ZoneBalanceRef, String> {
    let path = Path::new(REFERENCE_DATA_DIR)
        .join("zone_balance")
        .join(format!("case_{}_energy_reference.csv", case_id));
    let content = fs::read_to_string(&path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    parse_zone_balance(case_id, &content)
}

fn parse_zone_balance(case_id: &str, content: &str) -> Result<ZoneBalanceRef, String> {
    let mut metrics: Vec<ZoneBalanceMetric> = Vec::new();
    let mut header_seen = false;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !header_seen {
            header_seen = true;
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }
        let metric = parts[0].trim();
        let ref_min: f64 = parts[2].trim().parse().unwrap_or(0.0);
        let ref_max: f64 = parts[3].trim().parse().unwrap_or(0.0);
        let ref_midpoint: f64 = parts[4].trim().parse().unwrap_or(0.0);
        let tolerance_pct: f64 = parts[5].trim().parse().unwrap_or(15.0) / 100.0;
        metrics.push(ZoneBalanceMetric {
            metric: leak_str(metric),
            ref_min,
            ref_max,
            ref_midpoint,
            tolerance_pct,
        });
    }

    if metrics.is_empty() {
        return Err(format!(
            "no metric rows found in case_{}_energy_reference.csv",
            case_id
        ));
    }

    Ok(ZoneBalanceRef {
        case_id: case_id.to_string(),
        metrics,
    })
}

/// Tiny `&'static str` leak to satisfy `ZoneBalanceMetric::metric`'s lifetime
/// without dragging in a string interner. Acceptable because metric names are
/// a small, fixed set (`annual_heating`, `peak_cooling`, …) parsed once at
/// load time.
fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

/// One row of the south-vertical surface irradiance CSV.
///
/// Columns: `hour(1-8760), beam_irradiance(W/m2), ground_diffuse_irradiance(W/m2)`.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceIrradianceRow {
    pub hour: usize,
    pub beam_wm2: f64,
    pub ground_diffuse_wm2: f64,
}

/// Load the south-vertical surface irradiance reference CSV.
pub fn load_surface_irradiance_south() -> Result<Vec<SurfaceIrradianceRow>, String> {
    let path = Path::new(REFERENCE_DATA_DIR)
        .join("solar")
        .join("surface_irradiance_south.csv");
    let content = fs::read_to_string(&path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    let mut rows = Vec::with_capacity(8760);
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("hour") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 3 {
            continue;
        }
        rows.push(SurfaceIrradianceRow {
            hour: parts[0].trim().parse().unwrap_or(0),
            beam_wm2: parts[1].trim().parse().unwrap_or(0.0),
            ground_diffuse_wm2: parts[2].trim().parse().unwrap_or(0.0),
        });
    }
    Ok(rows)
}

/// One row of the Denver infiltration reference CSV.
///
/// Columns: `hour(1-8760), outdoor_temp(C), wind_speed(m/s),
/// infiltration_ach(1/h), vent_conductance(W/K)`.
#[derive(Debug, Clone, Copy)]
pub struct InfiltrationRow {
    pub hour: usize,
    pub outdoor_temp: f64,
    pub wind_speed: f64,
    pub infiltration_ach: f64,
    pub vent_conductance: f64,
}

/// Load the Denver infiltration reference CSV.
pub fn load_infiltration_denver() -> Result<Vec<InfiltrationRow>, String> {
    let path = Path::new(REFERENCE_DATA_DIR)
        .join("ventilation")
        .join("infiltration_denver.csv");
    let content = fs::read_to_string(&path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    let mut rows = Vec::with_capacity(8760);
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("hour") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 5 {
            continue;
        }
        rows.push(InfiltrationRow {
            hour: parts[0].trim().parse().unwrap_or(0),
            outdoor_temp: parts[1].trim().parse().unwrap_or(0.0),
            wind_speed: parts[2].trim().parse().unwrap_or(0.0),
            infiltration_ach: parts[3].trim().parse().unwrap_or(0.0),
            vent_conductance: parts[4].trim().parse().unwrap_or(0.0),
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_concrete_step_response_is_populated() {
        let data = load_conduction_step_response("200mm_concrete")
            .expect("concrete step response CSV must exist");
        assert!(!data.rows.is_empty(), "step response has no rows");
        assert_eq!(
            data.rows.len(),
            288,
            "72 h × 4 steps/h = 288 rows for the concrete step response"
        );
        assert!(data.peak_abs_q_inside() > 0.0);
        // The free-floating concrete wall has near-zero mean inside-face flux
        // (thermal mass absorbs the step), so effective_r_value() is not
        // meaningful here.  Instead, verify the peak outside-face flux is
        // significant (driven by the cold January boundary).
        let peak_q_out = data
            .rows
            .iter()
            .map(|r| r.q_outside_wm2.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            peak_q_out > 50.0,
            "Peak outside-face flux for 200mm concrete should exceed 50 W/m², got {}",
            peak_q_out
        );
    }

    #[test]
    fn load_zone_balance_600_has_heating_and_cooling() {
        let case = load_zone_balance_case("600").expect("case 600 CSV must exist");
        let heating = case.annual_heating_mwh();
        let cooling = case.annual_cooling_mwh();
        assert!(heating > 0.0, "Case 600 annual heating must be > 0");
        assert!(cooling > 0.0, "Case 600 annual cooling must be > 0");
        // ASHRAE 140 Case 600 ref band: heating 4.36-5.79 MWh, cooling 3.92-6.14 MWh
        assert!((4.0..6.5).contains(&heating));
        assert!((3.5..6.5).contains(&cooling));
    }

    #[test]
    fn load_surface_irradiance_south_has_8760_hours() {
        let rows = load_surface_irradiance_south().expect("south irradiance CSV");
        assert_eq!(rows.len(), 8760, "expected full TMY year");
        let noon_summer = &rows[4350..4380];
        let max_beam = noon_summer
            .iter()
            .map(|r| r.beam_wm2)
            .fold(0.0_f64, f64::max);
        assert!(
            max_beam > 200.0,
            "Denver summer noon beam on south wall should exceed 200 W/m², got {}",
            max_beam
        );
    }

    #[test]
    fn load_infiltration_denver_constant_ach() {
        let rows = load_infiltration_denver().expect("denver infiltration CSV");
        assert_eq!(rows.len(), 8760);
        // E+ model uses constant 0.5 ACH and constant 21.6 W/K conductance.
        let ach_min = rows
            .iter()
            .map(|r| r.infiltration_ach)
            .fold(f64::INFINITY, f64::min);
        let ach_max = rows
            .iter()
            .map(|r| r.infiltration_ach)
            .fold(0.0_f64, f64::max);
        assert!(
            (ach_max - ach_min).abs() < 1e-9,
            "constant-ACH model should have zero variance, got min={}, max={}",
            ach_min,
            ach_max
        );
        let c_min = rows
            .iter()
            .map(|r| r.vent_conductance)
            .fold(f64::INFINITY, f64::min);
        let c_max = rows
            .iter()
            .map(|r| r.vent_conductance)
            .fold(0.0_f64, f64::max);
        assert!((c_max - c_min).abs() < 1e-9);
        assert!(
            (c_max - 21.6).abs() < 0.01,
            "0.5 ACH × 129.6 m³ × 1.2 kg/m³ × 1000 J/(kg·K) / 3600 ≈ 21.6 W/K, got {}",
            c_max
        );
    }
}