//! DX (direct-expansion) cooling equipment model driven by an ODB/EWB/EDB
//! performance map — the §9 CE100–CE200 equipment layer (Issue #4296 part 2).
//!
//! ## Design decisions (Alex's fidelity-first call, 2026-10-18)
//!
//! Alex's direction: "Do the option which is the most accurate to real dx
//! equipment usage and performance." Applied to the four engine-semantics
//! questions from the #4296 part-2 assessment:
//!
//! 1. **Standalone, not a #3984 precursor.** This is a steady-state
//!    validation component in `src/sim/hvac/`, independent of the teacher
//!    pressure-driven air loop. The CE100–CE200 cases need exactly one
//!    evaluation per case condition (fixed setpoint, fixed gains, fixed
//!    ODB/EWB); #3984's dynamic control semantics need none of this.
//!    Building it in the `hvac` module's existing vocabulary keeps the
//!    adoption path open without coupling it to the air-loop design.
//! 2. **Tabular map interpolation, not fitted curves.** The normative
//!    material of ASHRAE 140-2023 §9 is the tabular map (Tables 26a/26c/26e,
//!    transcribed in `data/reference/ashrae140/section9_ce100_performance_map.json`
//!    with SHA-256 provenance). The NREL BESTEST-GSR reference implementation
//!    runs EnergyPlus `CoilCoolingDXSingleSpeed` over *biquadratic curve fits*
//!    of that same table; fitting our own curves would invent numbers
//!    (RULES.md) where the standard supplies data. We interpolate the table
//!    directly (bilinear in EWB×EDB per ODB, linear in ODB), which is more
//!    faithful to the normative source; cross-engine comparability with the
//!    published EnergyPlus ensemble is diagnostic context, not the criterion.
//! 3. **Wet-coil SHR at the entering condition, from the map.** The map
//!    carries sensible capacity per (ODB, EWB, EDB) and total capacity per
//!    (ODB, EWB); SHR is evaluated at the case's entering conditions as
//!    interpolated sensible / interpolated total — not a constant rated SHR.
//!    The map's own dry-coil rule applies (where sensible exceeds total the
//!    coil is dry and total equals sensible).
//! 4. **Part-load cycling per the EnergyPlus formulation, with the GSR
//!    reference's coefficients.** Real DX equipment at low part-load ratio
//!    loses capacity and efficiency to cycling (thermostat anticipation,
//!    off-cycle refrigerant migration, wind-down). Following the NREL
//!    BESTEST-GSR reference (`shared_resources/bestest_model_methods.rb`,
//!    `clg_part_load_ratio` curve, verified 2026-10-18):
//!    `PartLoadFrac = 0.771 + 0.229·PLR`. Delivered capacity is
//!    `capacity × PLR`; consumed compressor power is
//!    `power × PLR / PartLoadFrac`. Constants are transcribed from the
//!    reference implementation, not invented.
//!
//! ## Fan heat
//!
//! Gross capacities (Table 26c) minus fan heat give net capacities. The map
//! binds the manufacturer default (329 W) and the actual measured indoor fan
//! power (230 W); the AHRI rating point reconciles gross 8181 W / net 7852 W
//! (default) / adjusted net 7951 W (actual).

use serde::Deserialize;
use std::collections::BTreeMap;

/// Parsed representation of `section9_ce100_performance_map.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct CePerformanceMap {
    /// `edb_columns_C` — entering dry-bulb grid [°C].
    #[serde(rename = "edb_columns_C")]
    edb_columns_c: Vec<f64>,
    /// `fan_heat` block.
    fan_heat: FanHeat,
    /// `ari_rating_point` block.
    ari_rating_point: AriRatingPoint,
    /// `tables` — each a 6-ODB × 4-EWB grid keyed by stringified °C values.
    tables: CeTables,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FanHeat {
    #[serde(rename = "manufacturer_default_W")]
    manufacturer_default_w: f64,
    #[serde(rename = "actual_W")]
    actual_w: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct AriRatingPoint {
    #[serde(rename = "net_total_capacity_W")]
    net_total_capacity_w: f64,
    #[serde(rename = "gross_total_capacity_W")]
    gross_total_capacity_w: f64,
    #[serde(rename = "adjusted_net_total_capacity_W")]
    adjusted_net_total_capacity_w: f64,
    #[serde(rename = "compressor_power_W")]
    #[allow(dead_code)]
    compressor_power_w: f64,
    #[serde(rename = "COP")]
    cop: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct CeTables {
    #[serde(rename = "net_Table_26a")]
    net_table_26a: CeGrid,
    #[serde(rename = "gross_Table_26c")]
    gross_table_26c: CeGrid,
    #[serde(rename = "adjusted_net_Table_26e")]
    adjusted_net_table_26e: CeGrid,
}

/// One 6-ODB × 4-EWB table; JSON keys are stringified temperatures
/// ("29.4", "19.4") at one decimal place, deserialized directly into
/// nested maps.
type CeGrid = BTreeMap<String, BTreeMap<String, CeCell>>;

#[derive(Debug, Clone, Deserialize)]
struct CeCell {
    #[serde(rename = "total_capacity_kW")]
    total_capacity_kw: f64,
    #[serde(rename = "sensible_capacity_kW_by_edb")]
    sensible_capacity_kw_by_edb: BTreeMap<String, f64>,
    #[serde(rename = "compressor_power_kW")]
    compressor_power_kw: f64,
    #[serde(rename = "apparatus_dew_point_C")]
    #[allow(dead_code)]
    apparatus_dew_point_c: f64,
}

/// Which committed table an evaluation reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CeTable {
    /// `net_Table_26a` — net of the manufacturer-default fan heat (329 W).
    Net26a,
    /// `gross_Table_26c` — the equipment's full-load physics.
    Gross26c,
    /// `adjusted_net_Table_26e` — net of the actual fan power (230 W).
    AdjustedNet26e,
}

/// Interpolated evaluation of the DX unit at one operating point.
#[derive(Debug, Clone, PartialEq)]
pub struct DxMapEvaluation {
    /// Total (sensible + latent) capacity at the entering conditions [W].
    pub total_capacity_w: f64,
    /// Sensible capacity at the entering dry-bulb [W].
    pub sensible_capacity_w: f64,
    /// Sensible heat ratio at the entering conditions (map-derived).
    pub shr: f64,
    /// Compressor power [W].
    pub compressor_power_w: f64,
    /// Indoor fan heat consistent with this evaluation's basis [W].
    pub fan_heat_w: f64,
    /// COP: total capacity / (compressor + fan) power at this point.
    pub cop: f64,
    /// Part-load fraction `0.771 + 0.229·PLR` (cycling degradation).
    pub part_load_fraction: f64,
    /// Delivered (time-averaged) capacity: `total_capacity_w × PLR` [W].
    pub delivered_capacity_w: f64,
    /// Delivered compressor power: `power × PLR / part_load_fraction` [W].
    pub delivered_power_w: f64,
}

impl CePerformanceMap {
    /// Parse the committed CE100 performance map JSON.
    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        let map: CePerformanceMap = serde_json::from_str(json)?;
        map.validate()?;
        Ok(map)
    }

    /// Structural validation: complete 6×4 grids with 5-point EDB rows.
    fn validate(&self) -> Result<(), serde_json::Error> {
        for (name, grid) in [
            ("net_Table_26a", &self.tables.net_table_26a),
            ("gross_Table_26c", &self.tables.gross_table_26c),
            (
                "adjusted_net_Table_26e",
                &self.tables.adjusted_net_table_26e,
            ),
        ] {
            if grid.len() != 6 {
                return Err(serde::de::Error::custom(format!(
                    "{name}: expected 6 ODB rows, got {}",
                    grid.len()
                )));
            }
            for (odb, row) in grid {
                if row.len() != 4 {
                    return Err(serde::de::Error::custom(format!(
                        "{name} ODB {odb}: expected 4 EWB cells, got {}",
                        row.len()
                    )));
                }
                for (ewb, cell) in row {
                    if cell.sensible_capacity_kw_by_edb.len() != 5 {
                        return Err(serde::de::Error::custom(format!(
                            "{name} ODB {odb} EWB {ewb}: expected 5 EDB columns"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn grid(&self, table: CeTable) -> &BTreeMap<String, BTreeMap<String, CeCell>> {
        match table {
            CeTable::Net26a => &self.tables.net_table_26a,
            CeTable::Gross26c => &self.tables.gross_table_26c,
            CeTable::AdjustedNet26e => &self.tables.adjusted_net_table_26e,
        }
    }

    fn axis(&self, table: CeTable, level: usize) -> Vec<f64> {
        let grid = self.grid(table);
        let mut keys: Vec<f64> = match level {
            0 => grid.keys().filter_map(|k| k.parse().ok()).collect(),
            _ => grid
                .values()
                .next()
                .map(|row| row.keys().filter_map(|k| k.parse().ok()).collect())
                .unwrap_or_default(),
        };
        keys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        keys
    }

    fn cell(&self, table: CeTable, odb: f64, ewb: f64) -> Option<&CeCell> {
        self.grid(table).get(&fmt_key(odb))?.get(&fmt_key(ewb))
    }

    /// Bilinear-in-(EWB, ODB) interpolation of one scalar field over the
    /// grid, clamped at the grid edges (CE cases stay inside the grid;
    /// clamping is documented behaviour for slightly out-of-grid calls).
    fn interpolate_scalar<F>(&self, table: CeTable, odb: f64, ewb: f64, field: F) -> f64
    where
        F: Fn(&CeCell) -> f64,
    {
        let odbs = self.axis(table, 0);
        let ewbs = self.axis(table, 1);
        let (o0, o1, to) = bracket(&odbs, odb);
        let (e0, e1, te) = bracket(&ewbs, ewb);
        let f = |o: f64, e: f64| self.cell(table, o, e).map(&field).unwrap_or(f64::NAN);
        let f00 = f(o0, e0);
        let f01 = f(o0, e1);
        let f10 = f(o1, e0);
        let f11 = f(o1, e1);
        let f_e0 = f00 + (f01 - f00) * te;
        let f_e1 = f10 + (f11 - f10) * te;
        f_e0 + (f_e1 - f_e0) * to
    }

    /// Sensible capacity interpolated over (EDB, EWB) at one ODB, then
    /// linearly in ODB — sensible is tabulated per EDB, total is not.
    fn interpolate_sensible(&self, table: CeTable, odb: f64, ewb: f64, edb: f64) -> f64 {
        let odbs = self.axis(table, 0);
        let ewbs = self.axis(table, 1);
        let (o0, o1, to) = bracket(&odbs, odb);
        let (e0, e1, te) = bracket(&ewbs, ewb);
        let (lo_d, hi_d, td) = bracket(&self.edb_columns_c, edb);
        let sens = |o: f64, e: f64| {
            self.cell(table, o, e)
                .and_then(|c| {
                    let s0 = c.sensible_capacity_kw_by_edb.get(&fmt_key(lo_d));
                    let s1 = c.sensible_capacity_kw_by_edb.get(&fmt_key(hi_d));
                    s0.zip(s1).map(|(a, b)| a + (b - a) * td)
                })
                .unwrap_or(f64::NAN)
        };
        let s_e0 = sens(o0, e0) + (sens(o0, e1) - sens(o0, e0)) * te;
        let s_e1 = sens(o1, e0) + (sens(o1, e1) - sens(o1, e0)) * te;
        s_e0 + (s_e1 - s_e0) * to
    }

    /// Evaluate the unit at one operating point.
    ///
    /// * `ewb_c` / `odb_c` / `edb_c` — entering wet-bulb / outdoor dry-bulb /
    ///   entering dry-bulb [°C].
    /// * `plr` — part-load ratio in [0, 1] (1.0 = full load).
    /// * `fan_heat_w` — fan heat consistent with the chosen table's basis
    ///   (329 W for the gross→net definition, 230 W for adjusted-net).
    pub fn evaluate(
        &self,
        table: CeTable,
        ewb_c: f64,
        odb_c: f64,
        edb_c: f64,
        plr: f64,
        fan_heat_w: f64,
    ) -> DxMapEvaluation {
        // Map values are tabulated in kW; evaluate in W.
        let total_kw = self.interpolate_scalar(table, odb_c, ewb_c, |c| c.total_capacity_kw);
        let sensible_kw = self.interpolate_sensible(table, odb_c, ewb_c, edb_c);
        let power_kw = self.interpolate_scalar(table, odb_c, ewb_c, |c| c.compressor_power_kw);
        // The map's dry-coil rule: where sensible exceeds total the coil is
        // dry and total capacity equals sensible capacity.
        let (total_kw, sensible_kw) = if sensible_kw > total_kw {
            (sensible_kw, sensible_kw)
        } else {
            (total_kw, sensible_kw)
        };
        let total_w = total_kw * 1000.0;
        let sensible_w = sensible_kw * 1000.0;
        let compressor_w = power_kw * 1000.0;
        let plr = plr.clamp(0.0, 1.0);
        let plf = 0.771 + 0.229 * plr;
        DxMapEvaluation {
            total_capacity_w: total_w,
            sensible_capacity_w: sensible_w,
            shr: if total_w > 0.0 {
                sensible_w / total_w
            } else {
                1.0
            },
            compressor_power_w: compressor_w,
            fan_heat_w,
            cop: if compressor_w + fan_heat_w > 0.0 {
                total_w / (compressor_w + fan_heat_w)
            } else {
                f64::NAN
            },
            part_load_fraction: plf,
            delivered_capacity_w: total_w * plr,
            delivered_power_w: compressor_w * plr / plf,
        }
    }

    /// The AHRI rating point as transcribed (`ari_rating_point`), for
    /// reconciliation tests and §9 reporting: (gross W, net W, adjusted
    /// net W, COP).
    pub fn ari_rating_point_watts(&self) -> (f64, f64, f64, f64) {
        (
            self.ari_rating_point.gross_total_capacity_w,
            self.ari_rating_point.net_total_capacity_w,
            self.ari_rating_point.adjusted_net_total_capacity_w,
            self.ari_rating_point.cop,
        )
    }

    /// Manufacturer-default indoor fan heat (329 W).
    pub fn fan_heat_default_w(&self) -> f64 {
        self.fan_heat.manufacturer_default_w
    }

    /// Actual measured indoor fan power (230 W).
    pub fn fan_heat_actual_w(&self) -> f64 {
        self.fan_heat.actual_w
    }

    /// Entering dry-bulb grid [°C].
    pub fn edb_columns(&self) -> &[f64] {
        &self.edb_columns_c
    }
}

/// Bracket `x` in an ascending axis: returns (lo, hi, t) with lo ≤ x ≤ hi,
/// t ∈ [0,1] the interpolation fraction; clamped at both ends.
fn bracket(axis: &[f64], x: f64) -> (f64, f64, f64) {
    debug_assert!(axis.windows(2).all(|w| w[0] < w[1]));
    let last = axis[axis.len() - 1];
    if x <= axis[0] {
        return (axis[0], axis[0], 0.0);
    }
    if x >= last {
        return (last, last, 0.0);
    }
    for w in axis.windows(2) {
        if x <= w[1] {
            let t = (x - w[0]) / (w[1] - w[0]);
            return (w[0], w[1], t);
        }
    }
    unreachable!("x inside axis")
}

/// JSON keys are stringified temperatures at one decimal place.
fn fmt_key(v: f64) -> String {
    format!("{v:.1}")
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAP_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/reference/ashrae140/section9_ce100_performance_map.json"
    ));

    fn load() -> CePerformanceMap {
        CePerformanceMap::from_json_str(MAP_JSON).expect("map parses and validates")
    }

    #[test]
    fn map_parses_with_complete_grids() {
        let m = load();
        assert_eq!(m.axis(CeTable::Net26a, 0).len(), 6);
        assert_eq!(m.axis(CeTable::Net26a, 1).len(), 4);
        assert_eq!(m.edb_columns().len(), 5);
    }

    #[test]
    fn grid_nodes_reproduce_tabulated_values_exactly() {
        let m = load();
        let e = m.evaluate(CeTable::Net26a, 15.0, 29.4, 22.2, 1.0, 0.0);
        assert_eq!(e.total_capacity_w, 7090.0);
        assert_eq!(e.sensible_capacity_w, 6210.0);
        assert_eq!(e.compressor_power_w, 1620.0);
        // SHR at entering conditions from the map itself.
        assert!((e.shr - 6.21 / 7.09).abs() < 1e-12);
    }

    #[test]
    fn ari_rating_point_reconciles() {
        let m = load();
        let (gross, net, adjusted_net, cop) = m.ari_rating_point_watts();
        assert_eq!(gross, 8181.0);
        assert_eq!(net, 7852.0);
        assert_eq!(adjusted_net, 7951.0);
        assert!((cop - 3.62).abs() < 1e-12);
        // net = gross - 329 W manufacturer default; adjusted = gross - 230 W.
        assert!((gross - net - m.fan_heat_default_w()).abs() < 1e-9);
        assert!((gross - adjusted_net - m.fan_heat_actual_w()).abs() < 1e-9);
    }

    #[test]
    fn interpolation_is_exact_at_nodes_and_smooth_between() {
        let m = load();
        let a = m.evaluate(CeTable::Gross26c, 19.4, 35.0, 26.7, 1.0, 0.0);
        let b = m.evaluate(CeTable::Gross26c, 19.4, 35.0, 22.2, 1.0, 0.0);
        // Node values reproduce the table exactly (26.7 column at ODB 35 / EWB 19.4).
        let c = m.cell(CeTable::Gross26c, 35.0, 19.4).unwrap();
        assert_eq!(
            a.sensible_capacity_w,
            c.sensible_capacity_kw_by_edb["26.7"] * 1000.0
        );
        assert_eq!(
            b.sensible_capacity_w,
            c.sensible_capacity_kw_by_edb["22.2"] * 1000.0
        );
        // Piecewise-linear: the midpoint of two ADJACENT EDB columns is
        // the mean of the two tabulated values.
        let s1 = c.sensible_capacity_kw_by_edb["24.4"] * 1000.0;
        let s2 = c.sensible_capacity_kw_by_edb["25.6"] * 1000.0;
        let mid = m.evaluate(CeTable::Gross26c, 19.4, 35.0, (24.4 + 25.6) / 2.0, 1.0, 0.0);
        assert!((mid.sensible_capacity_w - (s1 + s2) / 2.0).abs() < 1e-9);
    }

    #[test]
    fn shr_varies_with_entering_conditions() {
        let m = load();
        // Wet corner (high EWB): SHR well below 1.
        let wet = m.evaluate(CeTable::Net26a, 21.7, 35.0, 26.7, 1.0, 0.0);
        // Dry corner (low EWB, high ODB): SHR at or near 1 via the rule.
        let dry = m.evaluate(CeTable::Net26a, 15.0, 46.1, 26.7, 1.0, 0.0);
        assert!(wet.shr < 0.95, "wet corner SHR {}", wet.shr);
        assert!(
            dry.shr >= wet.shr,
            "SHR must rise as entering air dries ({} vs {})",
            dry.shr,
            wet.shr
        );
    }

    #[test]
    fn part_load_follows_energyplus_cycling_formulation() {
        let m = load();
        // GSR reference: PartLoadFrac = 0.771 + 0.229·PLR.
        let e = m.evaluate(CeTable::Net26a, 19.4, 35.0, 24.4, 0.05, 329.0);
        assert!((e.part_load_fraction - (0.771 + 0.229 * 0.05)).abs() < 1e-12);
        assert!((e.delivered_capacity_w - e.total_capacity_w * 0.05).abs() < 1e-9);
        assert!(
            (e.delivered_power_w - e.compressor_power_w * 0.05 / e.part_load_fraction).abs() < 1e-9
        );
        // At PLR = 1 the cycling loss is exactly 1.0 (no degradation).
        let full = m.evaluate(CeTable::Net26a, 19.4, 35.0, 24.4, 1.0, 329.0);
        assert!((full.part_load_fraction - 1.0).abs() < 1e-12);
        assert!((full.delivered_power_w - full.compressor_power_w).abs() < 1e-9);
    }

    #[test]
    fn clamping_at_grid_edges_is_documented() {
        let m = load();
        let edge = m.evaluate(CeTable::Net26a, 10.0, 50.0, 25.0, 1.0, 0.0);
        let corner = m.evaluate(CeTable::Net26a, 15.0, 46.1, 25.0, 1.0, 0.0);
        assert!((edge.total_capacity_w - corner.total_capacity_w).abs() < 1e-9);
        assert!((edge.compressor_power_w - corner.compressor_power_w).abs() < 1e-9);
    }

    #[test]
    fn out_of_grid_is_finite() {
        let m = load();
        let e = m.evaluate(CeTable::Gross26c, 18.0, 33.0, 24.0, 0.5, 230.0);
        assert!(e.total_capacity_w.is_finite());
        assert!(e.sensible_capacity_w.is_finite());
        assert!(e.shr.is_finite());
        assert!(e.delivered_power_w.is_finite());
    }
}
