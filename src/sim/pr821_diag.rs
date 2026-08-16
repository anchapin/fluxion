//! PR #821 — Free-Float diagnostic CSV writer.
//!
//! This module is compiled only when the `pr821-diag` cargo feature is enabled.
//! It exposes a tiny stateful logger that captures, once per simulation hour,
//! the variables relevant to the 600FF / 650FF peak-temperature investigation
//! and writes them to `target/diag/pr821_<case>.csv` after the run completes.
//!
//! The collector is intentionally allocation-light: it appends one row per
//! `record(...)` call and flushes everything in `flush_to_csv(...)`. Existing
//! code paths are not touched — the diagnostic helper is invoked only by the
//! `run_free_floating_simulation` test helper (or any caller that opts in).
//!
//! Usage:
//! ```ignore
//! #[cfg(feature = "pr821-diag")]
//! let mut diag = fluxion::sim::pr821_diag::DiagCollector::new("600FF");
//!
//! for step in 0..8760 {
//!     // ... step_physics ...
//!     #[cfg(feature = "pr821-diag")]
//!     diag.record(step, &model, outdoor_temp, solar_window_w);
//! }
//!
//! #[cfg(feature = "pr821-diag")]
//! diag.flush_to_csv().expect("write pr821 diagnostic CSV");
//! ```
//!
//! Acceptance for Phase 0 of the PR #821 plan:
//! `cargo test --test ashrae_140_case_600_series test_max_temperature \
//!     --features pr821-diag -- --test-threads=1 --nocapture`
//! must produce 8 760-row CSVs for both 600FF and 650FF in < 30 s.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;

/// One hourly snapshot of the 5R1C state for a free-float diagnostic run.
#[derive(Debug, Clone)]
pub struct DiagRow {
    pub t: usize,
    pub hour: u32,
    pub t_air: f64,
    pub t_mass: f64,
    pub t_surface: f64,
    pub cm: f64,
    pub h_tr_ms: f64,
    pub h_tr_is: f64,
    pub h_tr_em: f64,
    pub h_tr_floor: f64,
    pub t_ground: f64,
    pub t_outdoor: f64,
    pub solar_window_w: f64,
    pub phi_ia: f64,
    pub phi_st: f64,
    pub phi_m: f64,
    pub night_vent_active: u8,
    pub hvac_out_w: f64,
}

/// Diagnostic collector — accumulates rows and writes them to a CSV.
pub struct DiagCollector {
    case_id: String,
    rows: Vec<DiagRow>,
}

impl DiagCollector {
    /// Construct a collector for the given ASHRAE 140 case identifier.
    pub fn new(case_id: impl Into<String>) -> Self {
        Self {
            case_id: case_id.into(),
            rows: Vec::with_capacity(8_760),
        }
    }

    /// Capture the current state. `t_outdoor` and `solar_window_w` come from the
    /// simulation loop (the model itself does not retain them). `phi_*` are
    /// optional — pass `0.0` if unavailable.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        t: usize,
        model: &ThermalModel<VectorField>,
        t_outdoor: f64,
        solar_window_w: f64,
        phi_ia: f64,
        phi_st: f64,
        phi_m: f64,
        night_vent_active: bool,
        hvac_out_w: f64,
    ) {
        let zone_idx = 0;
        let t_air = model
            .setpoints
            .temperatures
            .as_slice()
            .get(zone_idx)
            .copied()
            .unwrap_or(f64::NAN);
        let t_mass = model
            .mass
            .mass_temperatures
            .as_slice()
            .get(zone_idx)
            .copied()
            .unwrap_or(f64::NAN);
        // The 5R1C model does not expose a separate T_surface; surface tracks
        // the implicit (h_is·T_air + h_ms·T_mass) / (h_is + h_ms) combination.
        // Reconstructed here for diagnostic clarity.
        let h_is = model
            .conduction
            .h_tr_is
            .as_slice()
            .get(zone_idx)
            .copied()
            .unwrap_or(0.0);
        let h_ms = model
            .conduction
            .h_tr_ms
            .as_slice()
            .get(zone_idx)
            .copied()
            .unwrap_or(0.0);
        let denom = h_is + h_ms;
        let t_surface = if denom > 0.0 {
            (h_is * t_air + h_ms * t_mass) / denom
        } else {
            t_air
        };

        let cm = model
            .mass
            .thermal_capacitance
            .as_slice()
            .get(zone_idx)
            .copied()
            .unwrap_or(0.0);
        let h_tr_em = model
            .conduction
            .h_tr_em
            .as_slice()
            .get(zone_idx)
            .copied()
            .unwrap_or(0.0);
        let h_tr_floor = model
            .conduction
            .h_tr_floor
            .as_slice()
            .get(zone_idx)
            .copied()
            .unwrap_or(0.0);

        let t_ground = model.conduction.ground_temperature.ground_temperature(t);

        self.rows.push(DiagRow {
            t,
            hour: (t % 24) as u32,
            t_air,
            t_mass,
            t_surface,
            cm,
            h_tr_ms: h_ms,
            h_tr_is: h_is,
            h_tr_em,
            h_tr_floor,
            t_ground,
            t_outdoor,
            solar_window_w,
            phi_ia,
            phi_st,
            phi_m,
            night_vent_active: u8::from(night_vent_active),
            hvac_out_w,
        });
    }

    /// Write accumulated rows to `target/diag/pr821_<case>.csv`.
    pub fn flush_to_csv(&self) -> io::Result<PathBuf> {
        let dir = PathBuf::from("target/diag");
        fs::create_dir_all(&dir)?;
        let path = dir.join(format!("pr821_{}.csv", self.case_id));
        let f = File::create(&path)?;
        let mut w = BufWriter::new(f);
        writeln!(
            w,
            "t,hour,T_air,T_mass,T_surface,Cm,h_tr_ms,h_tr_is,h_tr_em,h_tr_floor,\
             T_ground,T_outdoor,solar_window_W,phi_ia,phi_st,phi_m,night_vent_active,hvac_out_W"
        )?;
        for r in &self.rows {
            writeln!(
                w,
                "{},{},{:.4},{:.4},{:.4},{:.3e},{:.4},{:.4},{:.4},{:.4},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.3}",
                r.t,
                r.hour,
                r.t_air,
                r.t_mass,
                r.t_surface,
                r.cm,
                r.h_tr_ms,
                r.h_tr_is,
                r.h_tr_em,
                r.h_tr_floor,
                r.t_ground,
                r.t_outdoor,
                r.solar_window_w,
                r.phi_ia,
                r.phi_st,
                r.phi_m,
                r.night_vent_active,
                r.hvac_out_w
            )?;
        }
        w.flush()?;
        Ok(path)
    }

    /// Borrow the accumulated diagnostic rows. Used by the 600/650FF
    /// regression assertion in `tests/ashrae_140_case_600_series.rs` to verify
    /// the heat-balance phi_* terms are non-zero when expected (Issue #825).
    pub fn rows(&self) -> &[DiagRow] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}
