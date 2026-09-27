//! PCM (Phase-Change Material) test-box harness (Refs #3986 / #4118).
//!
//! This module is the skeleton of the PCM test-box sub-suite documented in
//! `tests/reference_data/pcm_test_box/PROVENANCE.md`. It defines:
//!
//! - [`PCMTestBox`] — a one-zone, one-PCM-layer rectangular test box with an
//!   isothermal wall (the geometry referenced in
//!   `docs/research/mazzeo-rt27-pcm-spike.md`).
//! - [`PCMTestBox::solid_fraction_at_time`] — the sentinel return documented in
//!   the BLOCKER branch of `PROVENANCE.md`.
//! - [`PCMTestBox::apparent_cp_at_wall`] — a thin pass-through to
//!   [`PhaseChangeMaterial::apparent_cp_j_per_kg_k`].
//!
//! ## BLOCKER branch
//!
//! The experimental reference curve that would let us validate
//! `solid_fraction_at_time(t)` against real data is **not available** in the
//! spike window. The skeleton ships with `reference_curve = None` and
//! `solid_fraction_at_time(t)` returns `None`. PR-B+1 unlocks this once a
//! licensable RT27 curve is acquired (see `PROVENANCE.md` §"Unlock criteria").

use crate::physics::phase_change_material::PhaseChangeMaterial;

/// Default PCM layer thickness in metres. 5 cm is the canonical reference
/// value for the rectangular test-box geometry (Kraiem 2019; matches the
/// Mazzeo group setup referenced in `docs/research/mazzeo-rt27-pcm-spike.md`).
pub const DEFAULT_LAYER_THICKNESS_M: f64 = 0.05;

/// Default wall temperature in °C. 40 °C is well above the RT27 liquidus
/// (28 °C), so the apparent-heat-capacity method returns the sensible c_p
/// (i.e., the test-box sits in the fully-liquid regime by default). This
/// matches the standard `isothermal wall above T_liquidus` setup used to
/// measure solid-front propagation rates.
pub const DEFAULT_WALL_TEMP_C: f64 = 40.0;

/// PCM test-box configuration.
///
/// Holds a [`PhaseChangeMaterial`], the layer geometry, and the wall boundary
/// condition. `reference_curve` is `None` in the skeleton (BLOCKER branch);
/// PR-B+1 fills it in with the experimental solid-fraction-vs-time curve.
#[derive(Debug, Clone, PartialEq)]
pub struct PCMTestBox {
    /// The PCM filling the test box.
    pub pcm: PhaseChangeMaterial,
    /// PCM layer thickness in metres.
    pub layer_thickness_m: f64,
    /// Isothermal wall temperature in °C.
    pub wall_temperature_c: f64,
    /// Optional experimental reference curve (solid fraction vs. time).
    /// `None` in the skeleton; PR-B+1 fills this in.
    pub reference_curve: Option<ReferenceCurve>,
}

/// Stub type for the experimental reference curve. PR-B+1 expands this to
/// hold the tabulated (time_s, solid_fraction) pairs loaded from
/// `tests/reference_data/pcm_test_box/rt27_solid_fraction.csv`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceCurve {
    /// Marker field so the type is non-trivial until PR-B+1. Future fields
    /// will hold the tabulated curve and the citation string.
    _marker: (),
}

impl PCMTestBox {
    /// Construct a test box with the default layer thickness (5 cm) and wall
    /// temperature (40 °C). The reference curve is `None` (BLOCKER branch).
    pub fn new(pcm: PhaseChangeMaterial) -> Self {
        Self {
            pcm,
            layer_thickness_m: DEFAULT_LAYER_THICKNESS_M,
            wall_temperature_c: DEFAULT_WALL_TEMP_C,
            reference_curve: None,
        }
    }

    /// Builder: override the PCM layer thickness in metres.
    #[must_use]
    pub fn with_layer_thickness(mut self, thickness_m: f64) -> Self {
        self.layer_thickness_m = thickness_m;
        self
    }

    /// Builder: override the isothermal wall temperature in °C.
    #[must_use]
    pub fn with_wall_temperature(mut self, wall_temp_c: f64) -> Self {
        self.wall_temperature_c = wall_temp_c;
        self
    }

    /// Solid fraction at time `t_s` (seconds from t = 0).
    ///
    /// Returns `None` in the skeleton (BLOCKER branch:
    /// `tests/reference_data/pcm_test_box/PROVENANCE.md`). PR-B+1 replaces
    /// this with a real curve lookup returning `Some(solid_fraction)` and
    /// asserting ±2% absolute match against the loaded reference data.
    pub fn solid_fraction_at_time(&self, _t_s: f64) -> Option<f64> {
        // Skeleton sentinel: no reference curve loaded yet.
        // PR-B+1 will return `Some(self.reference_curve.as_ref()?.interpolate(t_s))`.
        None
    }

    /// Apparent specific heat at the wall temperature, J/(kg·K).
    /// Delegates to [`PhaseChangeMaterial::apparent_cp_j_per_kg_k`].
    pub fn apparent_cp_at_wall(&self) -> f64 {
        self.pcm.apparent_cp_j_per_kg_k(self.wall_temperature_c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_layer_thickness_matches_documented_value() {
        // Defensive: if a future change bumps the default, this test fails
        // BEFORE the public-API test does, with a clearer diagnostic.
        assert!((DEFAULT_LAYER_THICKNESS_M - 0.05).abs() < 1e-12);
        assert!((DEFAULT_WALL_TEMP_C - 40.0).abs() < 1e-12);
    }

    #[test]
    fn builders_update_fields() {
        let box_ = PCMTestBox::new(PhaseChangeMaterial::rt27())
            .with_layer_thickness(0.10)
            .with_wall_temperature(30.0);
        assert_eq!(box_.layer_thickness_m, 0.10);
        assert_eq!(box_.wall_temperature_c, 30.0);
        // Wall at 30 °C is in the melting band (25–28 °C)? No — 30 > 28,
        // so it's in the liquid region. But we only check the builder here.
        assert!(box_.reference_curve.is_none());
    }
}
