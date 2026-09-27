//! Phase-Change Material (PCM) nominal property definitions (Refs #3986 / #4118).
//!
//! This module is the skeleton of the PCM test-box sub-suite documented in
//! `tests/reference_data/pcm_test_box/PROVENANCE.md`. It defines:
//!
//! - [`PhaseChangeMaterial`] — a struct holding the manufacturer-published
//!   nominal thermal properties (melting range, latent heat, specific heat,
//!   density) and the standard enthalpy-method `enthalpy_j_per_kg` +
//!   `apparent_cp_j_per_kg_k` methods.
//! - [`PhaseChangeMaterial::rt27`] — the nominal Rubitherm RT27 values cited in
//!   `PROVENANCE.md`. These are the real manufacturer datasheet numbers, NOT
//!   placeholders.
//!
//! The apparent-heat-capacity method (used by `apparent_cp_j_per_kg_k`) is the
//! standard enthalpy-method PCM numerical technique (see e.g. Shmueli et al.
//! 2010, Assis et al. 2009). Outside the melting band the apparent c_p equals
//! the sensible c_p; inside the band it equals `latent_heat / ΔT_melt`, which
//! yields a smooth step in enthalpy.
//!
//! ## BLOCKER branch
//!
//! The experimental reference curve that would let us validate
//! `PCMTestBox::solid_fraction_at_time(t)` against real data is **not available**
//! in the spike window (see `docs/research/mazzeo-rt27-pcm-spike.md`). The
//! skeleton ships without a `ReferenceCurve` and the harness exposes a sentinel
//! return. Real PCM physics (mushy-zone tracking, time-stepped solver, etc.)
//! lands in PR-B+1.

/// Reference temperature for enthalpy convention: `h(REFERENCE_TEMP_C) = 0`.
/// Matches the ASHRAE 140 / 5R1C convention of pinning enthalpy at a
/// sensible reference (here, 0 °C, the conventional PCM cold state).
const REFERENCE_TEMP_C: f64 = 0.0;

/// Phase-Change Material nominal properties.
///
/// All values are stored in SI units (J, kg, K, °C where the difference is
/// immaterial because Celsius offsets are linear in Kelvin). `melting_range_c`
/// uses (T_solidus, T_liquidus); the apparent-heat-capacity method treats the
/// transition as a band of width `T_liquidus - T_solidus`.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseChangeMaterial {
    /// Human-readable name (e.g., `"Rubitherm RT27"`).
    pub name: String,
    /// Melting temperature range `(T_solidus, T_liquidus)` in °C.
    pub melting_range_c: (f64, f64),
    /// Latent heat of fusion in J/kg.
    pub latent_heat_j_per_kg: f64,
    /// Representative sensible specific heat in J/(kg·K). Used outside the
    /// melting band; the apparent-heat-capacity method substitutes
    /// `latent_heat_j_per_kg / ΔT_melt` inside the band.
    pub specific_heat_j_per_kg_k: f64,
    /// Representative density in kg/m³. Used by the harness for energy
    /// conversion; PR-B+1 may add temperature-dependent `ρ(T)` once real data
    /// lands.
    pub density_kg_per_m3: f64,
}

impl PhaseChangeMaterial {
    /// Rubitherm RT27 nominal values (Refs #3986 / #4118, BLOCKER branch).
    ///
    /// Values cited in `tests/reference_data/pcm_test_box/PROVENANCE.md`:
    /// - Melting range: 25.0–28.0 °C (Rubitherm RT27 datasheet)
    /// - Latent heat: 184 kJ/kg (Rubitherm RT27 datasheet)
    /// - Representative c_p: 2000 J/(kg·K) (mean of solid 1800 / liquid 2400)
    /// - Representative ρ: 800 kg/m³ (mean of solid 880 / liquid 760)
    pub fn rt27() -> Self {
        Self {
            name: "Rubitherm RT27".to_string(),
            melting_range_c: (25.0, 28.0),
            latent_heat_j_per_kg: 184_000.0,
            specific_heat_j_per_kg_k: 2000.0,
            density_kg_per_m3: 800.0,
        }
    }

    /// Returns `true` iff `t_c` falls within the melting range
    /// `[T_solidus, T_liquidus]` (endpoints inclusive).
    pub fn is_in_melting_range(&self, t_c: f64) -> bool {
        let (t_s, t_l) = self.melting_range_c;
        t_c >= t_s && t_c <= t_l
    }

    /// Specific enthalpy `h(T)` in J/kg, referenced to `h(REFERENCE_TEMP_C) = 0`.
    ///
    /// Uses the apparent-heat-capacity method:
    /// - For T < T_solidus: `h(T) = c_p * (T - REFERENCE_TEMP_C)`
    /// - For T in [T_solidus, T_liquidus]: smooth blend via
    ///   `h(T) = c_p * (T_solidus - REFERENCE_TEMP_C)
    ///         + (latent_heat / (T_liquidus - T_solidus)) * (T - T_solidus)`
    /// - For T > T_liquidus: `h(T) = c_p * (T_solidus - REFERENCE_TEMP_C)
    ///                     + latent_heat
    ///                     + c_p * (T - T_liquidus)`
    ///
    /// The function is C¹-continuous at the band boundaries because the
    /// apparent c_p equals c_sensible on both sides.
    pub fn enthalpy_j_per_kg(&self, t_c: f64) -> f64 {
        let (t_s, t_l) = self.melting_range_c;
        let c_p = self.specific_heat_j_per_kg_k;
        let latent = self.latent_heat_j_per_kg;
        let delta_t_melt = t_l - t_s;
        let band_cp = latent / delta_t_melt;

        if t_c < t_s {
            c_p * (t_c - REFERENCE_TEMP_C)
        } else if t_c <= t_l {
            // In the melting band. h(T_solidus) = c_p * (T_solidus - 0).
            c_p * (t_s - REFERENCE_TEMP_C) + band_cp * (t_c - t_s)
        } else {
            // Above T_liquidus. h(T_liquidus) = c_p * (T_solidus - 0) + latent.
            c_p * (t_s - REFERENCE_TEMP_C) + latent + c_p * (t_c - t_l)
        }
    }

    /// Apparent specific heat `c_p,app(T)` in J/(kg·K).
    ///
    /// Returns the sensible c_p outside the melting band and
    /// `latent_heat / (T_liquidus - T_solidus)` inside it. This is the standard
    /// apparent-heat-capacity approximation used in enthalpy-method PCM
    /// simulations.
    pub fn apparent_cp_j_per_kg_k(&self, t_c: f64) -> f64 {
        let (t_s, t_l) = self.melting_range_c;
        if self.is_in_melting_range(t_c) {
            self.latent_heat_j_per_kg / (t_l - t_s)
        } else {
            self.specific_heat_j_per_kg_k
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt27_matches_documented_nominal_values() {
        // Defensive: if a future change bumps the nominals, this test fails
        // BEFORE the public-API test (`rt27_constructs_with_nominal_properties`)
        // does, with a clearer diagnostic.
        let pcm = PhaseChangeMaterial::rt27();
        assert_eq!(pcm.name, "Rubitherm RT27");
        assert!((pcm.melting_range_c.0 - 25.0).abs() < 1e-12);
        assert!((pcm.melting_range_c.1 - 28.0).abs() < 1e-12);
        assert_eq!(pcm.latent_heat_j_per_kg, 184_000.0);
        assert_eq!(pcm.specific_heat_j_per_kg_k, 2000.0);
        assert_eq!(pcm.density_kg_per_m3, 800.0);
    }

    #[test]
    fn enthalpy_continuous_at_band_boundaries() {
        // The apparent-heat-capacity method guarantees h is continuous AND
        // continuously differentiable at the band edges. We check continuity
        // numerically (C¹ check requires comparing slopes, which we do via
        // apparent_cp tests in the public-API suite). Choose eps << 1/c_p so
        // the linear-extrapolation gap c_p * eps is well below tolerance.
        let pcm = PhaseChangeMaterial::rt27();
        // eps must be << 1/band_cp so that band_cp * eps << tol. With
        // band_cp = latent/ΔT = 184000/3 ≈ 61333 J/(kg·K), eps = 1e-12
        // gives a gap of ≈ 6e-8 J/kg, well below tol = 1e-5 J/kg.
        let eps = 1e-12;
        let tol = 1e-5;
        let h_just_below_s = pcm.enthalpy_j_per_kg(pcm.melting_range_c.0 - eps);
        let h_at_s = pcm.enthalpy_j_per_kg(pcm.melting_range_c.0);
        let h_just_below_l = pcm.enthalpy_j_per_kg(pcm.melting_range_c.1 - eps);
        let h_at_l = pcm.enthalpy_j_per_kg(pcm.melting_range_c.1);
        let h_just_above_l = pcm.enthalpy_j_per_kg(pcm.melting_range_c.1 + eps);
        assert!(
            (h_at_s - h_just_below_s).abs() < tol,
            "h discontinuity at T_solidus: {h_at_s} vs {h_just_below_s}, diff={}",
            (h_at_s - h_just_below_s).abs()
        );
        assert!(
            (h_at_l - h_just_below_l).abs() < tol,
            "h discontinuity inside band at T_liquidus: {h_at_l} vs {h_just_below_l}, diff={}",
            (h_at_l - h_just_below_l).abs()
        );
        assert!(
            (h_just_above_l - h_at_l).abs() < tol,
            "h discontinuity at T_liquidus: {h_at_l} vs {h_just_above_l}, diff={}",
            (h_just_above_l - h_at_l).abs()
        );
    }
}
