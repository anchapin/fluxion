//! Cooling coil model with bypass-factor psychrometrics (Issue #1762).
//!
//! Implements the second core airside component after T2.1 (psychrometrics,
//! #1760). The coil removes sensible **and** latent load from moist air using
//! the ASHRAE bypass-factor method, tracking condensate removal.
//!
//! ## Physical model
//!
//! A fraction of the supply air bypasses the coil surface unchanged. The
//! remainder reaches saturation at the **apparatus dew point** (ADP — the
//! effective saturated coil-surface temperature). The leaving-air state lies on
//! the straight line connecting the entering-air and ADP points on the
//! psychrometric chart:
//!
//! ```text
//! T_db,out = T_adp + BF · (T_db,in − T_adp)
//! W_out    = W_adp + BF · (W_in    − W_adp)
//! h_out    = h_adp + BF · (h_in    − h_adp)
//! ```
//!
//! Capacities follow from the ASHRAE Handbook of Fundamentals (2021), Ch.1:
//!
//! ```text
//! q_total    = ṁ_da · (h_in − h_out)                  [kW → ×1000 for W]
//! q_sensible = ṁ_da · c_p,ma · (T_db,in − T_db,out)
//! q_latent   = q_total − q_sensible
//! SHR        = q_sensible / q_total
//! ṁ_cond     = ṁ_da · (W_in − W_out)                   [kg/s]
//! ```
//!
//! where `c_p,ma = 1.006 + 1.86·W` [kJ/(kg·K)].
//!
//! ## Part-load behaviour (stub)
//!
//! Part-load is modelled as time-averaged cycling: the on-cycle air-side
//! performance is computed at full coil effectiveness, and the **delivered**
//! capacity is scaled by the part-load ratio (PLR). This is a stub for the
//! future polynomial part-load curve integration (cf. `efficiency_curves.rs`).

use crate::sim::hvac::airside_state::{AirsideCouplingError, MoistAirState};
use fluxion_core::weather::psychrometrics::{calculate_enthalpy, calculate_humidity_ratio};
use serde::{Deserialize, Serialize};

/// Specific heat of dry air, kJ/(kg·K) — ASHRAE HoF Ch.1.
const CP_DRY_AIR_KJ_PER_KG_K: f64 = 1.006;
/// Specific heat of water vapor, kJ/(kg·K) — ASHRAE HoF Ch.1.
const CP_WATER_VAPOR_KJ_PER_KG_K: f64 = 1.86;

/// Result of a cooling-coil performance calculation.
///
/// All capacity values are in Watts and are positive when the coil is removing
/// heat from the air. `leaving_air` is the **on-cycle** supply-air state
/// (thermodynamic state during active cooling, before time-averaging by PLR).
#[derive(Debug, Clone, PartialEq)]
pub struct CoilPerformance {
    /// Total cooling capacity (sensible + latent) at the given conditions [W].
    pub total_capacity_w: f64,
    /// Sensible cooling capacity [W].
    pub sensible_capacity_w: f64,
    /// Latent cooling capacity [W].
    pub latent_capacity_w: f64,
    /// Sensible heat ratio (sensible / total), in [0, 1].
    pub shr: f64,
    /// Leaving (supply) moist-air state at full coil effectiveness.
    pub leaving_air: MoistAirState,
    /// Condensate removal rate [kg/s].
    pub condensate_rate_kg_per_s: f64,
}

/// Trait for cooling-coil behaviour (Issue #1762).
///
/// Defines the contract for any cooling-coil implementation so that the
/// airside coupling layer (`AirsideEnvelopeCoupler`, #1767) can swap in
/// alternative coil models (e.g. detailed NTU-ε or data-driven surrogates)
/// without changing the calling code.
pub trait CoolingCoilBehavior: Send + Sync {
    /// Compute full-load air-side performance for the given entering-air state.
    ///
    /// Returns the total, sensible, and latent capacity, the sensible heat
    /// ratio, the leaving-air state, and the condensate removal rate.
    ///
    /// # Arguments
    /// * `inlet` - Entering (return) moist-air state
    /// * `mass_flow_da_kg_per_s` - Dry-air mass flow rate [kg/s]
    fn compute_cooling_capacity(
        &self,
        inlet: &MoistAirState,
        mass_flow_da_kg_per_s: f64,
    ) -> Result<CoilPerformance, AirsideCouplingError>;

    /// Compute the sensible heat ratio (SHR) for the given conditions.
    ///
    /// Returns `sensible / total` in [0, 1]. Returns 0.0 when there is no
    /// cooling (inlet at or below apparatus dew point).
    fn compute_shr(&self, inlet: &MoistAirState, mass_flow_da_kg_per_s: f64) -> f64;

    /// Condensate removal rate [kg/s] for the given conditions.
    fn condensate_rate(&self, inlet: &MoistAirState, mass_flow_da_kg_per_s: f64) -> f64;

    /// Rated total cooling capacity at design/AHRI conditions [W].
    fn rated_total_capacity(&self) -> f64;

    /// Rated sensible cooling capacity at design/AHRI conditions [W].
    fn rated_sensible_capacity(&self) -> f64;

    /// Bypass factor (fraction of air not contacting the coil surface).
    fn bypass_factor(&self) -> f64;

    /// Current part-load ratio (0.0 to 1.0).
    fn current_plr(&self) -> f64;

    /// Delivered (time-averaged) total capacity at the current part-load ratio
    /// and given entering-air conditions [W].
    ///
    /// This applies the part-load stub: `full_capacity × PLR`, clamped to the
    /// rated capacity.
    fn delivered_capacity(&self, inlet: &MoistAirState, mass_flow_da_kg_per_s: f64) -> f64;
}

/// Cooling coil model using the ASHRAE bypass-factor method (Issue #1762).
///
/// Key characteristics:
/// - Removes **sensible and latent** load using psychrometrics from T2.1 (#1760)
/// - Bypass-factor model determines leaving-air state from apparatus dew point
/// - Part-load behaviour stub: capacity scales linearly with PLR (cycling model)
/// - Condensate removal tracked from humidity-ratio change across the coil
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolingCoil {
    /// Equipment identifier.
    pub id: String,
    /// Rated total cooling capacity at AHRI/design conditions [W].
    pub rated_total_capacity: f64,
    /// Rated sensible heat ratio at AHRI/design conditions (sensible / total).
    pub rated_shr: f64,
    /// Bypass factor — fraction of air that passes the coil unchanged (0.0–1.0).
    ///
    /// Physically related to the coil NTU by `BF = exp(−NTU)`. Typical values
    /// range from 0.05 (deep, many rows) to 0.30 (shallow coil).
    pub bypass_factor: f64,
    /// Apparatus dew point temperature — saturated coil-surface temp [°C].
    ///
    /// Typically 2–6 °C below the chilled-water entering temperature.
    pub apparatus_dew_point: f64,
    /// Design dry-air mass flow rate [kg/s].
    pub design_mass_flow: f64,
    /// Atmospheric pressure [Pa].
    pub pressure: f64,
    /// Overall heat-transfer coefficient (UA per face area) [W/(m²·K)].
    ///
    /// Characterises the coil's heat-transfer surface; combined with `face_area`
    /// gives `UA = u_value × face_area` [W/K].
    pub u_value: f64,
    /// Coil face area (frontal area normal to airflow) [m²].
    pub face_area: f64,
    /// Current part-load ratio (0.0 to 1.0).
    pub current_plr: f64,
}

impl CoolingCoil {
    /// Create a new cooling coil with the given design parameters.
    ///
    /// # Arguments
    /// * `id` - Equipment identifier
    /// * `rated_total_capacity` - Rated total capacity [W]
    /// * `rated_shr` - Rated sensible heat ratio (0.0–1.0)
    /// * `bypass_factor` - Bypass factor (0.0–1.0, exclusive of 1.0)
    /// * `apparatus_dew_point` - Apparatus dew point [°C]
    /// * `design_mass_flow` - Design dry-air mass flow [kg/s]
    pub fn new(
        id: String,
        rated_total_capacity: f64,
        rated_shr: f64,
        bypass_factor: f64,
        apparatus_dew_point: f64,
        design_mass_flow: f64,
    ) -> Self {
        Self {
            id,
            rated_total_capacity,
            rated_shr: rated_shr.clamp(0.0, 1.0),
            bypass_factor: bypass_factor.clamp(0.0, 0.999),
            apparatus_dew_point,
            design_mass_flow,
            pressure: 101_325.0,
            u_value: 50.0,
            face_area: 1.0,
            current_plr: 0.0,
        }
    }

    /// Rated sensible capacity derived from rated total capacity and SHR [W].
    pub fn rated_sensible(&self) -> f64 {
        self.rated_total_capacity * self.rated_shr
    }

    /// Rated latent capacity derived from rated total capacity and SHR [W].
    pub fn rated_latent(&self) -> f64 {
        self.rated_total_capacity * (1.0 - self.rated_shr)
    }

    /// Coil NTU (Number of Transfer Units) for a given mass flow.
    ///
    /// `NTU = UA / (ṁ_da · c_p,ma)`. Related to the bypass factor by
    /// `BF = exp(−NTU)` for a dry coil.
    pub fn ntu(&self, mass_flow_da_kg_per_s: f64) -> f64 {
        if mass_flow_da_kg_per_s <= 0.0 {
            return f64::INFINITY;
        }
        let ua = self.u_value * self.face_area; // W/K
        let cp_j_per_kg_k = CP_DRY_AIR_KJ_PER_KG_K * 1000.0;
        ua / (mass_flow_da_kg_per_s * cp_j_per_kg_k)
    }

    /// Set the current part-load ratio.
    pub fn set_plr(&mut self, plr: f64) {
        self.current_plr = plr.clamp(0.0, 1.0);
    }

    /// Leaving-air dry-bulb temperature for given inlet [°C].
    fn leaving_db(&self, inlet_db: f64) -> f64 {
        self.apparatus_dew_point + self.bypass_factor * (inlet_db - self.apparatus_dew_point)
    }

    /// Humidity ratio at the apparatus dew point (saturated) [kg/kg].
    fn w_adp(&self) -> f64 {
        calculate_humidity_ratio(self.apparatus_dew_point, 100.0, self.pressure)
    }

    /// Enthalpy at the apparatus dew point (saturated) [kJ/kg].
    fn h_adp(&self) -> f64 {
        calculate_enthalpy(self.apparatus_dew_point, 100.0, self.pressure)
    }
}

impl CoolingCoilBehavior for CoolingCoil {
    fn compute_cooling_capacity(
        &self,
        inlet: &MoistAirState,
        mass_flow_da_kg_per_s: f64,
    ) -> Result<CoilPerformance, AirsideCouplingError> {
        // No cooling if no airflow or inlet already at/below the ADP.
        if mass_flow_da_kg_per_s <= 0.0 || inlet.dry_bulb_c <= self.apparatus_dew_point {
            return Ok(zero_performance(*inlet));
        }

        let w_in = inlet.humidity_ratio_kg_per_kg_dry_air;
        let h_in = inlet.enthalpy_kj_per_kg_dry_air;
        let w_adp = self.w_adp();
        let h_adp = self.h_adp();

        // Bypass-factor interpolation toward the ADP state.
        let t_out = self.leaving_db(inlet.dry_bulb_c);
        let w_out = w_adp + self.bypass_factor * (w_in - w_adp);
        let h_out = h_adp + self.bypass_factor * (h_in - h_adp);

        let leaving_air = MoistAirState::from_humidity_ratio(t_out, w_out, self.pressure)?;

        // Capacities — enthalpy is kJ/kg, multiply by 1000 for Watts.
        let cp_ma = CP_DRY_AIR_KJ_PER_KG_K + CP_WATER_VAPOR_KJ_PER_KG_K * w_in;
        let total = mass_flow_da_kg_per_s * (h_in - h_out) * 1000.0;
        let sensible = mass_flow_da_kg_per_s * cp_ma * (inlet.dry_bulb_c - t_out) * 1000.0;
        let latent = (total - sensible).max(0.0);
        let shr = if total > 0.0 { sensible / total } else { 0.0 };
        let condensate = mass_flow_da_kg_per_s * (w_in - w_out).max(0.0);

        Ok(CoilPerformance {
            total_capacity_w: total,
            sensible_capacity_w: sensible,
            latent_capacity_w: latent,
            shr: shr.clamp(0.0, 1.0),
            leaving_air,
            condensate_rate_kg_per_s: condensate.max(0.0),
        })
    }

    fn compute_shr(&self, inlet: &MoistAirState, mass_flow_da_kg_per_s: f64) -> f64 {
        if mass_flow_da_kg_per_s <= 0.0 || inlet.dry_bulb_c <= self.apparatus_dew_point {
            return 0.0;
        }
        let w_in = inlet.humidity_ratio_kg_per_kg_dry_air;
        let h_in = inlet.enthalpy_kj_per_kg_dry_air;
        let h_adp = self.h_adp();
        let h_out = h_adp + self.bypass_factor * (h_in - h_adp);
        let t_out = self.leaving_db(inlet.dry_bulb_c);
        let cp_ma = CP_DRY_AIR_KJ_PER_KG_K + CP_WATER_VAPOR_KJ_PER_KG_K * w_in;
        let total = (h_in - h_out) * 1000.0;
        if total <= 0.0 {
            return 0.0;
        }
        let sensible = cp_ma * (inlet.dry_bulb_c - t_out) * 1000.0;
        (sensible / total).clamp(0.0, 1.0)
    }

    fn condensate_rate(&self, inlet: &MoistAirState, mass_flow_da_kg_per_s: f64) -> f64 {
        if mass_flow_da_kg_per_s <= 0.0 || inlet.dry_bulb_c <= self.apparatus_dew_point {
            return 0.0;
        }
        let w_adp = self.w_adp();
        let w_out = w_adp + self.bypass_factor * (inlet.humidity_ratio_kg_per_kg_dry_air - w_adp);
        mass_flow_da_kg_per_s * (inlet.humidity_ratio_kg_per_kg_dry_air - w_out).max(0.0)
    }

    fn rated_total_capacity(&self) -> f64 {
        self.rated_total_capacity
    }

    fn rated_sensible_capacity(&self) -> f64 {
        self.rated_sensible()
    }

    fn bypass_factor(&self) -> f64 {
        self.bypass_factor
    }

    fn current_plr(&self) -> f64 {
        self.current_plr
    }

    fn delivered_capacity(&self, inlet: &MoistAirState, mass_flow_da_kg_per_s: f64) -> f64 {
        let full = self
            .compute_cooling_capacity(inlet, mass_flow_da_kg_per_s)
            .map(|p| p.total_capacity_w.min(self.rated_total_capacity))
            .unwrap_or(0.0);
        full * self.current_plr
    }
}

/// Zero-capacity result (coil off or no cooling needed), leaving air = inlet.
fn zero_performance(inlet: MoistAirState) -> CoilPerformance {
    CoilPerformance {
        total_capacity_w: 0.0,
        sensible_capacity_w: 0.0,
        latent_capacity_w: 0.0,
        shr: 0.0,
        leaving_air: inlet,
        condensate_rate_kg_per_s: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a standard test coil: 30 kW total, SHR 0.75, BF 0.15, ADP 10 °C.
    fn test_coil() -> CoolingCoil {
        CoolingCoil::new(
            "CC-1".to_string(),
            30_000.0, // 30 kW rated total
            0.75,     // rated SHR
            0.15,     // bypass factor
            10.0,     // ADP 10 °C
            1.0,      // 1 kg/s design mass flow
        )
    }

    fn inlet_30c_50rh() -> MoistAirState {
        MoistAirState::try_new(30.0, 50.0, 101_325.0).unwrap()
    }

    // ---- Structural / accessor tests --------------------------------------

    #[test]
    fn test_constructor_clamps_bf_and_shr() {
        let coil = CoolingCoil::new(
            "CC".to_string(),
            10_000.0,
            1.5, // invalid SHR > 1
            1.5, // invalid BF >= 1
            10.0,
            1.0,
        );
        assert!((coil.bypass_factor - 0.999).abs() < 1e-9);
        assert!((coil.rated_shr - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_rated_sensible_latent() {
        let coil = test_coil();
        assert!((coil.rated_sensible() - 22_500.0).abs() < 1e-6); // 30000 * 0.75
        assert!((coil.rated_latent() - 7_500.0).abs() < 1e-6); // 30000 * 0.25
        assert!((coil.rated_sensible_capacity() - 22_500.0).abs() < 1e-6);
    }

    #[test]
    fn test_plr_setter_clamps() {
        let mut coil = test_coil();
        coil.set_plr(0.5);
        assert!((coil.current_plr() - 0.5).abs() < 1e-9);
        coil.set_plr(-1.0);
        assert!((coil.current_plr() - 0.0).abs() < 1e-9);
        coil.set_plr(2.0);
        assert!((coil.current_plr() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ntu_finite_and_positive() {
        let coil = test_coil();
        let ntu = coil.ntu(1.0);
        assert!(ntu.is_finite());
        assert!(ntu > 0.0);
        // NTU = UA / (ṁ·cp). With defaults u_value=50, face_area=1, cp=1006:
        // NTU = 50 / 1006 ≈ 0.0497.
        assert!((ntu - 50.0 / 1006.0).abs() < 1e-6);
        // NTU increases when flow decreases (less heat capacity rate).
        assert!(coil.ntu(0.5) > ntu);
        assert!(coil.ntu(0.0).is_infinite()); // zero flow → infinite NTU
    }

    // ---- Bypass-factor / capacity tests (ASHRAE-validated) ----------------

    /// Core acceptance test: sensible + latent capacity validated against an
    /// ASHRAE-style bypass-factor example coil.
    ///
    /// Inlet: 30 °C DB, 50 % RH; ADP 10 °C; BF 0.15; 1 kg/s dry air.
    /// Reference values computed independently from ASHRAE HoF (2021) Ch.1:
    ///   q_total ≈ 29 668 W, q_sensible ≈ 17 523 W, SHR ≈ 0.591, leaving ≈ 91 % RH.
    #[test]
    fn test_ashrae_bypass_factor_capacity() {
        let coil = test_coil();
        let inlet = inlet_30c_50rh();
        let perf = coil.compute_cooling_capacity(&inlet, 1.0).unwrap();

        // Total capacity within 1% of reference (29 668 W).
        assert!(
            (perf.total_capacity_w - 29_668.0).abs() / 29_668.0 < 0.01,
            "total {} vs 29668",
            perf.total_capacity_w
        );

        // Sensible capacity within 1% of reference (17 523 W).
        assert!(
            (perf.sensible_capacity_w - 17_523.0).abs() / 17_523.0 < 0.01,
            "sensible {} vs 17523",
            perf.sensible_capacity_w
        );

        // SHR within 1% of reference (0.591).
        assert!((perf.shr - 0.591).abs() < 0.01, "SHR {} vs 0.591", perf.shr);

        // Energy balance: sensible + latent == total.
        assert!(
            (perf.sensible_capacity_w + perf.latent_capacity_w - perf.total_capacity_w).abs() < 1.0,
            "sensible + latent != total"
        );

        // Leaving air is near-saturated but subsaturated (typical cooling coil).
        assert!(
            perf.leaving_air.relative_humidity_percent > 85.0
                && perf.leaving_air.relative_humidity_percent <= 100.0,
            "leaving RH {} not in (85, 100]",
            perf.leaving_air.relative_humidity_percent
        );

        // Leaving dry-bulb: 10 + 0.15*(30-10) = 13.0 °C exactly.
        assert!(
            (perf.leaving_air.dry_bulb_c - 13.0).abs() < 0.01,
            "leaving DB {} vs 13.0",
            perf.leaving_air.dry_bulb_c
        );
    }

    #[test]
    fn test_condensate_tracked() {
        let coil = test_coil();
        let inlet = inlet_30c_50rh();
        let perf = coil.compute_cooling_capacity(&inlet, 1.0).unwrap();

        // Condensate ≈ 0.00482 kg/s (4.82 g/s) from humidity-ratio change.
        assert!(
            perf.condensate_rate_kg_per_s > 0.0,
            "condensate should be positive"
        );
        assert!(
            (perf.condensate_rate_kg_per_s - 0.00482).abs() / 0.00482 < 0.02,
            "condensate {} vs 0.00482",
            perf.condensate_rate_kg_per_s
        );

        // Direct method should match the bundled result.
        let direct = coil.condensate_rate(&inlet, 1.0);
        assert!((direct - perf.condensate_rate_kg_per_s).abs() < 1e-9);
    }

    #[test]
    fn test_shr_standalone_matches_bundled() {
        let coil = test_coil();
        let inlet = inlet_30c_50rh();
        let perf = coil.compute_cooling_capacity(&inlet, 1.0).unwrap();
        let shr = coil.compute_shr(&inlet, 1.0);
        assert!((shr - perf.shr).abs() < 1e-9);
    }

    #[test]
    fn test_capacity_scales_with_mass_flow() {
        let coil = test_coil();
        let inlet = inlet_30c_50rh();
        let q1 = coil
            .compute_cooling_capacity(&inlet, 1.0)
            .unwrap()
            .total_capacity_w;
        let q2 = coil
            .compute_cooling_capacity(&inlet, 2.0)
            .unwrap()
            .total_capacity_w;
        // Doubling mass flow doubles capacity (SHR unchanged since it's intensive).
        assert!((q2 / q1 - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_shr_invariant_to_mass_flow() {
        let coil = test_coil();
        let inlet = inlet_30c_50rh();
        let shr1 = coil.compute_shr(&inlet, 1.0);
        let shr2 = coil.compute_shr(&inlet, 5.0);
        assert!((shr1 - shr2).abs() < 1e-9);
    }

    #[test]
    fn test_lower_bypass_factor_increases_capacity() {
        let inlet = inlet_30c_50rh();
        let deep_coil = CoolingCoil::new("deep".into(), 30_000.0, 0.75, 0.05, 10.0, 1.0);
        let shallow_coil = CoolingCoil::new("shallow".into(), 30_000.0, 0.75, 0.30, 10.0, 1.0);
        let q_deep = deep_coil
            .compute_cooling_capacity(&inlet, 1.0)
            .unwrap()
            .total_capacity_w;
        let q_shallow = shallow_coil
            .compute_cooling_capacity(&inlet, 1.0)
            .unwrap()
            .total_capacity_w;
        assert!(
            q_deep > q_shallow,
            "deeper coil (lower BF) should remove more heat: {} vs {}",
            q_deep,
            q_shallow
        );
    }

    // ---- Edge cases --------------------------------------------------------

    #[test]
    fn test_no_cooling_at_or_below_adp() {
        let coil = test_coil();
        // Inlet at the ADP temperature → no temperature driving force.
        let cold = MoistAirState::try_new(10.0, 90.0, 101_325.0).unwrap();
        let perf = coil.compute_cooling_capacity(&cold, 1.0).unwrap();
        assert_eq!(perf.total_capacity_w, 0.0);
        assert_eq!(perf.sensible_capacity_w, 0.0);
        assert_eq!(perf.condensate_rate_kg_per_s, 0.0);

        // Below ADP — should also be zero (coil would heat, not cool).
        let colder = MoistAirState::try_new(5.0, 80.0, 101_325.0).unwrap();
        let perf2 = coil.compute_cooling_capacity(&colder, 1.0).unwrap();
        assert_eq!(perf2.total_capacity_w, 0.0);
    }

    #[test]
    fn test_zero_mass_flow() {
        let coil = test_coil();
        let inlet = inlet_30c_50rh();
        let perf = coil.compute_cooling_capacity(&inlet, 0.0).unwrap();
        assert_eq!(perf.total_capacity_w, 0.0);
        // Leaving air should equal inlet (no change).
        assert_eq!(perf.leaving_air, inlet);
    }

    #[test]
    fn test_dry_air_no_condensate() {
        // Very dry inlet air (low RH) → mostly sensible cooling, minimal latent.
        let coil = test_coil();
        let dry = MoistAirState::try_new(30.0, 5.0, 101_325.0).unwrap();
        let perf = coil.compute_cooling_capacity(&dry, 1.0).unwrap();
        assert!(perf.condensate_rate_kg_per_s >= 0.0);
        // SHR should be high (mostly sensible) for dry air.
        assert!(
            perf.shr > 0.9,
            "SHR {} should be high for dry air",
            perf.shr
        );
    }

    // ---- Part-load stub ----------------------------------------------------

    #[test]
    fn test_part_load_scales_delivered_capacity() {
        let mut coil = test_coil();
        let inlet = inlet_30c_50rh();

        let full = coil
            .compute_cooling_capacity(&inlet, 1.0)
            .unwrap()
            .total_capacity_w;

        coil.set_plr(1.0);
        let delivered_full = coil.delivered_capacity(&inlet, 1.0);
        // At PLR=1.0, delivered = min(full, rated).
        assert!((delivered_full - full.min(30_000.0)).abs() < 1e-3);

        coil.set_plr(0.5);
        let delivered_half = coil.delivered_capacity(&inlet, 1.0);
        assert!(
            (delivered_half - delivered_full * 0.5).abs() < 1e-3,
            "PLR=0.5 should halve delivered capacity"
        );

        coil.set_plr(0.0);
        let delivered_off = coil.delivered_capacity(&inlet, 1.0);
        assert!(
            delivered_off.abs() < 1e-9,
            "PLR=0.0 should give zero delivered capacity"
        );
    }

    #[test]
    fn test_delivered_capacity_capped_at_rated() {
        let mut coil = CoolingCoil::new(
            "small".into(),
            5_000.0, // 5 kW rated (smaller than air-side demand)
            0.75,
            0.05, // very effective coil
            5.0,
            2.0, // high flow
        );
        coil.set_plr(1.0);
        let inlet = inlet_30c_50rh();
        let delivered = coil.delivered_capacity(&inlet, 2.0);
        assert!(
            delivered <= 5_000.0 + 1e-6,
            "delivered {} should not exceed rated 5000",
            delivered
        );
    }
}
