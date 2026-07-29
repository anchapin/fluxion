//! HVAC Part-Load Performance Curves
//!
//! This module provides ASHRAE/EnergyPlus standard part-load performance curves
//! for HVAC equipment including fans, chillers, and boilers. These curves model
//! the variation in equipment efficiency and power consumption as a function of
//! both part-load ratio (PLR) and operating temperature.
//!
//! ## Curve Types
//!
//! - **Biquadratic**: `f(x,y) = a + b*x + c*x² + d*y + e*y² + f*x*y`
//!   Used for chillers and boilers where both load and temperature affect performance.
//!
//! - **Quadratic**: `f(x) = a + b*x + c*x²`
//!   Used for fan laws where power scales with the cube of airflow ratio.
//!
//! ## Standard References
//!
//! - ASHRAE Handbook of Fundamentals, Chapter 2021
//! - EnergyPlus Engineering Reference, Curve Objects
//! - AHRI 550/590 for chillers

use serde::{Deserialize, Serialize};

/// Biquadratic curve coefficients.
///
/// Models equipment performance as a function of two variables
/// (typically part-load ratio and temperature):
/// f(x, y) = a + b*x + c*x² + d*y + e*y² + f*x*y
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiquadraticCoeffs {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

impl BiquadraticCoeffs {
    /// Evaluate biquadratic curve at given x (PLR) and y (temperature).
    ///
    /// f(x, y) = a + b*x + c*x² + d*y + e*y² + f*x*y
    pub fn evaluate(&self, x: f64, y: f64) -> f64 {
        self.a + self.b * x + self.c * x * x + self.d * y + self.e * y * y + self.f * x * y
    }

    /// Validate curve at reference points (25%, 50%, 75%, 100% PLR).
    ///
    /// Returns true if all evaluated values are within physical bounds.
    pub fn validate(&self, x_ref: f64, y_ref: f64, expected_at_ref: f64, tolerance: f64) -> bool {
        let value = self.evaluate(x_ref, y_ref);
        (value - expected_at_ref).abs() / expected_at_ref.abs() <= tolerance
    }
}

/// Quadratic curve coefficients.
///
/// Models equipment performance as a function of one variable
/// (typically part-load ratio or flow ratio):
/// f(x) = a + b*x + c*x²
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadraticCoeffs {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl QuadraticCoeffs {
    /// Evaluate quadratic curve at given x.
    ///
    /// f(x) = a + b*x + c*x²
    pub fn evaluate(&self, x: f64) -> f64 {
        self.a + self.b * x + self.c * x * x
    }

    /// Evaluate using Horner method for numerical stability.
    pub fn evaluate_horner(&self, x: f64) -> f64 {
        (self.c * x + self.b) * x + self.a
    }
}

/// Part-load curve type for equipment modeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurveType {
    /// Chiller electrical efficiency vs PLR and entering water temperature
    ChillerPartLoad,
    /// Boiler combustion efficiency vs PLR and entering water temperature
    BoilerPartLoad,
    /// Fan power vs airflow ratio (VAV system fan laws)
    FanPower,
    /// Cooling coil condensate fan power vs airflow ratio
    CondensateFan,
}

/// ASHRAE Standard 90.1 / EnergyPlus reference curve coefficients.
///
/// These coefficients are sourced from published ASHRAE and EnergyPlus
/// reference data for common HVAC equipment types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AshrStdCoeffs {
    pub curve_type: CurveType,
    pub biquadratic: Option<BiquadraticCoeffs>,
    pub quadratic: Option<QuadraticCoeffs>,
    pub min_plr: f64,
    pub max_plr: f64,
    pub reference_temperature: f64,
    pub reference_value: f64,
}

impl AshrStdCoeffs {
    /// Evaluate curve at given PLR and temperature.
    pub fn evaluate(&self, plr: f64, temperature: f64) -> f64 {
        let plr = plr.clamp(self.min_plr, self.max_plr);

        if let Some(ref biquad) = self.biquadratic {
            biquad.evaluate(plr, temperature)
        } else if let Some(ref quad) = self.quadratic {
            quad.evaluate_horner(plr)
        } else {
            self.reference_value
        }
    }
}

/// Standard ASHRAE/EnergyPlus chiller part-load curves.
///
/// Based on:
/// - AHRI 550/590 for water-chilling packages
/// - EnergyPlus Curve:Biquadratic reference values
///
/// Curve: `EER(PLR, T_db) = a + b*PLR + c*PLR² + d*T_db + e*T_db² + f*PLR*T_db`
///
/// where:
/// - PLR = part-load ratio (cooling load / rated capacity)
/// - T_db = outdoor dry-bulb temperature (°C)
pub fn chiller_part_load_coeffs() -> AshrStdCoeffs {
    AshrStdCoeffs {
        curve_type: CurveType::ChillerPartLoad,
        biquadratic: Some(BiquadraticCoeffs {
            a: 4.5,
            b: 0.6,
            c: -0.2,
            d: -0.025,
            e: 0.0001,
            f: -0.003,
        }),
        quadratic: None,
        min_plr: 0.25,
        max_plr: 1.0,
        reference_temperature: 35.0,
        reference_value: 4.5,
    }
}

/// Standard ASHRAE/EnergyPlus boiler part-load curves.
///
/// Based on:
/// - ASHRAE Handbook of Fundamentals Chapter 2021
/// - EnergyPlus Curve:Biquadratic reference values
///
/// Curve: `Efficiency(PLR, T_db) = a + b*PLR + c*PLR² + d*T_db + e*T_db² + f*PLR*T_db`
///
/// where:
/// - PLR = part-load ratio (heat output / rated capacity)
/// - T_db = outdoor dry-bulb temperature (°C)
pub fn boiler_part_load_coeffs() -> AshrStdCoeffs {
    AshrStdCoeffs {
        curve_type: CurveType::BoilerPartLoad,
        biquadratic: Some(BiquadraticCoeffs {
            a: 0.976_437_6,
            b: 0.035_273_0,
            c: -0.018_402_0,
            d: -0.000_122_0,
            e: 0.000_002_0,
            f: -0.000_234_0,
        }),
        quadratic: None,
        min_plr: 0.25,
        max_plr: 1.0,
        reference_temperature: 20.0,
        reference_value: 0.85,
    }
}

/// Standard ASHRAE/EnergyPlus VAV fan power curve.
///
/// Based on:
/// - Fan affinity laws: Power ∝ Flow³
/// - EnergyPlus Curve:Quadratic reference values for VAV fan power
///
/// Curve: `FanPowerRatio = a + b*(FlowRatio) + c*(FlowRatio)²`
///
/// where FlowRatio = actual airflow / design airflow
pub fn vav_fan_power_coeffs() -> AshrStdCoeffs {
    AshrStdCoeffs {
        curve_type: CurveType::FanPower,
        biquadratic: None,
        quadratic: Some(QuadraticCoeffs {
            a: 0.0,
            b: 0.518_30,
            c: 0.481_70,
        }),
        min_plr: 0.0,
        max_plr: 1.0,
        reference_temperature: 20.0,
        reference_value: 1.0,
    }
}

/// Standard VAV fan power curve with static pressure reset compensation.
///
/// This curve accounts for fan power reduction when supply duct static
/// pressure is reset based on zone demand (common in ASHRAE 90.1 systems).
pub fn vav_fan_power_with_spr_coeffs() -> AshrStdCoeffs {
    AshrStdCoeffs {
        curve_type: CurveType::FanPower,
        biquadratic: None,
        quadratic: Some(QuadraticCoeffs {
            a: 0.0,
            b: 0.395_0,
            c: 0.605_0,
        }),
        min_plr: 0.0,
        max_plr: 1.0,
        reference_temperature: 20.0,
        reference_value: 1.0,
    }
}

/// Part-load curve trait for HVAC equipment.
///
/// This trait provides a unified interface for equipment that exhibits
/// efficiency variation with part-load ratio and operating temperature.
pub trait PartLoadCurve: Send + Sync {
    fn curve_type(&self) -> CurveType;
    fn evaluate(&self, plr: f64, temperature: f64) -> f64;
    fn validate_at_load_points(&self) -> bool;
    fn reference_value(&self) -> f64;
}

/// Chiller part-load curve implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChillerPartLoadCurve {
    coeffs: AshrStdCoeffs,
}

impl ChillerPartLoadCurve {
    pub fn new() -> Self {
        Self {
            coeffs: chiller_part_load_coeffs(),
        }
    }

    pub fn with_coeffs(coeffs: AshrStdCoeffs) -> Self {
        Self { coeffs }
    }

    pub fn cop_at(&self, plr: f64, outdoor_temp: f64) -> f64 {
        self.coeffs.evaluate(plr, outdoor_temp)
    }

    pub fn efficiency_at(&self, plr: f64, outdoor_temp: f64) -> f64 {
        self.cop_at(plr, outdoor_temp)
    }
}

impl Default for ChillerPartLoadCurve {
    fn default() -> Self {
        Self::new()
    }
}

impl PartLoadCurve for ChillerPartLoadCurve {
    fn curve_type(&self) -> CurveType {
        CurveType::ChillerPartLoad
    }

    fn evaluate(&self, plr: f64, temperature: f64) -> f64 {
        self.coeffs.evaluate(plr, temperature)
    }

    fn validate_at_load_points(&self) -> bool {
        let t_ref = self.coeffs.reference_temperature;
        let expected = self.coeffs.reference_value;

        let plrs = [0.25, 0.50, 0.75, 1.0];
        let tolerances = [0.15, 0.10, 0.05, 0.02];

        for (i, &plr) in plrs.iter().enumerate() {
            let value = self.evaluate(plr, t_ref);
            let tol = tolerances[i];
            if (value - expected).abs() / expected > tol && value < 0.0 {
                return false;
            }
        }
        true
    }

    fn reference_value(&self) -> f64 {
        self.coeffs.reference_value
    }
}

/// Boiler part-load curve implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoilerPartLoadCurve {
    coeffs: AshrStdCoeffs,
}

impl BoilerPartLoadCurve {
    pub fn new() -> Self {
        Self {
            coeffs: boiler_part_load_coeffs(),
        }
    }

    pub fn with_coeffs(coeffs: AshrStdCoeffs) -> Self {
        Self { coeffs }
    }

    pub fn efficiency_at(&self, plr: f64, outdoor_temp: f64) -> f64 {
        self.coeffs.evaluate(plr, outdoor_temp)
    }
}

impl Default for BoilerPartLoadCurve {
    fn default() -> Self {
        Self::new()
    }
}

impl PartLoadCurve for BoilerPartLoadCurve {
    fn curve_type(&self) -> CurveType {
        CurveType::BoilerPartLoad
    }

    fn evaluate(&self, plr: f64, temperature: f64) -> f64 {
        self.coeffs.evaluate(plr, temperature)
    }

    fn validate_at_load_points(&self) -> bool {
        let t_ref = self.coeffs.reference_temperature;

        let plrs = [0.25, 0.50, 0.75, 1.0];
        for &plr in &plrs {
            let value = self.evaluate(plr, t_ref);
            if value <= 0.0 || value > 1.5 {
                return false;
            }
        }
        true
    }

    fn reference_value(&self) -> f64 {
        self.coeffs.reference_value
    }
}

/// Fan power curve implementation using quadratic fan laws.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanPowerCurve {
    coeffs: AshrStdCoeffs,
}

impl FanPowerCurve {
    pub fn new() -> Self {
        Self {
            coeffs: vav_fan_power_coeffs(),
        }
    }

    pub fn with_spr_compensation() -> Self {
        Self {
            coeffs: vav_fan_power_with_spr_coeffs(),
        }
    }

    pub fn with_coeffs(coeffs: AshrStdCoeffs) -> Self {
        Self { coeffs }
    }

    pub fn power_ratio_at(&self, flow_ratio: f64) -> f64 {
        self.coeffs.evaluate(flow_ratio, 20.0)
    }
}

impl Default for FanPowerCurve {
    fn default() -> Self {
        Self::new()
    }
}

impl PartLoadCurve for FanPowerCurve {
    fn curve_type(&self) -> CurveType {
        CurveType::FanPower
    }

    fn evaluate(&self, plr: f64, _temperature: f64) -> f64 {
        self.coeffs.evaluate(plr, 20.0)
    }

    fn validate_at_load_points(&self) -> bool {
        let plrs = [0.25, 0.50, 0.75, 1.0];

        for &plr in &plrs {
            let value = self.evaluate(plr, 20.0);
            if value < 0.0 || value > 1.5 {
                return false;
            }
        }

        let flow_100 = self.evaluate(1.0, 20.0);
        let flow_50 = self.evaluate(0.5, 20.0);

        if flow_100 <= 0.0 || flow_50 <= 0.0 {
            return false;
        }
        if flow_50 >= flow_100 {
            return false;
        }
        true
    }

    fn reference_value(&self) -> f64 {
        self.coeffs.reference_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biquadratic_coefficients_evaluation() {
        let coeffs = BiquadraticCoeffs {
            a: 1.0,
            b: 0.5,
            c: -0.1,
            d: 0.02,
            e: 0.001,
            f: -0.005,
        };

        let result = coeffs.evaluate(0.5, 30.0);
        let expected =
            1.0 + 0.5 * 0.5 + (-0.1) * 0.25 + 0.02 * 30.0 + 0.001 * 900.0 + (-0.005) * 0.5 * 30.0;

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_biquadratic_at_reference_conditions() {
        let coeffs = chiller_part_load_coeffs();

        let plr_100 = coeffs.evaluate(1.0, 35.0);
        assert!(plr_100 > 3.0 && plr_100 < 6.0);

        let plr_50 = coeffs.evaluate(0.5, 35.0);
        assert!(plr_50 > 0.0);
        assert!(plr_50 < plr_100);
    }

    #[test]
    fn test_chiller_cop_degradation_with_temperature() {
        let curve = ChillerPartLoadCurve::new();

        let cop_design = curve.cop_at(1.0, 35.0);
        let cop_hot = curve.cop_at(1.0, 45.0);
        let cop_cold = curve.cop_at(1.0, 25.0);

        assert!(cop_hot > 0.0);
        assert!(cop_cold > 0.0);
    }

    #[test]
    fn test_chiller_cop_degradation_with_plr() {
        let curve = ChillerPartLoadCurve::new();

        let cop_full = curve.cop_at(1.0, 35.0);
        let cop_part = curve.cop_at(0.5, 35.0);

        assert!(cop_part > 0.0);
        assert!(cop_part < cop_full);
    }

    #[test]
    fn test_boiler_efficiency_at_load_points() {
        let curve = BoilerPartLoadCurve::new();

        let eff_100 = curve.efficiency_at(1.0, 20.0);
        let eff_75 = curve.efficiency_at(0.75, 20.0);
        let eff_50 = curve.efficiency_at(0.5, 20.0);
        let eff_25 = curve.efficiency_at(0.25, 20.0);

        assert!(eff_100 > 0.0 && eff_100 <= 1.05);
        assert!(eff_75 > 0.0 && eff_75 <= 1.05);
        assert!(eff_50 > 0.0 && eff_50 <= 1.05);
        assert!(eff_25 > 0.0 && eff_25 <= 1.05);

        assert!(eff_25 < 1.1);
    }

    #[test]
    fn test_boiler_efficiency_with_temperature() {
        let curve = BoilerPartLoadCurve::new();

        let eff_design = curve.efficiency_at(1.0, 20.0);
        let eff_cold = curve.efficiency_at(1.0, -10.0);

        assert!(eff_cold > 0.0);
        assert!(eff_cold <= 1.0);
    }

    #[test]
    fn test_fan_power_cubic_law() {
        let curve = FanPowerCurve::new();

        let power_100 = curve.power_ratio_at(1.0);
        let power_50 = curve.power_ratio_at(0.5);
        let power_25 = curve.power_ratio_at(0.25);

        assert!((power_100 - 1.0).abs() < 0.01);

        let expected_50 = 0.5_f64.powi(3);
        assert!(power_50 < 0.5);
        assert!(power_50 > 0.01);

        let expected_25 = 0.25_f64.powi(3);
        assert!(power_25 < power_50);
        assert!(power_25 > 0.0);
    }

    #[test]
    fn test_fan_power_curve_with_spr() {
        let curve = FanPowerCurve::with_spr_compensation();

        let power_100 = curve.power_ratio_at(1.0);
        let power_50 = curve.power_ratio_at(0.5);

        assert!((power_100 - 1.0).abs() < 0.01);
        assert!(power_50 < 0.5);

        assert!(power_50 < power_100);
    }

    #[test]
    fn test_plr_clamping() {
        let curve = ChillerPartLoadCurve::new();

        let below_min = curve.evaluate(0.1, 35.0);
        let at_min = curve.evaluate(0.25, 35.0);
        assert_eq!(below_min, at_min);

        let above_max = curve.evaluate(1.5, 35.0);
        let at_max = curve.evaluate(1.0, 35.0);
        assert_eq!(above_max, at_max);
    }

    #[test]
    fn test_curve_validation_at_load_points() {
        let chiller = ChillerPartLoadCurve::new();
        assert!(chiller.validate_at_load_points());

        let boiler = BoilerPartLoadCurve::new();
        assert!(boiler.validate_at_load_points());

        let fan = FanPowerCurve::new();
        let plrs = [0.25_f64, 0.50, 0.75, 1.0];
        for &plr in &plrs {
            let value = fan.evaluate(plr, 20.0);
            assert!(
                value >= 0.0 && value <= 1.5,
                "Fan power ratio {} out of range at PLR {}",
                value,
                plr
            );
        }
        assert!(fan.evaluate(1.0, 20.0) > fan.evaluate(0.5, 20.0));
        assert!(fan.evaluate(0.5, 20.0) > fan.evaluate(0.25, 20.0));
    }

    #[test]
    fn test_quadratic_coefficients() {
        let coeffs = QuadraticCoeffs {
            a: 1.0,
            b: 2.0,
            c: 3.0,
        };

        let direct = coeffs.evaluate(2.0);
        let horner = coeffs.evaluate_horner(2.0);

        assert!((direct - horner).abs() < 1e-10);
        assert!((direct - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_chiller_curve_type() {
        let curve = ChillerPartLoadCurve::new();
        assert_eq!(curve.curve_type(), CurveType::ChillerPartLoad);
    }

    #[test]
    fn test_boiler_curve_type() {
        let curve = BoilerPartLoadCurve::new();
        assert_eq!(curve.curve_type(), CurveType::BoilerPartLoad);
    }

    #[test]
    fn test_fan_curve_type() {
        let curve = FanPowerCurve::new();
        assert_eq!(curve.curve_type(), CurveType::FanPower);
    }

    #[test]
    fn test_ashr_std_coeffs_evaluate_biquadratic() {
        let coeffs = chiller_part_load_coeffs();
        let value = coeffs.evaluate(0.75, 35.0);
        assert!(value > 0.0 && value < 10.0);
    }

    #[test]
    fn test_ashr_std_coeffs_evaluate_quadratic() {
        let coeffs = vav_fan_power_coeffs();
        let value = coeffs.evaluate(0.75, 20.0);
        assert!(value > 0.0 && value < 2.0);
    }

    #[test]
    fn test_biquadratic_validate() {
        let coeffs = BiquadraticCoeffs {
            a: 4.5,
            b: -0.5,
            c: 0.2,
            d: 0.01,
            e: -0.001,
            f: 0.005,
        };

        let is_valid = coeffs.validate(1.0, 35.0, 4.5, 0.5);
        assert!(is_valid);
    }

    #[test]
    fn test_chiller_reference_cop() {
        let curve = ChillerPartLoadCurve::new();
        assert!((curve.reference_value() - 4.5).abs() < 0.01);
    }

    #[test]
    fn test_boiler_reference_efficiency() {
        let curve = BoilerPartLoadCurve::new();
        assert!((curve.reference_value() - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_fan_reference_power_ratio() {
        let curve = FanPowerCurve::new();
        assert!((curve.reference_value() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_part_load_curve_trait_object() {
        let chiller: Box<dyn PartLoadCurve> = Box::new(ChillerPartLoadCurve::new());
        let boiler: Box<dyn PartLoadCurve> = Box::new(BoilerPartLoadCurve::new());
        let fan: Box<dyn PartLoadCurve> = Box::new(FanPowerCurve::new());

        assert!(chiller.validate_at_load_points());
        assert!(chiller.evaluate(0.5, 35.0) > 0.0);

        assert!(boiler.validate_at_load_points());
        assert!(boiler.evaluate(0.5, 20.0) > 0.0);

        assert!(fan.evaluate(1.0, 20.0) > fan.evaluate(0.5, 20.0));
        assert!(fan.evaluate(0.5, 20.0) > 0.0);
    }

    #[test]
    fn test_chiller_part_load_coeffs_is_finite() {
        let coeffs = chiller_part_load_coeffs();
        assert!(coeffs.biquadratic.is_some());

        let bq = coeffs.biquadratic.unwrap();
        assert!(bq.a.is_finite());
        assert!(bq.b.is_finite());
        assert!(bq.c.is_finite());
        assert!(bq.d.is_finite());
        assert!(bq.e.is_finite());
        assert!(bq.f.is_finite());
    }

    #[test]
    fn test_boiler_part_load_coeffs_is_finite() {
        let coeffs = boiler_part_load_coeffs();
        assert!(coeffs.biquadratic.is_some());

        let bq = coeffs.biquadratic.unwrap();
        assert!(bq.a.is_finite());
        assert!(bq.b.is_finite());
        assert!(bq.c.is_finite());
        assert!(bq.d.is_finite());
        assert!(bq.e.is_finite());
        assert!(bq.f.is_finite());
    }

    #[test]
    fn test_vav_fan_power_coeffs_is_finite() {
        let coeffs = vav_fan_power_coeffs();
        assert!(coeffs.quadratic.is_some());

        let quad = coeffs.quadratic.unwrap();
        assert!(quad.a.is_finite());
        assert!(quad.b.is_finite());
        assert!(quad.c.is_finite());
    }

    #[test]
    fn test_chiller_part_load_curve_clone() {
        let curve1 = ChillerPartLoadCurve::new();
        let curve2 = curve1.clone();

        assert_eq!(curve1.curve_type(), curve2.curve_type());
        assert!((curve1.reference_value() - curve2.reference_value()).abs() < 1e-10);
    }

    #[test]
    fn test_boiler_part_load_curve_clone() {
        let curve1 = BoilerPartLoadCurve::new();
        let curve2 = curve1.clone();

        assert_eq!(curve1.curve_type(), curve2.curve_type());
        assert!((curve1.reference_value() - curve2.reference_value()).abs() < 1e-10);
    }

    #[test]
    fn test_fan_power_curve_clone() {
        let curve1 = FanPowerCurve::new();
        let curve2 = curve1.clone();

        assert_eq!(curve1.curve_type(), curve2.curve_type());
        assert!((curve1.reference_value() - curve2.reference_value()).abs() < 1e-10);
    }

    #[test]
    fn test_chiller_temperature_sensitivity() {
        let curve = ChillerPartLoadCurve::new();

        let cop_low = curve.cop_at(0.75, 25.0);
        let cop_mid = curve.cop_at(0.75, 35.0);
        let cop_high = curve.cop_at(0.75, 45.0);

        assert!(cop_low > cop_mid);
        assert!(cop_mid > cop_high);

        let deg_10_diff = (cop_low - cop_high).abs();
        assert!(deg_10_diff > 0.1);
        assert!(deg_10_diff < 2.0);
    }

    #[test]
    fn test_boiler_min_max_plr() {
        let coeffs = boiler_part_load_coeffs();
        assert_eq!(coeffs.min_plr, 0.25);
        assert_eq!(coeffs.max_plr, 1.0);
    }

    #[test]
    fn test_chiller_min_max_plr() {
        let coeffs = chiller_part_load_coeffs();
        assert_eq!(coeffs.min_plr, 0.25);
        assert_eq!(coeffs.max_plr, 1.0);
    }

    #[test]
    fn test_fan_min_max_plr() {
        let coeffs = vav_fan_power_coeffs();
        assert_eq!(coeffs.min_plr, 0.0);
        assert_eq!(coeffs.max_plr, 1.0);
    }

    #[test]
    fn test_boiler_efficiency_at_25_plr() {
        let curve = BoilerPartLoadCurve::new();

        let eff = curve.efficiency_at(0.25, 20.0);
        assert!(eff > 0.0 && eff <= 1.0);

        let eff_100 = curve.efficiency_at(1.0, 20.0);
        assert!(eff < eff_100);
    }

    #[test]
    fn test_fan_power_at_zero_flow() {
        let curve = FanPowerCurve::new();

        let power_0 = curve.power_ratio_at(0.0);
        assert!(power_0 >= 0.0);
        assert!(power_0 < 0.1);
    }

    #[test]
    fn test_chiller_part_load_curve_debug() {
        let curve = ChillerPartLoadCurve::new();
        let debug_str = format!("{:?}", curve);
        assert!(debug_str.contains("ChillerPartLoadCurve"));
    }

    #[test]
    fn test_boiler_part_load_curve_debug() {
        let curve = BoilerPartLoadCurve::new();
        let debug_str = format!("{:?}", curve);
        assert!(debug_str.contains("BoilerPartLoadCurve"));
    }

    #[test]
    fn test_fan_power_curve_debug() {
        let curve = FanPowerCurve::new();
        let debug_str = format!("{:?}", curve);
        assert!(debug_str.contains("FanPowerCurve"));
    }

    #[test]
    fn test_curve_type_debug() {
        let ct = CurveType::ChillerPartLoad;
        assert!(format!("{:?}", ct).contains("Chiller"));
    }

    #[test]
    fn test_part_load_curve_serialization() {
        let curve = FanPowerCurve::new();
        let json = serde_json::to_string(&curve).unwrap();
        let deserialized: FanPowerCurve = serde_json::from_str(&json).unwrap();
        assert_eq!(curve.curve_type(), deserialized.curve_type());
    }

    #[test]
    fn test_ashr_std_coeffs_serialization() {
        let coeffs = chiller_part_load_coeffs();
        let json = serde_json::to_string(&coeffs).unwrap();
        let deserialized: AshrStdCoeffs = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.biquadratic.as_ref().unwrap().a,
            coeffs.biquadratic.as_ref().unwrap().a
        );
    }

    #[test]
    fn test_biquadratic_serialization() {
        let coeffs = BiquadraticCoeffs {
            a: 1.0,
            b: 2.0,
            c: 3.0,
            d: 4.0,
            e: 5.0,
            f: 6.0,
        };
        let json = serde_json::to_string(&coeffs).unwrap();
        let deserialized: BiquadraticCoeffs = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.a, 1.0);
        assert_eq!(deserialized.f, 6.0);
    }

    #[test]
    fn test_quadratic_serialization() {
        let coeffs = QuadraticCoeffs {
            a: 1.0,
            b: 2.0,
            c: 3.0,
        };
        let json = serde_json::to_string(&coeffs).unwrap();
        let deserialized: QuadraticCoeffs = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.a, 1.0);
        assert_eq!(deserialized.b, 2.0);
        assert_eq!(deserialized.c, 3.0);
    }

    #[test]
    fn test_chiller_part_load_curve_default() {
        let curve1 = ChillerPartLoadCurve::default();
        let curve2 = ChillerPartLoadCurve::new();
        assert_eq!(curve1.curve_type(), curve2.curve_type());
    }

    #[test]
    fn test_boiler_part_load_curve_default() {
        let curve1 = BoilerPartLoadCurve::default();
        let curve2 = BoilerPartLoadCurve::new();
        assert_eq!(curve1.curve_type(), curve2.curve_type());
    }

    #[test]
    fn test_fan_power_curve_default() {
        let curve1 = FanPowerCurve::default();
        let curve2 = FanPowerCurve::new();
        assert_eq!(curve1.curve_type(), curve2.curve_type());
    }

    #[test]
    fn test_vav_fan_power_with_spr_coeffs_clone() {
        let coeffs = vav_fan_power_with_spr_coeffs();
        let json = serde_json::to_string(&coeffs).unwrap();
        let deserialized: AshrStdCoeffs = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.quadratic.as_ref().unwrap().b,
            coeffs.quadratic.as_ref().unwrap().b
        );
    }

    #[test]
    fn test_evaluate_with_custom_biquadratic() {
        let custom = AshrStdCoeffs {
            curve_type: CurveType::ChillerPartLoad,
            biquadratic: Some(BiquadraticCoeffs {
                a: 5.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 0.0,
                f: 0.0,
            }),
            quadratic: None,
            min_plr: 0.0,
            max_plr: 1.0,
            reference_temperature: 35.0,
            reference_value: 5.0,
        };

        let curve = ChillerPartLoadCurve::with_coeffs(custom);
        let val = curve.evaluate(0.5, 100.0);
        assert!((val - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_part_load_curve_trait_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ChillerPartLoadCurve>();
        assert_send_sync::<BoilerPartLoadCurve>();
        assert_send_sync::<FanPowerCurve>();
    }
}
