use crate::{GridModelError, VoltageCoupler, VoltagePu};

#[derive(Debug, Clone, Default)]
pub struct HeatPumpVoltageModel {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub voltage_nominal: f64,
}

impl HeatPumpVoltageModel {
    pub fn new(a: f64, b: f64, c: f64, voltage_nominal: f64) -> Self {
        Self {
            a,
            b,
            c,
            voltage_nominal,
        }
    }

    pub fn cop_adjustment_factor(&self, voltage_pu: VoltagePu) -> Result<f64, GridModelError> {
        if !(0.5..=1.5).contains(&voltage_pu) {
            return Err(GridModelError::VoltageOutOfRange {
                voltage: voltage_pu,
            });
        }
        let factor = self.a + self.b * voltage_pu + self.c * voltage_pu * voltage_pu;
        Ok(factor)
    }

    pub fn apply_to_coupler(
        &self,
        coupler: &mut VoltageCoupler,
        voltage_pu: VoltagePu,
    ) -> Result<(), GridModelError> {
        if coupler.thermal_mass_j_per_k <= 0.0 {
            return Err(GridModelError::ZeroThermalMass);
        }
        let factor = self.cop_adjustment_factor(voltage_pu)?;
        if factor < 0.0 {
            return Err(GridModelError::NegativeAdjustment { factor });
        }
        coupler.current_voltage_pu = voltage_pu;
        coupler.current_cop *= factor;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn default_model() -> HeatPumpVoltageModel {
        HeatPumpVoltageModel::new(0.85, 0.30, -0.15, 230.0)
    }

    #[test]
    fn test_cop_at_nominal_is_unity() {
        let model = default_model();
        let factor = model.cop_adjustment_factor(1.0).unwrap();
        assert_relative_eq!(factor, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cop_at_09pu_less_than_cop_at_10pu() {
        let model = default_model();
        let f_09 = model.cop_adjustment_factor(0.9).unwrap();
        let f_10 = model.cop_adjustment_factor(1.0).unwrap();
        assert!(
            f_09 < f_10,
            "COP at 0.9 pu ({f_09}) should be less than COP at 1.0 pu ({f_10})"
        );
    }

    #[test]
    fn test_cop_polynomial_values() {
        let model = default_model();
        let f_10 = model.cop_adjustment_factor(1.0).unwrap();
        let f_09 = model.cop_adjustment_factor(0.9).unwrap();
        let f_05 = model.cop_adjustment_factor(1.05).unwrap();
        assert_relative_eq!(f_10, 1.0, epsilon = 1e-10);
        assert!(f_09 < 1.0, "f(0.9) = {f_09} should be < 1.0");
        assert!(
            f_05 < 1.0 && f_05 > f_09,
            "f(1.05) = {f_05} should be < 1.0 but > f(0.9)"
        );
    }

    #[test]
    fn test_voltage_out_of_range() {
        let model = default_model();
        let result = model.cop_adjustment_factor(1.6);
        assert!(result.is_err());
        let result = model.cop_adjustment_factor(0.4);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_to_coupler() {
        let model = default_model();
        let mut coupler = VoltageCoupler::new(1000.0, 5000.0);
        coupler.set_cop(3.5);

        model.apply_to_coupler(&mut coupler, 0.9).unwrap();

        let expected_factor = model.cop_adjustment_factor(0.9).unwrap();
        let expected_cop = 3.5 * expected_factor;
        assert_relative_eq!(coupler.current_cop, expected_cop, epsilon = 1e-10);
        assert_eq!(coupler.current_voltage_pu, 0.9);
    }

    #[test]
    fn test_apply_zero_thermal_mass() {
        let model = default_model();
        let mut coupler = VoltageCoupler::new(1000.0, 0.0);
        let result = model.apply_to_coupler(&mut coupler, 0.9);
        assert!(result.is_err());
    }
}
