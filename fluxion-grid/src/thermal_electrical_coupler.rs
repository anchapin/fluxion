use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ThermalElectricalCoupler {
    pub compressor_power_demand_w: f64,
    pub thermal_mass_j_per_k: f64,
    pub current_cop: f64,
    pub nominal_voltage_v: f64,
    pub current_voltage_pu: f64,
}

impl ThermalElectricalCoupler {
    pub fn new(compressor_power_demand_w: f64, thermal_mass_j_per_k: f64) -> Self {
        Self {
            compressor_power_demand_w,
            thermal_mass_j_per_k,
            current_cop: 1.0,
            nominal_voltage_v: 230.0,
            current_voltage_pu: 1.0,
        }
    }

    pub fn with_voltage(mut self, voltage_pu: f64) -> Self {
        self.current_voltage_pu = voltage_pu;
        self
    }

    pub fn set_cop(&mut self, cop: f64) {
        self.current_cop = cop;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupler_default_voltage() {
        let coupler = ThermalElectricalCoupler::new(1000.0, 5000.0);
        assert_eq!(coupler.current_voltage_pu, 1.0);
        assert_eq!(coupler.current_cop, 1.0);
    }

    #[test]
    fn test_coupler_with_voltage() {
        let coupler = ThermalElectricalCoupler::new(1000.0, 5000.0).with_voltage(0.95);
        assert_eq!(coupler.current_voltage_pu, 0.95);
    }
}
