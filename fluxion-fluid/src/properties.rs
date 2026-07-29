//! Thermophysical fluid properties.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluidProperties {
    pub temperature: f64,
    pub pressure: f64,
    pub mass_flow_rate: f64,
    pub density: f64,
    pub specific_heat: f64,
}

impl FluidProperties {
    #[must_use]
    pub fn new(
        temperature: f64,
        pressure: f64,
        mass_flow_rate: f64,
        density: f64,
        specific_heat: f64,
    ) -> Self {
        Self {
            temperature,
            pressure,
            mass_flow_rate,
            density,
            specific_heat,
        }
    }

    #[must_use]
    pub fn volumetric_flow_rate(&self) -> f64 {
        if self.density > 0.0 {
            self.mass_flow_rate / self.density
        } else {
            0.0
        }
    }

    #[must_use]
    pub fn enthalpy(&self) -> f64 {
        self.specific_heat * self.temperature
    }

    #[must_use]
    pub fn mass_flow_rate_from_volumetric(&self, volumetric_flow_rate: f64) -> f64 {
        self.density * volumetric_flow_rate
    }

    #[must_use]
    pub fn heat_transfer_rate(&self, outlet_temperature: f64, inlet_temperature: f64) -> f64 {
        self.mass_flow_rate * self.specific_heat * (outlet_temperature - inlet_temperature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fluid_properties_creation() {
        let props = FluidProperties::new(293.15, 101325.0, 0.5, 998.0, 4182.0);
        assert!((props.temperature - 293.15).abs() < 0.01);
        assert!((props.mass_flow_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_volumetric_flow_rate() {
        let props = FluidProperties::new(293.15, 101325.0, 0.5, 998.0, 4182.0);
        let vfr = props.volumetric_flow_rate();
        assert!((vfr - 0.0005).abs() < 0.00001);
    }

    #[test]
    fn test_heat_transfer_rate() {
        let props = FluidProperties::new(293.15, 101325.0, 0.5, 998.0, 4182.0);
        let q = props.heat_transfer_rate(303.15, 293.15);
        assert!((q - 20910.0).abs() < 100.0);
    }

    #[test]
    fn test_mass_flow_rate_from_volumetric() {
        let props = FluidProperties::new(293.15, 101325.0, 0.0, 998.0, 4182.0);
        let mfr = props.mass_flow_rate_from_volumetric(0.0005);
        assert!((mfr - 0.499).abs() < 0.001);
    }
}
