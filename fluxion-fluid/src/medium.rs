//! Medium types and the `FluidMedium` trait for thermophysical properties.

use std::fmt;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Medium {
    Water,
    Air,
    Refrigerant,
    Steam,
    Glycol,
    Oil,
}

impl fmt::Display for Medium {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Medium {
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Water => "Water",
            Self::Air => "Air",
            Self::Refrigerant => "Refrigerant",
            Self::Steam => "Steam",
            Self::Glycol => "Glycol",
            Self::Oil => "Oil",
        }
    }
}

#[derive(Debug, Error)]
pub enum MediumError {
    #[error("Property evaluation failed for {0}")]
    PropertyEvaluation(String),
    #[error("Invalid temperature for {medium}: {temperature} K")]
    InvalidTemperature { medium: Medium, temperature: f64 },
    #[error("Invalid pressure for {medium}: {pressure} Pa")]
    InvalidPressure { medium: Medium, pressure: f64 },
}

pub trait FluidMedium: Clone + Copy {
    fn medium(&self) -> Medium;

    fn density(&self, temperature: f64, pressure: f64) -> Result<f64, MediumError>;

    fn specific_heat(&self, temperature: f64, pressure: f64) -> Result<f64, MediumError>;

    fn dynamic_viscosity(&self, temperature: f64, pressure: f64) -> Result<f64, MediumError>;

    fn thermal_conductivity(&self, temperature: f64, pressure: f64) -> Result<f64, MediumError>;

    fn prandtl_number(&self, temperature: f64, pressure: f64) -> Result<f64, MediumError> {
        let cp = self.specific_heat(temperature, pressure)?;
        let mu = self.dynamic_viscosity(temperature, pressure)?;
        let k = self.thermal_conductivity(temperature, pressure)?;
        if k.abs() < f64::EPSILON {
            return Err(MediumError::PropertyEvaluation(format!(
                "Prandtl calculation failed: thermal conductivity is zero for {:?}",
                self.medium()
            )));
        }
        Ok(cp * mu / k)
    }

    fn saturation_temperature(&self, pressure: f64) -> Result<f64, MediumError>;

    fn saturation_pressure(&self, temperature: f64) -> Result<f64, MediumError>;

    fn validate_temperature(&self, temperature: f64) -> Result<(), MediumError> {
        let (t_min, t_max) = self.operating_temperature_range();
        if temperature < t_min || temperature > t_max {
            return Err(MediumError::InvalidTemperature {
                medium: self.medium(),
                temperature,
            });
        }
        Ok(())
    }

    fn validate_pressure(&self, pressure: f64) -> Result<(), MediumError> {
        if pressure <= 0.0 {
            return Err(MediumError::InvalidPressure {
                medium: self.medium(),
                pressure,
            });
        }
        Ok(())
    }

    fn operating_temperature_range(&self) -> (f64, f64);

    fn operating_pressure_range(&self) -> (f64, f64);
}

#[derive(Debug, Clone, Copy)]
pub struct WaterMedium;

impl FluidMedium for WaterMedium {
    fn medium(&self) -> Medium {
        Medium::Water
    }

    fn density(&self, temperature: f64, _pressure: f64) -> Result<f64, MediumError> {
        self.validate_temperature(temperature)?;
        let t_celsius = temperature - 273.15;
        let rho = 1_000.0 - 0.0178 * t_celsius - 0.000_001_2 * t_celsius.powi(2);
        Ok(rho.max(500.0))
    }

    fn specific_heat(&self, temperature: f64, _pressure: f64) -> Result<f64, MediumError> {
        self.validate_temperature(temperature)?;
        let t_celsius = temperature - 273.15;
        let cp = 4_182.0 + t_celsius * 0.000_6;
        Ok(cp.max(4_000.0))
    }

    fn dynamic_viscosity(&self, temperature: f64, _pressure: f64) -> Result<f64, MediumError> {
        self.validate_temperature(temperature)?;
        let t_celsius = temperature - 273.15;
        let mu = 0.000_001 * (1.0 + 0.000_03 * t_celsius.powi(2));
        Ok(mu.max(0.000_000_1))
    }

    fn thermal_conductivity(&self, temperature: f64, _pressure: f64) -> Result<f64, MediumError> {
        self.validate_temperature(temperature)?;
        let t_celsius = temperature - 273.15;
        let k = 0.5984 - 0.000_085 * t_celsius;
        Ok(k.max(0.01))
    }

    fn saturation_temperature(&self, pressure: f64) -> Result<f64, MediumError> {
        self.validate_pressure(pressure)?;
        let p_bar = pressure / 100_000.0;
        let t_sat = 99.6 + 28.1 * (p_bar / 1.013_25).ln() - 0.000_12 * p_bar;
        Ok((t_sat + 273.15).max(273.15))
    }

    fn saturation_pressure(&self, temperature: f64) -> Result<f64, MediumError> {
        self.validate_temperature(temperature)?;
        let t_celsius = temperature - 273.15;
        let p_sat = 101_325.0 * (t_celsius / 100.0).exp();
        Ok(p_sat.max(600.0))
    }

    fn operating_temperature_range(&self) -> (f64, f64) {
        (273.15, 373.15)
    }

    fn operating_pressure_range(&self) -> (f64, f64) {
        (1_000.0, 10_000_000.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AirMedium;

impl FluidMedium for AirMedium {
    fn medium(&self) -> Medium {
        Medium::Air
    }

    fn density(&self, temperature: f64, pressure: f64) -> Result<f64, MediumError> {
        self.validate_temperature(temperature)?;
        self.validate_pressure(pressure)?;
        let r_specific = 287.05;
        Ok(pressure / (r_specific * temperature))
    }

    fn specific_heat(&self, _temperature: f64, _pressure: f64) -> Result<f64, MediumError> {
        Ok(1_006.0)
    }

    fn dynamic_viscosity(&self, temperature: f64, _pressure: f64) -> Result<f64, MediumError> {
        self.validate_temperature(temperature)?;
        let mu = 1.458e-6 * temperature.powf(1.5) / (temperature + 110.4);
        Ok(mu.max(0.000_001))
    }

    fn thermal_conductivity(&self, temperature: f64, _pressure: f64) -> Result<f64, MediumError> {
        self.validate_temperature(temperature)?;
        let k = 0.024 + 0.000_07 * (temperature - 273.15);
        Ok(k.max(0.001))
    }

    fn saturation_temperature(&self, _pressure: f64) -> Result<f64, MediumError> {
        Err(MediumError::PropertyEvaluation(
            "Air does not saturate".to_string(),
        ))
    }

    fn saturation_pressure(&self, _temperature: f64) -> Result<f64, MediumError> {
        Err(MediumError::PropertyEvaluation(
            "Air does not saturate".to_string(),
        ))
    }

    fn operating_temperature_range(&self) -> (f64, f64) {
        (200.0, 2_000.0)
    }

    fn operating_pressure_range(&self) -> (f64, f64) {
        (100.0, 10_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_medium_name() {
        assert_eq!(Medium::Water.name(), "Water");
        assert_eq!(Medium::Air.name(), "Air");
    }

    #[test]
    fn test_water_density() {
        let water = WaterMedium;
        let rho = water.density(293.15, 101_325.0).unwrap();
        assert!((rho - 998.0).abs() < 10.0);
    }

    #[test]
    fn test_water_specific_heat() {
        let water = WaterMedium;
        let cp = water.specific_heat(293.15, 101_325.0).unwrap();
        assert!((cp - 4_182.0).abs() < 10.0);
    }

    #[test]
    fn test_air_density() {
        let air = AirMedium;
        let rho = air.density(293.15, 101_325.0).unwrap();
        assert!((rho - 1.2).abs() < 0.2);
    }

    #[test]
    fn test_air_prandtl() {
        let air = AirMedium;
        let pr = air.prandtl_number(300.0, 101_325.0).unwrap();
        assert!((pr - 0.71).abs() < 0.5);
    }

    #[test]
    fn test_water_invalid_temperature() {
        let water = WaterMedium;
        let result = water.density(173.15, 101_325.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_water_invalid_pressure() {
        let water = WaterMedium;
        let result = water.validate_pressure(-100.0);
        assert!(result.is_err());
    }
}
