//! Typed fluid ports for HVAC component connections.
//!
//! Ports are compile-time typed by their medium, preventing runtime type mismatches
//! in acausal component connections.

use crate::medium::{FluidMedium, Medium};
use crate::properties::FluidProperties;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortDirection {
    Inlet,
    Outlet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortSide {
    Supply,
    Return,
}

#[derive(Debug, Error)]
pub enum PortError {
    #[error("Medium mismatch: port expects {expected}, got {actual}")]
    MediumMismatch { expected: Medium, actual: Medium },
    #[error("Invalid mass flow rate: {0} kg/s")]
    InvalidMassFlowRate(f64),
    #[error("Invalid temperature: {0} K")]
    InvalidTemperature(f64),
    #[error("Invalid pressure: {0} Pa")]
    InvalidPressure(f64),
    #[error("Port property access failed: {0}")]
    PropertyAccess(String),
}

pub trait FluidPort: Sized {
    type Medium: FluidMedium;

    fn direction(&self) -> PortDirection;

    fn side(&self) -> PortSide;

    fn medium(&self) -> Medium;

    fn temperature(&self) -> f64;

    fn pressure(&self) -> f64;

    fn mass_flow_rate(&self) -> f64;

    fn set_temperature(&mut self, temperature: f64) -> Result<(), PortError>;

    fn set_pressure(&mut self, pressure: f64) -> Result<(), PortError>;

    fn set_mass_flow_rate(&mut self, mass_flow_rate: f64) -> Result<(), PortError>;

    fn validate(&self) -> Result<(), PortError> {
        if self.mass_flow_rate() < 0.0 {
            return Err(PortError::InvalidMassFlowRate(self.mass_flow_rate()));
        }
        if self.temperature() < 0.0 {
            return Err(PortError::InvalidTemperature(self.temperature()));
        }
        if self.pressure() <= 0.0 {
            return Err(PortError::InvalidPressure(self.pressure()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FluidPortState<M: FluidMedium> {
    temperature: f64,
    pressure: f64,
    mass_flow_rate: f64,
    medium: M,
}

impl<M: FluidMedium> FluidPortState<M> {
    #[must_use]
    pub fn new(medium: M, temperature: f64, pressure: f64, mass_flow_rate: f64) -> Self {
        Self {
            temperature,
            pressure,
            mass_flow_rate,
            medium,
        }
    }

    #[must_use]
    pub fn medium(&self) -> M {
        self.medium
    }

    #[must_use]
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    #[must_use]
    pub fn pressure(&self) -> f64 {
        self.pressure
    }

    #[must_use]
    pub fn mass_flow_rate(&self) -> f64 {
        self.mass_flow_rate
    }

    pub fn set_temperature(&mut self, temperature: f64) -> Result<(), PortError> {
        self.medium
            .validate_temperature(temperature)
            .map_err(|_| PortError::InvalidTemperature(temperature))?;
        self.temperature = temperature;
        Ok(())
    }

    pub fn set_pressure(&mut self, pressure: f64) -> Result<(), PortError> {
        self.medium
            .validate_pressure(pressure)
            .map_err(|_| PortError::InvalidPressure(pressure))?;
        self.pressure = pressure;
        Ok(())
    }

    pub fn set_mass_flow_rate(&mut self, mass_flow_rate: f64) -> Result<(), PortError> {
        if mass_flow_rate < 0.0 {
            return Err(PortError::InvalidMassFlowRate(mass_flow_rate));
        }
        self.mass_flow_rate = mass_flow_rate;
        Ok(())
    }

    pub fn properties(&self) -> Result<FluidProperties, PortError> {
        Ok(FluidProperties::new(
            self.temperature,
            self.pressure,
            self.mass_flow_rate,
            self.medium.density(self.temperature, self.pressure).map_err(|e| {
                PortError::PropertyAccess(format!("density: {e}"))
            })?,
            self.medium.specific_heat(self.temperature, self.pressure).map_err(|e| {
                PortError::PropertyAccess(format!("specific_heat: {e}"))
            })?,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct InletPort<M: FluidMedium> {
    side: PortSide,
    state: FluidPortState<M>,
}

#[derive(Debug, Clone)]
pub struct OutletPort<M: FluidMedium> {
    side: PortSide,
    state: FluidPortState<M>,
}

impl<M: FluidMedium> InletPort<M> {
    #[must_use]
    pub fn new(
        side: PortSide,
        medium: M,
        temperature: f64,
        pressure: f64,
        mass_flow_rate: f64,
    ) -> Self {
        Self {
            side,
            state: FluidPortState::new(medium, temperature, pressure, mass_flow_rate),
        }
    }

    #[must_use]
    pub fn side(&self) -> PortSide {
        self.side
    }

    #[must_use]
    pub fn state(&self) -> &FluidPortState<M> {
        &self.state
    }

    #[must_use]
    pub fn mut_state(&mut self) -> &mut FluidPortState<M> {
        &mut self.state
    }
}

impl<M: FluidMedium> FluidPort for InletPort<M> {
    type Medium = M;

    fn direction(&self) -> PortDirection {
        PortDirection::Inlet
    }

    fn side(&self) -> PortSide {
        self.side
    }

    fn medium(&self) -> Medium {
        self.state.medium.medium()
    }

    fn temperature(&self) -> f64 {
        self.state.temperature()
    }

    fn pressure(&self) -> f64 {
        self.state.pressure()
    }

    fn mass_flow_rate(&self) -> f64 {
        self.state.mass_flow_rate()
    }

    fn set_temperature(&mut self, temperature: f64) -> Result<(), PortError> {
        self.state.set_temperature(temperature)
    }

    fn set_pressure(&mut self, pressure: f64) -> Result<(), PortError> {
        self.state.set_pressure(pressure)
    }

    fn set_mass_flow_rate(&mut self, mass_flow_rate: f64) -> Result<(), PortError> {
        self.state.set_mass_flow_rate(mass_flow_rate)
    }
}

impl<M: FluidMedium> OutletPort<M> {
    #[must_use]
    pub fn new(
        side: PortSide,
        medium: M,
        temperature: f64,
        pressure: f64,
        mass_flow_rate: f64,
    ) -> Self {
        Self {
            side,
            state: FluidPortState::new(medium, temperature, pressure, mass_flow_rate),
        }
    }

    #[must_use]
    pub fn side(&self) -> PortSide {
        self.side
    }

    #[must_use]
    pub fn state(&self) -> &FluidPortState<M> {
        &self.state
    }

    #[must_use]
    pub fn mut_state(&mut self) -> &mut FluidPortState<M> {
        &mut self.state
    }
}

impl<M: FluidMedium> FluidPort for OutletPort<M> {
    type Medium = M;

    fn direction(&self) -> PortDirection {
        PortDirection::Outlet
    }

    fn side(&self) -> PortSide {
        self.side
    }

    fn medium(&self) -> Medium {
        self.state.medium.medium()
    }

    fn temperature(&self) -> f64 {
        self.state.temperature()
    }

    fn pressure(&self) -> f64 {
        self.state.pressure()
    }

    fn mass_flow_rate(&self) -> f64 {
        self.state.mass_flow_rate()
    }

    fn set_temperature(&mut self, temperature: f64) -> Result<(), PortError> {
        self.state.set_temperature(temperature)
    }

    fn set_pressure(&mut self, pressure: f64) -> Result<(), PortError> {
        self.state.set_pressure(pressure)
    }

    fn set_mass_flow_rate(&mut self, mass_flow_rate: f64) -> Result<(), PortError> {
        self.state.set_mass_flow_rate(mass_flow_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::{AirMedium, WaterMedium};

    #[test]
    fn test_inlet_port_creation() {
        let water = WaterMedium;
        let port = InletPort::new(PortSide::Supply, water, 293.15, 101325.0, 0.5);
        assert_eq!(port.direction(), PortDirection::Inlet);
        assert_eq!(port.side(), PortSide::Supply);
        assert_eq!(port.medium(), Medium::Water);
        assert!((port.temperature() - 293.15).abs() < 0.01);
    }

    #[test]
    fn test_outlet_port_creation() {
        let air = AirMedium;
        let port = OutletPort::new(PortSide::Return, air, 303.15, 101325.0, 0.3);
        assert_eq!(port.direction(), PortDirection::Outlet);
        assert_eq!(port.side(), PortSide::Return);
        assert_eq!(port.medium(), Medium::Air);
    }

    #[test]
    fn test_port_validate() {
        let water = WaterMedium;
        let port = InletPort::new(PortSide::Supply, water, 293.15, 101325.0, 0.5);
        assert!(port.validate().is_ok());
    }

    #[test]
    fn test_port_validate_negative_flow() {
        let water = WaterMedium;
        let port = InletPort::new(PortSide::Supply, water, 293.15, 101325.0, -0.5);
        assert!(port.validate().is_err());
    }

    #[test]
    fn test_port_set_temperature() {
        let water = WaterMedium;
        let mut port = InletPort::new(PortSide::Supply, water, 293.15, 101325.0, 0.5);
        port.set_temperature(303.15).unwrap();
        assert!((port.temperature() - 303.15).abs() < 0.01);
    }

    #[test]
    fn test_port_properties() {
        let water = WaterMedium;
        let port = InletPort::new(PortSide::Supply, water, 293.15, 101325.0, 0.5);
        let props = port.state.properties().unwrap();
        assert!((props.temperature - 293.15).abs() < 0.01);
        assert!((props.mass_flow_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_compile_time_type_safety() {
        let water = WaterMedium;
        let air = AirMedium;

        let _water_port: InletPort<WaterMedium> =
            InletPort::new(PortSide::Supply, water, 293.15, 101325.0, 0.5);
        let _air_port: InletPort<AirMedium> =
            InletPort::new(PortSide::Supply, air, 293.15, 101325.0, 0.3);
    }
}
