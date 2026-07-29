//! Fluid port trait and implementations.
//!
//! Defines the `FluidPort` trait that all fluid ports must implement,
//! along with concrete port types for different mediums.

use crate::mediums::{
    Air, AirConservedVars, AirPotentialVars, CompatibleWith, Medium, Refrigerant,
    RefrigerantConservedVars, RefrigerantPotentialVars, Steam, SteamConservedVars,
    SteamPotentialVars, Water, WaterConservedVars, WaterPotentialVars,
};
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum PortError {
    #[error("Incompatible medium connection")]
    IncompatibleMedium,
    #[error("Boundary condition not set")]
    BoundaryConditionNotSet,
    #[error("Conservation violation: {0}")]
    ConservationViolation(String),
}

pub type PortResult<T> = Result<T, PortError>;

pub trait FluidPort: Send + Sync {
    type Medium: Medium + 'static;
    type State: Clone + Send + Sync + 'static;

    fn potential_vars(&self) -> <Self::Medium as Medium>::PotentialVars;
    fn conserved_vars(&self) -> <Self::Medium as Medium>::ConservedVars;
    fn set_boundary_conditions(&mut self, bc: BoundaryConditions);
    fn compute_residual(&self, eq: &mut EquationSystem);
}

#[derive(Debug, Clone, Default)]
pub struct BoundaryConditions {
    pub is_prescribed: bool,
    pub prescribed_mass_flow: Option<f32>,
    pub prescribed_enthalpy: Option<f32>,
    pub prescribed_temperature: Option<f32>,
    pub prescribed_pressure: Option<f32>,
}

impl BoundaryConditions {
    pub fn prescribed_flow(flow: f32) -> Self {
        Self {
            is_prescribed: true,
            prescribed_mass_flow: Some(flow),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct EquationSystem {
    pub equations: Vec<f32>,
}

impl EquationSystem {
    pub fn new(size: usize) -> Self {
        Self {
            equations: vec![0.0; size],
        }
    }
}

#[derive(Debug, Clone)]
pub struct AirPort {
    state: AirPortState,
}

#[derive(Debug, Clone)]
pub struct AirPortState {
    potential: AirPotentialVars,
    conserved: AirConservedVars,
    boundary: BoundaryConditions,
}

impl Default for AirPort {
    fn default() -> Self {
        Self {
            state: AirPortState {
                potential: AirPotentialVars::default(),
                conserved: AirConservedVars::default(),
                boundary: BoundaryConditions::default(),
            },
        }
    }
}

impl AirPort {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_state(potential: AirPotentialVars, conserved: AirConservedVars) -> Self {
        Self {
            state: AirPortState {
                potential,
                conserved,
                boundary: BoundaryConditions::default(),
            },
        }
    }
}

impl FluidPort for AirPort {
    type Medium = Air;
    type State = AirPortState;

    fn potential_vars(&self) -> AirPotentialVars {
        self.state.potential.clone()
    }

    fn conserved_vars(&self) -> AirConservedVars {
        self.state.conserved.clone()
    }

    fn set_boundary_conditions(&mut self, bc: BoundaryConditions) {
        self.state.boundary = bc;
    }

    fn compute_residual(&self, _eq: &mut EquationSystem) {}
}

#[derive(Debug, Clone)]
pub struct HydronicPort {
    state: HydronicPortState,
}

#[derive(Debug, Clone)]
pub struct HydronicPortState {
    potential: WaterPotentialVars,
    conserved: WaterConservedVars,
    boundary: BoundaryConditions,
}

impl Default for HydronicPort {
    fn default() -> Self {
        Self {
            state: HydronicPortState {
                potential: WaterPotentialVars::default(),
                conserved: WaterConservedVars::default(),
                boundary: BoundaryConditions::default(),
            },
        }
    }
}

impl HydronicPort {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_state(potential: WaterPotentialVars, conserved: WaterConservedVars) -> Self {
        Self {
            state: HydronicPortState {
                potential,
                conserved,
                boundary: BoundaryConditions::default(),
            },
        }
    }
}

impl FluidPort for HydronicPort {
    type Medium = Water;
    type State = HydronicPortState;

    fn potential_vars(&self) -> WaterPotentialVars {
        self.state.potential.clone()
    }

    fn conserved_vars(&self) -> WaterConservedVars {
        self.state.conserved.clone()
    }

    fn set_boundary_conditions(&mut self, bc: BoundaryConditions) {
        self.state.boundary = bc;
    }

    fn compute_residual(&self, _eq: &mut EquationSystem) {}
}

#[derive(Debug, Clone)]
pub struct RefrigerantPort {
    state: RefrigerantPortState,
}

#[derive(Debug, Clone)]
pub struct RefrigerantPortState {
    potential: RefrigerantPotentialVars,
    conserved: RefrigerantConservedVars,
    boundary: BoundaryConditions,
}

impl Default for RefrigerantPort {
    fn default() -> Self {
        Self {
            state: RefrigerantPortState {
                potential: RefrigerantPotentialVars::default(),
                conserved: RefrigerantConservedVars::default(),
                boundary: BoundaryConditions::default(),
            },
        }
    }
}

impl RefrigerantPort {
    pub fn new() -> Self {
        Self::default()
    }
}

impl FluidPort for RefrigerantPort {
    type Medium = Refrigerant;
    type State = RefrigerantPortState;

    fn potential_vars(&self) -> RefrigerantPotentialVars {
        self.state.potential.clone()
    }

    fn conserved_vars(&self) -> RefrigerantConservedVars {
        self.state.conserved.clone()
    }

    fn set_boundary_conditions(&mut self, bc: BoundaryConditions) {
        self.state.boundary = bc;
    }

    fn compute_residual(&self, _eq: &mut EquationSystem) {}
}

#[derive(Debug, Clone)]
pub struct SteamPort {
    state: SteamPortState,
}

#[derive(Debug, Clone)]
pub struct SteamPortState {
    potential: SteamPotentialVars,
    conserved: SteamConservedVars,
    boundary: BoundaryConditions,
}

impl Default for SteamPort {
    fn default() -> Self {
        Self {
            state: SteamPortState {
                potential: SteamPotentialVars::default(),
                conserved: SteamConservedVars::default(),
                boundary: BoundaryConditions::default(),
            },
        }
    }
}

impl SteamPort {
    pub fn new() -> Self {
        Self::default()
    }
}

impl FluidPort for SteamPort {
    type Medium = Steam;
    type State = SteamPortState;

    fn potential_vars(&self) -> SteamPotentialVars {
        self.state.potential.clone()
    }

    fn conserved_vars(&self) -> SteamConservedVars {
        self.state.conserved.clone()
    }

    fn set_boundary_conditions(&mut self, bc: BoundaryConditions) {
        self.state.boundary = bc;
    }

    fn compute_residual(&self, _eq: &mut EquationSystem) {}
}

pub fn connect<P1, P2>(_p1: &mut P1, _p2: &mut P2) -> PortResult<()>
where
    P1: FluidPort,
    P2: FluidPort,
    P1::Medium: CompatibleWith<P2::Medium>,
{
    if !<P1::Medium as CompatibleWith<P2::Medium>>::is_compatible_with() {
        return Err(PortError::IncompatibleMedium);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_air_port_creation() {
        let port = AirPort::new();
        assert!(matches!(port.potential_vars(), AirPotentialVars { .. }));
    }

    #[test]
    fn test_hydronic_port_creation() {
        let port = HydronicPort::new();
        assert!(matches!(port.potential_vars(), WaterPotentialVars { .. }));
    }

    #[test]
    fn test_refrigerant_port_creation() {
        let port = RefrigerantPort::new();
        assert!(matches!(
            port.potential_vars(),
            RefrigerantPotentialVars { .. }
        ));
    }

    #[test]
    fn test_steam_port_creation() {
        let port = SteamPort::new();
        assert!(matches!(port.potential_vars(), SteamPotentialVars { .. }));
    }

    #[test]
    fn test_same_medium_compatible() {
        assert!(<Air as CompatibleWith<Air>>::is_compatible_with());
        assert!(<Water as CompatibleWith<Water>>::is_compatible_with());
    }

    #[test]
    fn test_different_mediums_incompatible() {
        assert!(!<Air as CompatibleWith<Water>>::is_compatible_with());
        assert!(!<Water as CompatibleWith<Air>>::is_compatible_with());
        assert!(!<Air as CompatibleWith<Refrigerant>>::is_compatible_with());
        assert!(!<Steam as CompatibleWith<Water>>::is_compatible_with());
    }

    #[test]
    fn test_connect_same_medium_succeeds() {
        let mut port1 = AirPort::new();
        let mut port2 = AirPort::new();
        assert!(connect(&mut port1, &mut port2).is_ok());
    }

    #[test]
    fn test_connect_different_mediums_fails() {
        let mut air_port = AirPort::new();
        let mut water_port = HydronicPort::new();
        assert!(connect(&mut air_port, &mut water_port).is_err());
    }

    #[test]
    fn test_boundary_conditions() {
        let mut port = AirPort::new();
        let bc = BoundaryConditions::prescribed_flow(0.5);
        port.set_boundary_conditions(bc);
        assert!(port.state.boundary.is_prescribed);
        assert_eq!(port.state.boundary.prescribed_mass_flow, Some(0.5));
    }
}
