//! Medium types for fluid systems.
//!
//! Defines the physical mediums (Air, Water, Refrigerant, Steam) that can flow
//! through fluid ports. Each medium has associated thermodynamic properties.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Air {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Water {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Refrigerant {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Steam {}

pub trait Medium: Sized + Send + Sync + 'static {
    type PotentialVars: Clone + Send + Sync + 'static;
    type ConservedVars: Clone + Send + Sync + 'static;
}

impl Medium for Air {
    type PotentialVars = AirPotentialVars;
    type ConservedVars = AirConservedVars;
}

impl Medium for Water {
    type PotentialVars = WaterPotentialVars;
    type ConservedVars = WaterConservedVars;
}

impl Medium for Refrigerant {
    type PotentialVars = RefrigerantPotentialVars;
    type ConservedVars = RefrigerantConservedVars;
}

impl Medium for Steam {
    type PotentialVars = SteamPotentialVars;
    type ConservedVars = SteamConservedVars;
}

pub trait CompatibleWith<Other: Medium> {
    fn is_compatible_with() -> bool;
}

impl<T: Medium> CompatibleWith<T> for T {
    fn is_compatible_with() -> bool {
        true
    }
}

impl CompatibleWith<Water> for Air {
    fn is_compatible_with() -> bool {
        false
    }
}

impl CompatibleWith<Air> for Water {
    fn is_compatible_with() -> bool {
        false
    }
}

impl CompatibleWith<Refrigerant> for Air {
    fn is_compatible_with() -> bool {
        false
    }
}

impl CompatibleWith<Air> for Refrigerant {
    fn is_compatible_with() -> bool {
        false
    }
}

impl CompatibleWith<Steam> for Air {
    fn is_compatible_with() -> bool {
        false
    }
}

impl CompatibleWith<Air> for Steam {
    fn is_compatible_with() -> bool {
        false
    }
}

impl CompatibleWith<Steam> for Water {
    fn is_compatible_with() -> bool {
        false
    }
}

impl CompatibleWith<Water> for Steam {
    fn is_compatible_with() -> bool {
        false
    }
}

impl CompatibleWith<Refrigerant> for Water {
    fn is_compatible_with() -> bool {
        false
    }
}

impl CompatibleWith<Water> for Refrigerant {
    fn is_compatible_with() -> bool {
        false
    }
}

impl CompatibleWith<Steam> for Refrigerant {
    fn is_compatible_with() -> bool {
        false
    }
}

impl CompatibleWith<Refrigerant> for Steam {
    fn is_compatible_with() -> bool {
        false
    }
}

#[derive(Debug, Clone, Default)]
pub struct AirPotentialVars {
    pub t_db: f32,
    pub t_wb: f32,
    pub omega: f32,
}

#[derive(Debug, Clone, Default)]
pub struct AirConservedVars {
    pub m_dot_da: f32,
}

#[derive(Debug, Clone, Default)]
pub struct WaterPotentialVars {
    pub temperature: f32,
    pub pressure: f32,
}

#[derive(Debug, Clone, Default)]
pub struct WaterConservedVars {
    pub mass_flow: f32,
    pub density: f32,
}

#[derive(Debug, Clone, Default)]
pub struct RefrigerantPotentialVars {
    pub pressure: f32,
    pub quality: f32,
}

#[derive(Debug, Clone, Default)]
pub struct RefrigerantConservedVars {
    pub mass_flow: f32,
}

#[derive(Debug, Clone, Default)]
pub struct SteamPotentialVars {
    pub temperature: f32,
    pub pressure: f32,
}

#[derive(Debug, Clone, Default)]
pub struct SteamConservedVars {
    pub mass_flow: f32,
    pub enthalpy: f32,
}
