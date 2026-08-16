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

#[cfg(test)]
mod tests {
    use super::*;

    fn compat<A, B>() -> bool
    where
        A: Medium + CompatibleWith<B>,
        B: Medium,
    {
        <A as CompatibleWith<B>>::is_compatible_with()
    }

    #[test]
    fn air_is_not_compatible_with_water() {
        assert!(!compat::<Air, Water>());
        assert!(!compat::<Water, Air>());
    }

    #[test]
    fn air_is_not_compatible_with_refrigerant_or_steam() {
        assert!(!compat::<Air, Refrigerant>());
        assert!(!compat::<Refrigerant, Air>());
        assert!(!compat::<Air, Steam>());
        assert!(!compat::<Steam, Air>());
    }

    #[test]
    fn water_is_not_compatible_with_refrigerant_or_steam() {
        assert!(!compat::<Water, Refrigerant>());
        assert!(!compat::<Refrigerant, Water>());
        assert!(!compat::<Water, Steam>());
        assert!(!compat::<Steam, Water>());
    }

    #[test]
    fn refrigerant_is_not_compatible_with_steam() {
        assert!(!compat::<Refrigerant, Steam>());
        assert!(!compat::<Steam, Refrigerant>());
    }

    #[test]
    fn same_medium_is_always_compatible() {
        // The blanket `impl<T: Medium> CompatibleWith<T> for T` covers all
        // marker types. Verify it actually applies to every variant.
        assert!(compat::<Air, Air>());
        assert!(compat::<Water, Water>());
        assert!(compat::<Refrigerant, Refrigerant>());
        assert!(compat::<Steam, Steam>());
    }

    #[test]
    fn marker_types_are_zero_sized_and_distinct() {
        // Marker types must be uninhabited enums (zero-sized) and remain distinct.
        assert_eq!(std::mem::size_of::<Air>(), 0);
        assert_eq!(std::mem::size_of::<Water>(), 0);
        assert_eq!(std::mem::size_of::<Refrigerant>(), 0);
        assert_eq!(std::mem::size_of::<Steam>(), 0);
        // Distinctness: a `fn` pointer to each type is a unique ZST.
        fn id<T>() {}
        assert_ne!(
            id::<Air> as *const () as usize,
            id::<Water> as *const () as usize
        );
        assert_ne!(
            id::<Refrigerant> as *const () as usize,
            id::<Steam> as *const () as usize
        );
    }

    #[test]
    fn medium_trait_associates_potential_and_conserved_vars() {
        // The `Medium` trait must surface a non-trivial pair of associated types.
        // This locks down the type-level wiring used by `FluidPort` consumers.
        fn assert_associated_types<M: Medium>() {}
        assert_associated_types::<Air>();
        assert_associated_types::<Water>();
        assert_associated_types::<Refrigerant>();
        assert_associated_types::<Steam>();
    }

    #[test]
    fn air_potential_vars_default_is_zeroed() {
        let v = AirPotentialVars::default();
        assert_eq!(v.t_db, 0.0);
        assert_eq!(v.t_wb, 0.0);
        assert_eq!(v.omega, 0.0);
    }

    #[test]
    fn water_potential_vars_default_is_zeroed() {
        let v = WaterPotentialVars::default();
        assert_eq!(v.temperature, 0.0);
        assert_eq!(v.pressure, 0.0);
    }
}
