//! Component types for HVAC equipment ECS.
//!
//! Each component type stores scalar f64 values in SoA layout (contiguous arrays).

/// Physical state component: temperature, pressure, mass_flowrate, enthalpy
///
/// Used by MassBalanceSystem and HeatTransferSystem to track thermodynamic state.
#[derive(Clone, Debug)]
pub struct PhysicalState {
    /// Temperature in Celsius
    pub temperature: f64,
    /// Pressure in Pascals
    pub pressure: f64,
    /// Mass flow rate in kg/s
    pub mass_flowrate: f64,
    /// Enthalpy in J/kg
    pub enthalpy: f64,
}

impl PhysicalState {
    pub fn new(temperature: f64, pressure: f64, mass_flowrate: f64, enthalpy: f64) -> Self {
        Self {
            temperature,
            pressure,
            mass_flowrate,
            enthalpy,
        }
    }

    pub fn default_for(kind: super::EquipmentKind) -> Self {
        match kind {
            super::EquipmentKind::Chiller | super::EquipmentKind::CoilCooling => {
                Self::new(7.0, 101_325.0, 0.5, 2500.0)
            }
            super::EquipmentKind::Boiler | super::EquipmentKind::CoilHeating => {
                Self::new(60.0, 101_325.0, 0.5, 280_000.0)
            }
            super::EquipmentKind::VavBox
            | super::EquipmentKind::Damper
            | super::EquipmentKind::Fan => Self::new(24.0, 101_325.0, 0.2, 2800.0),
            super::EquipmentKind::CoolingTower => Self::new(25.0, 101_325.0, 1.0, 83_000.0),
            super::EquipmentKind::Pump => Self::new(20.0, 101_325.0, 0.5, 84_000.0),
        }
    }
}

impl Default for PhysicalState {
    fn default() -> Self {
        Self::new(20.0, 101_325.0, 0.0, 0.0)
    }
}

/// Equipment parameters component: rated_capacity, efficiency, nominal_flowrate, control_type
///
/// Stores rated/design parameters for HVAC equipment.
#[derive(Clone, Debug)]
pub struct EquipmentParameters {
    /// Rated capacity in Watts
    pub rated_capacity: f64,
    /// Efficiency (COP for chillers, thermal efficiency for boilers)
    pub efficiency: f64,
    /// Nominal flow rate in kg/s
    pub nominal_flowrate: f64,
    /// Control type enum encoded as f64 (0=constant, 1=variable, 2=modulating)
    pub control_type: f64,
}

impl EquipmentParameters {
    pub fn new(rated_capacity: f64, efficiency: f64, nominal_flowrate: f64) -> Self {
        Self {
            rated_capacity,
            efficiency,
            nominal_flowrate,
            control_type: 1.0, // variable speed default
        }
    }

    pub fn chiller(rated_capacity: f64, cop: f64) -> Self {
        Self::new(rated_capacity, cop, 0.5)
    }

    pub fn boiler(rated_capacity: f64, eta: f64) -> Self {
        Self::new(rated_capacity, eta, 0.3)
    }

    pub fn pump(rated_flow: f64, rated_head: f64, rated_power: f64) -> Self {
        Self::new(rated_power, rated_head / rated_flow, rated_flow)
    }

    pub fn vav_box(rated_demand: f64, k_factor: f64) -> Self {
        Self::new(rated_demand, k_factor, 0.2)
    }
}

impl Default for EquipmentParameters {
    fn default() -> Self {
        Self::new(100_000.0, 5.0, 0.5)
    }
}

/// Control signal component: setpoint, position, on_off
///
/// Used by ControlLoopSystem for feedback control.
#[derive(Clone, Debug)]
pub struct ControlSignal {
    /// Setpoint value (temperature, pressure, etc.)
    pub setpoint: f64,
    /// Position of actuator (0.0 to 1.0 for dampers, valves)
    pub position: f64,
    /// On/off state (1.0 = on, 0.0 = off)
    pub on_off: f64,
}

impl ControlSignal {
    pub fn new(setpoint: f64, position: f64, on_off: bool) -> Self {
        Self {
            setpoint,
            position: position.clamp(0.0, 1.0),
            on_off: if on_off { 1.0 } else { 0.0 },
        }
    }

    pub fn is_on(&self) -> bool {
        self.on_off > 0.5
    }
}

impl Default for ControlSignal {
    fn default() -> Self {
        Self::new(20.0, 0.5, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_state_new_preserves_fields() {
        let s = PhysicalState::new(35.0, 200_000.0, 0.42, 1_234.5);
        assert_eq!(s.temperature, 35.0);
        assert_eq!(s.pressure, 200_000.0);
        assert_eq!(s.mass_flowrate, 0.42);
        assert_eq!(s.enthalpy, 1_234.5);
    }

    #[test]
    fn physical_state_default_is_zeroed_flow_and_enthalpy() {
        let s = PhysicalState::default();
        assert_eq!(s.temperature, 20.0);
        assert_eq!(s.pressure, 101_325.0);
        assert_eq!(s.mass_flowrate, 0.0);
        assert_eq!(s.enthalpy, 0.0);
    }

    #[test]
    fn physical_state_default_for_covers_every_kind() {
        let kinds = [
            super::super::EquipmentKind::Chiller,
            super::super::EquipmentKind::Boiler,
            super::super::EquipmentKind::CoolingTower,
            super::super::EquipmentKind::Pump,
            super::super::EquipmentKind::VavBox,
            super::super::EquipmentKind::Damper,
            super::super::EquipmentKind::Fan,
            super::super::EquipmentKind::CoilHeating,
            super::super::EquipmentKind::CoilCooling,
        ];
        for k in kinds {
            let s = PhysicalState::default_for(k);
            // Field invariant: default must always populate non-NaN finite values.
            for field in [s.temperature, s.pressure, s.mass_flowrate, s.enthalpy] {
                assert!(
                    field.is_finite(),
                    "PhysicalState::default_for({k:?}) produced non-finite field {field}"
                );
            }
            assert!(s.pressure > 0.0, "pressure must be positive for {k:?}");
        }
    }

    #[test]
    fn physical_state_chiller_default_is_cold() {
        let s = PhysicalState::default_for(super::super::EquipmentKind::Chiller);
        // Chiller evaporator side runs cold (chilled water).
        assert!(s.temperature < 15.0, "chiller default T should be cold");
    }

    #[test]
    fn physical_state_boiler_default_is_hot() {
        let s = PhysicalState::default_for(super::super::EquipmentKind::Boiler);
        // Boiler supply side is hot.
        assert!(s.temperature > 40.0, "boiler default T should be hot");
    }

    #[test]
    fn equipment_parameters_chiller_constructor() {
        let p = EquipmentParameters::chiller(120_000.0, 4.8);
        assert_eq!(p.rated_capacity, 120_000.0);
        assert_eq!(p.efficiency, 4.8);
        assert!(p.nominal_flowrate > 0.0);
        // Default control_type is variable speed (1.0).
        assert_eq!(p.control_type, 1.0);
    }

    #[test]
    fn equipment_parameters_boiler_constructor() {
        let p = EquipmentParameters::boiler(60_000.0, 0.85);
        assert_eq!(p.rated_capacity, 60_000.0);
        assert_eq!(p.efficiency, 0.85);
        assert_eq!(p.control_type, 1.0);
    }

    #[test]
    fn equipment_parameters_pump_constructor() {
        let p = EquipmentParameters::pump(0.5, 100_000.0, 5_000.0);
        assert_eq!(p.rated_capacity, 5_000.0);
        // efficiency is rated_head / rated_flow = 100_000 / 0.5 = 200_000.
        assert_eq!(p.efficiency, 200_000.0);
        assert_eq!(p.nominal_flowrate, 0.5);
    }

    #[test]
    fn equipment_parameters_vav_box_constructor() {
        let p = EquipmentParameters::vav_box(5_000.0, 0.05);
        assert_eq!(p.rated_capacity, 5_000.0);
        assert_eq!(p.efficiency, 0.05);
        assert!(p.nominal_flowrate > 0.0);
    }

    #[test]
    fn equipment_parameters_default_has_positive_capacity() {
        let p = EquipmentParameters::default();
        assert!(p.rated_capacity > 0.0);
        assert!(p.efficiency > 0.0);
        assert!(p.nominal_flowrate >= 0.0);
    }

    #[test]
    fn control_signal_new_clamps_position() {
        let s = ControlSignal::new(22.0, 2.0, true);
        assert_eq!(s.position, 1.0, "position above 1.0 must clamp to 1.0");
        let s = ControlSignal::new(22.0, -0.5, true);
        assert_eq!(s.position, 0.0, "position below 0.0 must clamp to 0.0");
        assert_eq!(s.setpoint, 22.0);
        assert_eq!(s.on_off, 1.0);
    }

    #[test]
    fn control_signal_is_on_matches_threshold() {
        assert!(ControlSignal::new(20.0, 0.5, true).is_on());
        assert!(!ControlSignal::new(20.0, 0.5, false).is_on());
        // Custom on_off value still obeys > 0.5 threshold.
        let s_high = ControlSignal {
            on_off: 0.6,
            ..ControlSignal::default()
        };
        assert!(s_high.is_on());
        let s_low = ControlSignal {
            on_off: 0.4,
            ..ControlSignal::default()
        };
        assert!(!s_low.is_on());
    }
}
