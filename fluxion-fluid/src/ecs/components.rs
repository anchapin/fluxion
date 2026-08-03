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
