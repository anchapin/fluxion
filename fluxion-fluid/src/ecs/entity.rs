//! Entity types for HVAC equipment ECS.

/// Entity ID wrapper for HVAC equipment.
///
/// Wraps a u64 index into the component storage arrays.
/// The inner value is the entity's index in the ECS storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EquipmentEntity(u64);

impl EquipmentEntity {
    /// Create a new entity with the given index.
    pub fn new(index: u64) -> Self {
        Self(index)
    }

    /// Get the raw index value.
    pub fn index(&self) -> u64 {
        self.0
    }
}

impl From<u64> for EquipmentEntity {
    fn from(index: u64) -> Self {
        Self(index)
    }
}

impl From<EquipmentEntity> for u64 {
    fn from(entity: EquipmentEntity) -> Self {
        entity.0
    }
}

/// Kind of HVAC equipment entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EquipmentKind {
    Chiller,
    Boiler,
    CoolingTower,
    Pump,
    VavBox,
    Damper,
    Fan,
    CoilHeating,
    CoilCooling,
}

impl EquipmentKind {
    /// Returns the number of scalar fields needed for equipment parameters.
    pub fn params_len(&self) -> usize {
        match self {
            EquipmentKind::Chiller => 2,      // rated_capacity, efficiency
            EquipmentKind::Boiler => 2,       // rated_capacity, efficiency
            EquipmentKind::CoolingTower => 2, // rated_capacity, efficiency
            EquipmentKind::Pump => 3,         // rated_flow, rated_head, rated_power
            EquipmentKind::VavBox => 2,       // rated_demand, k_factor
            EquipmentKind::Damper => 1,       // k_valve
            EquipmentKind::Fan => 3,          // rated_flow, rated_pressure, rated_power
            EquipmentKind::CoilHeating => 2,  // rated_capacity, rated_flow
            EquipmentKind::CoilCooling => 2,  // rated_capacity, rated_flow
        }
    }
}
