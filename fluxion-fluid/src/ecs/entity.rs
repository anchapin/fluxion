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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_index_round_trip() {
        for raw in [0u64, 1, 7, 42, u64::MAX] {
            let e = EquipmentEntity::new(raw);
            assert_eq!(e.index(), raw);
        }
    }

    #[test]
    fn entity_from_u64_conversions() {
        let e: EquipmentEntity = 17u64.into();
        assert_eq!(e, EquipmentEntity::new(17));
        let back: u64 = e.into();
        assert_eq!(back, 17u64);
    }

    #[test]
    fn entity_equality_and_hash() {
        let a = EquipmentEntity::new(3);
        let b = EquipmentEntity::new(3);
        let c = EquipmentEntity::new(4);
        assert_eq!(a, b);
        assert_ne!(a, c);
        // Hash consistency for use in HashSet/HashMap.
        let mut set = std::collections::HashSet::new();
        set.insert(a);
        set.insert(b);
        set.insert(c);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn entity_is_copy_and_clone() {
        let e = EquipmentEntity::new(5);
        let copy = e;
        let clone2 = e;
        assert_eq!(copy, clone2);
    }

    #[test]
    fn equipment_kind_variants_are_distinct() {
        let all = [
            EquipmentKind::Chiller,
            EquipmentKind::Boiler,
            EquipmentKind::CoolingTower,
            EquipmentKind::Pump,
            EquipmentKind::VavBox,
            EquipmentKind::Damper,
            EquipmentKind::Fan,
            EquipmentKind::CoilHeating,
            EquipmentKind::CoilCooling,
        ];
        // Pairwise distinctness: count how many entries have no later twin.
        let unique = all
            .iter()
            .enumerate()
            .filter(|(i, a)| !all.iter().skip(i + 1).any(|b| **a == *b))
            .count();
        assert_eq!(
            unique,
            all.len(),
            "every EquipmentKind variant must be unique"
        );
    }

    #[test]
    fn params_len_field_invariant() {
        // Field invariant documented in `params_len`: every kind declares between 1
        // and 3 parameters. If a future contributor adds a parameter, the contract
        // is "scalar fields" — assert the documented lower and upper bounds.
        let kinds = [
            EquipmentKind::Chiller,
            EquipmentKind::Boiler,
            EquipmentKind::CoolingTower,
            EquipmentKind::Pump,
            EquipmentKind::VavBox,
            EquipmentKind::Damper,
            EquipmentKind::Fan,
            EquipmentKind::CoilHeating,
            EquipmentKind::CoilCooling,
        ];
        for k in kinds {
            let n = k.params_len();
            assert!(
                (1..=3).contains(&n),
                "EquipmentKind::{k:?} has params_len={n} outside documented [1, 3] range"
            );
        }
        // Damper is the unique single-parameter kind (k_valve).
        assert_eq!(EquipmentKind::Damper.params_len(), 1);
    }
}
