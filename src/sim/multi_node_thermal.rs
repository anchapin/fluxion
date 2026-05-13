//! Multi-Node Thermal Model Data Structures (Phase 6)
//!
//! This module defines the data structures for the 9R4C multi-node thermal model
//! used for heavy mass buildings (Case 900+ series, Issue #715).
//!
//! The 9R4C model separates thermal mass into 4 nodes:
//! - Wall node: exterior wall thermal mass
//! - Roof node: roof/ceiling thermal mass
//! - Floor node: floor slab thermal mass
//! - Internal node: furniture, partitions, internal mass

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ThermalMassNode {
    pub temperature: f64,
    pub capacitance: f64,
    pub h_tr_ms: f64,
    pub h_tr_em: f64,
    pub h_tr_me: f64,
    pub heat_flux_cumulative: f64,
}

impl ThermalMassNode {
    pub fn new(temperature: f64, capacitance: f64, h_tr_ms: f64, h_tr_em: f64) -> Self {
        Self {
            temperature,
            capacitance,
            h_tr_ms,
            h_tr_em,
            h_tr_me: 0.0,
            heat_flux_cumulative: 0.0,
        }
    }

    pub fn with_h_tr_me(mut self, h_tr_me: f64) -> Self {
        self.h_tr_me = h_tr_me;
        self
    }

    pub fn update_heat_flux(&mut self, heat_flux: f64, dt: f64) {
        self.heat_flux_cumulative += heat_flux * dt;
    }

    pub fn reset_heat_flux(&mut self) {
        self.heat_flux_cumulative = 0.0;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MultiNodeThermalMass {
    pub wall: ThermalMassNode,
    pub roof: ThermalMassNode,
    pub floor: ThermalMassNode,
    pub internal: ThermalMassNode,
}

impl MultiNodeThermalMass {
    pub fn new(
        wall: ThermalMassNode,
        roof: ThermalMassNode,
        floor: ThermalMassNode,
        internal: ThermalMassNode,
    ) -> Self {
        Self {
            wall,
            roof,
            floor,
            internal,
        }
    }

    pub fn wall_mut(&mut self) -> &mut ThermalMassNode {
        &mut self.wall
    }

    pub fn roof_mut(&mut self) -> &mut ThermalMassNode {
        &mut self.roof
    }

    pub fn floor_mut(&mut self) -> &mut ThermalMassNode {
        &mut self.floor
    }

    pub fn internal_mut(&mut self) -> &mut ThermalMassNode {
        &mut self.internal
    }

    pub fn reset_all_heat_flux(&mut self) {
        self.wall.reset_heat_flux();
        self.roof.reset_heat_flux();
        self.floor.reset_heat_flux();
        self.internal.reset_heat_flux();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiNodeModelType {
    FiveR1C,
    SixR2C,
    EightR3C,
    NineRFourC,
}

impl Default for MultiNodeModelType {
    fn default() -> Self {
        MultiNodeModelType::FiveR1C
    }
}
