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

/// How the air node couples to the per-surface mass nodes in the 9R4C network.
///
/// This enum is the switch behind Issue #1281's fix proposal. The two formulations
/// give materially different air-temperature predictions because they treat the
/// mass-to-air coupling network differently.
///
/// # Background — Issue #1281
///
/// `docs/KNOWN_ISSUES.md` LIMIT-05 UPDATE (Phase 36) identifies the additive
/// `h_ms_total = h_ms_wall + h_ms_roof + h_ms_floor` formulation as suspect: it
/// treats the three per-surface mass nodes as if their conductances sum at the
/// shared interior air node, when physically each surface has its own T_s_k
/// with h_is feedback.
///
/// Python verification (`.agents/results/issue-1281-python-verification.py`)
/// confirms the additive formulation overcounts the effective mass-to-air
/// coupling by ~32.7 % for ASHRAE 140 Case 900 (h_ms_total=127.3 W/K vs
/// h_path_total=96.0 W/K). However, the cooling-load direction is opposite to
/// the issue hypothesis: switching to `ParallelResistance` produces a *lower*
/// peak cooling demand (3.27 kW vs 4.10 kW) because the air node receives less
/// heat from the masses.
///
/// The actual ASHRAE 140 cooling underestimate is documented in
/// `docs/investigations/issue-1280-ctf-peak-load.md` §4 as a **roof solar
/// under-counting** issue (separate follow-up). The `ParallelResistance` mode
/// is still the more physically correct formulation and ships as the
/// architecturally-improved 9R4C coupling; it does not by itself close the
/// ASHRAE 140 cooling gap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MassAirCouplingMode {
    /// Original 9R4C formulation:
    /// `T_s = (Σ h_ms_k × T_m_k) / Σ h_ms_k`  (shared surface, conductance-weighted mean)
    /// `T_air = (h_tr_is × T_s + h_ve × T_out + phi_ia) / (h_tr_is + h_ve)`
    ///
    /// Backward-compatible default. Matches `multi_node_solver.rs::step_*`
    /// and `compute_zone_air_temperature` prior to Issue #1281.
    AdditiveSum,

    /// Per-surface series paths:
    /// Each surface has its own steady-state `T_s_k = (h_ms_k × T_m_k + h_tr_is × T_air)
    ///                                        / (h_ms_k + h_tr_is)`.
    /// The air node sees the parallel combination:
    /// `h_path_k = h_ms_k × h_tr_is / (h_ms_k + h_tr_is)`  (series combination of mass→surface→air)
    /// `T_air = (Σ h_path_k × T_m_k + h_ve × T_out + phi_ia) / (Σ h_path_k + h_ve)`.
    ///
    /// Eliminates the additive-sum overcounting of the mass-to-air coupling.
    /// See `.agents/results/issue-1281-python-verification.py` for derivation
    /// and parameter sensitivity.
    ParallelResistance,
}

impl Default for MassAirCouplingMode {
    fn default() -> Self {
        MassAirCouplingMode::AdditiveSum
    }
}
