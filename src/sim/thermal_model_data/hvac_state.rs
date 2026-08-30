//! HVAC state — equipment, controllers, peak/energy output, scratch, tracer.
//!
//! Extracted from `ThermalModelData` (Issue #2878) so the wrapper-level clone
//! visits exactly 6 fields (one per sub-struct). All peak-power, annual-energy,
//! per-zone energy, scratch-pool, tracer, and `last_phi_*` diagnostic fields
//! are owned here because they are written/read by HVAC control and the
//! 5R1C/9R4C physics hot loops.

use super::{ContinuousTensor, VectorField};
use crate::sim::hvac::{
    AnyEquipment, CyclingTracker, EconomizerMode, IdealLoadsSystem, PredictiveController,
};
use crate::sim::hvac_controller::{HvacSystemMode, IdealHVACController};
use crate::sim::thermal_model_core::DoorGeometry;
use crate::sim::thermal_model_scratch::PhysicsScratchPool;
use crate::sim::thermal_selector::ThermalSelector;
use crate::testing::integration::wiring::WiringTracer;
use fluxion_core::ashrae_cases::NightVentilation;
use std::sync::Arc;

pub struct HvacState<T: ContinuousTensor<f64>> {
    // Building metadata — set once at construction.
    pub num_zones: usize,
    pub case_id: String,
    pub building_type: crate::sim::occupancy::BuildingType,
    pub thermal_model_type: crate::sim::thermal_model_core::ThermalModelType,
    pub timestep_mode: crate::sim::adaptive_timestep::TimestepMode,
    pub door_geometry: DoorGeometry,
    /// Selector driving the production-path dispatch (Issue #3277 / umbrella #3291).
    /// The default `Gauge` selector routes the unified single-zone / multi-zone solver
    /// when the `gauge-solver` feature is enabled. Non-default selectors (`FiveROneC`,
    /// `NineRFourC`) opt out of the gauge path and pin `thermal_model_type` to the
    /// matching legacy network.
    pub thermal_selector: ThermalSelector,

    // HVAC equipment + control.
    pub hvac_heating_capacity: f64,
    pub hvac_cooling_capacity: f64,
    pub hvac_controller: IdealHVACController,
    pub predictive_controller: PredictiveController,
    pub cycling_tracker: CyclingTracker,
    pub economizer_mode: EconomizerMode,
    pub hvac_system_mode: HvacSystemMode,
    pub hvac_equipment: Option<AnyEquipment>,
    pub ideal_loads_system: Vec<Option<IdealLoadsSystem>>,
    pub ideal_air_loads_mode: bool,
    pub free_float: bool,
    pub warm_up_years: u32,
    pub night_ventilation: Option<NightVentilation>,

    // Per-zone HVAC state.
    pub hvac_enabled: T,
    pub previous_temperatures: VectorField,
    pub current_hvac_output: Option<T>,

    // Output metrics.
    pub peak_power_heating: f64,
    pub peak_power_cooling: f64,
    /// Issue #1289 — per-zone peak heating power in kW.
    pub zone_peak_heating_kw: T,
    /// Issue #1289 — per-zone peak cooling power in kW.
    pub zone_peak_cooling_kw: T,
    /// Issue #1628 — timestep index when peak heating occurred for each zone.
    pub zone_peak_heating_timestep: Vec<usize>,
    /// Issue #1628 — timestep index when peak cooling occurred for each zone.
    pub zone_peak_cooling_timestep: Vec<usize>,
    pub annual_heating_energy: f64,
    pub annual_cooling_energy: f64,
    pub annual_electrical_energy: f64,
    /// Issue #1288 — per-zone heating energy (kWh).
    pub zone_heating_energy_kwh: T,
    /// Issue #1288 — per-zone cooling energy (kWh).
    pub zone_cooling_energy_kwh: T,

    // PR #821 / Issue #825 — most recent zone-0 phi_ia/phi_st/phi_m captured for
    // the `pr821-diag` CSV writer. Always 0.0 when the feature is disabled.
    #[cfg(feature = "pr821-diag")]
    pub last_phi_ia: f64,
    #[cfg(feature = "pr821-diag")]
    pub last_phi_st: f64,
    #[cfg(feature = "pr821-diag")]
    pub last_phi_m: f64,

    // Test-only wiring tracer (Issue #2543 / Plan 21-10).
    pub tracer: Option<Arc<WiringTracer>>,

    /// Issue #1966 / #2756 — pooled physics scratch buffers (per-timestep).
    /// On clone, get a fresh empty pool — scratch reuse is a hot-path concern.
    pub(crate) scratch_pool: PhysicsScratchPool,
}

impl<T: ContinuousTensor<f64> + Clone> Clone for HvacState<T> {
    fn clone(&self) -> Self {
        Self {
            num_zones: self.num_zones,
            case_id: self.case_id.clone(),
            building_type: self.building_type,
            thermal_model_type: self.thermal_model_type,
            timestep_mode: self.timestep_mode.clone(),
            door_geometry: self.door_geometry,
            thermal_selector: self.thermal_selector,

            hvac_heating_capacity: self.hvac_heating_capacity,
            hvac_cooling_capacity: self.hvac_cooling_capacity,
            hvac_controller: self.hvac_controller.clone(),
            predictive_controller: self.predictive_controller.clone(),
            cycling_tracker: self.cycling_tracker.clone(),
            economizer_mode: self.economizer_mode,
            hvac_system_mode: self.hvac_system_mode,
            hvac_equipment: self.hvac_equipment.clone(),
            ideal_loads_system: self.ideal_loads_system.clone(),
            ideal_air_loads_mode: self.ideal_air_loads_mode,
            free_float: self.free_float,
            warm_up_years: self.warm_up_years,
            night_ventilation: self.night_ventilation,

            hvac_enabled: self.hvac_enabled.clone(),
            previous_temperatures: self.previous_temperatures.clone(),
            current_hvac_output: self.current_hvac_output.clone(),

            peak_power_heating: self.peak_power_heating,
            peak_power_cooling: self.peak_power_cooling,
            zone_peak_heating_kw: self.zone_peak_heating_kw.clone(),
            zone_peak_cooling_kw: self.zone_peak_cooling_kw.clone(),
            zone_peak_heating_timestep: self.zone_peak_heating_timestep.clone(),
            zone_peak_cooling_timestep: self.zone_peak_cooling_timestep.clone(),
            annual_heating_energy: self.annual_heating_energy,
            annual_cooling_energy: self.annual_cooling_energy,
            annual_electrical_energy: self.annual_electrical_energy,
            zone_heating_energy_kwh: self.zone_heating_energy_kwh.clone(),
            zone_cooling_energy_kwh: self.zone_cooling_energy_kwh.clone(),

            #[cfg(feature = "pr821-diag")]
            last_phi_ia: self.last_phi_ia,
            #[cfg(feature = "pr821-diag")]
            last_phi_st: self.last_phi_st,
            #[cfg(feature = "pr821-diag")]
            last_phi_m: self.last_phi_m,

            tracer: self.tracer.clone(),
            scratch_pool: PhysicsScratchPool::new(),
        }
    }
}
