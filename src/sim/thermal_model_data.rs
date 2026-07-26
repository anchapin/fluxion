//! Thermal Model Data - newtype data container
//!
//! This module contains ThermalModelData which holds all the thermal model state.
//! The ThermalModel wrapper provides Deref to access ThermalModelData fields.

use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::physics::ctf_coefficients::CTFCoefficients;
use crate::physics::ctf_solver::CTFSolver;
use crate::physics::ctf_zone_coupling::CtfZoneCouplingSolver;
use crate::physics::fd_solver::ImplicitFDSolver;
use crate::physics::multi_node_solver::MultiNodeSolver;
use crate::physics::solver_manager::SolverManager;
use crate::sim::adaptive_timestep::TimestepMode;
use crate::sim::boundary::GroundTemperature;
use crate::sim::construction::WallSurface;
use crate::sim::hvac::{
    AnyEquipment, CyclingTracker, EconomizerMode, IdealLoadsSystem, PredictiveController,
};
use crate::sim::hvac_controller::{HvacSystemMode, IdealHVACController};
use crate::sim::occupancy::BuildingType;
use crate::sim::schedule::DailySchedule;
use crate::sim::solar::{SolarPosition, WindowProperties};
use crate::sim::thermal_model_core::DoorGeometry;
use crate::testing::integration::wiring::WiringTracer;
use crate::validation::diagnostics::SimulationDiagnostics;
use crate::weather::HourlyWeatherData;
use fluxion_core::ashrae_cases::{NightVentilation, Orientation};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Accumulator for per-surface incident solar radiation tracking.
///
/// Tracks annual incident solar energy (kWh/m²) and peak irradiance (W/m²)
/// for each surface. Per ASHRAE 140-2023 Section 8.2.3.
#[derive(Clone, Debug, Default)]
pub struct IncidentSolarAccumulator {
    pub annual_kwh_m2: f64,
    pub peak_wm2: f64,
}

impl IncidentSolarAccumulator {
    pub fn new() -> Self {
        Self {
            annual_kwh_m2: 0.0,
            peak_wm2: 0.0,
        }
    }

    pub fn accumulate(&mut self, irradiance_wm2: f64, _area_m2: f64, dt_seconds: f64) {
        self.annual_kwh_m2 += irradiance_wm2 * dt_seconds / 3_600_000.0;
        self.peak_wm2 = self.peak_wm2.max(irradiance_wm2);
    }
}

pub struct ThermalModelData<T: ContinuousTensor<f64> + Clone> {
    pub num_zones: usize,
    pub temperatures: T,
    pub loads: T,
    pub solar_gains: T,
    pub opaque_solar_gains: T,
    pub surfaces: Vec<Vec<WallSurface>>,
    pub window_u_value: f64,
    pub heating_setpoint: f64,
    pub cooling_setpoint: f64,
    pub heating_setpoints: T,
    pub cooling_setpoints: T,
    pub hvac_enabled: T,
    pub heating_schedule: DailySchedule,
    pub cooling_schedule: DailySchedule,
    pub hvac_heating_capacity: f64,
    pub hvac_cooling_capacity: f64,
    pub hvac_controller: IdealHVACController,
    pub predictive_controller: PredictiveController,
    pub cycling_tracker: CyclingTracker,
    pub economizer_mode: EconomizerMode,
    pub previous_temperatures: VectorField,
    pub hvac_equipment: Option<AnyEquipment>,
    pub zone_area: T,
    pub ceiling_height: T,
    pub air_density: T,
    pub heat_capacity: T,
    pub window_ratio: T,
    pub aspect_ratio: T,
    pub infiltration_rate: T,
    pub wall_u_value: f64,
    pub roof_u_value: f64,
    pub floor_u_value: f64,
    pub case_id: String,
    pub building_type: BuildingType,
    pub thermal_model_type: crate::sim::thermal_model_core::ThermalModelType,
    pub timestep_mode: TimestepMode,
    pub mass_temperatures: T,
    pub thermal_capacitance: T,
    /// Air-node thermal capacitance C_air = ρ_air · cp_air · V_zone  [J/K].
    ///
    /// Issue #1522 (option (a)): restores a real capacitance on the 5R1C air
    /// node so it decouples from the slow mass node on sub-timestep
    /// timescales. Closes the algebraic pinning that drove peak_cooling OVER,
    /// peak_heating UNDER, annual_cooling UNDER, and free-float min too warm
    /// for ASHRAE 140 Case 600 series. Per ISO 13790 §12.2.2 and ASHRAE 140
    /// §5.2.2, the air node is a true ODE state with relaxation time
    /// τ_air = C_air / den (≈0.28 h for Case 600), not algebraically pinned
    /// to the mass node. Used in `step_physics_5r1c` only; the 9R4C
    /// (`step_physics_9r4c`) and 6R2C paths maintain their own air-node
    /// handling.
    pub air_thermal_capacitance: T,
    /// Independent air-node temperature state for the 5R1C model.
    ///
    /// This field stores the free-floating air temperature (pre-HVAC) from the
    /// previous timestep, used as the ODE state `t_air_old` in the exact
    /// exponential solution of the air-node energy balance:
    ///   t_air_new = steady + (t_air_old - steady) * exp(-dt / τ_air)
    /// where τ_air = C_air · term_rest_1 / den.
    ///
    /// This replaces the legacy algebraic pinning `t_i_free = num / den` that
    /// prevented the air node from decoupling on sub-timestep timescales.
    /// Used in `step_physics_5r1c` only; the 9R4C and 6R2C paths
    /// maintain their own air-node temperature handling.
    pub air_temperatures: T,
    /// Independent interior wall-surface temperature state for the 5R1C model.
    ///
    /// Tracks the temperature at the interior wall surface `T_si` for each
    /// zone, evolved via the exact exponential solution of the surface-node
    /// ODE:
    ///
    /// ```text
    /// T_si_new = T_si_eq + (T_si_old - T_si_eq) · exp(-dt / τ_si)
    /// τ_si = C_m · (R_1 ∥ R_si) = C_m · R_1·R_si / (R_1 + R_si)
    /// T_si_eq = (T_m / R_1 + T_int / R_si) / (1/R_1 + 1/R_si)
    /// ```
    ///
    /// Here `R_1 = R_ms = 1 / h_tr_ms`; `h_tr_ms` excludes the interior film,
    /// which is represented separately by `R_si = 1 / h_tr_is`.
    /// The flux to the zone air is coupled into the 5R1C air-node numerator.
    pub wall_surface_temperatures: T,
    pub envelope_mass_temperatures: T,
    pub internal_mass_temperatures: T,
    pub envelope_thermal_capacitance: T,
    pub internal_thermal_capacitance: T,
    pub h_tr_me: T,
    pub ceiling_mass_temperatures: Option<T>,
    pub floor_mass_temperatures: Option<T>,
    pub partition_mass_temperatures: Option<T>,
    pub ceiling_thermal_capacitance: Option<T>,
    pub floor_thermal_capacitance: Option<T>,
    pub partition_thermal_capacitance: Option<T>,
    pub h_tr_ceiling: Option<T>,
    pub h_tr_floor_mass: Option<T>,
    pub h_tr_partition: Option<T>,
    pub tracer: Option<Arc<WiringTracer>>,
    pub h_tr_em: T,
    pub h_tr_ms: T,
    pub h_tr_is: T,
    /// h_tr_is excluding south wall contribution (for south wall bypass fix, Issue #715)
    pub h_tr_is_no_south: T,
    /// South wall's h_tr_em for series path computation (Issue #715)
    pub h_tr_em_south: T,
    pub h_tr_w: T,
    pub h_ve: T,
    pub h_tr_floor: T,
    pub ground_temperature: Box<dyn GroundTemperature>,
    pub h_tr_iz: T,
    pub h_tr_iz_rad: T,
    pub surface_emissivity: T,
    pub zone_volume: T,
    pub common_wall_area: f64,
    pub hvac_system_mode: HvacSystemMode,
    pub night_ventilation: Option<NightVentilation>,
    pub h_vent_mass: f64,
    pub thermal_bridge_coefficient: f64,
    pub ideal_air_loads_mode: bool,
    pub ideal_loads_system: Vec<Option<IdealLoadsSystem>>,
    pub free_float: bool, // When true, HVAC output is forced to zero (for free-floating cases)
    /// Number of warm-up years for annual simulation (Issue #744).
    /// Runs the full 8760-hour simulation this many times before collecting results,
    /// ensuring periodic steady-state for high-mass constructions.
    /// Default: 2.
    pub warm_up_years: u32,
    pub ctf_coefficients: Option<CTFCoefficients>,
    pub ctf_solvers: Vec<CTFSolver>,
    pub ctf_enabled: bool,
    pub ctf_timestep: f64,
    pub ctf_zone_coupling_solver: Option<CtfZoneCouplingSolver>,
    pub ctf_primary: bool,
    pub fd_solvers: Vec<ImplicitFDSolver>,
    pub fd_enabled: bool,
    pub fd_timestep: f64,
    pub multi_node_solvers: Vec<MultiNodeSolver>,
    pub solver_manager: Option<SolverManager>,
    pub convective_fraction: f64,
    pub solar_distribution_to_air: f64,
    pub solar_beam_to_mass_fraction: f64,
    pub previous_mass_temperatures: T,
    pub mass_energy_change_cumulative: f64,
    pub envelope_mass_energy_change_cumulative: f64,
    pub internal_mass_energy_change_cumulative: f64,
    pub peak_power_heating: f64,
    pub peak_power_cooling: f64,
    /// Issue #1289 — per-zone peak heating power in kW
    pub zone_peak_heating_kw: T,
    /// Issue #1289 — per-zone peak cooling power in kW
    pub zone_peak_cooling_kw: T,
    /// Issue #1628 — timestep index when peak heating occurred for each zone
    pub zone_peak_heating_timestep: Vec<usize>,
    /// Issue #1628 — timestep index when peak cooling occurred for each zone
    pub zone_peak_cooling_timestep: Vec<usize>,
    pub annual_heating_energy: f64,
    pub annual_cooling_energy: f64,
    pub annual_electrical_energy: f64,
    // Per-zone energy tracking (Issue #1288)
    pub zone_heating_energy_kwh: T,
    pub zone_cooling_energy_kwh: T,
    pub weather: Option<HourlyWeatherData>,
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    /// Issue #1416: explicit EPW LOCATION time-zone offset (decimal hours).
    /// When `Some`, forwarded to `calculate_solar_position` so half-hour zones
    /// and 7.5°-offset longitudes produce correct solar positions. `None`
    /// preserves the legacy longitude-inferred fallback (the original
    /// ASHRAE-140 baseline).
    pub utc_offset_hours: Option<f64>,
    pub window_properties: Vec<WindowProperties>,
    pub window_orientations: Vec<Vec<Orientation>>,
    pub door_geometry: DoorGeometry,
    pub derived_h_ext: T,
    pub derived_term_rest_1: T,
    pub derived_h_ms_is_prod: T,
    pub derived_den: T,
    pub derived_ground_coeff: T,
    /// ISO 13790 §C.6: H_tr_1 = 1/(1/H_ve_adj + 1/H_tr_is) — combined ventilation + surface-to-air
    pub derived_h_tr_1: T,
    /// ISO 13790 §C.7: H_tr_2 = H_tr_1 + H_tr_w — adds window conductance
    pub derived_h_tr_2: T,
    /// ISO 13790 §C.8: H_tr_3 = 1/(1/H_tr_2 + 1/H_tr_ms) — combined air-to-mass (~40 W/K for Case 900)
    pub derived_h_tr_3: T,
    pub diagnostics: Option<SimulationDiagnostics>,
    pub current_hvac_output: Option<T>,
    pub internal_radiative_to_mass: f64,
    // Per-surface thermal mass conductances for 9R4C model (Issue #715, Phase 6B)
    pub h_tr_ms_wall: Option<T>,
    pub h_tr_ms_roof: Option<T>,
    pub h_tr_ms_floor: Option<T>,
    pub h_tr_em_wall: Option<T>,
    pub h_tr_em_roof: Option<T>,
    pub h_tr_em_floor: Option<T>,
    // Per-surface thermal capacitances for 9R4C model
    pub cm_wall: Option<T>,
    pub cm_roof: Option<T>,
    pub cm_floor: Option<T>,
    pub cm_internal: Option<T>,
    // Multi-node thermal mass state for 9R4C model
    pub multi_node_thermal_mass: Option<fluxion_core::multi_node::MultiNodeThermalMass>,
    /// PR #821 / Issue #825 — most recent zone-0 phi_ia (W to air node) computed
    /// inside `step_physics_5r1c`. Captured for the `pr821-diag` CSV writer so
    /// the heat-balance terms can be inspected hour by hour. Always 0.0 when
    /// the `pr821-diag` feature is disabled (and the field does not exist).
    #[cfg(feature = "pr821-diag")]
    pub last_phi_ia: f64,
    /// PR #821 / Issue #825 — most recent zone-0 phi_st (W to surface node).
    #[cfg(feature = "pr821-diag")]
    pub last_phi_st: f64,
    /// PR #821 / Issue #825 — most recent zone-0 phi_m (W to mass node).
    #[cfg(feature = "pr821-diag")]
    pub last_phi_m: f64,
    /// Issue #763 — full hourly zone temperature profiles for post-processing and reporting.
    /// Format: [num_zones][8760] hourly temperatures in °C.
    pub hourly_temperatures: Option<Vec<Vec<f64>>>,
    /// Issue #1799 — sub-hourly 9R4C node temperatures per zone (diagnostics + ML features).
    ///
    /// `nodal_temperatures[zone][node][timestep]` where `node` follows the
    /// canonical `MultiNodeSolver::NODE_NAMES` ordering:
    /// `0=wall, 1=roof, 2=floor, 3=internal`.
    ///
    /// Only populated when the model carries at least one `MultiNodeSolver`
    /// (i.e. 9R4C / heavy-mass construction like ASHRAE 140 Case 900+).
    /// `None` for low-mass models or before a simulation has been run.
    pub nodal_temperatures: Option<Vec<Vec<Vec<f64>>>>,
    /// Issue #762 — per-surface incident solar radiation tracking for ASHRAE 140-2023 Section 8.2.3.
    /// Key: surface identifier (e.g., "wall_N", "window_S", "roof").
    /// BTreeMap for deterministic iteration order across platforms (Issue #1297)
    pub incident_solar_per_surface: BTreeMap<String, IncidentSolarAccumulator>,
    /// Issue #1212 — solar position cache keyed by `(timestep, hour_slot)`.
    /// 2 slots per timestep (integer-hour for 5R1C, mid-hour for 9R4C) prevent
    /// the 5R1C caller from overwriting the 9R4C caller's value (see
    /// `cached_solar_position` in `thermal_model_core.rs`).
    pub sun_pos_cache: std::collections::HashMap<(usize, i32), SolarPosition>,
}

impl<T: ContinuousTensor<f64> + Clone> Clone for ThermalModelData<T> {
    fn clone(&self) -> Self {
        Self {
            num_zones: self.num_zones,
            temperatures: self.temperatures.clone(),
            loads: self.loads.clone(),
            solar_gains: self.solar_gains.clone(),
            opaque_solar_gains: self.opaque_solar_gains.clone(),
            surfaces: self.surfaces.clone(),
            window_u_value: self.window_u_value,
            heating_setpoint: self.heating_setpoint,
            cooling_setpoint: self.cooling_setpoint,
            heating_setpoints: self.heating_setpoints.clone(),
            cooling_setpoints: self.cooling_setpoints.clone(),
            hvac_enabled: self.hvac_enabled.clone(),
            heating_schedule: self.heating_schedule.clone(),
            cooling_schedule: self.cooling_schedule.clone(),
            hvac_heating_capacity: self.hvac_heating_capacity,
            hvac_cooling_capacity: self.hvac_cooling_capacity,
            hvac_controller: self.hvac_controller.clone(),
            predictive_controller: self.predictive_controller.clone(),
            cycling_tracker: self.cycling_tracker.clone(),
            economizer_mode: self.economizer_mode,
            previous_temperatures: self.previous_temperatures.clone(),
            hvac_equipment: self.hvac_equipment.clone(),
            zone_area: self.zone_area.clone(),
            ceiling_height: self.ceiling_height.clone(),
            air_density: self.air_density.clone(),
            heat_capacity: self.heat_capacity.clone(),
            window_ratio: self.window_ratio.clone(),
            aspect_ratio: self.aspect_ratio.clone(),
            infiltration_rate: self.infiltration_rate.clone(),
            wall_u_value: self.wall_u_value,
            roof_u_value: self.roof_u_value,
            floor_u_value: self.floor_u_value,
            case_id: self.case_id.clone(),
            building_type: self.building_type,
            thermal_model_type: self.thermal_model_type,
            timestep_mode: self.timestep_mode.clone(),
            mass_temperatures: self.mass_temperatures.clone(),
            thermal_capacitance: self.thermal_capacitance.clone(),
            air_thermal_capacitance: self.air_thermal_capacitance.clone(),
            air_temperatures: self.air_temperatures.clone(),
            wall_surface_temperatures: self.wall_surface_temperatures.clone(),
            envelope_mass_temperatures: self.envelope_mass_temperatures.clone(),
            internal_mass_temperatures: self.internal_mass_temperatures.clone(),
            envelope_thermal_capacitance: self.envelope_thermal_capacitance.clone(),
            internal_thermal_capacitance: self.internal_thermal_capacitance.clone(),
            h_tr_me: self.h_tr_me.clone(),
            ceiling_mass_temperatures: self.ceiling_mass_temperatures.clone(),
            floor_mass_temperatures: self.floor_mass_temperatures.clone(),
            partition_mass_temperatures: self.partition_mass_temperatures.clone(),
            ceiling_thermal_capacitance: self.ceiling_thermal_capacitance.clone(),
            floor_thermal_capacitance: self.floor_thermal_capacitance.clone(),
            partition_thermal_capacitance: self.partition_thermal_capacitance.clone(),
            h_tr_ceiling: self.h_tr_ceiling.clone(),
            h_tr_floor_mass: self.h_tr_floor_mass.clone(),
            h_tr_partition: self.h_tr_partition.clone(),
            tracer: self.tracer.clone(),
            h_tr_em: self.h_tr_em.clone(),
            h_tr_ms: self.h_tr_ms.clone(),
            h_tr_is: self.h_tr_is.clone(),
            h_tr_is_no_south: self.h_tr_is_no_south.clone(),
            h_tr_em_south: self.h_tr_em_south.clone(),
            h_tr_w: self.h_tr_w.clone(),
            h_ve: self.h_ve.clone(),
            h_tr_floor: self.h_tr_floor.clone(),
            ground_temperature: self.ground_temperature.clone_box(),
            h_tr_iz: self.h_tr_iz.clone(),
            h_tr_iz_rad: self.h_tr_iz_rad.clone(),
            surface_emissivity: self.surface_emissivity.clone(),
            zone_volume: self.zone_volume.clone(),
            common_wall_area: self.common_wall_area,
            hvac_system_mode: self.hvac_system_mode,
            night_ventilation: self.night_ventilation,
            h_vent_mass: self.h_vent_mass,
            thermal_bridge_coefficient: self.thermal_bridge_coefficient,
            ideal_air_loads_mode: self.ideal_air_loads_mode,
            ideal_loads_system: self.ideal_loads_system.clone(),
            free_float: self.free_float,
            warm_up_years: self.warm_up_years,
            ctf_coefficients: self.ctf_coefficients.clone(),
            ctf_solvers: self.ctf_solvers.clone(),
            ctf_enabled: self.ctf_enabled,
            ctf_timestep: self.ctf_timestep,
            ctf_zone_coupling_solver: self.ctf_zone_coupling_solver.clone(),
            ctf_primary: self.ctf_primary,
            fd_solvers: vec![],
            fd_enabled: self.fd_enabled,
            fd_timestep: self.fd_timestep,
            multi_node_solvers: vec![],
            solver_manager: None,
            convective_fraction: self.convective_fraction,
            solar_distribution_to_air: self.solar_distribution_to_air,
            solar_beam_to_mass_fraction: self.solar_beam_to_mass_fraction,
            previous_mass_temperatures: self.previous_mass_temperatures.clone(),
            mass_energy_change_cumulative: self.mass_energy_change_cumulative,
            envelope_mass_energy_change_cumulative: self.envelope_mass_energy_change_cumulative,
            internal_mass_energy_change_cumulative: self.internal_mass_energy_change_cumulative,
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
            weather: self.weather.clone(),
            latitude_deg: self.latitude_deg,
            longitude_deg: self.longitude_deg,
            utc_offset_hours: self.utc_offset_hours,
            window_properties: self.window_properties.clone(),
            window_orientations: self.window_orientations.clone(),
            door_geometry: self.door_geometry,
            derived_h_ext: self.derived_h_ext.clone(),
            derived_term_rest_1: self.derived_term_rest_1.clone(),
            derived_h_ms_is_prod: self.derived_h_ms_is_prod.clone(),
            derived_den: self.derived_den.clone(),
            derived_ground_coeff: self.derived_ground_coeff.clone(),
            derived_h_tr_1: self.derived_h_tr_1.clone(),
            derived_h_tr_2: self.derived_h_tr_2.clone(),
            derived_h_tr_3: self.derived_h_tr_3.clone(),
            diagnostics: None,
            current_hvac_output: self.current_hvac_output.clone(),
            internal_radiative_to_mass: self.internal_radiative_to_mass,
            h_tr_ms_wall: self.h_tr_ms_wall.clone(),
            h_tr_ms_roof: self.h_tr_ms_roof.clone(),
            h_tr_ms_floor: self.h_tr_ms_floor.clone(),
            h_tr_em_wall: self.h_tr_em_wall.clone(),
            h_tr_em_roof: self.h_tr_em_roof.clone(),
            h_tr_em_floor: self.h_tr_em_floor.clone(),
            cm_wall: self.cm_wall.clone(),
            cm_roof: self.cm_roof.clone(),
            cm_floor: self.cm_floor.clone(),
            cm_internal: self.cm_internal.clone(),
            multi_node_thermal_mass: self.multi_node_thermal_mass.clone(),
            #[cfg(feature = "pr821-diag")]
            last_phi_ia: self.last_phi_ia,
            #[cfg(feature = "pr821-diag")]
            last_phi_st: self.last_phi_st,
            #[cfg(feature = "pr821-diag")]
            last_phi_m: self.last_phi_m,
            hourly_temperatures: None,
            nodal_temperatures: None,
            incident_solar_per_surface: self.incident_solar_per_surface.clone(),
            sun_pos_cache: self.sun_pos_cache.clone(),
        }
    }
}
