//! Zone-level HVAC control logic.
//!
//! This module implements independent HVAC control for each thermal zone
//! based on current temperatures and configured setpoints.
//!
//! # Layered Controllers
//!
//! This module supports multiple control strategies organized in layers:
//!
//! - **Layer 1: Status Determination** - Determines Heating/Cooling/Off based on deadband
//! - **Layer 2: Control Strategy** - Selects between Ideal Loads, Staged Equipment, or Schedule-Aware
//! - **Layer 3: Energy Calculation** - Calculates actual energy input based on selected strategy
//!
//! ## Control Strategies
//!
//! - `IdealLoadsController`: Uses thermodynamic formulas (Q = ṁ·cp·ΔT) for infinite-capacity ideal response
//! - `StagedEquipmentController`: Models equipment with cycling losses and part-load degradation
//! - `ScheduleAwareController`: Uses predictive control with time-varying setpoints (setback/setup)

use std::sync::Arc;

use crate::physics::cta::VectorField;
use crate::thermal::thermal_model::ThermalModel;

/// HVAC system status for a zone.
#[derive(Debug, Clone, PartialEq)]
pub enum HVACStatus {
    Heating,
    Cooling,
    Off,
}

/// Control strategy for HVAC operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlStrategy {
    /// Ideal loads: thermodynamic-based calculation with infinite capacity
    IdealLoads,
    /// Staged equipment: models cycling losses and part-load efficiency degradation
    StagedEquipment,
    /// Schedule-aware: predictive control with time-varying setpoints
    ScheduleAware,
}

impl Default for ControlStrategy {
    fn default() -> Self {
        ControlStrategy::IdealLoads
    }
}

/// Layered controller configuration for a zone.
#[derive(Debug, Clone)]
pub struct LayeredControllerConfig {
    /// Control strategy to use
    pub strategy: ControlStrategy,
    /// Zone volume for ideal loads calculation (m³)
    pub zone_volume: f64,
    /// Air changes per hour for ventilation
    pub air_changes_per_hour: f64,
    /// Supply air temperature for cooling (°C), typically 13°C
    pub supply_cooling_temp: f64,
    /// Supply air temperature for heating (°C), typically 40°C
    pub supply_heating_temp: f64,
    /// Cooling COP for equipment efficiency
    pub cooling_cop: f64,
    /// Heating efficiency for equipment
    pub heating_efficiency: f64,
    /// Minimum runtime for staged equipment (timesteps)
    pub min_runtime_timesteps: u32,
    /// Startup penalty (kWh) for staged equipment
    pub startup_penalty_kwh: f64,
}

impl Default for LayeredControllerConfig {
    fn default() -> Self {
        Self {
            strategy: ControlStrategy::default(),
            zone_volume: 129.6,        // ASHRAE 140 standard: 8m × 6m × 2.7m
            air_changes_per_hour: 0.5, // ASHRAE 140 standard
            supply_cooling_temp: 13.0,
            supply_heating_temp: 40.0,
            cooling_cop: 3.0,        // ASHRAE 140 standard
            heating_efficiency: 0.9, // ASHRAE 140 standard (electric resistance)
            min_runtime_timesteps: 5,
            startup_penalty_kwh: 0.1,
        }
    }
}

/// Layered controller for a zone.
///
/// This controller supports three distinct control strategies:
/// 1. Ideal Loads - uses thermodynamic formulas for perfect load tracking
/// 2. Staged Equipment - models realistic equipment with cycling losses
/// 3. Schedule-Aware - uses predictive control with dynamic setpoints
#[derive(Debug, Clone)]
pub struct LayeredController {
    /// Configuration
    config: LayeredControllerConfig,
    /// Cycling tracker for staged equipment
    cycling_tracker: crate::sim::hvac::cycling::CyclingTracker,
    /// Previous zone temperature for rate calculation
    previous_zone_temp: f64,
    /// Previous setpoints for schedule-aware control
    prev_heating_setpoint: f64,
    /// Previous cooling setpoint
    prev_cooling_setpoint: f64,
}

impl LayeredController {
    /// Create a new layered controller with default configuration.
    pub fn new() -> Self {
        Self {
            config: LayeredControllerConfig::default(),
            cycling_tracker: crate::sim::hvac::cycling::CyclingTracker::new(),
            previous_zone_temp: 20.0,
            prev_heating_setpoint: 20.0,
            prev_cooling_setpoint: 24.0,
        }
    }

    /// Create a layered controller with custom configuration.
    pub fn with_config(config: LayeredControllerConfig) -> Self {
        Self {
            cycling_tracker: crate::sim::hvac::cycling::CyclingTracker::new(),
            previous_zone_temp: 20.0,
            prev_heating_setpoint: 20.0,
            prev_cooling_setpoint: 24.0,
            config,
        }
    }

    /// Calculate energy input using the configured control strategy.
    ///
    /// # Arguments
    /// * `zone_id` - Zone index
    /// * `current_temp` - Current zone temperature (°C)
    /// * `status` - Current HVAC status
    /// * `heating_setpoint` - Current heating setpoint (°C)
    /// * `cooling_setpoint` - Current cooling setpoint (°C)
    ///
    /// # Returns
    /// Energy input in Watts
    pub fn calculate_energy_input(
        &mut self,
        zone_id: usize,
        current_temp: f64,
        status: &HVACStatus,
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> f64 {
        match self.config.strategy {
            ControlStrategy::IdealLoads => self.calculate_ideal_loads(
                zone_id,
                current_temp,
                status,
                heating_setpoint,
                cooling_setpoint,
            ),
            ControlStrategy::StagedEquipment => self.calculate_staged_equipment(
                zone_id,
                current_temp,
                status,
                heating_setpoint,
                cooling_setpoint,
            ),
            ControlStrategy::ScheduleAware => self.calculate_schedule_aware(
                zone_id,
                current_temp,
                status,
                heating_setpoint,
                cooling_setpoint,
            ),
        }
    }

    /// Calculate energy using ideal loads thermodynamics: Q = ṁ·cp·ΔT
    #[allow(clippy::unused_self)]
    fn calculate_ideal_loads(
        &self,
        _zone_id: usize,
        current_temp: f64,
        status: &HVACStatus,
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> f64 {
        match status {
            HVACStatus::Heating => {
                let load =
                    crate::sim::hvac::ideal_loads::ZoneIdealLoads::calculate_sensible_heating_load(
                        current_temp,
                        heating_setpoint,
                        self.config.supply_heating_temp,
                        self.config.zone_volume,
                        self.config.air_changes_per_hour,
                    );
                // Convert to electrical input using heating efficiency
                load / self.config.heating_efficiency
            }
            HVACStatus::Cooling => {
                let load =
                    crate::sim::hvac::ideal_loads::ZoneIdealLoads::calculate_sensible_cooling_load(
                        current_temp,
                        cooling_setpoint,
                        self.config.supply_cooling_temp,
                        self.config.zone_volume,
                        self.config.air_changes_per_hour,
                    );
                // Convert to electrical input using COP
                load / self.config.cooling_cop
            }
            HVACStatus::Off => 0.0,
        }
    }

    /// Calculate energy with staged equipment including cycling losses.
    fn calculate_staged_equipment(
        &mut self,
        zone_id: usize,
        current_temp: f64,
        status: &HVACStatus,
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> f64 {
        let base_energy = self.calculate_ideal_loads(
            zone_id,
            current_temp,
            status,
            heating_setpoint,
            cooling_setpoint,
        );

        if base_energy == 0.0 {
            return 0.0;
        }

        // Determine if equipment is on
        let is_on = !matches!(status, HVACStatus::Off);

        // Calculate part-load ratio based on demand vs some nominal capacity
        // For staged equipment, we assume 3 stages with ~33%, 66%, 100% capacity
        let nominal_capacity = self.config.zone_volume * 50.0; // Rough estimate: 50 W/m³
        let plr = (base_energy / nominal_capacity).clamp(0.0, 1.0);

        // Apply cycling losses
        let (efficiency_mult, startup_penalty) =
            self.cycling_tracker.calculate_cycling_loss(is_on, plr);

        // Calculate final energy with cycling penalty (in watts)
        base_energy * efficiency_mult + startup_penalty * 1000.0
    }

    /// Calculate energy using schedule-aware predictive control.
    ///
    /// Uses thermal inertia and temperature rate of change to anticipate
    /// load needs and prevent oscillation.
    fn calculate_schedule_aware(
        &mut self,
        zone_id: usize,
        current_temp: f64,
        status: &HVACStatus,
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> f64 {
        // Calculate temperature rate of change
        let temp_rate = current_temp - self.previous_zone_temp;

        // Use ideal loads as the base calculation
        let base_energy = self.calculate_ideal_loads(
            zone_id,
            current_temp,
            status,
            heating_setpoint,
            cooling_setpoint,
        );

        if base_energy == 0.0 {
            self.previous_zone_temp = current_temp;
            self.prev_heating_setpoint = heating_setpoint;
            self.prev_cooling_setpoint = cooling_setpoint;
            return 0.0;
        }

        // Apply predictive modulation factor based on temperature rate
        // Rising temperature in cooling mode -> reduce modulation
        // Falling temperature in heating mode -> reduce modulation
        let predictive_factor = match status {
            HVACStatus::Cooling if temp_rate > 0.01 => 0.8, // Anticipate overshoot
            HVACStatus::Heating if temp_rate < -0.01 => 0.8, // Anticipate undershoot
            _ => 1.0,
        };

        // Detect setpoint changes (schedule transitions)
        let setpoint_changed = (heating_setpoint - self.prev_heating_setpoint).abs() > 0.1
            || (cooling_setpoint - self.prev_cooling_setpoint).abs() > 0.1;

        // Apply boost for setpoint transitions (setup/setback)
        let transition_boost = if setpoint_changed { 1.2 } else { 1.0 };

        self.previous_zone_temp = current_temp;
        self.prev_heating_setpoint = heating_setpoint;
        self.prev_cooling_setpoint = cooling_setpoint;

        base_energy * predictive_factor * transition_boost
    }

    /// Reset controller state for new simulation.
    pub fn reset(&mut self) {
        self.previous_zone_temp = 20.0;
        self.prev_heating_setpoint = 20.0;
        self.prev_cooling_setpoint = 24.0;
        self.cycling_tracker.reset();
    }
}

impl Default for LayeredController {
    fn default() -> Self {
        Self::new()
    }
}

/// Zone-level HVAC control system.
pub struct ZoneControl {
    /// Reference to the thermal model
    pub thermal_model: Arc<ThermalModel>,

    /// Zone setpoints configuration
    setpoints: crate::sim::hvac::zones::zone_setpoints::ZoneSetpoints,

    /// Current HVAC status for each zone
    zone_status: VectorField,

    /// Layered controllers for each zone
    layered_controllers: Vec<LayeredController>,
}

impl std::fmt::Debug for ZoneControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZoneControl")
            .field("thermal_model.num_zones", &self.thermal_model.num_zones)
            .field("setpoints", &self.setpoints)
            .field("zone_status", &self.zone_status)
            .field("layered_controllers", &self.layered_controllers)
            .finish()
    }
}

impl ZoneControl {
    /// Create a new ZoneControl instance with default layered controllers.
    ///
    /// # Arguments
    /// * `thermal_model` - Arc-wrapped thermal model
    /// * `setpoints` - Zone setpoints configuration
    ///
    /// # Returns
    /// A new ZoneControl instance
    pub fn new(
        thermal_model: Arc<ThermalModel>,
        setpoints: crate::sim::hvac::zones::zone_setpoints::ZoneSetpoints,
    ) -> Self {
        let num_zones = thermal_model.num_zones;
        let layered_controllers = (0..num_zones).map(|_| LayeredController::new()).collect();
        ZoneControl {
            thermal_model,
            setpoints,
            zone_status: VectorField::from_scalar(0.0, num_zones),
            layered_controllers,
        }
    }

    /// Create a new ZoneControl instance with custom layered controller configuration.
    ///
    /// # Arguments
    /// * `thermal_model` - Arc-wrapped thermal model
    /// * `setpoints` - Zone setpoints configuration
    /// * `controller_configs` - Per-zone layered controller configurations
    ///
    /// # Returns
    /// A new ZoneControl instance
    pub fn with_layered_controllers(
        thermal_model: Arc<ThermalModel>,
        setpoints: crate::sim::hvac::zones::zone_setpoints::ZoneSetpoints,
        controller_configs: Vec<LayeredControllerConfig>,
    ) -> Self {
        let num_zones = thermal_model.num_zones;
        let layered_controllers: Vec<LayeredController> = controller_configs
            .into_iter()
            .map(LayeredController::with_config)
            .collect();
        ZoneControl {
            thermal_model,
            setpoints,
            zone_status: VectorField::from_scalar(0.0, num_zones),
            layered_controllers,
        }
    }

    /// Create a ZoneControl with a specific default control strategy.
    ///
    /// # Arguments
    /// * `thermal_model` - Arc-wrapped thermal model
    /// * `setpoints` - Zone setpoints configuration
    /// * `default_strategy` - Default control strategy for all zones
    ///
    /// # Returns
    /// A new ZoneControl instance
    pub fn with_strategy(
        thermal_model: Arc<ThermalModel>,
        setpoints: crate::sim::hvac::zones::zone_setpoints::ZoneSetpoints,
        default_strategy: ControlStrategy,
    ) -> Self {
        let num_zones = thermal_model.num_zones;
        let layered_controllers: Vec<LayeredController> = (0..num_zones)
            .map(|_| {
                let mut c = LayeredController::new();
                c.config.strategy = default_strategy;
                c
            })
            .collect();
        ZoneControl {
            thermal_model,
            setpoints,
            zone_status: VectorField::from_scalar(0.0, num_zones),
            layered_controllers,
        }
    }

    /// Set the control strategy for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zone index (0-based)
    /// * `strategy` - Control strategy to use
    pub fn set_zone_strategy(&mut self, zone_id: usize, strategy: ControlStrategy) {
        if let Some(controller) = self.layered_controllers.get_mut(zone_id) {
            controller.config.strategy = strategy;
        }
    }

    /// Get the control strategy for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zone index (0-based)
    ///
    /// # Returns
    /// Current control strategy for the zone
    pub fn get_zone_strategy(&self, zone_id: usize) -> Option<ControlStrategy> {
        self.layered_controllers
            .get(zone_id)
            .map(|c| c.config.strategy)
    }

    /// Update HVAC controls for all zones based on current temperatures.
    ///
    /// # Arguments
    /// * `current_temperatures` - Current zone temperatures
    ///
    /// # Returns
    /// Energy input vector (Watts) for each zone
    pub fn update_zone_controls(&mut self, current_temperatures: &VectorField) -> VectorField {
        let mut energy_input = VectorField::from_scalar(0.0, self.thermal_model.num_zones);

        for zone_id in 0..self.thermal_model.num_zones {
            let current_temp = current_temperatures.as_slice()[zone_id];
            let heating_setpoint = self.setpoints.get_heating_setpoint(zone_id);
            let cooling_setpoint = self.setpoints.get_cooling_setpoint(zone_id);
            let deadband = self.setpoints.get_deadband(zone_id);

            let status = self.determine_hvac_status(
                zone_id,
                current_temp,
                heating_setpoint,
                cooling_setpoint,
                deadband,
            );

            // Update status tracking
            self.zone_status.as_mut_slice()[zone_id] = match status {
                HVACStatus::Heating => 1.0,
                HVACStatus::Cooling => -1.0,
                HVACStatus::Off => 0.0,
            };

            // Calculate energy input based on status
            let energy = self.calculate_energy_input(zone_id, current_temp, &status);
            energy_input.as_mut_slice()[zone_id] = energy;
        }

        energy_input
    }

    /// Determine HVAC status for a zone.
    fn determine_hvac_status(
        &self,
        _zone_id: usize,
        current_temp: f64,
        heating_setpoint: f64,
        cooling_setpoint: f64,
        deadband: f64,
    ) -> HVACStatus {
        let heating_threshold = heating_setpoint - deadband / 2.0;
        let cooling_threshold = cooling_setpoint + deadband / 2.0;

        if current_temp < heating_threshold {
            HVACStatus::Heating
        } else if current_temp > cooling_threshold {
            HVACStatus::Cooling
        } else {
            HVACStatus::Off
        }
    }

    /// Get current HVAC status for a zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zone index (0-based)
    ///
    /// # Returns
    /// HVACStatus enum value
    pub fn get_zone_hvac_status(&self, zone_id: usize) -> HVACStatus {
        let status_value = self.zone_status.as_slice()[zone_id];
        if status_value > 0.0 {
            HVACStatus::Heating
        } else if status_value < 0.0 {
            HVACStatus::Cooling
        } else {
            HVACStatus::Off
        }
    }

    /// Calculate energy input for a zone using layered controllers.
    ///
    /// # Arguments
    /// * `zone_id` - Zone index (0-based)
    /// * `current_temp` - Current zone temperature (°C)
    /// * `status` - Current HVAC status
    ///
    /// # Returns
    /// Energy input in Watts
    pub fn calculate_energy_input(
        &mut self,
        zone_id: usize,
        current_temp: f64,
        status: &HVACStatus,
    ) -> f64 {
        let heating_setpoint = self.setpoints.get_heating_setpoint(zone_id);
        let cooling_setpoint = self.setpoints.get_cooling_setpoint(zone_id);

        if let Some(controller) = self.layered_controllers.get_mut(zone_id) {
            controller.calculate_energy_input(
                zone_id,
                current_temp,
                status,
                heating_setpoint,
                cooling_setpoint,
            )
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thermal::thermal_model::ThermalModel;

    #[test]
    fn test_zone_control_creation() {
        let thermal_model = Arc::new(ThermalModel::new(3));
        let setpoints = crate::sim::hvac::zones::zone_setpoints::ZoneSetpoints::new(3);
        let zone_control = ZoneControl::new(thermal_model, setpoints);

        // Initial status should be Off for all zones
        for zone_id in 0..3 {
            assert_eq!(zone_control.get_zone_hvac_status(zone_id), HVACStatus::Off);
        }
    }

    #[test]
    fn test_heating_control() {
        let thermal_model = Arc::new(ThermalModel::new(1));
        let mut setpoints = crate::sim::hvac::zones::zone_setpoints::ZoneSetpoints::new(1);
        setpoints.set_heating_setpoint(0, 22.0).unwrap();
        setpoints.set_cooling_setpoint(0, 26.0).unwrap();

        let mut zone_control = ZoneControl::new(thermal_model.clone(), setpoints);
        let current_temps = VectorField::from_scalar(18.0, 1);

        let energy_input = zone_control.update_zone_controls(&current_temps);

        assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Heating);
        assert!(energy_input.as_slice()[0] > 0.0);
    }

    #[test]
    fn test_cooling_control() {
        let thermal_model = Arc::new(ThermalModel::new(1));
        let mut setpoints = crate::sim::hvac::zones::zone_setpoints::ZoneSetpoints::new(1);
        setpoints.set_heating_setpoint(0, 22.0).unwrap();
        setpoints.set_cooling_setpoint(0, 26.0).unwrap();

        let mut zone_control = ZoneControl::new(thermal_model.clone(), setpoints);
        let current_temps = VectorField::from_scalar(28.0, 1);

        let energy_input = zone_control.update_zone_controls(&current_temps);

        assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Cooling);
        assert!(energy_input.as_slice()[0] > 0.0);
    }

    #[test]
    fn test_deadband_control() {
        let thermal_model = Arc::new(ThermalModel::new(1));
        let mut setpoints = crate::sim::hvac::zones::zone_setpoints::ZoneSetpoints::new(1);
        setpoints.set_heating_setpoint(0, 22.0).unwrap();
        setpoints.set_cooling_setpoint(0, 26.0).unwrap();
        setpoints.set_deadband(0, 2.0).unwrap();

        let mut zone_control = ZoneControl::new(thermal_model.clone(), setpoints);
        let current_temps = VectorField::from_scalar(23.0, 1);

        let energy_input = zone_control.update_zone_controls(&current_temps);

        assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Off);
        assert_eq!(energy_input.as_slice()[0], 0.0);
    }

    #[test]
    fn test_independent_zone_control() {
        let thermal_model = Arc::new(ThermalModel::new(2));
        let mut setpoints = crate::sim::hvac::zones::zone_setpoints::ZoneSetpoints::new(2);
        setpoints.set_heating_setpoint(0, 22.0).unwrap();
        setpoints.set_cooling_setpoint(0, 26.0).unwrap();
        setpoints.set_heating_setpoint(1, 18.0).unwrap();
        setpoints.set_cooling_setpoint(1, 22.0).unwrap();

        let mut zone_control = ZoneControl::new(thermal_model.clone(), setpoints);
        let current_temps = VectorField::new(vec![19.0, 25.0]); // Zone 0: heating, Zone 1: cooling

        let energy_input = zone_control.update_zone_controls(&current_temps);

        assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Heating);
        assert_eq!(zone_control.get_zone_hvac_status(1), HVACStatus::Cooling);
        assert!(energy_input.as_slice()[0] > 0.0);
        assert!(energy_input.as_slice()[1] > 0.0);
    }

    #[test]
    fn test_energy_calculation_ideal_loads() {
        let thermal_model = Arc::new(ThermalModel::new(1));
        let mut setpoints = crate::sim::hvac::zones::zone_setpoints::ZoneSetpoints::new(1);
        setpoints.set_heating_setpoint(0, 22.0).unwrap();
        setpoints.set_cooling_setpoint(0, 26.0).unwrap();

        let mut zone_control = ZoneControl::new(thermal_model.clone(), setpoints);
        let current_temps = VectorField::from_scalar(18.0, 1);

        let energy_input = zone_control.update_zone_controls(&current_temps);

        // Ideal loads calculation: Q = m_dot * cp * delta_t
        // With zone_volume=129.6, ACH=0.5, supply_heating_temp=40°C
        // airflow = 129.6 * 0.5 / 3600 = 0.018 m³/s
        // mass_flow = 0.018 * 1.2 = 0.0216 kg/s
        // delta_t = 40 - 18 = 22°C
        // Q = 0.0216 * 1005 * 22 = ~477 W (thermal)
        // Electrical = 477 / 0.9 = ~530 W
        assert!(energy_input.as_slice()[0] > 0.0);
    }

    #[test]
    fn test_hvac_status_transitions() {
        let thermal_model = Arc::new(ThermalModel::new(1));
        let mut setpoints = crate::sim::hvac::zones::zone_setpoints::ZoneSetpoints::new(1);
        setpoints.set_heating_setpoint(0, 22.0).unwrap();
        setpoints.set_cooling_setpoint(0, 26.0).unwrap();

        let mut zone_control = ZoneControl::new(thermal_model.clone(), setpoints);

        // Start with heating
        let mut current_temps = VectorField::from_scalar(20.0, 1);
        zone_control.update_zone_controls(&current_temps);
        assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Heating);

        // Transition to deadband
        current_temps = VectorField::from_scalar(23.0, 1);
        zone_control.update_zone_controls(&current_temps);
        assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Off);

        // Transition to cooling (above cooling threshold of 27.0°C)
        current_temps = VectorField::from_scalar(27.1, 1);
        zone_control.update_zone_controls(&current_temps);
        assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Cooling);
    }

    // ==================== Layered Controller Tests ====================

    #[test]
    fn test_layered_controller_ideal_loads_strategy() {
        let config = LayeredControllerConfig {
            strategy: ControlStrategy::IdealLoads,
            zone_volume: 129.6,
            air_changes_per_hour: 0.5,
            supply_cooling_temp: 13.0,
            supply_heating_temp: 40.0,
            cooling_cop: 3.0,
            heating_efficiency: 0.9,
            ..Default::default()
        };
        let controller = LayeredController::with_config(config);

        assert_eq!(controller.config.strategy, ControlStrategy::IdealLoads);
        assert_eq!(controller.config.zone_volume, 129.6);
    }

    #[test]
    fn test_layered_controller_staged_equipment_strategy() {
        let config = LayeredControllerConfig {
            strategy: ControlStrategy::StagedEquipment,
            zone_volume: 129.6,
            air_changes_per_hour: 0.5,
            min_runtime_timesteps: 5,
            startup_penalty_kwh: 0.1,
            ..Default::default()
        };
        let controller = LayeredController::with_config(config);

        assert_eq!(controller.config.strategy, ControlStrategy::StagedEquipment);
    }

    #[test]
    fn test_layered_controller_schedule_aware_strategy() {
        let config = LayeredControllerConfig {
            strategy: ControlStrategy::ScheduleAware,
            zone_volume: 129.6,
            air_changes_per_hour: 0.5,
            ..Default::default()
        };
        let controller = LayeredController::with_config(config);

        assert_eq!(controller.config.strategy, ControlStrategy::ScheduleAware);
    }

    #[test]
    fn test_ideal_loads_heating_energy() {
        let mut controller = LayeredController::new();
        controller.config.strategy = ControlStrategy::IdealLoads;
        controller.config.zone_volume = 129.6;
        controller.config.air_changes_per_hour = 0.5;
        controller.config.supply_heating_temp = 40.0;
        controller.config.heating_efficiency = 0.9;

        let energy = controller.calculate_energy_input(0, 18.0, &HVACStatus::Heating, 22.0, 26.0);

        // Energy should be positive for heating
        assert!(energy > 0.0);
    }

    #[test]
    fn test_ideal_loads_cooling_energy() {
        let mut controller = LayeredController::new();
        controller.config.strategy = ControlStrategy::IdealLoads;
        controller.config.zone_volume = 129.6;
        controller.config.air_changes_per_hour = 0.5;
        controller.config.supply_cooling_temp = 13.0;
        controller.config.cooling_cop = 3.0;

        let energy = controller.calculate_energy_input(0, 28.0, &HVACStatus::Cooling, 22.0, 26.0);

        // Energy should be positive for cooling
        assert!(energy > 0.0);
    }

    #[test]
    fn test_ideal_loads_off_energy() {
        let mut controller = LayeredController::new();
        controller.config.strategy = ControlStrategy::IdealLoads;

        let energy = controller.calculate_energy_input(0, 23.0, &HVACStatus::Off, 22.0, 26.0);

        // Energy should be zero when off
        assert_eq!(energy, 0.0);
    }

    #[test]
    fn test_staged_equipment_cycling_losses() {
        let mut controller = LayeredController::new();
        controller.config.strategy = ControlStrategy::StagedEquipment;
        controller.config.zone_volume = 129.6;
        controller.config.air_changes_per_hour = 0.5;
        controller.config.min_runtime_timesteps = 5;

        // First call - startup
        let energy1 = controller.calculate_energy_input(0, 18.0, &HVACStatus::Heating, 22.0, 26.0);

        // Second call - still running within min runtime
        let energy2 = controller.calculate_energy_input(0, 18.0, &HVACStatus::Heating, 22.0, 26.0);

        // Both should have energy
        assert!(energy1 > 0.0);
        assert!(energy2 > 0.0);
    }

    #[test]
    fn test_schedule_aware_predictive_control() {
        let mut controller = LayeredController::new();
        controller.config.strategy = ControlStrategy::ScheduleAware;
        controller.config.zone_volume = 129.6;
        controller.config.air_changes_per_hour = 0.5;
        controller.config.supply_heating_temp = 40.0;
        controller.config.heating_efficiency = 0.9;

        // First call - establish baseline
        let energy1 = controller.calculate_energy_input(0, 18.0, &HVACStatus::Heating, 22.0, 26.0);

        // Second call with rising temperature - should reduce modulation
        controller.previous_zone_temp = 17.0; // Simulate rising temp
        let energy2 = controller.calculate_energy_input(0, 18.0, &HVACStatus::Heating, 22.0, 26.0);

        assert!(energy1 > 0.0);
        assert!(energy2 > 0.0);
    }

    #[test]
    fn test_zone_control_with_strategy() {
        let thermal_model = Arc::new(ThermalModel::new(1));
        let mut setpoints = crate::sim::hvac::zones::zone_setpoints::ZoneSetpoints::new(1);
        setpoints.set_heating_setpoint(0, 22.0).unwrap();
        setpoints.set_cooling_setpoint(0, 26.0).unwrap();

        let zone_control = ZoneControl::with_strategy(
            thermal_model.clone(),
            setpoints,
            ControlStrategy::StagedEquipment,
        );

        assert_eq!(
            zone_control.get_zone_strategy(0),
            Some(ControlStrategy::StagedEquipment)
        );
    }

    #[test]
    fn test_zone_control_set_zone_strategy() {
        let thermal_model = Arc::new(ThermalModel::new(1));
        let setpoints = crate::sim::hvac::zones::zone_setpoints::ZoneSetpoints::new(1);

        let mut zone_control = ZoneControl::new(thermal_model.clone(), setpoints);

        assert_eq!(
            zone_control.get_zone_strategy(0),
            Some(ControlStrategy::IdealLoads)
        );

        zone_control.set_zone_strategy(0, ControlStrategy::StagedEquipment);
        assert_eq!(
            zone_control.get_zone_strategy(0),
            Some(ControlStrategy::StagedEquipment)
        );
    }

    #[test]
    fn test_layered_controller_reset() {
        let mut controller = LayeredController::new();
        controller.previous_zone_temp = 25.0;
        controller.prev_heating_setpoint = 18.0;

        controller.reset();

        assert_eq!(controller.previous_zone_temp, 20.0);
        assert_eq!(controller.prev_heating_setpoint, 20.0);
        assert_eq!(controller.prev_cooling_setpoint, 24.0);
    }

    #[test]
    fn test_ashrae_140_standard_values() {
        // ASHRAE 140 standard zone: 8m × 6m × 2.7m = 129.6 m³
        let config = LayeredControllerConfig {
            strategy: ControlStrategy::IdealLoads,
            zone_volume: 129.6,
            air_changes_per_hour: 0.5,
            supply_cooling_temp: 13.0,
            supply_heating_temp: 40.0,
            cooling_cop: 3.0,
            heating_efficiency: 0.9,
            ..Default::default()
        };
        let controller = LayeredController::with_config(config);

        assert_eq!(controller.config.cooling_cop, 3.0);
        assert_eq!(controller.config.heating_efficiency, 0.9);
        assert_eq!(controller.config.zone_volume, 129.6);
    }

    #[test]
    fn test_zone_volume_affects_ideal_loads() {
        let mut small_controller = LayeredController::new();
        small_controller.config.zone_volume = 50.0; // Smaller zone
        small_controller.config.air_changes_per_hour = 0.5;
        small_controller.config.supply_heating_temp = 40.0;
        small_controller.config.heating_efficiency = 0.9;

        let mut large_controller = LayeredController::new();
        large_controller.config.zone_volume = 200.0; // Larger zone
        large_controller.config.air_changes_per_hour = 0.5;
        large_controller.config.supply_heating_temp = 40.0;
        large_controller.config.heating_efficiency = 0.9;

        let small_energy =
            small_controller.calculate_energy_input(0, 18.0, &HVACStatus::Heating, 22.0, 26.0);
        let large_energy =
            large_controller.calculate_energy_input(0, 18.0, &HVACStatus::Heating, 22.0, 26.0);

        // Larger zone should have higher energy demand
        assert!(large_energy > small_energy);
    }

    #[test]
    fn test_ach_affects_ideal_loads() {
        let mut low_ach_controller = LayeredController::new();
        low_ach_controller.config.zone_volume = 129.6;
        low_ach_controller.config.air_changes_per_hour = 0.5;
        low_ach_controller.config.supply_heating_temp = 40.0;
        low_ach_controller.config.heating_efficiency = 0.9;

        let mut high_ach_controller = LayeredController::new();
        high_ach_controller.config.zone_volume = 129.6;
        high_ach_controller.config.air_changes_per_hour = 2.0; // Higher ACH
        high_ach_controller.config.supply_heating_temp = 40.0;
        high_ach_controller.config.heating_efficiency = 0.9;

        let low_ach_energy =
            low_ach_controller.calculate_energy_input(0, 18.0, &HVACStatus::Heating, 22.0, 26.0);
        let high_ach_energy =
            high_ach_controller.calculate_energy_input(0, 18.0, &HVACStatus::Heating, 22.0, 26.0);

        // Higher ACH should have higher energy demand
        assert!(high_ach_energy > low_ach_energy);
    }

    #[test]
    fn test_control_strategy_default() {
        assert_eq!(ControlStrategy::default(), ControlStrategy::IdealLoads);
    }

    #[test]
    fn test_layered_controller_config_default() {
        let config = LayeredControllerConfig::default();
        assert_eq!(config.strategy, ControlStrategy::default());
        assert_eq!(config.zone_volume, 129.6);
        assert_eq!(config.air_changes_per_hour, 0.5);
        assert_eq!(config.cooling_cop, 3.0);
        assert_eq!(config.heating_efficiency, 0.9);
    }
}
