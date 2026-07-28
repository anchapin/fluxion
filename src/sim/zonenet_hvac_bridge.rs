//! Bridge between ZoneNet ([`MultiZoneAirflowNetwork`]) and MultiZone HVAC system model.
//!
//! This module provides the integration layer that connects the N-zone inter-zone
//! thermal coupling network with HVAC systems that serve multiple zones. The bridge
//! handles:
//!
//! - Converting HVAC heating/cooling demands into external heat inputs (`q_ext`)
//!   for the [`MultiZoneAirflowNetwork::solve_step`] call
//! - Propagating inter-zone heat transfer results back to the HVAC system
//! - Moisture/humidity tracking where occupancy-driven latent gains affect zone RH
//!
//! ## Sign Convention
//!
//! - Positive `q_hvac` = heating demand (heat added to zone)
//! - Negative `q_hvac` = cooling demand (heat removed from zone)
//! - Positive `q_latent` = latent heat gain (humidification, e.g., from occupancy)
//! - Negative `q_latent` = latent heat loss (dehumidification)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                      ZoneNetHvacBridge                            │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  MultiZoneAirflowNetwork  ←── q_ext (HVAC + internal gains)       │
//! │         ↓ solve_step()                                            │
//! │  InterZoneResult ←── q_iz_w (inter-zone transfers)                │
//! │         ↓                                                         │
//! │  Zone HVAC Controller  ←── updated temperatures + loads             │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Moisture Balance
//!
//! Latent gains from occupancy are tracked separately and contribute to zone
//! humidity ratio. The moisture balance equation:
//!
//! ```text
//! C_w · dω/dt = m_dot_vent · (ω_outdoor − ω_zone) + q_latent / h_fg
//! ```
//!
//! where:
//! - `C_w` = air moisture capacity [kg/K]
//! - `ω` = humidity ratio [kg_water/kg_dry_air]
//! - `m_dot_vent` = ventilation mass flow rate [kg_dry_air/s]
//! - `h_fg` = latent heat of vaporization [J/kg]

use crate::sim::multi_zone_network::{InterZoneResult, MultiZoneAirflowNetwork, ZoneState};
use crate::sim::occupancy::OccupancyProfile;
use nalgebra::DMatrix;

/// Configuration for the ZoneNet-HVAC bridge.
#[derive(Debug, Clone)]
pub struct ZoneNetHvacBridgeConfig {
    /// Number of zones in the network.
    pub num_zones: usize,
    /// Timestep duration in seconds.
    pub dt_seconds: f64,
    /// Supply air temperature for heating [°C].
    pub supply_heating_temp: f64,
    /// Supply air temperature for cooling [°C].
    pub supply_cooling_temp: f64,
    /// Zone air volume [m³].
    pub zone_volumes: Vec<f64>,
    /// HVAC enabled flag per zone.
    pub hvac_enabled: Vec<bool>,
}

impl ZoneNetHvacBridgeConfig {
    /// Validate the bridge configuration.
    pub fn validate(&self) -> Result<(), ZoneNetHvacBridgeError> {
        if self.num_zones == 0 {
            return Err(ZoneNetHvacBridgeError::InvalidZoneCount(self.num_zones));
        }
        if self.zone_volumes.len() != self.num_zones {
            return Err(ZoneNetHvacBridgeError::ZoneVolumeMismatch {
                expected: self.num_zones,
                got: self.zone_volumes.len(),
            });
        }
        if self.hvac_enabled.len() != self.num_zones {
            return Err(ZoneNetHvacBridgeError::HvacEnabledMismatch {
                expected: self.num_zones,
                got: self.hvac_enabled.len(),
            });
        }
        if self.dt_seconds <= 0.0 {
            return Err(ZoneNetHvacBridgeError::InvalidTimestep(self.dt_seconds));
        }
        Ok(())
    }
}

/// Errors that can occur in the ZoneNet-HVAC bridge.
#[derive(Debug, Clone, PartialEq)]
pub enum ZoneNetHvacBridgeError {
    /// Zone count must be positive.
    InvalidZoneCount(usize),
    /// Zone volume slice length mismatch.
    ZoneVolumeMismatch { expected: usize, got: usize },
    /// HVAC enabled flag slice length mismatch.
    HvacEnabledMismatch { expected: usize, got: usize },
    /// Timestep must be positive.
    InvalidTimestep(f64),
    /// Multi-zone network error propagated.
    NetworkError(String),
    /// Moisture balance error.
    MoistureBalanceError(String),
}

impl std::fmt::Display for ZoneNetHvacBridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidZoneCount(n) => write!(f, "zone count must be positive, got {n}"),
            Self::ZoneVolumeMismatch { expected, got } => write!(
                f,
                "zone_volumes length {got} does not match num_zones {expected}"
            ),
            Self::HvacEnabledMismatch { expected, got } => write!(
                f,
                "hvac_enabled length {got} does not match num_zones {expected}"
            ),
            Self::InvalidTimestep(dt) => write!(f, "timestep {dt} must be positive"),
            Self::NetworkError(msg) => write!(f, "network error: {msg}"),
            Self::MoistureBalanceError(msg) => write!(f, "moisture balance error: {msg}"),
        }
    }
}

impl std::error::Error for ZoneNetHvacBridgeError {}

/// Result of a bridged HVAC step.
#[derive(Debug, Clone)]
pub struct BridgedHvacResult {
    /// Inter-zone heat transfer results.
    pub inter_zone: InterZoneResult,
    /// HVAC heating demand per zone [W].
    pub hvac_heating_w: Vec<f64>,
    /// HVAC cooling demand per zone [W].
    pub hvac_cooling_w: Vec<f64>,
    /// Zone humidity ratios after the step [kg_water/kg_dry_air].
    pub humidity_ratios: Vec<f64>,
    /// Zone relative humidities after the step [%].
    pub relative_humidities: Vec<f64>,
}

/// Per-zone internal gains including latent heat from occupancy.
#[derive(Debug, Clone, Default)]
pub struct InternalGains {
    /// Sensible convective gain [W].
    pub sensible_convective_w: f64,
    /// Sensible radiative gain [W].
    pub sensible_radiative_w: f64,
    /// Latent heat gain [W].
    pub latent_w: f64,
}

impl InternalGains {
    /// Total sensible gain [W].
    pub fn sensible_total_w(&self) -> f64 {
        self.sensible_convective_w + self.sensible_radiative_w
    }

    /// Total gain (sensible + latent) [W].
    pub fn total_w(&self) -> f64 {
        self.sensible_total_w() + self.latent_w
    }
}

/// Bridge between ZoneNet (MultiZoneAirflowNetwork) and MultiZone HVAC systems.
///
/// This struct manages the data and computation needed to couple the inter-zone
/// thermal network with HVAC heating/cooling delivery. It maintains:
///
/// - The multi-zone airflow network state
/// - Current zone temperatures and humidity ratios
/// - Zone HVAC controllers
/// - Moisture balance state
pub struct ZoneNetHvacBridge {
    network: MultiZoneAirflowNetwork,
    config: ZoneNetHvacBridgeConfig,
    /// Current zone states (temperature + heat capacity).
    zone_states: Vec<ZoneState>,
    /// Air moisture capacity per zone [kg/K] = ρ_air · V · (1 + ω).
    air_moisture_capacities: Vec<f64>,
    /// Current humidity ratios per zone [kg_water/kg_dry_air].
    humidity_ratios: Vec<f64>,
    /// Outdoor humidity ratio [kg_water/kg_dry_air].
    outdoor_humidity_ratio: f64,
    /// Heating setpoints per zone [°C].
    heating_setpoints: Vec<f64>,
    /// Cooling setpoints per zone [°C].
    cooling_setpoints: Vec<f64>,
}

impl ZoneNetHvacBridge {
    /// Create a new bridge from a conductance matrix and configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is invalid.
    pub fn new(
        h_tr_iz: DMatrix<f64>,
        config: ZoneNetHvacBridgeConfig,
    ) -> Result<Self, ZoneNetHvacBridgeError> {
        config.validate()?;
        let num_zones = config.num_zones;

        let zone_states = vec![ZoneState::new(20.0, config.dt_seconds * 100.0); num_zones];

        // Air moisture capacity (approximate): ρ_air · V
        let rho_air = 1.2;
        let air_moisture_capacities: Vec<f64> =
            config.zone_volumes.iter().map(|&v| rho_air * v).collect();

        let outdoor_humidity_ratio = 0.008;
        let humidity_ratios = vec![0.008; num_zones];

        let network = MultiZoneAirflowNetwork::from_matrix(h_tr_iz);

        Ok(Self {
            network,
            config,
            zone_states,
            air_moisture_capacities,
            humidity_ratios,
            outdoor_humidity_ratio,
            heating_setpoints: vec![20.0; num_zones],
            cooling_setpoints: vec![26.0; num_zones],
        })
    }

    /// Initialize zone temperatures and heat capacities.
    pub fn initialize_zones(&mut self, temperatures: &[f64], heat_capacities: &[f64]) {
        for (i, (t, &c)) in temperatures.iter().zip(heat_capacities.iter()).enumerate() {
            if i < self.zone_states.len() {
                self.zone_states[i].temperature = *t;
                self.zone_states[i].heat_capacity = c;
            }
        }
    }

    /// Set the outdoor humidity ratio.
    pub fn set_outdoor_humidity_ratio(&mut self, omega: f64) {
        self.outdoor_humidity_ratio = omega;
    }

    /// Set HVAC enabled flag for a zone.
    pub fn set_hvac_enabled(&mut self, zone: usize, enabled: bool) {
        if zone < self.config.hvac_enabled.len() {
            self.config.hvac_enabled[zone] = enabled;
        }
    }

    /// Set heating setpoint for a zone.
    pub fn set_heating_setpoint(&mut self, zone: usize, temp: f64) {
        if zone < self.heating_setpoints.len() {
            self.heating_setpoints[zone] = temp;
        }
    }

    /// Set cooling setpoint for a zone.
    pub fn set_cooling_setpoint(&mut self, zone: usize, temp: f64) {
        if zone < self.cooling_setpoints.len() {
            self.cooling_setpoints[zone] = temp;
        }
    }

    /// Compute HVAC demand for a zone given its temperature and setpoints.
    fn compute_zone_hvac_demand(&self, zone_idx: usize, t_zone: f64) -> (f64, f64) {
        let h_sp = self.heating_setpoints[zone_idx];
        let c_sp = self.cooling_setpoints[zone_idx];

        // Simple ideal HVAC: demand proportional to temperature error
        // Using a nominal h_coeff of 100 W/K per zone
        let h_coeff = 100.0;

        let heating = if t_zone < h_sp {
            h_coeff * (h_sp - t_zone)
        } else {
            0.0
        };

        let cooling = if t_zone > c_sp {
            -h_coeff * (t_zone - c_sp)
        } else {
            0.0
        };

        (heating, cooling)
    }

    /// Compute internal gains from occupancy for a given hour.
    ///
    /// This combines sensible and latent heat gains from occupants into a
    /// single [`InternalGains`] struct.
    pub fn compute_occupancy_gains(
        &self,
        occupancy: &OccupancyProfile,
        hour_of_week: usize,
    ) -> InternalGains {
        InternalGains {
            sensible_convective_w: occupancy.convective_sensible_heat_per_person()
                * occupancy.occupancy_at(hour_of_week),
            sensible_radiative_w: occupancy.radiative_sensible_heat_per_person()
                * occupancy.occupancy_at(hour_of_week),
            latent_w: occupancy.latent_heat_per_person * occupancy.occupancy_at(hour_of_week),
        }
    }

    /// Compute moisture balance for a zone.
    ///
    /// Returns the new humidity ratio after one timestep.
    ///
    /// # Arguments
    /// * `zone_idx` - Zone index
    /// * `q_latent` - Latent heat gain [W] (positive = humidification)
    /// * `ventilation_ach` - Ventilation rate [ACH]
    ///
    /// # Returns
    /// New humidity ratio [kg_water/kg_dry_air]
    fn compute_moisture_balance(
        &self,
        zone_idx: usize,
        q_latent: f64,
        ventilation_ach: f64,
    ) -> Result<f64, ZoneNetHvacBridgeError> {
        let c_w = self.air_moisture_capacities[zone_idx];
        if c_w <= 0.0 {
            return Err(ZoneNetHvacBridgeError::MoistureBalanceError(format!(
                "zone {zone_idx} has non-positive moisture capacity {c_w}"
            )));
        }

        // Latent heat of vaporization at 20°C [J/kg]
        let h_fg = 2.45e6;

        // Ventilation mass flow rate [kg_dry_air/s]
        // ACH → m³/s: ACH / 3600
        // ρ_air * V * ACH / 3600
        let rho_air = 1.2;
        let m_dot_vent = rho_air * self.config.zone_volumes[zone_idx] * ventilation_ach / 3600.0;

        // Moisture production rate from latent heat [kg_water/s]
        let m_dot_production = q_latent / h_fg;

        // Moisture accumulation term
        let moisture_storage = m_dot_production * self.config.dt_seconds;

        // Ventilation moisture transfer
        let ventilation_transfer = m_dot_vent
            * (self.outdoor_humidity_ratio - self.humidity_ratios[zone_idx])
            * self.config.dt_seconds;

        // Update humidity ratio
        let omega_new =
            self.humidity_ratios[zone_idx] + (moisture_storage + ventilation_transfer) / c_w;

        // Clamp to physically reasonable range [0.001, 0.030] kg/kg
        let omega_clamped = omega_new.clamp(0.001, 0.030);

        Ok(omega_clamped)
    }

    /// Run one coupled timestep step.
    ///
    /// This method:
    /// 1. Computes HVAC heating/cooling demand for each zone
    /// 2. Computes latent gains from occupancy
    /// 3. Builds the external heat vector `q_ext` = HVAC + occupancy gains
    /// 4. Calls `MultiZoneAirflowNetwork::solve_step`
    /// 5. Updates humidity ratios via moisture balance
    /// 6. Returns the bridged result
    ///
    /// # Arguments
    /// * `occupancy` - Optional occupancy profile for latent gains
    /// * `hour_of_week` - Hour of week (0-167) for occupancy schedule
    /// * `ventilation_ach` - Ventilation rate [ACH] per zone
    ///
    /// # Errors
    /// Returns an error if the network solve fails.
    pub fn step(
        &mut self,
        occupancy: Option<&OccupancyProfile>,
        hour_of_week: usize,
        ventilation_ach: &[f64],
    ) -> Result<BridgedHvacResult, ZoneNetHvacBridgeError> {
        let n = self.config.num_zones;

        // Step 1: Compute HVAC demands and occupancy gains
        let mut q_ext = vec![0.0; n];
        let mut hvac_heating = vec![0.0; n];
        let mut hvac_cooling = vec![0.0; n];

        for i in 0..n {
            // HVAC demand
            let (heat, cool) = if self.config.hvac_enabled[i] {
                self.compute_zone_hvac_demand(i, self.zone_states[i].temperature)
            } else {
                (0.0, 0.0)
            };
            hvac_heating[i] = heat;
            hvac_cooling[i] = cool;

            // HVAC heat input to zone air node (convective)
            // Positive = heating, negative = cooling
            q_ext[i] += heat - cool;

            // Occupancy gains
            if let Some(occ) = occupancy {
                let gains = self.compute_occupancy_gains(occ, hour_of_week);
                // Convective sensible gains go directly to zone air
                q_ext[i] += gains.sensible_convective_w;

                // Update moisture balance for latent gains
                let vent_ach = ventilation_ach.get(i).copied().unwrap_or(0.5);
                match self.compute_moisture_balance(i, gains.latent_w, vent_ach) {
                    Ok(omega_new) => {
                        self.humidity_ratios[i] = omega_new;
                    }
                    Err(e) => {
                        return Err(e);
                    }
                }
            }
        }

        // Step 2: Run the multi-zone network solve
        let inter_zone_result = self
            .network
            .solve_step(&mut self.zone_states, &q_ext, self.config.dt_seconds)
            .map_err(|e| ZoneNetHvacBridgeError::NetworkError(e.to_string()))?;

        // Step 3: Compute relative humidities from humidity ratios
        // Using approximate relationship: RH ≈ ω / ω_sat(T)
        // where ω_sat(20°C) ≈ 0.0148 kg/kg
        let omega_sat_20c = 0.0148;
        let relative_humidities: Vec<f64> = self
            .humidity_ratios
            .iter()
            .map(|&omega| (omega / omega_sat_20c * 100.0).clamp(0.0, 100.0))
            .collect();

        Ok(BridgedHvacResult {
            inter_zone: inter_zone_result,
            hvac_heating_w: hvac_heating,
            hvac_cooling_w: hvac_cooling,
            humidity_ratios: self.humidity_ratios.clone(),
            relative_humidities,
        })
    }

    /// Get current zone temperatures.
    pub fn zone_temperatures(&self) -> Vec<f64> {
        self.zone_states.iter().map(|z| z.temperature).collect()
    }

    /// Get current humidity ratios.
    pub fn humidity_ratios(&self) -> &[f64] {
        &self.humidity_ratios
    }

    /// Access the underlying network for conservation checking.
    pub fn network(&self) -> &MultiZoneAirflowNetwork {
        &self.network
    }

    /// Generate a conservation report for the current state.
    pub fn conservation_report(&self) -> crate::sim::multi_zone_network::MultiZoneNetworkReport {
        self.network.conservation_report()
    }
}

/// Create a symmetric fully-connected conductance matrix for N zones.
///
/// This is a convenience constructor for building a uniform N×N inter-zone
/// conductance matrix where every zone pair has the same conductance.
pub fn fully_connected_conductance(n: usize, h_tr_ij: f64) -> DMatrix<f64> {
    let mut m = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if i != j {
                m[(i, j)] = h_tr_ij;
            }
        }
    }
    m
}

/// Create a ring conductance matrix where each zone connects only to its neighbors.
///
/// For a ring of N zones, each zone is connected to its two neighbors with
/// conductance `h_tr`.
pub fn ring_conductance(n: usize, h_tr: f64) -> DMatrix<f64> {
    let mut m = DMatrix::<f64>::zeros(n, n);
    if n < 2 {
        return m;
    }
    for i in 0..n {
        let j_next = (i + 1) % n;
        let j_prev = if i == 0 { n - 1 } else { i - 1 };
        m[(i, j_next)] = h_tr;
        m[(j_next, i)] = h_tr;
        m[(i, j_prev)] = h_tr;
        m[(j_prev, i)] = h_tr;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(n: usize) -> ZoneNetHvacBridgeConfig {
        ZoneNetHvacBridgeConfig {
            num_zones: n,
            dt_seconds: 3600.0,
            supply_heating_temp: 40.0,
            supply_cooling_temp: 13.0,
            zone_volumes: vec![129.6; n], // 8m × 6m × 2.7m
            hvac_enabled: vec![true; n],
        }
    }

    #[test]
    fn test_bridge_initialization() {
        let h = fully_connected_conductance(3, 50.0);
        let config = make_config(3);
        let bridge = ZoneNetHvacBridge::new(h, config).expect("valid config");
        assert_eq!(bridge.zone_temperatures().len(), 3);
    }

    #[test]
    fn test_bridge_step_conserves_energy() {
        let h = fully_connected_conductance(3, 50.0);
        let mut config = make_config(3);
        config.hvac_enabled = vec![false; 3]; // Disable HVAC to isolate inter-zone transfer
        let mut bridge = ZoneNetHvacBridge::new(h, config).expect("valid config");

        // Initialize zones at different temperatures
        bridge.initialize_zones(&[20.0, 25.0, 15.0], &[1.0e6, 1.0e6, 1.0e6]);

        let result = bridge
            .step(None, 0, &[0.5, 0.5, 0.5])
            .expect("step should succeed");

        // Energy conservation: sum of inter-zone transfers should be ~0
        assert!(
            result.inter_zone.net_w.abs() < 1e-6,
            "inter-zone transfer should conserve energy, got net_w = {}",
            result.inter_zone.net_w
        );
    }

    #[test]
    fn test_occupancy_spike_increases_latent_gain() {
        let h = fully_connected_conductance(2, 30.0);
        let config = make_config(2);
        let bridge = ZoneNetHvacBridge::new(h, config).expect("valid config");

        // Create a simple occupancy profile
        let occupancy = OccupancyProfile::new(
            "test".to_string(),
            crate::sim::occupancy::BuildingType::Office,
            10.0,
        )
        .office_schedule();

        // Morning hour (hour 9 AM Monday = 9 + 24 = 33)
        let hour_morning = 33;
        let gains_morning = bridge.compute_occupancy_gains(&occupancy, hour_morning);

        // There should be latent gain from occupancy
        assert!(
            gains_morning.latent_w > 0.0,
            "occupancy should produce latent gain, got {} W",
            gains_morning.latent_w
        );

        // Sensible gains should also be positive
        assert!(
            gains_morning.sensible_total_w() > 0.0,
            "occupancy should produce sensible gain, got {} W",
            gains_morning.sensible_total_w()
        );
    }

    #[test]
    fn test_hvac_demand_heating() {
        let h = fully_connected_conductance(1, 0.0);
        let config = make_config(1);
        let bridge = ZoneNetHvacBridge::new(h, config).expect("valid config");

        // Zone at 18°C, heating setpoint 20°C
        // Should demand heating
        let (heat, cool) = bridge.compute_zone_hvac_demand(0, 18.0);
        assert!(heat > 0.0, "should demand heating when below setpoint");
        assert_eq!(cool, 0.0, "should not demand cooling");
    }

    #[test]
    fn test_hvac_demand_cooling() {
        let h = fully_connected_conductance(1, 0.0);
        let config = make_config(1);
        let bridge = ZoneNetHvacBridge::new(h, config).expect("valid config");

        // Zone at 28°C, cooling setpoint 26°C
        // Should demand cooling
        let (heat, cool) = bridge.compute_zone_hvac_demand(0, 28.0);
        assert_eq!(heat, 0.0, "should not demand heating");
        assert!(cool < 0.0, "should demand cooling when above setpoint");
    }

    #[test]
    fn test_moisture_balance_during_occupancy() {
        let h = fully_connected_conductance(2, 30.0);
        let mut config = make_config(2);
        config.hvac_enabled = vec![false; 2]; // Disable HVAC
        let mut bridge = ZoneNetHvacBridge::new(h, config).expect("valid config");

        // Initialize at 20°C
        bridge.initialize_zones(&[20.0, 20.0], &[1.0e6, 1.0e6]);

        let occupancy = OccupancyProfile::new(
            "test".to_string(),
            crate::sim::occupancy::BuildingType::Office,
            10.0,
        )
        .office_schedule();

        // Initial humidity ratio
        let omega_initial = bridge.humidity_ratios()[0];

        // Step with occupancy (produces latent heat)
        let result = bridge
            .step(Some(&occupancy), 33, &[0.5, 0.5])
            .expect("step should succeed");

        // Humidity ratio should have increased due to latent gains from occupancy
        let omega_after = result.humidity_ratios[0];
        assert!(
            omega_after > omega_initial,
            "humidity ratio should increase with occupancy latent gain: {} -> {}",
            omega_initial,
            omega_after
        );

        // Relative humidity should also reflect the increase
        assert!(
            result.relative_humidities[0] > 30.0,
            "RH should be reasonable after occupancy spike"
        );
    }

    #[test]
    fn test_ring_conductance_symmetry() {
        let h = ring_conductance(4, 25.0);
        // Ring should be symmetric
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    h[(i, j)],
                    h[(j, i)],
                    "conductance matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_invalid_config_rejected() {
        let h = fully_connected_conductance(2, 30.0);
        let mut config = make_config(2);
        config.num_zones = 0; // Invalid
        let result = ZoneNetHvacBridge::new(h, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_fully_connected_conductance() {
        let h = fully_connected_conductance(3, 50.0);
        // Off-diagonal should be 50.0, diagonal should be 0.0
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(h[(i, j)], 0.0);
                } else {
                    assert_eq!(h[(i, j)], 50.0);
                }
            }
        }
    }
}
