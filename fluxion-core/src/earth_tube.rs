//! Earth tube (ground-air heat exchanger) model.
//!
//! Earth tubes are buried pipes that pre-heat or pre-cool intake ventilation air
//! using the ground's thermal mass. At burial depths of 2-3 meters, ground temperature
//! remains relatively stable year-round, approximately equal to the annual average
//! outdoor air temperature with significantly damped seasonal amplitude.
//!
//! # Physics Model
//!
//! The model uses the NTU-effectiveness method for heat exchangers:
//!
//! ```text
//! Q = C_p × ρ × V̇ × (T_ground - T_air) × (1 - exp(-NTU))
//! ```
//!
//! Where:
//! - `C_p` = specific heat of air (1005 J/kg·K)
//! - `ρ` = density of air (1.2 kg/m³)
//! - `V̇` = volumetric flow rate (m³/s)
//! - `NTU` = Number of Transfer Units = UA / C_min
//! - `UA` = overall heat transfer coefficient × area
//!
//! # Earth Tube Heat Transfer Components
//!
//! 1. **Inside convection** (air to pipe wall): Dittus-Boelter correlation
//! 2. **Pipe wall conduction**: cylindrical thermal resistance
//! 3. **Soil conduction**: cylindrical heat transfer to infinite medium
//!
//! # References
//!
//! - EnergyPlus Engineering Reference: EarthTube object
//! - ASHRAE Handbook 2021 Chapter 26 (Ventilation)
//! - Incropera et al., Fundamentals of Heat and Mass Transfer

use serde::{Deserialize, Serialize};

/// Standard air density at typical conditions (kg/m³).
const RHO_AIR: f64 = 1.2;

/// Standard air specific heat capacity (J/kg·K).
const CP_AIR: f64 = 1005.0;

/// Air thermal conductivity (W/m·K) at moderate temperatures.
const K_AIR: f64 = 0.025;

/// Prandtl number for air (dimensionless).
const PR_AIR: f64 = 0.71;

/// Air dynamic viscosity (Pa·s) at ~0°C (winter condition).
const MU_WINTER: f64 = 1.7e-5;

/// Air dynamic viscosity (Pa·s) at ~35°C (summer condition).
const MU_SUMMER: f64 = 1.85e-5;

/// Earth tube configuration and state.
///
/// # Example
///
/// ```
/// use fluxion_core::earth_tube::EarthTube;
///
/// // Typical residential earth tube: 6-inch PVC, 30m length, 2.5m burial depth
/// let et = EarthTube::new()
///     .burial_depth_m(2.5)
///     .pipe_diameter_m(0.15)
///     .pipe_length_m(30.0)
///     .flow_rate_m3_s(0.05)
///     .ground_conductivity(1.5)
///     .soil_temperature_K(285.15); // ~12°C
///
/// // Winter: outdoor -5°C → supply ~8°C (pre-heated by ~13°C)
/// let supply_K = et.supply_temperature(268.15);
/// let preheat = supply_K - 268.15;
/// assert!(preheat >= 5.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarthTube {
    /// Pipe burial depth (meters). Typical range: 1.5-3.0 m.
    /// Deeper burial gives more stable ground temperature.
    burial_depth_m: f64,

    /// Pipe inner diameter (meters). Typical: 0.10-0.20 m (4-8 inch).
    pipe_diameter_m: f64,

    /// Total pipe length (meters). Typical: 20-50 m.
    pipe_length_m: f64,

    /// Volumetric air flow rate (m³/s). Should match ventilation requirement.
    flow_rate_m3_s: f64,

    /// Soil thermal conductivity (W/m·K). Typical: 1.0-2.0 W/m·K.
    /// Sandy soil: ~0.4-1.0, Clay soil: ~1.0-2.0, Wet soil: ~2.0-4.0.
    ground_conductivity: f64,

    /// Ground temperature at burial depth (K).
    /// At 2-3m depth, this approximates annual average outdoor temperature
    /// (~10-15°C in temperate climates) with damped seasonal variation.
    soil_temperature_K: f64,

    /// Pipe wall thermal conductivity (W/m·K).
    /// PVC: ~0.16, HDPE: ~0.50, Steel: ~45.0.
    /// Defaults to PVC (common for earth tubes).
    pipe_conductivity: f64,

    /// Pipe wall thickness (m). Default: 5mm PVC.
    pipe_wall_thickness_m: f64,
}

impl EarthTube {
    /// Creates a new EarthTube with default parameters.
    ///
    /// Default configuration:
    /// - Burial depth: 2.5 m
    /// - Pipe diameter: 0.15 m (6 inch)
    /// - Pipe length: 30 m
    /// - Flow rate: 0.05 m³/s
    /// - Soil conductivity: 1.5 W/m·K
    /// - Soil temperature: 285.15 K (12°C)
    /// - Pipe conductivity: 0.16 W/m·K (PVC)
    /// - Pipe wall thickness: 0.005 m
    pub fn new() -> Self {
        Self {
            burial_depth_m: 2.5,
            pipe_diameter_m: 0.15,
            pipe_length_m: 30.0,
            flow_rate_m3_s: 0.05,
            ground_conductivity: 1.5,
            soil_temperature_K: 285.15,
            pipe_conductivity: 0.16,
            pipe_wall_thickness_m: 0.005,
        }
    }

    /// Sets the pipe burial depth in meters.
    pub fn burial_depth_m(mut self, depth: f64) -> Self {
        self.burial_depth_m = depth;
        self
    }

    /// Sets the pipe inner diameter in meters.
    pub fn pipe_diameter_m(mut self, diameter: f64) -> Self {
        self.pipe_diameter_m = diameter;
        self
    }

    /// Sets the total pipe length in meters.
    pub fn pipe_length_m(mut self, length: f64) -> Self {
        self.pipe_length_m = length;
        self
    }

    /// Sets the volumetric air flow rate in m³/s.
    pub fn flow_rate_m3_s(mut self, flow_rate: f64) -> Self {
        self.flow_rate_m3_s = flow_rate;
        self
    }

    /// Sets the soil thermal conductivity in W/m·K.
    pub fn ground_conductivity(mut self, conductivity: f64) -> Self {
        self.ground_conductivity = conductivity;
        self
    }

    /// Sets the ground temperature at burial depth in Kelvin.
    pub fn soil_temperature_K(mut self, temperature: f64) -> Self {
        self.soil_temperature_K = temperature;
        self
    }

    /// Sets the pipe wall thermal conductivity in W/m·K.
    pub fn pipe_conductivity(mut self, conductivity: f64) -> Self {
        self.pipe_conductivity = conductivity;
        self
    }

    /// Sets the pipe wall thickness in meters.
    pub fn pipe_wall_thickness_m(mut self, thickness: f64) -> Self {
        self.pipe_wall_thickness_m = thickness;
        self
    }

    /// Calculates the inside convection heat transfer coefficient.
    ///
    /// Uses Dittus-Boelter correlation for turbulent flow (Re > 2300):
    /// `h_i = 0.023 × Re^0.8 × Pr^0.4 × k / D`
    ///
    /// For laminar flow (Re ≤ 2300):
    /// `h_i = 3.66 × k / D`
    fn inside_convection_coefficient(&self, mu: f64) -> f64 {
        let d = self.pipe_diameter_m;
        let v = self.velocity();
        let re = RHO_AIR * v * d / mu;

        if re < 2300.0 {
            // Laminar flow - use analytical solution
            3.66 * K_AIR / d
        } else {
            // Turbulent flow - Dittus-Boelter
            0.023 * re.powf(0.8) * PR_AIR.powf(0.4) * K_AIR / d
        }
    }

    /// Calculates pipe wall thermal resistance (cylindrical, K/W).
    fn wall_thermal_resistance(&self) -> f64 {
        let d_inner = self.pipe_diameter_m;
        let d_outer = d_inner + 2.0 * self.pipe_wall_thickness_m;
        let l = self.pipe_length_m;
        let k_pipe = self.pipe_conductivity;

        // R_wall = ln(r_outer/r_inner) / (2 * π * k * L)
        (d_outer / d_inner).ln() / (2.0 * std::f64::consts::PI * k_pipe * l)
    }

    /// Calculates soil thermal resistance (cylindrical heat transfer to infinite medium).
    fn soil_thermal_resistance(&self) -> f64 {
        let d_inner = self.pipe_diameter_m;
        let d_outer = d_inner + 2.0 * self.pipe_wall_thickness_m;
        let l = self.pipe_length_m;
        let k_soil = self.ground_conductivity;

        // R_soil = ln(r_outer/r_inner) / (2 * π * k_soil * L)
        (d_outer / d_inner).ln() / (2.0 * std::f64::consts::PI * k_soil * l)
    }

    /// Calculates pipe inner surface area (m²).
    fn inner_surface_area(&self) -> f64 {
        std::f64::consts::PI * self.pipe_diameter_m * self.pipe_length_m
    }

    /// Calculates air velocity in pipe (m/s).
    fn velocity(&self) -> f64 {
        let a = std::f64::consts::PI / 4.0 * self.pipe_diameter_m.powi(2);
        self.flow_rate_m3_s / a
    }

    /// Calculates the UA product (overall heat transfer coefficient × area).
    fn ua(&self, mu: f64) -> f64 {
        let a_pipe = self.inner_surface_area();
        let h_i = self.inside_convection_coefficient(mu);
        let r_wall = self.wall_thermal_resistance();
        let r_soil = self.soil_thermal_resistance();

        // UA = 1 / (1/(h_i*A) + R_wall + R_soil)
        1.0 / (1.0 / (h_i * a_pipe) + r_wall + r_soil)
    }

    /// Calculates the heat capacity rate of the air stream (W/K).
    fn capacity_rate(&self) -> f64 {
        CP_AIR * RHO_AIR * self.flow_rate_m3_s
    }

    /// Calculates the NTU (Number of Transfer Units) for the heat exchanger.
    fn ntu(&self, mu: f64) -> f64 {
        self.ua(mu) / self.capacity_rate()
    }

    /// Calculates the heat exchanger effectiveness (dimensionless).
    ///
    /// For a single-stream heat exchanger (counter-flow approximation):
    /// `ε = 1 - exp(-NTU)`
    fn effectiveness(&self, mu: f64) -> f64 {
        let ntu = self.ntu(mu);
        1.0 - (-ntu).exp()
    }

    /// Calculates the supply air temperature after earth tube (Kelvin).
    ///
    /// Uses the NTU-effectiveness method:
    /// ```text
    /// T_supply = T_outdoor + ε × (T_ground - T_outdoor)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `outdoor_temp_K` - Outdoor air temperature entering the earth tube (K)
    ///
    /// # Returns
    ///
    /// Supply air temperature after earth tube (K)
    pub fn supply_temperature(&self, outdoor_temp_K: f64) -> f64 {
        // Zero flow rate: no temperature change
        if self.flow_rate_m3_s <= 0.0 {
            return outdoor_temp_K;
        }

        // Select viscosity based on whether heating or cooling
        let mu = if outdoor_temp_K < self.soil_temperature_K {
            MU_WINTER
        } else {
            MU_SUMMER
        };

        let eps = self.effectiveness(mu);
        outdoor_temp_K + eps * (self.soil_temperature_K - outdoor_temp_K)
    }

    /// Calculates the heat transfer rate through the earth tube (Watts).
    ///
    /// Positive = heat gain (heating mode), Negative = heat loss (cooling mode).
    ///
    /// # Arguments
    ///
    /// * `outdoor_temp_K` - Outdoor air temperature entering the earth tube (K)
    ///
    /// # Returns
    ///
    /// Heat transfer rate in Watts
    pub fn heat_transfer_rate(&self, outdoor_temp_K: f64) -> f64 {
        // Zero flow rate: no heat transfer
        if self.flow_rate_m3_s <= 0.0 {
            return 0.0;
        }

        // Select viscosity based on whether heating or cooling
        let mu = if outdoor_temp_K < self.soil_temperature_K {
            MU_WINTER
        } else {
            MU_SUMMER
        };

        let eps = self.effectiveness(mu);
        let c_dot = self.capacity_rate();
        c_dot * eps * (self.soil_temperature_K - outdoor_temp_K)
    }

    /// Returns the pre-heating or pre-cooling temperature difference (Kelvin).
    ///
    /// Positive = pre-heating (outdoor colder than ground)
    /// Negative = pre-cooling (outdoor warmer than ground)
    pub fn temperature_difference(&self, outdoor_temp_K: f64) -> f64 {
        self.supply_temperature(outdoor_temp_K) - outdoor_temp_K
    }
}

impl Default for EarthTube {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_earth_tube_winter_preheating() {
        // Winter test: outdoor -5°C, ground ~12°C → supply ≥ 5°C (pre-heated by ≥5°C)
        let et = EarthTube::new()
            .soil_temperature_K(285.15) // 12°C
            .flow_rate_m3_s(0.05);

        let outdoor_K = 268.15; // -5°C
        let supply_K = et.supply_temperature(outdoor_K);
        let preheat = supply_K - outdoor_K;

        println!(
            "Winter: outdoor={}°C, supply={}°C, preheat={}°C",
            outdoor_K - 273.15,
            supply_K - 273.15,
            preheat
        );

        assert!(
            preheat >= 5.0,
            "Winter pre-heating should be at least 5°C, got {}°C",
            preheat
        );
    }

    #[test]
    fn test_earth_tube_summer_precooling() {
        // Summer test: outdoor 35°C, ground ~18°C → supply ≤ 30°C (pre-cooled by ≥5°C)
        let et = EarthTube::new()
            .soil_temperature_K(291.15) // 18°C
            .flow_rate_m3_s(0.05);

        let outdoor_K = 308.15; // 35°C
        let supply_K = et.supply_temperature(outdoor_K);
        let precool = supply_K - outdoor_K; // negative

        println!(
            "Summer: outdoor={}°C, supply={}°C, precool={}°C",
            outdoor_K - 273.15,
            supply_K - 273.15,
            precool
        );

        assert!(
            precool <= -5.0,
            "Summer pre-cooling should be at least 5°C, got {}°C",
            precool.abs()
        );
    }

    #[test]
    fn test_earth_tube_energy_balance() {
        // Verify energy balance: Q = m_dot * cp * delta_T
        let et = EarthTube::new()
            .soil_temperature_K(285.15)
            .flow_rate_m3_s(0.05);

        let outdoor_K = 268.15; // -5°C
        let supply_K = et.supply_temperature(outdoor_K);
        let Q_calc = et.heat_transfer_rate(outdoor_K);

        let m_dot = RHO_AIR * et.flow_rate_m3_s;
        let Q_check = m_dot * CP_AIR * (supply_K - outdoor_K);

        println!("Q_calc = {} W, Q_check = {} W", Q_calc, Q_check);
        assert!(
            (Q_calc - Q_check).abs() < 0.1,
            "Energy balance check failed"
        );
    }

    #[test]
    fn test_earth_tube_supply_temp_bounds() {
        // Supply temp should always be between outdoor and ground temp
        let et = EarthTube::new();

        // Winter case (outdoor < ground)
        let outdoor_winter = 268.15; // -5°C
        let ground = et.soil_temperature_K; // ~12°C
        let supply_winter = et.supply_temperature(outdoor_winter);
        assert!(
            supply_winter > outdoor_winter && supply_winter < ground,
            "Winter supply should be between outdoor and ground"
        );

        // Summer case (outdoor > ground)
        let outdoor_summer = 308.15; // 35°C
        let supply_summer = et.supply_temperature(outdoor_summer);
        assert!(
            supply_summer < outdoor_summer && supply_summer > ground,
            "Summer supply should be between outdoor and ground"
        );
    }

    #[test]
    fn test_earth_tube_zero_flow() {
        // Zero flow rate should give no temperature change
        let et = EarthTube::new()
            .flow_rate_m3_s(0.0)
            .soil_temperature_K(285.15);

        let outdoor_K = 268.15;
        let supply_K = et.supply_temperature(outdoor_K);

        assert_eq!(
            supply_K, outdoor_K,
            "Zero flow should give no temperature change"
        );
    }

    #[test]
    fn test_earth_tube_flow_scaling() {
        // Higher flow rate → lower effectiveness (less contact time)
        let et_low = EarthTube::new().flow_rate_m3_s(0.02);
        let et_high = EarthTube::new().flow_rate_m3_s(0.20);

        let outdoor_K = 268.15;

        let preheat_low = et_low.supply_temperature(outdoor_K) - outdoor_K;
        let preheat_high = et_high.supply_temperature(outdoor_K) - outdoor_K;

        println!(
            "Low flow preheat: {}°C, High flow preheat: {}°C",
            preheat_low, preheat_high
        );
        assert!(
            preheat_low > preheat_high,
            "Lower flow should give more preheating"
        );
    }

    #[test]
    fn test_earth_tube_length_scaling() {
        // Longer pipe → higher effectiveness (more contact time)
        let et_short = EarthTube::new().pipe_length_m(15.0);
        let et_long = EarthTube::new().pipe_length_m(60.0);

        let outdoor_K = 268.15;

        let preheat_short = et_short.supply_temperature(outdoor_K) - outdoor_K;
        let preheat_long = et_long.supply_temperature(outdoor_K) - outdoor_K;

        println!(
            "Short pipe preheat: {}°C, Long pipe preheat: {}°C",
            preheat_short, preheat_long
        );
        assert!(
            preheat_long > preheat_short,
            "Longer pipe should give more preheating"
        );
    }

    #[test]
    fn test_earth_tube_depth_effect() {
        // Deeper burial → less extreme ground temp variation (but here we set absolute T)
        // With fixed soil temperature, deeper depth in constructor doesn't affect output
        // This test just verifies the builder pattern works
        let et = EarthTube::new()
            .burial_depth_m(3.0)
            .pipe_diameter_m(0.20)
            .pipe_length_m(40.0)
            .flow_rate_m3_s(0.08)
            .ground_conductivity(2.0)
            .soil_temperature_K(288.15)
            .pipe_conductivity(0.50)
            .pipe_wall_thickness_m(0.006);

        let outdoor_K = 268.15;
        let supply_K = et.supply_temperature(outdoor_K);

        println!("Custom config: supply={}°C", supply_K - 273.15);
        assert!(supply_K > outdoor_K);
    }

    #[test]
    fn test_earth_tube_ground_same_as_outdoor() {
        // If ground temp equals outdoor temp, no heat transfer
        let et = EarthTube::new()
            .soil_temperature_K(283.15)
            .flow_rate_m3_s(0.05);

        let outdoor_K = 283.15; // same as ground
        let supply_K = et.supply_temperature(outdoor_K);
        let Q = et.heat_transfer_rate(outdoor_K);

        assert_eq!(supply_K, outdoor_K, "Same temps should give same supply");
        assert_eq!(Q, 0.0, "No heat transfer when temps are equal");
    }

    #[test]
    fn test_earth_tube_negative_delta_t() {
        // Verify temperature difference is negative when outdoor > ground (cooling)
        let et = EarthTube::new()
            .soil_temperature_K(285.15)
            .flow_rate_m3_s(0.05);

        let outdoor_K = 308.15; // 35°C
        let delta = et.temperature_difference(outdoor_K);

        assert!(
            delta < 0.0,
            "Should be pre-cooling (negative delta), got {}K",
            delta
        );
    }

    #[test]
    fn test_earth_tube_positive_delta_t() {
        // Verify temperature difference is positive when outdoor < ground (heating)
        let et = EarthTube::new()
            .soil_temperature_K(285.15)
            .flow_rate_m3_s(0.05);

        let outdoor_K = 268.15; // -5°C
        let delta = et.temperature_difference(outdoor_K);

        assert!(
            delta > 0.0,
            "Should be pre-heating (positive delta), got {}K",
            delta
        );
    }
}
