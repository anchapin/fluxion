//! Sweep configuration — the central builder for Monte Carlo parameter
//! sweeps (Issue #1776, Task T5.1).
//!
//! [`SweepConfig`] aggregates all sampling distributions for building geometry,
//! insulation properties, and weather/climate conditions.  It can be
//! constructed from an existing [`SurrogateDomain`] (per epic #719 scope)
//! or built from scratch with the [`SweepConfigBuilder`].
//!
//! # Design decisions
//!
//! - **Driven by `SurrogateDomain`**: the weather/occupancy/zone-temp
//!   distributions inherit their bounds from the domain so that generated
//!   samples always fall within the surrogate model's valid input range.
//! - **Geometry & insulation are additional sweep dimensions**: the issue
//!   requires building-geometry and insulation sweeps that are not covered
//!   by `SurrogateDomain`.  These get sensible defaults derived from
//!   ASHRAE 90.1 and common residential typologies.
//! - **Discrete climate zones**: sampled uniformly from
//!   `SurrogateDomain.climate_zones` to ensure multi-climate coverage.
//! - **Reproducible**: every config carries a `seed` that is forwarded to
//!   the sampling strategies.

use crate::ai::surrogate::SurrogateDomain;
use crate::ai::sweeps::distributions::Choice;
use crate::ai::sweeps::distributions::ParameterDistribution;
use crate::ai::sweeps::sampling::SamplingStrategy;
use crate::ai::sweeps::weather::WeatherFileRegistry;
use serde::{Deserialize, Serialize};

/// Building-geometry parameter distributions.
///
/// These describe the physical shape of the zone being simulated.  All
/// values use standard SI units.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BuildingGeometryParams {
    /// Conditioned floor area [m²].
    pub floor_area: ParameterDistribution,
    /// Floor-to-ceiling height [m].
    pub ceiling_height: ParameterDistribution,
    /// Window-to-wall ratio [fraction, 0–1].
    pub window_to_wall_ratio: ParameterDistribution,
    /// Length-to-width aspect ratio [dimensionless].
    pub aspect_ratio: ParameterDistribution,
}

impl Default for BuildingGeometryParams {
    fn default() -> Self {
        Self {
            floor_area: ParameterDistribution::uniform(50.0, 500.0),
            ceiling_height: ParameterDistribution::uniform(2.4, 4.0),
            window_to_wall_ratio: ParameterDistribution::uniform(0.1, 0.6),
            aspect_ratio: ParameterDistribution::uniform(1.0, 3.0),
        }
    }
}

/// Insulation / envelope thermal-property distributions.
///
/// These describe the thermal resistance and mass of the building envelope,
/// enabling sweeps over insulation levels from minimally-code-compliant to
/// super-insulated.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InsulationParams {
    /// Overall wall U-value [W/m²K] (lower = better insulated).
    pub wall_u_value: ParameterDistribution,
    /// Insulation layer thickness [m].
    pub insulation_thickness: ParameterDistribution,
    /// Insulation thermal conductivity [W/m·K].
    pub insulation_conductivity: ParameterDistribution,
    /// Roof U-value [W/m²K].
    pub roof_u_value: ParameterDistribution,
}

impl Default for InsulationParams {
    fn default() -> Self {
        Self {
            // ASHRAE 90.1 range: ~0.3 (cold) to ~1.5 (hot/mild)
            wall_u_value: ParameterDistribution::uniform(0.2, 1.5),
            insulation_thickness: ParameterDistribution::uniform(0.025, 0.30),
            // Foam/fiberglass range
            insulation_conductivity: ParameterDistribution::uniform(0.022, 0.060),
            roof_u_value: ParameterDistribution::uniform(0.15, 0.8),
        }
    }
}

/// Weather / climate sampling parameters.
///
/// The continuous distributions (temp, solar, humidity, wind) inherit
/// their bounds from [`SurrogateDomain`].  Climate zones and building
/// types are discrete and sampled from the domain's lists.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeatherSamplingParams {
    /// Exterior dry-bulb temperature [°C].
    pub exterior_temp: ParameterDistribution,
    /// Solar irradiance [W/m²].
    pub solar_rad: ParameterDistribution,
    /// Relative humidity [%].
    pub humidity: ParameterDistribution,
    /// Wind speed [m/s].
    pub wind_speed: ParameterDistribution,
    /// Discrete climate-zone labels.
    pub climate_zones: Choice,
    /// Discrete building-type labels.
    pub building_types: Choice,
}

/// Occupancy and internal-gain distributions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OccupancyParams {
    /// Occupant density [fraction, 0–1].
    pub occupancy: ParameterDistribution,
    /// Zone (interior) temperature setpoint [°C].
    pub zone_temp: ParameterDistribution,
    /// Internal heat gain density [W/m²] (equipment + lighting).
    pub internal_gain_density: ParameterDistribution,
}

impl Default for OccupancyParams {
    fn default() -> Self {
        Self {
            occupancy: ParameterDistribution::uniform(0.0, 1.0),
            zone_temp: ParameterDistribution::uniform(18.0, 26.0),
            internal_gain_density: ParameterDistribution::uniform(2.0, 25.0),
        }
    }
}

/// Master configuration for a Monte Carlo parameter sweep.
///
/// Construct via [`SweepConfig::from_domain`] or [`SweepConfig::builder`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SweepConfig {
    /// Building geometry distributions.
    pub geometry: BuildingGeometryParams,
    /// Insulation / envelope distributions.
    pub insulation: InsulationParams,
    /// Weather / climate distributions.
    pub weather: WeatherSamplingParams,
    /// Occupancy / internal-gain distributions.
    pub occupancy: OccupancyParams,
    /// Sampling strategy (LHS, Sobol, or random).
    pub strategy: SamplingStrategy,
    /// Number of parameter sets to generate.
    pub num_samples: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Registry mapping climate-zone labels to representative weather files
    /// (multi-climate coverage, per Issue #1776 AC1).
    #[serde(default = "WeatherFileRegistry::standard")]
    pub weather_registry: WeatherFileRegistry,
}

/// Number of continuous dimensions swept by a [`SweepConfig`].
///
/// This must match the dimension count used by the sampling strategies.
/// Discrete dimensions (climate zones, building types) are not counted here
/// because they are sampled separately.
pub const NUM_CONTINUOUS_DIMENSIONS: usize = 15;

impl SweepConfig {
    /// Build a sweep config from an existing [`SurrogateDomain`], using the
    /// domain's bounds for weather/occupancy parameters and sensible defaults
    /// for geometry/insulation.
    ///
    /// This satisfies the acceptance criterion: *"Sampling config driven by
    /// existing `SurrogateDomain`"*.
    pub fn from_domain(domain: &SurrogateDomain) -> Self {
        let weather = WeatherSamplingParams {
            exterior_temp: ParameterDistribution::uniform(
                domain.temp_bounds.0,
                domain.temp_bounds.1,
            ),
            solar_rad: ParameterDistribution::uniform(domain.solar_bounds.0, domain.solar_bounds.1),
            humidity: ParameterDistribution::uniform(
                domain.humidity_bounds.0,
                domain.humidity_bounds.1,
            ),
            // Wind speed is not in SurrogateDomain; use a wide default.
            wind_speed: ParameterDistribution::uniform(0.0, 10.0),
            climate_zones: Choice::new(domain.climate_zones.clone()),
            building_types: Choice::new(domain.building_types.clone()),
        };

        let occupancy = OccupancyParams {
            occupancy: ParameterDistribution::uniform(
                domain.occupancy_bounds.0,
                domain.occupancy_bounds.1,
            ),
            zone_temp: ParameterDistribution::uniform(
                domain.zone_temp_bounds.0,
                domain.zone_temp_bounds.1,
            ),
            internal_gain_density: ParameterDistribution::uniform(2.0, 25.0),
        };

        SweepConfig {
            geometry: BuildingGeometryParams::default(),
            insulation: InsulationParams::default(),
            weather,
            occupancy,
            strategy: SamplingStrategy::LatinHypercube,
            num_samples: 1000,
            seed: 42,
            weather_registry: WeatherFileRegistry::standard(),
        }
    }

    /// Start a builder for custom configurations.
    pub fn builder() -> SweepConfigBuilder {
        SweepConfigBuilder::default()
    }

    /// Validate all distributions and parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_samples == 0 {
            return Err("num_samples must be > 0".to_string());
        }

        // Geometry
        self.geometry
            .floor_area
            .validate()
            .map_err(|e| format!("floor_area: {e}"))?;
        self.geometry
            .ceiling_height
            .validate()
            .map_err(|e| format!("ceiling_height: {e}"))?;
        self.geometry
            .window_to_wall_ratio
            .validate()
            .map_err(|e| format!("window_to_wall_ratio: {e}"))?;
        self.geometry
            .aspect_ratio
            .validate()
            .map_err(|e| format!("aspect_ratio: {e}"))?;

        // Insulation
        self.insulation
            .wall_u_value
            .validate()
            .map_err(|e| format!("wall_u_value: {e}"))?;
        self.insulation
            .insulation_thickness
            .validate()
            .map_err(|e| format!("insulation_thickness: {e}"))?;
        self.insulation
            .insulation_conductivity
            .validate()
            .map_err(|e| format!("insulation_conductivity: {e}"))?;
        self.insulation
            .roof_u_value
            .validate()
            .map_err(|e| format!("roof_u_value: {e}"))?;

        // Weather
        self.weather
            .exterior_temp
            .validate()
            .map_err(|e| format!("exterior_temp: {e}"))?;
        self.weather
            .solar_rad
            .validate()
            .map_err(|e| format!("solar_rad: {e}"))?;
        self.weather
            .humidity
            .validate()
            .map_err(|e| format!("humidity: {e}"))?;
        self.weather
            .wind_speed
            .validate()
            .map_err(|e| format!("wind_speed: {e}"))?;
        self.weather
            .climate_zones
            .validate()
            .map_err(|e| format!("climate_zones: {e}"))?;
        self.weather
            .building_types
            .validate()
            .map_err(|e| format!("building_types: {e}"))?;

        // Occupancy
        self.occupancy
            .occupancy
            .validate()
            .map_err(|e| format!("occupancy: {e}"))?;
        self.occupancy
            .zone_temp
            .validate()
            .map_err(|e| format!("zone_temp: {e}"))?;
        self.occupancy
            .internal_gain_density
            .validate()
            .map_err(|e| format!("internal_gain_density: {e}"))?;

        Ok(())
    }
}

/// Builder for [`SweepConfig`].
#[derive(Default)]
pub struct SweepConfigBuilder {
    geometry: Option<BuildingGeometryParams>,
    insulation: Option<InsulationParams>,
    weather: Option<WeatherSamplingParams>,
    occupancy: Option<OccupancyParams>,
    strategy: Option<SamplingStrategy>,
    num_samples: Option<usize>,
    seed: Option<u64>,
    weather_registry: Option<WeatherFileRegistry>,
}

impl SweepConfigBuilder {
    pub fn geometry(mut self, geometry: BuildingGeometryParams) -> Self {
        self.geometry = Some(geometry);
        self
    }

    pub fn insulation(mut self, insulation: InsulationParams) -> Self {
        self.insulation = Some(insulation);
        self
    }

    pub fn weather(mut self, weather: WeatherSamplingParams) -> Self {
        self.weather = Some(weather);
        self
    }

    pub fn occupancy(mut self, occupancy: OccupancyParams) -> Self {
        self.occupancy = Some(occupancy);
        self
    }

    pub fn strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.strategy = Some(strategy);
        self
    }

    pub fn num_samples(mut self, num_samples: usize) -> Self {
        self.num_samples = Some(num_samples);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Override the default standard weather-file registry.
    pub fn weather_registry(mut self, registry: WeatherFileRegistry) -> Self {
        self.weather_registry = Some(registry);
        self
    }

    /// Build the config, filling in defaults from `SurrogateDomain::default_residential()`.
    pub fn build(self) -> SweepConfig {
        let domain = SurrogateDomain::default_residential();
        let base = SweepConfig::from_domain(&domain);
        SweepConfig {
            geometry: self.geometry.unwrap_or(base.geometry),
            insulation: self.insulation.unwrap_or(base.insulation),
            weather: self.weather.unwrap_or(base.weather),
            occupancy: self.occupancy.unwrap_or(base.occupancy),
            strategy: self.strategy.unwrap_or(base.strategy),
            num_samples: self.num_samples.unwrap_or(base.num_samples),
            seed: self.seed.unwrap_or(base.seed),
            weather_registry: self.weather_registry.unwrap_or(base.weather_registry),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_domain_defaults() {
        let domain = SurrogateDomain::default_residential();
        let config = SweepConfig::from_domain(&domain);

        assert_eq!(config.num_samples, 1000);
        assert_eq!(config.seed, 42);
        assert_eq!(config.strategy, SamplingStrategy::LatinHypercube);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_from_domain_inherits_bounds() {
        let mut domain = SurrogateDomain::default_residential();
        domain.temp_bounds = (-30.0, 50.0);
        let config = SweepConfig::from_domain(&domain);

        match &config.weather.exterior_temp {
            ParameterDistribution::Uniform { min, max } => {
                assert_eq!(*min, -30.0);
                assert_eq!(*max, 50.0);
            }
            _ => panic!("expected uniform"),
        }
    }

    #[test]
    fn test_from_domain_inherits_climate_zones() {
        let mut domain = SurrogateDomain::default_residential();
        domain.climate_zones = vec!["4A".into(), "5A".into(), "6A".into()];
        let config = SweepConfig::from_domain(&domain);
        assert_eq!(config.weather.climate_zones.values.len(), 3);
    }

    #[test]
    fn test_builder_custom() {
        let config = SweepConfig::builder()
            .num_samples(500)
            .seed(99)
            .strategy(SamplingStrategy::Sobol)
            .build();
        assert_eq!(config.num_samples, 500);
        assert_eq!(config.seed, 99);
        assert_eq!(config.strategy, SamplingStrategy::Sobol);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_zero_samples() {
        let domain = SurrogateDomain::default_residential();
        let mut config = SweepConfig::from_domain(&domain);
        config.num_samples = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_bad_distribution() {
        let domain = SurrogateDomain::default_residential();
        let mut config = SweepConfig::from_domain(&domain);
        config.geometry.floor_area = ParameterDistribution::uniform(100.0, 50.0);
        let err = config.validate().unwrap_err();
        assert!(err.contains("floor_area"));
    }

    #[test]
    fn test_num_continuous_dimensions() {
        //   geometry: floor_area, ceiling_height, window_to_wall_ratio, aspect_ratio = 4
        //   insulation: wall_u_value, insulation_thickness, insulation_conductivity, roof_u_value = 4
        //   weather: exterior_temp, solar_rad, humidity, wind_speed = 4
        //   occupancy: occupancy, zone_temp, internal_gain_density = 3
        // Total = 4 + 4 + 4 + 3 = 15
        assert_eq!(NUM_CONTINUOUS_DIMENSIONS, 15);
    }

    #[test]
    fn test_from_domain_includes_standard_weather_registry() {
        // Issue #1776 AC1: from_domain must ship with the standard weather
        // registry so climate zones resolve to weather files.
        let domain = SurrogateDomain::default_residential();
        let config = SweepConfig::from_domain(&domain);
        for z in &["4A", "5A", "6A"] {
            assert!(
                config.weather_registry.lookup(z).is_some(),
                "standard registry missing {z}"
            );
        }
        assert!(!config.weather_registry.is_empty());
    }

    #[test]
    fn test_builder_custom_weather_registry() {
        use crate::ai::sweeps::weather::{WeatherFileEntry, WeatherFileRegistry};
        let custom = WeatherFileRegistry::with_entries([WeatherFileEntry::new(
            "9Z",
            "Custom City",
            1.0,
            2.0,
            "custom.epw",
        )]);
        let config = SweepConfig::builder()
            .weather_registry(custom)
            .num_samples(10)
            .build();
        assert!(config.weather_registry.lookup("9Z").is_some());
        assert!(config.weather_registry.lookup("4A").is_none());
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let domain = SurrogateDomain::default_residential();
        let config = SweepConfig::from_domain(&domain);
        let json = serde_json::to_string(&config).expect("serialize");
        let restored: SweepConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.seed, config.seed);
        assert_eq!(restored.num_samples, config.num_samples);
        assert!(restored.validate().is_ok());
    }
}
