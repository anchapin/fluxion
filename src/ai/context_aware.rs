//! Context-aware parameter initializer for quantum state space reduction.
//!
//! This module provides intelligent parameter initialization strategies that:
//! - Reduce the quantum state space for optimization algorithms
//! - Use context (weather, building characteristics) to guide initial population
//! - Improve convergence by starting from promising regions of parameter space

use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;
use crate::weather::HourlyWeatherData;
use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// Simple Gaussian random number generator using Box-Muller transform
struct GaussianRng<R: Rng> {
    rng: R,
}

impl<R: Rng> GaussianRng<R> {
    fn new(rng: R) -> Self {
        Self { rng }
    }

    fn sample(&mut self, mean: f64, std_dev: f64) -> f64 {
        // Box-Muller transform to generate Gaussian random numbers
        let u1: f64 = self.rng.gen();
        let u2: f64 = self.rng.gen();

        // Avoid log(0)
        let u1 = if u1 == 0.0 { f64::MIN_POSITIVE } else { u1 };

        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + z0 * std_dev
    }
}

/// Initialization strategy for parameter vectors
#[derive(Clone, Debug)]
pub enum InitializationStrategy {
    /// Uniform random initialization within bounds
    Uniform {
        /// Random seed for reproducibility
        seed: Option<u64>,
    },
    /// Gaussian initialization around a mean
    Gaussian {
        /// Random seed for reproducibility
        seed: Option<u64>,
        /// Standard deviation as fraction of parameter range
        std_fraction: f64,
    },
    /// Smart initialization using context (weather, building characteristics)
    Smart {
        /// Random seed for reproducibility
        seed: Option<u64>,
        /// Weather data for context-aware initialization
        weather: Option<HourlyWeatherData>,
        /// Building thermal mass (kg) for context-aware initialization
        thermal_mass: Option<f64>,
    },
    /// Latin Hypercube Sampling for better space coverage
    LHS {
        /// Random seed for reproducibility
        seed: Option<u64>,
        /// Number of samples to generate
        samples: usize,
    },
}

/// Configuration for the context-aware parameter initializer
#[derive(Clone, Debug)]
pub struct ContextAwareInitializerConfig {
    /// The initialization strategy to use
    pub strategy: InitializationStrategy,
    /// Parameter bounds: [min_u_value, max_u_value, min_heating, max_heating, min_cooling, max_cooling]
    pub param_bounds: ParamBounds,
    /// Number of individuals to generate
    pub population_size: usize,
}

impl Default for ContextAwareInitializerConfig {
    fn default() -> Self {
        Self {
            strategy: InitializationStrategy::Smart {
                seed: Some(42),
                weather: None,
                thermal_mass: None,
            },
            param_bounds: ParamBounds::default(),
            population_size: 100,
        }
    }
}

/// Parameter bounds for window U-value and HVAC setpoints
#[derive(Clone, Debug)]
pub struct ParamBounds {
    pub u_value_min: f64,
    pub u_value_max: f64,
    pub heating_setpoint_min: f64,
    pub heating_setpoint_max: f64,
    pub cooling_setpoint_min: f64,
    pub cooling_setpoint_max: f64,
}

impl Default for ParamBounds {
    fn default() -> Self {
        Self {
            u_value_min: 0.1,
            u_value_max: 5.0,
            heating_setpoint_min: 15.0,
            heating_setpoint_max: 25.0,
            cooling_setpoint_min: 22.0,
            cooling_setpoint_max: 32.0,
        }
    }
}

/// Context information for smart initialization
#[derive(Clone, Debug)]
pub struct InitializationContext {
    /// Average outdoor temperature (°C) for the simulation period
    pub avg_outdoor_temp: f64,
    /// Heating degree days (base 18°C)
    pub heating_degree_days: f64,
    /// Cooling degree days (base 18°C)
    pub cooling_degree_days: f64,
    /// Building thermal mass (kJ/K)
    pub thermal_mass: f64,
    /// Number of zones
    pub num_zones: usize,
    /// Window to wall ratio
    pub window_to_wall_ratio: f64,
}

impl Default for InitializationContext {
    fn default() -> Self {
        Self {
            avg_outdoor_temp: 15.0,
            heating_degree_days: 2000.0,
            cooling_degree_days: 500.0,
            thermal_mass: 10000.0,
            num_zones: 1,
            window_to_wall_ratio: 0.3,
        }
    }
}

/// Context-aware parameter initializer
pub struct ContextAwareParameterInitializer {
    config: ContextAwareInitializerConfig,
    context: InitializationContext,
}

impl ContextAwareParameterInitializer {
    /// Create a new initializer with the given configuration
    pub fn new(config: ContextAwareInitializerConfig, context: InitializationContext) -> Self {
        Self { config, context }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self {
            config: ContextAwareInitializerConfig::default(),
            context: InitializationContext::default(),
        }
    }

    /// Initialize a population of parameter vectors
    pub fn initialize(&self) -> Vec<Vec<f64>> {
        let population_size = self.config.population_size;

        match &self.config.strategy {
            InitializationStrategy::Uniform { seed } => {
                self.initialize_uniform(population_size, *seed)
            }
            InitializationStrategy::Gaussian { seed, std_fraction } => {
                self.initialize_gaussian(population_size, *seed, *std_fraction)
            }
            InitializationStrategy::Smart {
                seed,
                weather,
                thermal_mass,
            } => self.initialize_smart(population_size, *seed, weather.as_ref(), *thermal_mass),
            InitializationStrategy::LHS { seed, samples } => self.initialize_lhs(*samples, *seed),
        }
    }

    /// Initialize with uniform random distribution
    fn initialize_uniform(&self, size: usize, seed: Option<u64>) -> Vec<Vec<f64>> {
        let bounds = &self.config.param_bounds;
        let mut rng = create_rng(seed);

        let u_dist = Uniform::new(bounds.u_value_min, bounds.u_value_max);
        let heating_dist = Uniform::new(bounds.heating_setpoint_min, bounds.heating_setpoint_max);
        let cooling_dist = Uniform::new(bounds.cooling_setpoint_min, bounds.cooling_setpoint_max);

        (0..size)
            .map(|_| {
                let heating = heating_dist.sample(&mut rng);
                let cooling = (cooling_dist.sample(&mut rng)).max(heating + 1.0);
                vec![u_dist.sample(&mut rng), heating, cooling]
            })
            .collect()
    }

    /// Initialize with Gaussian distribution centered on promising values
    fn initialize_gaussian(
        &self,
        size: usize,
        seed: Option<u64>,
        std_fraction: f64,
    ) -> Vec<Vec<f64>> {
        let bounds = &self.config.param_bounds;
        let mut rng = create_rng(seed);
        let mut gaussian = GaussianRng::new(&mut rng);

        // Calculate means based on context
        let u_mean = (bounds.u_value_min + bounds.u_value_max) / 2.0;
        let heating_mean = (bounds.heating_setpoint_min + bounds.heating_setpoint_max) / 2.0;
        let cooling_mean = (bounds.cooling_setpoint_min + bounds.cooling_setpoint_max) / 2.0;

        let u_std = (bounds.u_value_max - bounds.u_value_min) * std_fraction;
        let heating_std =
            (bounds.heating_setpoint_max - bounds.heating_setpoint_min) * std_fraction;
        let cooling_std =
            (bounds.cooling_setpoint_max - bounds.cooling_setpoint_min) * std_fraction;

        (0..size)
            .map(|_| {
                let mut heating = gaussian.sample(heating_mean, heating_std);
                let mut cooling = gaussian.sample(cooling_mean, cooling_std);

                // Clamp to valid ranges
                heating = heating.clamp(bounds.heating_setpoint_min, bounds.heating_setpoint_max);
                cooling = cooling
                    .max(heating + 1.0)
                    .clamp(bounds.cooling_setpoint_min, bounds.cooling_setpoint_max);

                let u = gaussian
                    .sample(u_mean, u_std)
                    .clamp(bounds.u_value_min, bounds.u_value_max);

                vec![u, heating, cooling]
            })
            .collect()
    }

    /// Initialize using smart context-aware strategy
    fn initialize_smart(
        &self,
        size: usize,
        seed: Option<u64>,
        _weather: Option<&HourlyWeatherData>,
        thermal_mass: Option<f64>,
    ) -> Vec<Vec<f64>> {
        let bounds = &self.config.param_bounds;
        let ctx = &self.context;
        let mut rng = create_rng(seed);
        let mut gaussian = GaussianRng::new(&mut rng);

        // Use provided thermal mass or fall back to context
        let mass = thermal_mass.unwrap_or(ctx.thermal_mass);

        // Calculate climate-based setpoints
        // In heating-dominated climates, lower heating setpoints save energy
        // In cooling-dominated climates, higher cooling setpoints save energy
        let heating_base = if ctx.heating_degree_days > ctx.cooling_degree_days {
            // Heating-dominated: favor lower heating, moderate cooling
            bounds.heating_setpoint_min + 2.0
        } else {
            // Balanced/mixed: use middle ground
            (bounds.heating_setpoint_min + bounds.heating_setpoint_max) / 2.0
        };

        let cooling_base = if ctx.cooling_degree_days > ctx.heating_degree_days {
            // Cooling-dominated: favor higher cooling
            bounds.cooling_setpoint_max - 2.0
        } else {
            // Balanced/mixed: use middle ground
            (bounds.cooling_setpoint_min + bounds.cooling_setpoint_max) / 2.0
        };

        // For high thermal mass buildings, we can afford more aggressive setpoints
        // (they buffer temperature swings better)
        let mass_factor = (mass / 10000.0).clamp(0.5, 2.0);

        // Adjust spreads based on thermal mass (high mass = wider acceptable range)
        let heating_spread =
            (bounds.heating_setpoint_max - bounds.heating_setpoint_min) * 0.3 / mass_factor.sqrt();
        let cooling_spread =
            (bounds.cooling_setpoint_max - bounds.cooling_setpoint_min) * 0.3 / mass_factor.sqrt();

        // U-value: better insulated buildings need more exploration
        let u_base = if ctx.avg_outdoor_temp < 10.0 {
            // Cold climate: prioritize better insulation
            (bounds.u_value_min + bounds.u_value_max) / 3.0
        } else if ctx.avg_outdoor_temp > 20. {
            // Warm climate: can use less insulation
            (bounds.u_value_min + bounds.u_value_max) * 2.0 / 3.0
        } else {
            // Moderate: use middle
            (bounds.u_value_min + bounds.u_value_max) / 2.0
        };

        let u_spread = (bounds.u_value_max - bounds.u_value_min) * 0.2;

        (0..size)
            .map(|_| {
                let mut heating = gaussian.sample(heating_base, heating_spread);
                let mut cooling = gaussian.sample(cooling_base, cooling_spread);

                // Clamp to valid ranges
                heating = heating.clamp(bounds.heating_setpoint_min, bounds.heating_setpoint_max);
                cooling = cooling
                    .max(heating + 1.0)
                    .clamp(bounds.cooling_setpoint_min, bounds.cooling_setpoint_max);

                let u = gaussian
                    .sample(u_base, u_spread)
                    .clamp(bounds.u_value_min, bounds.u_value_max);

                vec![u, heating, cooling]
            })
            .collect()
    }

    /// Initialize using Latin Hypercube Sampling for better space coverage
    fn initialize_lhs(&self, samples: usize, seed: Option<u64>) -> Vec<Vec<f64>> {
        let bounds = &self.config.param_bounds;
        let mut rng = create_rng(seed);

        let u_range = bounds.u_value_max - bounds.u_value_min;
        let heating_range = bounds.heating_setpoint_max - bounds.heating_setpoint_min;
        let cooling_range = bounds.cooling_setpoint_max - bounds.cooling_setpoint_min;

        // Generate LHS for each parameter
        let mut lhs_samples: Vec<Vec<f64>> = Vec::with_capacity(samples);

        for i in 0..samples {
            let u = bounds.u_value_min + u_range * ((i as f64 + rng.gen::<f64>()) / samples as f64);
            let heating = bounds.heating_setpoint_min
                + heating_range * ((i as f64 + rng.gen::<f64>()) / samples as f64);

            // Ensure heating < cooling with a gap
            let min_cooling = heating + 1.0;
            let max_cooling = bounds.cooling_setpoint_max;
            let cooling = min_cooling
                + (max_cooling - min_cooling) * ((i as f64 + rng.gen::<f64>()) / samples as f64);

            // Add some randomization within cells
            let u = u + u_range * (rng.gen::<f64>() - 0.5) / samples as f64;
            let heating = heating + heating_range * (rng.gen::<f64>() - 0.5) / samples as f64;
            let cooling = cooling + cooling_range * (rng.gen::<f64>() - 0.5) / samples as f64;

            lhs_samples.push(vec![
                u.clamp(bounds.u_value_min, bounds.u_value_max),
                heating.clamp(bounds.heating_setpoint_min, bounds.heating_setpoint_max),
                cooling
                    .max(heating + 1.0)
                    .clamp(bounds.cooling_setpoint_min, bounds.cooling_setpoint_max),
            ]);
        }

        lhs_samples
    }

    /// Generate initial population from a reference model
    pub fn initialize_from_model(
        &self,
        base_model: &ThermalModel<VectorField>,
        population_size: usize,
    ) -> Vec<Vec<f64>> {
        let bounds = &self.config.param_bounds;

        // Extract current parameters from base model
        let base_u = base_model.window_u_value;
        let base_heating = base_model.hvac_controller.heating_setpoint;
        let base_cooling = base_model.hvac_controller.cooling_setpoint;

        // Generate variations around the base model
        let mut rng = create_rng(Some(42));
        let mut gaussian = GaussianRng::new(&mut rng);

        let u_spread = (bounds.u_value_max - bounds.u_value_min) * 0.15;
        let heating_spread = (bounds.heating_setpoint_max - bounds.heating_setpoint_min) * 0.1;
        let cooling_spread = (bounds.cooling_setpoint_max - bounds.cooling_setpoint_min) * 0.1;

        (0..population_size)
            .map(|_| {
                let mut heating = gaussian.sample(base_heating, heating_spread);
                let mut cooling = gaussian.sample(base_cooling, cooling_spread);

                heating = heating.clamp(bounds.heating_setpoint_min, bounds.heating_setpoint_max);
                cooling = cooling
                    .max(heating + 1.0)
                    .clamp(bounds.cooling_setpoint_min, bounds.cooling_setpoint_max);

                let u = gaussian
                    .sample(base_u, u_spread)
                    .clamp(bounds.u_value_min, bounds.u_value_max);

                vec![u, heating, cooling]
            })
            .collect()
    }

    /// Get statistics about the initialized population
    pub fn get_population_stats(&self, population: &[Vec<f64>]) -> PopulationStats {
        if population.is_empty() {
            return PopulationStats::default();
        }

        let mut u_values = Vec::with_capacity(population.len());
        let mut heating_values = Vec::with_capacity(population.len());
        let mut cooling_values = Vec::with_capacity(population.len());

        for params in population {
            if params.len() >= 3 {
                u_values.push(params[0]);
                heating_values.push(params[1]);
                cooling_values.push(params[2]);
            }
        }

        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let std = |v: &[f64]| {
            let m = mean(v);
            (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
        };
        let min = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        PopulationStats {
            u_value_mean: mean(&u_values),
            u_value_std: std(&u_values),
            u_value_min: min(&u_values),
            u_value_max: max(&u_values),
            heating_mean: mean(&heating_values),
            heating_std: std(&heating_values),
            heating_min: min(&heating_values),
            heating_max: max(&heating_values),
            cooling_mean: mean(&cooling_values),
            cooling_std: std(&cooling_values),
            cooling_min: min(&cooling_values),
            cooling_max: max(&cooling_values),
            population_size: population.len(),
        }
    }
}

/// Statistics about an initialized population
#[derive(Clone, Debug, Default)]
pub struct PopulationStats {
    pub u_value_mean: f64,
    pub u_value_std: f64,
    pub u_value_min: f64,
    pub u_value_max: f64,
    pub heating_mean: f64,
    pub heating_std: f64,
    pub heating_min: f64,
    pub heating_max: f64,
    pub cooling_mean: f64,
    pub cooling_std: f64,
    pub cooling_min: f64,
    pub cooling_max: f64,
    pub population_size: usize,
}

/// Create a random number generator with optional seed
fn create_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_initialization() {
        let config = ContextAwareInitializerConfig {
            strategy: InitializationStrategy::Uniform { seed: Some(42) },
            param_bounds: ParamBounds::default(),
            population_size: 100,
        };

        let initializer =
            ContextAwareParameterInitializer::new(config, InitializationContext::default());
        let population = initializer.initialize();

        assert_eq!(population.len(), 100);
        assert!(population.iter().all(|p| p.len() == 3));

        // Check bounds
        for params in &population {
            assert!(params[0] >= 0.1 && params[0] <= 5.0);
            assert!(params[1] >= 15.0 && params[1] <= 25.0);
            assert!(params[2] >= 22.0 && params[2] <= 32.0);
            assert!(params[1] < params[2]); // heating < cooling
        }
    }

    #[test]
    fn test_gaussian_initialization() {
        let config = ContextAwareInitializerConfig {
            strategy: InitializationStrategy::Gaussian {
                seed: Some(42),
                std_fraction: 0.2,
            },
            param_bounds: ParamBounds::default(),
            population_size: 100,
        };

        let initializer =
            ContextAwareParameterInitializer::new(config, InitializationContext::default());
        let population = initializer.initialize();

        assert_eq!(population.len(), 100);

        let stats = initializer.get_population_stats(&population);
        // Gaussian should be more concentrated around the center
        assert!(stats.u_value_std < 2.0);
    }

    #[test]
    fn test_smart_initialization_heating_climate() {
        let config = ContextAwareInitializerConfig {
            strategy: InitializationStrategy::Smart {
                seed: Some(42),
                weather: None,
                thermal_mass: None,
            },
            param_bounds: ParamBounds::default(),
            population_size: 100,
        };

        let context = InitializationContext {
            avg_outdoor_temp: 10.0,
            heating_degree_days: 3000.0,
            cooling_degree_days: 500.0,
            thermal_mass: 10000.0,
            num_zones: 1,
            window_to_wall_ratio: 0.3,
        };

        let initializer = ContextAwareParameterInitializer::new(config, context);
        let population = initializer.initialize();

        let stats = initializer.get_population_stats(&population);

        // In heating-dominated climate, should favor lower heating setpoints
        assert!(stats.heating_mean < 20.0);
    }

    #[test]
    fn test_lhs_initialization() {
        let config = ContextAwareInitializerConfig {
            strategy: InitializationStrategy::LHS {
                seed: Some(42),
                samples: 50,
            },
            param_bounds: ParamBounds::default(),
            population_size: 50,
        };

        let initializer =
            ContextAwareParameterInitializer::new(config, InitializationContext::default());
        let population = initializer.initialize();

        assert_eq!(population.len(), 50);

        // LHS should provide good coverage - check that min/max span the range well
        let stats = initializer.get_population_stats(&population);
        assert!(stats.u_value_max - stats.u_value_min > 3.0);
    }

    #[test]
    fn test_population_stats() {
        let config = ContextAwareInitializerConfig {
            strategy: InitializationStrategy::Uniform { seed: Some(42) },
            param_bounds: ParamBounds::default(),
            population_size: 100,
        };

        let initializer =
            ContextAwareParameterInitializer::new(config, InitializationContext::default());
        let population = initializer.initialize();
        let stats = initializer.get_population_stats(&population);

        assert_eq!(stats.population_size, 100);
        assert!(stats.u_value_min < stats.u_value_max);
        assert!(stats.heating_min < stats.heating_max);
        assert!(stats.cooling_min < stats.cooling_max);
    }

    #[test]
    fn test_reproducibility() {
        let config1 = ContextAwareInitializerConfig {
            strategy: InitializationStrategy::Uniform { seed: Some(12345) },
            param_bounds: ParamBounds::default(),
            population_size: 10,
        };

        let config2 = ContextAwareInitializerConfig {
            strategy: InitializationStrategy::Uniform { seed: Some(12345) },
            param_bounds: ParamBounds::default(),
            population_size: 10,
        };

        let initializer1 =
            ContextAwareParameterInitializer::new(config1, InitializationContext::default());
        let initializer2 =
            ContextAwareParameterInitializer::new(config2, InitializationContext::default());

        let pop1 = initializer1.initialize();
        let pop2 = initializer2.initialize();

        assert_eq!(pop1, pop2);
    }
}
