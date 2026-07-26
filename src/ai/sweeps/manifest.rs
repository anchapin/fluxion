//! Reproducible parameter manifest and individual sweep samples.
//!
//! The [`ParameterManifest`] records the exact configuration used to generate
//! a batch of sweep samples so that the run is fully reproducible.  It is
//! emitted alongside the generated dataset (Issue #1776 acceptance criterion:
//! *"Reproducible (seeded) parameter manifest emitted"*).
//!
//! [`SweepSample`] is a single realised parameter set — a point in parameter
//! space that can be fed to the physics solver to produce training data for
//! the ML surrogate.

use crate::ai::surrogate::SurrogateInputs;
use crate::ai::sweeps::config::SweepConfig;
use crate::ai::sweeps::config::NUM_CONTINUOUS_DIMENSIONS;
use crate::ai::sweeps::distributions::ParameterDistribution;
use crate::ai::sweeps::sampling::generate_unit_samples;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// A single realised parameter set from a sweep.
///
/// Each field corresponds to a physical quantity.  The struct can be
/// converted into [`SurrogateInputs`] (for the weather/occupancy portion)
/// and into a wall-insulation specification (for the envelope portion).
#[derive(Clone, Debug)]
pub struct SweepSample {
    // --- Building geometry ---
    /// Conditioned floor area [m²].
    pub floor_area: f64,
    /// Floor-to-ceiling height [m].
    pub ceiling_height: f64,
    /// Window-to-wall ratio [fraction, 0–1].
    pub window_to_wall_ratio: f64,
    /// Length-to-width aspect ratio [dimensionless].
    pub aspect_ratio: f64,

    // --- Insulation / envelope ---
    /// Overall wall U-value [W/m²K].
    pub wall_u_value: f64,
    /// Insulation layer thickness [m].
    pub insulation_thickness: f64,
    /// Insulation thermal conductivity [W/m·K].
    pub insulation_conductivity: f64,
    /// Roof U-value [W/m²K].
    pub roof_u_value: f64,

    // --- Weather / climate ---
    /// Exterior dry-bulb temperature [°C].
    pub exterior_temp: f64,
    /// Solar irradiance [W/m²].
    pub solar_rad: f64,
    /// Relative humidity [%].
    pub humidity: f64,
    /// Wind speed [m/s].
    pub wind_speed: f64,
    /// Climate zone label (e.g. "4A").
    pub climate_zone: String,

    // --- Occupancy / internal gains ---
    /// Occupant density [fraction, 0–1].
    pub occupancy: f64,
    /// Zone temperature setpoint [°C].
    pub zone_temp: f64,
    /// Internal heat gain density [W/m²].
    pub internal_gain_density: f64,

    // --- Building type ---
    /// Building type label (e.g. "residential").
    pub building_type: String,
}

impl SweepSample {
    /// Convert the weather/occupancy portion to [`SurrogateInputs`] for
    /// surrogate-model inference.
    pub fn to_surrogate_inputs(&self) -> SurrogateInputs {
        SurrogateInputs {
            exterior_temp: self.exterior_temp,
            zone_temp: self.zone_temp,
            solar_rad: self.solar_rad,
            humidity: self.humidity,
            occupancy: self.occupancy,
            climate_zone: self.climate_zone.clone(),
        }
    }

    /// Effective envelope area estimate [m²].
    ///
    /// Simple box model: given `floor_area` and `aspect_ratio`, computes the
    /// perimeter-wall area (minus windows) plus the roof area.
    pub fn envelope_area(&self) -> f64 {
        let footprint = self.floor_area;
        let l = (footprint * self.aspect_ratio).sqrt();
        let w = footprint / l;
        let perimeter = 2.0 * (l + w);
        let wall_area = perimeter * self.ceiling_height;
        let window_area = wall_area * self.window_to_wall_ratio;
        let opaque_wall_area = wall_area - window_area;
        let roof_area = footprint;
        opaque_wall_area + roof_area
    }
}

/// Reproducible manifest for a completed sweep run.
///
/// Contains all information needed to re-generate the exact same set of
/// [`SweepSample`]s.
#[derive(Clone, Debug)]
pub struct ParameterManifest {
    /// Random seed used by the sampling strategy.
    pub seed: u64,
    /// Sampling strategy name.
    pub strategy: String,
    /// Number of samples generated.
    pub num_samples: usize,
    /// Number of continuous dimensions.
    pub num_dimensions: usize,
    /// ISO-8601-ish timestamp of when the manifest was created.
    pub created_at: String,
    /// Short description of the sweep (human-readable).
    pub description: String,
}

impl ParameterManifest {
    fn now_iso8601() -> String {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let days = secs / 86400;
        let year = 1970 + (days / 365);
        let day_of_year = days % 365;
        format!("{year}-{:03}T{}Z", day_of_year, secs % 86400)
    }

    /// Create a manifest from a config.
    pub fn from_config(config: &SweepConfig) -> Self {
        ParameterManifest {
            seed: config.seed,
            strategy: config.strategy.name().to_string(),
            num_samples: config.num_samples,
            num_dimensions: NUM_CONTINUOUS_DIMENSIONS,
            created_at: Self::now_iso8601(),
            description: format!(
                "Monte Carlo sweep: {} samples, {} strategy, seed {}",
                config.num_samples,
                config.strategy.name(),
                config.seed
            ),
        }
    }
}

/// Result of a sweep generation: the samples + a reproducible manifest.
#[derive(Clone, Debug)]
pub struct SweepResult {
    /// Generated parameter samples.
    pub samples: Vec<SweepSample>,
    /// Manifest for reproducibility.
    pub manifest: ParameterManifest,
}

impl SweepResult {
    /// Number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the result is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Extract all [`SurrogateInputs`] from the samples.
    pub fn to_surrogate_inputs(&self) -> Vec<SurrogateInputs> {
        self.samples
            .iter()
            .map(|s| s.to_surrogate_inputs())
            .collect()
    }
}

/// Generate a full set of sweep samples from a [`SweepConfig`].
///
/// This is the main entry point for Phase 2 data generation.  It:
/// 1. Generates unit-cube samples via the configured strategy.
/// 2. Maps each unit-cube point through the parameter distributions using
///    inverse-CDF transforms (preserves LHS stratification).
/// 3. Samples discrete parameters (climate zones, building types).
/// 4. Returns a [`SweepResult`] containing the samples and manifest.
///
/// # Panics
/// Panics if `config.validate()` returns an error.
pub fn generate_samples(config: &SweepConfig) -> SweepResult {
    config.validate().expect("invalid sweep config");

    let unit_samples = generate_unit_samples(
        config.strategy,
        config.num_samples,
        NUM_CONTINUOUS_DIMENSIONS,
        config.seed,
    );

    // Separate RNG for discrete sampling (climate zones, building types).
    let mut discrete_rng = StdRng::seed_from_u64(config.seed.wrapping_add(0xDEAD_BEEF));

    let distributions: [(&str, &ParameterDistribution); NUM_CONTINUOUS_DIMENSIONS] = [
        ("floor_area", &config.geometry.floor_area),
        ("ceiling_height", &config.geometry.ceiling_height),
        (
            "window_to_wall_ratio",
            &config.geometry.window_to_wall_ratio,
        ),
        ("aspect_ratio", &config.geometry.aspect_ratio),
        ("wall_u_value", &config.insulation.wall_u_value),
        (
            "insulation_thickness",
            &config.insulation.insulation_thickness,
        ),
        (
            "insulation_conductivity",
            &config.insulation.insulation_conductivity,
        ),
        ("roof_u_value", &config.insulation.roof_u_value),
        ("exterior_temp", &config.weather.exterior_temp),
        ("solar_rad", &config.weather.solar_rad),
        ("humidity", &config.weather.humidity),
        ("wind_speed", &config.weather.wind_speed),
        ("occupancy", &config.occupancy.occupancy),
        ("zone_temp", &config.occupancy.zone_temp),
        (
            "internal_gain_density",
            &config.occupancy.internal_gain_density,
        ),
    ];

    let mut samples = Vec::with_capacity(config.num_samples);
    for i in 0..config.num_samples {
        let mut vals = [0.0f64; NUM_CONTINUOUS_DIMENSIONS];
        for (j, (_, dist)) in distributions.iter().enumerate() {
            let unit_val = unit_samples[i * NUM_CONTINUOUS_DIMENSIONS + j];
            vals[j] = transform_unit_to_distribution(unit_val, dist);
        }

        let climate_zone = config
            .weather
            .climate_zones
            .sample(&mut discrete_rng)
            .to_string();
        let building_type = config
            .weather
            .building_types
            .sample(&mut discrete_rng)
            .to_string();

        let humidity = vals[10].clamp(0.0, 100.0);
        let window_to_wall_ratio = vals[2].clamp(0.0, 1.0);
        let occupancy = vals[12].clamp(0.0, 1.0);

        samples.push(SweepSample {
            floor_area: vals[0],
            ceiling_height: vals[1],
            window_to_wall_ratio,
            aspect_ratio: vals[3],
            wall_u_value: vals[4],
            insulation_thickness: vals[5],
            insulation_conductivity: vals[6],
            roof_u_value: vals[7],
            exterior_temp: vals[8],
            solar_rad: vals[9],
            humidity,
            wind_speed: vals[11],
            climate_zone,
            occupancy,
            zone_temp: vals[13],
            internal_gain_density: vals[14],
            building_type,
        });
    }

    let manifest = ParameterManifest::from_config(config);
    SweepResult { samples, manifest }
}

/// Transform a unit-cube `[0,1)` value to the target distribution using
/// the inverse-CDF method.
///
/// For `Uniform`, this is a simple linear interpolation.  For `Normal` and
/// `LogNormal`, the unit value is treated as a probability `p` and the
/// quantile function maps it to the distribution.  This preserves LHS
/// stratification: each stratum in the unit cube maps to a stratum in the
/// target distribution.
fn transform_unit_to_distribution(unit_val: f64, dist: &ParameterDistribution) -> f64 {
    match dist {
        ParameterDistribution::Uniform { min, max } => {
            let clamped = unit_val.clamp(0.0, 1.0);
            min + clamped * (max - min)
        }
        ParameterDistribution::Normal { mean, std_dev } => {
            let p = unit_val.clamp(1e-10, 1.0 - 1e-10);
            let z = inverse_normal_cdf(p);
            mean + std_dev * z
        }
        ParameterDistribution::LogNormal { mu, sigma } => {
            let p = unit_val.clamp(1e-10, 1.0 - 1e-10);
            let z = inverse_normal_cdf(p);
            (mu + sigma * z).exp()
        }
    }
}

/// Inverse error function `erfinv(y)` using Newton-Raphson with `libm::erf`.
///
/// Returns `x` such that `erf(x) = y` for `y` in `(-1, 1)`.
/// Accurate to machine epsilon (~1e-15) after a few iterations.
const SQRT_PI: f64 = 1.7724538509055159;

fn erfinv(y: f64) -> f64 {
    if y.abs() >= 1.0 {
        return if y > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }
    if y.abs() < 1e-10 {
        let c = 2.0 / SQRT_PI;
        return y / c;
    }

    let x0 = if y.abs() < 0.9 {
        let s = (1.0 - y) * (1.0 + y);
        0.88622692545 * y * (-s.ln()).sqrt()
    } else {
        if y > 0.0 {
            (1.0 / (2.0 * SQRT_PI * (1.0 - y))).ln().sqrt()
        } else {
            -(1.0 / (2.0 * SQRT_PI * (1.0 + y))).ln().sqrt()
        }
    };

    let mut x = x0;
    for _ in 0..50 {
        let fx = libm::erf(x) - y;
        if fx.abs() < 1e-15 {
            break;
        }
        let dfx = 2.0 / SQRT_PI * (-x * x).exp();
        x -= fx / dfx;
    }
    x
}

/// Inverse standard normal CDF via the erfinv relation:
///
/// `ndtri(p) = -sqrt(2) * erfinv(1 - 2p)`
///
/// This is the scipy-compatible formula; Newton-Raphson with libm::erf
/// gives machine-epsilon accuracy for all `p` in `(0, 1)`.
fn inverse_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let y = 1.0 - 2.0 * p;
    -std::f64::consts::SQRT_2 * erfinv(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::surrogate::SurrogateDomain;
    use crate::ai::sweeps::sampling::SamplingStrategy;

    #[test]
    fn test_inverse_normal_cdf_symmetric() {
        let z_half = inverse_normal_cdf(0.5);
        assert!(z_half.abs() < 1e-6, "ICDF(0.5) should be 0, got {z_half}");

        let z_25 = inverse_normal_cdf(0.25);
        let z_75 = inverse_normal_cdf(0.75);
        assert!((z_25 + z_75).abs() < 1e-6, "should be antisymmetric");
    }

    #[test]
    fn test_inverse_normal_cdf_known_values() {
        let z = inverse_normal_cdf(0.8413447460685429);
        assert!(
            (z - 1.0).abs() < 1e-3,
            "ICDF(0.8413) should be ~1.0, got {z}"
        );

        let z = inverse_normal_cdf(0.02275013194817921);
        assert!(
            (z + 2.0).abs() < 1e-3,
            "ICDF(0.0228) should be ~-2.0, got {z}"
        );
    }

    #[test]
    fn test_generate_samples_count() {
        let domain = SurrogateDomain::default_residential();
        let config = SweepConfig::from_domain(&domain);
        let result = generate_samples(&config);
        assert_eq!(result.len(), 1000);
    }

    #[test]
    fn test_generate_samples_reproducible() {
        let domain = SurrogateDomain::default_residential();
        let config = SweepConfig::from_domain(&domain);
        let result1 = generate_samples(&config);
        let result2 = generate_samples(&config);

        assert_eq!(result1.len(), result2.len());
        for (s1, s2) in result1.samples.iter().zip(result2.samples.iter()) {
            assert!((s1.floor_area - s2.floor_area).abs() < 1e-10);
            assert!((s1.exterior_temp - s2.exterior_temp).abs() < 1e-10);
            assert_eq!(s1.climate_zone, s2.climate_zone);
        }
    }

    #[test]
    fn test_different_seeds_different_samples() {
        let domain = SurrogateDomain::default_residential();
        let mut config1 = SweepConfig::from_domain(&domain);
        config1.seed = 1;
        let mut config2 = SweepConfig::from_domain(&domain);
        config2.seed = 2;

        let result1 = generate_samples(&config1);
        let result2 = generate_samples(&config2);

        let diff = result1
            .samples
            .iter()
            .zip(result2.samples.iter())
            .filter(|(a, b)| (a.floor_area - b.floor_area).abs() > 1e-10)
            .count();
        assert!(
            diff > 900,
            "different seeds should produce mostly different samples, got {diff} matches out of 1000"
        );
    }

    #[test]
    fn test_samples_in_bounds() {
        let mut domain = SurrogateDomain::default_residential();
        domain.temp_bounds = (-20.0, 40.0);
        domain.humidity_bounds = (0.0, 100.0);
        let config = SweepConfig::from_domain(&domain);
        let result = generate_samples(&config);

        for s in &result.samples {
            assert!(s.exterior_temp >= -20.0 && s.exterior_temp <= 40.0);
            assert!(s.humidity >= 0.0 && s.humidity <= 100.0);
            assert!(s.window_to_wall_ratio >= 0.0 && s.window_to_wall_ratio <= 1.0);
            assert!(s.occupancy >= 0.0 && s.occupancy <= 1.0);
            assert!(s.floor_area > 0.0);
            assert!(s.ceiling_height > 0.0);
            assert!(s.wall_u_value > 0.0);
        }
    }

    #[test]
    fn test_climate_zone_coverage() {
        let domain = SurrogateDomain::default_residential();
        let config = SweepConfig::from_domain(&domain);
        let result = generate_samples(&config);

        let zones: std::collections::HashSet<_> = result
            .samples
            .iter()
            .map(|s| s.climate_zone.as_str())
            .collect();
        assert!(zones.contains("4A"));
        assert!(zones.contains("5A"));
        assert!(zones.contains("6A"));
    }

    #[test]
    fn test_manifest_content() {
        let domain = SurrogateDomain::default_residential();
        let config = SweepConfig::from_domain(&domain);
        let result = generate_samples(&config);

        assert_eq!(result.manifest.seed, 42);
        assert_eq!(result.manifest.strategy, "latin_hypercube");
        assert_eq!(result.manifest.num_samples, 1000);
        assert_eq!(result.manifest.num_dimensions, NUM_CONTINUOUS_DIMENSIONS);
    }

    #[test]
    fn test_sobol_strategy_samples() {
        let domain = SurrogateDomain::default_residential();
        let mut config = SweepConfig::from_domain(&domain);
        config.strategy = SamplingStrategy::Sobol;
        config.num_samples = 100;
        let result = generate_samples(&config);
        assert_eq!(result.len(), 100);
        assert_eq!(result.manifest.strategy, "sobol");
    }

    #[test]
    fn test_random_mc_strategy_samples() {
        let domain = SurrogateDomain::default_residential();
        let mut config = SweepConfig::from_domain(&domain);
        config.strategy = SamplingStrategy::RandomMonteCarlo;
        config.num_samples = 100;
        let result = generate_samples(&config);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_sweep_sample_to_surrogate_inputs() {
        let domain = SurrogateDomain::default_residential();
        let config = SweepConfig::from_domain(&domain);
        let result = generate_samples(&config);

        let inputs = result.to_surrogate_inputs();
        assert_eq!(inputs.len(), 1000);
        for (s, inp) in result.samples.iter().zip(inputs.iter()) {
            assert_eq!(s.exterior_temp, inp.exterior_temp);
            assert_eq!(s.climate_zone, inp.climate_zone);
        }
    }

    #[test]
    fn test_envelope_area_positive() {
        let domain = SurrogateDomain::default_residential();
        let config = SweepConfig::from_domain(&domain);
        let result = generate_samples(&config);
        for s in &result.samples {
            assert!(s.envelope_area() > 0.0);
        }
    }
}
