use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::BatchOracle;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const MIN_CONFIGS_PER_SEC: f64 = 10.0;
const ZONE_COUNT: usize = 10;

fn generate_synthetic_population(size: usize) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut population = Vec::with_capacity(size);
    for _ in 0..size {
        let u_value = rng.random_range(0.1..5.0);
        let heating_setpoint = rng.random_range(15.0..25.0);
        let cooling_setpoint = rng.random_range(22.0..32.0);
        population.push(vec![u_value, heating_setpoint, cooling_setpoint]);
    }
    population
}

#[test]
fn test_multi_zone_throughput() {
    let base_model = ThermalModel::<VectorField>::new(ZONE_COUNT);
    let oracle = BatchOracle::from_model(base_model);

    let population = generate_synthetic_population(100);

    let start = Instant::now();
    let _ = oracle
        .evaluate_population(population, false)
        .expect("evaluate_population should succeed");
    let elapsed = start.elapsed();

    let n = 100;
    let secs = elapsed.as_secs_f64();
    let throughput = n as f64 / secs;

    println!(
        "Multi-zone ({} zone) throughput: {:.2} configs/sec",
        ZONE_COUNT, throughput
    );
    assert!(
        throughput >= MIN_CONFIGS_PER_SEC,
        "Multi-zone throughput {} configs/sec is below minimum {}",
        throughput,
        MIN_CONFIGS_PER_SEC
    );
}

#[test]
fn test_ci_validator_creation() {
    let _validator = fluxion::validation::performance::ci::CiPerformanceValidator::new(None);
}
