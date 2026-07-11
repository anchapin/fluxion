//! Parity test for the PerezSkyModel dedup (Issue #1414).
//!
//! `PerezSkyModel`, `extraterrestrial_irradiance`, and `relative_airmass` are
//! defined exactly once in `fluxion::solar::surface_irradiance` (the leaf
//! module). `fluxion::sim::sky_radiation` re-exports them so callers depending
//! on either path observe bit-identical results.
//!
//! This test asserts the leaf path (`fluxion::solar::*`) and the sim path
//! (`fluxion::sim::sky_radiation::*`) agree for 50 deterministic random
//! samples within `f64::EPSILON * 1e3`. Because both paths now resolve to the
//! same function items via `pub use`, they should match exactly — we use a
//! generous epsilon (1e3 ulps) only as an explicit allow-list for any future
//! platform-specific floating-point variance.

use fluxion::sim::sky_radiation as sim_path;
use fluxion::solar::surface_irradiance as leaf_path;
use rand::{rngs::StdRng, Rng, SeedableRng};

const SAMPLES: usize = 50;
const SEED: u64 = 0x1414_DEAD_BEEF;

fn assert_bit_identical(name: &str, a: f64, b: f64) {
    assert!(
        (a - b).abs() <= f64::EPSILON * 1e3,
        "{name} parity drift: leaf={a} sim={b} delta={} (tol={})",
        (a - b).abs(),
        f64::EPSILON * 1e3,
    );
}

#[test]
fn test_leaf_and_sim_perez_identical() {
    let mut rng = StdRng::seed_from_u64(SEED);

    // 50 deterministic random (DNI, DHI, zenith, tilt, az) tuples.
    for i in 0..SAMPLES {
        let dhi: f64 = rng.gen_range(0.0..500.0);
        let dni: f64 = rng.gen_range(0.0..900.0);
        let dni_extra: f64 = rng.gen_range(1300.0..1400.0);
        let airmass: f64 = rng.gen_range(1.0..10.0);
        let zenith_deg: f64 = rng.gen_range(0.0..85.0);
        let surface_tilt_deg: f64 = rng.gen_range(0.0..90.0);
        let surface_azimuth_deg: f64 = rng.gen_range(0.0..360.0);
        let solar_azimuth_deg: f64 = rng.gen_range(0.0..360.0);

        let leaf_diffuse = leaf_path::PerezSkyModel::calculate_diffuse_tilted(
            dhi,
            dni,
            dni_extra,
            airmass,
            zenith_deg,
            surface_tilt_deg,
            surface_azimuth_deg,
            solar_azimuth_deg,
        );
        let sim_diffuse = sim_path::PerezSkyModel::calculate_diffuse_tilted(
            dhi,
            dni,
            dni_extra,
            airmass,
            zenith_deg,
            surface_tilt_deg,
            surface_azimuth_deg,
            solar_azimuth_deg,
        );
        assert_bit_identical(
            &format!("diffuse_tilted[{i}]"),
            leaf_diffuse,
            sim_diffuse,
        );

        // extraterrestrial_irradiance varies only by day_of_year.
        let day_of_year: usize = rng.gen_range(1..=365);
        assert_bit_identical(
            &format!("extraterrestrial_irradiance[{i}]"),
            leaf_path::extraterrestrial_irradiance(day_of_year),
            sim_path::extraterrestrial_irradiance(day_of_year),
        );

        assert_bit_identical(
            &format!("relative_airmass[{i}]"),
            leaf_path::relative_airmass(zenith_deg),
            sim_path::relative_airmass(zenith_deg),
        );
    }
}

/// Belt-and-braces: every re-export must resolve to the exact same function
/// address as the leaf definition. If a future contributor reintroduces a
/// local copy in `sim::sky_radiation`, this test trips immediately rather
/// than relying on coincidental numerical agreement.
#[test]
fn test_re_exports_point_at_single_definition() {
    let leaf_perez: fn(f64, f64, f64, f64, f64, f64, f64, f64) -> f64 =
        leaf_path::PerezSkyModel::calculate_diffuse_tilted;
    let sim_perez: fn(f64, f64, f64, f64, f64, f64, f64, f64) -> f64 =
        sim_path::PerezSkyModel::calculate_diffuse_tilted;
    assert_eq!(
        leaf_perez as usize, sim_perez as usize,
        "PerezSkyModel::calculate_diffuse_tilted re-export diverged from leaf"
    );

    let leaf_eti: fn(usize) -> f64 = leaf_path::extraterrestrial_irradiance;
    let sim_eti: fn(usize) -> f64 = sim_path::extraterrestrial_irradiance;
    assert_eq!(
        leaf_eti as usize, sim_eti as usize,
        "extraterrestrial_irradiance re-export diverged from leaf"
    );

    let leaf_am: fn(f64) -> f64 = leaf_path::relative_airmass;
    let sim_am: fn(f64) -> f64 = sim_path::relative_airmass;
    assert_eq!(
        leaf_am as usize, sim_am as usize,
        "relative_airmass re-export diverged from leaf"
    );
}
