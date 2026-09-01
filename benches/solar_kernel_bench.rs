use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fluxion::solar::solar_position::{
    calculate_day_of_year, calculate_solar_position, SolarPosition,
};
use fluxion::solar::surface_irradiance::calculate_surface_irradiance;
use fluxion::solar::surface_irradiance::{Orientation, PerezSkyModel};

fn bench_perez_diffuse_tilted(c: &mut Criterion) {
    let dni = 800.0;
    let dhi = 100.0;
    let day_of_year = 172;

    let surface_tilts = [0.0, 30.0, 60.0, 90.0];
    let surface_azimuths = [0.0, 90.0, 180.0, 270.0];

    let mut group = c.benchmark_group("perez_diffuse_tilted");
    for &tilt in &surface_tilts {
        for &azimuth in &surface_azimuths {
            let sun_pos = SolarPosition {
                altitude_deg: 45.0,
                azimuth_deg: 180.0,
                zenith_deg: 45.0,
            };
            let dni_extra =
                fluxion::solar::surface_irradiance::extraterrestrial_irradiance(day_of_year);
            let airmass = fluxion::solar::surface_irradiance::relative_airmass(sun_pos.zenith_deg);

            let id = BenchmarkId::new(format!("tilt{}_az{}", tilt as u32, azimuth as u32), dhi);
            group.bench_with_input(id, &dhi, |b, &dhi_val| {
                b.iter(|| {
                    let _result = PerezSkyModel::calculate_diffuse_tilted(
                        black_box(dhi_val),
                        black_box(dni),
                        black_box(dni_extra),
                        black_box(airmass),
                        black_box(sun_pos.zenith_deg),
                        black_box(tilt),
                        black_box(azimuth),
                        black_box(sun_pos.azimuth_deg),
                    );
                })
            });
        }
    }
    group.finish();
}

fn bench_perez_diffuse_vectorized(c: &mut Criterion) {
    let day_of_year = 172;
    let dni_extra = fluxion::solar::surface_irradiance::extraterrestrial_irradiance(day_of_year);
    let airmass = fluxion::solar::surface_irradiance::relative_airmass(45.0);

    let n_calls = 10_000;
    let dni_vals = vec![800.0; n_calls];
    let dhi_vals = vec![100.0; n_calls];
    let tilt_vals = vec![90.0; n_calls];
    let azimuth_vals = vec![180.0; n_calls];
    let zenith_vals = vec![45.0; n_calls];
    let solar_azimuth_vals = vec![180.0; n_calls];

    c.bench_function("perez_diffuse_vectorized_10k", |b| {
        b.iter(|| {
            for i in 0..n_calls {
                black_box(PerezSkyModel::calculate_diffuse_tilted(
                    black_box(dhi_vals[i]),
                    black_box(dni_vals[i]),
                    black_box(dni_extra),
                    black_box(airmass),
                    black_box(zenith_vals[i]),
                    black_box(tilt_vals[i]),
                    black_box(azimuth_vals[i]),
                    black_box(solar_azimuth_vals[i]),
                ));
            }
        })
    });
}

fn bench_calculate_surface_irradiance(c: &mut Criterion) {
    let sun_pos = SolarPosition {
        altitude_deg: 45.0,
        azimuth_deg: 180.0,
        zenith_deg: 45.0,
    };
    let dni = 800.0;
    let dhi = 100.0;

    c.bench_function("calculate_surface_irradiance_south_vertical", |b| {
        b.iter(|| {
            let _result = calculate_surface_irradiance(
                black_box(&sun_pos),
                black_box(dni),
                black_box(dhi),
                black_box(None),
                black_box(Orientation::South),
                black_box(0.2),
                black_box(172),
            );
        })
    });
}

fn bench_solar_position_diurnal_cycle(c: &mut Criterion) {
    let latitude = 39.7;
    let longitude = -105.0;
    let year = 2024;
    let month = 6;
    let day = 21;

    let hours: Vec<f64> = (0..24).map(|h| h as f64).collect();

    c.bench_function("solar_position_diurnal_cycle_jun_solstice", |b| {
        b.iter(|| {
            for &hour in &hours {
                let _pos = calculate_solar_position(
                    black_box(latitude),
                    black_box(longitude),
                    black_box(year),
                    black_box(month),
                    black_box(day),
                    black_box(hour),
                    black_box(None),
                );
            }
        })
    });
}

fn bench_solar_position_single(c: &mut Criterion) {
    c.bench_function("solar_position_noon_jun_solstice_denver", |b| {
        b.iter(|| {
            let _pos = calculate_solar_position(
                black_box(39.7),
                black_box(-105.0),
                black_box(2024),
                black_box(6),
                black_box(21),
                black_box(12.0),
                black_box(None),
            );
        })
    });
}

fn bench_day_of_year(c: &mut Criterion) {
    c.bench_function("day_of_year_jun_21", |b| {
        b.iter(|| {
            let _doy = calculate_day_of_year(black_box(2024), black_box(6), black_box(21));
        })
    });
}

fn bench_surface_irradiance_multiple_orientations(c: &mut Criterion) {
    let sun_pos = SolarPosition {
        altitude_deg: 45.0,
        azimuth_deg: 180.0,
        zenith_deg: 45.0,
    };
    let orientations = [
        Orientation::South,
        Orientation::East,
        Orientation::West,
        Orientation::North,
        Orientation::Up,
    ];

    c.bench_function("surface_irradiance_5_orientations", |b| {
        b.iter(|| {
            for &orient in &orientations {
                let _result = calculate_surface_irradiance(
                    black_box(&sun_pos),
                    black_box(800.0),
                    black_box(100.0),
                    black_box(None),
                    black_box(orient),
                    black_box(0.2),
                    black_box(172),
                );
            }
        })
    });
}

criterion_group!(
    solar_kernel_benches,
    bench_perez_diffuse_tilted,
    bench_perez_diffuse_vectorized,
    bench_calculate_surface_irradiance,
    bench_solar_position_diurnal_cycle,
    bench_solar_position_single,
    bench_day_of_year,
    bench_surface_irradiance_multiple_orientations,
);
criterion_main!(solar_kernel_benches);
