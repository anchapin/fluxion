use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fluxion_city::{
    ashrae140,
    nusselt::{self, ViewFactorMatrix},
    sparse::{create_sparse_from_urban_canyon, SparseViewFactorMatrix, UrbanRadiationSolver},
};

fn create_urban_canyon_surfaces(n_buildings: usize) -> (Vec<(f64, f64, f64)>, f64) {
    let mut walls = Vec::with_capacity(n_buildings);
    for i in 0..n_buildings {
        let height = 10.0 + (i as f64 * 0.5);
        let width = 5.0 + (i as f64 * 0.3);
        let spacing = 3.0 + (i as f64 * 0.2);
        walls.push((width * 10.0, height, spacing));
    }
    let ground_area = 1000.0 + (n_buildings as f64 * 10.0);
    (walls, ground_area)
}

fn create_rectangular_enclosure(n_surfaces: usize) -> Vec<(f64, f64)> {
    let mut surfaces = Vec::with_capacity(n_surfaces);
    for i in 0..n_surfaces {
        let area = 50.0 + (i as f64 * 10.0);
        let height = 3.0 + (i as f64 * 0.1);
        surfaces.push((area, height));
    }
    surfaces
}

fn benchmark_view_factor_enclosure(c: &mut Criterion) {
    let mut group = c.benchmark_group("view_factor_enclosure");

    for n in [3, 5, 10, 20, 50].iter() {
        let surfaces = create_rectangular_enclosure(*n);

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter(|| nusselt::view_factor_enclosure(black_box(&surfaces)).unwrap());
        });
    }

    group.finish();
}

fn benchmark_urban_canyon_view_factors(c: &mut Criterion) {
    let mut group = c.benchmark_group("urban_canyon_view_factors");

    for n in [5, 10, 20, 50, 100].iter() {
        let (walls, ground_area) = create_urban_canyon_surfaces(*n);

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter(|| {
                nusselt::compute_urban_canyon_view_factors(
                    black_box(&walls),
                    black_box(ground_area),
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

fn benchmark_sparse_matrix_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matrix_creation");

    for n in [5, 10, 20, 50, 100].iter() {
        let (walls, ground_area) = create_urban_canyon_surfaces(*n);

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter(|| {
                create_sparse_from_urban_canyon(black_box(&walls), black_box(ground_area)).unwrap()
            });
        });
    }

    group.finish();
}

fn benchmark_sparse_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matrix_multiplication");

    for n in [5, 10, 20, 50, 100].iter() {
        let (walls, ground_area) = create_urban_canyon_surfaces(*n);
        let sparse = create_sparse_from_urban_canyon(&walls, ground_area).unwrap();
        let vec: Vec<f64> = vec![1.0; n + 1];

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter(|| sparse.multiply_dense(black_box(&vec)));
        });
    }

    group.finish();
}

fn benchmark_radiation_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("radiation_solver");

    for n in [5, 10, 20, 50].iter() {
        let (walls, ground_area) = create_urban_canyon_surfaces(*n);
        let dense = nusselt::compute_urban_canyon_view_factors(&walls, ground_area).unwrap();

        let mut all_areas = Vec::new();
        for &(area, _, _) in &walls {
            all_areas.push(area);
        }
        all_areas.push(ground_area);

        let emissivities = vec![0.9; walls.len() + 1];
        let solver =
            UrbanRadiationSolver::from_dense_enclosure(&dense, all_areas.clone(), emissivities)
                .unwrap();
        let temperatures: Vec<f64> = vec![293.15; walls.len() + 1];

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter(|| solver.compute_radiation_exchange(black_box(&temperatures)));
        });
    }

    group.finish();
}

fn benchmark_reciprocity_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("reciprocity_verification");

    for n in [5, 10, 20, 50, 100].iter() {
        let surfaces = create_rectangular_enclosure(*n);
        let f = nusselt::view_factor_enclosure(&surfaces).unwrap();
        let matrix = ViewFactorMatrix::from_dense(f);
        let areas: Vec<f64> = surfaces.iter().map(|(a, _)| *a).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter(|| matrix.verify_reciprocity(black_box(&areas)));
        });
    }

    group.finish();
}

fn benchmark_summation_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("summation_verification");

    for n in [5, 10, 20, 50, 100].iter() {
        let surfaces = create_rectangular_enclosure(*n);
        let f = nusselt::view_factor_enclosure(&surfaces).unwrap();
        let matrix = ViewFactorMatrix::from_dense(f);

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter(|| matrix.verify_summation());
        });
    }

    group.finish();
}

fn benchmark_ashrae140_cases(c: &mut Criterion) {
    let cases = ashrae140::reference_configurations();

    let mut group = c.benchmark_group("ashrae140_cases");

    for case in cases {
        group.bench_function(case.name.clone(), |b| {
            b.iter(|| fluxion_city::verify_ashrae_case(black_box(&case)).unwrap());
        });
    }

    group.finish();
}

fn benchmark_100_surface_urban_canopy(c: &mut Criterion) {
    let n = 100;
    let (walls, ground_area) = create_urban_canyon_surfaces(n);

    let mut group = c.benchmark_group("target_100_surface_urban_canopy");

    group.bench_function("full_pipeline", |b| {
        b.iter(|| {
            let sparse =
                create_sparse_from_urban_canyon(black_box(&walls), black_box(ground_area)).unwrap();
            let vec: Vec<f64> = vec![1.0; n + 1];
            sparse.multiply_dense(black_box(&vec))
        });
    });

    group.bench_function("view_factors_only", |b| {
        b.iter(|| {
            nusselt::compute_urban_canyon_view_factors(black_box(&walls), black_box(ground_area))
                .unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_view_factor_enclosure,
    benchmark_urban_canyon_view_factors,
    benchmark_sparse_matrix_creation,
    benchmark_sparse_matrix_multiplication,
    benchmark_radiation_solver,
    benchmark_reciprocity_verification,
    benchmark_summation_verification,
    benchmark_ashrae140_cases,
    benchmark_100_surface_urban_canopy,
);
criterion_main!(benches);
