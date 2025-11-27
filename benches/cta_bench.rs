use criterion::{criterion_group, criterion_main, Criterion};
use fluxion::physics::cta::{ContinuousTensor, VectorField};
use fluxion::physics::nd_array::NDArrayField;

fn bench_map_vector(c: &mut Criterion) {
    let v = VectorField::new((0..10_000).map(|i| i as f64).collect());
    c.bench_function("vector_map", |b| {
        b.iter(|| {
            let _ = v.map(|x| x * 1.001);
        })
    });
}

fn bench_map_ndarray(c: &mut Criterion) {
    let v = NDArrayField::from_shape_vec(vec![10_000], (0..10_000).map(|i| i as f64).collect());
    c.bench_function("ndarray_map", |b| {
        b.iter(|| {
            let _ = v.map(|x| x * 1.001);
        })
    });
}

criterion_group!(cta_benches, bench_map_vector, bench_map_ndarray);
criterion_main!(cta_benches);
