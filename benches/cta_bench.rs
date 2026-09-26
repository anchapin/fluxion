use criterion::{criterion_group, criterion_main, Criterion};
use fluxion::physics::cta::{ContinuousTensor, VectorField};

fn bench_map_raw(c: &mut Criterion) {
    let v: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    c.bench_function("raw_map", |b| {
        b.iter(|| v.iter().map(|x| x * 1.001).collect::<Vec<f64>>())
    });
}

fn bench_map_vector(c: &mut Criterion) {
    let v = VectorField::new((0..10_000).map(|i| i as f64).collect());
    c.bench_function("vector_map", |b| b.iter(|| v.map(|x| x * 1.001)));
}

criterion_group!(cta_benches, bench_map_raw, bench_map_vector);
criterion_main!(cta_benches);
