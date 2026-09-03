use criterion::{criterion_group, criterion_main, Criterion};

mod shipyard_bench {
    use shipyard::{Component, Get};

    #[derive(Clone, Copy, Debug, PartialEq, Component)]
    pub struct Position {
        pub x: f64,
        pub y: f64,
        pub z: f64,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Component)]
    pub struct Velocity {
        pub dx: f64,
        pub dy: f64,
        pub dz: f64,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Component)]
    pub struct Mass {
        pub value: f64,
    }

    pub fn create_entities(count: usize) {
        let world = &mut shipyard::World::new();

        for i in 0..count {
            let entity = world.add_entity((
                Position {
                    x: i as f64,
                    y: i as f64,
                    z: i as f64,
                },
                Velocity {
                    dx: 1.0,
                    dy: 1.0,
                    dz: 1.0,
                },
                Mass { value: 1.0 },
            ));
            std::hint::black_box(entity);
        }
    }

    pub fn archetype_iteration(count: usize) {
        let world = &mut shipyard::World::new();

        for i in 0..count {
            world.add_entity((
                Position {
                    x: i as f64,
                    y: i as f64,
                    z: i as f64,
                },
                Velocity {
                    dx: 1.0,
                    dy: 1.0,
                    dz: 1.0,
                },
                Mass { value: 1.0 },
            ));
        }

        let entities = world.borrow::<shipyard::EntitiesView>().unwrap();
        let positions = world.borrow::<shipyard::View<Position>>().unwrap();
        let velocities = world.borrow::<shipyard::View<Velocity>>().unwrap();

        let mut sum = 0.0;
        for entity in entities.iter() {
            let pos = positions.get(entity).unwrap();
            let vel = velocities.get(entity).unwrap();
            sum += pos.x * vel.dx + pos.y * vel.dy + pos.z * vel.dz;
            std::hint::black_box(sum);
        }
    }

    pub fn insert_and_remove(count: usize) {
        let world = &mut shipyard::World::new();

        let ids: Vec<_> = (0..count)
            .map(|i| {
                world.add_entity((
                    Position {
                        x: i as f64,
                        y: i as f64,
                        z: i as f64,
                    },
                    Velocity {
                        dx: 1.0,
                        dy: 1.0,
                        dz: 1.0,
                    },
                    Mass { value: 1.0 },
                ))
            })
            .collect();

        std::hint::black_box(&ids);

        drop(ids);
    }
}

mod hecs_bench {
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Position {
        pub x: f64,
        pub y: f64,
        pub z: f64,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Velocity {
        pub dx: f64,
        pub dy: f64,
        pub dz: f64,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Mass {
        pub value: f64,
    }

    pub fn create_entities(count: usize) {
        let mut world = hecs::World::new();

        for i in 0..count {
            world.spawn((
                Position {
                    x: i as f64,
                    y: i as f64,
                    z: i as f64,
                },
                Velocity {
                    dx: 1.0,
                    dy: 1.0,
                    dz: 1.0,
                },
                Mass { value: 1.0 },
            ));
        }
        std::hint::black_box(&world);
    }

    pub fn archetype_iteration(count: usize) {
        let mut world = hecs::World::new();

        for i in 0..count {
            world.spawn((
                Position {
                    x: i as f64,
                    y: i as f64,
                    z: i as f64,
                },
                Velocity {
                    dx: 1.0,
                    dy: 1.0,
                    dz: 1.0,
                },
                Mass { value: 1.0 },
            ));
        }

        let mut sum = 0.0;
        for (pos, vel) in world.query::<(&Position, &Velocity)>().iter() {
            sum += pos.x * vel.dx + pos.y * vel.dy + pos.z * vel.dz;
            std::hint::black_box(sum);
        }
    }

    pub fn insert_and_remove(count: usize) {
        let mut world = hecs::World::new();

        let ids: Vec<_> = (0..count)
            .map(|i| {
                world.spawn((
                    Position {
                        x: i as f64,
                        y: i as f64,
                        z: i as f64,
                    },
                    Velocity {
                        dx: 1.0,
                        dy: 1.0,
                        dz: 1.0,
                    },
                    Mass { value: 1.0 },
                ))
            })
            .collect();

        std::hint::black_box(&ids);

        for id in ids {
            world.despawn(id).unwrap();
        }
    }
}

pub fn shipyard_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("shipyard");
    group.measurement_time(std::time::Duration::from_secs(3));

    group.bench_function("create_entities_10k", |b| {
        b.iter(|| shipyard_bench::create_entities(std::hint::black_box(10_000)));
    });

    group.bench_function("archetype_iteration_10k", |b| {
        b.iter(|| shipyard_bench::archetype_iteration(std::hint::black_box(10_000)));
    });

    group.bench_function("insert_remove_10k", |b| {
        b.iter(|| shipyard_bench::insert_and_remove(std::hint::black_box(10_000)));
    });

    group.finish();
}

pub fn hecs_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("hecs");
    group.measurement_time(std::time::Duration::from_secs(3));

    group.bench_function("create_entities_10k", |b| {
        b.iter(|| hecs_bench::create_entities(std::hint::black_box(10_000)));
    });

    group.bench_function("archetype_iteration_10k", |b| {
        b.iter(|| hecs_bench::archetype_iteration(std::hint::black_box(10_000)));
    });

    group.bench_function("insert_remove_10k", |b| {
        b.iter(|| hecs_bench::insert_and_remove(std::hint::black_box(10_000)));
    });

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = shipyard_benchmarks, hecs_benchmarks
);
criterion_main!(benches);
