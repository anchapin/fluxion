//! Criterion bench for Issue #2525 — `ParallelLoopDispatcher::step`.
//!
//! Before #2525 the step closure was wrapped in `Arc<Mutex<F>>` and invoked
//! under a lock inside `par_iter().map(..)`, which collapsed the parallel
//! iteration back to serial execution (every rayon worker had to acquire the
//! same lock to call the closure).
//!
//! This bench measures the fixed `step` path with **> 16 subgraphs** and
//! contrasts it against an equivalent fully-sequential loop. Reading the
//! `decoupled_loop_dispatch/step_parallel_*` vs `step_serial_baseline_*` rows
//! in the criterion output gives the parallel/serial ratio directly. On an
//! 8-core machine the parallel path should be **≥ 4× faster** than the serial
//! baseline now that the `Mutex` is gone (the serial baseline mimics the old
//! locked behaviour).
//!
//! Run with:
//!
//! ```bash
//! cargo bench --bench decoupled_loop_dispatch_bench
//! ```
//!
//! > Parallelism lives at exactly this single `par_iter` level. The closure
//! > body is pure / sequential per subgraph — no nested `par_iter` (AGENTS.md).

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use fluxion::sim::decoupled_loop_rayon::{
    DispatchError, GraphNodeId, ParallelLoopDispatcher, Subgraph,
};

/// Number of independent (no-feedback) subgraphs — kept above 16 per the
/// #2525 acceptance criterion so the workload spans more rayon workers than a
/// typical 8-core machine has hardware threads.
const SUBGRAPH_COUNT: usize = 32;
/// Nodes per subgraph. Deliberately large so the per-subgraph closure is
/// genuinely CPU-bound and dominates rayon scheduling overhead — this lets
/// the parallel/serial ratio approach the hardware core count (≥ 4× on an
/// 8-core machine, per the #2525 AC).
const NODES_PER_SUBGRAPH: usize = 4096;

/// Build `count` independent (no-feedback) subgraphs each holding `nodes_per`
/// graph nodes.
fn build_subgraphs(count: usize, nodes_per: usize) -> Vec<Subgraph> {
    (0..count)
        .map(|id| Subgraph {
            id,
            nodes: (0..nodes_per)
                .map(|n| GraphNodeId::new(id * nodes_per + n))
                .collect(),
            edges: vec![],
            has_feedback: false,
        })
        .collect()
}

/// CPU-bound but pure work for the closure: a deterministic mix of
/// transcendental terms over the subgraph node ids. Stateless (`Fn`), `Send +
/// Sync`, and free of allocation — exactly the shape #2525 requires of `f`.
#[inline]
fn per_subgraph_work(subgraph: &Subgraph) -> u64 {
    let mut acc = 0u64;
    for node in &subgraph.nodes {
        let x = node.0 as f64;
        // Two transcendental evaluations per node keeps the work dense enough
        // to dominate rayon scheduling overhead.
        acc = acc.wrapping_add((x.sin().abs() * 1e6) as u64);
        acc = acc.wrapping_add((x.ln().max(0.0) * 1e6) as u64);
    }
    acc
}

/// Parallel dispatch path — the function fixed by #2525. Reports one
/// "element" per subgraph so criterion prints subgraphs/sec as throughput.
fn bench_step_parallel(c: &mut Criterion) {
    let subgraphs = build_subgraphs(SUBGRAPH_COUNT, NODES_PER_SUBGRAPH);
    let mut dispatcher = ParallelLoopDispatcher::new(subgraphs);
    let count = dispatcher.num_subgraphs() as u64;

    let mut group = c.benchmark_group("decoupled_loop_dispatch");
    group.throughput(Throughput::Elements(count));
    group.sample_size(20);

    group.bench_function("step_parallel_32_subgraphs", |b| {
        b.iter(|| {
            dispatcher
                .step(0.0, 0.001, |sg| {
                    let r = per_subgraph_work(sg);
                    Ok::<u64, DispatchError>(black_box(r))
                })
                .expect("parallel dispatch should succeed");
        })
    });

    group.finish();
}

/// Serial baseline mimicking the pre-#2525 behaviour, where the `Mutex`
/// forced one-at-a-time invocation. Same per-subgraph work, evaluated
/// sequentially. Compare against `step_parallel_32_subgraphs` to read the
/// speedup ratio.
fn bench_step_serial_baseline(c: &mut Criterion) {
    let subgraphs = build_subgraphs(SUBGRAPH_COUNT, NODES_PER_SUBGRAPH);
    let count = subgraphs.len() as u64;

    let mut group = c.benchmark_group("decoupled_loop_dispatch");
    group.throughput(Throughput::Elements(count));
    group.sample_size(20);

    group.bench_function("step_serial_baseline_32_subgraphs", |b| {
        b.iter(|| {
            // Sequential invocation — what the `Arc<Mutex<F>>` enforced.
            let mut acc = 0u64;
            for sg in &subgraphs {
                acc = acc.wrapping_add(per_subgraph_work(sg));
            }
            black_box(acc);
        })
    });

    group.finish();
}

criterion_group!(
    decoupled_loop_dispatch,
    bench_step_parallel,
    bench_step_serial_baseline,
);
criterion_main!(decoupled_loop_dispatch);
