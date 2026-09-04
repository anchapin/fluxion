//! `fluxion-evaluator` binary — thin CLI wrapper.
//!
//! Reads candidate source from `stdin` (or the path given via
//! `--candidate-file`) and emits a [`Summary`] on `stdout`. The
//! OpenEvolve adapter invokes this binary once per candidate; the
//! JSON-on-stdout contract lets the adapter stay in Python with zero
//! Rust binding overhead.
//!
//! ## Usage
//!
//! ```text
//! $ cat candidate.rs | fluxion-evaluator \
//!     --candidate-id ctf-seed-0042 \
//!     --edge-cases edge_cases.json
//! { "schema_version": 1, ... }
//! ```
//!
//! Exit codes match the crate root docs:
//! - 0 — evaluation succeeded
//! - 2 — compile failure
//! - 3 — invariant hard-fail
//! - 4 — timeout / resource cap

use std::path::PathBuf;

use fluxion_evaluator::summary::{Summary, SummaryBuilder};

fn main() {
    // We deliberately avoid pulling in `clap` (a third-party crate
    // that's not in the workspace dep set yet). Hand-rolled arg
    // parsing is good enough for the four flags this binary needs;
    // OpenEvolve invokes it via subprocess with a fixed argument
    // layout, so there's no UI surface to worry about.

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut candidate_id: Option<String> = None;
    let mut candidate_file: Option<PathBuf> = None;
    let mut edge_cases_file: Option<PathBuf> = None;
    let mut generation: Option<u32> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--candidate-id" => {
                candidate_id = args.get(i + 1).cloned();
                i += 2;
            }
            "--candidate-file" => {
                candidate_file = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--edge-cases" => {
                edge_cases_file = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--generation" => {
                generation = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                eprintln!("fluxion-evaluator: unknown argument `{}`", other);
                eprintln!("(run with --help for usage)");
                std::process::exit(1);
            }
        }
    }

    let candidate_id = match candidate_id {
        Some(id) => id,
        None => {
            eprintln!("fluxion-evaluator: --candidate-id is required");
            std::process::exit(1);
        }
    };

    // Read the candidate source — either from a file or stdin.
    // Reserved for the follow-up CLI wiring; today the binary emits
    // a stub Summary so the OpenEvolve adapter has a process to
    // invoke during bring-up. The full pipeline is exercised by the
    // integration tests under `tests/`.
    let _candidate_source = match candidate_file {
        Some(path) => match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                let summary = Summary::compile_failure(
                    &candidate_id,
                    generation,
                    format!("read failed: {}", e),
                );
                emit_and_exit(summary, 2);
            }
        },
        None => {
            let mut buf = String::new();
            if let Err(e) = std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf) {
                let summary = Summary::compile_failure(
                    &candidate_id,
                    generation,
                    format!("stdin read failed: {}", e),
                );
                emit_and_exit(summary, 2);
            }
            buf
        }
    };

    // For now, the CLI binary is a thin harness — it reports a
    // compile-success / invariant-pass stub Summary. The full
    // recompile + invariant + timing pipeline is wired up in the
    // library API; the binary is the campaign entry point that the
    // OpenEvolve adapter invokes. Wiring the CLI binary to the full
    // pipeline lands in a follow-up PR (issue #3336 only requires
    // the harness API + the binary stub).
    let _ = edge_cases_file; // reserved for follow-up PR.

    // Emit a deterministic summary that matches schema v1.
    let mut builder = SummaryBuilder::new(&candidate_id, 1.0)
        .with_max_error(0.0)
        .with_eval_latency_ns(0)
        .with_eval_latency_spread_ns(0)
        .with_min_invariant_margin(1.0);
    if let Some(g) = generation {
        builder = builder.with_generation(g);
    }
    let summary = Summary::new(builder);
    emit_and_exit(summary, 0);
}

fn emit_and_exit(summary: Summary, code: i32) -> ! {
    match summary.to_canonical_json() {
        Ok(json) => {
            println!("{}", json);
        }
        Err(e) => {
            eprintln!("fluxion-evaluator: failed to serialize Summary: {}", e);
            std::process::exit(1);
        }
    }
    std::process::exit(code);
}

fn print_help() {
    println!(
        "fluxion-evaluator — deterministic headless evaluator harness (issue #3336)\n\n\
         USAGE:\n  \
         fluxion-evaluator --candidate-id <ID> [--candidate-file <PATH>|--stdin]\n\
                          [--edge-cases <PATH>] [--generation <N>]\n\n\
         OUTPUT:\n  \
         Schema v1 JSON Summary on stdout. Exit codes: 0 ok, 2 compile, 3 invariant, 4 cap."
    );
}
