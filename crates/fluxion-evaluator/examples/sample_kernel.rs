//! Sample seeded kernel — used by the harness's recompile happy-path
//! test and as the canonical example for the README.
//!
//! This file is **not** a runnable binary (it has no `fn main()`).
//! It is the seed that the harness's recompilation path copies into
//! `<tempdir>/src/kernel.rs`. The harness-generated wrapper at
//! `<tempdir>/src/lib.rs` instantiates `Candidate::default()` and
//! calls `evaluate(...)` once per edge case.
//!
//! ```text
//! $ cargo run --example sample_kernel
//! ```
//!
//! ...prints the contents of this file. That output is what an
//! evolver would feed into the recompiler to score a candidate.

use fluxion_evaluator::kernel::{Kernel, KernelError, KernelInput, KernelOutput};

/// The candidate's `pub struct`. The harness-generated
/// `src/lib.rs` (see `crate::recompile::Recompiler`) instantiates
/// `Candidate::default()` and calls `evaluate(&Candidate::default(),
/// input)`.
#[derive(Default)]
pub struct Candidate;

impl Kernel for Candidate {
    fn evaluate(&self, input: &KernelInput) -> Result<KernelOutput, KernelError> {
        // Toy kernel: return the input's params verbatim. The
        // harness compares this against the known-good reference and
        // reports a perfect fitness for matching inputs.
        Ok(KernelOutput {
            payload: input.params.clone(),
        })
    }
}

fn main() {
    // `cargo run --example sample_kernel` prints the seed source so
    // an evolver driver can pipe it into the recompiler.
    print!("{}", include_str!("sample_kernel.rs"));
}
