//! In-tree sample kernels.
//!
//! These concrete implementations of [`crate::kernel::Kernel`] are
//! used by tests in multiple modules (and by `examples/sample_kernel.rs`)
//! to drive the harness's invariant battery and timing aggregations
//! without depending on the recompile path.
//!
//! They are `pub` so integration tests under `tests/` can also use them
//! without re-implementing the contract.

use crate::kernel::{Kernel, KernelError, KernelInput, KernelOutput};

/// A toy kernel that returns the input's params verbatim. Used as
/// the canonical "passes every invariant" reference in the harness's
/// tests.
#[derive(Default, Debug)]
pub struct IdentityKernel;

impl Kernel for IdentityKernel {
    fn evaluate(&self, input: &KernelInput) -> Result<KernelOutput, KernelError> {
        Ok(KernelOutput {
            payload: input.params.clone(),
        })
    }
}
