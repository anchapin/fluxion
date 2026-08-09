//! GPU-accelerated kernels for FFD (issue #2386 / #2456).
//!
//! **Status: scaffolding only.** No CUDA / OpenCL kernels are
//! implemented as of issue #2456. The `GpuBackend` enum and the
//! `get_available_backend` / `supports_gpu` accessors are the
//! architecture contract that a future kernel port must respect, but
//! the dispatch paths are not wired — the accessors return `CPU` and
//! `false` respectively.
//!
//! The CPU baseline the GPU port must beat lives in
//! [`crate::cpu`] (issue #2456 First Step) and is bit-identical to
//! the top-level loops on a 32³ grid (enforced by
//! `fluxion-cfd/tests/cpu_gpu_parity.rs`). CUDA kernel authoring is
//! out of scope for #2456.

// Imports kept available for the future CUDA/OpenCL kernel stubs (issue #2456
// is GPU-scaffolding-only; the kernel code that will reference these types is
// out of scope for this issue).
#[allow(unused_imports)]
use crate::{CfdError, CfdResult, Field3d, Grid3d, VelocityField};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GpuBackend {
    #[default]
    CPU,
    CUDA,
    OpenCL,
}

pub struct GpuConfig {
    pub backend: GpuBackend,
    pub device_id: usize,
    pub max_blocks: usize,
    pub threads_per_block: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::CPU,
            device_id: 0,
            max_blocks: 256,
            threads_per_block: 256,
        }
    }
}

pub fn get_available_backend() -> GpuBackend {
    GpuBackend::CPU
}

pub fn supports_gpu() -> bool {
    false
}
