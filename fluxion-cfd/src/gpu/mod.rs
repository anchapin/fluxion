//! GPU-accelerated kernels for FFD

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
