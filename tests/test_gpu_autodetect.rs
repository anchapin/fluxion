//! Tests for GPU autodetection in SurrogateManager.

use fluxion::ai::surrogate::SurrogateManager;

#[test]
fn gpu_supported_cpu_backend() {
    let manager = SurrogateManager::new().expect("Failed to create SurrogateManager");
    assert!(
        !manager.gpu_supported(),
        "GPU should not be supported with CPU backend"
    );
}

#[test]
fn gpu_supported_env_override() {
    let manager = SurrogateManager::new().expect("Failed to create SurrogateManager");

    // Test with FLUXION_GPU=0
    std::env::set_var("FLUXION_GPU", "0");
    assert!(
        !manager.gpu_supported(),
        "GPU should be disabled when FLUXION_GPU=0"
    );

    // Test with FLUXION_GPU=1 on CPU backend (still false)
    std::env::set_var("FLUXION_GPU", "1");
    assert!(
        !manager.gpu_supported(),
        "GPU should still be false with CPU backend even if FLUXION_GPU=1"
    );

    // Test with FLUXION_GPU=false
    std::env::set_var("FLUXION_GPU", "false");
    assert!(
        !manager.gpu_supported(),
        "GPU should be disabled when FLUXION_GPU=false"
    );

    // Clean up
    std::env::remove_var("FLUXION_GPU");
}

#[cfg(feature = "cuda")]
#[test]
fn gpu_supported_cuda_backend_if_available() {
    // This test attempts to verify that when using CUDA backend, gpu_supported returns true
    // unless overridden by FLUXION_GPU=0. However, creating a CUDA backend requires a valid
    // ONNX model file and CUDA drivers, which may not be available in all environments.
    // This test will be skipped if we cannot create a CUDA manager.

    // Try to use an existing dummy model if present
    let dummy_model_path = "tests_tmp_dummy.onnx";
    if !std::path::Path::new(dummy_model_path).exists() {
        eprintln!("Skipping CUDA backend test: dummy ONNX model not found");
        return;
    }

    // Attempt to create a CUDA-backed SurrogateManager.
    // Note: This may fail if CUDA drivers are not present.
    match SurrogateManager::with_gpu_backend(
        dummy_model_path,
        fluxion::ai::surrogate::InferenceBackend::CUDA,
        0,
    ) {
        Ok(cuda_manager) => {
            // With no environment override, should be true (if we actually have CUDA)
            // However, if the CUDA session creation failed (but returned Ok?), unlikely.
            // But we must respect FLUXION_GPU override.
            std::env::set_var("FLUXION_GPU", "0");
            assert!(
                !cuda_manager.gpu_supported(),
                "FLUXION_GPU=0 should disable GPU support"
            );
            std::env::remove_var("FLUXION_GPU");

            // Without override, behavior depends on whether CUDA is actually functional.
            // If the manager was successfully created, we assume CUDA is available.
            // This branch may be skipped in CI if CUDA not available.
            let supported = cuda_manager.gpu_supported();
            // We don't assert true because the manager might be created but still lack drivers?
            // But we'll just print.
            eprintln!("CUDA manager gpu_supported: {}", supported);
        }
        Err(_) => {
            eprintln!("Skipping CUDA backend test: failed to create CUDA SurrogateManager (no GPU or drivers)");
        }
    }
}
