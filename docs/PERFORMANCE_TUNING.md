# Performance Tuning Guide

This guide provides recommendations and instructions for optimizing the performance of Fluxion's surrogate models, specifically focusing on inference throughput and latency.

## Hardware Recommendations

For massive population evaluations (100K+ configs/sec), we recommend the following hardware configurations:

*   **GPU**: NVIDIA GPU with Tensor Cores (e.g., V100, A100, RTX 3090/4090).
*   **CPU**: High core count CPU (e.g., AMD EPYC, Intel Xeon) to feed the GPU.
*   **RAM**: Sufficient system RAM to hold the population data (16GB+).

## Inference Optimization

### 1. Batch Size

Batching is critical for maximizing GPU utilization.

*   **Small Batches (1-10)**: High latency per sample, low throughput. Suitable for real-time, single-sample predictions.
*   **Medium Batches (100-1000)**: Good balance between latency and throughput.
*   **Large Batches (10000+)**: Highest throughput. Recommended for population-based optimization (Genetic Algorithms, Particle Swarm).

Use `tools/benchmark_throughput.py` to find the optimal batch size for your hardware:

```bash
python tools/benchmark_throughput.py --model assets/thermal_surrogate.onnx --batch-sizes 100,1000,10000 --backends cuda
```

### 2. Model Quantization

Quantization reduces model size and precision (e.g., Float32 -> Int8) to improve inference speed with minimal accuracy loss.

*   **Int8 Quantization**: approximately 4x smaller model size, 1.5x-2x speedup on supported hardware.

Use `tools/quantize_model.py` to create a quantized model:

```bash
python tools/quantize_model.py --model assets/thermal_surrogate.onnx --output assets/thermal_surrogate_int8.onnx
```

### 3. GPU Acceleration

Fluxion supports multiple GPU backends via ONNX Runtime:

*   **CUDA**: Standard for NVIDIA GPUs.
*   **TensorRT**: Optimized inference engine for NVIDIA GPUs (requires TensorRT installation).
*   **CoreML**: For Apple Silicon (M1/M2/M3) devices.
*   **DirectML**: For Windows devices with DirectX 12 compatible GPUs.

Ensure you have the appropriate drivers and libraries installed (e.g., CUDA Toolkit, cuDNN).

## Multi-Device Inference

For extremely large scale simulations, Fluxion can distribute inference across multiple GPUs.

```rust
use fluxion::ai::distributed::DistributedSurrogateManager;
use fluxion::ai::surrogate::InferenceBackend;

let device_ids = vec![0, 1, 2, 3]; // Use 4 GPUs
let manager = DistributedSurrogateManager::new("model.onnx", InferenceBackend::CUDA, &device_ids)?;

let results = manager.evaluate_population_distributed(population)?;
```

## Benchmarking

Regularly benchmark your setup to ensure optimal performance.

```bash
python tools/benchmark_throughput.py --model assets/thermal_surrogate.onnx --backends cpu,cuda
```

## Troubleshooting

*   **"Provider not available"**: Check if `onnxruntime-gpu` is installed and compatible with your CUDA version.
*   **Out of Memory (OOM)**: Reduce batch size.
*   **Low GPU Usage**: Increase batch size or use multiple threads to feed data.
