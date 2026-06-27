import argparse
import os
import time

import numpy as np
import onnxruntime as ort


def benchmark_batch_sizes(model_path: str, max_batch: int = 1000):
    """Find optimal batch size for throughput."""

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    # Check available providers
    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")

    # Use CUDA if available, otherwise CPU
    providers = (
        ["CUDAExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers
        else ["CPUExecutionProvider"]
    )
    print(f"Using providers: {providers}")

    try:
        sess = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        print(f"Failed to create inference session: {e}")
        return

    # Inspect model input to determine shape
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    # Assuming input shape is [batch_size, input_dim] or similar.
    # If shape has a symbolic dimension (like None or 'batch'), we can use
    # variable batch size.
    # If dimensions are fixed, we might have issues with batching if the model
    # doesn't support it.

    # Try to infer input dimension from shape
    input_dim = 2  # Default fallback
    if len(input_shape) == 2:
        if isinstance(input_shape[1], int):
            input_dim = input_shape[1]

    print(
        f"Input name: {input_name}, Input shape: {input_shape}, "
        f"inferred dim: {input_dim}"
    )

    for batch_size in [1, 10, 100, 1000, 10000]:
        if batch_size > max_batch:
            break

        # Generate dummy batch
        X = np.random.randn(batch_size, input_dim).astype(np.float32)

        # Measure inference time
        t0 = time.time()
        iterations = 100
        for _ in range(iterations):
            sess.run(None, {input_name: X})
        t1 = time.time()

        total_time = t1 - t0
        throughput = (batch_size * iterations) / total_time
        print(
            f"Batch {batch_size:5d}: {throughput:10.0f} configs/sec "
            f"({total_time / iterations * 1000:.2f} ms/batch)"
        )


def compare_cpu_cuda(model_path: str, rel_tol: float = 1e-5, max_batch: int = 1000):
    """Side-by-side CPU vs CUDA parity report (issue #1336).

    Loads the same ONNX model under both execution providers, runs the
    same fixed input batch, and asserts max relative error <= `rel_tol`
    per tensor element. If CUDAExecutionProvider is not available at
    runtime the function exits early with a clear message — the CUDA
    branch cannot run without GPU hardware (per AGENTS.md: no parameter
    tuning, hardware-in-loop CI handles the GPU path).
    """
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return 1

    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")

    if "CUDAExecutionProvider" not in available_providers:
        print(
            "compare_cpu_cuda: CUDAExecutionProvider not available at runtime — "
            "skipping parity sweep. Run on hardware-in-loop CI with an NVIDIA "
            "GPU to exercise the full CPU-vs-CUDA envelope (issue #1336)."
        )
        return 0

    input_dim = 2  # fallback
    cpu_sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    cuda_sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
    input_name = cpu_sess.get_inputs()[0].name
    input_shape = cpu_sess.get_inputs()[0].shape
    if len(input_shape) == 2 and isinstance(input_shape[1], int):
        input_dim = input_shape[1]

    rng = np.random.default_rng(seed=1336)
    print(f"Input name: {input_name}, Input shape: {input_shape}, dim: {input_dim}")
    print(f"Relative-error envelope: {rel_tol}")

    overall_max_rel = 0.0
    overall_worst = (None, None)
    for batch_size in [1, 10, 100, max(1000, min(max_batch, 1000))]:
        if batch_size > max_batch:
            break
        X = rng.standard_normal((batch_size, input_dim)).astype(np.float32)

        cpu_out = cpu_sess.run(None, {input_name: X})[0]
        cuda_out = cuda_sess.run(None, {input_name: X})[0]

        # Per-element relative error, denominator clamped to 1e-9 to
        # avoid spurious blow-up at near-zero outputs.
        denom = np.maximum(np.abs(cpu_out), 1e-9)
        rel_err = np.abs(cpu_out - cuda_out) / denom
        max_rel = float(rel_err.max())
        if max_rel > overall_max_rel:
            overall_max_rel = max_rel
            idx = int(np.argmax(rel_err))
            overall_worst = (
                float(cpu_out.flatten()[idx]) if cpu_out.size else None,
                float(cuda_out.flatten()[idx]) if cuda_out.size else None,
            )

        status = "PASS" if max_rel <= rel_tol else "FAIL"
        print(
            f"Batch {batch_size:5d}: max_rel_err={max_rel:.3e}  "
            f"shape={cpu_out.shape}  [{status}]"
        )

    verdict = "PASS" if overall_max_rel <= rel_tol else "FAIL"
    print(
        f"\nCPU-vs-CUDA parity report (issue #1336): {verdict}\n"
        f"  overall_max_rel_err = {overall_max_rel:.3e}\n"
        f"  tolerance           = {rel_tol:.3e}\n"
        f"  worst_cpu_value     = {overall_worst[0]}\n"
        f"  worst_cuda_value    = {overall_worst[1]}"
    )
    return 0 if overall_max_rel <= rel_tol else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ONNX inference throughput.")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument(
        "--max-batch", type=int, default=10000, help="Maximum batch size to test"
    )
    parser.add_argument(
        "--compare-cpu-cuda",
        action="store_true",
        help=(
            "Run a side-by-side CPU-vs-CUDA parity sweep instead of "
            "throughput benchmarking (issue #1336). Requires an ONNX "
            "model and a CUDA-capable runtime; the sweep is skipped "
            "(exit 0) when CUDAExecutionProvider is unavailable."
        ),
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-5,
        help=(
            "Per-element max relative error tolerance for the "
            "--compare-cpu-cuda parity sweep (default: 1e-5, "
            "matching issue #1336 acceptance criterion)."
        ),
    )

    args = parser.parse_args()
    if args.compare_cpu_cuda:
        raise SystemExit(compare_cpu_cuda(args.model, args.rel_tol, args.max_batch))
    benchmark_batch_sizes(args.model, args.max_batch)
