import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort


def benchmark_throughput(
    model_path: str, batch_sizes: list[int], backends: list[str], output_file: str
):
    """
    Run comprehensive throughput benchmarks.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    results: dict = {"model": model_path, "timestamp": time.time(), "benchmarks": []}

    # Inspect model
    try:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape

        input_dim = 2
        if len(input_shape) == 2 and isinstance(input_shape[1], int):
            input_dim = input_shape[1]

        print(f"Model input dim: {input_dim}")
    except Exception as e:
        print(f"Failed to inspect model: {e}")
        return

    available_providers = ort.get_available_providers()
    print(f"Available ORT providers: {available_providers}")

    for backend in backends:
        provider = None
        if backend.lower() == "cpu":
            provider = "CPUExecutionProvider"
        elif backend.lower() == "cuda":
            provider = "CUDAExecutionProvider"
        elif backend.lower() == "coreml":
            provider = "CoreMLExecutionProvider"
        elif backend.lower() == "directml":
            provider = "DirectMLExecutionProvider"
        elif backend.lower() == "openvino":
            provider = "OpenVINOExecutionProvider"

        if not provider:
            print(f"Unknown backend: {backend}")
            continue

        if provider not in available_providers:
            print(f"Provider {provider} not available, skipping.")
            continue

        print(f"Benchmarking backend: {backend} ({provider})")

        try:
            sess = ort.InferenceSession(model_path, providers=[provider])
        except Exception as e:
            print(f"Failed to create session for {backend}: {e}")
            continue

        for batch_size in batch_sizes:
            print(f"  Batch size: {batch_size}")

            # Generate dummy data
            X = np.random.randn(batch_size, input_dim).astype(np.float32)

            # Warmup
            for _ in range(10):
                sess.run(None, {input_name: X})

            # Benchmark
            iterations = max(
                10, 10000 // batch_size
            )  # Adjust iterations based on batch size
            start_time = time.time()
            for _ in range(iterations):
                sess.run(None, {input_name: X})
            end_time = time.time()

            duration = end_time - start_time
            throughput = (batch_size * iterations) / duration

            print(f"    Throughput: {throughput:.2f} samples/sec")

            results["benchmarks"].append(
                {
                    "backend": backend,
                    "batch_size": batch_size,
                    "throughput": throughput,
                    "latency_ms": (duration / iterations) * 1000,
                }
            )

    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive inference throughput benchmark."
    )
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,10,100,1000,10000",
        help="Comma-separated batch sizes",
    )
    parser.add_argument(
        "--backends", type=str, default="cpu,cuda", help="Comma-separated backends"
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.json", help="Output JSON file"
    )

    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    backends = args.backends.split(",")

    benchmark_throughput(args.model, batch_sizes, backends, args.output)
