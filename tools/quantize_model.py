#!/usr/bin/env python3
"""Quantize ONNX models using ONNX Runtime for optimized CPU inference.

This script quantizes FP32 ONNX models to INT8 for faster CPU inference.
Quantized models typically achieve 2-4x speedup on CPU with <1% accuracy loss.

Usage:
    python tools/quantize_model.py --model input.onnx --output quantized.onnx --type int8

References:
    - ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/quantization.html
    - onnxruntime.transformers.quantize: https://github.com/microsoft/onnxruntime
"""

import argparse
import sys
import time
from pathlib import Path

try:
    import onnx
except ImportError:
    print("Error: onnx package is required. Install with: pip install onnx")
    sys.exit(1)

try:
    from onnxruntime.transformers.quantize import quantize
except ImportError:
    print("Error: onnxruntime package is required. Install with: pip install onnxruntime")
    sys.exit(1)


def quantize_model(
    model_path: str,
    output_path: str,
    quantization_type: str = "int8",
    benchmark: bool = False,
    benchmark_runs: int = 100,
) -> dict:
    """Quantize an ONNX model to INT8.

    Args:
        model_path: Path to input FP32 ONNX model
        output_path: Path to output quantized model
        quantization_type: Type of quantization ("int8", "uint8", "fp16")
        benchmark: Whether to run benchmark after quantization
        benchmark_runs: Number of benchmark runs

    Returns:
        Dictionary with quantization results and benchmark data
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Validate quantization type
    valid_types = ["int8", "uint8", "fp16"]
    if quantization_type.lower() not in valid_types:
        raise ValueError(f"Invalid quantization type: {quantization_type}. Must be one of {valid_types}")

    print(f"Loading model: {model_path}")
    print(f"Quantization type: {quantization_type}")

    # Determine quantization mode based on type
    if quantization_type.lower() == "fp16":
        quantization_mode = "FP16"
        force_f16 = True
    elif quantization_type.lower() in ["int8", "uint8"]:
        quantization_mode = "IntegerOps"
        force_f16 = False
    else:
        quantization_mode = "IntegerOps"
        force_f16 = False

    # Quantize the model
    start_time = time.time()
    print(f"Quantizing model (this may take a moment)...")

    quantize(
        model=str(model_path),
        quantization_mode=quantization_mode,
        force_f16=force_f16,
        output_model_path=str(output_path),
    )

    quantize_time = time.time() - start_time
    print(f"Quantization completed in {quantize_time:.2f}s")

    # Get file sizes
    fp32_size = model_path.stat().st_size / (1024 * 1024)
    quantized_size = output_path.stat().st_size / (1024 * 1024)
    compression_ratio = fp32_size / quantized_size if quantized_size > 0 else 0

    print(f"FP32 model size:     {fp32_size:.2f} MB")
    print(f"Quantized size:     {quantized_size:.2f} MB")
    print(f"Compression ratio:  {compression_ratio:.2f}x")

    result = {
        "input_path": str(model_path),
        "output_path": str(output_path),
        "quantization_type": quantization_type,
        "fp32_size_mb": fp32_size,
        "quantized_size_mb": quantized_size,
        "compression_ratio": compression_ratio,
        "quantization_time_s": quantize_time,
    }

    # Run benchmark if requested
    if benchmark:
        print(f"\nRunning benchmark ({benchmark_runs} inferences)...")
        benchmark_result = run_benchmark(output_path, benchmark_runs)
        result["benchmark"] = benchmark_result

    return result


def run_benchmark(model_path: str, num_runs: int = 100) -> dict:
    """Run inference benchmark on a quantized model.

    Args:
        model_path: Path to quantized ONNX model
        num_runs: Number of inference runs

    Returns:
        Dictionary with benchmark results
    """
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        print("Warning: onnxruntime required for benchmark. Skipping benchmark.")
        return {}

    # Load model
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Get input shape
    inputs = session.get_inputs()
    if not inputs:
        print("Warning: No inputs found in model. Using default shape.")
        input_shape = [1, 10]
    else:
        input_shape = inputs[0].shape
        # Replace None/dynamic dims with 1
        input_shape = [1 if d is None or d == "" else int(d) for d in input_shape]

    print(f"Input shape: {input_shape}")

    # Generate random input data
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Warmup run
    session.run(None, {inputs[0].name: input_data})

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {inputs[0].name: input_data})
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    throughput = 1000.0 / avg_time  # inferences per second

    print(f"\nBenchmark Results:")
    print(f"  Average: {avg_time:.3f} ms ({throughput:.1f} inf/s)")
    print(f"  Std Dev: {std_time:.3f} ms")
    print(f"  Min:     {min_time:.3f} ms")
    print(f"  Max:     {max_time:.3f} ms")

    return {
        "num_runs": num_runs,
        "avg_time_ms": avg_time,
        "std_time_ms": std_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "throughput_per_sec": throughput,
    }


def compare_models(fp32_path: str, int8_path: str, num_samples: int = 100) -> dict:
    """Compare FP32 and INT8 model outputs to measure accuracy loss.

    Args:
        fp32_path: Path to FP32 model
        int8_path: Path to INT8 model
        num_samples: Number of test samples

    Returns:
        Dictionary with comparison results
    """
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        print("Warning: onnxruntime required for comparison. Skipping.")
        return {}

    # Load models
    fp32_session = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    # Get input info
    fp32_inputs = fp32_session.get_inputs()
    int8_inputs = int8_session.get_inputs()

    if not fp32_inputs or not int8_inputs:
        print("Warning: Could not get input info from models. Skipping comparison.")
        return {}

    input_shape = fp32_inputs[0].shape
    input_shape = [1 if d is None or d == "" else int(d) for d in input_shape]
    input_name = fp32_inputs[0].name
    int8_input_name = int8_inputs[0].name

    # Run comparison
    errors = []
    max_rel_errors = []

    for _ in range(num_samples):
        input_data = np.random.randn(*input_shape).astype(np.float32)

        fp32_out = fp32_session.run(None, {input_name: input_data})[0]
        int8_out = int8_session.run(None, {int8_input_name: input_data})[0]

        # Compute relative error
        diff = np.abs(fp32_out - int8_out)
        rel_error = diff / (np.abs(fp32_out) + 1e-8)
        mean_rel_error = np.mean(rel_error)
        max_rel_error = np.max(rel_error)

        errors.append(mean_rel_error)
        max_rel_errors.append(max_rel_error)

    mean_error = np.mean(errors)
    max_error = np.max(max_rel_errors)

    print(f"\nAccuracy Comparison ({num_samples} samples):")
    print(f"  Mean relative error: {mean_error * 100:.4f}%")
    print(f"  Max relative error:  {max_error * 100:.4f}%")

    return {
        "num_samples": num_samples,
        "mean_relative_error": mean_error,
        "max_relative_error": max_error,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Quantize ONNX models for optimized CPU inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize to INT8
  python tools/quantize_model.py --model model.onnx --output model_int8.onnx

  # Quantize to INT8 with benchmark
  python tools/quantize_model.py --model model.onnx --output model_int8.onnx --benchmark

  # Quantize to FP16
  python tools/quantize_model.py --model model.onnx --output model_fp16.onnx --type fp16
        """,
    )

    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to input ONNX model",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output quantized model",
    )
    parser.add_argument(
        "--type", "-t",
        default="int8",
        choices=["int8", "uint8", "fp16"],
        help="Quantization type (default: int8)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmark after quantization",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=100,
        help="Number of benchmark runs (default: 100)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare FP32 and quantized model outputs",
    )

    args = parser.parse_args()

    try:
        # Quantize the model
        result = quantize_model(
            model_path=args.model,
            output_path=args.output,
            quantization_type=args.type,
            benchmark=args.benchmark,
            benchmark_runs=args.benchmark_runs,
        )

        # Compare models if requested
        if args.compare:
            compare_result = compare_models(args.model, args.output)
            if compare_result:
                result["comparison"] = compare_result

        # Print summary
        print("\n" + "=" * 50)
        print("Quantization Summary")
        print("=" * 50)
        print(f"Input:  {result['input_path']}")
        print(f"Output: {result['output_path']}")
        print(f"Type:   {result['quantization_type']}")
        print(f"Size:   {result['fp32_size_mb']:.2f} MB -> {result['quantized_size_mb']:.2f} MB")
        print(f"Ratio:  {result['compression_ratio']:.2f}x")

        if "benchmark" in result:
            b = result["benchmark"]
            print(f"\nBenchmark:")
            print(f"  Latency: {b['avg_time_ms']:.3f} ms")
            print(f"  Throughput: {b['throughput_per_sec']:.1f} inf/s")

        if "comparison" in result:
            c = result["comparison"]
            print(f"\nAccuracy:")
            print(f"  Mean error: {c['mean_relative_error'] * 100:.4f}%")
            print(f"  Max error:  {c['max_relative_error'] * 100:.4f}%")

        print("=" * 50)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
