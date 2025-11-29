import argparse
import sys
import os

# Add tools directory to path to allow importing benchmark_throughput
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark_throughput import benchmark_throughput

def benchmark_gpu_crossover(model_path: str, output_file: str, batch_sizes: list[int] = None):
    """
    Run benchmarks to find CPU/GPU crossover point.
    """
    if batch_sizes is None:
        # Batch sizes focusing on finding the crossover (typically 100-10000 range)
        # The issue specifically asks to benchmark 100 vs 10000, but we want to find the crossover.
        batch_sizes = [1, 10, 50, 100, 250, 500, 750, 1000, 2500, 5000, 10000, 20000]

    backends = ["cpu", "cuda"]

    print("=== GPU Throughput Crossover Benchmark ===")
    print(f"Model: {model_path}")
    print(f"Backends: {backends}")
    print(f"Batch sizes: {batch_sizes}")

    results = benchmark_throughput(model_path, batch_sizes, backends, output_file)

    if not results or "benchmarks" not in results:
        print("No results returned.")
        return

    # Analyze results
    cpu_results = {b['batch_size']: b['throughput'] for b in results['benchmarks'] if b['backend'] == 'cpu'}
    cuda_results = {b['batch_size']: b['throughput'] for b in results['benchmarks'] if b['backend'] == 'cuda'}

    if not cuda_results:
        print("\nWARNING: No CUDA results found. Is CUDA available?")
        return

    print("\n=== Crossover Analysis ===")
    crossover_found = False

    sorted_batches = sorted(batch_sizes)

    print(f"{'Batch Size':<12} | {'CPU (samples/s)':<18} | {'GPU (samples/s)':<18} | {'Speedup':<10}")
    print("-" * 65)

    for bs in sorted_batches:
        cpu_tp = cpu_results.get(bs, 0)
        gpu_tp = cuda_results.get(bs, 0)

        if cpu_tp > 0:
            speedup = gpu_tp / cpu_tp
        else:
            speedup = 0.0

        print(f"{bs:<12} | {cpu_tp:<18.2f} | {gpu_tp:<18.2f} | {speedup:<10.2f}x")

        if not crossover_found and speedup > 1.0:
            print(f"\n[!] Crossover point found around batch size {bs} (Speedup: {speedup:.2f}x)")
            crossover_found = True

    if not crossover_found:
        print("\n[!] No crossover point found (CPU might be faster for all tested batch sizes).")
    elif crossover_found:
        # Find where speedup exceeds 10x
        for bs in sorted_batches:
             cpu_tp = cpu_results.get(bs, 0)
             gpu_tp = cuda_results.get(bs, 0)
             if cpu_tp > 0 and (gpu_tp / cpu_tp) > 10.0:
                 print(f"[!] >10x Speedup achieved at batch size {bs}")
                 break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Crossover Benchmark")
    parser.add_argument("--model", type=str, default="assets/loads_predictor.onnx", help="Path to ONNX model")
    parser.add_argument("--output", type=str, default="gpu_benchmark_results.json", help="Output JSON file")
    parser.add_argument("--batch-sizes", type=str, help="Comma-separated batch sizes (optional)")

    args = parser.parse_args()

    batch_sizes = None
    if args.batch_sizes:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    benchmark_gpu_crossover(args.model, args.output, batch_sizes)
