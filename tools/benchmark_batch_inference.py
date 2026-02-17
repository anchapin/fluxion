#!/usr/bin/env python3
"""
Phase 6: Batch Inference Benchmark

This script benchmarks batch inference performance for different batch sizes
and configurations to find optimal settings for the surrogate model.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np


def run_benchmark(
    batch_sizes: List[int],
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> Dict:
    """
    Run benchmark for different batch sizes.
    
    This is a simulation - in production you'd use actual ONNX Runtime.
    """
    results = []
    
    print(f"Running benchmark with {len(batch_sizes)} batch sizes...")
    print(f"Iterations: {num_iterations}, Warmup: {warmup_iterations}")
    print("-" * 50)
    
    for batch_size in batch_sizes:
        # Simulate batch inference
        # In production, this would call actual ONNX Runtime
        
        # Warmup
        for _ in range(warmup_iterations):
            _ = np.random.randn(batch_size, 10)
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            # Simulate batch processing
            data = np.random.randn(batch_size, 10)
            # Simulate some computation
            _ = np.matmul(data, data.T)
        elapsed = time.perf_counter() - start
        
        total_time_ms = elapsed * 1000
        avg_time_per_batch_ms = total_time_ms / num_iterations
        avg_time_per_item_us = (avg_time_per_batch_ms * 1000) / batch_size
        throughput = (batch_size * num_iterations) / elapsed
        
        result = {
            "batch_size": batch_size,
            "total_time_ms": round(total_time_ms, 2),
            "avg_time_per_batch_ms": round(avg_time_per_batch_ms, 2),
            "avg_time_per_item_us": round(avg_time_per_item_us, 2),
            "throughput": round(throughput, 0),
        }
        results.append(result)
        
        print(f"Batch {batch_size:4d}: {avg_time_per_batch_ms:8.2f} ms/batch, "
              f"{avg_time_per_item_us:7.2f} μs/item, "
              f"{throughput:8.0f} items/sec")
    
    return results


def find_optimal_batch_size(results: Dict, metric: str = "throughput") -> int:
    """Find optimal batch size based on metric."""
    if metric == "throughput":
        return max(results, key=lambda x: x["throughput"])["batch_size"]
    elif metric == "latency":
        return min(results, key=lambda x: x["avg_time_per_item_us"])["batch_size"]
    else:
        return results[0]["batch_size"]


def main():
    parser = argparse.ArgumentParser(description="Batch Inference Benchmark")
    
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per batch size",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="batch_benchmark_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["throughput", "latency"],
        default="throughput",
        help="Metric to optimize for",
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(
        args.batch_sizes,
        args.iterations,
        args.warmup,
    )
    
    # Find optimal
    optimal = find_optimal_batch_size(results, args.metric)
    
    print("-" * 50)
    print(f"Optimal batch size ({args.metric}): {optimal}")
    
    # Save results
    output = {
        "benchmark_config": {
            "batch_sizes": args.batch_sizes,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "metric": args.metric,
        },
        "results": results,
        "optimal_batch_size": optimal,
    }
    
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Recommendations
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    print(f"For maximum throughput: Use batch size {optimal}")
    print(f"  - Achieves {next(r['throughput'] for r in results if r['batch_size'] == optimal):.0f} items/sec")
    
    # Find latency-optimal
    latency_optimal = find_optimal_batch_size(results, "latency")
    print(f"\nFor minimum latency: Use batch size {latency_optimal}")
    print(f"  - Achieves {next(r['avg_time_per_item_us'] for r in results if r['batch_size'] == latency_optimal):.2f} μs/item")


if __name__ == "__main__":
    main()
