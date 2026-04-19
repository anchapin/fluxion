"""
Performance Metrics Module

This module provides comprehensive performance metrics and profiling
capabilities for large batch operations in building energy simulation
and optimization workflows.

Features:
- Detailed timing metrics for batch operations
- Memory usage profiling
- Parallelization efficiency analysis
- Batch operation statistics
- Performance reporting and visualization

Usage:
    from tools.performance_metrics import PerformanceProfiler

    # Create profiler
    profiler = PerformanceProfiler()

    # Profile a batch operation
    with profiler.profile_batch("simulation_batch", batch_size=100):
        for i in range(100):
            run_simulation(case_i)

    # Get performance report
    report = profiler.generate_report()
    print(report)
"""

import json
import logging
import os
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatchOperationMetrics:
    """Performance metrics for a single batch operation."""

    name: str
    batch_size: int
    start_time: float
    end_time: float
    peak_memory_mb: float
    cpu_usage_percent: float

    @property
    def duration_seconds(self) -> float:
        """Duration of the operation in seconds."""
        return self.end_time - self.start_time

    @property
    def operations_per_second(self) -> float:
        """Operations per second."""
        return (
            self.batch_size / self.duration_seconds
            if self.duration_seconds > 0
            else 0.0
        )

    @property
    def seconds_per_operation(self) -> float:
        """Seconds per operation."""
        return self.duration_seconds / self.batch_size if self.batch_size > 0 else 0.0

    @property
    def memory_per_operation_mb(self) -> float:
        """Memory usage per operation in MB."""
        return self.peak_memory_mb / self.batch_size if self.batch_size > 0 else 0.0


@dataclass
class ParallelOperationMetrics:
    """Performance metrics for parallel operations."""

    name: str
    sequential_time: float
    parallel_time: float
    worker_count: int
    batch_size: int

    @property
    def speedup(self) -> float:
        """Speedup factor from parallelization."""
        return (
            self.sequential_time / self.parallel_time if self.parallel_time > 0 else 0.0
        )

    @property
    def efficiency(self) -> float:
        """Parallelization efficiency (0-1)."""
        return self.speedup / self.worker_count if self.worker_count > 0 else 0.0

    @property
    def overhead_percent(self) -> float:
        """Parallelization overhead percentage."""
        ideal_time = self.sequential_time / self.worker_count
        overhead = (
            ((self.parallel_time - ideal_time) / ideal_time) * 100
            if ideal_time > 0
            else 0.0
        )
        return overhead


class PerformanceProfiler:
    """
    Performance profiler for batch operations.

    Tracks timing, memory usage, and other performance metrics
    for batch operations in building energy simulation workflows.
    """

    def __init__(self):
        """Initialize performance profiler."""
        self.batch_operations: List[BatchOperationMetrics] = []
        self.parallel_operations: List[ParallelOperationMetrics] = []
        self.memory_snapshots: List[Tuple[float, float]] = []  # (timestamp, memory_mb)
        self.custom_metrics: Dict[str, List[float]] = defaultdict(list)
        self.start_time = time.time()

        # System information
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict:
        """Get system information."""
        return {
            "cpu_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "os": os.name,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        }

    @contextmanager
    def profile_batch(self, name: str, batch_size: int):
        """
        Context manager for profiling batch operations.

        Args:
            name: Name of the batch operation
            batch_size: Number of operations in the batch

        Yields:
            None (context manager)
        """
        start_time = time.time()
        start_memory = self._get_current_memory_mb()
        start_cpu = psutil.cpu_percent(interval=None)

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_current_memory_mb()
            end_cpu = psutil.cpu_percent(interval=None)

            # Record peak memory during operation
            peak_memory = max(start_memory, end_memory)

            # Create metrics object
            metrics = BatchOperationMetrics(
                name=name,
                batch_size=batch_size,
                start_time=start_time,
                end_time=end_time,
                peak_memory_mb=peak_memory,
                cpu_usage_percent=end_cpu,
            )

            self.batch_operations.append(metrics)
            self._record_memory_snapshot()

    @contextmanager
    def profile_parallel(self, name: str, worker_count: int, batch_size: int):
        """
        Context manager for profiling parallel operations.

        Args:
            name: Name of the parallel operation
            worker_count: Number of parallel workers
            batch_size: Total batch size

        Yields:
            Tuple of (sequential_context, parallel_context) context managers
        """
        sequential_metrics = None
        parallel_metrics = None

        class SequentialContext:
            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.end_time = time.time()
                nonlocal sequential_metrics
                sequential_metrics = (self.start_time, self.end_time)

        class ParallelContext:
            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.end_time = time.time()
                nonlocal parallel_metrics
                parallel_metrics = (self.start_time, self.end_time)

        try:
            yield SequentialContext(), ParallelContext()
        finally:
            if sequential_metrics and parallel_metrics:
                seq_start, seq_end = sequential_metrics
                par_start, par_end = parallel_metrics

                metrics = ParallelOperationMetrics(
                    name=name,
                    sequential_time=seq_end - seq_start,
                    parallel_time=par_end - par_start,
                    worker_count=worker_count,
                    batch_size=batch_size,
                )

                self.parallel_operations.append(metrics)

    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def _record_memory_snapshot(self):
        """Record current memory usage snapshot."""
        timestamp = time.time() - self.start_time
        memory_mb = self._get_current_memory_mb()
        self.memory_snapshots.append((timestamp, memory_mb))

    def record_custom_metric(self, name: str, value: float):
        """Record a custom performance metric."""
        self.custom_metrics[name].append(value)

    def get_batch_statistics(self) -> Dict:
        """Get statistics for all batch operations."""
        if not self.batch_operations:
            return {}

        durations = [op.duration_seconds for op in self.batch_operations]
        ops_per_sec = [op.operations_per_second for op in self.batch_operations]
        sec_per_op = [op.seconds_per_operation for op in self.batch_operations]
        memory_per_op = [op.memory_per_operation_mb for op in self.batch_operations]

        return {
            "total_operations": sum(op.batch_size for op in self.batch_operations),
            "total_time_seconds": sum(
                op.duration_seconds for op in self.batch_operations
            ),
            "average_duration_seconds": statistics.mean(durations),
            "min_duration_seconds": min(durations),
            "max_duration_seconds": max(durations),
            "std_duration_seconds": (
                statistics.stdev(durations) if len(durations) > 1 else 0.0
            ),
            "average_ops_per_second": statistics.mean(ops_per_sec),
            "average_seconds_per_op": statistics.mean(sec_per_op),
            "average_memory_per_op_mb": statistics.mean(memory_per_op),
            "peak_memory_mb": max(op.peak_memory_mb for op in self.batch_operations),
        }

    def get_parallel_statistics(self) -> Dict:
        """Get statistics for parallel operations."""
        if not self.parallel_operations:
            return {}

        speedups = [op.speedup for op in self.parallel_operations]
        efficiencies = [op.efficiency for op in self.parallel_operations]
        overheads = [op.overhead_percent for op in self.parallel_operations]

        return {
            "average_speedup": statistics.mean(speedups),
            "min_speedup": min(speedups),
            "max_speedup": max(speedups),
            "average_efficiency": statistics.mean(efficiencies),
            "min_efficiency": min(efficiencies),
            "max_efficiency": max(efficiencies),
            "average_overhead_percent": statistics.mean(overheads),
            "min_overhead_percent": min(overheads),
            "max_overhead_percent": max(overheads),
        }

    def generate_report(self, format: str = "text") -> str:
        """Generate performance report."""
        if format.lower() == "json":
            return self._generate_json_report()
        else:
            return self._generate_text_report()

    def _generate_text_report(self) -> str:
        """Generate text performance report."""
        report = [
            "=" * 70,
            "PERFORMANCE PROFILING REPORT",
            "=" * 70,
            "",
            "SYSTEM INFORMATION:",
            "-" * 50,
        ]

        # System info
        for key, value in self.system_info.items():
            report.append(f"  {key:15}: {value}")

        # Batch operations summary
        batch_stats = self.get_batch_statistics()
        if batch_stats:
            report.extend(["", "BATCH OPERATIONS SUMMARY:", "-" * 50])

            report.append(f"  Total Operations: {batch_stats['total_operations']:,}")
            report.append(
                f"  Total Time: {batch_stats['total_time_seconds']:.2f} seconds"
            )
            report.append(
                f"  Average Duration: {batch_stats['average_duration_seconds']:.3f} ± {batch_stats['std_duration_seconds']:.3f} seconds"
            )
            report.append(
                f"  Throughput: {batch_stats['average_ops_per_second']:.2f} operations/second"
            )
            report.append(
                f"  Latency: {batch_stats['average_seconds_per_op']:.6f} seconds/operation"
            )
            report.append(
                f"  Memory Usage: {batch_stats['average_memory_per_op_mb']:.3f} MB/operation"
            )
            report.append(f"  Peak Memory: {batch_stats['peak_memory_mb']:.1f} MB")

        # Parallel operations summary
        parallel_stats = self.get_parallel_statistics()
        if parallel_stats:
            report.extend(["", "PARALLEL OPERATIONS SUMMARY:", "-" * 50])

            report.append(
                f"  Average Speedup: {parallel_stats['average_speedup']:.2f}x"
            )
            report.append(
                f"  Parallel Efficiency: {parallel_stats['average_efficiency']:.2%}"
            )
            report.append(
                f"  Average Overhead: {parallel_stats['average_overhead_percent']:.1f}%"
            )
            report.append(
                f"  Speedup Range: {parallel_stats['min_speedup']:.1f}x - {parallel_stats['max_speedup']:.1f}x"
            )
            report.append(
                f"  Efficiency Range: {parallel_stats['min_efficiency']:.2%} - {parallel_stats['max_efficiency']:.2%}"
            )

        # Detailed batch operations
        if self.batch_operations:
            report.extend(["", "DETAILED BATCH OPERATIONS:", "-" * 50])
            report.append(
                f"{'Name':<20} {'Size':>8} {'Time':>10} {'Ops/sec':>12} {'Sec/op':>12} {'Mem/op':>12}"
            )
            report.append("-" * 70)

            for op in self.batch_operations:
                report.append(
                    f"{op.name:<20} {op.batch_size:>8,} {op.duration_seconds:>10.3f} "
                    f"{op.operations_per_second:>12.2f} {op.seconds_per_operation:>12.6f} "
                    f"{op.memory_per_operation_mb:>12.3f}"
                )

        # Detailed parallel operations
        if self.parallel_operations:
            report.extend(["", "DETAILED PARALLEL OPERATIONS:", "-" * 50])
            report.append(
                f"{'Name':<20} {'Workers':>8} {'Speedup':>10} {'Eff.':>8} {'Overhead':>10}"
            )
            report.append("-" * 70)

            for op in self.parallel_operations:
                report.append(
                    f"{op.name:<20} {op.worker_count:>8} {op.speedup:>10.2f}x "
                    f"{op.efficiency:>8.2%} {op.overhead_percent:>10.1f}%"
                )

        # Custom metrics
        if self.custom_metrics:
            report.extend(["", "CUSTOM METRICS:", "-" * 50])
            for metric_name, values in self.custom_metrics.items():
                report.append(f"  {metric_name}:")
                report.append(f"    Count: {len(values)}")
                report.append(f"    Average: {statistics.mean(values):.4f}")
                report.append(f"    Min: {min(values):.4f}")
                report.append(f"    Max: {max(values):.4f}")
                if len(values) > 1:
                    report.append(f"    Std Dev: {statistics.stdev(values):.4f}")

        # Overall summary
        total_time = time.time() - self.start_time
        report.extend(["", "OVERALL SUMMARY:", "-" * 50])
        report.append(f"  Total Profiling Time: {total_time:.2f} seconds")
        report.append(f"  Batch Operations: {len(self.batch_operations)}")
        report.append(f"  Parallel Operations: {len(self.parallel_operations)}")
        report.append(f"  Memory Snapshots: {len(self.memory_snapshots)}")

        report.append("\n" + "=" * 70)
        return "\n".join(report)

    def _generate_json_report(self) -> str:
        """Generate JSON performance report."""
        report_data = {
            "system_info": self.system_info,
            "batch_operations": [
                {
                    "name": op.name,
                    "batch_size": op.batch_size,
                    "duration_seconds": op.duration_seconds,
                    "operations_per_second": op.operations_per_second,
                    "seconds_per_operation": op.seconds_per_operation,
                    "peak_memory_mb": op.peak_memory_mb,
                    "cpu_usage_percent": op.cpu_usage_percent,
                    "memory_per_operation_mb": op.memory_per_operation_mb,
                }
                for op in self.batch_operations
            ],
            "parallel_operations": [
                {
                    "name": op.name,
                    "worker_count": op.worker_count,
                    "batch_size": op.batch_size,
                    "sequential_time": op.sequential_time,
                    "parallel_time": op.parallel_time,
                    "speedup": op.speedup,
                    "efficiency": op.efficiency,
                    "overhead_percent": op.overhead_percent,
                }
                for op in self.parallel_operations
            ],
            "batch_statistics": self.get_batch_statistics(),
            "parallel_statistics": self.get_parallel_statistics(),
            "custom_metrics": {k: v for k, v in self.custom_metrics.items()},
            "memory_snapshots": self.memory_snapshots,
            "total_profiling_time_seconds": time.time() - self.start_time,
        }

        return json.dumps(report_data, indent=2)

    def save_report_to_file(self, filename: str, format: str = "text"):
        """Save performance report to file."""
        report = self.generate_report(format)

        with open(filename, "w") as f:
            f.write(report)

        logger.info(f"Performance report saved to {filename}")

    def reset(self):
        """Reset all collected metrics."""
        self.batch_operations.clear()
        self.parallel_operations.clear()
        self.memory_snapshots.clear()
        self.custom_metrics.clear()
        self.start_time = time.time()


def create_batch_performance_benchmark() -> Callable:
    """Create a benchmark function for testing batch performance."""

    def mock_simulation(case_data):
        """Mock simulation function with variable computation time."""
        # Simulate computation time based on case complexity
        complexity = case_data.get("complexity", 1.0)
        time.sleep(0.01 * complexity)

        # Simulate memory usage
        dummy_data = [0] * int(1000 * complexity)

        return {
            "energy": 100 * complexity,
            "comfort": 90 - complexity,
            "computation_time": 0.01 * complexity,
        }

    def benchmark_batch_operation(batch_size: int, complexity: float = 1.0):
        """
        Benchmark function for batch operations.

        Args:
            batch_size: Number of simulations to run
            complexity: Complexity factor (1.0 = normal, 2.0 = 2x more complex)

        Returns:
            Dictionary with performance metrics
        """
        profiler = PerformanceProfiler()

        with profiler.profile_batch(f"simulation_batch_{complexity}", batch_size):
            results = []
            for i in range(batch_size):
                case_data = {"id": i, "complexity": complexity}
                result = mock_simulation(case_data)
                results.append(result)

                # Record custom metric
                profiler.record_custom_metric("simulation_energy", result["energy"])

        # Get statistics
        stats = profiler.get_batch_statistics()

        return {
            "batch_size": batch_size,
            "complexity": complexity,
            "total_time": stats["total_time_seconds"],
            "throughput": stats["average_ops_per_second"],
            "latency": stats["average_seconds_per_op"],
            "memory_usage": stats["average_memory_per_op_mb"],
            "results": results,
        }

    return benchmark_batch_operation


def analyze_parallel_scaling() -> Dict:
    """Analyze parallel scaling efficiency."""

    def mock_parallel_operation(item, complexity: float = 1.0):
        """Mock operation for parallel testing."""
        time.sleep(0.005 * complexity)
        return {"result": item * complexity}

    def run_scaling_analysis(max_workers: int = 8, batch_size: int = 100):
        """Run parallel scaling analysis."""
        profiler = PerformanceProfiler()

        # Sequential baseline
        sequential_results = []
        with profiler.profile_batch("sequential_baseline", batch_size) as seq_ctx:
            for i in range(batch_size):
                result = mock_parallel_operation(i, complexity=1.0)
                sequential_results.append(result)

        # Parallel runs with different worker counts
        for worker_count in range(1, max_workers + 1):
            parallel_results = []

            # Simulate parallel execution
            start_time = time.time()

            # In real implementation, this would use ThreadPoolExecutor
            # For testing, we simulate parallel speedup
            parallel_time = (
                sequential_results[0]["computation_time"] * batch_size / worker_count
            )
            time.sleep(parallel_time)

            for i in range(batch_size):
                result = mock_parallel_operation(i, complexity=1.0)
                parallel_results.append(result)

            end_time = time.time()

            # Record parallel metrics
            seq_start, seq_end = seq_ctx.start_time, seq_ctx.end_time
            par_start, par_end = start_time, end_time

            metrics = ParallelOperationMetrics(
                name=f"parallel_{worker_count}_workers",
                sequential_time=seq_end - seq_start,
                parallel_time=par_end - par_start,
                worker_count=worker_count,
                batch_size=batch_size,
            )

            profiler.parallel_operations.append(metrics)

        # Get parallel statistics
        parallel_stats = profiler.get_parallel_statistics()

        return {
            "sequential_time": sequential_results[0]["computation_time"] * batch_size,
            "parallel_stats": parallel_stats,
            "worker_counts": list(range(1, max_workers + 1)),
            "scaling_curve": [
                {
                    "workers": w,
                    "speedup": profiler.parallel_operations[w - 1].speedup,
                    "efficiency": profiler.parallel_operations[w - 1].efficiency,
                }
                for w in range(1, max_workers + 1)
            ],
        }

    return run_scaling_analysis


if __name__ == "__main__":
    # Demonstration of performance profiling
    print("Performance Profiling Demo")
    print("=" * 50)

    # Create profiler
    profiler = PerformanceProfiler()

    # Profile a batch operation
    print("Profiling batch operation...")
    with profiler.profile_batch("test_simulations", batch_size=25):
        # Simulate 25 building energy simulations
        for i in range(25):
            # Simulate computation
            time.sleep(0.01)

            # Record custom metric
            profiler.record_custom_metric("simulation_energy", 100 + i * 2)

    # Profile another batch
    with profiler.profile_batch("validation_checks", batch_size=50):
        for i in range(50):
            time.sleep(0.005)

    # Generate and display report
    report = profiler.generate_report()
    print("\nPerformance Report:")
    print(report)

    # Test JSON report
    json_report = profiler.generate_report("json")
    print("\nJSON Report (first 500 chars):")
    print(json_report[:500] + "...")

    # Test benchmark function
    print("\nRunning benchmark...")
    benchmark = create_batch_performance_benchmark()

    # Test with different complexities
    for complexity in [1.0, 1.5, 2.0]:
        result = benchmark(batch_size=10, complexity=complexity)
        print(
            f"Complexity {complexity}: {result['throughput']:.1f} ops/sec, "
            f"{result['latency']:.4f} sec/op"
        )

    # Test parallel scaling analysis
    print("\nParallel scaling analysis...")
    scaling_analysis = analyze_parallel_scaling()
    scaling_result = scaling_analysis(max_workers=4, batch_size=10)

    print(f"Sequential time: {scaling_result['sequential_time']:.3f}s")
    print(
        f"Average speedup: {scaling_result['parallel_stats']['average_speedup']:.2f}x"
    )
    print(
        f"Average efficiency: {scaling_result['parallel_stats']['average_efficiency']:.2%}"
    )

    print("\nScaling curve:")
    for point in scaling_result["scaling_curve"]:
        print(
            f"  {point['workers']} workers: {point['speedup']:.2f}x speedup, "
            f"{point['efficiency']:.2%} efficiency"
        )
