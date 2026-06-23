#!/usr/bin/env python3
"""
Benchmark script comparing JSON/CSV serialization vs zero-copy state extraction.

This demonstrates the performance benefit of native zero-copy bindings for ML training.
"""

import time
import json
import csv
import io
import sys
import os

# Add the fluxion package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "target", "debug" if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "target", "debug")) else "release"))

try:
    import fluxion
    from fluxion.multi_zone import MultiZoneThermalModel
    HAS_FLUXION = True
except ImportError as e:
    print(f"Warning: fluxion not available ({e})")
    HAS_FLUXION = False


def benchmark_json_serialization(model, timesteps):
    """Simulate JSON serialization overhead (traditional approach)."""
    start = time.perf_counter()

    # Get hourly temperatures as Python data
    temps = model.get_hourly_temperatures()

    # Simulate JSON serialization overhead
    json_bytes = json.dumps(temps).encode('utf-8')
    deserialized = json.loads(json_bytes.decode('utf-8'))

    elapsed = time.perf_counter() - start
    size_kb = len(json_bytes) / 1024

    return elapsed, size_kb


def benchmark_csv_serialization(model, timesteps):
    """Simulate CSV serialization overhead."""
    start = time.perf_counter()

    temps = model.get_hourly_temperatures()

    # Simulate CSV serialization
    output = io.StringIO()
    writer = csv.writer(output)
    for zone_temps in temps:
        writer.writerow(zone_temps)

    csv_bytes = output.getvalue().encode('utf-8')

    elapsed = time.perf_counter() - start
    size_kb = len(csv_bytes) / 1024

    return elapsed, size_kb


def benchmark_zero_copy(model, timesteps):
    """Zero-copy numpy extraction (our approach)."""
    start = time.perf_counter()

    # Get hourly temperatures as numpy array (zero-copy when possible)
    temps, shape = model.get_hourly_temperatures_numpy()

    elapsed = time.perf_counter() - start
    size_kb = temps.nbytes / 1024

    return elapsed, size_kb, shape


def run_benchmark(num_zones=3, years=1):
    """Run full benchmark suite."""
    if not HAS_FLUXION:
        print("Fluxion Python bindings not available - benchmark skipped")
        print("Build with: cargo build --features python-bindings")
        return

    print(f"Benchmark: {num_zones} zones, {years} year(s)")
    print("=" * 60)

    # Create and simulate model
    model = MultiZoneThermalModel(num_zones)
    print(f"Running simulation ({years * 8760} timesteps)...")
    eui = model.simulate_multi_zone(years, False)
    print(f"Simulation complete, EUI = {eui:.2f} kWh/m²/year")
    print()

    timesteps = years * 8760

    # JSON benchmark
    json_time, json_size = benchmark_json_serialization(model, timesteps)
    print(f"JSON serialization:")
    print(f"  Time:  {json_time * 1000:.2f} ms")
    print(f"  Size:  {json_size:.1f} KB")
    print()

    # CSV benchmark
    csv_time, csv_size = benchmark_csv_serialization(model, timesteps)
    print(f"CSV serialization:")
    print(f"  Time:  {csv_time * 1000:.2f} ms")
    print(f"  Size:  {csv_size:.1f} KB")
    print()

    # Zero-copy benchmark
    zc_time, zc_size, shape = benchmark_zero_copy(model, timesteps)
    print(f"Zero-copy numpy:")
    print(f"  Time:  {zc_time * 1000:.2f} ms")
    print(f"  Size:  {zc_size:.1f} KB")
    print(f"  Shape: {shape}")
    print()

    # Speedup calculations
    print("Speedup vs traditional methods:")
    print(f"  vs JSON: {json_time / zc_time:.1f}x faster")
    print(f"  vs CSV:  {csv_time / zc_time:.1f}x faster")
    print()

    # Memory savings
    print("Memory efficiency:")
    print(f"  JSON: {json_size / zc_size:.1f}x larger than native")
    print(f"  CSV:  {csv_size / zc_size:.1f}x larger than native")


if __name__ == "__main__":
    run_benchmark(num_zones=3, years=1)
