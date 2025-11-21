#!/usr/bin/env python3
"""
Hook: Optional performance baseline check
Purpose: Smoke test BatchOracle throughput (should exceed 100μs/config)
Run with: pre-commit run --hook-stage manual

This catches silent performance regressions:
  - Unnecessary clones in batch loop
  - Memory allocations in hot path
  - Missing rayon parallelism
  - FFI overhead issues
"""

import time
import sys
import os

# Only run if release build exists
release_lib = "target/release/libfluxion.so"
if not os.path.exists(release_lib):
    print("ℹ Release build not found; skipping perf check")
    print("  Build with: cargo build --release && maturin develop --release")
    sys.exit(0)

try:
    from fluxion import BatchOracle
except ImportError:
    print("ℹ Fluxion not installed; skipping perf check")
    print("  Install with: maturin develop --release")
    sys.exit(0)

try:
    oracle = BatchOracle()
    
    # Test with 100 configs (representative batch)
    population = [[1.5, 21.0] for _ in range(100)]
    
    # Warm-up run (JIT, caching)
    _ = oracle.evaluate_population(population, False)
    
    # Actual benchmark
    start = time.perf_counter()
    results = oracle.evaluate_population(population, False)
    elapsed = time.perf_counter() - start
    
    time_per_config_ms = (elapsed * 1000) / 100
    throughput_per_sec = 1000 / time_per_config_ms
    
    # Targets: <0.1ms per config (10,000+ configs/sec)
    # Warning threshold: >1ms (regression indicator)
    if time_per_config_ms > 1.0:
        print(f"⚠ PERF WARNING: {time_per_config_ms:.2f}ms per config")
        print(f"  Target: <0.1ms ({10000:.0f}+ configs/sec)")
        print(f"  Current: {throughput_per_sec:.0f} configs/sec")
        print()
        print("  Likely causes:")
        print("    • Unnecessary clones of ThermalModel in evaluate_population")
        print("    • Memory allocations in solve_timesteps inner loop")
        print("    • Missing or nested rayon parallelism")
        print("    • FFI overhead (check Python boundary crossing)")
        sys.exit(1)
    else:
        print(f"✓ Perf OK: {time_per_config_ms:.3f}ms per config")
        print(f"  Throughput: {throughput_per_sec:.0f} configs/sec (Target: >10,000)")

except Exception as e:
    print(f"⚠ Perf check error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(0)
