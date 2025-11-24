#!/usr/bin/env python3
"""Example showing how to use `BatchOracle` to evaluate a small population.

This demonstrates the expected population format and prints a few results.
"""

import random
import time

try:
    import fluxion
except Exception as e:
    raise SystemExit(
        "Failed to import `fluxion`. Build & install the Python bindings first: "
        "`maturin develop`\n" + f"Original error: {e}"
    )


def make_population(n: int):
    # Window U-value range: 0.5 - 3.0
    # HVAC setpoint range: 19.0 - 24.0
    pop = []
    for _ in range(n):
        u = 0.5 + random.random() * (3.0 - 0.5)
        setpoint = 19.0 + random.random() * (24.0 - 19.0)
        pop.append([u, setpoint])
    return pop


def main() -> None:
    print("Creating BatchOracle...")
    oracle = fluxion.BatchOracle()

    # Ensure dummy surrogate ONNX exists and register it with the oracle
    import os

    dummy_path = "examples/dummy_surrogate.onnx"
    if not os.path.exists(dummy_path):
        print("Generating dummy ONNX surrogate...")
        import subprocess

        subprocess.check_call(
            [
                "python",
                "tools/generate_dummy_surrogate.py",
                "--zones",
                "10",
                "--out",
                dummy_path,
            ]
        )

    try:
        oracle.load_surrogate(dummy_path)
    except Exception as e:
        print(f"Warning: failed to load surrogate: {e}")

    pop = make_population(20)
    print("Evaluating population of 20 candidates (surrogates ON)...")
    t0 = time.time()
    results = oracle.evaluate_population(pop, True)
    t1 = time.time()

    print(f"Elapsed: {t1 - t0:.3f}s")
    best_idx = min(range(len(results)), key=lambda i: results[i])
    print(f"Best candidate index: {best_idx}, EUI: {results[best_idx]:.4f}")
    print("Sample results:")
    for i, (params, r) in enumerate(zip(pop[:5], results[:5])):
        print(f"  #{i}: U={params[0]:.3f}, setpoint={params[1]:.2f} -> EUI={r:.4f}")


if __name__ == "__main__":
    main()
