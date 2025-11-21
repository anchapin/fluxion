#!/usr/bin/env python3
"""Validation example: compare surrogate vs analytical predictions.

This script demonstrates how to:
1. Create a BatchOracle
2. Load a surrogate model (optional)
3. Evaluate the same population using both analytical and surrogate paths
4. Compare results and compute error metrics

The dummy surrogate model returns constant loads (1.2 W/m²), which should produce
results similar to analytical in certain scenarios. This is a simple validation
framework that can be extended for real trained models.
"""

import random
import time

try:
    import fluxion
except Exception as e:
    raise SystemExit(
        "Failed to import `fluxion`. Build & install the Python bindings first: `maturin develop`\n"
        + f"Original error: {e}"
    )


def make_population(n: int, seed: int = 42):
    """Generate n random candidates with reproducible randomness."""
    random.seed(seed)
    pop = []
    for _ in range(n):
        u = 0.5 + random.random() * (3.0 - 0.5)
        setpoint = 19.0 + random.random() * (24.0 - 19.0)
        pop.append([u, setpoint])
    return pop


def compute_error_metrics(analytical, surrogate):
    """Compute MAE, RMSE, and max error between two result sets."""
    import math

    mae = sum(abs(a - s) for a, s in zip(analytical, surrogate)) / len(analytical)
    rmse = math.sqrt(
        sum((a - s) ** 2 for a, s in zip(analytical, surrogate)) / len(analytical)
    )
    max_error = max(abs(a - s) for a, s in zip(analytical, surrogate))

    return {"mae": mae, "rmse": rmse, "max_error": max_error}


def main() -> None:
    print("=" * 70)
    print("Fluxion Surrogate Validation Example")
    print("=" * 70)

    # Create oracle
    print("\nCreating BatchOracle...")
    oracle = fluxion.BatchOracle()

    # Ensure dummy surrogate ONNX exists
    import os

    dummy_path = "assets/loads_predictor.onnx"
    if not os.path.exists(dummy_path):
        print(f"Warning: dummy surrogate not found at {dummy_path}")
        print("Using mock surrogates (both paths return 1.2 W/m²)")
        has_surrogate = False
    else:
        print(f"Dummy surrogate found at {dummy_path}")
        try:
            oracle.load_surrogate(dummy_path)
            has_surrogate = True
            print("✓ Surrogate model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load surrogate: {e}")
            print("  (libonnxruntime not installed; will use mock loads instead)")
            has_surrogate = False

    # Generate test population
    pop_size = 10
    pop = make_population(pop_size)
    print(f"\nGenerated population of {pop_size} candidates")
    print("Parameter ranges: U-value=[0.5-3.0], setpoint=[19-24]")

    # Evaluate using analytical (no AI)
    print("\n" + "-" * 70)
    print("Running ANALYTICAL evaluation (use_surrogates=False)...")
    t0 = time.time()
    analytical = oracle.evaluate_population(pop, use_surrogates=False)
    t_analytical = time.time() - t0
    print(f"  Time: {t_analytical:.4f}s")
    print(
        f"  Results: min={min(analytical):.4f}, max={max(analytical):.4f}, "
        f"mean={sum(analytical) / len(analytical):.4f}"
    )

    # Evaluate using surrogates (with AI)
    print("\n" + "-" * 70)
    print("Running SURROGATE evaluation (use_surrogates=True)...")
    t0 = time.time()
    surrogate = oracle.evaluate_population(pop, use_surrogates=True)
    t_surrogate = time.time() - t0
    print(f"  Time: {t_surrogate:.4f}s")
    print(
        f"  Results: min={min(surrogate):.4f}, max={max(surrogate):.4f}, "
        f"mean={sum(surrogate) / len(surrogate):.4f}"
    )

    # Compute speedup
    speedup = t_analytical / t_surrogate if t_surrogate > 0 else float("inf")
    print(f"\n  Speedup: {speedup:.2f}x (surrogate vs analytical)")

    # Compare results
    print("\n" + "-" * 70)
    print("VALIDATION COMPARISON")
    print("-" * 70)

    if has_surrogate:
        print("\n✓ Using real ONNX model (dummy constant predictor)")
        errors = compute_error_metrics(analytical, surrogate)
        print(f"  Mean Absolute Error:  {errors['mae']:.6f}")
        print(f"  Root Mean Squared Error: {errors['rmse']:.6f}")
        print(f"  Max Error:            {errors['max_error']:.6f}")

        print("\nDetailed comparison (first 5 candidates):")
        print("  Idx | Analytical  | Surrogate   | Difference")
        print("  " + "-" * 44)
        for i, (a, s) in enumerate(zip(analytical[:5], surrogate[:5])):
            diff = abs(a - s)
            print(f"  {i:3d} | {a:11.6f} | {s:11.6f} | {diff:10.6f}")
    else:
        print("\n⚠ Using mock surrogates (both return constant 1.2)")
        print("  Analytical results likely similar to surrogate results")
        print("  (This is expected; real ONNX model not available)")

        print("\nSample results:")
        for i, (a, s, params) in enumerate(zip(analytical[:5], surrogate[:5], pop[:5])):
            print(
                f"  {i}: U={params[0]:.3f}, setpoint={params[1]:.2f} -> "
                f"analytical={a:.4f}, surrogate={s:.4f}"
            )

    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
