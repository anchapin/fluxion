#!/usr/bin/env python3
"""Simple example showing how to use the `Model` API.

Try running after `maturin develop` (see README). This example creates a
`Model`, runs a 1-year simulation with analytical loads and with surrogates,
and prints the resulting energy values.
"""

import time

try:
    import fluxion
except Exception as e:
    raise SystemExit(
        "Failed to import `fluxion`. Build & install the Python bindings first: `maturin develop`\n"
        + f"Original error: {e}"
    )


def main() -> None:
    print("Creating Model (this uses a simple default thermal model)...")

    # Model accepts a config path string but the current constructor ignores it.
    model = fluxion.Model("examples/simple_config.json")

    # Analytical (physics) simulation
    print("Running analytical simulation (1 year)...")
    t0 = time.time()
    e_analytical = model.simulate(1, False)
    t1 = time.time()
    print(f"Analytical EUI: {e_analytical:.4f} (elapsed {t1 - t0:.3f}s)")

    # Surrogate-enabled (fast) simulation
    print("Running surrogate-enabled simulation (1 year)...")
    t0 = time.time()
    # Generate and load dummy ONNX surrogate if present
    import os

    dummy_path = "examples/dummy_surrogate.onnx"
    if not os.path.exists(dummy_path):
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
        model.load_surrogate(dummy_path)
    except Exception as e:
        print(f"Warning: failed to load surrogate: {e}")

    e_surrogate = model.simulate(1, True)
    t1 = time.time()
    print(f"Surrogate EUI: {e_surrogate:.4f} (elapsed {t1 - t0:.3f}s)")


if __name__ == "__main__":
    main()
