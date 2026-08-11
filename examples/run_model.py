#!/usr/bin/env python3
"""Single-zone simulation example using the real `Model` API.

This script:

1. Creates a 1-zone `fluxion.Model` (the constructor only takes a
   zone count — see `src/lib.rs:189`).
2. Runs an analytical 1-year simulation and prints the EUI.
3. Optionally loads a pre-trained ONNX surrogate and re-runs the
   simulation for comparison. The surrogate path is skipped silently
   if the file is missing or the `ort` runtime is unavailable, so the
   script still succeeds on a fresh checkout.

The `eui` value is a raw cumulative temperature-departure metric —
a relative energy-cost objective, not a calibrated `kWh/m²/year`
value (see the "Production scope" section in `docs/QUICKSTART.md`
and issues #749 / #767).
"""

from __future__ import annotations

import os
import time

try:
    import fluxion
except Exception as e:  # noqa: BLE001 — surface any import error verbatim
    raise SystemExit(
        "Failed to import `fluxion`. Build & install the Python bindings first: "
        "`maturin develop --release`\n"
        f"Original error: {e}"
    )


def main() -> None:
    print("Creating Model (1 zone, default setpoints)…")
    model = fluxion.Model(num_zones=1)

    print("Running analytical simulation (1 year)…")
    t0 = time.time()
    e_analytical = model.simulate(1, False)
    t1 = time.time()
    print(f"Analytical EUI: {e_analytical:.4f} (elapsed {t1 - t0:.3f}s)")

    surrogate_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dummy_surrogate.onnx"
    )
    if not os.path.exists(surrogate_path):
        print(
            f"No surrogate found at {surrogate_path} — skipping surrogate run. "
            "Generate one with `python tools/generate_dummy_surrogate.py "
            "--zones 1 --out examples/dummy_surrogate.onnx`."
        )
        return

    print(f"Loading ONNX surrogate from {surrogate_path}…")
    try:
        model.load_surrogate(surrogate_path)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: failed to load surrogate: {e}")
        return

    print("Running surrogate-enabled simulation (1 year)…")
    t0 = time.time()
    e_surrogate = model.simulate(1, True)
    t1 = time.time()
    print(f"Surrogate EUI: {e_surrogate:.4f} (elapsed {t1 - t0:.3f}s)")


if __name__ == "__main__":
    main()
