#!/usr/bin/env python3
"""Population evaluation example using the real `BatchOracle` API.

`BatchOracle.evaluate_population` expects a `List[List[float]]` of
design candidates, each with **three** elements (window U-value,
heating setpoint, cooling setpoint) — see `src/lib.rs:1015` and
`BatchOracle::validate_parameters`. The previous two-element
`[u, setpoint]` form would fail validation with
"Cooling setpoint (index 2) … out of range".

`BatchOracle` does **not** expose a `load_surrogate` method (only
`fluxion.Model` does). The oracle always uses its internal
`SurrogateManager`, which falls back to deterministic mock loads
when no ONNX is loaded. We therefore do not call `load_surrogate`
here — the legacy call in earlier revisions of this example was
removed in #1411.
"""

from __future__ import annotations

import random
import time

try:
    import fluxion
except Exception as e:  # noqa: BLE001 — surface any import error verbatim
    raise SystemExit(
        "Failed to import `fluxion`. Build & install the Python bindings first: "
        "`maturin develop --release`\n"
        f"Original error: {e}"
    )


def make_population(n: int) -> list[list[float]]:
    # Parameter ranges (see `BatchOracle::MIN_*` / `MAX_*` in `src/lib.rs:997-1003`):
    #   window_u_value    : 0.1 – 5.0   W/m²K
    #   heating_setpoint  : 15.0 – 25.0 °C
    #   cooling_setpoint  : 22.0 – 32.0 °C  (must be > heating_setpoint)
    pop: list[list[float]] = []
    for _ in range(n):
        u = 0.5 + random.random() * (3.0 - 0.5)
        heating = 19.0 + random.random() * (22.0 - 19.0)
        cooling = heating + 2.0 + random.random() * (28.0 - (heating + 2.0))
        pop.append([u, heating, cooling])
    return pop


def main() -> None:
    print("Creating BatchOracle (uses its internal SurrogateManager)…")
    oracle = fluxion.BatchOracle()

    pop = make_population(20)
    print("Evaluating population of 20 candidates (analytical, no surrogate)…")
    t0 = time.time()
    results = oracle.evaluate_population(pop, False)
    t1 = time.time()

    print(f"Elapsed: {t1 - t0:.3f}s")
    best_idx = min(range(len(results)), key=lambda i: results[i])
    print(f"Best candidate index: {best_idx}, EUI: {results[best_idx]:.4f}")
    print("Sample results:")
    for i, (params, r) in enumerate(zip(pop[:5], results[:5])):
        print(
            f"  #{i}: U={params[0]:.3f}, "
            f"heating={params[1]:.2f}, cooling={params[2]:.2f} -> EUI={r:.4f}"
        )


if __name__ == "__main__":
    main()
