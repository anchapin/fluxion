#!/usr/bin/env python3
"""Generate `tests/surrogate_models/golden/golden_v3_1_0.json`.

This script reproduces the deterministic analytical fallback from
`src/ai/surrogate.rs::SurrogateManager::deterministic_analytical_loads`:

    output[i] = max(0.0, 50.0 * sin(pi * (exterior_temp - 6.0) / 12.0))

Use it to re-baseline the golden file after an intentional change to the
deterministic helper. See ADR-0004 (docs/adr/0004-onnx-model-versioning.md)
for the re-baseline policy.

Usage:
    python3 scripts/gen_golden_outputs.py > tests/surrogate_models/golden/golden_v3_1_0.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "tests" / "surrogate_models" / "golden" / "golden_v3_1_0.json"


def build_inputs(n: int = 100) -> list[dict]:
    """100 fixed inputs spread across the residential domain."""
    out: list[dict] = []
    for i in range(n):
        out.append(
            {
                "exterior_temp": -10.0 + (40.0 * i / (n - 1)),
                "zone_temp": 18.0 + (8.0 * (i % 7) / 6.0),
                "solar_rad": 0.0 + (1000.0 * ((i * 17) % 100) / 100.0),
                "humidity": 20.0 + (60.0 * ((i * 31) % 100) / 100.0),
                "occupancy": (i % 11) / 10.0,
                "climate_zone": ["4A", "5A", "3A", "2A", "6A"][i % 5],
            }
        )
    return out


def compute_outputs(inputs: list[dict]) -> list[float]:
    """Mirror of `deterministic_analytical_loads`."""
    return [
        max(0.0, 50.0 * math.sin(math.pi * (inp["exterior_temp"] - 6.0) / 12.0))
        for inp in inputs
    ]


def main() -> int:
    inputs = build_inputs(100)
    outputs = compute_outputs(inputs)
    golden = {
        "_meta": {
            "schema": "fluxion-golden-output/v1",
            "version": "3.1.0",
            "description": (
                "100 fixed inputs through "
                "SurrogateManager::deterministic_analytical_loads. "
                "Regenerate with: python3 scripts/gen_golden_outputs.py "
                "(see ADR-0004)."
            ),
            "tolerance_rel": 1e-6,
            "tolerance_abs": 1e-9,
        },
        "inputs": inputs,
        "outputs": outputs,
    }
    json_str = json.dumps(golden, indent=2, sort_keys=False)

    if "--write" in sys.argv:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_text(json_str + "\n")
        print(f"wrote {TARGET}", file=sys.stderr)
        return 0

    print(json_str)
    return 0


if __name__ == "__main__":
    sys.exit(main())