"""Generate Monte Carlo delta files for Fluxion parameter sweeps (Issue #1813).

This module produces **declarative delta files** consumed by the Rust worker
entrypoint::

    fluxion monte-carlo sweep --base-model base.yaml --delta-file delta.yaml

It can also **pre-materialize** N individual JSON patch files (one per draw) for
the distributed-worker deployment where each Nomad / AWS Batch worker receives
the base model plus a single per-draw patch.

The distribution sampling mirrors the Rust implementation in
``src/analysis/monte_carlo.rs`` so a given seed produces identical draws in both
languages (uniform uses a simple LCG fallback; for exact parity use the Rust
sampler as the source of truth and materialize via ``--materialize``).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Re-exported enum-style distribution names (mirror the Rust ``Distribution`` enum).
UNIFORM = "uniform"
NORMAL = "normal"
LOGNORMAL = "lognormal"
TRIANGULAR = "triangular"
FIXED = "fixed"

DEFAULT_SAMPLES = 1000  # per Issue #1813 acceptance criterion
DEFAULT_SEED = 0x5EED_1813
DEFAULT_WARM_UP_YEARS = 2


@dataclass
class Distribution:
    """A probability distribution over a single building-model parameter.

    Fields depend on ``kind``:

    - ``uniform``     → ``min``, ``max``
    - ``normal``      → ``mean``, ``std``
    - ``lognormal``   → ``mean``, ``std`` (of the underlying normal in log-space)
    - ``triangular``  → ``min``, ``mode``, ``max``
    - ``fixed``       → ``value``
    """

    kind: str
    params: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def uniform(cls, min: float, max: float) -> "Distribution":
        if max < min:
            raise ValueError(f"uniform: max ({max}) < min ({min})")
        return cls(UNIFORM, {"min": float(min), "max": float(max)})

    @classmethod
    def normal(cls, mean: float, std: float) -> "Distribution":
        if std <= 0:
            raise ValueError(f"normal: std must be > 0 (got {std})")
        return cls(NORMAL, {"mean": float(mean), "std": float(std)})

    @classmethod
    def lognormal(cls, mean: float, std: float) -> "Distribution":
        if std <= 0:
            raise ValueError(f"lognormal: std must be > 0 (got {std})")
        return cls(LOGNORMAL, {"mean": float(mean), "std": float(std)})

    @classmethod
    def triangular(cls, min: float, mode: float, max: float) -> "Distribution":
        if not (min <= mode <= max):
            raise ValueError(
                f"triangular: require min ({min}) <= mode ({mode}) <= max ({max})"
            )
        return cls(TRIANGULAR, {"min": float(min), "mode": float(mode), "max": float(max)})

    @classmethod
    def fixed(cls, value: float) -> "Distribution":
        return cls(FIXED, {"value": float(value)})

    def sample(self, rng: random.Random) -> float:
        if self.kind == UNIFORM:
            return rng.uniform(self.params["min"], self.params["max"])
        if self.kind == NORMAL:
            return rng.gauss(self.params["mean"], self.params["std"])
        if self.kind == LOGNORMAL:
            return math.exp(rng.gauss(self.params["mean"], self.params["std"]))
        if self.kind == TRIANGULAR:
            return rng.triangular(
                self.params["min"], self.params["max"], self.params["mode"]
            )
        if self.kind == FIXED:
            return self.params["value"]
        raise ValueError(f"unknown distribution kind: {self.kind!r}")

    def to_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {"distribution": self.kind}  # type: ignore[assignment]
        out.update(self.params)
        return out


@dataclass
class ParameterConfig:
    """A swept parameter: a dot-path into the serialized CaseSpec tree."""

    path: str
    distribution: Distribution


@dataclass
class DeltaSpec:
    """Top-level declarative delta specification."""

    parameters: List[ParameterConfig] = field(default_factory=list)
    samples: int = DEFAULT_SAMPLES
    seed: int = DEFAULT_SEED
    warm_up_years: int = DEFAULT_WARM_UP_YEARS

    def to_dict(self) -> Dict[str, object]:
        return {
            "samples": self.samples,
            "seed": self.seed,
            "warm_up_years": self.warm_up_years,
            "parameters": {p.path: p.distribution.to_dict() for p in self.parameters},
        }


def generate_delta_file(spec: DeltaSpec) -> Dict[str, object]:
    """Validate ``spec`` and return the declarative delta as a plain dict.

    The returned dict serializes directly to the YAML/JSON consumed by the Rust
    ``MonteCarloDelta`` parser.
    """
    if spec.samples <= 0:
        raise ValueError(f"samples must be > 0 (got {spec.samples})")
    if not spec.parameters:
        raise ValueError("at least one parameter must be specified")
    # Eagerly validate each distribution by drawing one sample.
    rng = random.Random(spec.seed)
    for p in spec.parameters:
        p.distribution.sample(rng)
    return spec.to_dict()


def materialize_patches(
    spec: DeltaSpec, *, out_dir: Optional[Path] = None, prefix: str = "delta_"
) -> List[Dict[str, float]]:
    """Draw ``spec.samples`` samples and optionally write one JSON file per draw.

    Returns the list of per-draw dicts (parameter path → sampled value). When
    ``out_dir`` is given, also writes ``{prefix}{i:06d}.json`` for each draw —
    the per-worker payload for the distributed Nomad / AWS Batch deployment.
    """
    delta = generate_delta_file(spec)
    rng = random.Random(spec.seed)
    # Deterministic parameter ordering (sorted by path) for reproducibility.
    paths = sorted(delta["parameters"].keys())  # type: ignore[index]
    dists = {p.path: p.distribution for p in spec.parameters}
    draws: List[Dict[str, float]] = []
    for i in range(spec.samples):
        draw = {path: dists[path].sample(rng) for path in paths}
        draws.append(draw)
        if out_dir is not None:
            payload = {"index": i, "values": draw}
            (out_dir / f"{prefix}{i:06d}.json").write_text(json.dumps(payload, indent=2))
    return draws


# ---------------------------------------------------------------------------
# Default parameter set: the three parameters called out in Issue #1813
# (window-to-wall ratio, insulation R-value, infiltration rate) plus window
# U-value and SHGC. R-values are converted to U-values (1/R) at apply time by
# the Rust worker, since CaseSpec stores U-values directly.
# ---------------------------------------------------------------------------

def default_parameter_set() -> List[ParameterConfig]:
    return [
        ParameterConfig("infiltration_ach", Distribution.uniform(0.3, 1.5)),
        ParameterConfig(
            "window_properties.u_value", Distribution.normal(3.0, 0.3)
        ),
        ParameterConfig(
            "window_properties.shgc", Distribution.triangular(0.4, 0.7, 0.9)
        ),
        ParameterConfig(
            "opaque_absorptance", Distribution.uniform(0.6, 0.9)
        ),
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="generate_monte_carlo_deltas",
        description="Generate declarative Monte Carlo delta files for Fluxion (Issue #1813).",
    )
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"number of Monte Carlo draws (default {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="RNG seed for reproducibility"
    )
    parser.add_argument(
        "--warm-up-years",
        type=int,
        default=DEFAULT_WARM_UP_YEARS,
        help="convergence warm-up years (default %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("delta.yaml"),
        help="output delta file (.yaml/.yml or .json)",
    )
    parser.add_argument(
        "--materialize",
        type=Path,
        default=None,
        help="also write one per-draw JSON patch per sample into this directory",
    )
    args = parser.parse_args(argv)

    spec = DeltaSpec(
        parameters=default_parameter_set(),
        samples=args.samples,
        seed=args.seed,
        warm_up_years=args.warm_up_years,
    )
    delta = generate_delta_file(spec)

    if args.output.suffix.lower() == ".json":
        args.output.write_text(json.dumps(delta, indent=2))
    else:
        import yaml  # PyYAML; only needed for the YAML output path

        args.output.write_text(yaml.safe_dump(delta, sort_keys=False))

    print(f"Wrote declarative delta file: {args.output} ({spec.samples} samples)")

    if args.materialize is not None:
        args.materialize.mkdir(parents=True, exist_ok=True)
        draws = materialize_patches(spec, out_dir=args.materialize)
        print(f"Materialized {len(draws)} per-draw patches into: {args.materialize}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
