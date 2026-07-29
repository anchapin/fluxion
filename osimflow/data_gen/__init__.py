"""OSimFlow data generation utilities (Issue #1813).

Generates declarative Monte Carlo delta files (and optionally pre-materialized
per-draw JSON patches) consumed by the Fluxion Rust worker entrypoint
(``fluxion monte-carlo sweep --base-model ... --delta-file ...``).

Phase 1 (Declarative Deltas) of the OSimFlow hybrid-measure approach: send one
base model plus thousands of lightweight delta files instead of one full model
per cloud worker.
"""

from .generate_monte_carlo_deltas import (
    Distribution,
    DeltaSpec,
    ParameterConfig,
    generate_delta_file,
    materialize_patches,
    main,
)

__all__ = [
    "Distribution",
    "DeltaSpec",
    "ParameterConfig",
    "generate_delta_file",
    "materialize_patches",
    "main",
]
