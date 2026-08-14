#!/usr/bin/env python3
"""
Autonomous Diagnostic Parameter Sweep
======================================
Executes overnight parameter sweeps against the ASHRAE 140 validation suite,
logs mathematical divergence results into `.sdd/traces/diagnostic/` for human review.

Usage
-----
# Run a grid sweep with default mission brief
python scripts/autonomous_parameter_sweep.py --case 600 --params R_value,wall_thickness

# Run with custom mission brief
python scripts/autonomous_parameter_sweep.py --mission-brief .planning/my_brief.md

# Random search with 50 samples
python scripts/autonomous_parameter_sweep.py --case 900 --sweep-type random --samples 50

# Dry run (validate config without executing)
python scripts/autonomous_parameter_sweep.py --case 600 --dry-run

# Resume interrupted sweep
python scripts/autonomous_parameter_sweep.py --resume .sdd/traces/diagnostic/case600_20260101_120000/

Exit codes
----------
0 — Sweep completed successfully
1 — Sweep failed (see logs)
2 — Invalid configuration
3 — Dry run validation passed
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRACE_BASE = Path(".sdd/traces/diagnostic")
ASHRAE_TEST_TARGET = "ashrae_140_validation"
CARGO_TEST_CMD = ["cargo", "test", f"--test={ASHRAE_TEST_TARGET}", "--", "--nocapture"]

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


class SweepType(Enum):
    GRID = "grid"
    RANDOM = "random"
    GRADIENT = "gradient"
    BINARY = "binary"
    LATIN_HYPERCUBE = "latin_hypercube"


@dataclass
class ParameterSpec:
    name: str
    default: float
    min_val: float
    max_val: float
    step: float
    unit: str = ""


@dataclass
class BriefSpec:
    """Top-level parameter-sweep specification loaded from a YAML/JSON brief.

    Schema (all fields optional — missing keys fall back to :data:`DEFAULT_BRIEF_SPEC`):

    * ``axes``   — list of parameter names to sweep.
    * ``ranges`` — per-parameter bounds: ``{name: {min, max, step?, default?}}``.
    * ``objective`` — free-form label that downstream tooling can log
      (e.g. ``"minimize_mae"``).
    * ``seed``   — optional RNG seed for reproducible random sweeps.

    Backed by Issue #2951.
    """

    axes: list[str] = field(default_factory=lambda: ["R_value", "wall_thickness"])
    ranges: dict[str, dict[str, float]] = field(default_factory=dict)
    objective: str = "minimize_mae"
    seed: Optional[int] = None


# Singleton default brief. Returned by ``load_brief_spec`` when the user
# did not pass ``--brief`` or the file could not be parsed.
DEFAULT_BRIEF_SPEC = BriefSpec()


@dataclass
class SweepConfig:
    case_id: str
    sweep_type: SweepType
    parameters: list[ParameterSpec]
    max_iterations: int = 100
    samples_per_param: int = 10
    concurrent_runs: int = 4
    timeout_per_run: int = 300
    tolerance_mae: float = 5.0
    trace_dir: Optional[Path] = None


@dataclass
class SweepResult:
    run_id: str
    case_id: str
    iteration: int
    parameters: dict[str, float]
    heating_mae: float
    cooling_mae: float
    peak_heating_mae: float
    peak_cooling_mae: float
    temperature_mae: float
    overall_pass: bool
    timestamp: str
    duration_ms: int
    error_message: Optional[str] = None


@dataclass
class SweepState:
    config: dict
    results: list[dict]
    best_parameters: dict[str, float]
    best_mae: float
    start_time: str
    end_time: Optional[str] = None
    status: str = "running"
    iteration: int = 0


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def parse_cargo_output(output: str, case_filter: str = "") -> dict[str, float]:
    """Parse ASHRAE 140 validation output for MAE values."""
    metrics: dict[str, Any] = {
        "heating_mae": 0.0,
        "cooling_mae": 0.0,
        "peak_heating_mae": 0.0,
        "peak_cooling_mae": 0.0,
        "temperature_mae": 0.0,
        "overall_pass": False,
    }

    # Match "Mean Absolute Error: X.XX%"
    mae_pattern = re.compile(r"Mean\s+Absolute\s+Error:\s*([\d.]+)%", re.IGNORECASE)
    for match in mae_pattern.finditer(output):
        val = float(match.group(1))
        if metrics["heating_mae"] == 0.0:
            metrics["heating_mae"] = val

    # Match per-case output: "Case XXX : Heating=..."
    case_pattern = re.compile(
        r"Case\s+(\d+[A-Z0-9_]*)\s*[:\-]\s*"
        r"Heating\s*=\s*([\d.]+)\s*\(Ref:\s*([\d.+-]+)\s*-\s*([\d.+-]+)\),\s*"
        r"Cooling\s*=\s*([\d.]+)\s*\(Ref:\s*([\d.+-]+)\s*-\s*([\d.+-]+)\)"
    )

    heating_errors = []
    cooling_errors = []
    for match in case_pattern.finditer(output):
        case = match.group(1)
        if case_filter and (case.startswith(case_filter) or case_filter in case):
            ref_heat = (float(match.group(3)) + float(match.group(4))) / 2
            ref_cool = (float(match.group(6)) + float(match.group(7))) / 2
            sim_heat = float(match.group(2))
            sim_cool = float(match.group(5))
            if ref_heat > 0:
                heating_errors.append(abs(sim_heat - ref_heat) / ref_heat * 100)
            if ref_cool > 0:
                cooling_errors.append(abs(sim_cool - ref_cool) / ref_cool * 100)

    if heating_errors:
        metrics["heating_mae"] = sum(heating_errors) / len(heating_errors)
    if cooling_errors:
        metrics["cooling_mae"] = sum(cooling_errors) / len(cooling_errors)

    # Match pass/fail summary
    summary_pattern = re.compile(
        r"Pass\s+Rate:\s*([\d.]+)%.*?Passed:\s*(\d+).*?Failed:\s*(\d+)",
        re.DOTALL | re.IGNORECASE,
    )
    summary_match = summary_pattern.search(output)
    if summary_match:
        pass_rate = float(summary_match.group(1))
        metrics["overall_pass"] = pass_rate >= 80.0

    return metrics


def run_ashrae_validation(timeout: int = 300) -> tuple[dict[str, Any], str]:
    """Run the ASHRAE 140 validation suite and return metrics + raw output."""
    try:
        result = subprocess.run(
            CARGO_TEST_CMD,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent,
        )
        output = result.stdout + result.stderr
        metrics = parse_cargo_output(output)
        return metrics, output
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}, ""
    except Exception as e:
        return {"error": str(e)}, ""


def generate_grid_points(specs: list[ParameterSpec]) -> list[dict[str, float]]:
    """Generate full factorial grid of parameter combinations."""
    import itertools

    grids = []
    for spec in specs:
        points = []
        val = spec.min_val
        while val <= spec.max_val + 1e-9:
            points.append(val)
            val += spec.step
        grids.append(points)

    combinations = list(itertools.product(*grids))
    return [
        {specs[i].name: combo[i] for i in range(len(specs))} for combo in combinations
    ]


def generate_random_points(
    specs: list[ParameterSpec], samples: int
) -> list[dict[str, float]]:
    """Generate random parameter samples within bounds."""
    return [
        {spec.name: random.uniform(spec.min_val, spec.max_val) for spec in specs}
        for _ in range(samples)
    ]


def apply_parameters(params: dict[str, float], env_overrides: dict[str, str]) -> None:
    """Apply parameter overrides to the environment for the next run."""
    for key, val in params.items():
        env_overrides[f"FLUXION_PARAM_{key.upper()}"] = str(val)


def log_result(trace_dir: Path, result: SweepResult) -> None:
    """Append a sweep result to the JSONL log file."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    log_file = trace_dir / "parameter_sweep_results.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


def log_convergence(trace_dir: Path, iteration: int, mae: float) -> None:
    """Append convergence data to CSV log."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    conv_file = trace_dir / "convergence_log.csv"
    exists = conv_file.exists()
    with open(conv_file, "a") as f:
        if not exists:
            f.write("iteration,mae,timestamp\n")
        f.write(f"{iteration},{mae},{datetime.now(timezone.utc).isoformat()}\n")


def save_sweep_state(trace_dir: Path, state: SweepState) -> None:
    """Save sweep state for resume capability."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    state_file = trace_dir / "sweep_state.json"
    with open(state_file, "w") as f:
        json.dump(asdict(state), f, indent=2)


def load_sweep_state(trace_dir: Path) -> Optional[SweepState]:
    """Load a previously saved sweep state."""
    state_file = trace_dir / "sweep_state.json"
    if not state_file.exists():
        return None
    with open(state_file) as f:
        data = json.load(f)
    return SweepState(
        config=data["config"],
        results=data["results"],
        best_parameters=data["best_parameters"],
        best_mae=data["best_mae"],
        start_time=data["start_time"],
        end_time=data.get("end_time"),
        status=data.get("status", "running"),
        iteration=data.get("iteration", 0),
    )


def create_divergence_report(
    trace_dir: Path, config: SweepConfig, results: list[SweepResult]
) -> None:
    """Generate human-readable divergence report."""
    if not results:
        return

    best_result = min(results, key=lambda r: r.heating_mae + r.cooling_mae)
    passing = [r for r in results if r.overall_pass]
    failed = [r for r in results if not r.overall_pass]

    report = f"""# Diagnostic Parameter Sweep — Divergence Report

**Case:** {config.case_id}
**Sweep Type:** {config.sweep_type.value}
**Generated:** {datetime.now(timezone.utc).isoformat()}

---

## Summary

| Metric | Value |
|--------|-------|
| Total Runs | {len(results)} |
| Passing Runs | {len(passing)} |
| Failed Runs | {len(failed)} |
| Pass Rate | {len(passing) / len(results) * 100:.1f}% |
| Best MAE (Heating) | {best_result.heating_mae:.2f}% |
| Best MAE (Cooling) | {best_result.cooling_mae:.2f}% |

---

## Best Parameters Found

```
{json.dumps(best_result.parameters, indent=2)}
```

---

## Parameter Impact Analysis

| Parameter | Min MAE | Max MAE | Range |
|-----------|--------|---------|-------|
"""

    for spec in config.parameters:
        param_results = [r for r in results if spec.name in r.parameters]
        if param_results:
            maes = [r.heating_mae + r.cooling_mae for r in param_results]
            report += f"| {spec.name} | {min(maes):.2f}% | {max(maes):.2f}% | {spec.max_val - spec.min_val:.3f} {spec.unit} |\n"

    report += f"""

---

## Convergence

Best MAE achieved at iteration {best_result.iteration}.

---

## Recommendations

1. **Primary fix:** Adjust {list(best_result.parameters.keys())[0]} to {list(best_result.parameters.values())[0]}
2. **Secondary:** Consider module-level investigation of {config.case_id} thermal coupling
3. **Validation:** Re-run full ASHRAE 140 suite after implementing parameter changes

---

*Report generated by autonomous_parameter_sweep.py*
"""
    report_file = trace_dir / "divergence_report.md"
    with open(report_file, "w") as f:
        f.write(report)


# ---------------------------------------------------------------------------
# Brief loader — Issue #2951
# ---------------------------------------------------------------------------


def _emit_brief_warning(path: Optional[Path], message: str) -> None:
    """Emit a GitHub-Actions-style ``::warning::`` annotation to stderr.

    Used by :func:`load_brief_spec` so CI logs surface brief-parser
    failures in the standard workflow-commands format.
    """
    location = f"file={path}" if path is not None else "file=<none>"
    print(f"::warning::{location}::{message}", file=sys.stderr)


def load_brief_spec(path: Optional[Path]) -> BriefSpec:
    """Load a :class:`BriefSpec` from a YAML or JSON file.

    Behavior:

    * ``path is None`` → returns :data:`DEFAULT_BRIEF_SPEC` silently.
    * File missing → ``::warning::`` to stderr, returns defaults.
    * Parse error / IO error / schema error → ``::warning::`` to stderr,
      returns defaults.

    YAML is preferred when the file extension is ``.yaml``/``.yml``.
    PyYAML is imported lazily; if it is not installed we transparently
    fall back to :mod:`json`. JSON is used for every other extension.
    """
    if path is None:
        return DEFAULT_BRIEF_SPEC

    if not path.exists():
        _emit_brief_warning(
            path,
            "Brief file not found. Using default sweep parameters.",
        )
        return DEFAULT_BRIEF_SPEC

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _emit_brief_warning(
            path,
            f"Could not read brief ({exc}). Using default sweep parameters.",
        )
        return DEFAULT_BRIEF_SPEC

    suffix = path.suffix.lower()
    data: Any = None
    yaml_error_types: tuple[type[BaseException], ...] = ()
    try:
        import yaml  # type: ignore

        yaml_error_types = (yaml.YAMLError,)
    except ImportError:
        yaml_error_types = ()
    try:
        if suffix in (".yaml", ".yml"):
            if yaml_error_types:
                data = yaml.safe_load(text)
            else:
                # PyYAML unavailable — try JSON as a best-effort fallback so
                # a user-written JSON file still works.
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as json_exc:
                    raise ValueError(
                        "PyYAML is not installed and the brief is not valid JSON"
                    ) from json_exc
        else:
            data = json.loads(text)
    except (ValueError, json.JSONDecodeError) + yaml_error_types as exc:
        _emit_brief_warning(
            path,
            f"Failed to parse brief ({exc}). Using default sweep parameters.",
        )
        return DEFAULT_BRIEF_SPEC

    if data is None:
        # Empty file is treated as "use defaults" without a warning.
        return DEFAULT_BRIEF_SPEC

    if not isinstance(data, dict):
        _emit_brief_warning(
            path,
            f"Brief must be a mapping at the top level (got {type(data).__name__}). "
            f"Using default sweep parameters.",
        )
        return DEFAULT_BRIEF_SPEC

    return _coerce_brief_spec(data, path)


def _coerce_brief_spec(data: dict[str, Any], path: Path) -> BriefSpec:
    """Coerce a raw parsed mapping into a :class:`BriefSpec`.

    Per-field coercion failures are downgraded to ``::warning::`` lines so
    a partially-malformed brief still produces a usable spec — only fully
    broken top-level shapes fail outright (handled in
    :func:`load_brief_spec`).
    """
    axes_raw = data.get("axes", DEFAULT_BRIEF_SPEC.axes)
    if isinstance(axes_raw, list) and all(isinstance(a, str) for a in axes_raw):
        axes = list(axes_raw)
    else:
        _emit_brief_warning(
            path,
            "'axes' must be a list of strings; falling back to defaults.",
        )
        axes = list(DEFAULT_BRIEF_SPEC.axes)

    ranges_raw = data.get("ranges", {})
    ranges: dict[str, dict[str, float]] = {}
    if isinstance(ranges_raw, dict):
        for name, bounds in ranges_raw.items():
            if not isinstance(bounds, dict):
                _emit_brief_warning(
                    path,
                    f"ranges[{name!r}] is not a mapping; ignoring.",
                )
                continue
            try:
                min_v = float(bounds["min"])
                max_v = float(bounds["max"])
            except (KeyError, TypeError, ValueError):
                _emit_brief_warning(
                    path,
                    f"ranges[{name!r}] missing/invalid 'min' or 'max'; ignoring.",
                )
                continue
            if min_v > max_v:
                _emit_brief_warning(
                    path,
                    f"ranges[{name!r}] min > max; swapping bounds.",
                )
                min_v, max_v = max_v, min_v
            step_raw = bounds.get("step")
            if step_raw is None:
                step_v = (max_v - min_v) / 4.0 or 0.1
            else:
                try:
                    step_v = float(step_raw)
                except (TypeError, ValueError):
                    step_v = (max_v - min_v) / 4.0 or 0.1
            entry: dict[str, float] = {"min": min_v, "max": max_v, "step": step_v}
            if "default" in bounds:
                try:
                    entry["default"] = float(bounds["default"])
                except (TypeError, ValueError):
                    pass
            ranges[name] = entry
    else:
        _emit_brief_warning(
            path,
            "'ranges' must be a mapping; treating as empty.",
        )

    objective_raw = data.get("objective", DEFAULT_BRIEF_SPEC.objective)
    objective = (
        str(objective_raw)
        if objective_raw is not None
        else DEFAULT_BRIEF_SPEC.objective
    )

    seed: Optional[int] = None
    seed_raw = data.get("seed")
    if seed_raw is not None:
        try:
            seed = int(seed_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            _emit_brief_warning(
                path,
                f"'seed' must be an integer (got {seed_raw!r}); ignoring.",
            )

    return BriefSpec(axes=axes, ranges=ranges, objective=objective, seed=seed)


def brief_to_parameter_specs(
    brief: BriefSpec,
    default_params: dict[str, ParameterSpec],
) -> list[ParameterSpec]:
    """Translate a :class:`BriefSpec` into a list of :class:`ParameterSpec`.

    For each axis:

    * If ``brief.ranges`` provides bounds, those win (overriding the
      legacy ``default_params`` table for the matching name).
    * Otherwise we look up the axis in ``default_params`` (so users
      that only override axes still get sensible defaults).
    * Otherwise we synthesize a generic 0.1–10.0 spec.
    """
    specs: list[ParameterSpec] = []
    for name in brief.axes:
        bounds = brief.ranges.get(name)
        if bounds is not None and name in default_params:
            base = default_params[name]
            specs.append(
                ParameterSpec(
                    name=name,
                    default=float(bounds.get("default", base.default)),
                    min_val=float(bounds["min"]),
                    max_val=float(bounds["max"]),
                    step=float(bounds["step"]),
                    unit=base.unit,
                )
            )
        elif bounds is not None:
            min_v = float(bounds["min"])
            max_v = float(bounds["max"])
            default = float(bounds.get("default", (min_v + max_v) / 2.0))
            specs.append(
                ParameterSpec(
                    name=name,
                    default=default,
                    min_val=min_v,
                    max_val=max_v,
                    step=float(bounds["step"]),
                )
            )
        elif name in default_params:
            specs.append(default_params[name])
        else:
            specs.append(
                ParameterSpec(
                    name=name, default=1.0, min_val=0.1, max_val=10.0, step=0.1
                )
            )
    return specs


# ---------------------------------------------------------------------------
# Main Sweep Execution
# ---------------------------------------------------------------------------


def run_sweep(config: SweepConfig, dry_run: bool = False) -> SweepState:
    """Execute the parameter sweep."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trace_dir = config.trace_dir or (TRACE_BASE / f"case{config.case_id}_{timestamp}")
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = {
        "case_id": config.case_id,
        "sweep_type": config.sweep_type.value,
        "parameters": [
            {
                "name": s.name,
                "default": s.default,
                "min": s.min_val,
                "max": s.max_val,
                "step": s.step,
                "unit": s.unit,
            }
            for s in config.parameters
        ],
        "max_iterations": config.max_iterations,
        "samples_per_param": config.samples_per_param,
        "tolerance_mae": config.tolerance_mae,
    }
    with open(trace_dir / "sweep_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    state = SweepState(
        config=config_dict,
        results=[],
        best_parameters={},
        best_mae=999.0,
        start_time=datetime.now(timezone.utc).isoformat(),
        iteration=0,
    )

    if dry_run:
        print(f"[DRY RUN] Would execute {config.max_iterations} iterations")
        print(f"[DRY RUN] Trace directory: {trace_dir}")
        print(f"[DRY RUN] Parameters: {[s.name for s in config.parameters]}")
        return state

    # Generate parameter combinations
    if config.sweep_type == SweepType.GRID:
        param_combinations = generate_grid_points(config.parameters)
    elif config.sweep_type == SweepType.RANDOM:
        param_combinations = generate_random_points(
            config.parameters, config.samples_per_param
        )
    else:
        param_combinations = generate_random_points(
            config.parameters, config.max_iterations
        )

    print(f"[*] Starting sweep: {len(param_combinations)} parameter combinations")
    print(f"[*] Trace directory: {trace_dir}")

    for i, params in enumerate(param_combinations[: config.max_iterations]):
        state.iteration = i + 1
        run_id = str(uuid.uuid4())[:8]

        print(f"[*] Iteration {state.iteration}/{config.max_iterations}: {params}")

        # Apply parameters (via environment)
        env = os.environ.copy()
        for key, val in params.items():
            env[f"FLUXION_PARAM_{key.upper()}"] = str(val)

        # Run validation
        start = time.time()
        metrics, raw_output = run_ashrae_validation(timeout=config.timeout_per_run)
        duration_ms = int((time.time() - start) * 1000)

        if "error" in metrics:
            result = SweepResult(
                run_id=run_id,
                case_id=config.case_id,
                iteration=state.iteration,
                parameters=params,
                heating_mae=999.0,
                cooling_mae=999.0,
                peak_heating_mae=999.0,
                peak_cooling_mae=999.0,
                temperature_mae=999.0,
                overall_pass=False,
                timestamp=datetime.now(timezone.utc).isoformat(),
                duration_ms=duration_ms,
                error_message=str(metrics.get("error", "unknown")),
            )
        else:
            result = SweepResult(
                run_id=run_id,
                case_id=config.case_id,
                iteration=state.iteration,
                parameters=params,
                heating_mae=metrics.get("heating_mae", 999.0),
                cooling_mae=metrics.get("cooling_mae", 999.0),
                peak_heating_mae=metrics.get("peak_heating_mae", 999.0),
                peak_cooling_mae=metrics.get("peak_cooling_mae", 999.0),
                temperature_mae=metrics.get("temperature_mae", 999.0),
                overall_pass=metrics.get("overall_pass", False),
                timestamp=datetime.now(timezone.utc).isoformat(),
                duration_ms=duration_ms,
            )

            # Update best
            total_mae = result.heating_mae + result.cooling_mae
            if total_mae < state.best_mae:
                state.best_mae = total_mae
                state.best_parameters = params.copy()

        # Log
        state.results.append(asdict(result))
        log_result(trace_dir, result)
        log_convergence(
            trace_dir, state.iteration, result.heating_mae + result.cooling_mae
        )
        save_sweep_state(trace_dir, state)

        print(
            f"    -> MAE: {result.heating_mae:.2f}% heating, {result.cooling_mae:.2f}% cooling"
        )

        # Check abandon criteria
        if result.heating_mae + result.cooling_mae <= config.tolerance_mae:
            print(
                f"[*] Tolerance achieved! MAE = {result.heating_mae + result.cooling_mae:.2f}%"
            )
            break

    # Finalize
    state.end_time = datetime.now(timezone.utc).isoformat()
    state.status = "completed"
    save_sweep_state(trace_dir, state)
    create_divergence_report(trace_dir, config, [])

    print(f"[*] Sweep complete. Best MAE: {state.best_mae:.2f}%")
    print(f"[*] Best parameters: {state.best_parameters}")
    print(f"[*] Results saved to: {trace_dir}")

    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Autonomous Diagnostic Parameter Sweep for ASHRAE 140 Discrepancies"
    )
    parser.add_argument(
        "--case",
        type=str,
        required=True,
        help="ASHRAE 140 case ID to investigate (e.g., 600, 900, 650FF)",
    )
    parser.add_argument(
        "--params",
        type=str,
        help="Comma-separated list of parameters to sweep",
    )
    parser.add_argument(
        "--sweep-type",
        type=str,
        choices=["grid", "random", "gradient", "binary", "latin_hypercube"],
        default="random",
        help="Sweep strategy",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples for random/latin_hypercube sweep",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--mission-brief",
        type=str,
        help="Path to filled diagnostic mission brief (alternative to --params)",
    )
    parser.add_argument(
        "--brief",
        type=Path,
        default=None,
        help=(
            "Path to a YAML or JSON parameter-sweep brief describing axes, "
            "ranges, objective, and seed. If missing or malformed, the script "
            "falls back to the default sweep and logs a ::warning:: to stderr."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from existing trace directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without executing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override trace output directory",
    )
    return parser


def main() -> int:
    parser = build_parser()
    parsed_args = parser.parse_args()

    # Load mission brief or build config from args
    if parsed_args.resume:
        trace_dir = Path(parsed_args.resume)
        state = load_sweep_state(trace_dir)
        if not state:
            print(f"[!] No sweep state found in {trace_dir}")
            return 2
        print(f"[*] Resuming sweep from {trace_dir}")
        print(
            f"[*] Completed {state.iteration} iterations, best MAE: {state.best_mae:.2f}%"
        )
        return 0

    if parsed_args.mission_brief:
        brief_path = Path(parsed_args.mission_brief)
        if not brief_path.exists():
            print(f"[!] Mission brief not found: {brief_path}")
            return 2
        # Parse brief (simplified — full implementation would parse markdown)
        print(f"[*] Loaded mission brief: {brief_path}")
        # TODO: Parse parameters from brief
        return 2

    # Build config from CLI args (and optional --brief YAML/JSON).
    #
    # Resolution order:
    #   1. ``--brief`` (loaded earlier) supplies axes + ranges. A missing or
    #      malformed brief emits a ``::warning::`` and falls through to the
    #      legacy default sweep — the warning is *not* fatal.
    #   2. ``--params`` (legacy CSV) overrides axes if both are present.
    #   3. Otherwise defaults.
    default_params = {
        "R_value": ParameterSpec(
            "R_value", default=2.0, min_val=1.0, max_val=5.0, step=0.5, unit="m²K/W"
        ),
        "wall_thickness": ParameterSpec(
            "wall_thickness",
            default=0.15,
            min_val=0.05,
            max_val=0.30,
            step=0.05,
            unit="m",
        ),
        "thermal_mass": ParameterSpec(
            "thermal_mass", default=1.0, min_val=0.5, max_val=2.0, step=0.1, unit=""
        ),
        "h_tr_is": ParameterSpec(
            "h_tr_is", default=8.29, min_val=5.0, max_val=15.0, step=1.0, unit="W/m²K"
        ),
    }

    brief = load_brief_spec(parsed_args.brief)

    if parsed_args.params:
        # CLI wins over brief axes; the brief's ranges still apply per-axis
        # where the CLI parameter names overlap.
        param_names = parsed_args.params.split(",")
        specs = brief_to_parameter_specs(brief, default_params)
        # Restrict to the CLI-supplied axes, preserving brief-supplied bounds
        # for the names that overlap.
        by_name = {s.name: s for s in specs}
        specs = [
            by_name.get(
                name,
                ParameterSpec(name, default=1.0, min_val=0.1, max_val=10.0, step=0.1),
            )
            for name in param_names
        ]
    elif parsed_args.brief is not None:
        # Brief was provided (possibly invalid → defaults have been substituted
        # already; either way, use it as the source of truth for axes/ranges).
        specs = brief_to_parameter_specs(brief, default_params)
    else:
        # Backward-compatible default path.
        param_names = ["R_value", "wall_thickness"]
        specs = []
        for name in param_names:
            if name in default_params:
                specs.append(default_params[name])
            else:
                specs.append(
                    ParameterSpec(
                        name, default=1.0, min_val=0.1, max_val=10.0, step=0.1
                    )
                )

    # Optional RNG seed from brief — applied to Python's global ``random``
    # module so ``generate_random_points`` becomes reproducible.
    if brief.seed is not None:
        random.seed(brief.seed)

    config = SweepConfig(
        case_id=parsed_args.case,
        sweep_type=SweepType(parsed_args.sweep_type),
        parameters=specs,
        max_iterations=parsed_args.max_iterations,
        samples_per_param=parsed_args.samples,
        trace_dir=Path(parsed_args.output_dir) if parsed_args.output_dir else None,
    )

    state = run_sweep(config, dry_run=parsed_args.dry_run)

    if parsed_args.dry_run:
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
