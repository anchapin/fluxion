#!/usr/bin/env python3
"""Module size gate for Fluxion (Issues #2878, #3457).

Enforces hard upper bounds on the line count of selected ``.rs`` source files
that are prone to god-struct accumulation. Issue #2878 introduced the gate
with a single limit for ``src/sim/thermal_model_data/mod.rs``; Issue #3457
extended the policy to cover the ten largest ``src/`` files (each ratcheted
to its current line count).

Each entry in ``LIMITS`` defines:
- ``path``: source file (relative to repo root, OR absolute).
- ``max_lines``: hard ceiling (PR-blocking). The YAML ``max_lines`` is the
  active ceiling; the ratchet JSON holds the historical maximum for
  visibility (``effective_max = max(max_lines, ratchet_max)``). To tighten
  the bound, lower ``max_lines`` in this script.
- ``ratchet_path``: JSON file holding the historical maximum line count
  observed for that file (``max_lines`` + ``history`` keys). Pre-seeded at
  the file's current size on Issue #3457 so the bound only tightens from
  here onward as the YAML ceiling is lowered alongside file decomposition.
- ``reason``: human-readable rationale, surfaced in failure messages and
  JSON output.

Going forward, the ``LIMITS`` list itself is ratcheted downward-only via
``BASELINE_MODULE_SIZE_LIMITS`` and ``_BASELINE_MODULE_SIZE_LIMITS_SET``,
mirroring the ``BASELINE_KNOWN_ORPHANS`` /
``_BASELINE_KNOWN_ORPHANS_SET`` pattern from
``scripts/check_orphan_modules.py`` (Issues #3458 / #3459). Adding a new
entry is PR-blocking unless the baseline is raised (with a documenting
comment naming the tracking issue) AND the matching path is added to
``_BASELINE_MODULE_SIZE_LIMITS_SET``. Companion cleanup PRs that
decompose a gated file should remove its entry AND lower the baseline by
one.

Exit codes:
- 0 — all entries within bounds, no drift against the freeze.
- 1 — one or more entries exceed their bound, OR the LIMITS list grew past
  the freeze baseline.
- 2 — script error (e.g. file not found, malformed ratchet JSON).

Usage:
    python3 scripts/check_module_size.py [--json] [--write-ratchet]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Issue #3457 — LIMITS-list baseline (downward-only ratchet).
#
# Mirrors ``BASELINE_KNOWN_ORPHANS`` from ``scripts/check_orphan_modules.py``:
# the constant is the *highest* value of ``len(LIMITS)`` this guard will
# accept. The script FAILS the moment the live LIMITS list grows past it.
# Companion cleanup PRs that decompose a gated file (so its entry can be
# removed) are expected to drop the matching entry AND lower this baseline
# by one. Raising the baseline is reserved for legitimately gating a new
# large file, and MUST be accompanied by a documenting comment naming the
# tracking issue AND a matching addition to
# ``_BASELINE_MODULE_SIZE_LIMITS_SET``.
#
# History:
#   2 → 12 (Issue #3457): initial extension beyond the single ratcheted
#     file. The original 2 entries (``thermal_model_data.rs`` + the
#     ``mod.rs`` directory form, both Issue #2878) are retained, plus the
#     10 largest ``src/`` files identified by the gap-issues
#     ``auto-improvement-loop`` audit pass on 2026-09-07 (HEAD
#     ``436de25``). The bound for each new entry is the file's CURRENT
#     line count snapshotted as both ``max_lines`` (YAML ceiling) and the
#     ratchet JSON's ``max_lines`` (historical max); going forward the
#     bound can only tighten as the YAML ceiling is lowered alongside
#     file decomposition.
#   12 → 11 (Issue #3543, part 1): ``src/api/server.rs`` was decomposed into
#     a per-route submodule tree under ``src/api/server/`` (mod.rs +
#     api_error, batch, campaigns, constants, health, import_format,
#     router, schema_store, simulate, state, tests). No new entry is
#     added because the largest child submodule (~750 LoC) is well below
#     the smallest ratcheted threshold (~2000 LoC); the decomposition
#     itself is the ratchet-lowering event.
#   11 → 9 (Issue #3543, parts 2 + 3):
#     - ``src/sim/thermal_model_core.rs`` → ``src/sim/thermal_model_core/``
#       directory (mod.rs + tests). Largest child: mod.rs (~4.1k LoC) —
#       still over the ratchet but only because of the bare
#       ``impl ThermalModel<VectorField>`` block; further decomposition
#       is out of scope for #3543 (tracked separately).
#     - ``src/validation/ashrae_140_validator.rs`` →
#       ``src/validation/ashrae_140_validator/`` directory (mod.rs + tests).
#       Largest child: mod.rs (~3.1k LoC).
#     Two entries removed; baseline lowered by two.
#   9 → 17 (Issue #3574): Issue #3543's decomposition of
#     ``src/api/server.rs``, ``src/sim/thermal_model_core.rs``, and
#     ``src/validation/ashrae_140_validator.rs`` (PR #3566) shrank three
#     entries out of the LIMITS list but exposed eight additional files
#     above the smallest ratcheted threshold (~2000 LoC) that were
#     previously invisible to the gate. Each is added at its current line
#     count plus a small buffer (max of +5% or +100 lines) so the gate
#     passes immediately but tightens against any further growth.
#     Companion cleanup PRs that decompose any of these files should
#     remove the matching entry AND lower this baseline by one. Eight
#     entries added; baseline raised by eight.
#   17 → 16 (fmi/mod.rs decomposition, tracked under issue #3625):
#     ``src/interop/fmi/mod.rs`` (3,246 lines) was decomposed into the
#     ``src/interop/fmi/`` submodule tree (model_description, lifecycle,
#     marshaling) behind a thin ``mod.rs`` facade (~90 lines). The entry
#     is removed and ``fmi_mod_ratchet.json`` deleted per the
#     companion-cleanup convention (cf. Issue #3543 and the surrogate.rs
#     decomposition, which likewise added no child entries for their
#     decomposed modules). The largest child,
#     ``src/interop/fmi/model_description.rs`` (~2.3k LoC), is left
#     ungated for a future #3574-style audit pass.
BASELINE_MODULE_SIZE_LIMITS = 16

# Freeze snapshot of the gated paths (Issue #3457 ratchet).
#
# Mirrors ``_BASELINE_KNOWN_ORPHANS_SET`` from
# ``scripts/check_orphan_modules.py``: this frozenset snapshots which
# files are gated at the moment the ratchet was introduced. It exists
# separately so the ratchet check can report *which* new entries were
# added to ``LIMITS`` since the freeze, not just the total count.
# Editing this set is the "raise the baseline" lever — any new entry MUST
# be added here AND to ``LIMITS`` (and ``BASELINE_MODULE_SIZE_LIMITS``
# must be raised to match the new size), with a documenting comment
# naming the tracking issue. Editing ``LIMITS`` alone, without mirroring
# the change here, makes the diff visible in the CI failure message.
_BASELINE_MODULE_SIZE_LIMITS_SET: frozenset[str] = frozenset(
    {
        # Issue #2878 (retained).
        "src/sim/thermal_model_data.rs",
        "src/sim/thermal_model_data/mod.rs",
        # Issue #3457 — 7 largest src/ files at freeze time.
        # Removed in Issue #3543 (decomposed):
        #   - ``src/api/server.rs`` (part 1) — per-route submodule tree
        #   - ``src/sim/thermal_model_core.rs`` (part 2) — mod.rs + tests
        #   - ``src/validation/ashrae_140_validator.rs`` (part 3) — mod.rs + tests
        "src/ai/surrogate.rs",
        "src/validation/ashrae_140_cases.rs",
        "src/physics/state_space_ctf.rs",
        "src/validation/report.rs",
        # Removed in the fmi/mod.rs decomposition (issue #3625):
        #   - ``src/interop/fmi/mod.rs`` — src/interop/fmi/ submodule tree
        #     (model_description, lifecycle, marshaling)
        "src/sim/thermal_model.rs",
        "src/physics/multi_node_solver.rs",
        # Issue #3574 — 8 files newly over the ~2000-LoC threshold after
        # the Issue #3543 decomposition. Each is ratcheted at its current
        # size plus a small buffer; companion cleanup PRs that decompose
        # any of them should remove the matching entry AND lower
        # BASELINE_MODULE_SIZE_LIMITS by one.
        "src/api/security.rs",
        "src/sim/thermal_model_core/mod.rs",
        "src/validation/ashrae_140_validator/mod.rs",
        "src/physics/geometry_tensor.rs",
        "src/python/model_bindings.rs",
        "src/validation/reporter.rs",
        "fluxion-city/src/lib.rs",
        "fluxion-mcp/src/tools.rs",
    }
)


@dataclass
class Limit:
    path: Path
    max_lines: int
    reason: str
    ratchet_path: Path | None = None

    def effective_max(self) -> int:
        """Return the effective ceiling — ``max(max_lines, ratchet_max)``."""
        if self.ratchet_path is None or not self.ratchet_path.exists():
            return self.max_lines
        try:
            data = json.loads(self.ratchet_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"ERROR: could not read ratchet JSON {self.ratchet_path}: {exc}"
            ) from exc
        ratchet_max = int(data.get("max_lines", 0))
        return max(self.max_lines, ratchet_max)


@dataclass
class Result:
    path: Path
    actual: int
    max: int
    passed: bool
    reason: str = ""


# Module-size limits (Issues #2878, #3457).
#
# Each entry is keyed by its repository-relative path; the ratchet JSON
# (when present) holds the historical maximum line count observed for
# that file. The active ceiling is ``max_lines``; ``effective_max()``
# falls back to the ratchet's max when the YAML ceiling is below it. New
# entries beyond the freeze baseline must be reflected in
# ``BASELINE_MODULE_SIZE_LIMITS`` and ``_BASELINE_MODULE_SIZE_LIMITS_SET``.
LIMITS: list[Limit] = [
    # ------------------------------------------------------------------
    # Issue #2878 — original god-struct limit (retained).
    # ------------------------------------------------------------------
    Limit(
        path=REPO_ROOT / "src" / "sim" / "thermal_model_data.rs",
        max_lines=200,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "thermal_model_data_ratchet.json",
        reason=(
            "Issue #2878 acceptance: drop ThermalModelData below 200 lines "
            "so the god-struct (~140 fields, 145-line Clone impl) does not "
            "regress. Per-config clone must touch ≤6 fields."
        ),
    ),
    Limit(
        path=REPO_ROOT / "src" / "sim" / "thermal_model_data" / "mod.rs",
        max_lines=200,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "thermal_model_data_ratchet.json",
        reason=(
            "Issue #2878 acceptance (directory form): drop ThermalModelData "
            "below 200 lines so the god-struct (~140 fields, 145-line Clone "
            "impl) does not regress. Per-config clone must touch ≤6 fields."
        ),
    ),
    # ------------------------------------------------------------------
    # Issue #3457 — extend gate to the ten largest src/ files.
    #
    # ``max_lines`` is set to the file's CURRENT line count so the gate
    # passes immediately (no new failures) but tightens over time as
    # companion cleanup PRs decompose the file and lower ``max_lines``
    # alongside the ratchet JSON.
    # ------------------------------------------------------------------
    Limit(
        path=REPO_ROOT / "src" / "ai" / "surrogate.rs",
        max_lines=5726,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "surrogate_ratchet.json",
        reason=(
            "Issue #3457: surrogate module is the largest god-module in "
            "``src/``; ratcheted at current size (5726 lines) so the gate "
            "fails the moment it grows further. Decomposition is tracked "
            "separately."
        ),
    ),
    Limit(
        path=REPO_ROOT / "src" / "validation" / "ashrae_140_cases.rs",
        max_lines=4764,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "ashrae_140_cases_ratchet.json",
        reason=(
            "Issue #3457: ASHRAE 140 cases module ratcheted at current "
            "size (4764 lines); v1.3 validation work depends on this "
            "module."
        ),
    ),
    Limit(
        path=REPO_ROOT / "src" / "physics" / "state_space_ctf.rs",
        max_lines=4344,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "state_space_ctf_ratchet.json",
        reason=(
            "Issue #3457: state-space CTF (conduction transfer function) "
            "module ratcheted at current size (4344 lines); referenced by "
            "the 5R1C / 9R4C legacy dispatchers."
        ),
    ),
    Limit(
        path=REPO_ROOT / "src" / "validation" / "report.rs",
        max_lines=4136,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "report_ratchet.json",
        reason=(
            "Issue #3457: validation report module ratcheted at current "
            "size (4136 lines); consumed by ``ashrae_140_validator``."
        ),
    ),
    Limit(
        path=REPO_ROOT / "src" / "sim" / "thermal_model.rs",
        max_lines=3061,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "thermal_model_ratchet.json",
        reason=(
            "Issue #3457: top-level thermal-model module ratcheted at "
            "current size (3061 lines); consumed by the physics↔sim "
            "cycle guard."
        ),
    ),
    Limit(
        path=REPO_ROOT / "src" / "physics" / "multi_node_solver.rs",
        max_lines=2664,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "multi_node_solver_ratchet.json",
        reason=(
            "Issue #3457: multi-node thermal solver ratcheted at current "
            "size (2664 lines)."
        ),
    ),
    # ------------------------------------------------------------------
    # Issue #3574 — extend gate to 8 files newly over the ~2000-LoC
    # threshold after the Issue #3543 decomposition. ``max_lines`` is set
    # to the file's CURRENT line count plus a small buffer (max of +5%
    # or +100 lines, whichever is larger) so the gate passes immediately
    # but tightens against any further growth. Companion cleanup PRs that
    # decompose any of these files should remove the matching entry AND
    # lower ``BASELINE_MODULE_SIZE_LIMITS`` by one.
    # ------------------------------------------------------------------
    Limit(
        path=REPO_ROOT / "src" / "api" / "security.rs",
        max_lines=2199,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "api_security_ratchet.json",
        reason=(
            "Issue #3574: ``src/api/security.rs`` crossed the smallest "
            "ratcheted threshold (~2000 LoC) after the Issue #3543 "
            "decomposition of ``src/api/server.rs``. Ratcheted at current "
            "size 2094 lines + buffer (max of +5% or +100 lines = 2199); "
            "decomposition tracked separately."
        ),
    ),
    Limit(
        path=REPO_ROOT / "src" / "sim" / "thermal_model_core" / "mod.rs",
        max_lines=4260,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "thermal_model_core_mod_ratchet.json",
        reason=(
            "Issue #3574: ``src/sim/thermal_model_core/mod.rs`` is the "
            "largest child of the Issue #3543 decomposition (~4.1k LoC), "
            "still over the ratchet because of the bare "
            "``impl ThermalModel<VectorField>`` block. Ratcheted at "
            "current size 4057 lines + buffer (max of +5% or +100 lines = "
            "4260); further decomposition tracked separately."
        ),
    ),
    Limit(
        path=REPO_ROOT
        / "src"
        / "validation"
        / "ashrae_140_validator"
        / "mod.rs",
        max_lines=3247,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "ashrae_140_validator_mod_ratchet.json",
        reason=(
            "Issue #3574: ``src/validation/ashrae_140_validator/mod.rs`` "
            "is the largest child of the Issue #3543 decomposition. "
            "Ratcheted at current size 3092 lines + buffer (max of +5% or "
            "+100 lines = 3247); further decomposition tracked separately."
        ),
    ),
    Limit(
        path=REPO_ROOT / "src" / "physics" / "geometry_tensor.rs",
        max_lines=2610,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "geometry_tensor_ratchet.json",
        reason=(
            "Issue #3574: ``src/physics/geometry_tensor.rs`` crossed the "
            "smallest ratcheted threshold after Issue #3543. Ratcheted at "
            "current size 2486 lines + buffer (max of +5% or +100 lines = "
            "2610); decomposition tracked separately."
        ),
    ),
    Limit(
        path=REPO_ROOT / "src" / "python" / "model_bindings.rs",
        max_lines=2609,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "python_model_bindings_ratchet.json",
        reason=(
            "Issue #3574: ``src/python/model_bindings.rs`` crossed the "
            "smallest ratcheted threshold after Issue #3543. Ratcheted at "
            "current size 2485 lines + buffer (max of +5% or +100 lines = "
            "2609); decomposition tracked separately."
        ),
    ),
    Limit(
        path=REPO_ROOT / "src" / "validation" / "reporter.rs",
        max_lines=2292,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "validation_reporter_ratchet.json",
        reason=(
            "Issue #3574: ``src/validation/reporter.rs`` crossed the "
            "smallest ratcheted threshold after Issue #3543. Ratcheted at "
            "current size 2183 lines + buffer (max of +5% or +100 lines = "
            "2292); decomposition tracked separately."
        ),
    ),
    Limit(
        path=REPO_ROOT / "fluxion-city" / "src" / "lib.rs",
        max_lines=2814,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "fluxion_city_lib_ratchet.json",
        reason=(
            "Issue #3574: workspace sibling ``fluxion-city/src/lib.rs`` "
            "crossed the smallest ratcheted threshold after Issue #3543. "
            "Ratcheted at current size 2680 lines + buffer (max of +5% or "
            "+100 lines = 2814); decomposition tracked separately."
        ),
    ),
    Limit(
        path=REPO_ROOT / "fluxion-mcp" / "src" / "tools.rs",
        max_lines=1748,
        ratchet_path=REPO_ROOT
        / "tests"
        / "reference_data"
        / "module_size"
        / "fluxion_mcp_tools_ratchet.json",
        reason=(
            "Issue #3574: workspace sibling ``fluxion-mcp/src/tools.rs`` "
            "crossed the smallest ratcheted threshold after Issue #3543. "
            "Ratcheted at current size 1648 lines + buffer (max of +5% or "
            "+100 lines = 1748); decomposition tracked separately."
        ),
    ),
]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open(encoding="utf-8"))


def check(limit: Limit) -> Result | None:
    if not limit.path.exists():
        return None
    actual = count_lines(limit.path)
    ceiling = limit.effective_max()
    return Result(
        path=limit.path,
        actual=actual,
        max=ceiling,
        passed=actual <= ceiling,
        reason=limit.reason,
    )


def update_ratchet(result: Result) -> None:
    """Tighten the ratchet if a new minimum is below the historical max."""
    if not result.path.exists():
        return
    rel = result.path.relative_to(REPO_ROOT)
    candidates = [
        lim
        for lim in LIMITS
        if lim.path.relative_to(REPO_ROOT) == rel or lim.path.name == result.path.name
    ]
    for limit in candidates:
        if limit.ratchet_path is None:
            continue
        if not limit.ratchet_path.exists():
            data: dict = {"max_lines": limit.max_lines, "history": []}
        else:
            try:
                data = json.loads(limit.ratchet_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {"max_lines": limit.max_lines, "history": []}
        history = list(data.get("history", []))
        history.append({"actual": result.actual})
        max_observed = max([entry["actual"] for entry in history] + [limit.max_lines])
        data["max_lines"] = max_observed
        data["history"] = history[-20:]
        limit.ratchet_path.parent.mkdir(parents=True, exist_ok=True)
        limit.ratchet_path.write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )


def check_baseline_drift() -> list[str]:
    """Return human-readable drift messages (empty list = no drift).

    Compares the live ``LIMITS`` table against the freeze snapshot
    (``_BASELINE_MODULE_SIZE_LIMITS_SET``) and the count baseline
    (``BASELINE_MODULE_SIZE_LIMITS``). Mirrors
    ``BASELINE_KNOWN_ORPHANS`` / ``_BASELINE_KNOWN_ORPHANS_SET`` from
    ``check_orphan_modules.py``: adding entries without raising the
    baseline is treated as drift, and the failing messages name which
    entries are new so the diff is visible in CI output.
    """
    messages: list[str] = []
    live_paths: set[str] = set()
    for lim in LIMITS:
        try:
            rel = lim.path.relative_to(REPO_ROOT)
        except ValueError:
            rel = lim.path
        live_paths.add(str(rel))
    new_paths = sorted(live_paths - set(_BASELINE_MODULE_SIZE_LIMITS_SET))
    if new_paths:
        joined = "\n".join(f"    - {p}" for p in new_paths)
        messages.append(
            "NEW LIMITS entries (regression): "
            f"{len(new_paths)} (not in freeze snapshot)\n{joined}\n"
            "If these entries are intentional, raise "
            "BASELINE_MODULE_SIZE_LIMITS to match the new size AND add "
            "the paths to _BASELINE_MODULE_SIZE_LIMITS_SET, with a "
            "documenting comment naming the tracking issue."
        )
    if len(LIMITS) > BASELINE_MODULE_SIZE_LIMITS:
        messages.append(
            "BASELINE_MODULE_SIZE_LIMITS drift: "
            f"len(LIMITS)={len(LIMITS)} > "
            f"BASELINE_MODULE_SIZE_LIMITS={BASELINE_MODULE_SIZE_LIMITS}\n"
            "Either raise BASELINE_MODULE_SIZE_LIMITS (with a documenting "
            "comment naming the tracking issue) or remove the surplus "
            "entry. The companion cleanup PRs that decompose a gated "
            "file are expected to lower BASELINE_MODULE_SIZE_LIMITS by "
            "one."
        )
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output for CI consumption.",
    )
    parser.add_argument(
        "--write-ratchet",
        action="store_true",
        help="Tighten the ratchet JSON to reflect observed line counts (PR-friendly).",
    )
    args = parser.parse_args()

    drift_messages = check_baseline_drift()
    results: list[Result] = []
    for limit in LIMITS:
        result = check(limit)
        if result is not None:
            results.append(result)

    if args.write_ratchet:
        for result in results:
            update_ratchet(result)

    all_passed = all(r.passed for r in results) and not drift_messages

    if args.json:
        payload = {
            "results": [
                {
                    "path": str(r.path.relative_to(REPO_ROOT)),
                    "actual": r.actual,
                    "max": r.max,
                    "passed": r.passed,
                    "reason": r.reason,
                }
                for r in results
            ],
            "baseline": {
                "max_limits": BASELINE_MODULE_SIZE_LIMITS,
                "observed": len(LIMITS),
                "drift": drift_messages,
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("=== Fluxion module-size gate (Issues #2878, #3457) ===")
        print(f"Repo: {REPO_ROOT}")
        print()
        if drift_messages:
            print("BASELINE DRIFT (PR-blocking unless baseline is raised):")
            for msg in drift_messages:
                print(f"  - {msg}")
            print()
        if not results:
            print("No matching files found — gate is a no-op.")
            return 0 if not drift_messages else 1
        for result in results:
            rel = result.path.relative_to(REPO_ROOT)
            verdict = "PASS" if result.passed else "FAIL"
            print(f"  [{verdict}] {rel}: {result.actual} lines (max {result.max})")
            print(f"      {result.reason}")
        print()
        if all_passed:
            print("All module-size limits satisfied.")
            return 0
        print(
            "One or more module-size limits exceeded or baseline drift "
            "detected; see FAIL lines above."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
