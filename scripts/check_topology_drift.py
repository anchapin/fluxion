#!/usr/bin/env python3
"""Topology diagram drift gate — Issue #3966.

Regenerates every artifact produced by ``scripts/generate_topology_diagrams.py``
into a temp directory and byte-compares the result against the committed tree
(``tests/reference_data/topology/**`` and ``docs/architecture/topology/**``).
Also re-runs ``fluxion topology lint --strict`` for every registry case via the
generator (any non-zero lint exit fails the gate).

Exit codes: 0 = no drift, 1 = drift or lint failure, 2 = environment error
(missing fluxion CLI).

Wired into ``.github/workflows/topology_visualizer.yml``. Contributors who see
this fail should run, then commit the results in the same PR:

    python3 scripts/generate_topology_diagrams.py
"""

from __future__ import annotations

import argparse
import filecmp
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_topology_diagrams as gen  # noqa: E402

MANAGED_TREES = [gen.REFERENCE_DIR, gen.DOCS_DIR]


def collect_files(root: Path, subtree: Path) -> set[Path]:
    base = root / subtree
    if not base.is_dir():
        return set()
    return {p.relative_to(root) for p in base.rglob("*") if p.is_file()}


def check_drift(repo_root: Path, fluxion_bin: str) -> tuple[int, list[str], list[dict]]:
    """Returns (exit_code, report_lines, case_index_entries)."""
    report: list[str] = []
    with tempfile.TemporaryDirectory(prefix="fluxion-topology-drift-") as td:
        tmp_root = Path(td)
        gen_rc = gen.generate(tmp_root, fluxion_bin)
        expected: set[Path] = set()
        for tree in MANAGED_TREES:
            expected |= collect_files(tmp_root, tree)
        committed: set[Path] = set()
        for tree in MANAGED_TREES:
            committed |= collect_files(repo_root, tree)

        missing = sorted(expected - committed)
        extra = sorted(committed - expected)
        changed: list[Path] = []
        for rel in sorted(expected & committed):
            if not filecmp.cmp(repo_root / rel, tmp_root / rel, shallow=False):
                changed.append(rel)

        if missing:
            report.append(f"missing committed artifacts ({len(missing)}):")
            report += [f"  - {rel}" for rel in missing]
        if extra:
            report.append(f"stale committed artifacts not produced by the generator ({len(extra)}):")
            report += [f"  - {rel}" for rel in extra]
        if changed:
            report.append(f"drifted artifacts ({len(changed)}):")
            report += [f"  - {rel}" for rel in changed]

        drift = bool(missing or extra or changed)
        entries: list[dict] = []
        index_path = tmp_root / gen.REFERENCE_DIR / "index.json"
        if index_path.is_file():
            import json

            entries = json.loads(index_path.read_text(encoding="utf-8")).get("cases", [])

        if drift:
            report.append(
                "topology artifacts are out of date: regenerate with "
                "`python3 scripts/generate_topology_diagrams.py` (requires `cargo build -p fluxion`) "
                "and commit the results in the same PR. Do not hand-edit generated files."
            )
        if gen_rc != 0:
            report.append("topology lint --strict reported failures for at least one case (see above).")
        return (1 if (drift or gen_rc != 0) else 0), report, entries


def write_summary(path: str, exit_code: int, report: list[str], entries: list[dict]) -> None:
    lines = [
        "## Topology Diagram Drift Gate (Issue #3966)",
        "",
        "**Verdict:** " + ("✅ no drift — references, diagrams, and lint reports all match" if exit_code == 0 else "❌ drift detected"),
        "",
        "| Case | Model | Nodes | Couplings | Lint (strict) |",
        "|---|---|---|---|---|",
    ]
    for e in entries:
        lint = e.get("lint", {})
        lint_cell = "✅ clean" if lint.get("clean") else "❌ failures"
        lines.append(
            f"| {e.get('case')} | {e.get('model_name')} | {e.get('node_count')} | "
            f"{e.get('edge_count')} | {lint_cell} |"
        )
    if report:
        lines += ["", "<details><summary>Gate output</summary>", "", "```"]
        lines += [line for line in report]
        lines += ["```", "", "</details>"]
    lines.append("")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Byte-compare committed topology artifacts against regenerated ones (issue #3966)."
    )
    parser.add_argument(
        "--write-summary",
        default=None,
        help="append a markdown summary to this file (e.g. $GITHUB_STEP_SUMMARY)",
    )
    args = parser.parse_args(argv)

    repo_root = gen.REPO_ROOT
    try:
        fluxion_bin = gen.locate_fluxion(None)
    except SystemExit:
        if args.write_summary and os.environ.get("GITHUB_STEP_SUMMARY"):
            write_summary(
                os.environ["GITHUB_STEP_SUMMARY"], 2, ["environment error: fluxion CLI unavailable"], []
            )
        raise

    exit_code, report, entries = check_drift(repo_root, fluxion_bin)
    for line in report:
        print(line)
    if exit_code == 0:
        print(f"topology drift gate: OK ({len(entries)} cases, byte-identical)")
    else:
        print("topology drift gate: FAILED", file=sys.stderr)
    if args.write_summary:
        write_summary(args.write_summary, exit_code, report, entries)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
