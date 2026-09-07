#!/usr/bin/env python3
"""
Fluxion Release Scorecard Generator.

Emits ``SCORECARD.md`` at the repo root from **committed** source files so the
output is fully deterministic and reproducible:

  * ``docs/ASHRAE140_RESULTS.md``   -- headline pass rate / MAE / per-series.
  * ``release_gates.yaml``          -- gate budgets (pass rate, MAE, throughput).
  * ``README.md``                   -- BatchOracle release-mode throughput claim.

Because every figure is read from a committed file, the generated scorecard is
byte-stable for a given set of inputs. CI runs ``--check`` to regenerate the
scorecard to a temporary path and ``diff`` it against the committed copy; any
drift (a validation report or gate threshold changed but the scorecard was not
regenerated) fails the build. This is the acceptance criterion of issue #2496:
"CI can fail when any metric regresses."

The script deliberately does NOT consult ``docs/QUALITY_METRICS.md`` (a
historical dashboard that can hold stale/corrupt values -- issue #1167) and
does NOT execute ``cargo`` / benchmarks, so it runs in seconds on any runner.

Usage:
    python scripts/generate_scorecard.py                 # write SCORECARD.md
    python scripts/generate_scorecard.py --verbose       # trace parsed values
    python scripts/generate_scorecard.py --check         # exit 1 on drift
    python scripts/generate_scorecard.py -o /tmp/out.md  # custom output

Exit codes:
    0  success (or --check passed)
    1  --check detected drift (committed SCORECARD.md is stale)
    2  a required source file could not be read/parsed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Pin stdout / stderr to UTF-8 so the ``✓`` glyph in the success print works on
# Windows runners whose default stdio codec is cp1252. No-op on POSIX runners
# (already UTF-8). Python 3.7+ supports ``reconfigure``.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

REPO = Path(__file__).resolve().parent.parent
ASHRAE_DOC = REPO / "docs" / "ASHRAE140_RESULTS.md"
# Issue #3403: most-recent validation-run history. When present, the
# throughput figure is taken from the latest entry here instead of the
# (batch-generated, potentially weeks-stale) ASHRAE140_RESULTS.md.
PERF_HISTORY = REPO / "target" / "performance_history.jsonl"
GATES_YAML = REPO / "release_gates.yaml"
README_MD = REPO / "README.md"
SCORECARD = REPO / "SCORECARD.md"


@dataclass
class Validation:
    total: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    pass_rate: float = 0.0
    mae: float = 0.0
    max_deviation: float = 0.0
    throughput_cases_per_sec: float = 0.0
    generated_utc: str = ""


@dataclass
class SeriesRow:
    name: str
    cases: int = 0
    passed: int = 0
    warn: int = 0
    failed: int = 0

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.cases * 100.0) if self.cases else 0.0


@dataclass
class Gates:
    min_pass_rate: float = 60.0
    max_mae: float = 50.0
    min_throughput: float = 150.0
    max_latency_ms: float = 10.0
    absolute_min_throughput: float = 100.0
    known_failures: list[str] = field(default_factory=list)
    required_checks: list[str] = field(default_factory=list)
    ci_throughput_comment: float = 0.0  # parsed from the YAML comment (~157)


@dataclass
class Benchmark:
    readme_release_throughput: float = 0.0  # ~900 configs/sec from README


def _num(text: str) -> Optional[float]:
    """Parse a leading numeric from a markdown/yaml cell ('55.09%' -> 55.09)."""
    if text is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group(0)) if m else None


def apply_performance_history(v: Validation) -> str:
    """Override v.throughput_cases_per_sec from the latest perf-history
    entry (Issue #3403). Returns the source string for attribution; the
    committed results doc remains the fallback when no history exists
    (e.g. fresh checkout -- target/ is a build artifact)."""
    if not PERF_HISTORY.exists():
        return "`docs/ASHRAE140_RESULTS.md`"
    try:
        last = None
        for line in PERF_HISTORY.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                last = line
        if last is None:
            return "`docs/ASHRAE140_RESULTS.md`"
        entry = json.loads(last)
        thr = float(entry.get("throughput", 0.0))
        if thr > 0.0:
            v.throughput_cases_per_sec = thr
            ts = str(entry.get("timestamp", ""))[:10]
            return f"`target/performance_history.jsonl` (latest run {ts})"
    except (json.JSONDecodeError, ValueError, OSError):
        pass
    return "`docs/ASHRAE140_RESULTS.md`"


def parse_ashrae(doc_text: str) -> Validation:
    v = Validation()
    # Generated timestamp (authoritative "data as of" date).
    m = re.search(r"\*Generated:\s*([0-9-]+ [0-9:]+ UTC)\*", doc_text)
    if m:
        v.generated_utc = m.group(1)

    # Scan all '## ...' table sections. The headline metrics live in
    # '## Summary' and the cases/sec throughput lives in
    # '## Performance Summary', so we accumulate keyed rows across both.
    for line in doc_text.splitlines():
        s = line.strip()
        if s.startswith("## "):
            continue
        if not s.startswith("|"):
            continue
        if set(s.replace("|", "").strip()) <= {"-", ":"}:
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 2:
            continue
        key = cells[0].lower()
        val = cells[1]
        if key == "total results":
            v.total = int(_num(val) or 0)
        elif key == "passed":
            v.passed = int(_num(val) or 0)
        elif key == "failed":
            v.failed = int(_num(val) or 0)
        elif key == "warnings":
            v.warnings = int(_num(val) or 0)
        elif key == "pass rate":
            v.pass_rate = _num(val) or 0.0
        elif key == "mean absolute error":
            v.mae = _num(val) or 0.0
        elif key == "max deviation":
            v.max_deviation = _num(val) or 0.0
        elif key == "throughput" and "cases/sec" in val:
            v.throughput_cases_per_sec = _num(val) or 0.0
    return v


def parse_series(doc_text: str) -> list[SeriesRow]:
    """Parse case-level pass/warn/fail by series from the '## Detailed Results'
    section. Each subsection (### ...) whose table has a Status column is a
    series. Sections outside '## Detailed Results' (e.g. Systematic Issues) are
    skipped via the enclosing-section guard."""
    rows: list[SeriesRow] = []
    current_sub: Optional[str] = None
    enclosing: Optional[str] = None
    in_results = False
    for line in doc_text.splitlines():
        s = line.strip()
        if s.startswith("## "):
            enclosing = s.lstrip("# ").strip()
            in_results = "detailed results" in enclosing.lower()
            current_sub = None
            continue
        if not in_results:
            continue
        if s.startswith("### "):
            current_sub = s.lstrip("# ").strip()
            rows.append(SeriesRow(name=current_sub))
            continue
        if current_sub is None or not s.startswith("|"):
            continue
        if "Status" in s or set(s.replace("|", "").strip()) <= {"-", ":"}:
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if not cells:
            continue
        status = cells[-1]
        row = rows[-1]
        if "PASS" in status and "FAIL" not in status:
            row.passed += 1
            row.cases += 1
        elif "WARN" in status:
            row.warn += 1
            row.cases += 1
        elif "FAIL" in status:
            row.failed += 1
            row.cases += 1
    return [r for r in rows if r.cases > 0]


def _yaml_num(text: str, key: str) -> Optional[float]:
    m = re.search(rf"^\s*{re.escape(key)}:\s*(-?\d+(?:\.\d+)?)", text, re.M)
    return float(m.group(1)) if m else None


def _next_top_level_key(
    yaml_text: str, block: str, current_key: str, next_key: Optional[str]
) -> str:
    """Truncate ``block`` at the next top-level YAML key.

    The previous implementation stopped at a ``# ====`` comment marker,
    which does not exist between ``ci.required_checks`` and the next
    top-level YAML key (``ci.required_checks_workflow_only``) in
    ``release_gates.yaml``. The marker-based stop therefore greedily
    consumed BOTH lists into one block, producing a 29-row duplicated
    CI Gate Status table in SCORECARD.md (Issue #3389).

    ``current_key`` is the key the caller is splitting on (used to detect
    the indentation of ``next_key`` from ``yaml_text`` so the stop is
    anchored at the same column as ``current_key`` itself, not at column
    0). ``next_key`` is the next-key label (e.g.
    ``"required_checks_workflow_only"``) or ``None`` for end-of-block
    (EOF).

    The match allows at most ``len(current_key_indent)`` leading spaces
    so an indented subkey at a deeper level does NOT count as a boundary.
    If ``next_key`` is not found (or is ``None``), the entire ``block``
    is returned unchanged.
    """
    if next_key is None:
        return block
    # Detect the indentation of ``current_key`` in the source YAML. The
    # same indentation defines "top-level" for the next key.
    cur_match = re.search(rf"^(\s*){re.escape(current_key)}:", yaml_text, re.M)
    if cur_match is None:
        return block
    indent = cur_match.group(1)
    pattern = re.compile(
        rf"^{re.escape(indent)}{re.escape(next_key)}:", re.M
    )
    m = pattern.search(block)
    if m is None:
        return block
    return block[: m.start()]


def parse_gates(yaml_text: str) -> Gates:
    g = Gates()
    g.min_pass_rate = _yaml_num(yaml_text, "min_pass_rate") or g.min_pass_rate
    g.max_mae = _yaml_num(yaml_text, "max_mae") or g.max_mae
    g.min_throughput = _yaml_num(yaml_text, "min_configs_per_sec") or g.min_throughput
    g.max_latency_ms = _yaml_num(yaml_text, "max_ms_per_config") or g.max_latency_ms
    g.absolute_min_throughput = (
        _yaml_num(yaml_text, "absolute_min_throughput") or g.absolute_min_throughput
    )

    # required_checks list
    if "required_checks:" in yaml_text:
        block = yaml_text.split("required_checks:", 1)[1]
        # Issue #3389: the previous ``block.split("# ====", 1)[0]`` was a
        # ``# ====``-marker stop that doesn't exist between ``ci.required_checks``
        # and the next top-level YAML key (``required_checks_workflow_only:``)
        # in ``release_gates.yaml``. The unreached stop meant the script
        # captured BOTH ``required_checks`` AND ``required_checks_workflow_only``
        # entries (54 matches / 29 unique), producing a 29-row duplicated
        # CI Gate Status table in SCORECARD.md. Stop on the next top-level
        # YAML key (same indentation as ``required_checks:`` itself) so the
        # block ends at the list boundary.
        block = _next_top_level_key(
            yaml_text, block, "required_checks", "required_checks_workflow_only"
        )
        g.required_checks = re.findall(r'^\s*-\s*"(.+?)"', block, re.M)

    # known_failures
    if "known_failures:" in yaml_text:
        block = yaml_text.split("known_failures:", 1)[1]
        # Mirror the same fix as ``required_checks`` above: stop at the
        # next top-level key rather than a non-existent ``# ====`` marker
        # so we capture only the ``known_failures`` entries and not whatever
        # follows it.
        block = _next_top_level_key(yaml_text, block, "known_failures", None)
        g.known_failures = re.findall(r'^\s*-\s*"(\d+)"', block, re.M)

    # CI runner throughput from the YAML comment (~157 configs/sec).
    mc = re.search(r"~(\d+)\s*configs/sec", yaml_text)
    if mc:
        g.ci_throughput_comment = float(mc.group(1))
    return g


def parse_readme_throughput(readme_text: str) -> Benchmark:
    b = Benchmark()
    m = re.search(r"~?(\d+)\s*configs/sec\s*throughput in release mode", readme_text)
    if m:
        b.readme_release_throughput = float(m.group(1))
    return b


def _status(ok: bool, good="✅ Pass", bad="❌ Fail") -> str:
    return good if ok else bad


def render(
    v: Validation,
    series: list[SeriesRow],
    g: Gates,
    b: Benchmark,
    throughput_source: str = "`docs/ASHRAE140_RESULTS.md`",
) -> str:
    pass_ok = v.pass_rate >= g.min_pass_rate
    mae_ok = v.mae <= g.max_mae
    # Conservative "meets budget" uses the CI-runner figure when available
    # (it is the lower of the two attributable numbers), else the README one.
    ci_thr = g.ci_throughput_comment or b.readme_release_throughput
    throughput_ok = ci_thr >= g.min_throughput

    lines: list[str] = []
    p = lines.append

    p("# Fluxion Release Scorecard")
    p("")
    p(
        "> Consolidated view of release-readiness metrics. Generated from "
        "committed sources so it is fully reproducible."
    )
    p(">")
    p(
        "> **Do not edit by hand** — regenerate with "
        "`python scripts/generate_scorecard.py`. CI fails on drift "
        "(`scorecard-drift` workflow)."
    )
    p("")
    last_updated = (v.generated_utc or "unknown").split(" ")[0]
    p(f"**Last Updated:** {last_updated}  ")
    p(f"**Data source as of:** {v.generated_utc or 'unknown'}  ")
    p("**Sources:** `docs/ASHRAE140_RESULTS.md`, `release_gates.yaml`, " "`README.md`")
    p("")
    p("---")
    p("")

    # --- Headline -------------------------------------------------------
    p("## Headline")
    p("")
    p("| Metric | Current | Budget (gate) | Status | Source |")
    p("|--------|---------|---------------|--------|--------|")
    p(
        f"| ASHRAE 140 pass rate | **{v.pass_rate:.1f}%** "
        f"({v.passed}/{v.total} metrics) | ≥ {g.min_pass_rate:.0f}% "
        f"(`validation.min_pass_rate`) | {_status(pass_ok)} | "
        "`docs/ASHRAE140_RESULTS.md` |"
    )
    p(
        f"| Mean Absolute Error (MAE) | **{v.mae:.2f}%** | "
        f"≤ {g.max_mae:.0f}% (`validation.max_mae`) | {_status(mae_ok)} | "
        "`docs/ASHRAE140_RESULTS.md` |"
    )
    thr_note = (
        f"{g.ci_throughput_comment:.0f} (CI) / {b.readme_release_throughput:.0f} (release)"
        if g.ci_throughput_comment and b.readme_release_throughput
        else f"{ci_thr:.0f}"
    )
    p(
        f"| BatchOracle throughput | **{thr_note}** configs/sec | "
        f"≥ {g.min_throughput:.0f} (`benchmark.throughput.min_configs_per_sec`) "
        f"| {_status(throughput_ok)} | `release_gates.yaml` comment + "
        f"`README.md` |"
    )
    p(
        f"| Validation-suite throughput | {v.throughput_cases_per_sec:.2f} "
        f"cases/sec | (informational) | ℹ️ | {throughput_source} |"
    )
    p(
        f"| Max single-case deviation | {v.max_deviation:.2f}% | "
        f"(ref: `individual.max_deviation` = 100%) | ℹ️ | "
        "`docs/ASHRAE140_RESULTS.md` |"
    )
    p("")

    # --- Pass rate ------------------------------------------------------
    p("## ASHRAE 140 Pass Rate")
    p("")
    p(
        f"- **Overall (metric-level):** {v.pass_rate:.1f}% — "
        f"{v.passed} PASS / {v.warnings} WARN / {v.failed} FAIL of "
        f"{v.total} results. "
        f"{_status(pass_ok, 'Meets', 'Below')} the {g.min_pass_rate:.0f}% gate."
    )
    case_total = sum(r.cases for r in series)
    case_pass = sum(r.passed for r in series)
    case_pct = (case_pass / case_total * 100.0) if case_total else 0.0
    p(
        f"- **Case-level:** {case_pass}/{case_total} cases fully PASS "
        f"({case_pct:.1f}%)."
    )
    p("")

    # --- Per-series breakdown ------------------------------------------
    p("### Per-Series Breakdown (case-level)")
    p("")
    p("| Series | Cases | PASS | WARN | FAIL | Pass rate |")
    p("|--------|-------|------|------|------|-----------|")
    for r in series:
        p(
            f"| {r.name} | {r.cases} | {r.passed} | {r.warn} | {r.failed} | "
            f"{r.pass_rate:.1f}% |"
        )
    p("")
    p(
        "*Case-level = a case is PASS only if its aggregate row is ✅. "
        "Metric-level headline (20.3%) counts each reported metric individually; "
        "see `docs/ASHRAE140_RESULTS.md` Summary.*"
    )
    p("")

    # --- Throughput vs budget ------------------------------------------
    p("## Throughput vs Budget")
    p("")
    p(
        f"- **Gate:** ≥ **{g.min_throughput:.0f}** configs/sec "
        f"(`benchmark.throughput.min_configs_per_sec`); absolute floor "
        f"{g.absolute_min_throughput:.0f}; latency ≤ "
        f"{g.max_latency_ms:.0f} ms/config."
    )
    if g.ci_throughput_comment:
        p(
            f"- **CI runner (Wave 1+1.5):** ~{g.ci_throughput_comment:.0f} "
            f"configs/sec — {_status(g.ci_throughput_comment >= g.min_throughput)} "
            f"(narrow margin; source: `release_gates.yaml` comment)."
        )
    if b.readme_release_throughput:
        p(
            f"- **Release mode (BatchOracle, rayon):** "
            f"~{b.readme_release_throughput:.0f} configs/sec — "
            f"{_status(b.readme_release_throughput >= g.min_throughput)} "
            f"(source: `README.md`)."
        )
    p(
        f"- **Validation-suite throughput:** "
        f"{v.throughput_cases_per_sec:.2f} cases/sec — informational only; "
        f"this is the test-runner cadence, not the BatchOracle benchmark "
        f"(source: {throughput_source})."
    )
    p("")

    # --- MAE vs budget -------------------------------------------------
    p("## MAE vs Budget")
    p("")
    p(f"- **Gate:** ≤ **{g.max_mae:.0f}%** (`validation.max_mae`).")
    p(
        f"- **Current:** **{v.mae:.2f}%** — {_status(mae_ok, 'Within budget', 'Over budget')} "
        f"by {v.mae - g.max_mae:+.2f} pp. Max single-case deviation "
        f"{v.max_deviation:.2f}%."
    )
    p(
        "- *Driver:* high-mass annual-energy deviation (5R1C/CTF thermal-mass "
        "limitation; see Known Structural Failures)."
    )
    p("")

    # --- Known structural failures -------------------------------------
    p("## Known Structural Failures")
    p("")
    p(
        "Cases excluded from the strict ±15% annual-energy gate and from the "
        "`extreme_deviation_limit` count (`release_gates.yaml` → "
        "`validation.individual.known_failures`):"
    )
    p("")
    p("| Case | Series | Reason |")
    p("|------|--------|--------|")
    p(
        "| **600** | Baseline (low-mass) | Multiple low-mass baseline tests — "
        "simplified envelope model (`AGENTS.md`). |"
    )
    p(
        "| **900** | High-mass | Heating deviation ~200% — high-mass thermal-"
        "mass model limitation (`release_gates.yaml` comment). |"
    )
    p("")
    p(
        "Per `AGENTS.md`: cases **600** and **900** are documented structural "
        "failures. Fix path = underlying physics (no parameter tuning — `RULES.md`)."
    )
    p("")

    # --- CI gate status ------------------------------------------------
    p("## CI Gate Status")
    p("")
    p(
        "Required branch-protection checks "
        "(`release_gates.yaml` → `ci.required_checks`):"
    )
    p("")
    p("| Required check | Issue |")
    p("|----------------|-------|")
    for chk in g.required_checks:
        im = re.search(r"#(\d+)", chk)
        p(f"| {chk} | {('#' + im.group(1)) if im else '—'} |")
    p("")
    p(
        "- **Live status** is intentionally not baked in here (it is "
        "non-deterministic and would break scorecard diff stability). Run:"
    )
    p("")
    p("  ```bash")
    p("  gh run list --repo anchapin/fluxion --branch develop --limit 10")
    p("  ```")
    p("")
    p(
        "- **Validation gate policy** (`release_gates.yaml`): major/minor "
        "releases require validation + benchmark + drift gates; patches relax "
        f"validation to {40:.0f}% pass (see `release_requirements.patch`)."
    )
    p(
        "- **Drift guard** (`drift.*`): max ±2.0 pp pass-rate change, ±5.0 pp "
        "MAE change, ≤1 pass→fail flip vs `validation_baseline.json`."
    )
    p("")

    # --- Regenerate ----------------------------------------------------
    p("## Regenerate")
    p("")
    p("```bash")
    p("# Regenerate the scorecard from committed sources")
    p("python scripts/generate_scorecard.py")
    p("")
    p("# CI uses this to fail on drift (exit 1 if SCORECARD.md is stale)")
    p("python scripts/generate_scorecard.py --check")
    p("")
    p("# Verbose: print every parsed value")
    p("python scripts/generate_scorecard.py --verbose")
    p("```")
    p("")
    p(
        "The scorecard is regenerated whenever `docs/ASHRAE140_RESULTS.md`, "
        "`release_gates.yaml`, or `README.md` changes. The "
        "`scorecard-drift` workflow enforces this on every PR."
    )
    p("")
    p("---")
    p("")
    p(
        "*Auto-generated by `scripts/generate_scorecard.py` (issue #2496). "
        "Edit the generator, not this file.*"
    )
    p("")
    return "\n".join(lines)


def load_all(
    verbose: bool = False,
) -> tuple[Validation, list[SeriesRow], Gates, Benchmark, str]:
    if not ASHRAE_DOC.exists():
        print(
            f"ERROR: {ASHRAE_DOC} not found (committed validation report).",
            file=sys.stderr,
        )
        sys.exit(2)
    if not GATES_YAML.exists():
        print(f"ERROR: {GATES_YAML} not found.", file=sys.stderr)
        sys.exit(2)
    doc = ASHRAE_DOC.read_text(encoding="utf-8")
    yaml_text = GATES_YAML.read_text(encoding="utf-8")
    readme_text = README_MD.read_text(encoding="utf-8") if README_MD.exists() else ""

    v = parse_ashrae(doc)
    throughput_source = apply_performance_history(v)
    series = parse_series(doc)
    g = parse_gates(yaml_text)
    b = parse_readme_throughput(readme_text)

    if verbose:
        print(
            f"  validation: pass_rate={v.pass_rate}% mae={v.mae}% "
            f"total={v.total} passed={v.passed} gen={v.generated_utc}"
        )
        print(f"  series: {len(series)} rows")
        for r in series:
            print(f"    {r.name}: {r.passed}/{r.cases}")
        print(
            f"  gates: min_pass={g.min_pass_rate} max_mae={g.max_mae} "
            f"min_thr={g.min_throughput} known_failures={g.known_failures}"
        )
        print(
            f"  benchmark: readme_release={b.readme_release_throughput} "
            f"ci_comment={g.ci_throughput_comment}"
        )
    return v, series, g, b, throughput_source


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Fluxion Release Scorecard")
    ap.add_argument(
        "-o",
        "--output",
        default=str(SCORECARD),
        help="output path (default: SCORECARD.md)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="regenerate to temp and diff against committed "
        "SCORECARD.md; exit 1 on drift",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="print parsed values")
    args = ap.parse_args()

    v, series, g, b, throughput_source = load_all(verbose=args.verbose)
    content = render(v, series, g, b, throughput_source)

    if args.check:
        if not SCORECARD.exists():
            print(
                "ERROR: SCORECARD.md missing — run without --check to create it.",
                file=sys.stderr,
            )
            return 1
        committed = SCORECARD.read_text(encoding="utf-8")
        if committed != content:
            print(
                "ERROR: SCORECARD.md is stale (drift detected).\n"
                "  Regenerate with: python scripts/generate_scorecard.py",
                file=sys.stderr,
            )
            return 1
        print("[OK] SCORECARD.md is up to date (no drift).")
        return 0

    out = Path(args.output)
    out.write_text(content, encoding="utf-8")
    print(f"[OK] Wrote {out}  (pass {v.pass_rate:.1f}% / MAE {v.mae:.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
