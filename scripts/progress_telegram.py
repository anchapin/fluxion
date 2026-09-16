#!/usr/bin/env python3
"""
scripts/progress_telegram.py — weekly v1.3 progress telegram (plan
pm-throughput-20260915 task T5; ADR-0016 companion; tracker issue #3804).

Emits the weekly "progress telegram" (see .planning/CONTEXT.md glossary)
as GitHub-flavoured markdown for a comment on the pinned v1.3 progress
tracker issue, following the stateless β-soak pattern
(.github/workflows/nightly-ashrae-140-gauge.yml): no state is committed
back to the repo. Week-over-week deltas are derived by parsing the
machine-readable marker embedded in the *previous* telegram comment on
the tracker issue — if none exists yet (first week), deltas render as
``n/a``.

Metrics (each with week-over-week delta where derivable):

  1. ASHRAE 140 metric pass rate + cases fully passing
     (canonical generated sources: validation/performance_history.latest.json
     for the pass rate / MAE; SCORECARD.md for the metric-count breakdown
     and the case-level summary — never hand-copied stale figures).
  2. milestone-labelled merges vs hygiene merges
     (gh pr list --state merged, label:milestone filter, same window).
  3. cycle-time p50 per lane — hygiene PRs (no physics-path files) vs
     physics PRs — from PR createdAt→mergedAt + files. Physics path class
     per ADR-0016: src/**, fluxion-core/**, fluxion-fluid/**, tests/**,
     models/**, weather/**.
  4. fix-loop rate — merged PRs with >1 commit in the same window
     (pre-push-checklist.md fix-push etiquette telemetry).
  5. cancelled-run share — gh run list conclusion=cancelled / completed,
     same window (ADR-0015 cancel-storm telemetry).
  6. LIMIT-gap count — distinct LIMIT-NN ids under ``###``-style
     headings in docs/KNOWN_ISSUES.md (UPDATE headings under the same
     LIMIT id do not count as separate gaps).

Only stdlib + the ``gh`` CLI (read-only data collection, one comment
write when posting). Matches the scripts/check_*.py house style.

Usage:
    python3 scripts/progress_telegram.py --weeks 1 --output markdown
    python3 scripts/progress_telegram.py --dry-run
    python3 scripts/progress_telegram.py --issue 3804          # post via gh
    python3 scripts/progress_telegram.py --selftest            # offline

Exit codes:
    0 — success (telegram rendered; posted, dry-run, or selftest passed)
    1 — failure: local canonical source missing/unreadable, selftest
        assertion failure, or the gh comment post failed
    2 — usage error (unknown --output mode, bad --weeks/--issue)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical generated sources (do not hand-edit; see AGENTS.md).
PERFORMANCE_HISTORY = REPO_ROOT / "validation" / "performance_history.latest.json"
SCORECARD = REPO_ROOT / "SCORECARD.md"
KNOWN_ISSUES = REPO_ROOT / "docs" / "KNOWN_ISSUES.md"

DEFAULT_TRACKER_ISSUE = 3804  # pinned "v1.3 progress tracker" issue

# ADR-0016 physics path class (§Summary 3/7). A PR touching ANY of these
# path prefixes is a physics-lane PR; everything else is hygiene.
PHYSICS_PATH_PREFIXES = (
    "src/",
    "fluxion-core/",
    "fluxion-fluid/",
    "tests/",
    "models/",
    "weather/",
)

# Marker embedded in every posted telegram. The next run parses the most
# recent comment containing this marker to derive week-over-week deltas
# (stateless — no committed state file, mirroring the β-soak pattern).
MARKER_KEY = "progress-telegram-v1"


# ---------------------------------------------------------------------------
# Pure functions (offline-testable; no I/O, no network)
# ---------------------------------------------------------------------------


def classify_lane(files: list[str]) -> str:
    """Classify a PR into ``"physics"`` or ``"hygiene"`` by its files.

    ADR-0016 deny-list semantics: ANY file under the physics path class
    puts the PR in the physics lane (the gauntlet triggers on the union,
    fail-safe by construction). An empty file list classifies as hygiene
    (data-unavailable defaults to the cheaper lane; reported n is still
    surfaced so the reader can see the sample size).
    """
    for path in files:
        if any(path.startswith(prefix) for prefix in PHYSICS_PATH_PREFIXES):
            return "physics"
    return "hygiene"


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile (numpy 'linear' method).

    ``q`` is in [0, 100]. Returns None for an empty sample. For p50 on
    even samples this equals statistics.median (the mean of the two
    middle values); on odd samples the middle value.
    """
    if not values:
        return None
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile q out of range: {q}")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def parse_iso8601(ts: str) -> datetime | None:
    """Parse a GitHub API ISO-8601 timestamp (``...Z`` or with offset)."""
    if not ts:
        return None
    text = ts.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def cycle_time_minutes(created_at: str, merged_at: str) -> float | None:
    """PR cycle time in minutes (createdAt → mergedAt).

    Returns None when either timestamp is unparsable or the interval is
    negative (defensive: GitHub sorts by mergedAt; a bad datum is dropped
    rather than poisoning the percentile).
    """
    created = parse_iso8601(created_at)
    merged = parse_iso8601(merged_at)
    if created is None or merged is None:
        return None
    minutes = (merged - created).total_seconds() / 60.0
    if minutes < 0:
        return None
    return minutes


def count_limit_gaps(text: str) -> tuple[int, int]:
    """Count LIMIT gaps in docs/KNOWN_ISSUES.md content.

    Returns ``(distinct_gaps, heading_count)``. Headings are ``#``-prefixed
    lines mentioning ``LIMIT-<digits>``; the LIMIT-05 UPDATE blocks are
    updates to the same gap, so the distinct-id count is the true gap
    count and the heading count (including UPDATEs) is reported for
    transparency.
    """
    ids = re.findall(r"^#{1,6}[^\n]*?LIMIT-(\d+)", text, re.MULTILINE)
    return (len(set(ids)), len(ids))


_CASE_LEVEL_RE = re.compile(
    r"\*\*Case-level:\*\*\s*(\d+)\s*/\s*(\d+)\s*cases fully PASS"
)
_PASS_RATE_HEADLINE_RE = re.compile(
    r"\|\s*ASHRAE 140 pass rate\s*\|\s*\*\*([\d.]+)%\*\*\s*\((\d+)/(\d+) metrics\)"
)


def parse_case_level(scorecard_text: str) -> tuple[int, int] | None:
    """Extract ``(cases_fully_passing, total_cases)`` from SCORECARD.md.

    Parses the generated line ``**Case-level:** 0/18 cases fully PASS``.
    Returns None when the generated shape changes (fail-visible, never a
    guessed number).
    """
    m = _CASE_LEVEL_RE.search(scorecard_text)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def parse_pass_rate_headline(scorecard_text: str) -> tuple[float, int, int] | None:
    """Extract ``(pass_rate_pct, passing_metrics, total_metrics)`` from the
    SCORECARD.md headline table row (``**14.1%** (12/84 metrics)``)."""
    m = _PASS_RATE_HEADLINE_RE.search(scorecard_text)
    if not m:
        return None
    return (float(m.group(1)), int(m.group(2)), int(m.group(3)))


def merge_stats(prs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the merge-mix / lane / fix-loop block from merged-PR dicts.

    Each PR dict: ``number``, ``labels`` (list[str]), ``files``
    (list[str]), ``commits`` (int), ``createdAt`` / ``mergedAt`` (ISO).
    """
    milestone = [p for p in prs if "milestone" in (p.get("labels") or [])]
    hygiene = [p for p in prs if "milestone" not in (p.get("labels") or [])]

    physics_cycles: list[float] = []
    hygiene_cycles: list[float] = []
    for pr in prs:
        minutes = cycle_time_minutes(pr.get("createdAt", ""), pr.get("mergedAt", ""))
        if minutes is None:
            continue
        lane = classify_lane(pr.get("files") or [])
        (physics_cycles if lane == "physics" else hygiene_cycles).append(minutes)

    multi_commit = [p for p in prs if int(p.get("commits") or 0) > 1]

    return {
        "merged_total": len(prs),
        "milestone_merges": len(milestone),
        "hygiene_merges": len(hygiene),
        "physics_p50_min": percentile(physics_cycles, 50),
        "hygiene_p50_min": percentile(hygiene_cycles, 50),
        "physics_n": len(physics_cycles),
        "hygiene_n": len(hygiene_cycles),
        "multi_commit_prs": len(multi_commit),
        "fix_loop_rate_pct": (100.0 * len(multi_commit) / len(prs) if prs else None),
    }


def cancelled_share(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Cancelled-run share from ``gh run list --json conclusion`` output.

    In-flight runs (empty conclusion) are excluded from both numerator
    and denominator: the share measures ADR-0015 cancel-storm waste
    among *completed* runs.
    """
    completed = [r for r in runs if r.get("conclusion")]
    cancelled = [r for r in completed if r["conclusion"] == "cancelled"]
    return {
        "completed_runs": len(completed),
        "cancelled_runs": len(cancelled),
        "cancelled_run_share_pct": (
            100.0 * len(cancelled) / len(completed) if completed else None
        ),
    }


# ---------------------------------------------------------------------------
# Telegram dataclass + marker (stateless WoW-delta channel)
# ---------------------------------------------------------------------------


@dataclass
class TelegramMetrics:
    """All numbers rendered into the weekly telegram."""

    generated_at: str = ""
    window_days: int = 7
    # scorecard
    pass_rate_pct: float | None = None
    passing_metrics: int | None = None
    total_metrics: int | None = None
    mae_pct: float | None = None
    cases_fully_passing: int | None = None
    cases_total: int | None = None
    # pipeline
    milestone_merges: int = 0
    hygiene_merges: int = 0
    merged_total: int = 0
    physics_p50_min: float | None = None
    hygiene_p50_min: float | None = None
    physics_n: int = 0
    hygiene_n: int = 0
    multi_commit_prs: int = 0
    fix_loop_rate_pct: float | None = None
    cancelled_run_share_pct: float | None = None
    cancelled_runs: int = 0
    completed_runs: int = 0
    # validation debt
    limit_gaps: int = 0
    limit_headings: int = 0
    # provenance
    warnings: list[str] = field(default_factory=list)

    def to_marker(self) -> dict[str, float]:
        """Numeric fields serialized into the HTML marker comment."""
        out: dict[str, float] = {}
        for name in (
            "pass_rate_pct",
            "mae_pct",
            "cases_fully_passing",
            "cases_total",
            "milestone_merges",
            "hygiene_merges",
            "physics_p50_min",
            "hygiene_p50_min",
            "fix_loop_rate_pct",
            "cancelled_run_share_pct",
            "limit_gaps",
        ):
            value = getattr(self, name)
            if value is not None:
                out[name] = float(value)
        return out


def build_marker(metrics: TelegramMetrics) -> str:
    """Render the machine-readable marker line embedded in the comment."""
    pairs = " ".join(f"{k}={v}" for k, v in sorted(metrics.to_marker().items()))
    return f"<!-- {MARKER_KEY} {pairs} -->"


def parse_marker(body: str) -> dict[str, float] | None:
    """Extract the metrics dict from a previous telegram comment body.

    Returns the LAST marker in the body (the freshest write wins) or
    None when no marker is present.
    """
    matches = re.findall(rf"<!--\s*{re.escape(MARKER_KEY)}\s+([^\n]*?)\s*-->", body)
    if not matches:
        return None
    out: dict[str, float] = {}
    for pair in matches[-1].split():
        if "=" not in pair:
            continue
        key, _, raw = pair.partition("=")
        try:
            out[key] = float(raw)
        except ValueError:
            continue
    return out or None


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def _fmt_min(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} min"


def fmt_delta(
    current: float | None,
    previous: float | None,
    unit: str,
    decimals: int = 1,
) -> str:
    """WoW delta suffix: `` (+1.2 pp)`` / `` (-2)`` / `` (±0)`` / `` (n/a)``."""
    if current is None or previous is None:
        return " (n/a)"
    diff = round(current, decimals + 2) - round(previous, decimals + 2)
    if abs(diff) < 10 ** (-decimals) / 2:
        return " (±0)"
    sign = "+" if diff > 0 else ""
    return f" ({sign}{diff:.{decimals}f}{unit})"


def render_markdown(
    metrics: TelegramMetrics,
    prev: dict[str, float] | None,
) -> str:
    """Render the telegram comment body (GitHub-flavoured markdown)."""
    m = metrics
    date = m.generated_at[:10] if m.generated_at else "unknown-date"
    lines: list[str] = []
    lines.append(f"## 📟 v1.3 progress telegram — week of {date}")
    lines.append("")
    lines.append(build_marker(m))
    lines.append("")

    lines.append("### Scorecard (ASHRAE 140)")
    lines.append("")
    if m.passing_metrics is not None and m.total_metrics is not None:
        headline = (
            f"**{_fmt_pct(m.pass_rate_pct)}** "
            f"({m.passing_metrics}/{m.total_metrics} metrics)"
        )
    else:
        headline = f"**{_fmt_pct(m.pass_rate_pct)}**"
    lines.append(
        f"- Metric pass rate: {headline}"
        f"{fmt_delta(m.pass_rate_pct, prev_get(prev, 'pass_rate_pct'), ' pp')}"
    )
    if m.cases_total is not None:
        case_pct = (
            100.0 * m.cases_fully_passing / m.cases_total
            if m.cases_fully_passing is not None
            else None
        )
        lines.append(
            f"- Cases fully passing: **{m.cases_fully_passing}/{m.cases_total}**"
            f" ({_fmt_pct(case_pct)})"
            f"{fmt_delta(m.cases_fully_passing, prev_get(prev, 'cases_fully_passing'), '', 0)}"
        )
    lines.append(
        f"- MAE: **{_fmt_pct(m.mae_pct)}** (budget ≤ 50%)"
        f"{fmt_delta(m.mae_pct, prev_get(prev, 'mae_pct'), ' pp')}"
    )
    lines.append("")

    lines.append(f"### Pipeline throughput ({m.window_days}d window)")
    lines.append("")
    lines.append(
        f"- Merge mix: **{m.milestone_merges} milestone** / "
        f"**{m.hygiene_merges} hygiene** ({m.merged_total} merged)"
        f"{fmt_delta(m.milestone_merges, prev_get(prev, 'milestone_merges'), '', 0)}"
    )
    physics_lane = (
        f"**{_fmt_min(m.physics_p50_min)}** (n={m.physics_n})"
        if m.physics_p50_min is not None
        else f"n/a (n={m.physics_n})"
    )
    hygiene_lane = (
        f"**{_fmt_min(m.hygiene_p50_min)}** (n={m.hygiene_n})"
        if m.hygiene_p50_min is not None
        else f"n/a (n={m.hygiene_n})"
    )
    lines.append(
        f"- Cycle-time p50: physics {physics_lane} · hygiene {hygiene_lane}"
        f"{fmt_delta(m.physics_p50_min, prev_get(prev, 'physics_p50_min'), ' min', 0)}"
    )
    lines.append(
        f"- Fix-loop rate: **{_fmt_pct(m.fix_loop_rate_pct)}** "
        f"({m.multi_commit_prs}/{m.merged_total} PRs >1 commit)"
        f"{fmt_delta(m.fix_loop_rate_pct, prev_get(prev, 'fix_loop_rate_pct'), ' pp')}"
    )
    lines.append(
        f"- Cancelled-run share: **{_fmt_pct(m.cancelled_run_share_pct)}** "
        f"({m.cancelled_runs}/{m.completed_runs} completed runs)"
        f"{fmt_delta(m.cancelled_run_share_pct, prev_get(prev, 'cancelled_run_share_pct'), ' pp')}"
    )
    lines.append("")

    lines.append("### Validation debt")
    lines.append("")
    lines.append(
        f"- Known LIMIT gaps: **{m.limit_gaps}** distinct "
        f"({m.limit_headings} headings incl. updates)"
        f"{fmt_delta(m.limit_gaps, prev_get(prev, 'limit_gaps'), '', 0)}"
    )
    lines.append("")

    if m.warnings:
        lines.append(
            f"> ⚠️ {len(m.warnings)} data source(s) unavailable this week: "
            + "; ".join(m.warnings)
        )
        lines.append("")

    lines.append(
        "_generated by `scripts/progress_telegram.py` "
        "(plan-pm-throughput-20260915 T5, ADR-0016 companion; "
        "sources: `validation/performance_history.latest.json`, `SCORECARD.md`, "
        "`gh pr list`/`gh run list`, `docs/KNOWN_ISSUES.md`)_"
    )
    lines.append("")
    return "\n".join(lines)


def prev_get(prev: dict[str, float] | None, key: str) -> float | None:
    """Fetch a previous-week value (None-safe)."""
    if prev is None:
        return None
    return prev.get(key)


# ---------------------------------------------------------------------------
# Data collection (gh subprocess; every failure is a warning, not a crash —
# the weekly post must still go out with visible n/a slots)
# ---------------------------------------------------------------------------


def run_gh(args: list[str]) -> str:
    """Run a read-only gh command and return stdout (raises on failure)."""
    proc = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh {' '.join(args[:2])} failed ({proc.returncode}): "
            f"{proc.stderr.strip()[:300]}"
        )
    return proc.stdout


def collect_merged_prs(
    repo: str, since: datetime, limit: int = 300
) -> list[dict[str, Any]]:
    """Merged PRs since ``since`` with labels, files, and commit counts.

    Uses the ``merged:>YYYY-MM-DD`` search qualifier for the window, then
    re-filters precisely on mergedAt in Python (the qualifier is
    day-granular). Raises RuntimeError when the result hits the fetch
    limit (truncation would silently bias the mix/cycle stats toward one
    end of the window — a busy week must fail visibly, not under-report).
    """
    since_date = since.date().isoformat()
    out = run_gh(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "merged",
            "--search",
            f"merged:>{since_date}",
            "--limit",
            str(limit),
            "--json",
            "number,title,labels,createdAt,mergedAt",
        ]
    )
    listed = json.loads(out)
    if len(listed) >= limit:
        raise RuntimeError(
            f"merged-PR window truncated at the gh --limit ({limit}); "
            "raise the limit or narrow --weeks"
        )
    prs = [
        p for p in listed if (parse_iso8601(p.get("mergedAt", "")) or since) >= since
    ]
    # Files and commit counts are per-PR fields: one `gh pr view` each.
    for p in prs:
        detail = json.loads(
            run_gh(
                [
                    "pr",
                    "view",
                    str(p["number"]),
                    "--repo",
                    repo,
                    "--json",
                    "files,commits",
                ]
            )
        )
        p["files"] = [f["path"] for f in detail.get("files", [])]
        p["commits"] = len(detail.get("commits", []))
    return prs


def collect_workflow_runs(
    repo: str, since: datetime, max_pages: int = 150
) -> list[dict[str, Any]]:
    """Workflow runs created since ``since`` (conclusion, incl. in-flight).

    ``gh run list`` hard-caps at 1000 results and this repo exceeds that
    in cancel-storm weeks (measured: >7k runs in the 2026-09-15 lane-split
    week), so this collector paginates the REST endpoint
    (``repos/{repo}/actions/runs?created=...``) directly via ``gh api``.
    Raises RuntimeError when ``max_pages`` is exhausted before
    ``total_count`` is reached (fail-visible, never a silent undercount).
    """
    per_page = 100
    # Full ISO timestamp — date-only values misbehave on this endpoint
    # (measured: non-monotonic total_count), while `>=` + RFC3339 is
    # exact, so the API-side window matches the Python-side re-filter.
    since_iso = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    runs: list[dict[str, Any]] = []
    total_count: int | None = None
    for page in range(1, max_pages + 1):
        out = run_gh(
            [
                "api",
                f"repos/{repo}/actions/runs",
                "-X",
                "GET",
                "-f",
                f"created=>={since_iso}",
                "-f",
                f"per_page={per_page}",
                "-f",
                f"page={page}",
            ]
        )
        payload = json.loads(out)
        total_count = int(payload.get("total_count") or 0)
        batch = payload.get("workflow_runs") or []
        if not batch:
            break
        for r in batch:
            runs.append(
                {
                    "conclusion": r.get("conclusion") or "",
                    "createdAt": r.get("created_at") or "",
                }
            )
        if len(runs) >= total_count:
            break
    else:
        raise RuntimeError(
            f"workflow-run pagination exhausted {max_pages} pages "
            f"without reaching total_count={total_count}"
        )
    return [
        r for r in runs if (parse_iso8601(r.get("createdAt", "")) or since) >= since
    ]


def collect_previous_metrics(repo: str, issue: int) -> dict[str, float] | None:
    """Parse the most recent telegram marker on the tracker issue.

    The tracker issue is public, so any GitHub commenter can post a
    body containing ``MARKER_KEY``. Trust the marker ONLY when the
    comment was authored by ``github-actions[bot]`` — the same
    identity that runs ``.github/workflows/progress_telegram.yml``
    and emits the genuine weekly posts (per Issue #3814). This is
    the integrity mechanism for the WoW deltas that feed the v1.3
    progress dashboard.
    """
    out = run_gh(
        [
            "issue",
            "view",
            str(issue),
            "--repo",
            repo,
            "--json",
            "comments",
            "--jq",
            "[.comments[] | select(.author.login == \"github-actions[bot]\""
            " and .body | contains("
            f'"{MARKER_KEY}"'
            "))][-.1:] | map(.body) | .[0] // empty",
        ]
    ).strip()
    if not out:
        return None
    return parse_marker(out)


def resolve_repo(explicit: str | None) -> str | None:
    """Default the gh --repo target: flag > GH_REPO > origin remote."""
    if explicit:
        return explicit
    import os

    env_repo = os.environ.get("GH_REPO")
    if env_repo:
        return env_repo
    try:
        url = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"[:/]([^/:]+/[^/]+?)(?:\.git)?$", url)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Selftest (offline fixtures — exercised by --selftest and by
# scripts/ci/test_progress_telegram.py)
# ---------------------------------------------------------------------------

SELFTEST_KNOWN_ISSUES = """# Known Issues
### LIMIT-01: High-Mass Annual Energy Discrepancy
body
### LIMIT-02: Free-Floating Temperature Range for Low-Mass
body
### LIMIT-05: High-Mass Peak Cooling
body
### LIMIT-05 UPDATE (Phase 36): more detail
body
### LIMIT-05 UPDATE (Issue #2300): even more detail
"""

SELFTEST_SCORECARD = """# Fluxion Release Scorecard
| ASHRAE 140 pass rate | **14.1%** (12/84 metrics) | ≥ 60% | ❌ Fail |
- **Overall (metric-level):** 14.1% — 12 PASS / 8 WARN / 64 FAIL of 84 results.
- **Case-level:** 2/18 cases fully PASS (11.1%).
"""

SELFTEST_PRS = [
    {
        "number": 101,
        "labels": ["milestone"],
        "files": ["src/sim/thermal.rs", "docs/x.md"],
        "commits": 1,
        "createdAt": "2026-09-10T10:00:00Z",
        "mergedAt": "2026-09-10T13:00:00Z",  # physics, 180 min
    },
    {
        "number": 102,
        "labels": [],
        "files": ["scripts/check_foo.py"],
        "commits": 2,
        "createdAt": "2026-09-11T09:00:00Z",
        "mergedAt": "2026-09-11T10:00:00Z",  # hygiene, 60 min, multi-commit
    },
    {
        "number": 103,
        "labels": [],
        "files": ["docs/adr/0001-x.md"],
        "commits": 1,
        "createdAt": "2026-09-12T09:00:00Z",
        "mergedAt": "2026-09-12T09:30:00Z",  # hygiene, 30 min
    },
]

SELFTEST_RUNS = [
    {"conclusion": "success"},
    {"conclusion": "cancelled"},
    {"conclusion": "failure"},
    {"conclusion": "success"},
    {"conclusion": ""},  # in-flight — excluded
]


def run_selftest() -> list[str]:
    """Run the offline fixture assertions; return failure descriptions."""
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        if not cond:
            failures.append(name)

    check(
        "lane: physics prefixes",
        classify_lane(["src/a.rs"]) == "physics"
        and classify_lane(["fluxion-core/src/lib.rs"]) == "physics"
        and classify_lane(["fluxion-fluid/src/x.rs"]) == "physics"
        and classify_lane(["tests/foo.rs"]) == "physics"
        and classify_lane(["models/m.onnx"]) == "physics"
        and classify_lane(["weather/epw/file.epw"]) == "physics",
    )
    check(
        "lane: hygiene + mixed",
        classify_lane(["docs/a.md", "scripts/x.py", ".github/workflows/y.yml"])
        == "hygiene"
        and classify_lane(["README.md"]) == "hygiene"
        and classify_lane([]) == "hygiene"
        and classify_lane(["docs/a.md", "src/sim/x.rs"]) == "physics",
    )
    check(
        "percentile p50",
        percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
        and percentile([3.0, 1.0, 2.0], 50) == 2.0
        and percentile([5.0], 50) == 5.0
        and percentile([], 50) is None,
    )
    check(
        "cycle time",
        cycle_time_minutes("2026-09-10T10:00:00Z", "2026-09-10T13:00:00Z") == 180.0
        and cycle_time_minutes("bad", "2026-09-10T13:00:00Z") is None
        and cycle_time_minutes("2026-09-10T13:00:00Z", "2026-09-10T10:00:00Z") is None,
    )
    gaps, headings = count_limit_gaps(SELFTEST_KNOWN_ISSUES)
    check("limit gaps", gaps == 3 and headings == 5)
    check(
        "case level",
        parse_case_level(SELFTEST_SCORECARD) == (2, 18)
        and parse_case_level("no line here") is None,
    )
    check(
        "pass-rate headline",
        parse_pass_rate_headline(SELFTEST_SCORECARD) == (14.1, 12, 84),
    )
    stats = merge_stats(SELFTEST_PRS)
    check(
        "merge stats",
        stats["merged_total"] == 3
        and stats["milestone_merges"] == 1
        and stats["hygiene_merges"] == 2
        and stats["physics_p50_min"] == 180.0
        and stats["hygiene_p50_min"] == 45.0  # median of [60, 30]
        and stats["multi_commit_prs"] == 1
        and abs(stats["fix_loop_rate_pct"] - 33.3333) < 0.01,
    )
    share = cancelled_share(SELFTEST_RUNS)
    check(
        "cancelled share",
        share["completed_runs"] == 4
        and share["cancelled_runs"] == 1
        and abs(share["cancelled_run_share_pct"] - 25.0) < 0.01,
    )

    metrics = TelegramMetrics(
        generated_at="2026-09-15T07:00:00+00:00",
        window_days=7,
        pass_rate_pct=14.1,
        passing_metrics=12,
        total_metrics=84,
        mae_pct=49.8,
        cases_fully_passing=2,
        cases_total=18,
        **{
            k: v
            for k, v in merge_stats(SELFTEST_PRS).items()
            if k in TelegramMetrics.__dataclass_fields__
        },
        **cancelled_share(SELFTEST_RUNS),
        limit_gaps=gaps,
        limit_headings=headings,
    )
    body = render_markdown(metrics, None)
    check(
        "render: sections",
        "Scorecard (ASHRAE 140)" in body
        and "Pipeline throughput (7d window)" in body
        and "Validation debt" in body
        and "**2/18**" in body
        and "**1 milestone** / **2 hygiene**" in body
        and "(1/3 PRs >1 commit)" in body,
    )
    check("render: deltas n/a on first week", "(n/a)" in body)
    round_trip = parse_marker(build_marker(metrics))
    check(
        "marker round-trip",
        round_trip is not None
        and round_trip["pass_rate_pct"] == 14.1
        and round_trip["limit_gaps"] == 3.0
        and round_trip["physics_p50_min"] == 180.0,
    )
    prev = parse_marker(
        "## 📟 telegram\n\n<!-- progress-telegram-v1 pass_rate_pct=10.0 "
        "limit_gaps=5 cases_fully_passing=0 -->\n\nbody"
    )
    check(
        "prev-marker parse",
        prev is not None
        and prev["pass_rate_pct"] == 10.0
        and prev["limit_gaps"] == 5.0,
    )
    body2 = render_markdown(metrics, prev)
    check("render: WoW deltas", "(+4.1 pp)" in body2 and "(-2)" in body2)
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def collect_metrics(repo: str, weeks: int) -> TelegramMetrics:
    """Collect every metric; failures degrade to n/a + warning."""
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=weeks * 7)
    metrics = TelegramMetrics(
        generated_at=now.isoformat(timespec="seconds"),
        window_days=weeks * 7,
    )

    # Local canonical files — missing/unreadable here is a hard error
    # (they are committed artifacts; absence means wrong cwd or a broken
    # checkout, and the telegram must not post fabricated numbers).
    try:
        history = json.loads(PERFORMANCE_HISTORY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"ERROR: cannot read {PERFORMANCE_HISTORY}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    metrics.pass_rate_pct = history.get("pass_rate")
    metrics.mae_pct = history.get("mae")

    try:
        scorecard_text = SCORECARD.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot read {SCORECARD}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    headline = parse_pass_rate_headline(scorecard_text)
    if headline is not None:
        (
            metrics.passing_metrics,
            metrics.total_metrics,
        ) = headline[1], headline[2]
    else:
        metrics.warnings.append("SCORECARD.md pass-rate headline unparsable")
    case_level = parse_case_level(scorecard_text)
    if case_level is not None:
        metrics.cases_fully_passing, metrics.cases_total = case_level
    else:
        metrics.warnings.append("SCORECARD.md case-level line unparsable")

    try:
        known_issues = KNOWN_ISSUES.read_text(encoding="utf-8")
        metrics.limit_gaps, metrics.limit_headings = count_limit_gaps(known_issues)
    except OSError:
        metrics.warnings.append("docs/KNOWN_ISSUES.md unreadable")

    # gh-sourced metrics degrade to n/a (the weekly post still ships).
    prs: list[dict[str, Any]] = []
    try:
        prs = collect_merged_prs(repo, since)
    except (RuntimeError, OSError, json.JSONDecodeError) as exc:
        metrics.warnings.append(f"merged-PR stats unavailable ({exc})")
    if prs:
        stats = merge_stats(prs)
        for key, value in stats.items():
            setattr(metrics, key, value)
    elif "merged-PR stats unavailable" not in ";".join(metrics.warnings):
        metrics.warnings.append("no merged PRs in window")

    try:
        runs = collect_workflow_runs(repo, since)
        for key, value in cancelled_share(runs).items():
            setattr(metrics, key, value)
    except (RuntimeError, OSError, json.JSONDecodeError) as exc:
        metrics.warnings.append(f"workflow-run stats unavailable ({exc})")

    return metrics


def post_comment(repo: str, issue: int, body: str) -> None:
    """Post the telegram as a comment on the tracker issue via gh."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".md", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(body)
        path = handle.name
    run_gh(["issue", "comment", str(issue), "--repo", repo, "--body-file", path])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Weekly v1.3 progress telegram (plan T5, ADR-0016 companion)"
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=1,
        help="window length in weeks (default: 1)",
    )
    parser.add_argument(
        "--output",
        choices=["markdown"],
        default="markdown",
        help="output format (default: markdown)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the telegram to stdout; do not post",
    )
    parser.add_argument(
        "--issue",
        type=int,
        default=DEFAULT_TRACKER_ISSUE,
        help=f"tracker issue number to comment on (default: {DEFAULT_TRACKER_ISSUE})",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="owner/repo for gh (default: GH_REPO env or origin remote)",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run the offline fixture selftest and exit",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        failures = run_selftest()
        if failures:
            print(f"SELFTEST FAIL ({len(failures)}):", file=sys.stderr)
            for name in failures:
                print(f"  - {name}", file=sys.stderr)
            return 1
        print("SELFTEST OK: all offline fixture assertions passed")
        return 0

    if args.weeks < 1:
        print("ERROR: --weeks must be >= 1", file=sys.stderr)
        return 2

    repo = resolve_repo(args.repo)
    if repo is None:
        print(
            "ERROR: cannot resolve --repo (flag, GH_REPO, or origin remote)",
            file=sys.stderr,
        )
        return 2

    metrics = collect_metrics(repo, args.weeks)

    prev: dict[str, float] | None = None
    try:
        prev = collect_previous_metrics(repo, args.issue)
    except (RuntimeError, OSError) as exc:
        metrics.warnings.append(f"previous telegram unavailable ({exc})")

    body = render_markdown(metrics, prev)

    # The rendered body always goes to stdout (the scheduled workflow
    # tees it into the run log + artifact); status lines go to stderr so
    # the captured body stays byte-identical to the posted comment.
    print(body)

    if args.dry_run:
        return 0

    try:
        post_comment(repo, args.issue, body)
    except (RuntimeError, OSError) as exc:
        print(f"ERROR: posting to issue #{args.issue} failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Posted progress telegram to issue #{args.issue} ({repo})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
