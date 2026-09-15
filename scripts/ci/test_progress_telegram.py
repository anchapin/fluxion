"""Tests for ``scripts/progress_telegram.py`` (plan pm-throughput-20260915
task T5; tracker issue #3804; ADR-0016 companion).

Pure-function coverage over hermetic fixtures: lane classification
(ADR-0016 physics path class), cycle-time percentile, LIMIT-gap counting,
SCORECARD.md parsing, marker round-trip, markdown rendering with and
without week-over-week deltas, plus the offline ``--selftest`` CLI flag.
No network access — the gh collectors are thin subprocess wrappers and
are exercised only via the mocking-free dry paths in CI (the workflow
itself is the integration test).
"""

from __future__ import annotations

import subprocess
import sys

FIXTURE_KNOWN_ISSUES = """# Known Issues
### LIMIT-01: High-Mass Annual Energy Discrepancy
body
### LIMIT-02: Free-Floating Temperature Range for Low-Mass
body
### LIMIT-03: Hardcoded HVAC Capacity
body
### LIMIT-03 UPDATE (Phase 36): more detail
body
### LIMIT-04: Case 960 Peak Heating
body
### LIMIT-04 UPDATE (Issue #2300): even more
"""

FIXTURE_SCORECARD = """# Fluxion Release Scorecard
|--------|---------|---------------|--------|-------|
| ASHRAE 140 pass rate | **14.1%** (12/84 metrics) | ≥ 60% | ❌ Fail | src |
- **Overall (metric-level):** 14.1% — 12 PASS / 8 WARN / 64 FAIL of 84 results.
- **Case-level:** 2/18 cases fully PASS (11.1%).
"""

FIXTURE_PRS = [
    {
        # physics lane (src/**), milestone-labelled, 1 commit, 180 min
        "number": 101,
        "labels": ["milestone"],
        "files": ["src/sim/thermal.rs", "docs/x.md"],
        "commits": 1,
        "createdAt": "2026-09-10T10:00:00Z",
        "mergedAt": "2026-09-10T13:00:00Z",
    },
    {
        # hygiene lane (scripts only), 2 commits (fix-loop), 60 min
        "number": 102,
        "labels": [],
        "files": ["scripts/check_foo.py"],
        "commits": 2,
        "createdAt": "2026-09-11T09:00:00Z",
        "mergedAt": "2026-09-11T10:00:00Z",
    },
    {
        # hygiene lane (docs only), 1 commit, 30 min
        "number": 103,
        "labels": [],
        "files": ["docs/adr/0001-x.md"],
        "commits": 1,
        "createdAt": "2026-09-12T09:00:00Z",
        "mergedAt": "2026-09-12T09:30:00Z",
    },
    {
        # physics lane (tests/** under fluxion-fluid sibling), 1 commit,
        # 120 min — exercises the sibling-crate physics prefixes
        "number": 104,
        "labels": [],
        "files": ["fluxion-fluid/tests/hvac.rs"],
        "commits": 1,
        "createdAt": "2026-09-13T08:00:00Z",
        "mergedAt": "2026-09-13T10:00:00Z",
    },
]

FIXTURE_RUNS = [
    {"conclusion": "success"},
    {"conclusion": "cancelled"},
    {"conclusion": "failure"},
    {"conclusion": "success"},
    {"conclusion": "cancelled"},
    {"conclusion": ""},  # in-flight — must be excluded from both counters
]


class TestLaneClassification:
    def test_physics_path_class_triggers_physics_lane(self, load_script):
        mod = load_script("progress_telegram")
        for prefix in mod.PHYSICS_PATH_PREFIXES:
            assert mod.classify_lane([f"{prefix}some/file"]) == "physics"

    def test_non_physics_paths_are_hygiene(self, load_script):
        mod = load_script("progress_telegram")
        assert (
            mod.classify_lane(["docs/a.md", "scripts/x.py", ".github/workflows/y.yml"])
            == "hygiene"
        )
        assert mod.classify_lane(["README.md", "Cargo.toml", "npm/index.js"]) == (
            "hygiene"
        )

    def test_mixed_pr_takes_physics_lane_any_match(self, load_script):
        mod = load_script("progress_telegram")
        assert mod.classify_lane(["docs/a.md", "src/sim/x.rs"]) == "physics"

    def test_empty_file_list_defaults_hygiene(self, load_script):
        mod = load_script("progress_telegram")
        assert mod.classify_lane([]) == "hygiene"


class TestPercentileAndCycleTime:
    def test_p50_odd_even_single_empty(self, load_script):
        mod = load_script("progress_telegram")
        assert mod.percentile([3.0, 1.0, 2.0], 50) == 2.0
        assert mod.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
        assert mod.percentile([5.0], 50) == 5.0
        assert mod.percentile([], 50) is None

    def test_p50_matches_linear_interpolation(self, load_script):
        mod = load_script("progress_telegram")
        # (5-1)*0.25 = 1.0 → s[1] + 0.0*(s[2]-s[1]) ... q=25 on 5 samples
        assert mod.percentile([10.0, 20.0, 30.0, 40.0, 50.0], 25) == 20.0

    def test_percentile_rejects_out_of_range_q(self, load_script):
        import pytest

        mod = load_script("progress_telegram")
        with pytest.raises(ValueError):
            mod.percentile([1.0], 101)

    def test_cycle_time_minutes(self, load_script):
        mod = load_script("progress_telegram")
        assert (
            mod.cycle_time_minutes("2026-09-10T10:00:00Z", "2026-09-10T13:30:00Z")
            == 210.0
        )

    def test_cycle_time_rejects_bad_and_negative(self, load_script):
        mod = load_script("progress_telegram")
        assert mod.cycle_time_minutes("garbage", "2026-09-10T13:00:00Z") is None
        assert mod.cycle_time_minutes("", "2026-09-10T13:00:00Z") is None
        assert (
            mod.cycle_time_minutes("2026-09-10T13:00:00Z", "2026-09-10T10:00:00Z")
            is None
        )


class TestLimitCounting:
    def test_distinct_ids_vs_headings(self, load_script):
        mod = load_script("progress_telegram")
        gaps, headings = mod.count_limit_gaps(FIXTURE_KNOWN_ISSUES)
        # LIMIT-01, -02, -03, -04 distinct; -03/-04 carry UPDATE headings
        assert gaps == 4
        assert headings == 6

    def test_empty_document(self, load_script):
        mod = load_script("progress_telegram")
        assert mod.count_limit_gaps("# Known Issues\n\nnothing\n") == (0, 0)

    def test_inline_mentions_do_not_count(self, load_script):
        mod = load_script("progress_telegram")
        # 'LIMIT-9' inside a paragraph body is not a heading
        text = "## Section\nsee LIMIT-9 above\n"
        assert mod.count_limit_gaps(text) == (0, 0)


class TestScorecardParsing:
    def test_case_level_line(self, load_script):
        mod = load_script("progress_telegram")
        assert mod.parse_case_level(FIXTURE_SCORECARD) == (2, 18)

    def test_case_level_absent_returns_none(self, load_script):
        mod = load_script("progress_telegram")
        assert mod.parse_case_level("# no case-level line\n") is None

    def test_pass_rate_headline(self, load_script):
        mod = load_script("progress_telegram")
        assert mod.parse_pass_rate_headline(FIXTURE_SCORECARD) == (
            14.1,
            12,
            84,
        )

    def test_pass_rate_headline_absent(self, load_script):
        mod = load_script("progress_telegram")
        assert mod.parse_pass_rate_headline("no table here") is None


class TestMergeStatsAndRuns:
    def test_merge_stats_lanes_mix_fixloop(self, load_script):
        mod = load_script("progress_telegram")
        stats = mod.merge_stats(FIXTURE_PRS)
        assert stats["merged_total"] == 4
        assert stats["milestone_merges"] == 1
        assert stats["hygiene_merges"] == 3
        # physics cycles [180, 120] → p50 150; hygiene [60, 30] → p50 45
        assert stats["physics_p50_min"] == 150.0
        assert stats["hygiene_p50_min"] == 45.0
        assert stats["physics_n"] == 2
        assert stats["hygiene_n"] == 2
        assert stats["multi_commit_prs"] == 1
        assert stats["fix_loop_rate_pct"] == 25.0

    def test_merge_stats_empty_window(self, load_script):
        mod = load_script("progress_telegram")
        stats = mod.merge_stats([])
        assert stats["merged_total"] == 0
        assert stats["physics_p50_min"] is None
        assert stats["hygiene_p50_min"] is None
        assert stats["fix_loop_rate_pct"] is None

    def test_cancelled_share_excludes_inflight(self, load_script):
        mod = load_script("progress_telegram")
        share = mod.cancelled_share(FIXTURE_RUNS)
        assert share["completed_runs"] == 5
        assert share["cancelled_runs"] == 2
        assert share["cancelled_run_share_pct"] == 40.0

    def test_cancelled_share_empty(self, load_script):
        mod = load_script("progress_telegram")
        assert mod.cancelled_share([])["cancelled_run_share_pct"] is None


class TestMarkerAndRendering:
    def _metrics(self, mod):
        stats = mod.merge_stats(FIXTURE_PRS)
        share = mod.cancelled_share(FIXTURE_RUNS)
        gaps, headings = mod.count_limit_gaps(FIXTURE_KNOWN_ISSUES)
        return mod.TelegramMetrics(
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
                for k, v in stats.items()
                if k in mod.TelegramMetrics.__dataclass_fields__
            },
            **share,
            limit_gaps=gaps,
            limit_headings=headings,
        )

    def test_marker_round_trip(self, load_script):
        mod = load_script("progress_telegram")
        marker = mod.build_marker(self._metrics(mod))
        parsed = mod.parse_marker(marker)
        assert parsed is not None
        assert parsed["pass_rate_pct"] == 14.1
        assert parsed["limit_gaps"] == 4.0
        assert parsed["physics_p50_min"] == 150.0
        assert parsed["cases_fully_passing"] == 2.0

    def test_marker_parse_from_full_comment_body(self, load_script):
        mod = load_script("progress_telegram")
        body = (
            "## 📟 v1.3 progress telegram — week of 2026-09-08\n\n"
            "<!-- progress-telegram-v1 pass_rate_pct=10.0 limit_gaps=5 "
            "cases_fully_passing=0 -->\n\n- body\n"
        )
        parsed = mod.parse_marker(body)
        assert parsed is not None
        assert parsed["pass_rate_pct"] == 10.0
        assert parsed["limit_gaps"] == 5.0

    def test_marker_parse_absent_returns_none(self, load_script):
        mod = load_script("progress_telegram")
        assert mod.parse_marker("no marker here") is None

    def test_render_first_week_nas(self, load_script):
        mod = load_script("progress_telegram")
        body = mod.render_markdown(self._metrics(mod), None)
        assert "Scorecard (ASHRAE 140)" in body
        assert "Pipeline throughput (7d window)" in body
        assert "Validation debt" in body
        assert "**14.1%** (12/84 metrics)" in body
        assert "**2/18**" in body
        assert "**1 milestone** / **3 hygiene**" in body
        assert "(1/4 PRs >1 commit)" in body
        assert "**4** distinct" in body
        assert "(n/a)" in body  # first week: no deltas derivable

    def test_render_with_wow_deltas(self, load_script):
        mod = load_script("progress_telegram")
        prev = {
            "pass_rate_pct": 10.0,
            "cases_fully_passing": 0.0,
            "limit_gaps": 5.0,
            "physics_p50_min": 170.0,
        }
        body = mod.render_markdown(self._metrics(mod), prev)
        assert "(+4.1 pp)" in body  # pass rate 14.1 vs 10.0
        assert "(+2)" in body  # cases fully passing 2 vs 0
        assert "(-1)" in body  # LIMIT gaps 4 vs 5
        assert "(-20 min)" in body  # physics p50 150 vs 170

    def test_render_zero_delta_uses_pm_symbol(self, load_script):
        mod = load_script("progress_telegram")
        prev = {"pass_rate_pct": 14.1, "limit_gaps": 4.0}
        body = mod.render_markdown(self._metrics(mod), prev)
        assert "(±0)" in body

    def test_render_warns_on_missing_sources(self, load_script):
        mod = load_script("progress_telegram")
        metrics = self._metrics(mod)
        metrics.warnings.append("workflow-run stats unavailable (boom)")
        body = mod.render_markdown(metrics, None)
        assert "⚠️" in body
        assert "workflow-run stats unavailable" in body


class TestSelftestCli:
    def test_selftest_flag_exits_zero_offline(self, repo_root):
        proc = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "progress_telegram.py"),
                "--selftest",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert "SELFTEST OK" in proc.stdout

    def test_usage_error_on_bad_weeks(self, load_script):
        mod = load_script("progress_telegram")
        assert mod.main(["--weeks", "0", "--dry-run"]) == 2
