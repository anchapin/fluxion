"""
Tests for ``scripts/check_cycle_downward_trend.py`` -- Issue #2768.

The cycle downward-trend guard consumes the scan primitives of the two
existing cycle-check scripts (which already have their own pytest suite
under ``scripts/ci/test_check_physics_sim_cycle.py``) and layers three
directional rules on top: R1 (no growth), R2 (downward progress, nightly
only), and R3 (no net-flat edge swap).

These tests pin the rule logic. They use synthetic history ledgers and
monkey-patch ``collect_current_edges`` so the rule evaluator can be driven
through every branch without checking out real cycle regressions.

The script is imported as a module (same pattern as
``test_check_physics_sim_cycle.py``) so each test gets a fresh copy and
``REPO_ROOT`` can be redirected via monkeypatch when needed.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_cycle_downward_trend.py"
)


def _load_guard():
    """Load scripts/check_cycle_downward_trend.py as a fresh module."""
    spec = importlib.util.spec_from_file_location(
        "check_cycle_downward_trend", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def guard():
    """Return a freshly-loaded copy of the trend-guard script."""
    return _load_guard()


def _snapshot(total: int, signature: str = "s0", **bucket_counts) -> dict:
    """Build a minimal snapshot dict for the ledger / current-edge call shape."""
    totals = {
        "sim_to_validation": 0,
        "validation_to_sim": 0,
        "validation_to_physics": 0,
        "validation_to_weather": 0,
        "physics_to_sim": 0,
        "sim_to_physics": 0,
    }
    totals.update(bucket_counts)
    return {
        "timestamp": "2026-08-12T00:00:00+00:00",
        "commit": "deadbee",
        "source": "test",
        "totals": totals,
        "total": total,
        "edge_signature": signature,
    }


def _current(total: int, signature: str = "s0", **bucket_counts) -> dict:
    """Build a synthetic ``collect_current_edges`` return value.

    Shape: ``{"totals": {bucket: int}, "total": int, "signature": str}``.
    The ledger stores the same data under ``edge_signature`` instead of
    ``signature``; ``evaluate_per_pr`` accepts either key for forward
    compatibility with snapshots recorded before the rename.
    """
    totals = {
        "sim_to_validation": 0,
        "validation_to_sim": 0,
        "validation_to_physics": 0,
        "validation_to_weather": 0,
        "physics_to_sim": 0,
        "sim_to_physics": 0,
    }
    totals.update(bucket_counts)
    return {"totals": totals, "total": total, "signature": signature}


# ---------------------------------------------------------------------------
# evaluate_per_pr (R1 + R3)
# ---------------------------------------------------------------------------


def test_per_pr_warmup_when_no_history(guard):
    """Cold-start path: empty ledger -> exit 0 with WARMUP message."""
    code, msgs = guard.evaluate_per_pr(_current(215), last=None)
    assert code == 0
    assert any("WARMUP" in m for m in msgs)


def test_per_pr_passes_on_unchanged_total_and_signature(guard):
    last = _snapshot(215, signature="abc")
    code, msgs = guard.evaluate_per_pr(_current(215, signature="abc"), last)
    assert code == 0
    assert any("holds at 215" in m for m in msgs)


def test_per_pr_r1_fails_on_total_growth(guard):
    """R1: total > last_total -> exit 1 with per-bucket breakdown."""
    last = _snapshot(
        215,
        signature="abc",
        sim_to_validation=72,
        validation_to_sim=58,
    )
    cur = _current(
        217,
        signature="xyz",
        sim_to_validation=74,
        validation_to_sim=58,
    )
    code, msgs = guard.evaluate_per_pr(cur, last)
    assert code == 1
    assert any("R1 FAIL" in m and "215" in m and "217" in m for m in msgs)
    # Per-bucket breakdown must call out sim_to_validation growing.
    assert any("sim_to_validation: 72 -> 74" in m for m in msgs)


def test_per_pr_r3_fails_on_net_flat_swap(guard):
    """R3: same total, different signature -> exit 1 (net-flat swap)."""
    last = _snapshot(215, signature="aaaa")
    cur = _current(215, signature="bbbb")
    code, msgs = guard.evaluate_per_pr(cur, last)
    assert code == 1
    assert any("R3 FAIL" in m and "swap" in m for m in msgs)


def test_per_pr_accepts_downward_progress(guard):
    """Downward progress passes; message hints at the baseline-update workflow."""
    last = _snapshot(215, signature="abc")
    cur = _current(210, signature="xyz")
    code, msgs = guard.evaluate_per_pr(cur, last)
    assert code == 0
    assert any("downward progress" in m and "210" in m for m in msgs)


def test_per_pr_reads_legacy_signature_key(guard):
    """Snapshots recorded with ``edge_signature`` (ledger key) must work."""
    last = _snapshot(215)
    # Snapshot uses 'edge_signature'; current uses 'signature'.
    assert last["edge_signature"] == "s0"
    code, msgs = guard.evaluate_per_pr(_current(215, signature="s0"), last)
    assert code == 0


# ---------------------------------------------------------------------------
# evaluate_nightly (R1 + R2 + R3)
# ---------------------------------------------------------------------------


def test_nightly_r2_defers_below_threshold(guard):
    """R2 must defer when the ledger has fewer than STALE_THRESHOLD entries.

    The last snapshot's signature must match the current signature so R3
    (net-flat swap) does not fire first and mask the R2-deferred path.
    """
    history = {
        "schema_version": 1,
        "buckets": list(guard.BUCKETS),
        "snapshots": [
            _snapshot(215, signature="earlier"),
            _snapshot(215, signature="earlier"),
            _snapshot(215, signature="current"),
        ],
    }
    code, msgs = guard.evaluate_nightly(history, _current(215, signature="current"))
    assert code == 0
    assert any("R2 deferred" in m for m in msgs)


def test_nightly_r2_fails_when_frozen_across_threshold(guard):
    """R2 must fail when the last STALE_THRESHOLD snapshots are flat."""
    n = guard.STALE_THRESHOLD_NIGHTS
    history = {
        "schema_version": 1,
        "buckets": list(guard.BUCKETS),
        # n+1 snapshots, all at total=215 with the SAME signature so R3 also
        # passes -- we are isolating R2.
        "snapshots": [_snapshot(215, signature="same") for _ in range(n + 1)],
    }
    code, msgs = guard.evaluate_nightly(history, _current(215, signature="same"))
    assert code == 1
    assert any("R2 FAIL" in m and "frozen" in m for m in msgs)


def test_nightly_r2_passes_when_progress_inside_window(guard):
    """A drop within the trailing window resets R2; total stays below last."""
    n = guard.STALE_THRESHOLD_NIGHTS
    snaps = [_snapshot(215, signature="same") for _ in range(n - 1)]
    # The most recent snapshot dropped to 210 (downward progress inside window).
    snaps.append(_snapshot(210, signature="dropped"))
    history = {
        "schema_version": 1,
        "buckets": list(guard.BUCKETS),
        "snapshots": snaps,
    }
    # Current stays at 210 with the same signature as the latest snapshot.
    code, msgs = guard.evaluate_nightly(history, _current(210, signature="dropped"))
    assert code == 0
    assert any("R2 OK" in m and "within window" in m for m in msgs)


def test_nightly_r2_strict_fails_on_monotonic_growth(guard):
    """Issue #3385: R2 strict form must FAIL when total GREW over the window.

    The legacy R2 rule fired only when the trailing window was monotonic-flat
    at the current total; it silently admitted unbounded growth because each
    new high water mark reset the stale window. Issue #3385 tightens the
    rule to FAIL when the total has not *decreased* relative to the oldest
    snapshot in the trailing window (modulo the configured tolerance).
    """
    n = guard.STALE_THRESHOLD_NIGHTS
    snaps = [_snapshot(215, signature="baseline") for _ in range(n - 1)]
    snaps.append(_snapshot(230, signature="grew"))
    history = {
        "schema_version": 1,
        "buckets": list(guard.BUCKETS),
        "snapshots": snaps,
    }
    code, msgs = guard.evaluate_nightly(history, _current(230, signature="grew"))
    assert code == 1
    assert any("R2 FAIL" in m and "grew by 15" in m for m in msgs)


def test_nightly_r2_strict_tolerance_absorbs_small_growth(guard):
    """Issue #3385: ``r2_upward_tolerance`` lets a small growth pass without
    tripping the strict rule, absorbing legitimate one-shot feature work."""
    n = guard.STALE_THRESHOLD_NIGHTS
    snaps = [_snapshot(215, signature="baseline") for _ in range(n - 1)]
    snaps.append(_snapshot(220, signature="grew-a-bit"))
    history = {
        "schema_version": 1,
        "buckets": list(guard.BUCKETS),
        "snapshots": snaps,
    }
    # Tolerance=10 absorbs the +5 growth (215 -> 220).
    code, msgs = guard.evaluate_nightly(
        history,
        _current(220, signature="grew-a-bit"),
        r2_upward_tolerance=10,
    )
    assert code == 0
    assert any("R2 OK" in m and "tolerance=10" in m for m in msgs)


def test_nightly_r2_strict_tolerance_still_fails_beyond_threshold(guard):
    """Tolerance absorbs small growth but a bigger growth still trips R2."""
    n = guard.STALE_THRESHOLD_NIGHTS
    snaps = [_snapshot(215, signature="baseline") for _ in range(n - 1)]
    snaps.append(_snapshot(230, signature="grew-lots"))
    history = {
        "schema_version": 1,
        "buckets": list(guard.BUCKETS),
        "snapshots": snaps,
    }
    code, msgs = guard.evaluate_nightly(
        history,
        _current(230, signature="grew-lots"),
        r2_upward_tolerance=10,
    )
    assert code == 1
    assert any("R2 FAIL" in m and "grew by 15" in m for m in msgs)


def test_nightly_fails_on_growth_even_with_clean_window(guard):
    """R1 takes precedence over R2: a growth regression still fails nightly."""
    n = guard.STALE_THRESHOLD_NIGHTS
    history = {
        "schema_version": 1,
        "buckets": list(guard.BUCKETS),
        "snapshots": [_snapshot(215, signature="same") for _ in range(n)],
    }
    code, msgs = guard.evaluate_nightly(history, _current(220, signature="bigger"))
    assert code == 1
    assert any("R1 FAIL" in m for m in msgs)


# ---------------------------------------------------------------------------
# load_history / save_history / append_snapshot
# ---------------------------------------------------------------------------


def test_load_history_synthesises_empty_when_missing(guard, tmp_path):
    """A missing ledger file is the cold-start path -- not an error."""
    ledger = tmp_path / "cycle_baseline_history.json"
    data = guard.load_history(ledger)
    assert data["schema_version"] == guard.SCHEMA_VERSION
    assert data["snapshots"] == []
    assert data["buckets"] == list(guard.BUCKETS)


def test_load_history_rejects_corrupt_json(guard, tmp_path):
    ledger = tmp_path / "cycle_baseline_history.json"
    ledger.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="corrupt history"):
        guard.load_history(ledger)


def test_load_history_rejects_wrong_schema_version(guard, tmp_path):
    ledger = tmp_path / "cycle_baseline_history.json"
    ledger.write_text(
        json.dumps({"schema_version": 99, "snapshots": []}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="schema_version"):
        guard.load_history(ledger)


def test_load_history_rejects_missing_snapshots_key(guard, tmp_path):
    ledger = tmp_path / "cycle_baseline_history.json"
    ledger.write_text(
        json.dumps({"schema_version": 1, "buckets": []}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="missing 'snapshots'"):
        guard.load_history(ledger)


def test_save_history_round_trips(guard, tmp_path):
    ledger = tmp_path / "cycle_baseline_history.json"
    history = {
        "schema_version": guard.SCHEMA_VERSION,
        "buckets": list(guard.BUCKETS),
        "snapshots": [],
    }
    cur = _current(215, signature="abc")
    guard.append_snapshot(history, cur, commit="deadbee", source="rt")
    guard.save_history(history, ledger)
    reloaded = guard.load_history(ledger)
    assert len(reloaded["snapshots"]) == 1
    snap = reloaded["snapshots"][0]
    assert snap["total"] == 215
    assert snap["edge_signature"] == "abc"
    assert snap["commit"] == "deadbee"
    assert snap["source"] == "rt"
    # Stable formatting: trailing newline so git diffs stay clean.
    assert ledger.read_text(encoding="utf-8").endswith("}\n")


def test_append_snapshot_does_not_mutate_current(guard):
    """``current`` must not be modified by the append."""
    history = {
        "schema_version": guard.SCHEMA_VERSION,
        "buckets": list(guard.BUCKETS),
        "snapshots": [],
    }
    cur = _current(215, signature="abc")
    cur_before = copy.deepcopy(cur)
    guard.append_snapshot(history, cur, commit="x", source="t")
    assert cur == cur_before


# ---------------------------------------------------------------------------
# main() end-to-end via subprocess-style argv on a synthetic tree
# ---------------------------------------------------------------------------


def _seed_ledger(path: Path, total: int, signature: str) -> None:
    history = {
        "schema_version": 1,
        "buckets": list(_load_guard().BUCKETS),
        "snapshots": [_snapshot(total, signature=signature)],
    }
    path.write_text(json.dumps(history), encoding="utf-8")


def test_main_per_pr_passes_against_real_repo_ledger(
    guard, tmp_path, monkeypatch, capsys
):
    """Drive main() in per-PR mode against the real scan output + a seeded
    ledger. Uses a tmp ledger file so the committed baseline is not touched.

    The scan reads the real ``src/`` tree, so the seeded snapshot's total
    must equal the current real cycle count for R1 to pass. The signature
    is the real one computed by ``collect_current_edges``. The total is
    deliberately read dynamically (not hardcoded) so this test stays valid
    when a coverage-extension PR (e.g. Issue #2766) re-baselines the ledger.
    """
    real_current = guard.collect_current_edges(*guard._load_cycle_scripts())
    ledger = tmp_path / "history.json"
    _seed_ledger(ledger, real_current["total"], real_current["signature"])
    code = guard.main(["--history", str(ledger)])
    assert code == 0
    out = capsys.readouterr().out
    assert f"holds at {real_current['total']}" in out


def test_main_per_pr_fails_when_real_total_grew(guard, tmp_path, capsys):
    """Seed a lower total than the real one -> R1 must fire (exit 1).

    The seeded total is ``real_total - 1`` (dynamic, not hardcoded) so the
    test survives coverage-extension re-baselining (e.g. Issue #2766).
    """
    real_current = guard.collect_current_edges(*guard._load_cycle_scripts())
    seed_total = real_current["total"] - 1
    ledger = tmp_path / "history.json"
    _seed_ledger(ledger, total=seed_total, signature="never-matches-real")
    code = guard.main(["--history", str(ledger)])
    assert code == 1
    out = capsys.readouterr().out
    assert "R1 FAIL" in out
    assert str(seed_total) in out and str(real_current["total"]) in out


def test_main_per_pr_fails_on_real_signature_drift(guard, tmp_path, capsys):
    """Same total, wrong signature -> R3 (net-flat swap) must fire."""
    real_current = guard.collect_current_edges(*guard._load_cycle_scripts())
    ledger = tmp_path / "history.json"
    _seed_ledger(
        ledger,
        total=real_current["total"],
        signature="0" * 64,  # 64-char hex sha256 that won't match
    )
    code = guard.main(["--history", str(ledger)])
    assert code == 1
    out = capsys.readouterr().out
    assert "R3 FAIL" in out
    assert "swap" in out


def test_main_update_appends_and_exits_zero(guard, tmp_path, capsys):
    """`--update` appends a fresh snapshot and returns 0."""
    real_current = guard.collect_current_edges(*guard._load_cycle_scripts())
    ledger = tmp_path / "history.json"
    _seed_ledger(ledger, real_current["total"], real_current["signature"])
    code = guard.main(
        [
            "--history",
            str(ledger),
            "--update",
            "--source",
            "test-update",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Appended snapshot (1 -> 2 entries)" in out
    reloaded = guard.load_history(ledger)
    assert len(reloaded["snapshots"]) == 2
    assert reloaded["snapshots"][-1]["source"] == "test-update"
    # --reason omitted -> snapshot has no `reason` key (backward compatible).
    assert "reason" not in reloaded["snapshots"][-1]


def test_main_update_records_reason_when_provided(guard, tmp_path, capsys):
    """`--update --reason X` records the justification in the snapshot so a
    re-baseline (Issue #2810) is self-documenting in the ledger."""
    real_current = guard.collect_current_edges(*guard._load_cycle_scripts())
    ledger = tmp_path / "history.json"
    _seed_ledger(ledger, real_current["total"], real_current["signature"])
    code = guard.main(
        [
            "--history",
            str(ledger),
            "--update",
            "--source",
            "post-orchestration-2026-08-13",
            "--reason",
            "post-orchestration refactor wave shifted line numbers",
        ]
    )
    assert code == 0
    reloaded = guard.load_history(ledger)
    snap = reloaded["snapshots"][-1]
    assert snap["source"] == "post-orchestration-2026-08-13"
    assert snap["reason"] == "post-orchestration refactor wave shifted line numbers"


def test_append_snapshot_reason_is_optional(guard):
    """``reason=None`` (the default) omits the field for backward compat."""
    history = {
        "schema_version": guard.SCHEMA_VERSION,
        "buckets": list(guard.BUCKETS),
        "snapshots": [],
    }
    cur = _current(5, signature="abc")
    guard.append_snapshot(history, cur, commit="x", source="t")
    assert "reason" not in history["snapshots"][0]
    guard.append_snapshot(history, cur, commit="x", source="t", reason="why")
    assert history["snapshots"][1]["reason"] == "why"


def test_main_returns_2_on_corrupt_ledger(guard, tmp_path, capsys):
    ledger = tmp_path / "history.json"
    ledger.write_text("{not json", encoding="utf-8")
    code = guard.main(["--history", str(ledger)])
    assert code == 2
    err = capsys.readouterr().err
    assert "corrupt history" in err


# ---------------------------------------------------------------------------
# import-shape parity with the existing cycle scripts (regression guards)
# ---------------------------------------------------------------------------


def test_collect_current_edges_returns_all_six_buckets(guard):
    """The snapshot must report exactly the six documented bucket labels."""
    cur = guard.collect_current_edges(*guard._load_cycle_scripts())
    assert set(cur["totals"].keys()) == set(guard.BUCKETS)
    assert cur["total"] == sum(cur["totals"].values())
    # Signature is a 64-char lowercase hex sha256.
    assert len(cur["signature"]) == 64
    assert all(c in "0123456789abcdef" for c in cur["signature"])


def test_buckets_tuple_order_matches_committed_ledger(guard):
    """The committed ledger's ``buckets`` array must mirror ``BUCKETS``.

    A schema-drift here would silently re-order the snapshot's ``totals``
    dict and confuse any consumer that reads the ledger by index. The
    check below reads the real committed file (not a tmp copy).
    """
    ledger = guard.HISTORY_FILE
    if not ledger.exists():
        pytest.skip("ledger not seeded yet (cold-start)")
    data = json.loads(ledger.read_text(encoding="utf-8"))
    assert data["buckets"] == list(guard.BUCKETS)


# ---------------------------------------------------------------------------
# Reject the documented "frozen at 215" pathology end-to-end
# ---------------------------------------------------------------------------


def test_end_to_end_frozen_ledger_fails_nightly(guard, tmp_path, capsys):
    """The exact pathology issue #2768 documents: 14+ identical snapshots
    at total=215 must trip R2 in nightly mode. Per-PR mode must NOT trip
    (so an ordinary PR that doesn't touch the cycle still merges)."""
    n = guard.STALE_THRESHOLD_NIGHTS
    real_current = guard.collect_current_edges(*guard._load_cycle_scripts())
    history = {
        "schema_version": 1,
        "buckets": list(guard.BUCKETS),
        "snapshots": [
            _snapshot(real_current["total"], signature=real_current["signature"])
            for _ in range(n + 1)
        ],
    }
    ledger = tmp_path / "history.json"
    ledger.write_text(json.dumps(history), encoding="utf-8")
    # Per-PR mode passes: ordinary PRs are not blocked by the freeze.
    assert guard.main(["--history", str(ledger)]) == 0
    # Nightly mode fails with R2.
    code = guard.main(["--history", str(ledger), "--nightly"])
    assert code == 1
    out = capsys.readouterr().out
    assert "R2 FAIL" in out
    assert "frozen" in out


# ---------------------------------------------------------------------------
# Issue #2810: lineno-independent offender signature
#
# The offender identity used to include ``lineno``, so any refactor that
# inserted code *above* an unchanged cycle edge shifted its line number and
# flipped R3 (net-flat swap) even though no edge was added or removed. The
# signature now drops ``lineno`` (keeping ``file`` + scanned-line content);
# ``lineno`` survives only in the raw offender string for the report.
# ---------------------------------------------------------------------------


def test_signature_identity_strips_lineno(guard):
    """``_signature_identity`` must drop the lineno field, keeping file+text."""
    raw = "src/sim/thermal_model.rs:541: use crate::validation::Foo;"
    assert (
        guard._signature_identity(raw)
        == "src/sim/thermal_model.rs: use crate::validation::Foo;"
    )


def test_signature_identity_preserves_content_colons(guard):
    """Rust path separators (``::``) in the scanned line must survive."""
    raw = "src/validation/cases.rs:10: use crate::sim::thermal_model::Model;"
    assert guard._signature_identity(raw) == (
        "src/validation/cases.rs: use crate::sim::thermal_model::Model;"
    )


def test_signature_identity_fallback_on_unrecognised_shape(guard):
    """An offender without the ``file:line: text`` shape is passed through
    unchanged so the signature stays deterministic (defensive fallback)."""
    weird = "no-colons-here"
    assert guard._signature_identity(weird) == weird


def test_compute_signature_invariant_to_pure_line_shift(guard):
    """A benign line-shift refactor (insertion above an existing edge) moves
    the lineno but leaves file+content identical -> signature MUST NOT change.

    This is the exact false positive Issue #2810 documents: the
    post-orchestration refactor wave (#2798-#2804) shifted line numbers in
    cycle-edge files without altering any edge, yet R3 fired.
    """
    before = {
        "sim_to_validation": [
            "src/sim/a.rs:100: use crate::validation::Foo;",
            "src/sim/b.rs:50: use crate::validation::Bar;",
        ]
    }
    after = {
        "sim_to_validation": [
            # Both edges shifted down by 48 lines (e.g. an import block inserted
            # at the top of each file); content and file unchanged.
            "src/sim/a.rs:148: use crate::validation::Foo;",
            "src/sim/b.rs:98: use crate::validation::Bar;",
        ]
    }
    assert guard._compute_signature(before) == guard._compute_signature(after)


def test_compute_signature_detects_genuine_edge_swap(guard):
    """Removing one edge and adding a different-content edge (same total)
    MUST change the signature -- this is the swap R3 exists to catch."""
    before = {
        "sim_to_validation": [
            "src/sim/a.rs:100: use crate::validation::diagnostics::Foo;",
        ]
    }
    after = {
        "sim_to_validation": [
            # Same file/lineno, but a different (higher-criticality) edge.
            "src/sim/a.rs:100: use crate::validation::casespec::CaseSpec;",
        ]
    }
    assert guard._compute_signature(before) != guard._compute_signature(after)


def test_compute_signature_detects_edge_removal_and_addition_across_files(guard):
    """A swap spread across two files (remove in A, add in B) must still flip
    the signature even though file+lineno of each survivor is unchanged."""
    before = {
        "sim_to_validation": ["src/sim/a.rs:10: use crate::validation::Foo;"],
        "validation_to_sim": ["src/validation/c.rs:20: use crate::sim::Bar;"],
    }
    after = {
        "sim_to_validation": ["src/sim/b.rs:99: use crate::validation::Qux;"],
        "validation_to_sim": ["src/validation/c.rs:20: use crate::sim::Bar;"],
    }
    assert guard._compute_signature(before) != guard._compute_signature(after)


def test_compute_signature_counts_duplicate_edges_via_multiset(guard):
    """Two identical edges in the same file must register as count 2 in the
    multiset (so adding/removing one is detectable), even though their
    lineno-stripped identities are equal."""
    two = {
        "sim_to_validation": [
            "src/sim/a.rs:10: use crate::validation::Foo;",
            "src/sim/a.rs:20: use crate::validation::Foo;",
        ]
    }
    one = {
        "sim_to_validation": [
            "src/sim/a.rs:10: use crate::validation::Foo;",
        ]
    }
    assert guard._compute_signature(two) != guard._compute_signature(one)


class _MockScanModules:
    """Minimal stand-in for the acc/psc cycle-scan modules.

    Lets ``collect_current_edges`` be driven by synthetic offender lists so
    the lineno-independence contract can be exercised end-to-end (through the
    real signature path) without touching the real ``src/`` tree.
    """

    def __init__(self, sim_to_validation: list[str] | None = None) -> None:
        self._sim_to_validation = sim_to_validation or []

    def scan_sim_for_validation_deps(self):
        return list(self._sim_to_validation)

    def scan_validation_for_sim_deps(self):
        return []

    def scan_validation_for_physics_deps(self):
        return []

    def scan_validation_for_weather_deps(self):
        return []

    def scan_physics_for_sim_deps(self):
        return []

    def scan_protected_sim_files_for_physics_deps(self):
        return []


def test_collect_current_edges_signature_ignores_line_shift(guard):
    """End-to-end via ``collect_current_edges``: a pure lineno shift across a
    full re-scan must yield an identical signature (Issue #2810 contract)."""
    mock = _MockScanModules(["src/sim/a.rs:100: use crate::validation::Foo;"])
    before = guard.collect_current_edges(mock, mock)
    mock2 = _MockScanModules(["src/sim/a.rs:250: use crate::validation::Foo;"])
    after = guard.collect_current_edges(mock2, mock2)
    assert before["signature"] == after["signature"]
    # Total is unchanged (still one edge).
    assert before["total"] == after["total"] == 1


def test_collect_current_edges_preserves_lineno_in_offenders(guard):
    """``lineno`` is dropped from the signature but preserved in the raw
    offender strings (for the human-readable debug report)."""
    mock = _MockScanModules(["src/sim/a.rs:100: use crate::validation::Foo;"])
    res = guard.collect_current_edges(mock, mock)
    assert res["offenders"]["sim_to_validation"] == [
        "src/sim/a.rs:100: use crate::validation::Foo;"
    ]


def test_r3_does_not_fire_on_benign_line_shift(guard):
    """R3 must NOT trip when only line numbers drifted (the Issue #2810 fix).

    Builds two snapshots whose offender sets differ ONLY in lineno; their
    lineno-stripped signatures are equal, so ``evaluate_per_pr`` reports a
    clean hold rather than a net-flat swap.
    """
    before = guard._compute_signature(
        {"sim_to_validation": ["src/sim/a.rs:100: use crate::validation::Foo;"]}
    )
    after = guard._compute_signature(
        {"sim_to_validation": ["src/sim/a.rs:150: use crate::validation::Foo;"]}
    )
    assert before == after  # sanity: the signatures really are equal
    last = _snapshot(1, signature=before)
    cur = _current(1, signature=after)
    code, msgs = guard.evaluate_per_pr(cur, last)
    assert code == 0
    assert any("holds at 1" in m and "unchanged" in m for m in msgs)
    assert not any("R3 FAIL" in m for m in msgs)


def test_r3_fires_on_genuine_edge_swap(guard):
    """R3 MUST still trip on a real swap (different content, same total)."""
    before = guard._compute_signature(
        {"sim_to_validation": ["src/sim/a.rs:100: use crate::validation::Foo;"]}
    )
    after = guard._compute_signature(
        {"sim_to_validation": ["src/sim/a.rs:100: use crate::validation::Bar;"]}
    )
    last = _snapshot(1, signature=before)
    cur = _current(1, signature=after)
    code, msgs = guard.evaluate_per_pr(cur, last)
    assert code == 1
    assert any("R3 FAIL" in m and "swap" in m for m in msgs)
