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
