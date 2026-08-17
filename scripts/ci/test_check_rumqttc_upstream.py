"""
Tests for ``scripts/check_rumqttc_upstream.py`` -- Issue #2853.

Regression guard for the upstream-watch script that surfaces when the
rumqttc security cluster (RUSTSEC-2025-0134 / 2026-0049/0098/0099/0104) can
be cleared. Mirrors the ``load_script`` + ``tmp_path`` mock-repo pattern
from ``test_check_known_issues_stale.py`` and
``test_check_required_checks_sync.py``:

* load the script as a fresh module via the shared ``load_script`` fixture,
* redirect the module-level path constants (``CARGO_LOCK``,
  ``RUMQTTC_CARGO_TOML``) at a synthetic ``tmp_path`` tree,
* inject stubbed HTTP responses via ``monkeypatch.setattr`` against
  ``fetch_upstream_pr`` / ``fetch_crates_io_latest`` /
  ``fetch_crates_io_deps`` so the script can be exercised without network
  access, and
* drive ``main()`` through clean and planted-violation scenarios.

The script's network fetches are intentionally thin wrappers (they call
``urllib.request.urlopen``) so we replace them at the public function
boundary rather than mocking the lower-level transport.

Why this matters (issue #2853): the cluster cannot be cleared until
bytebeamio/rumqtt#1037 lands and a new ``rumqttc`` release ships. A
regression that turns the watcher into a silent pass (e.g. a stale
``_semver_tuple`` parse, a flipped truthy check) would mean the cluster
silently keeps blocking ``cargo audit --deny warnings`` forever. The
gate below is what makes the watcher enforceable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SCRIPT_NAME = "check_rumqttc_upstream"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the upstream-watch script."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Point the script's ``CARGO_LOCK`` and ``RUMQTTC_CARGO_TOML`` at
    synthetic files in ``tmp_path`` and return both resolved paths.

    Both constants are computed at import time from ``Path(__file__)``.
    Each test that wants a synthetic fixture must therefore redirect the
    constants before calling ``main()``.
    """
    lock = tmp_path / "Cargo.lock"
    cargo_toml = tmp_path / "crates" / "fluxion-twin" / "Cargo.toml"
    monkeypatch.setattr(checker, "CARGO_LOCK", lock)
    monkeypatch.setattr(checker, "RUMQTTC_CARGO_TOML", cargo_toml)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    return lock, cargo_toml


def _write_rumqttc_lock(lock: Path, version: str = "0.25.1") -> Path:
    """Write a minimal ``Cargo.lock`` that resolves ``rumqttc = <version>``."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(
        "# synthetic Cargo.lock for check_rumqttc_upstream tests\n"
        "[[package]]\n"
        'name = "fluxion-twin"\n'
        'version = "0.0.0"\n'
        '"fluxion-twin" = []\n'
        "[[package]]\n"
        'name = "rumqttc"\n'
        f'version = "{version}"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        "dependencies = []\n",
        encoding="utf-8",
    )
    return lock


def _write_rumqttc_decl(cargo_toml: Path, constraint: str = "0.25") -> Path:
    """Write a minimal ``fluxion-twin/Cargo.toml`` that pins ``rumqttc``."""
    cargo_toml.parent.mkdir(parents=True, exist_ok=True)
    cargo_toml.write_text(
        "# synthetic Cargo.toml for check_rumqttc_upstream tests\n"
        "[dependencies]\n"
        f'rumqttc = "{constraint}"\n',
        encoding="utf-8",
    )
    return cargo_toml


def _stub_network(
    checker,
    monkeypatch,
    *,
    pr_merged: bool = False,
    crates_io_latest: str | None = "0.25.1",
    release_pulls_fixed: bool = False,
) -> None:
    """Inject stubbed responses for the three network fetches the script
    makes in ``--online`` mode.

    Each stub returns a deterministic dict that exercises the parser
    paths the production code uses (the GitHub API payload shape, the
    crates.io ``crate`` envelope, and the crates.io ``dependencies``
    list). Tests that want to simulate a network failure can set
    ``crates_io_latest=None``.
    """

    def fake_pr():
        return {
            "state": "closed" if pr_merged else "open",
            "merged": pr_merged,
            "merged_at": "2026-08-16T00:00:00Z" if pr_merged else None,
            "mergeable": True,
            "mergeable_state": "clean" if pr_merged else "blocked",
        }

    def fake_latest():
        return crates_io_latest

    def fake_deps(_version):
        if release_pulls_fixed:
            return [{"crate_id": "rustls-webpki", "req": ">=0.103.13"}]
        return [{"crate_id": "rustls-webpki", "req": "=0.102.8"}]

    monkeypatch.setattr(checker, "fetch_upstream_pr", fake_pr)
    monkeypatch.setattr(checker, "fetch_crates_io_latest", fake_latest)
    monkeypatch.setattr(checker, "fetch_crates_io_deps", fake_deps)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_version(v: str) -> tuple[int, int, int]:
    """Local copy of the script's semver parser (used in assertions)."""
    parts = v.split(".")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


# ---------------------------------------------------------------------------
# Cargo.lock + Cargo.toml parsing (offline)
# ---------------------------------------------------------------------------


def test_parse_cargo_lock_rumqttc_returns_version(checker, tmp_path):
    """Locked ``rumqttc = 0.25.1`` is parsed correctly."""
    lock = tmp_path / "Cargo.lock"
    _write_rumqttc_lock(lock, "0.25.1")
    assert checker._parse_cargo_lock_rumqttc(lock) == "0.25.1"


def test_parse_cargo_lock_rumqttc_returns_none_when_absent(checker, tmp_path):
    """Lock file without ``rumqttc`` → ``None`` (e.g. ``fluxion-twin``
    feature disabled)."""
    lock = tmp_path / "Cargo.lock"
    lock.write_text("# no rumqttc here\n", encoding="utf-8")
    assert checker._parse_cargo_lock_rumqttc(lock) is None


def test_parse_cargo_lock_rumqttc_returns_none_when_missing(checker, tmp_path):
    """Missing Cargo.lock → ``None`` (caller maps to exit 2)."""
    assert checker._parse_cargo_lock_rumqttc(tmp_path / "absent.lock") is None


def test_parse_rumqttc_decl_returns_constraint(checker, tmp_path):
    """Declared ``rumqttc = "0.25"`` is parsed correctly."""
    cargo_toml = tmp_path / "crates" / "fluxion-twin" / "Cargo.toml"
    _write_rumqttc_decl(cargo_toml, "0.25")
    assert checker._parse_rumqttc_decl(cargo_toml) == "0.25"


def test_parse_rumqttc_decl_returns_none_when_missing(checker, tmp_path):
    """Missing Cargo.toml → ``None`` (caller surfaces as 'unknown')."""
    assert checker._parse_rumqttc_decl(tmp_path / "absent" / "Cargo.toml") is None


# ---------------------------------------------------------------------------
# Constraint parsing (the rumqttc→rustls-webpki fix detector)
# ---------------------------------------------------------------------------


def test_constraint_mentions_fixed_webpki_accepts_pinned_103(checker):
    """``>=0.103.13`` is recognised as a fixed webpki constraint."""
    assert checker._constraint_mentions_fixed_webpki(">=0.103.13") is True


def test_constraint_mentions_fixed_webpki_accepts_pessimistic_103(checker):
    """``>0.103.0, <0.104.0`` (a common caret-style pin) is recognised."""
    assert checker._constraint_mentions_fixed_webpki("^0.103") is True


def test_constraint_mentions_fixed_webpki_rejects_102(checker):
    """``=0.102.8`` is recognised as NOT fixed (the locked transitive)."""
    assert checker._constraint_mentions_fixed_webpki("=0.102.8") is False


def test_constraint_mentions_fixed_webpki_rejects_floor_in_102(checker):
    """``>=0.102, <0.103`` is rejected even though 0.103 appears in the floor."""
    assert checker._constraint_mentions_fixed_webpki(">=0.102, <0.103") is False


def test_constraint_mentions_fixed_webpki_rejects_empty(checker):
    """Empty / None → ``False`` (defensive)."""
    assert checker._constraint_mentions_fixed_webpki("") is False
    assert checker._constraint_mentions_fixed_webpki(">=1.0.0") is False


# ---------------------------------------------------------------------------
# main() — offline scenarios
# ---------------------------------------------------------------------------


def test_main_offline_returns_zero_with_clean_lock(
    checker, tmp_path, monkeypatch, capsys
):
    """Offline default: locked ``rumqttc = 0.25.1``, no network → exit 0."""
    lock, cargo_toml = _redirect(checker, tmp_path, monkeypatch)
    _write_rumqttc_lock(lock, "0.25.1")
    _write_rumqttc_decl(cargo_toml, "0.25")
    rc = checker.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "rumqttc = 0.25.1" in out
    assert "Mode:                 offline" in out
    assert "no upstream fix yet" in out


def test_main_offline_returns_two_when_lock_missing(
    checker, tmp_path, monkeypatch, capsys
):
    """Missing ``Cargo.lock`` → exit 2 (script error)."""
    _redirect(checker, tmp_path, monkeypatch)
    rc = checker.main([])
    err = capsys.readouterr().err
    assert rc == 2
    assert "Cargo.lock" in err


def test_main_offline_returns_two_when_rumqttc_not_in_lock(
    checker, tmp_path, monkeypatch, capsys
):
    """Lock without ``rumqttc`` → exit 2 (fluxion-twin feature may be
    disabled — script cannot reason about cluster state without the
    crate)."""
    lock, cargo_toml = _redirect(checker, tmp_path, monkeypatch)
    lock.write_text("# no rumqttc here\n", encoding="utf-8")
    _write_rumqttc_decl(cargo_toml, "0.25")
    rc = checker.main([])
    err = capsys.readouterr().err
    assert rc == 2
    assert "rumqttc not resolved" in err


# ---------------------------------------------------------------------------
# main() — online scenarios (network stubs)
# ---------------------------------------------------------------------------


def test_main_online_returns_zero_when_pr_open_and_no_fix(
    checker, tmp_path, monkeypatch, capsys
):
    """Online poll: PR open, no new crates.io release → exit 0."""
    lock, cargo_toml = _redirect(checker, tmp_path, monkeypatch)
    _write_rumqttc_lock(lock, "0.25.1")
    _write_rumqttc_decl(cargo_toml, "0.25")
    _stub_network(checker, monkeypatch, pr_merged=False, crates_io_latest="0.25.1")
    rc = checker.main(["--online"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "not merged" in out
    assert "no upstream fix yet" in out


def test_main_online_returns_zero_when_release_does_not_pull_fixed_webpki(
    checker, tmp_path, monkeypatch, capsys
):
    """A new crates.io release that still pulls ``rustls-webpki = 0.102.x``
    must NOT trigger the fix-available branch — the cluster would
    re-surface."""
    lock, cargo_toml = _redirect(checker, tmp_path, monkeypatch)
    _write_rumqttc_lock(lock, "0.25.1")
    _write_rumqttc_decl(cargo_toml, "0.25")
    _stub_network(
        checker,
        monkeypatch,
        pr_merged=True,
        crates_io_latest="0.26.0",
        release_pulls_fixed=False,
    )
    rc = checker.main(["--online"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no upstream fix yet" in out


def test_main_online_returns_one_when_pr_merged_and_release_fixes_webpki(
    checker, tmp_path, monkeypatch, capsys
):
    """Strict success: PR #1037 merged AND crates.io has a new release
    that pins ``rustls-webpki >= 0.103.13`` → exit 1 with action banner."""
    lock, cargo_toml = _redirect(checker, tmp_path, monkeypatch)
    _write_rumqttc_lock(lock, "0.25.1")
    _write_rumqttc_decl(cargo_toml, "0.25")
    _stub_network(
        checker,
        monkeypatch,
        pr_merged=True,
        crates_io_latest="0.26.0",
        release_pulls_fixed=True,
    )
    rc = checker.main(["--online"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "MERGED" in out
    assert "ACTION: upstream fix is available" in out
    assert "cargo audit --deny warnings" in out


def test_main_online_strict_returns_one_even_when_pr_not_merged(
    checker, tmp_path, monkeypatch, capsys
):
    """``--strict`` override: a new crates.io release that pulls fixed
    webpki is enough to trigger exit 1, even when the upstream PR is
    still open. This is the operator-override path for the rare case
    where upstream publishes a release without going through the PR."""
    lock, cargo_toml = _redirect(checker, tmp_path, monkeypatch)
    _write_rumqttc_lock(lock, "0.25.1")
    _write_rumqttc_decl(cargo_toml, "0.25")
    _stub_network(
        checker,
        monkeypatch,
        pr_merged=False,
        crates_io_latest="0.26.0",
        release_pulls_fixed=True,
    )
    rc = checker.main(["--online", "--strict"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "ACTION" in out


def test_main_online_handles_cratesio_lookup_failure(
    checker, tmp_path, monkeypatch, capsys
):
    """crates.io lookup fails (network down) → script degrades to
    'crates.io lookup failed' and exits 0 (offline-equivalent)."""
    lock, cargo_toml = _redirect(checker, tmp_path, monkeypatch)
    _write_rumqttc_lock(lock, "0.25.1")
    _write_rumqttc_decl(cargo_toml, "0.25")
    _stub_network(checker, monkeypatch, pr_merged=False, crates_io_latest=None)
    rc = checker.main(["--online"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "lookup failed" in out


def test_main_json_returns_machine_readable_summary(
    checker, tmp_path, monkeypatch, capsys
):
    """``--json`` flag emits a single-line JSON summary on stdout with
    the expected keys. Used by the scheduled CI job to feed a status
    badge / metrics scraper."""
    lock, cargo_toml = _redirect(checker, tmp_path, monkeypatch)
    _write_rumqttc_lock(lock, "0.25.1")
    _write_rumqttc_decl(cargo_toml, "0.25")
    _stub_network(checker, monkeypatch, pr_merged=False, crates_io_latest="0.25.1")
    rc = checker.main(["--online", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    summary = json.loads(out.strip())
    assert summary["locked_rumqttc"] == "0.25.1"
    assert summary["declared_constraint"] == "0.25"
    assert summary["crates_io_latest"] == "0.25.1"
    assert summary["upstream_pr_merged"] is False
    assert summary["fix_available"] is False
    assert "checked_on" in summary
    assert len(summary["tracked_advisories"]) == 5


# ---------------------------------------------------------------------------
# Module-level constants pin
# ---------------------------------------------------------------------------


def test_tracked_advisories_pinned(checker):
    """The five documented advisories must remain in the script's
    surface list. A regression that drops one would silently shrink
    the cluster the watcher covers."""
    assert set(checker.TRACKED_ADVISORIES) == {
        "RUSTSEC-2025-0134",
        "RUSTSEC-2026-0049",
        "RUSTSEC-2026-0098",
        "RUSTSEC-2026-0099",
        "RUSTSEC-2026-0104",
    }


def test_min_fixed_webpki_pinned(checker):
    """The strict minimum webpki floor (``0.103.13``) must not be
    regressed to ``0.103.10`` or ``0.103.12`` — those are insufficient
    because -0104 (CRL panic) was only fixed in 0.103.13."""
    assert checker.MIN_FIXED_WEBPKI == "0.103.13"


def test_semver_tuple_handles_partial_strings(checker):
    """``_semver_tuple`` must not raise on empty / partial strings —
    it's called on user-controlled input from the crates.io payload."""
    assert checker._semver_tuple("") == (0, 0, 0)
    assert checker._semver_tuple("0.103") == (0, 0, 0)
    assert checker._semver_tuple("0.103.13") == (0, 103, 13)
