"""
Tests for ``scripts/check_tauri_csp.py`` -- Issue #3727.

Regression coverage for the ``fluxion-tauri/src-tauri/tauri.conf.json``
Content-Security-Policy gate. Mirrors the ``load_script`` + ``tmp_path``
mock-repo pattern from ``test_check_audit_config_unique.py`` /
``test_check_known_issues_stale.py``:

* load the script as a fresh module via the shared ``load_script``
  fixture,
* monkey-patch its module-level ``TAURI_CONF`` to a synthetic
  ``tmp_path`` file, then
* drive ``main()`` / ``_check_directives`` / ``_check_security_block``
  through clean (strict production CSP) and violation scenarios (null
  csp, missing required directive, permissive ``script-src
  'unsafe-inline'``, non-JSON file).

The acceptance criteria from issue #3727 are realised as:

* (a) *clean* fixture (strict ``csp`` with all REQUIRED_DIRECTIVES,
  no forbidden tokens) -> ``main()`` returns ``0`` and prints ``OK``.
* (b) *null csp* fixture (``"csp": null``) -> ``main()`` returns ``1``
  with the Issue #3727 remediation banner.
* (c) *missing directive* fixture (e.g. ``script-src`` removed) ->
  ``main()`` returns ``1`` naming the missing directive and required
  token.
* (d) *permissive script-src* fixture (adds ``'unsafe-inline'`` to
  ``script-src``) -> ``main()`` returns ``1`` flagging the relaxation
  as a production violation.
* (e) *non-JSON conf* -> ``main()`` returns ``1`` with a JSON parse
  error.
* (f) *missing security block* -> ``main()`` returns ``1`` with the
  ``app.security block is missing entirely`` banner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SCRIPT_NAME = "check_tauri_csp"

STRICT_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "font-src 'self'; "
    "connect-src 'self' ipc: http://ipc.localhost; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'none'; "
    "frame-ancestors 'none'; "
    "upgrade-insecure-requests"
)


def _conf(csp_value, *, include_devcsp: bool = True) -> dict:
    """Build a minimal ``tauri.conf.json`` dict with the given csp value."""
    security: dict = {"csp": csp_value}
    if include_devcsp:
        security["devCsp"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "connect-src 'self' ipc: http://ipc.localhost "
            "ws://localhost:1420 http://localhost:1420"
        )
    return {
        "$schema": "https://schema.tauri.app/config/2",
        "productName": "Fluxion Viewer",
        "version": "0.1.0",
        "identifier": "ai.fluxion.viewer",
        "build": {
            "devUrl": "http://localhost:1420",
            "frontendDist": "../frontend/dist",
        },
        "app": {
            "withGlobalTauri": True,
            "windows": [{"title": "Fluxion", "width": 1280, "height": 720}],
            "security": security,
        },
        "bundle": {"active": True, "targets": "all", "icon": ["icons/icon.png"]},
    }


def _write_conf(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the tauri-csp check script."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, tmp_path: Path, monkeypatch) -> Path:
    """Point the script's ``TAURI_CONF`` at a synthetic file in ``tmp_path``.

    The constant is a module-level ``Path`` resolved at import time, so the
    freshly-loaded module carries the real repo path. Each test that
    wants a synthetic fixture must redirect the constant before calling
    ``main()``.
    """
    target = tmp_path / "fluxion-tauri" / "src-tauri" / "tauri.conf.json"
    monkeypatch.setattr(checker, "TAURI_CONF", target)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    return target


# ---------------------------------------------------------------------------
# _parse_csp_directives / _token_is_present — pure-function primitives
# ---------------------------------------------------------------------------


def test_parse_csp_directives_splits_on_semicolons(checker):
    """``a 'x'; b 'y'`` → ``{a: "'x'", b: "'y'"}``."""
    parsed = checker._parse_csp_directives("default-src 'self'; script-src 'self'")
    assert parsed == {"default-src": "'self'", "script-src": "'self'"}


def test_parse_csp_directives_last_wins_on_duplicates(checker):
    """``script-src 'a'; script-src 'b'`` → ``{script-src: "'b'"}``.

    CSP spec resolves duplicates last-wins; we mirror that so a
    regression that adds a permissive second ``script-src`` is caught.
    """
    parsed = checker._parse_csp_directives(
        "script-src 'self'; script-src 'self' 'unsafe-inline'"
    )
    assert parsed == {"script-src": "'self' 'unsafe-inline'"}


def test_token_is_present_does_not_match_substring_overlap(checker):
    """``'self'`` must NOT be detected inside ``'unsafe-inline'``."""
    assert checker._token_is_present("'unsafe-inline'", "'self'") is False
    assert checker._token_is_present("'self'", "'unsafe-inline'") is False


def test_token_is_present_matches_word_boundary(checker):
    """``'self'`` matches ``'self' `` (space-delimited) but not
    ``myself`` (substring)."""
    assert checker._token_is_present("'self' https://x", "'self'") is True
    assert checker._token_is_present("myself", "'self'") is False


# ---------------------------------------------------------------------------
# main() — clean / violation scenarios
# ---------------------------------------------------------------------------


def test_main_returns_zero_on_strict_production_csp(checker, tmp_path, monkeypatch, capsys):
    """Clean fixture (strict CSP, all required directives, no forbidden
    tokens) -> exit 0 with OK banner."""
    target = _redirect(checker, tmp_path, monkeypatch)
    _write_conf(target, _conf(STRICT_CSP))
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out
    assert "strict CSP enforced" in out


def test_main_returns_one_when_csp_is_null(checker, tmp_path, monkeypatch, capsys):
    """Issue #3727 planted-violation: ``"csp": null`` -> exit 1."""
    target = _redirect(checker, tmp_path, monkeypatch)
    _write_conf(target, _conf(None))
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "csp is null" in out
    assert "Issue #3727" in out
    # Remediation guidance
    assert "Remediation" in out
    assert "devCsp" in out


def test_main_returns_one_when_security_block_missing(checker, tmp_path, monkeypatch, capsys):
    """No ``app.security`` block at all -> exit 1 with explicit banner."""
    target = _redirect(checker, tmp_path, monkeypatch)
    payload = _conf(STRICT_CSP)
    del payload["app"]["security"]
    _write_conf(target, payload)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "app.security block is missing entirely" in out


def test_main_returns_one_when_required_directive_missing(checker, tmp_path, monkeypatch, capsys):
    """Drop ``script-src`` entirely -> exit 1 naming the missing directive."""
    target = _redirect(checker, tmp_path, monkeypatch)
    csp_without_script_src = (
        "default-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'"
    )
    _write_conf(target, _conf(csp_without_script_src))
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "script-src" in out
    assert "'self'" in out


def test_main_returns_one_when_required_token_missing(checker, tmp_path, monkeypatch, capsys):
    """``default-src 'unsafe-inline'`` instead of ``'self'`` -> exit 1
    flagging the required token absence."""
    target = _redirect(checker, tmp_path, monkeypatch)
    csp_with_wrong_default = (
        "default-src 'unsafe-inline'; "
        "script-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'"
    )
    _write_conf(target, _conf(csp_with_wrong_default))
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "default-src" in out
    assert "'self'" in out


def test_main_returns_one_when_script_src_includes_unsafe_inline(checker, tmp_path, monkeypatch, capsys):
    """``script-src 'self' 'unsafe-inline'`` -> exit 1 (production
    relaxation). The whole point of the gate is to keep
    ``'unsafe-inline'`` out of the production policy."""
    target = _redirect(checker, tmp_path, monkeypatch)
    csp_with_unsafe_inline = STRICT_CSP.replace(
        "script-src 'self'", "script-src 'self' 'unsafe-inline'"
    )
    _write_conf(target, _conf(csp_with_unsafe_inline))
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "'unsafe-inline'" in out
    assert "must NOT include" in out


def test_main_returns_one_when_script_src_includes_unsafe_eval(checker, tmp_path, monkeypatch, capsys):
    """``script-src 'self' 'unsafe-eval'`` -> exit 1 (same reasoning as
    ``'unsafe-inline'``)."""
    target = _redirect(checker, tmp_path, monkeypatch)
    csp_with_unsafe_eval = STRICT_CSP.replace(
        "script-src 'self'", "script-src 'self' 'unsafe-eval'"
    )
    _write_conf(target, _conf(csp_with_unsafe_eval))
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "'unsafe-eval'" in out


def test_main_returns_one_when_object_src_is_not_none(checker, tmp_path, monkeypatch, capsys):
    """``object-src 'self'`` -> exit 1 (defeats the
    plugin-blocker rationale for ``object-src 'none'``)."""
    target = _redirect(checker, tmp_path, monkeypatch)
    csp_with_permissive_object = STRICT_CSP.replace(
        "object-src 'none'", "object-src 'self'"
    )
    _write_conf(target, _conf(csp_with_permissive_object))
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "object-src" in out
    assert "'none'" in out


def test_main_returns_one_when_conf_is_invalid_json(checker, tmp_path, monkeypatch, capsys):
    """Non-JSON conf file -> exit 1 with parse error."""
    target = _redirect(checker, tmp_path, monkeypatch)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{ this is not json", encoding="utf-8")
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "not valid JSON" in out


def test_main_returns_one_when_conf_is_missing(checker, tmp_path, monkeypatch, capsys):
    """Missing conf file -> exit 1 (Issue #3727 — silent skip is not
    acceptable for a security gate)."""
    _redirect(checker, tmp_path, monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "not found" in out


def test_main_accepts_csp_with_devcsp_relaxations(checker, tmp_path, monkeypatch, capsys):
    """devCsp carries the Vite HMR relaxations
    (``'unsafe-inline'``, ``'unsafe-eval'``, ``ws://localhost:1420``);
    only the production ``csp`` is gated. The devCsp relaxations must
    NOT cause a failure."""
    target = _redirect(checker, tmp_path, monkeypatch)
    relaxed_devcsp = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self' ipc: http://ipc.localhost "
        "ws://localhost:1420 http://localhost:1420; "
        "object-src 'none'; "
        "base-uri 'self'"
    )
    payload = _conf(STRICT_CSP, include_devcsp=False)
    payload["app"]["security"]["devCsp"] = relaxed_devcsp
    _write_conf(target, payload)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out


def test_main_pins_against_real_tauri_conf(checker, capsys):
    """Sanity test: the real ``fluxion-tauri/src-tauri/tauri.conf.json``
    carries a strict production ``csp`` after the Issue #3727 fix. A
    regression in the scanner (or someone reverting the conf to
    ``"csp": null``) surfaces locally before it does in CI."""
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, (
        "real tauri.conf.json failed the strict-CSP gate; the "
        "fluxion-tauri/src-tauri/tauri.conf.json regression was "
        "either reintroduced or the scanner logic drifted.\n"
        f"--- script output ---\n{out}\n--- end ---"
    )
    assert "strict CSP enforced" in out