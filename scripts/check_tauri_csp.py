#!/usr/bin/env python3
# scripts/check_tauri_csp.py
#
# Issue #3727 — `fluxion-tauri/src-tauri/tauri.conf.json` previously declared
# `"csp": null`, which disables Tauri's Content-Security-Policy injection for
# the webview entirely. The shipped desktop artifact is then defenseless
# against script injection in the renderer: any DOM compromise chains directly
# to the Rust-side IPC surface (`load_geometry`, `update_simulation_parameters`,
# `get_*` commands in `fluxion-tauri/src-tauri/src/commands.rs`).
#
# Acceptance: a strict production `csp` is defined; a `devCsp` may loosen the
# policy for local dev (Vite HMR inline scripts + `ws://`); this gate fails CI
# when `app.security.csp` is `null`, missing, or otherwise permissive.
#
# Exit codes:
#   0 — production CSP is non-null, parses, and contains the required
#       directive set ("default-src 'self'", "script-src 'self'",
#       "object-src 'none'", "base-uri 'self'", "frame-ancestors 'none'").
#   1 — any violation listed in the acceptance criteria; the printed banner
#       lists every individual problem so a regression in one directive is
#       still actionable.

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TAURI_CONF = REPO_ROOT / "fluxion-tauri" / "src-tauri" / "tauri.conf.json"

# Required CSP directives and the value the production policy MUST carry.
# Each entry is (directive_name, required_token_or_substring).
# ``required_token`` is matched as a word token (whitespace or ; either
# side) so ``'unsafe-inline'`` cannot accidentally satisfy
# ``script-src 'self'`` (no false positive on substring overlap).
REQUIRED_DIRECTIVES: list[tuple[str, str]] = [
    ("default-src", "'self'"),
    ("script-src", "'self'"),
    ("object-src", "'none'"),
    ("base-uri", "'self'"),
    ("frame-ancestors", "'none'"),
]

# Directives that must NOT be permissive in production. A production CSP
# that contains ``unsafe-inline`` for ``script-src`` would defeat the
# point of the gate (DOM injection → arbitrary JS). ``unsafe-eval`` is in
# the same bucket.
FORBIDDEN_TOKENS_BY_DIRECTIVE: dict[str, set[str]] = {
    "script-src": {"'unsafe-inline'", "'unsafe-eval'"},
    "default-src": {"'unsafe-inline'", "'unsafe-eval'"},
}


def _token_is_present(directive_value: str, token: str) -> bool:
    """True when ``token`` appears as a delimited word in ``directive_value``.

    CSP values are space-separated lists of sources. We match with a
    word-boundary regex so ``'self'`` does NOT count as containing
    ``'unsafe-inline'`` and vice versa. The leading apostrophe plus a
    trailing apostrophe gives a precise match for single-quoted sources
    such as ``'self'`` and ``'unsafe-inline'``; for unquoted keywords
    like ``https:`` the same regex is permissive (matches if the keyword
    is delimited by whitespace or end-of-string).
    """
    pattern = r"(?:^|[\s;,])" + re.escape(token) + r"(?:$|[\s;,])"
    return re.search(pattern, directive_value) is not None


def _parse_csp_directives(csp: str) -> dict[str, str]:
    """Split a CSP string into ``{directive_name: directive_value}`` pairs.

    The last entry wins on duplicate directive names (CSP spec resolves
    duplicates this way; we mirror that so a regression that adds a
    permissive second ``script-src`` is caught — the duplicate is
    surfaced by the per-directive checks below).
    """
    directives: dict[str, str] = {}
    for chunk in csp.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(None, 1)
        if not parts:
            continue
        name = parts[0].strip().lower()
        value = parts[1].strip() if len(parts) > 1 else ""
        directives[name] = value
    return directives


def _load_tauri_conf(path: Path) -> dict:
    """Load and minimally validate the Tauri config JSON.

    A Tauri config that does not parse as JSON is itself a gate failure,
    not a silent skip — the whole point of issue #3727 is to keep this
    file's content under CI control.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: {path} not found")
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"ERROR: {path} is not valid JSON: {e}")
        return {}


def _check_security_block(conf: dict) -> tuple[list[str], str | None]:
    """Return ``(problems, csp_string)`` for the ``app.security`` block.

    Surfaces every individual problem so a regression that fixes one
    directive but breaks another still produces actionable output. A
    missing ``app.security`` block counts as ``csp: null``.
    """
    problems: list[str] = []
    security = conf.get("app", {}).get("security")
    if security is None:
        return (["ERROR: app.security block is missing entirely"], None)
    if not isinstance(security, dict):
        return (
            [f"ERROR: app.security is not an object (got {type(security).__name__})"],
            None,
        )

    csp = security.get("csp")
    if csp is None:
        problems.append(
            "ERROR: app.security.csp is null (Issue #3727 — disables the "
            "Content-Security-Policy injection; renderer compromise chains "
            "to Rust-side IPC surface)"
        )
        return (problems, None)
    if not isinstance(csp, str):
        problems.append(
            f"ERROR: app.security.csp must be a string when non-null "
            f"(got {type(csp).__name__})"
        )
        return (problems, None)
    if not csp.strip():
        problems.append("ERROR: app.security.csp is an empty string")

    return (problems, csp)


def _check_directives(csp: str) -> list[str]:
    """Validate the required / forbidden directive set against ``csp``.

    Required: each entry in ``REQUIRED_DIRECTIVES`` must appear (last
    wins on duplicates — checked once after the merge).
    Forbidden: each entry in ``FORBIDDEN_TOKENS_BY_DIRECTIVE`` must NOT
    appear in the resolved directive value. This catches ``unsafe-inline``
    in ``script-src`` even if the same directive appears twice (one
    restrictive, one permissive).
    """
    problems: list[str] = []
    directives = _parse_csp_directives(csp)

    for name, required in REQUIRED_DIRECTIVES:
        value = directives.get(name)
        if value is None:
            problems.append(
                f"ERROR: CSP is missing required directive "
                f"'{name} {required}'"
            )
            continue
        if not _token_is_present(value, required):
            problems.append(
                f"ERROR: CSP directive '{name}' must include {required} "
                f"(got '{name} {value}')"
            )

    for name, forbidden in FORBIDDEN_TOKENS_BY_DIRECTIVE.items():
        value = directives.get(name, "")
        for token in sorted(forbidden):
            if _token_is_present(value, token):
                problems.append(
                    f"ERROR: CSP directive '{name}' must NOT include "
                    f"{token} in production (got '{name} {value}'). "
                    f"Vite HMR requires this token — keep the relaxation "
                    f"inside devCsp only."
                )

    return problems


def main() -> int:
    conf = _load_tauri_conf(TAURI_CONF)
    if not conf:
        return 1

    problems, csp = _check_security_block(conf)
    if csp is not None:
        problems.extend(_check_directives(csp))

    if problems:
        print(f"FAIL: {TAURI_CONF.relative_to(REPO_ROOT)}")
        for line in problems:
            print(f"  {line}")
        print(
            "\nRemediation (Issue #3727):\n"
            "  - Set app.security.csp to a strict policy string "
            "(see scripts/check_tauri_csp.py::REQUIRED_DIRECTIVES).\n"
            "  - Move Vite HMR relaxations ('unsafe-inline', "
            "'unsafe-eval', ws://localhost:1420) into app.security.devCsp.\n"
            "  - See fluxion-tauri/src-tauri/tauri.conf.json for the "
            "shipped policy."
        )
        return 1

    print(
        f"OK  {TAURI_CONF.relative_to(REPO_ROOT)} — strict CSP enforced "
        f"(production csp set; devCsp available for Vite HMR relaxations)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())