#!/usr/bin/env python3
"""
Orphan-modules detector — Issue #2875.

Fails CI when any ``*.rs`` file under ``src/**`` (the main ``fluxion`` crate)
is **not** transitively reachable from ``src/lib.rs``. An "orphan module" is a
source file that exists on disk but is never included by the module graph the
compiler actually builds, e.g. a leftover after a rename / move / duplicate
(see issue #2875 — ``src/sim/components.rs`` was a duplicate of
``src/sim/construction.rs``'s ``WallSurface`` that nothing referenced).

Why this matters
----------------
Rust's ``#[cfg(...)]`` and ``mod foo;`` declarations form a tree rooted at
``lib.rs``. Files outside that tree are silently skipped by ``cargo build`` —
they are dead weight that drifts from reality and occasionally becomes a
duplicate-type trap (the precise failure mode #2875 documented). The
classifier detects them statically so the dead weight is visible before the
slower ``cargo check --workspace`` even runs.

How reachability is computed
----------------------------
1. Start at ``src/lib.rs`` (the crate root).
2. BFS over ``mod foo;`` / ``mod foo { ... }`` declarations whose *target*
   resolves to a file inside ``src/**/*.rs``. A ``mod`` is reachable if its
   name matches ``src/<dir>/<name>.rs`` OR ``src/<dir>/<name>/mod.rs``.
3. Inline ``mod foo { ... }`` bodies are scanned recursively for nested
   ``mod bar;`` declarations so feature-gated / test-only sub-modules are
   still recognised as wired into the tree.
4. ``cfg(feature = ...)`` / ``cfg_attr(...)`` attributes on the ``mod`` line
   are stripped before matching — we are checking that the module *can* be
   wired in, not that every feature is currently active.
5. Files under ``src/bin/`` are excluded from the universe: each ``.rs``
   there is its own crate root / target (Cargo auto-discovers binaries), not
   a child of the ``fluxion`` library module tree.

Allowlist
---------
``KNOWN_ORPHANS`` lists files that are *known* to be orphans as of the
baseline commit that introduced this guard. They do NOT cause the script to
fail today (so the script can pass in CI), but each entry is a future
cleanup target — the script fails the moment a new file becomes an orphan
that is not in the list.

The allowlist serves as a tracked cleanup backlog. Removing an entry from
``KNOWN_ORPHANS`` requires the corresponding orphan to have been deleted
(or wired into the module graph) in the same PR.

Usage
-----
    python3 scripts/check_orphan_modules.py

Exit codes
----------
    0 — no NEW orphan modules (allowlisted entries are not reported)
    1 — one or more NEW orphan modules detected
    2 — script error (e.g. ``src/lib.rs`` missing)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
LIB_RS = SRC_DIR / "lib.rs"
BIN_DIR = SRC_DIR / "bin"

# ---------------------------------------------------------------------------
# Known-orphan allowlist (#2875 baseline).
#
# Each entry is a path relative to ``REPO_ROOT`` for a ``.rs`` file that is
# known to be orphaned as of this commit. The script ignores these so the
# detector can pass today; the moment a new file becomes an orphan the
# script fails CI until either the file is wired in or its path is added
# here with justification.
#
# Categories are tracked in the comments so the future cleanup backlog is
# visible at a glance:
#
#   [deleted-in-crate-split]   the leaf moved into ``fluxion-core``; the
#                              in-crate clone is now dead.
#   [replaced-by-canonical]    a sibling file owns the same name (e.g.
#                              ``parallel_executor.rs`` vs ``parallel.rs``).
#   [pending-removal]          the module has no callers in-crate; slated
#                              for removal in a follow-up issue.
#   [feature-gated-no-decl]    declared under a cfg gate that no longer
#                              wires the file in (deferred cleanup).
# ---------------------------------------------------------------------------
KNOWN_ORPHANS: frozenset[str] = frozenset(
    {
        # [replaced-by-canonical] `crate::ai::rl_policy` is exposed elsewhere;
        # this file is no longer wired into src/ai/mod.rs.
        "src/ai/rl_policy.rs",
        # [pending-removal] src/cli/commands was the entry-point of the old
        # CLI surface (#2929 removed `mod commands;` from src/cli/mod.rs);
        # the directory is kept around pending a follow-up delete.
        "src/cli/commands/mod.rs",
        "src/cli/commands/cross_validation.rs",
        "src/cli/commands/import.rs",
        # [pending-removal] thermal_mass::construction duplicates construction
        # logic that lives in src/sim/construction; pending consolidation.
        "src/physics/thermal_mass/construction.rs",
        # [pending-removal] src/sim/hvac/tests/*.rs are reachable from the
        # ``mod tests { ... }`` inline body in src/sim/hvac/mod.rs:481, but
        # the inline body does not declare them as nested mods. Pending
        # either consolidation into the inline body or a dedicated module.
        "src/sim/hvac/tests/cycling_tests.rs",
        "src/sim/hvac/tests/efficiency_curve_tests.rs",
        "src/sim/hvac/tests/equipment_tests.rs",
        "src/sim/hvac/tests/fluid_adapter_tests.rs",
        # [pending-removal] no `mod solar_gain_distribution;` in src/sim/mod.rs.
        "src/sim/solar_gain_distribution.rs",
        # [pending-removal] src/thermal/mod.rs does not declare solver /
        # zone_coupling; the canonical home is src/thermal/* elsewhere.
        "src/thermal/solver.rs",
        "src/thermal/zone_coupling.rs",
        # [pending-removal] live_twin_broadcaster has no callers in-crate.
        "src/twin/live_twin_broadcaster.rs",
        # [pending-removal] src/validation/esp_r/* is fully orphaned; callers
        # route through src/validation/reports/* instead. Tracked separately.
        "src/validation/esp_r/cli_integration.rs",
        "src/validation/esp_r/comparison.rs",
        "src/validation/esp_r/examples.rs",
        "src/validation/esp_r/integration.rs",
        "src/validation/esp_r/mod.rs",
        "src/validation/esp_r/parser.rs",
        "src/validation/esp_r/test_automation.rs",
        "src/validation/esp_r/test_automation_test.rs",
        # [pending-removal] ML data collector never wired into
        # src/validation/mod.rs.
        "src/validation/ml_data_collector.rs",
        # [replaced-by-canonical] parallel_executor.rs owns the name;
        # executor.rs / parallel.rs are leftover siblings.
        "src/validation/performance/executor.rs",
        "src/validation/performance/parallel.rs",
        # [pending-removal] src/validation/reports/* has no callers in-crate.
        "src/validation/reports/cross_validation.rs",
        "src/validation/reports/mod.rs",
        # [pending-removal] validation_suite.rs is never wired into
        # src/validation/mod.rs.
        "src/validation/validation_suite.rs",
        # [deleted-in-crate-split] weather moved to fluxion-core (#1255);
        # the in-crate clones are dead weight.
        "src/weather/denver.rs",
        "src/weather/epw.rs",
        "src/weather/mod.rs",
    }
)

# Match `mod foo;` / `pub mod foo;` / `pub(crate) mod foo;` / `mod foo {`.
# We capture the *name* and skip a `;` or `{` terminator. ``cfg(...)`` and
# ``cfg_attr(...)`` attributes are stripped before matching (Rust lets you
# write ``#[cfg(feature = "x")] mod foo;``). The leading ``mod`` keyword is
# required so we don't pick up e.g. ``module_name`` identifiers or comments.
#
# Capture group 1 = module name. We deliberately accept ``pub``,
# ``pub(crate)``, ``pub(super)``, ``pub(in path)`` visibilities — the reach
# check is structural, not API-surface.
_MOD_RE = re.compile(
    r"""
    (?:^|\s)                              # boundary (start of line or whitespace)
    (?:\#[^\n]*\n\s*)*                    # optional cfg/cfg_attr attributes (one or more lines)
    (?:pub(?:\s*\([^)]*\))?\s+)?          # optional `pub` / `pub(crate)` / `pub(super)` / `pub(in path)`
    mod\s+                                # the `mod` keyword
    ([A-Za-z_][A-Za-z0-9_]*)              # module name
    \s*[;{]                               # terminator: `;` for out-of-line or `{` for inline
    """,
    re.VERBOSE | re.MULTILINE,
)


def _strip_block_comments(text: str) -> str:
    """Remove ``/* ... */`` block comments while preserving line numbers."""
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _strip_line_comments(text: str) -> str:
    """Remove ``//`` line comments. We don't try to honour `// /*` inside a
    string; in practice the module declarations we care about never appear
    inside string literals, and stripping aggressively matches the technique
    used by ``check_ashrae_cases_cycle.py`` / ``check_physics_sim_cycle.py``.
    """
    out_lines: list[str] = []
    for line in text.splitlines():
        # Preserve the line itself so file:line offsets in error messages are
        # meaningful; just blank out the comment suffix.
        stripped = re.sub(r"//.*$", "", line)
        out_lines.append(stripped)
    return "\n".join(out_lines)


def _clean_source(text: str) -> str:
    """Strip block + line comments for the mod scan."""
    return _strip_line_comments(_strip_block_comments(text))


def _candidate_paths_for_mod(mod_name: str, parent_dir: Path) -> list[Path]:
    """Return the candidate file paths a Rust ``mod foo;`` declaration could
    resolve to inside ``parent_dir``. We accept all three forms the compiler
    accepts (2015 + 2018 + inline) so a module declared in any style gets
    recognised.

    Order matters only for error messages: prefer the modern 2018 ``name.rs``
    form, then the 2015 ``name/mod.rs`` form.
    """
    return [
        parent_dir / f"{mod_name}.rs",
        parent_dir / mod_name / "mod.rs",
    ]


def _extract_mod_bodies(text: str) -> dict[str, str]:
    """Return a mapping from inline ``mod name { ... }`` name to its body
    text. We do this by walking the source string and tracking brace depth,
    because a regex over the body contents would over-match nested braces.
    """
    bodies: dict[str, str] = {}
    # Pre-find every `mod NAME {` header position so we know where to start.
    # The header regex strips cfg/cfg_attr attributes the same way _MOD_RE
    # does; the difference is we require the terminator to be `{` here.
    header_re = re.compile(
        r"(?:^|\s)"
        r"(?:\#[^\n]*\n\s*)*"
        r"(?:pub(?:\s*\([^)]*\))?\s+)?"
        r"mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{",
        re.MULTILINE,
    )
    for match in header_re.finditer(text):
        name = match.group(1)
        open_brace_index = match.end() - 1
        depth = 1
        i = open_brace_index + 1
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        bodies[name] = text[open_brace_index + 1 : i - 1]
    return bodies


def _collect_declared_mods(rs_file: Path) -> tuple[list[str], dict[str, str]]:
    """Return the list of ``mod`` names declared (transitively) in ``rs_file``
    that resolve to a child file in the same directory, plus a mapping of
    inline ``mod foo { ... }`` bodies for further recursion.

    Out-of-line ``mod foo;`` declarations reference a child file. Inline
    ``mod foo { ... }`` declarations do not directly reference a child
    file, but their body may contain *nested* out-of-line ``mod bar;``
    declarations that do. We surface both kinds so the caller can walk the
    full module tree.
    """
    raw = rs_file.read_text(encoding="utf-8", errors="replace")
    text = _clean_source(raw)
    declared: list[str] = []
    inline_bodies = _extract_mod_bodies(text)
    # Strip the inline bodies from `text` before scanning for `;` out-of-line
    # declarations so a `mod foo;` inside an inline body doesn't get counted
    # twice (the recursive walk will pick it up when we descend into the
    # inline body below).
    text_without_inline = text
    for body in inline_bodies.values():
        text_without_inline = text_without_inline.replace(body, "")
    for match in _MOD_RE.finditer(text_without_inline):
        name = match.group(1)
        end = match.end()
        terminator_index = end - 1
        original_char = text[terminator_index]
        if original_char == ";":
            declared.append(name)
    return declared, inline_bodies


def _walk_reachable(start: Path) -> set[Path]:
    """BFS over the module graph rooted at ``start`` (typically ``src/lib.rs``).

    Returns the set of ``*.rs`` files transitively reachable via ``mod foo;``
    declarations whose target is a real file under ``src/**``, including
    descent into inline ``mod foo { ... }`` bodies.
    """
    if not start.exists():
        raise FileNotFoundError(f"crate root not found: {start}")

    # (file_path, source_text) pairs queued for processing. We pass the
    # source text alongside the path so we can descend into inline mod
    # bodies without re-reading the file (and so the inline-body map is
    # available to the body walker below).
    queue: list[Path] = [start]
    visited: set[Path] = set()
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        raw = current.read_text(encoding="utf-8", errors="replace")
        cleaned = _clean_source(raw)
        parent_dir = current.parent
        out_of_line, inline_bodies = _collect_declared_mods(current)
        # Out-of-line mod declarations.
        for name in out_of_line:
            for candidate in _candidate_paths_for_mod(name, parent_dir):
                if candidate.exists() and candidate.is_file():
                    queue.append(candidate)
                    break
        # Inline mod bodies: any out-of-line `mod bar;` declarations inside
        # them point at child files of the *inline namespace's* directory,
        # which is `parent_dir/<inline_name>/`. We re-scan the inline body
        # with the same regex + resolver.
        for inline_name, body_text in inline_bodies.items():
            inline_parent = parent_dir / inline_name
            body_cleaned = _clean_source(body_text)
            for match in _MOD_RE.finditer(body_cleaned):
                name = match.group(1)
                end = match.end()
                terminator_index = end - 1
                original_char = body_cleaned[terminator_index]
                if original_char != ";":
                    continue
                for candidate in _candidate_paths_for_mod(name, inline_parent):
                    if candidate.exists() and candidate.is_file():
                        queue.append(candidate)
                        break
    return visited


def _all_rs_under_src() -> set[Path]:
    """Every ``*.rs`` file under ``src/**`` (excluding ``src/bin/`` — each
    file there is its own Cargo target, not a library-module child).
    """
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"src dir not found: {SRC_DIR}")
    out: set[Path] = set()
    for p in SRC_DIR.rglob("*.rs"):
        if not p.is_file():
            continue
        # src/bin/*.rs are standalone binaries (Cargo target roots). They
        # are not children of the lib module graph and are not orphans in
        # the sense this guard cares about.
        try:
            p.relative_to(BIN_DIR)
        except ValueError:
            out.add(p)
    return out


def main() -> int:
    print(f"Orphan-modules detector (#2875) — repo: {REPO_ROOT}")
    print()

    if not LIB_RS.exists():
        print(f"ERROR: crate root not found: {LIB_RS}", file=sys.stderr)
        return 2

    all_rs = _all_rs_under_src()
    reachable = _walk_reachable(LIB_RS)
    raw_orphans = sorted(all_rs - reachable, key=lambda p: str(p))
    # Apply the allowlist: known-existing orphans do not fail CI.
    new_orphans = [
        orphan
        for orphan in raw_orphans
        if orphan.relative_to(REPO_ROOT).as_posix() not in KNOWN_ORPHANS
    ]
    # Sanity check: entries in the allowlist that point to files that are
    # actually reachable (i.e. someone wired them in) get surfaced as a
    # cleanup nudge so the allowlist does not rot.
    allowlist_resolved = sorted(
        path
        for rel in KNOWN_ORPHANS
        if (path := REPO_ROOT / rel) in reachable
    )

    print(f"Total .rs files under src/ (excluding src/bin/): {len(all_rs)}")
    print(f"Transitively reachable from src/lib.rs: {len(reachable)}")
    print(f"Raw orphans (before allowlist): {len(raw_orphans)}")
    print(f"Allowlisted entries: {len(KNOWN_ORPHANS)}")
    print(f"NEW orphans (regression): {len(new_orphans)}")
    print()

    if allowlist_resolved:
        print(
            "ALLOWLIST CLEANUP NUDGE: the following entries in KNOWN_ORPHANS "
            "are now reachable from src/lib.rs and can be removed from the "
            "allowlist in a follow-up PR:"
        )
        for path in allowlist_resolved:
            print(f"  {path.relative_to(REPO_ROOT)}")
        print()

    if new_orphans:
        print("NEW ORPHAN MODULES DETECTED (CI FAILURE):")
        for orphan in new_orphans:
            rel = orphan.relative_to(REPO_ROOT)
            print(f"  {rel}")
        print()
        print(
            "These files are present on disk but never included by any\n"
            "`mod foo;` declaration reachable from src/lib.rs. They are\n"
            "invisible to `cargo build` and silently drift from the rest of\n"
            "the codebase.\n"
            "\n"
            "Either wire them into the module graph (add `pub mod <name>;`\n"
            "to the parent `mod.rs`) or, if they are dead, delete the file.\n"
            "If the orphan is intentional and out of scope for immediate\n"
            "cleanup, add its path to KNOWN_ORPHANS in\n"
            "scripts/check_orphan_modules.py with a justification comment."
        )
        return 1

    print(
        "No new orphan modules. "
        f"({len(raw_orphans)} known orphan(s) are tracked in KNOWN_ORPHANS "
        "and will be cleaned up in follow-up PRs.)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)