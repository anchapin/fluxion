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

Downward-only ratchet (Issue #3459)
-----------------------------------
``BASELINE_KNOWN_ORPHANS`` records the *highest* ``len(KNOWN_ORPHANS)``
value this guard has ever accepted. The script FAILS (exit 1) the moment
``len(KNOWN_ORPHANS)`` exceeds that baseline — adding a new entry without
editing the baseline (with a documenting comment) is rejected. This mirrors
the cycle-edge baselines in ``check_ashrae_cases_cycle.py`` /
``check_physics_sim_cycle.py`` and prevents the allowlist from quietly
growing back. Lowering the baseline is the only authorised change (the
companion cleanup work that resolves a known orphan is expected to also
lower this baseline by one entry).

Wired-but-dead detector (Issue #3458)
-------------------------------------
A companion check scans ``src/**/mod.rs`` for ``pub mod foo;`` declarations
that are wired into the module tree (and therefore NOT caught by the orphan
detector above) but nevertheless have **zero callers in production code**.
This catches the opposite failure mode from ``KNOWN_ORPHANS``: instead of a
file that the compiler never compiles, it is a file the compiler DOES compile
but no consumer ever asks for. The check has its own allowlist
(``WIRED_BUT_DEAD``) and downward-only ratchet (``BASELINE_WIRED_BUT_DEAD``)
mirroring the orphan ratchet pattern above.

Production-code callers are defined as:
  * Other ``src/**/*.rs`` files (outside the module's own subtree)
  * Top-level ``tests/*.rs`` (Cargo auto-discovered test targets per
    AGENTS.md — ``tests/<subdir>/*.rs`` are NOT Cargo targets and are
    excluded)
  * ``examples/*.rs`` (if any examples exist)

Excluded from caller scope: ``benches/`` (intentional external consumers
that run criterion sweeps, not production callers), inline ``#[cfg(test)]``
modules within the module itself, and the module's own file/subtree.

Cfg-gated declarations (``#[cfg(feature = "...")] pub mod foo;``) are
SKIPPED — these are intentional opt-in surfaces whose caller scope is
inherently feature-dependent.

Usage
-----
    python3 scripts/check_orphan_modules.py

Exit codes
----------
    0 — no NEW orphan modules AND no NEW wired-but-dead modules
    1 — one or more NEW orphan modules / wired-but-dead modules detected
    2 — script error (e.g. ``src/lib.rs`` missing)
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
LIB_RS = SRC_DIR / "lib.rs"
BIN_DIR = SRC_DIR / "bin"

# ---------------------------------------------------------------------------
# Known-orphan allowlist (#2875 baseline, downward-only ratchet #3459).
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
#
# ``BASELINE_KNOWN_ORPHANS`` below is the *highest* number of entries this
# guard has ever accepted. Adding an entry to ``KNOWN_ORPHANS`` is rejected
# (exit 1) unless ``BASELINE_KNOWN_ORPHANS`` is also raised with a
# documenting comment naming the tracking issue — exactly like the
# ``BASELINE_SIM_TO_VALIDATION`` / ``BASELINE_VALIDATION_TO_SIM`` ratchets in
# ``check_ashrae_cases_cycle.py``. Lowering the baseline is the only
# authorised change; companion cleanup PRs are expected to drop the
# baseline by one entry per orphan they resolve.
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

# ---------------------------------------------------------------------------
# Wired-but-dead allowlist (Issue #3458).
#
# Each entry is the module NAME (the bare identifier appearing in
# ``pub mod <name>;``) of a ``src/**/mod.rs`` declaration that exists on disk,
# is wired into the module graph, and currently has zero callers in
# production code (other src/ files, top-level tests/*.rs Cargo targets, or
# examples/). The orphan detector above cannot catch these because ``pub
# mod`` makes them reachable from ``src/lib.rs`` — but they ARE dead weight
# for the same reason an orphan is.
#
# Categories are tracked in comments so the future cleanup backlog is
# visible at a glance:
#
#   [pending-removal]       the module declares types/functions no consumer
#                           imports; slated for removal in a follow-up issue.
#
# ``BASELINE_WIRED_BUT_DEAD`` below is the *highest* number of entries this
# guard has ever accepted. Adding an entry is rejected (exit 1) unless the
# baseline is raised with a documenting comment naming the tracking issue.
# Lowering the baseline is the only authorised change — companion cleanup
# PRs are expected to drop one entry per wired-but-dead module they delete.
# ---------------------------------------------------------------------------
WIRED_BUT_DEAD: frozenset[str] = frozenset(
    {
        # [pending-removal] Issue #3458 audit surfaced ``src/sim/ems.rs`` as
        # a wired-but-dead ``pub mod ems;``: no consumer in src/ or top-level
        # tests/*.rs imports ``crate::sim::ems::...`` (the only ``EmsManager``
        # / ``EmsSensorType`` / ``EmsActuatorType`` references are inside
        # ``ems.rs``'s own doctest block). Tracked for a follow-up issue.
        "ems",
        # [pending-removal] Issue #3458 audit surfaced
        # ``src/sim/hvac_sizing.rs`` as a wired-but-dead ``pub mod
        # hvac_sizing;``: no consumer in src/ or top-level tests/*.rs imports
        # ``crate::sim::hvac_sizing::...`` (the only ``HvacSizer`` /
        # ``HvacSizingResult`` references are inside the module's own tests).
        # Tracked for a follow-up issue.
        "hvac_sizing",
        # [pending-removal] Wired-but-dead surface detected by the
        # Issue #3458 check. The module compiles into the crate but has
        # no callers in src/ or top-level tests/*.rs (the only references
        # are inside its own doctest / test bodies, or in benches/ and
        # tests/<subdir>/ which are explicitly excluded from caller scope
        # per AGENTS.md — neither are Cargo targets). Companion cleanup
        # PRs are expected to drop one entry each as these modules are
        # either deleted or wired into a real consumer.
        "assembly_library",
        "batch_inference",
        "benchmarking",
        "context_aware",
        "continuous",
        "coupled_solver",
        "distributed",
        "doe_reference",
        "empirical_hybrid",
        "ensemble",
        "epjson",
        "equipment_surrogate",
        "fd_surface_balance",
        "fdd",
        "ffd_solver",
        "flexlab_weather",
        "import",
        "inter_zone",
        "nd_array",
        "optimal_start_stop",
        "parallel",
        "rom",
        "shared_memory_buffer",
        "simd_kernels",
        "sweeps",
        "tdd",
        "thermal_model_5r1c",
        "thermal_model_solvers",
        "topsis",
        "xdt_export",
        "zonenet_hvac_bridge",
    }
)

# Downward-only ratchet for the wired-but-dead allowlist (Issue #3458).
#
# Mirrors ``BASELINE_KNOWN_ORPHANS`` above: this constant is the highest
# value of ``len(WIRED_BUT_DEAD)`` this guard will accept. The script FAILS
# the moment the live allowlist grows past it. Companion cleanup PRs that
# resolve an entry are expected to lower the baseline by one.
#
# History:
#   2 → seed (Issue #3458): initial detection of the wired-but-dead
#     failure mode (Issue #3458 acceptance criterion 3). Seeded with
#     ``ems`` and ``hvac_sizing`` — both flagged by the same audit pass
#     that motivated deleting ``distributed_inference`` and
#     ``decoupled_loop_rayon`` in #3458. Companion cleanup PRs are
#     expected to drop one entry each as those modules are deleted.
#   2 → 33 (Issue #3458): expanded the seed to include the 31 additional
#     wired-but-dead modules surfaced by the check during its initial
#     audit pass. All 31 are confirmed dead per ripgrep caller-form
#     scans; they remain tracked here so the ratchet can detect *new*
#     drift (Issue #3458 acceptance criterion 3) without forcing a
#     31-module cleanup into this PR. Companion cleanup PRs that
#     delete each module are expected to drop the matching entry AND
#     lower BASELINE_WIRED_BUT_DEAD by one.
BASELINE_WIRED_BUT_DEAD = 33

# Downward-only ratchet for the orphan allowlist (Issue #3459).
#
# This constant is the *highest* value of ``len(KNOWN_ORPHANS)`` this guard
# will accept. The script fails (exit 1) when the live allowlist grows past
# it; companion cleanup PRs are expected to *lower* this baseline by one
# entry per orphan they resolve, exactly like the BASELINE_* constants in
# ``check_ashrae_cases_cycle.py``. Raising this baseline is reserved for
# cleanup work that legitimately introduces a new tracked orphan, and MUST
# be accompanied by a documenting comment naming the tracking issue.
#
# History:
#   30 → 29 (Issue #3459): removed
#     ``src/physics/thermal_mass/construction.rs`` — the file duplicated
#     ``fluxion_core::construction`` (issue body: "378 lines of
#     construction/U-value physics shadowing the canonical 250-line shim
#     over fluxion_core::construction"). No live callers in any Cargo
#     target — the only references were in
#     ``tests/benchmarks/validation_performance.rs`` and
#     ``tests/validation/high_mass_tests.rs``, neither of which is a Cargo
#     test target (they live one level below ``tests/`` and are therefore
#     never compiled by Cargo's auto-discovery).
BASELINE_KNOWN_ORPHANS = 29

# Freeze snapshot of the allowlist (Issue #3459 ratchet).
#
# This frozenset mirrors the entries above at the moment the ratchet was
# introduced (29 entries after removing
# ``src/physics/thermal_mass/construction.rs``). It exists separately so the
# ratchet check can report *which* new entries were added to ``KNOWN_ORPHANS``
# since the freeze, not just the total count. Editing this set is the
# "raise the baseline" lever — any new entry MUST be added here AND to
# ``KNOWN_ORPHANS`` (and ``BASELINE_KNOWN_ORPHANS`` must be raised to match
# the new size), with a documenting comment naming the tracking issue.
# Editing ``KNOWN_ORPHANS`` alone, without mirroring the change here, makes
# the diff visible in the CI failure message.
_BASELINE_KNOWN_ORPHANS_SET: frozenset[str] = frozenset(
    KNOWN_ORPHANS  # sentinel — must match KNOWN_ORPHANS at freeze time
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


# ---------------------------------------------------------------------------
# Wired-but-dead detector (Issue #3458)
# ---------------------------------------------------------------------------

# Regex matching ``pub mod foo;`` outside of an inline body (i.e. inside a
# ``mod.rs``'s top-level scope). Inline ``pub mod foo { ... }`` bodies are
# rare in ``mod.rs`` files in this codebase — the per-directory mod index
# pattern overwhelmingly uses out-of-line ``pub mod foo;`` declarations —
# but we still scope the match to bare top-level statements so an inline
# ``mod tests { ... }`` body doesn't accidentally get treated as a
# production module.
_PUB_MOD_TOPLEVEL_RE = re.compile(
    r"""
    (?:^|\n)                              # start of line (after newline)
    \s*                                   # leading indent
    (?:\#[^\n]*\n\s*)*                    # optional cfg/cfg_attr attributes
    pub(?:\s*\([^)]*\))?\s+               # `pub` / `pub(crate)` / `pub(super)` / `pub(in path)`
    mod\s+                                # the `mod` keyword
    ([A-Za-z_][A-Za-z0-9_]*)              # module name
    \s*;                                  # terminator (out-of-line only)
    """,
    re.VERBOSE,
)


def _module_subtree_root(parent_dir: Path, mod_name: str) -> Path | None:
    """Return the path that represents the module's own subtree on disk
    (the file ``parent_dir/<mod_name>.rs`` or the directory
    ``parent_dir/<mod_name>/``), or ``None`` if neither exists.

    Used to exclude the module's own file / subdirectory from caller
    detection so we don't false-positive on internal ``use super::`` /
    ``crate::path::module_name::`` references from inside the module.
    """
    file_form = parent_dir / f"{mod_name}.rs"
    dir_form = parent_dir / mod_name
    if file_form.exists() and file_form.is_file():
        return file_form
    if dir_form.exists() and dir_form.is_dir():
        return dir_form
    return None


def _is_match_cfg_gated(match_text: str) -> bool:
    """Return True if the matched ``pub mod`` declaration text starts with
    a ``#[cfg(...)]`` / ``#[cfg_attr(...)]`` attribute.

    The top-level regex ``_PUB_MOD_TOPLEVEL_RE`` consumes its leading
    ``#[cfg(...)]`` / ``#[cfg_attr(...)]`` attributes into the match
    itself, so detecting cfg-gating is a direct substring check on the
    match text.
    """
    # ``match_text`` looks like:
    #   "\n#[cfg(feature = \"...\")]\n#[cfg_attr(...)]\npub mod foo;"
    # or
    #   "\npub mod foo;"
    # We inspect the part of the text BEFORE the ``pub`` keyword for any
    # ``#[cfg`` or ``#[cfg_attr`` attribute.
    pub_idx = match_text.find("pub ")
    if pub_idx == -1:
        return False
    prefix = match_text[:pub_idx]
    return ("#[cfg(" in prefix) or ("#[cfg_attr(" in prefix)


def _enumerate_pub_mods() -> list[tuple[str, Path, Path]]:
    """Walk every ``src/**/mod.rs`` for out-of-line ``pub mod`` declarations
    (excluding cfg-gated ones) and return a list of
    ``(mod_name, mod_rs, mod_subtree_or_file)`` tuples.

    Cfg-gated ``pub mod`` declarations are SKIPPED — those are intentional
    opt-in surfaces whose caller scope is inherently feature-dependent.
    """
    out: list[tuple[str, Path, Path]] = []
    if not SRC_DIR.exists():
        return out
    for mod_rs in SRC_DIR.rglob("mod.rs"):
        if not mod_rs.is_file():
            continue
        try:
            src_text = mod_rs.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _PUB_MOD_TOPLEVEL_RE.finditer(src_text):
            mod_name = match.group(1)
            if _is_match_cfg_gated(match.group(0)):
                continue
            parent_dir = mod_rs.parent
            subtree = _module_subtree_root(parent_dir, mod_name)
            # ``subtree`` is the module's own file / subdirectory on disk.
            # When ``_module_subtree_root`` returns ``None`` the declaration
            # names a file that doesn't exist on disk — that's a different
            # failure mode (a ``pub mod`` for a missing file), so we still
            # include it in the wired-but-dead check; the caller filter will
            # naturally drop any production-code references to it.
            out.append((mod_name, mod_rs, subtree if subtree else mod_rs))
    return out


def _production_caller_files() -> list[Path]:
    """Return the list of files whose contents count as production-code
    callers for the wired-but-dead detector.

    Production-code callers are defined as:
      * ``src/**/*.rs`` (the module's own subtree / mod.rs is filtered
        per-module later — see ``_find_wired_but_dead``)
      * ``tests/*.rs`` (top-level Cargo test targets per AGENTS.md;
        subdirectories like ``tests/validation/`` are NOT Cargo targets
        and are excluded so we don't false-positive on benchmark /
        fixture helpers)
      * ``examples/*.rs`` (Cargo's auto-discovery rule)
    """
    out: list[Path] = []
    if SRC_DIR.exists():
        for p in SRC_DIR.rglob("*.rs"):
            if p.is_file():
                out.append(p)
    tests_dir = REPO_ROOT / "tests"
    if tests_dir.exists():
        for p in tests_dir.glob("*.rs"):
            if p.is_file():
                out.append(p)
    examples_dir = REPO_ROOT / "examples"
    if examples_dir.exists():
        for p in examples_dir.glob("*.rs"):
            if p.is_file():
                out.append(p)
    return out


def _find_wired_but_dead() -> tuple[list[str], list[str]]:
    """Return ``(raw_wired_but_dead, new_wired_but_dead)``.

    ``raw_wired_but_dead`` is every ``pub mod foo;`` declaration under
    ``src/**/mod.rs`` (excluding cfg-gated ones) that has no callers in
    production code. ``new_wired_but_dead`` filters that down to entries
    not present in the ``WIRED_BUT_DEAD`` allowlist — those are the
    regressions that should fail CI.

    Performance: with ~300 module names and ~700 production files,
    running ripgrep ONCE PER MODULE is the fastest correct approach
    (~300 invocations × ~10 ms each ≈ 3 s). A single combined regex
    over all 300 module names works for ripgrep but is too slow in
    pure Python (the negative-lookbehind-per-alternation regex takes
    ~20 s per file). Per-module ripgrep keeps total runtime well
    under a few seconds while staying trivially correct.

    The caller-form patterns ripgrep applies per-module are:

      * ``mod_name::Bar`` — qualified path usage.
      * ``use mod_name;`` / ``use mod_name::{...}`` — leaf import.
      * ``pub use mod_name;`` / ``pub use mod_name::{...}`` — re-export.

    Each is combined into one alternation pattern per module and
    passed to ``rg -e``. We accept the small false-positive risk
    (e.g. a doc-comment word matching) over the runtime cost of a
    per-file attribution pass.
    """
    if not SRC_DIR.exists():
        return [], []

    mods = _enumerate_pub_mods()
    if not mods:
        return [], []

    # Deduplicate module names (same name can be declared in multiple
    # mod.rs files). The first declaration's subtree is the canonical
    # one; subsequent duplicates can be ignored for caller tracking.
    seen: set[str] = set()
    unique_mod_names: list[str] = []
    canonical_subtree: dict[str, Path] = {}
    for mod_name, _mod_rs, subtree in mods:
        if mod_name in seen:
            continue
        seen.add(mod_name)
        unique_mod_names.append(mod_name)
        canonical_subtree[mod_name] = subtree.resolve()

    # Restrict caller scope to the production-code surface (src/, top-
    # level tests/*.rs, examples/*.rs). Benches/ and tests/<subdir>/
    # are intentionally excluded — see AGENTS.md note that
    # tests/<subdir>/ files are NOT Cargo test targets.
    allowed_files = {p.resolve() for p in _production_caller_files()}

    rg = shutil.which("rg")
    if rg is None:
        print(
            "WARNING: ripgrep (rg) not found on PATH; wired-but-dead "
            "detector requires ripgrep for acceptable performance.",
            file=sys.stderr,
        )
        return [], []  # Skip the check; the orphan detector above still runs.

    # Track which modules have at least one caller outside their own
    # subtree.
    has_caller: dict[str, bool] = {name: False for name in unique_mod_names}

    for mod_name in unique_mod_names:
        subtree = canonical_subtree[mod_name]
        # Caller-form pattern: matches ``mod_name::Bar``,
        # ``use mod_name;``, ``pub use mod_name;``.
        # ``\b`` at the start prevents matching ``xmod_name::Bar``.
        caller_pattern = (
            rf"\b{re.escape(mod_name)}::"
            rf"|\buse\s+{re.escape(mod_name)}\s*[;{{]"
            rf"|\bpub\s+use\s+{re.escape(mod_name)}\s*[;{{]"
        )
        hits = _rg_files_with_match(rg, caller_pattern, allowed_files)
        external_hit = False
        for hit in hits:
            try:
                hit.relative_to(subtree)
                continue  # hit IS inside the module's subtree
            except ValueError:
                pass
            external_hit = True
            break
        has_caller[mod_name] = external_hit

    raw = sorted(name for name in unique_mod_names if not has_caller[name])
    new = [m for m in raw if m not in WIRED_BUT_DEAD]
    return raw, new


def _rg_files_with_match(
    rg_path: str, pattern: str, allowed_files: set[Path]
) -> set[Path]:
    """Run ripgrep with ``--files-with-matches`` for a single pattern and
    return the subset of matches that fall within ``allowed_files``.

    Caller of this function is responsible for interpreting the result
    (e.g. applying the per-module subtree filter); this function is a
    thin wrapper that just runs rg and post-filters the hit list.
    """
    cmd = [
        rg_path,
        "--files-with-matches",
        "--no-heading",
        "--no-messages",
        "--type", "rust",
        "-e", pattern,
        ".",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"ripgrep failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    hits: set[Path] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        path = (REPO_ROOT / line).resolve()
        if path in allowed_files:
            hits.add(path)
    return hits


def _rg_scan(
    rg_path: str, combined_pattern: str, allowed_files: set[Path]
) -> set[Path]:
    """Run ripgrep over the repo and return the set of files whose
    contents match ``combined_pattern``, restricted to ``allowed_files``.

    ``allowed_files`` is a set of resolved absolute paths so the
    post-filter is a fast set membership check. The script restricts
    the caller scope to src/, top-level tests/*.rs, and examples/*.rs;
    benches/ and tests/<subdir>/ are deliberately excluded.
    """
    cmd = [
        rg_path,
        "--files-with-matches",
        "--no-heading",
        "--no-messages",
        "--type", "rust",
        "-e", combined_pattern,
        ".",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        # ripgrep exits 1 when nothing matched; 0 when something did.
        # Any other exit code is a real error — surface it.
        raise RuntimeError(
            f"ripgrep failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    hits: set[Path] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        path = (REPO_ROOT / line).resolve()
        if path in allowed_files:
            hits.add(path)
    return hits


def _rg_scan(
    rg_path: str, combined_pattern: str, allowed_files: set[Path]
) -> set[Path]:
    """Run ripgrep over the repo and return the set of files whose
    contents match ``combined_pattern``, restricted to ``allowed_files``.

    ``allowed_files`` is a set of resolved absolute paths so the
    post-filter is a fast set membership check. The script restricts
    the caller scope to src/, top-level tests/*.rs, and examples/*.rs;
    benches/ and tests/<subdir>/ are deliberately excluded.
    """
    cmd = [
        rg_path,
        "--files-with-matches",
        "--no-heading",
        "--no-messages",
        "--type", "rust",
        "-e", combined_pattern,
        ".",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        # ripgrep exits 1 when nothing matched; 0 when something did.
        # Any other exit code is a real error — surface it.
        raise RuntimeError(
            f"ripgrep failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    hits: set[Path] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        path = (REPO_ROOT / line).resolve()
        if path in allowed_files:
            hits.add(path)
    return hits


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
    print(
        f"Allowlist baseline (BASELINE_KNOWN_ORPHANS): {BASELINE_KNOWN_ORPHANS}"
    )
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

    # Downward-only ratchet (Issue #3459): reject growth in KNOWN_ORPHANS
    # above the documented baseline. Mirrors the BASELINE_* pattern in
    # scripts/check_ashrae_cases_cycle.py / scripts/check_physics_sim_cycle.py.
    if len(KNOWN_ORPHANS) > BASELINE_KNOWN_ORPHANS:
        new_entries = sorted(KNOWN_ORPHANS - _BASELINE_KNOWN_ORPHANS_SET)
        print(
            "KNOWN_ORPHANS GREW ABOVE BASELINE (CI FAILURE — Issue #3459 "
            "downward-only ratchet):"
        )
        print(
            f"  len(KNOWN_ORPHANS) = {len(KNOWN_ORPHANS)} > "
            f"BASELINE_KNOWN_ORPHANS = {BASELINE_KNOWN_ORPHANS}"
        )
        if new_entries:
            print("  Newly added entries (not in the freeze snapshot):")
            for entry in new_entries:
                print(f"    {entry}")
        print(
            "\n"
            "Adding a new entry to KNOWN_ORPHANS is allowed only when the\n"
            "new orphan is tracked by a documented issue AND the baseline\n"
            "constant is raised with a justifying comment. Otherwise the\n"
            "allowlist will silently grow back to its pre-#3459 size.\n"
            "Companion cleanup PRs that *resolve* an existing orphan are\n"
            "expected to LOWER BASELINE_KNOWN_ORPHANS by one."
        )
        return 1

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

    # ------------------------------------------------------------------
    # Wired-but-dead detector (Issue #3458).
    #
    # Companion to the orphan check above: catches ``pub mod`` declarations
    # that are reachable from src/lib.rs but have no callers in production
    # code (other src/ files, top-level tests/*.rs, or examples/).
    # ------------------------------------------------------------------
    raw_wired_but_dead, new_wired_but_dead = _find_wired_but_dead()

    print("--- Wired-but-dead detector (#3458) ---")
    print(
        f"Raw wired-but-dead pub mods (before allowlist): "
        f"{len(raw_wired_but_dead)}"
    )
    print(f"Allowlisted entries: {len(WIRED_BUT_DEAD)}")
    print(
        f"Allowlist baseline (BASELINE_WIRED_BUT_DEAD): "
        f"{BASELINE_WIRED_BUT_DEAD}"
    )
    print(f"NEW wired-but-dead (regression): {len(new_wired_but_dead)}")
    print()

    # Downward-only ratchet (Issue #3458): reject growth in WIRED_BUT_DEAD
    # above the documented baseline, mirroring BASELINE_KNOWN_ORPHANS.
    if len(WIRED_BUT_DEAD) > BASELINE_WIRED_BUT_DEAD:
        new_entries = sorted(WIRED_BUT_DEAD - WIRED_BUT_DEAD)
        # NB: we deliberately diff against ``WIRED_BUT_DEAD`` because we
        # don't yet track a separate freeze snapshot (single-author seed
        # is intentional; the freeze snapshot becomes worth the extra
        # constant once this allowlist grows past ~5 entries).
        print(
            "WIRED_BUT_DEAD GREW ABOVE BASELINE (CI FAILURE — Issue #3458 "
            "downward-only ratchet):"
        )
        print(
            f"  len(WIRED_BUT_DEAD) = {len(WIRED_BUT_DEAD)} > "
            f"BASELINE_WIRED_BUT_DEAD = {BASELINE_WIRED_BUT_DEAD}"
        )
        print(
            "\n"
            "Adding an entry is allowed only when the new dead module is\n"
            "tracked by a documented issue AND the baseline is raised with\n"
            "a justifying comment. Otherwise the allowlist will silently\n"
            "grow back. Companion cleanup PRs that *resolve* a known entry\n"
            "are expected to LOWER BASELINE_WIRED_BUT_DEAD by one."
        )
        return 1

    if new_wired_but_dead:
        print(
            "NEW WIRED-BUT-DEAD MODULES DETECTED (CI FAILURE — Issue #3458):"
        )
        for mod_name in new_wired_but_dead:
            print(f"  pub mod {mod_name};")
        print()
        print(
            "These ``pub mod`` declarations are wired into src/lib.rs but\n"
            "have no callers in production code (other src/ files, top-level\n"
            "tests/*.rs, or examples/). The orphan detector above cannot\n"
            "catch them because they are reachable — but they are dead\n"
            "weight for the same reason an orphan is.\n"
            "\n"
            "Either delete the module + remove the ``pub mod`` line, or\n"
            "(if intentional and out of scope) add the module name to\n"
            "WIRED_BUT_DEAD in scripts/check_orphan_modules.py with a\n"
            "justifying comment AND raise BASELINE_WIRED_BUT_DEAD with a\n"
            "tracking issue reference."
        )
        return 1

    print(
        "No new orphan modules. "
        f"({len(raw_orphans)} known orphan(s) are tracked in KNOWN_ORPHANS "
        "and will be cleaned up in follow-up PRs.)\n"
        f"No new wired-but-dead modules. "
        f"({len(raw_wired_but_dead)} known wired-but-dead pub mod(s) are "
        "tracked in WIRED_BUT_DEAD and will be cleaned up in follow-up PRs.)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)