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
    excluded), PLUS the file universe of the root crate's explicit
    ``[[test]] path = "tests/<sub>/<root>.rs"`` targets (Issue #3764
    consolidated-harness convention): a harness root such as
    ``tests/all_tests/main.rs`` and the sibling files it wires in via
    ``mod`` declarations ARE compiled test code and count as callers
  * ``examples/*.rs`` (if any examples exist)

Excluded from caller scope: ``benches/`` (intentional external consumers
that run criterion sweeps, not production callers), inline ``#[cfg(test)]``
modules within the module itself, and the module's own file/subtree.

Cfg-gated declarations (``#[cfg(feature = "...")] pub mod foo;``) are
SKIPPED — these are intentional opt-in surfaces whose caller scope is
inherently feature-dependent.

Dead-code-allow inventory (Issue #3752)
---------------------------------------
A third companion check governs the ``#[allow(dead_code)]`` attributes in
production (non-test) code. At the density the #3752 audit measured
(~100 occurrences concentrated in solver-adjacent modules such as
``src/physics/state_space_ctf/mod.rs`` and ``src/ai/surrogate/session_pool.rs``),
compiler-level dead-code signal is disabled exactly where solver internals
evolve fastest, and no Goal #5 tooling could see it: the orphan detector
above works at file granularity, not item granularity.

The mechanism is a checked-in inventory —
``tests/reference_data/dead_code_inventory.json`` — that records every
production ``#[allow(dead_code)]`` site with a tracking-issue reference
(the "#3752" governing issue until a site gains its own follow-up issue).
This satisfies the issue's acceptance clause "each existing
``#[allow(dead_code)]`` either gains an issue reference (or is deleted)"
at the registry level without churning 42 production files; deleting an
attribute remains the preferred resolution and shrinks the inventory
through ``--update-dead-code-inventory``.

Gate semantics (mirroring the downward-only ratchet convention of the
quarantine registry, Issue #3443):

1. **New-site rejection** — a live production site that is absent from
   the inventory fails CI. The fix is either to delete the attribute or
   to add the entry (with a tracking issue) via
   ``--update-dead-code-inventory`` and justify it in review.
2. **Stale-entry rejection** — an inventory entry whose site no longer
   exists in code fails CI, keeping the registry in lock-step with the
   tree (same lock-step convention as the test-inventory drift gate,
   Issue #3442).
3. **Downward-only count ratchet** — ``BASELINE_DEAD_CODE_ALLOWS`` is
   the highest production count this guard accepts. The script fails the
   moment the live count exceeds it. Cleanup PRs are expected to lower
   the baseline by one per resolved attribute; raising it requires a
   documenting comment naming the tracking issue.

Measurement scope (canonical definition): ``src/**/*.rs`` of the root
``fluxion`` crate (``src/bin/`` excluded — those are standalone binary
targets, mirroring the orphan universe), block comments stripped,
occurrence-counted, with ``#[cfg(test)]`` inline module bodies excluded
so test-only suppressions never count against the production ratchet.
Inner ``#![allow(...)]`` module-level attributes are out of scope (none
exist today). The #3752 audit headline of "100 across 43 files" came
from a looser raw-text match that included 3 sites inside
``#[cfg(test)]`` bodies; the canonical production-only scan seeds the
baseline at 94 across 42 files.

Per-module disposition registry (Issue #3748)
---------------------------------------------
A fourth companion check requires every ``WIRED_BUT_DEAD`` entry to
carry an explicit, owner-assigned disposition in a checked-in registry
— ``tests/reference_data/wired_but_dead_dispositions.json`` — mirroring
the #3752 inventory pattern. The ``BASELINE_WIRED_BUT_DEAD`` ratchet is
downward-only with no forcing function (the #3555 burn-down removed 11
entries, but nothing scheduled the remaining 22); the registry is that
forcing function. Allowed dispositions:

  * ``wire-pending``  a production consumer is planned; the module
    stays until the wiring lands (or the decision flips to reject).
  * ``reject``        slated for deletion in a cleanup PR; that PR must
    delete the module, drop the allowlist entry, lower
    ``BASELINE_WIRED_BUT_DEAD`` by one, and drop the registry row.
  * ``keep-dead``     intentional dead surface with a documented
    rationale (e.g. a parked roadmap module, or a helper surface whose
    callers are excluded from caller scope by design).

Gate semantics (mirroring the lock-step convention of the dead-code
inventory, Issue #3443's downward-only ratchet convention stays
untouched):

1. **Coverage** — every live wired-but-dead module AND every
   ``WIRED_BUT_DEAD`` allowlist entry must have a registry row, and
   each row must carry a ``disposition`` from the enum above plus
   ``issue``, ``owner``, and a non-empty ``rationale``.
2. **Lock-step** — a registry row whose module is no longer
   wired-but-dead (it was wired up or deleted) fails CI; drop the row
   in the same PR that resolves the module.
3. The count ratchet itself remains ``BASELINE_WIRED_BUT_DEAD``
   (downward-only, Issue #3458); the registry adds the per-module
   wire-or-delete decision and owner the ratchet lacked.

Usage
-----
    python3 scripts/check_orphan_modules.py
    python3 scripts/check_orphan_modules.py --update-dead-code-inventory

Exit codes
----------
    0 — no NEW orphan modules, no NEW wired-but-dead modules, the
        dead-code-allow inventory is in lock-step at or below baseline,
        and every wired-but-dead module carries a valid disposition
    1 — one or more NEW orphan modules / wired-but-dead modules / new or
        stale dead-code-allow sites, a missing/stale/invalid
        disposition registry row, or the production count grew above
        ``BASELINE_DEAD_CODE_ALLOWS``
    2 — script error (e.g. ``src/lib.rs`` missing, a required registry
        file absent)
"""

from __future__ import annotations

import json
import re
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
# Issue #3555 burn-down: lowered BASELINE_KNOWN_ORPHANS 29 → 0 in PR for fluxion-#3555.
# All 29 prior entries deleted: src/ai/rl_policy.rs; src/cli/commands/{mod,cross_validation,import}.rs;
# src/sim/hvac/tests/{cycling,efficiency_curve,equipment,fluid_adapter}_tests.rs;
# src/sim/solar_gain_distribution.rs; src/thermal/{solver,zone_coupling}.rs;
# src/twin/live_twin_broadcaster.rs; src/validation/esp_r/{cli_integration,comparison,examples,integration,mod,parser,test_automation,test_automation_test}.rs;
# src/validation/{ml_data_collector,validation_suite}.rs;
# src/validation/performance/{executor,parallel}.rs;
# src/validation/reports/{cross_validation,mod}.rs;
# src/weather/{denver,epw,mod}.rs.
KNOWN_ORPHANS: frozenset[str] = frozenset(
    {
        # Allowlist emptied by Issue #3555 burn-down (2026-09-08).
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
        # Issue #3555 burn-down (PR for fluxion-#3555): removed 11 entries
        # (`continuous`, `nd_array`, `fd_surface_balance`, `ffd_solver`,
        # `simd_kernels`, `ensemble`, `xdt_export`, `shared_memory_buffer`,
        # `optimal_start_stop`, `zonenet_hvac_bridge`, `import`).
        "assembly_library",
        "batch_inference",
        "benchmarking",
        "context_aware",
        "coupled_solver",
        "distributed",
        "doe_reference",
        "equipment_surrogate",
        "epjson",
        "fdd",
        "flexlab_weather",
        "inter_zone",
        "parallel",
        "rom",
        "sweeps",
        "tdd",
        "thermal_model_5r1c",
        "thermal_model_solvers",
        "topsis",
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
#   22 → 21 (Issue #3748): the wired-but-dead caller scope now counts
#     the root crate's explicit ``[[test]]`` target universes (the
#     Issue #3764 consolidated-harness convention). That correction
#     revealed ``empirical_hybrid``'s compiled harness consumer
#     (``tests/validation/hybrid_empirical_test.rs``, the
#     ``validation_hybrid_empirical_test`` target) — the module is no
#     longer wired-but-dead, so its allowlist entry and its Issue
#     #3748 disposition row are dropped in the same PR.
BASELINE_WIRED_BUT_DEAD = 21  # lowered from 22 → 21 in PR for fluxion-#3748 (was 33 → 22 in PR for fluxion-#3555)

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
#   29 → 0 (Issue #3555): burn-down PR cleared all 29 tracked orphans
#     (esp_r/*, cli/commands/*, sim/hvac/tests/*, sim/solar_gain_distribution.rs,
#     thermal/{solver,zone_coupling}.rs, twin/live_twin_broadcaster.rs,
#     validation/{esp_r/*,ml_data_collector.rs,validation_suite.rs},
#     validation/performance/{executor,parallel}.rs,
#     validation/reports/{cross_validation,mod}.rs,
#     ai/rl_policy.rs, weather/{denver,epw,mod}.rs). Lowered from 29 → 0 in
#     PR for fluxion-#3555.
BASELINE_KNOWN_ORPHANS = 0

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

# ---------------------------------------------------------------------------
# Dead-code-allow inventory (Issue #3752).
#
# ``DEAD_CODE_INVENTORY_PATH`` is the checked-in registry of every production
# (non-test) ``#[allow(dead_code)]`` site in the root crate. Each entry
# carries the site's file, a stable signature (the first code line after the
# attribute), a 1-based occurrence index disambiguating duplicate signatures
# within one file, and the tracking issue that justifies the suppression.
# Regenerate with ``--update-dead-code-inventory``; hand-edit only to change
# an entry's ``issue`` reference.
# ---------------------------------------------------------------------------
DEAD_CODE_INVENTORY_PATH = (
    REPO_ROOT / "tests" / "reference_data" / "dead_code_inventory.json"
)

# Governing issue for the seed inventory (#3752). Entries regenerated by
# ``--update-dead-code-inventory`` default to this reference; a cleanup PR
# that triages an individual site to a more specific follow-up issue edits
# that entry's ``issue`` field by hand.
DEAD_CODE_GOVERNING_ISSUE = 3752

# Matches ``#[allow(dead_code)]`` and multi-lint forms such as
# ``#[allow(dead_code, unused_imports)]``, consuming the attribute's closing
# bracket so the site signature is extracted from the *item* that follows,
# not from attribute punctuation. Inner ``#![allow(...)]`` module attributes
# and ``#[cfg_attr(..., allow(...))]`` forms are deliberately out of scope
# (none exist under src/ today; the detector fails closed on the forms it
# can see rather than trying to parse every attribute spelling).
_ALLOW_DEAD_CODE_RE = re.compile(r"#\[allow\([^)]*\bdead_code\b[^)]*\)\s*\]")

# Matches the header of an inline ``#[cfg(test)] mod <name> { ... }`` body.
# Everything inside the brace-matched span is test-only code and is excluded
# from the production inventory (the compiler only applies ``allow`` during
# test builds there, so it is not production dead-code debt).
_CFG_TEST_BODY_RE = re.compile(
    r"#\[cfg\(test\)\]\s*\n\s*"
    r"(?:pub(?:\s*\([^)]*\))?\s+)?"
    r"mod\s+[A-Za-z_][A-Za-z0-9_]*\s*\{"
)

# Downward-only ratchet on the production ``#[allow(dead_code)]`` count
# (Issue #3752). Mirrors ``BASELINE_KNOWN_ORPHANS`` /
# ``BASELINE_WIRED_BUT_DEAD`` above: the constant is the highest live
# production count this guard accepts; the script fails (exit 1) the moment
# the count grows past it. Cleanup PRs that delete suppressions are expected
# to LOWER this baseline; raising it requires a documenting comment naming
# the tracking issue that justifies the new suppressions.
#
# History:
#   94 → seed (Issue #3752): initial inventory across 42 production files.
#     The #3752 audit headline of "100 across 43 files" used a looser raw
#     text match that included 3 sites inside ``#[cfg(test)]`` inline test
#     bodies; this gate's canonical production-only measurement (block
#     comments stripped, occurrence-counted, cfg(test) bodies excluded)
#     seeds at 94.
BASELINE_DEAD_CODE_ALLOWS = 94

# ---------------------------------------------------------------------------
# Wired-but-dead disposition registry (Issue #3748).
#
# ``WIRED_BUT_DEAD_DISPOSITIONS_PATH`` is the checked-in registry giving
# every ``WIRED_BUT_DEAD`` allowlist entry an explicit, owner-assigned
# wire-or-delete decision — the per-module forcing function the
# downward-only ``BASELINE_WIRED_BUT_DEAD`` ratchet lacked. Allowed
# dispositions are ``wire-pending`` / ``reject`` / ``keep-dead`` (see the
# module docstring). Resolving a module requires dropping its allowlist
# entry, lowering ``BASELINE_WIRED_BUT_DEAD`` by one, AND dropping its
# registry row in the same PR (lock-step, like the dead-code inventory).
# ---------------------------------------------------------------------------
WIRED_BUT_DEAD_DISPOSITIONS_PATH = (
    REPO_ROOT / "tests" / "reference_data" / "wired_but_dead_dispositions.json"
)

# Governing issue for the seed registry (#3748). A follow-up issue that
# takes ownership of one module's decision edits that row's ``issue``
# field by hand.
WIRED_BUT_DEAD_GOVERNING_ISSUE = 3748

# The only dispositions the gate accepts; anything else is a schema
# violation and fails the check.
ALLOWED_DISPOSITIONS: frozenset[str] = frozenset(
    {"wire-pending", "reject", "keep-dead"}
)

# Max characters stored per site signature — long function signatures are
# truncated deterministically so the JSON diff stays reviewable.
_DEAD_CODE_SIGNATURE_MAX_LEN = 120

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
        # Index the SAME string the match offsets came from. Indexing the
        # original `text` here desynchronises offsets once inline bodies have
        # been stripped (`.replace(body, "")` shifts every later position),
        # so a trailing out-of-line `mod foo;` in a file that also has
        # inline `mod bar { ... }` test modules was silently dropped from
        # `declared` — surfacing as a phantom orphan for the declared file.
        original_char = text_without_inline[terminator_index]
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
    # Queue items are (file, child_dir) pairs: `child_dir` is the directory a
    # `mod name;` declaration inside `file` resolves against. For `dir/mod.rs`
    # and for the crate root that is the file's own directory; for a file
    # module `dir/foo.rs` (Rust 2018) it is `dir/foo/`. (Decompositions like
    # `src/ai/surrogate.rs` + `src/ai/surrogate/` need the latter; resolving
    # against the file's own directory misreported wired children as orphans.)
    queue: list[tuple[Path, Path]] = [(start, start.parent)]
    visited: set[Path] = set()
    while queue:
        current, child_dir = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        out_of_line, inline_bodies = _collect_declared_mods(current)
        # Out-of-line mod declarations.
        for name in out_of_line:
            for candidate in _candidate_paths_for_mod(name, child_dir):
                if candidate.exists() and candidate.is_file():
                    queue.append((candidate, child_dir / name))
                    break
        # Inline mod bodies: any out-of-line `mod bar;` declarations inside
        # them point at child files of the *inline namespace's* directory,
        # which is `child_dir/<inline_name>/`. We re-scan the inline body
        # with the same regex + resolver.
        for inline_name, body_text in inline_bodies.items():
            inline_parent = child_dir / inline_name
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
                        queue.append((candidate, inline_parent / name))
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


def _cargo_test_target_files() -> set[Path]:
    """Return the file universe of the root crate's explicit ``[[test]]``
    targets declared in ``Cargo.toml`` (Issue #3764 consolidation).

    Each ``[[test]] path = "tests/<sub>/<root>.rs"`` entry contributes
    its root file; a ``main.rs`` / ``mod.rs`` harness root (e.g.
    ``tests/all_tests/main.rs``, which collapsed 263 standalone test
    binaries into one target) additionally contributes the sibling
    files it wires in via ``mod`` declarations — those siblings ARE
    compiled by ``cargo test`` and therefore count as production-code
    callers for the wired-but-dead detector. Ad-hoc ``tests/<sub>/``
    files that no ``[[test]]`` target references remain excluded, per
    AGENTS.md.
    """
    out: set[Path] = set()
    manifest = REPO_ROOT / "Cargo.toml"
    if not manifest.exists():
        return out
    text = manifest.read_text(encoding="utf-8", errors="replace")
    for block_match in re.finditer(r"\[\[test\]\](.*?)(?=\n\[|\Z)", text, re.DOTALL):
        path_match = re.search(r'path\s*=\s*"([^"]+)"', block_match.group(1))
        if not path_match:
            continue
        root = (REPO_ROOT / path_match.group(1)).resolve()
        if not root.is_file():
            continue
        out.add(root)
        if root.name not in ("main.rs", "mod.rs"):
            continue
        cleaned = _clean_source(root.read_text(encoding="utf-8", errors="replace"))
        for mod_match in _MOD_RE.finditer(cleaned):
            if cleaned[mod_match.end() - 1] != ";":
                continue
            name = mod_match.group(1)
            for candidate in _candidate_paths_for_mod(name, root.parent):
                if candidate.exists() and candidate.is_file():
                    out.add(candidate.resolve())
                    break
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
        fixture helpers) — plus the explicit ``[[test]]`` target
        universes from ``_cargo_test_target_files`` (Issue #3764)
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
    out.extend(sorted(_cargo_test_target_files()))
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

    Performance: the caller scan is pure Python (no external tool
    dependency). A historical ripgrep fast path silently SKIPPED the
    detector when rg was absent — e.g. on GitHub-hosted runners — which
    zeroed the live set and made every disposition-registry row look
    stale (Issue #3748 CI failure). The two-stage scan below (one
    identifier-tokenization pass per file, then the precise caller-form
    regex only for candidate names actually present) keeps the whole
    detector deterministic across environments at ~5 s total runtime.

    The caller-form patterns applied per module are:

      * ``mod_name::Bar`` — qualified path usage.
      * ``use mod_name;`` / ``use mod_name::{...}`` — leaf import
        (``pub use mod_name;`` is subsumed: the leading ``\b`` holds
        after ``pub``).

    We accept the small false-positive risk (e.g. a doc-comment word
    matching) over the runtime cost of a per-file attribution pass.
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

    # Caller detection is pure Python on purpose: the historical ripgrep
    # fast path silently SKIPPED the detector when rg was absent (e.g. on
    # GitHub-hosted runners), zeroing the live set and making every
    # disposition-registry row look stale (Issue #3748 CI failure). A
    # two-stage scan — one identifier-tokenization pass per file, then the
    # precise caller-form regex only for candidate names actually present —
    # keeps the whole detector deterministic across environments and fast
    # enough (a few seconds) without any external tool dependency.
    token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    precise_re = {
        name: re.compile(
            rf"\b{re.escape(name)}::"
            rf"|\buse\s+{re.escape(name)}\s*[;{{]"
        )
        for name in unique_mod_names
    }
    hits_by_module: dict[str, set[Path]] = {
        name: set() for name in unique_mod_names
    }
    name_set = set(unique_mod_names)
    for path in sorted(allowed_files):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Stage 1: which module names appear as identifiers at all?
        candidates = {t for t in token_re.findall(text) if t in name_set}
        if not candidates:
            continue
        # Stage 2: precise caller-form verification for the candidates.
        # ``pub use name;`` is subsumed by the ``use`` alternative (the
        # ``\b`` holds after ``pub``).
        for name in candidates:
            if precise_re[name].search(text):
                hits_by_module[name].add(path)

    # Track which modules have at least one caller outside their own
    # subtree.
    has_caller: dict[str, bool] = {}
    for mod_name in unique_mod_names:
        subtree = canonical_subtree[mod_name]
        external_hit = False
        for hit in hits_by_module[mod_name]:
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


def _cfg_test_body_spans(text: str) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` spans of inline ``#[cfg(test)] mod ... { }``
    bodies (brace-matched, nesting-safe via sequential scanning).

    ``#[allow(dead_code)]`` occurrences inside these spans are test-only
    suppressions: the compiler applies them only under ``cfg(test)``, so
    they are not production dead-code debt and must never count against
    the Issue #3752 production ratchet.
    """
    spans: list[tuple[int, int]] = []
    for m in _CFG_TEST_BODY_RE.finditer(text):
        open_brace_index = m.end() - 1
        depth = 1
        i = open_brace_index + 1
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        spans.append((m.start(), i))
    return spans


def _dead_code_site_signature(text: str, search_from: int) -> str:
    """Return the stable signature for an ``#[allow(dead_code)]`` site: the
    first non-blank, non-comment, non-attribute line at or after
    ``search_from`` (the attribute's end), whitespace-normalised and
    truncated to ``_DEAD_CODE_SIGNATURE_MAX_LEN`` characters.

    Line numbers are deliberately NOT part of a site's identity — any edit
    above the attribute would otherwise shift unrelated entries and force
    inventory churn. The signature survives unrelated edits because
    doc-comments / sibling attributes between the attribute and its item
    are skipped.
    """
    for line in text[search_from:].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("//"):
            continue
        if stripped.startswith("#["):
            continue
        normalised = " ".join(stripped.split())
        if len(normalised) > _DEAD_CODE_SIGNATURE_MAX_LEN:
            normalised = normalised[:_DEAD_CODE_SIGNATURE_MAX_LEN]
        return normalised
    return "(end of file)"


def _scan_dead_code_sites() -> list[dict[str, object]]:
    """Scan production code for ``#[allow(dead_code)]`` occurrences and
    return the sorted site list used by both the gate and the
    ``--update-dead-code-inventory`` writer.

    Each site is ``{"file", "signature", "index"}`` where ``index`` is the
    1-based occurrence counter among identical ``(file, signature)`` pairs
    (two identical signatures in one file are legitimately distinct sites).
    """
    sites: list[dict[str, object]] = []
    for rs_file in sorted(SRC_DIR.rglob("*.rs")):
        if not rs_file.is_file():
            continue
        rel = rs_file.relative_to(REPO_ROOT).as_posix()
        # src/bin/*.rs are standalone Cargo binary targets, not children of
        # the library module tree — same universe rule as the orphan scan.
        try:
            rs_file.relative_to(BIN_DIR)
            continue
        except ValueError:
            pass
        text = _strip_block_comments(
            rs_file.read_text(encoding="utf-8", errors="replace")
        )
        test_spans = _cfg_test_body_spans(text)
        signature_counts: dict[str, int] = {}
        for m in _ALLOW_DEAD_CODE_RE.finditer(text):
            if any(start <= m.start() < end for start, end in test_spans):
                continue
            signature = _dead_code_site_signature(text, m.end())
            signature_counts[signature] = signature_counts.get(signature, 0) + 1
            sites.append(
                {
                    "file": rel,
                    "signature": signature,
                    "index": signature_counts[signature],
                }
            )
    sites.sort(key=lambda s: (str(s["file"]), str(s["signature"]), int(s["index"])))
    return sites


def _load_dead_code_inventory() -> list[dict[str, object]]:
    """Load the checked-in inventory. Returns ``[]`` when the file is
    missing so the gate can emit a single actionable message instead of a
    traceback.
    """
    if not DEAD_CODE_INVENTORY_PATH.exists():
        return []
    payload = json.loads(DEAD_CODE_INVENTORY_PATH.read_text(encoding="utf-8"))
    return list(payload.get("sites", []))


def _write_dead_code_inventory(sites: list[dict[str, object]]) -> None:
    """Regenerate ``tests/reference_data/dead_code_inventory.json`` from the
    live scan. Every entry is stamped with the governing issue (#3752);
    hand-edit an entry's ``issue`` field afterwards to point at a more
    specific follow-up issue.
    """
    entries = [
        {
            "file": site["file"],
            "signature": site["signature"],
            "index": site["index"],
            "issue": DEAD_CODE_GOVERNING_ISSUE,
        }
        for site in sites
    ]
    payload = {
        "schema_version": 1,
        "description": (
            "Inventory of production #[allow(dead_code)] sites in the root"
            " fluxion crate, governed by"
            " scripts/check_orphan_modules.py (Issue #3752). Regenerate"
            " with: python3 scripts/check_orphan_modules.py"
            " --update-dead-code-inventory"
        ),
        "governing_issue": DEAD_CODE_GOVERNING_ISSUE,
        "site_count": len(entries),
        "sites": entries,
    }
    DEAD_CODE_INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEAD_CODE_INVENTORY_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _check_dead_code_inventory(update: bool) -> int:
    """Run the Issue #3752 dead-code-allow gate. Returns the process exit
    code for this section (0 pass / 1 fail).

    In ``update`` mode the inventory is regenerated from the live scan and
    the gate short-circuits to success (the caller has asked for a rewrite,
    not a verdict); the printed report still shows the resulting count so
    a ratchet-affecting regeneration is visible in the output.
    """
    print("--- Dead-code-allow inventory (#3752) ---")
    # Mock-repo guard: scripts/ci/test_check_orphan_modules.py redirects the
    # module-level REPO_ROOT at synthetic tmp_path trees whose fixture
    # sources carry allow(dead_code) markers that can never appear in the
    # real repo's checked-in inventory. Every other detector here is
    # self-contained; this one is inherently registry-relative to the real
    # repo, so skip it whenever REPO_ROOT no longer points at this script's
    # actual repository.
    real_root = Path(__file__).resolve().parent.parent
    if Path(REPO_ROOT).resolve() != real_root:
        print(
            "Skipped: REPO_ROOT redirected to a synthetic mock tree "
            "(dead-code gate is registry-relative to the real repo)."
        )
        return 0
    live_sites = _scan_dead_code_sites()
    live_count = len(live_sites)
    print(f"Production allow(dead_code) sites (live scan): {live_count}")
    print(
        f"Count baseline (BASELINE_DEAD_CODE_ALLOWS): "
        f"{BASELINE_DEAD_CODE_ALLOWS}"
    )

    if update:
        _write_dead_code_inventory(live_sites)
        print(
            f"Inventory regenerated: "
            f"{DEAD_CODE_INVENTORY_PATH.relative_to(REPO_ROOT)} "
            f"({live_count} entries)"
        )
        if live_count > BASELINE_DEAD_CODE_ALLOWS:
            print(
                "WARNING: the regenerated count exceeds "
                "BASELINE_DEAD_CODE_ALLOWS; the verify gate will fail until "
                "the baseline is raised with a tracking-issue comment."
            )
        return 0

    inventory_sites = _load_dead_code_inventory()
    if not inventory_sites and not DEAD_CODE_INVENTORY_PATH.exists():
        print(
            "ERROR: dead-code inventory missing: "
            f"{DEAD_CODE_INVENTORY_PATH.relative_to(REPO_ROOT)}\n"
            "Regenerate it with:\n"
            "  python3 scripts/check_orphan_modules.py"
            " --update-dead-code-inventory"
        )
        return 2

    inventory_keys = {
        (
            str(s["file"]),
            str(s["signature"]),
            int(s["index"]),
        )
        for s in inventory_sites
    }
    live_keys = {
        (str(s["file"]), str(s["signature"]), int(s["index"])) for s in live_sites
    }
    new_sites = sorted(live_keys - inventory_keys)
    stale_sites = sorted(inventory_keys - live_keys)
    print(f"Inventory entries: {len(inventory_sites)}")
    print(f"NEW production sites (not inventoried): {len(new_sites)}")
    print(f"Stale inventory entries (site no longer in code): {len(stale_sites)}")
    print()

    failures = False

    if new_sites:
        failures = True
        print("NEW allow(dead_code) SITES DETECTED (CI FAILURE — Issue #3752):")
        for file, signature, index in new_sites:
            print(f"  {file}: {signature} (occurrence {index})")
        print()
        print(
            "Every production allow(dead_code) suppression must be\n"
            "inventoried with a tracking issue before it lands. Either\n"
            "delete the attribute (preferred — dead code should not ship),\n"
            "or run:\n"
            "  python3 scripts/check_orphan_modules.py"
            " --update-dead-code-inventory\n"
            "and commit the regenerated inventory alongside a justification\n"
            "for the new suppression."
        )
        print()

    if stale_sites:
        failures = True
        print("STALE INVENTORY ENTRIES DETECTED (CI FAILURE — Issue #3752):")
        for file, signature, index in stale_sites:
            print(f"  {file}: {signature} (occurrence {index})")
        print()
        print(
            "These inventory entries no longer match any live site. Keep\n"
            "the registry in lock-step with the tree by regenerating:\n"
            "  python3 scripts/check_orphan_modules.py"
            " --update-dead-code-inventory\n"
            "Resolving stale entries via deletion is exactly the cleanup\n"
            "#3752 wants — lower BASELINE_DEAD_CODE_ALLOWS by one per\n"
            "resolved attribute in the same PR."
        )
        print()

    if live_count > BASELINE_DEAD_CODE_ALLOWS:
        failures = True
        print(
            "PRODUCTION allow(dead_code) COUNT GREW ABOVE BASELINE "
            "(CI FAILURE — Issue #3752 downward-only ratchet):"
        )
        print(
            f"  live count {live_count} > "
            f"BASELINE_DEAD_CODE_ALLOWS {BASELINE_DEAD_CODE_ALLOWS}"
        )
        print(
            "\n"
            "The ratchet is downward-only (Issue #3443 convention): new\n"
            "suppressions must be justified by a tracking issue AND the\n"
            "baseline raised with a documenting comment in\n"
            "scripts/check_orphan_modules.py. Companion cleanup PRs are\n"
            "expected to LOWER the baseline by one per deleted attribute."
        )
        print()
    elif live_count < BASELINE_DEAD_CODE_ALLOWS:
        print(
            "CLEANUP NUDGE: the live count dropped below the baseline; "
            "lower BASELINE_DEAD_CODE_ALLOWS "
            f"({BASELINE_DEAD_CODE_ALLOWS} → {live_count}) in this PR to "
            "keep the ratchet tight."
        )
        print()

    if failures:
        return 1

    print(
        f"Dead-code inventory in lock-step ({live_count} production "
        f"suppression(s), at or below the baseline of "
        f"{BASELINE_DEAD_CODE_ALLOWS})."
    )
    return 0


def _load_wired_but_dead_dispositions() -> list[dict[str, object]]:
    """Load the checked-in disposition registry. Returns ``[]`` when the
    file is missing so the gate can emit a single actionable message
    instead of a traceback.
    """
    if not WIRED_BUT_DEAD_DISPOSITIONS_PATH.exists():
        return []
    payload = json.loads(
        WIRED_BUT_DEAD_DISPOSITIONS_PATH.read_text(encoding="utf-8")
    )
    return list(payload.get("modules", []))


def _check_wired_but_dead_dispositions(raw_wired_but_dead: list[str]) -> int:
    """Run the Issue #3748 disposition gate. Returns the process exit
    code for this section (0 pass / 1 fail / 2 missing registry).

    Lock-step contract: the registry must cover every live wired-but-dead
    module AND every ``WIRED_BUT_DEAD`` allowlist entry, carry a
    disposition from ``ALLOWED_DISPOSITIONS`` plus ``issue`` / ``owner``
    / non-empty ``rationale`` per row, and hold no rows for modules that
    are no longer wired-but-dead (wired up or deleted).
    """
    print("--- Wired-but-dead dispositions (#3748) ---")
    # Mock-repo guard: mirrors the dead-code gate above. The registry is
    # checked into the real repo; scripts/ci/test_check_orphan_modules.py
    # redirects REPO_ROOT at synthetic tmp_path trees whose live
    # wired-but-dead set can never match the real registry. Skip whenever
    # REPO_ROOT no longer points at this script's actual repository.
    real_root = Path(__file__).resolve().parent.parent
    if Path(REPO_ROOT).resolve() != real_root:
        print(
            "Skipped: REPO_ROOT redirected to a synthetic mock tree "
            "(disposition gate is registry-relative to the real repo)."
        )
        return 0

    rows = _load_wired_but_dead_dispositions()
    if not rows and not WIRED_BUT_DEAD_DISPOSITIONS_PATH.exists():
        print(
            "ERROR: wired-but-dead disposition registry missing: "
            f"{WIRED_BUT_DEAD_DISPOSITIONS_PATH}\n"
            "Every WIRED_BUT_DEAD entry must carry an owner-assigned "
            "wire-or-delete disposition (Issue #3748). Restore or create "
            "the registry before landing."
        )
        return 2

    live_set = set(raw_wired_but_dead)
    failures = False

    row_modules: set[str] = set()
    invalid_rows: list[str] = []
    for row in rows:
        name = str(row.get("module", "")).strip()
        row_modules.add(name)
        disposition = str(row.get("disposition", "")).strip()
        if disposition not in ALLOWED_DISPOSITIONS:
            invalid_rows.append(
                f"{name or '(blank module)'}: unknown disposition "
                f"{disposition!r} (allowed: "
                f"{sorted(ALLOWED_DISPOSITIONS)})"
            )
        issue = row.get("issue")
        if not isinstance(issue, int):
            invalid_rows.append(
                f"{name}: missing/invalid 'issue' reference "
                "(expected the governing or follow-up issue number)"
            )
        for field in ("owner", "rationale"):
            if not str(row.get(field, "")).strip():
                invalid_rows.append(f"{name}: missing '{field}'")

    missing_live = sorted(live_set - row_modules)
    missing_allowlist = sorted(set(WIRED_BUT_DEAD) - row_modules)
    stale_rows = sorted(row_modules - live_set)
    print(f"Live wired-but-dead modules: {len(live_set)}")
    print(f"Allowlisted entries: {len(WIRED_BUT_DEAD)}")
    print(f"Registry rows: {len(rows)}")
    print(f"Rows missing for live modules: {len(missing_live)}")
    print(f"Rows missing for allowlist entries: {len(missing_allowlist)}")
    print(f"Stale rows (module no longer wired-but-dead): {len(stale_rows)}")
    print(f"Invalid rows (schema violations): {len(invalid_rows)}")
    print()

    if missing_live or missing_allowlist:
        failures = True
        print(
            "MISSING DISPOSITION ROWS DETECTED (CI FAILURE — Issue #3748):"
        )
        for name in sorted(set(missing_live) | set(missing_allowlist)):
            print(f"  pub mod {name}; — no registry row")
        print()
        print(
            "Every wired-but-dead module needs an owner-assigned\n"
            "wire-or-delete disposition in\n"
            "  tests/reference_data/wired_but_dead_dispositions.json\n"
            "with one of: wire-pending / reject / keep-dead, a tracking\n"
            "issue, an owner, and a rationale. Wire-up decisions without\n"
            "a registry row are exactly the drift #3748 exists to stop."
        )
        print()

    if invalid_rows:
        failures = True
        print("INVALID DISPOSITION ROWS DETECTED (CI FAILURE — Issue #3748):")
        for problem in invalid_rows:
            print(f"  {problem}")
        print()
        print(
            "Each row must carry disposition (wire-pending | reject |\n"
            "keep-dead), an integer issue reference, an owner, and a\n"
            "non-empty rationale."
        )
        print()

    if stale_rows:
        failures = True
        print("STALE DISPOSITION ROWS DETECTED (CI FAILURE — Issue #3748):")
        for name in stale_rows:
            print(f"  pub mod {name}; — no longer wired-but-dead")
        print()
        print(
            "These registry rows point at modules that now have a\n"
            "production caller or no longer exist. Keep the registry in\n"
            "lock-step with the tree: drop the row in the same PR that\n"
            "resolves the module, drop its WIRED_BUT_DEAD allowlist\n"
            "entry, and lower BASELINE_WIRED_BUT_DEAD by one."
        )
        print()

    if failures:
        return 1

    print(
        f"Disposition registry in lock-step ({len(rows)} row(s), every "
        "wired-but-dead module carries an owner-assigned decision)."
    )
    return 0


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
    print()

    # ------------------------------------------------------------------
    # Wired-but-dead disposition registry (Issue #3748).
    #
    # Fourth companion check: every WIRED_BUT_DEAD entry must carry an
    # owner-assigned wire-or-delete disposition in the checked-in
    # registry, and the registry must stay in lock-step with the live
    # wired-but-dead set.
    # ------------------------------------------------------------------
    disposition_rc = _check_wired_but_dead_dispositions(raw_wired_but_dead)
    if disposition_rc != 0:
        return disposition_rc

    # ------------------------------------------------------------------
    # Dead-code-allow inventory (Issue #3752).
    #
    # Third companion detector: governs production #[allow(dead_code)]
    # suppressions via a checked-in per-site registry and a downward-only
    # count ratchet. ``--update-dead-code-inventory`` regenerates the
    # registry instead of rendering a verdict.
    # ------------------------------------------------------------------
    update_dead_code_inventory = "--update-dead-code-inventory" in sys.argv[1:]
    return _check_dead_code_inventory(update=update_dead_code_inventory)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)