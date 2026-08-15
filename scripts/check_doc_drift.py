#!/usr/bin/env python3
"""
Doc-Comment Drift Detector for cycle claims (Issue #2895).

The architecture-drift detector (``scripts/check_architecture_drift.py``)
compares ``ARCHITECTURE.md`` against the *codebase* (traits, modules,
dataflow). It does **not** read the doc-comments inside Rust files, so a
stale sentence like

    //! The `sim::construction ↔ physics::continuous` cycle remains and is
    //! the next cycle-break target

can sit in ``fluxion-core/src/lib.rs`` for months even after the cycle
participants become one-line ``pub use`` re-export shims (issue #1718).

This script closes that gap. It:

1. Parses every doc-comment in ``fluxion-core/src/lib.rs`` that names a
   cycle (e.g. ``the `physics ↔ sim` cycle``, ``the
   `sim::construction ↔ physics::continuous` cycle``, or the bare
   ``physics<->sim cycle``).
2. Classifies the cycle claim by **tense**:
   * *Past-tense* (``broke``, ``was closed``, ``Breaks``, ...) describes
     a historical cycle-break — drift-immune.
   * *Present-tense* (``remains``, ``persists``, ``is the next
     cycle-break target``, ``still depends``, ...) describes the
     **current** state — drift-sensitive.
3. For each present-tense claim, cross-references the named module pair
   against the **current** codebase state:

   * One-line ``pub use`` re-export shims cannot participate in a cycle
     (issue #1718) — any present-tense claim that names two such
     modules, or a single shim named as a cycle participant, is flagged
     as drift.
   * A present-tense claim ("X ↔ Y cycle remains", "is the next
     cycle-break target", ...) must be **absent** from the
     ``Resolved by #...`` / strikethrough entries in
     ``ARCHITECTURE.md`` §"Remaining cycles (deferred to follow-up
     issues)" — if ARCHITECTURE.md explicitly marks the pair as
     resolved, the doc-comment is drift.

4. Reports each mismatch with ``file:line`` so a reviewer can update
   the doc-comment or revert the code change.

Usage::

    python3 scripts/check_doc_drift.py

Exit codes:

  0 — no doc-comment drift detected
  1 — drift detected (a present-tense cycle claim contradicts the
      current cycle baselines)
  2 — script error

See issue #2895 and ``ARCHITECTURE.md`` §"Remaining cycles" for the
source-of-truth list of currently-active cycles.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIB_RS = REPO_ROOT / "fluxion-core" / "src" / "lib.rs"
ARCHITECTURE_MD = REPO_ROOT / "ARCHITECTURE.md"
FLUXION_SRC = REPO_ROOT / "src"
FLUXION_CORE_SRC = REPO_ROOT / "fluxion-core" / "src"

# ---------------------------------------------------------------------------
# Cycle-claim detection in doc-comments
# ---------------------------------------------------------------------------
#
# Cycle claims appear in three shapes in the wild. We match all three:
#
#   1. Whole-phrase backticks (the most common in ARCHITECTURE.md /
#      long-form doc-comments):
#         the `physics ↔ sim` cycle
#         the `sim ↔ validation` cycle
#
#   2. Two-side backticks (module paths joined by a glyph):
#         the `sim::construction ↔ validation::ashrae_140_cases::Orientation` cycle
#
#   3. Bare arrow text (older prose):
#         the physics<->sim dependency cycle
#
# All three are normalised to ``(left_module, right_module, glyph)``
# tuples and joined into the cycle claim set.

# Whole-phrase backticks: `LEFT ↔ RIGHT` (or <-> / <->). LEFT and RIGHT
# may be a short label like "physics" / "sim" or a full module path.
_WHOLE_PHRASE_RE = re.compile(
    r"`([A-Za-z0-9_:<>{},\s]+?)\s*(?:↔|<->|<->)\s*([A-Za-z0-9_:<>{},\s]+?)`"
    r"\s+(?:dependency\s+)?cycle\b",
    re.IGNORECASE,
)

# Two-side backticks: `LEFT` ↔ `RIGHT` (modules joined by a glyph).
# Each module path component is `[A-Za-z0-9_]+` so we accept both
# lowercase module names (``sim::construction``) and CamelCase types
# (``validation::ashrae_140_cases::Orientation``).
_TWO_SIDE_RE = re.compile(
    r"`([A-Za-z0-9_:<>{},\s]+?)`\s*(?:↔|<->|<->)\s*`([A-Za-z0-9_:<>{},\s]+?)`"
    r"\s+cycle\b",
    re.IGNORECASE,
)

# Bare arrow text: physics<->sim cycle (no backticks).
_BARE_ARROW_RE = re.compile(
    r"\b([a-z_]+)\s*<->\s*([a-z_]+)\s+(?:dependency\s+)?cycle\b",
    re.IGNORECASE,
)

# Present-tense cycle-state verbs to flag (case-insensitive). Past-tense
# verbs (broke, was closed, was resolved, breaks, ...) are drift-immune
# and not listed.
_PRESENT_VERBS = re.compile(
    r"\b(?:remains?|persists?|is\s+(?:the\s+)?(?:next\s+)?(?:cycle-?break\s+)?target|"
    r"still\s+(?:depends?|has|holds?)|is\s+(?:currently\s+)?(?:the\s+)?(?:active|open|unresolved))\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# ARCHITECTURE.md §"Remaining cycles" parsing
# ---------------------------------------------------------------------------
_REMAINING_CYCLES_SECTION_RE = re.compile(
    r"### Remaining cycles \(deferred to follow-up issues\)(.*?)(?=\n### |\Z)",
    re.DOTALL,
)
# Match a bullet that wraps its label in `~~...~~` (Markdown strikethrough).
# Carries forward the same single-backtick `X ↔ Y` extraction used for the
# main cycle-pair detection.
_STRIKETHROUGH_BULLET_RE = re.compile(
    r"-\s+~~[^\n]*~~",
    re.DOTALL,
)


def _strip_doc_marker(line: str) -> str:
    """Remove the leading ``//!`` or ``///`` (or ``/**``) marker."""
    return re.sub(r"^/\*\*?!?", "", line).strip()


def _normalize_pair(left: str, right: str) -> tuple[str, str]:
    """Canonicalise module identifiers for shim lookups.

    Strips crate prefixes (``crate::``, ``fluxion::``,
    ``fluxion_core::``), drops surrounding whitespace, and lowercases so
    we can compare against on-disk ``.rs`` paths uniformly. Also
    truncates at the first ``{`` so braces-list forms like
    ``fluxion::physics::{wall_spec, method_selector}`` are reduced to
    the leading module.
    """

    def clean(s: str) -> str:
        s = s.strip().strip("`")
        for prefix in ("crate::", "fluxion::", "fluxion_core::"):
            if s.startswith(prefix):
                s = s[len(prefix) :]
        # Strip braces-list continuation
        if "{" in s:
            s = s.split("{", 1)[0].rstrip("::").rstrip(":")
        return s.strip().lower()

    return clean(left), clean(right)


def _pair_key(left: str, right: str) -> str:
    """Canonical pair key for set membership (``a↔b``)."""
    return f"{left}↔{right}"


def _shim_modules() -> set[str]:
    """Collect every cycle-incapable module under ``src/`` and ``fluxion-core/src/``.

    A "cycle-incapable" module is a one-line ``pub use`` re-export shim
    — pure forwarding that exists only to keep historical import paths
    alive after a crate split. Such a module contains no logic for a
    function call to traverse, so it cannot participate in a cycle.

    Returns the set of normalised module paths (e.g. ``{"physics::continuous"}``).
    """
    shims: set[str] = set()
    for root in (FLUXION_SRC, FLUXION_CORE_SRC):
        if not root.exists():
            continue
        for rs_file in root.rglob("*.rs"):
            rel = rs_file.relative_to(root)
            parts = rel.with_suffix("").as_posix().split("/")
            module = "::".join(parts).lower()
            if _is_one_line_reexport_shim(rs_file):
                shims.add(module)
    return shims


def _is_one_line_reexport_shim(rs_path: Path) -> bool:
    """True if ``rs_path`` is a one-line ``pub use`` re-export shim.

    Strips comments and attribute lines (carry-forward of the
    comment-strip technique used in ``check_physics_sim_cycle.py``).
    A file is a shim iff it has 1–2 non-comment, non-attribute lines,
    every one of which is a ``use`` / ``pub use`` statement.
    """
    if not rs_path.exists():
        return False
    content = rs_path.read_text(encoding="utf-8", errors="replace")
    code_lines: list[str] = []
    for raw in content.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("//") or stripped.startswith("*"):
            continue
        if stripped.startswith("#"):
            continue
        code_lines.append(stripped)
    if not code_lines or len(code_lines) > 2:
        return False
    return all(
        re.match(r"^(pub\s+)?use\s+[A-Za-z0-9_:{},\s\*]+;\s*$", line)
        for line in code_lines
    )


def _scan_lib_rs_for_cycle_claims() -> list[tuple[int, str, str, bool]]:
    """Find every doc-comment cycle claim in ``fluxion-core/src/lib.rs``.

    Returns a list of ``(line_no, left_module, right_module, present_tense)``
    tuples. ``present_tense`` is True iff the doc-comment asserts the
    cycle's current state (drift-sensitive); False for past-tense
    historical descriptions (drift-immune).
    """
    if not LIB_RS.exists():
        return []
    claims: list[tuple[int, str, str, bool]] = []
    for lineno, raw in enumerate(
        LIB_RS.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        stripped = raw.strip()
        if not (stripped.startswith("//!") or stripped.startswith("///")):
            continue
        body = _strip_doc_marker(stripped)
        present = bool(_PRESENT_VERBS.search(body))
        seen_pairs: set[tuple[str, str]] = set()
        for regex in (_WHOLE_PHRASE_RE, _TWO_SIDE_RE, _BARE_ARROW_RE):
            for match in regex.finditer(body):
                left, right = _normalize_pair(match.group(1), match.group(2))
                pair = (left, right)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                claims.append((lineno, left, right, present))
    return claims


def _parse_architecture_remaining_cycles() -> tuple[set[str], set[str]]:
    """Extract the cycle inventory from ``ARCHITECTURE.md``.

    Returns two sets of canonical pair keys (``left↔right``):

    * ``active`` — non-strikethrough bullets listing cycles that
      ARCHITECTURE.md says *remain*.
    * ``resolved`` — strikethrough (``~~...~~``) bullets or bullets
      explicitly labelled "Resolved by #..." — cycles ARCHITECTURE.md
      records as retired.

    Bullets are matched in the §"Remaining cycles (deferred to
    follow-up issues)" section. Module pairs are extracted from any
    backticked ``X ↔ Y`` (or ``X <-> Y``) phrase in the bullet; if a
    bullet has no ``↔`` pair, its leading text is treated as the
    bullet's identity (used to mark the entry as resolved/active
    without a specific pair).
    """
    if not ARCHITECTURE_MD.exists():
        return set(), set()
    text = ARCHITECTURE_MD.read_text(encoding="utf-8", errors="replace")
    section_match = _REMAINING_CYCLES_SECTION_RE.search(text)
    if not section_match:
        return set(), set()
    section = section_match.group(1)
    active: set[str] = set()
    resolved: set[str] = set()
    # Bullets are lines starting with `- ` after the section header.
    for bullet_match in re.finditer(r"- ([^\n]+)", section):
        bullet = bullet_match.group(1).strip()
        # Markdown strikethrough marker around the label, OR an explicit
        # "Resolved by #..." somewhere in the bullet text.
        is_resolved = (bullet.startswith("~~") and bullet.endswith("~~")) or (
            "Resolved by" in bullet
        )
        # Try to extract a `X ↔ Y` pair from the leading line of the
        # bullet. If none found, fall back to the first backticked
        # module on the bullet.
        pair = _extract_pair_from_bullet(bullet)
        if not pair:
            continue
        if is_resolved:
            resolved.add(pair)
        else:
            active.add(pair)
    return active, resolved


def _extract_pair_from_bullet(bullet: str) -> str | None:
    """Pull a canonical pair key from a bullet's first line.

    Resolution order:

    1. ``X ↔ Y`` / ``X <-> Y`` / ``X <-> Y`` glyph-joined pair (the
       most informative; appears in ARCHITECTURE.md §"Cycle break").
    2. ``X still depends on Y`` / ``X references Y`` prose: take the
       first two backticked module identifiers (e.g.
       ``~~`fluxion::sim::construction` still depends on
       `fluxion::physics::continuous`.~~``). This is the common form
       in §"Remaining cycles" strikethrough bullets.
    3. First backticked module only: marks the bullet as active/resolved
       against a one-sided dependency (``fluxion::physics::{...}``
       intra-physics edges).
    """
    pair_match = re.search(
        r"([A-Za-z0-9_:<>{}\s]+?)\s*(?:↔|<->|<->)\s*([A-Za-z0-9_:<>{}\s]+?)(?:~~|\.|`|$)",
        bullet,
    )
    if pair_match:
        left, right = _normalize_pair(pair_match.group(1), pair_match.group(2))
        if left and right:
            return _pair_key(left, right)
    # Fallback: collect all backticked module identifiers. If we get
    # >=2, treat them as a pair (matches "X still depends on Y" prose).
    backticked = re.findall(r"`([A-Za-z0-9_:<>{},\s]+?)`", bullet)
    if len(backticked) >= 2:
        left, right = _normalize_pair(backticked[0], backticked[1])
        if left and right:
            return _pair_key(left, right)
    if len(backticked) == 1:
        left = _normalize_pair(backticked[0], "")[0]
        if left:
            return _pair_key(left, "*")
    return None


def main() -> int:
    print(f"Doc-comment drift check for cycle claims (issue #2895; repo: {REPO_ROOT})")
    print()

    failures: list[str] = []

    print("[1/4] scanning fluxion-core/src/lib.rs doc-comments for cycle claims ...")
    claims = _scan_lib_rs_for_cycle_claims()
    if not claims:
        print("    OK: no cycle claims found in doc-comments")
    else:
        past = sum(1 for c in claims if not c[3])
        pres = sum(1 for c in claims if c[3])
        print(
            f"    OK: found {len(claims)} cycle claim(s) "
            f"({past} past-tense, {pres} present-tense)"
        )

    print("[2/4] resolving ARCHITECTURE.md §'Remaining cycles' inventory ...")
    active_pairs, resolved_pairs = _parse_architecture_remaining_cycles()
    print(
        f"    OK: {len(active_pairs)} active pair(s), "
        f"{len(resolved_pairs)} resolved pair(s) documented"
    )

    print("[3/4] indexing one-line `pub use` re-export shims ...")
    shim_modules = _shim_modules()
    print(f"    OK: {len(shim_modules)} shim module(s) found")
    if shim_modules:
        sample = sorted(shim_modules)[:8]
        print(
            f"    e.g. {', '.join(sample)}" + (" ..." if len(shim_modules) > 8 else "")
        )

    print(
        "[4/4] cross-referencing present-tense cycle claims against current state ..."
    )

    for line_no, left, right, present in claims:
        # Past-tense claims are historical context, not drift.
        if not present:
            continue
        key = _pair_key(left, right)
        key_rev = _pair_key(right, left)

        # Drift case 1: ARCHITECTURE.md marks this pair as resolved.
        if key in resolved_pairs or key_rev in resolved_pairs:
            failures.append(
                f"{LIB_RS.relative_to(REPO_ROOT)}:{line_no}: doc-comment "
                f"claims `{left} ↔ {right}` cycle currently remains/is the "
                f"next target, but ARCHITECTURE.md §'Remaining cycles' "
                f"lists this pair as resolved/struck-through"
            )
            continue

        # Drift case 2: both modules are one-line re-export shims.
        left_shim = left in shim_modules
        right_shim = right in shim_modules
        if left_shim and right_shim:
            failures.append(
                f"{LIB_RS.relative_to(REPO_ROOT)}:{line_no}: doc-comment "
                f"claims `{left} ↔ {right}` cycle currently remains, but "
                f"both `{left}` and `{right}` are one-line `pub use` "
                f"re-export shims (no logic to participate in a cycle; "
                f"see ARCHITECTURE.md §'Remaining cycles' entry for this pair)"
            )
            continue

        # Drift case 3: one side is a shim AND the pair is not in the
        # active-cycles list. Conservative — a shim could legitimately
        # remain until the cycle is fully retired (the ARCHITECTURE.md
        # active list is the source of truth for "this cycle is real").
        if (left_shim or right_shim) and (
            key not in active_pairs and key_rev not in active_pairs
        ):
            shim_side = left if left_shim else right
            failures.append(
                f"{LIB_RS.relative_to(REPO_ROOT)}:{line_no}: doc-comment "
                f"claims `{left} ↔ {right}` cycle currently remains, but "
                f"`{shim_side}` is a one-line `pub use` re-export shim and "
                f"the pair is not listed in ARCHITECTURE.md §'Remaining "
                f"cycles' as active"
            )

    print()
    if failures:
        print("DOC-COMMENT DRIFT DETECTED:")
        for f in failures:
            print(f"  {f}")
        print()
        print("A doc-comment in fluxion-core/src/lib.rs names a cycle state")
        print("that the current codebase contradicts. Either:")
        print("  1. Update the doc-comment to reflect the resolved state")
        print("     (the cycle was broken by a prior issue, e.g. #2462).")
        print("  2. Revert the code change that retired the cycle edge")
        print("     (only valid if the cycle was re-introduced).")
        print("  3. Add the cycle to ARCHITECTURE.md §'Remaining cycles'")
        print("     as a struck-through '~~...~~' entry to acknowledge it")
        print("     was retired by a specific issue.")
        return 1

    present_claims = [c for c in claims if c[3]]
    print(
        f"No doc-comment drift. {len(present_claims)} present-tense "
        f"cycle claim(s) in fluxion-core/src/lib.rs are consistent with "
        f"the current cycle baselines ({len(active_pairs)} active, "
        f"{len(resolved_pairs)} resolved)."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
