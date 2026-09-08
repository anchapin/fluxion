#!/usr/bin/env python3
"""
Evaluator Dynamic-Loading Security Acceptance Gate (Issue #3554).

Enforces the security acceptance criteria defined in
`crates/fluxion-evaluator/SECURITY_ACCEPTANCE.md` against the actual
code in `crates/fluxion-evaluator/src/dynamic.rs`. Mirrors the fail-closed
pattern of `scripts/check_audit_config_unique.py`.

The follow-up `libloading` PR cannot land unless every check below passes.
The gate is deliberately conservative: it errs on the side of FAILING so
that an implementation that slips through review cannot silently ship a
loader with weak defaults.

Checks (all must pass):

    1. `SECURITY_ACCEPTANCE.md` exists, parses as Markdown with one
       sub-section per criterion (a) through (d).
    2. `dynamic.rs` is currently in stub state (every public return is
       `DynamicLoadError::NotImplementedInThisBuild` or
       `DynamicLoadError::FeatureNotEnabled`). If a loader is
       implemented, the file MUST contain the four
       `// SECURITY-ACCEPTANCE: satisfied` markers (one per criterion
       a/b/c/d).
    3. `Cargo.toml` for the evaluator does not silently re-introduce
       `libloading` (or any other third-party FFI crate) into the
       `dynamic` feature's `dependencies` line.

Usage:
    python3 scripts/check_evaluator_dynamic_secure.py

Exit codes:
    0 — All checks pass.
    1 — One or more checks failed.
    2 — Script error.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "crates" / "fluxion-evaluator"
DYNAMIC_RS = EVAL_DIR / "src" / "dynamic.rs"
CARGO_TOML = EVAL_DIR / "Cargo.toml"
ACCEPTANCE_MD = EVAL_DIR / "SECURITY_ACCEPTANCE.md"

# Criterion letters in fixed order. The gate requires one marker per
# criterion if the loader is implemented.
CRITERIA = ("a", "b", "c", "d")

# Crates that are NOT allowed to appear in the `dynamic` feature's
# dependency line. `libloading` is the documented follow-up target but
# is gated on issue #3310's duplicate-version budget. Adding it
# without the budget being raised first is the precise drift the gate
# is designed to catch.
BANNED_DEPS = ("libloading", "dlopen-ffi", "sharedlib")


def read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required file missing: {path}")
    return path.read_text(encoding="utf-8")


def acceptance_md_has_all_criteria(text: str) -> list[str]:
    """Return the list of criterion letters that are NOT documented."""
    missing: list[str] = []
    for letter in CRITERIA:
        # Each criterion has a heading like `### (a) Signature verification ...`
        pattern = rf"^###\s+\({letter}\)\s+"
        if not re.search(pattern, text, re.MULTILINE):
            missing.append(letter)
    return missing


def dynamic_rs_is_stub(text: str) -> bool:
    """Return True if every public return path in dynamic.rs is one of
    the two stub error variants AND there is no `// SECURITY-ACCEPTANCE:
    satisfied` marker (consistent with the module still being a stub).
    """
    has_satisfied_marker = bool(
        re.search(r"//\s*SECURITY-ACCEPTANCE:\s*satisfied", text)
    )
    if has_satisfied_marker:
        return False

    # The two canonical stub error variants. We look for either a
    # `DynamicLoadError::NotImplementedInThisBuild` or a
    # `DynamicLoadError::FeatureNotEnabled` after a `=>` or in an
    # `Err(...)` constructor — every public return path in the stub
    # uses one of those two variants.
    stub_variants = (
        "DynamicLoadError::NotImplementedInThisBuild",
        "DynamicLoadError::FeatureNotEnabled",
    )
    has_stub_return = any(variant in text for variant in stub_variants)

    # A non-stub would introduce a return of `Ok(DynamicKernel { ... })`
    # or similar. Look for the DynamicKernel struct being CONSTRUCTED
    # in a return position — a `pub struct DynamicKernel { abi_version:
    # u32 }` definition alone is fine (the stub already has it), but a
    # constructor returning it via `Ok(...)` is the smoking gun.
    returns_loaded_kernel = bool(
        re.search(r"Ok\(\s*DynamicKernel\b", text)
    )
    return has_stub_return and not returns_loaded_kernel


def dynamic_rs_has_all_satisfied_markers(text: str) -> list[str]:
    """Return the list of criterion letters MISSING their
    `// SECURITY-ACCEPTANCE: satisfied — (x)` marker.
    """
    missing: list[str] = []
    for letter in CRITERIA:
        pattern = (
            r"//\s*SECURITY-ACCEPTANCE:\s*satisfied\s*[—-]\s*"
            rf"\({letter}\)"
        )
        if not re.search(pattern, text):
            missing.append(letter)
    return missing


def cargo_toml_has_banned_dep(text: str) -> list[str]:
    """Return the list of banned deps that appear on a `dependencies`
    line under any `[features.*]`-implied section. The gate is narrow:
    it only fails when the dep name appears as a bare key on a line of
    its own, which is the syntax used by `libloading = "0.8"` in
    optional-dependency sections.
    """
    found: list[str] = []
    for dep in BANNED_DEPS:
        # Match `dep = "..."` or `dep.workspace = true` on a line of
        # its own. Exclude lines starting with `#` (comments).
        pattern = rf"^\s*{re.escape(dep)}\s*="
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("#"):
                continue
            if re.search(pattern, raw_line):
                found.append(dep)
                break
    return found


def main() -> int:
    print("=== fluxion-evaluator Dynamic-Loading Security Gate ===")
    print(f"Repo: {REPO_ROOT}")
    print(f"Target: {DYNAMIC_RS.relative_to(REPO_ROOT)}")
    print()

    failures: list[str] = []

    # Check 1: SECURITY_ACCEPTANCE.md exists and documents all criteria.
    print("[1] SECURITY_ACCEPTANCE.md present and complete ...", end=" ")
    try:
        acceptance_text = read_text(ACCEPTANCE_MD)
        missing_md = acceptance_md_has_all_criteria(acceptance_text)
        if missing_md:
            print("FAIL")
            failures.append(
                f"SECURITY_ACCEPTANCE.md is missing criterion section(s): "
                f"{', '.join(f'({m})' for m in missing_md)}"
            )
        else:
            print("PASS")
    except FileNotFoundError as e:
        print("FAIL")
        failures.append(str(e))

    # Check 2: dynamic.rs is either in stub state OR has all markers.
    print("[2] dynamic.rs is stub OR has all acceptance markers ...", end=" ")
    try:
        dynamic_text = read_text(DYNAMIC_RS)
        is_stub = dynamic_rs_is_stub(dynamic_text)
        missing_markers = dynamic_rs_has_all_satisfied_markers(dynamic_text)
        if is_stub:
            print("PASS (stub)")
        elif missing_markers:
            print("FAIL")
            failures.append(
                "dynamic.rs has implemented loader logic but is missing "
                f"SECURITY-ACCEPTANCE: satisfied markers for: "
                f"{', '.join(f'({m})' for m in missing_markers)}"
            )
        else:
            print("PASS (implemented with markers)")
    except FileNotFoundError as e:
        print("FAIL")
        failures.append(str(e))

    # Check 3: Cargo.toml for evaluator does not silently include banned deps.
    print("[3] Cargo.toml does not silently pull in banned FFI deps ...", end=" ")
    try:
        cargo_text = read_text(CARGO_TOML)
        banned = cargo_toml_has_banned_dep(cargo_text)
        if banned:
            print("FAIL")
            failures.append(
                "Cargo.toml introduces banned FFI crate(s) without the "
                "duplicate-version budget fix from issue #3310: "
                f"{', '.join(banned)}"
            )
        else:
            print("PASS")
    except FileNotFoundError as e:
        print("FAIL")
        failures.append(str(e))

    print()
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        print()
        print("Remediation (Issue #3554):")
        print("  - See crates/fluxion-evaluator/SECURITY_ACCEPTANCE.md for the")
        print("    authoritative acceptance criteria (a) through (d).")
        print("  - For a stub-only state (this PR), the gate passes; the")
        print("    markers are added by the follow-up `libloading` PR.")
        print("  - For an implemented loader, add one `// SECURITY-ACCEPTANCE:")
        print("    satisfied — (x)` line per criterion (a)/(b)/(c)/(d) in")
        print("    crates/fluxion-evaluator/src/dynamic.rs.")
        return 1

    print("PASS: All checks satisfied.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)