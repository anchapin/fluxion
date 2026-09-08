#!/usr/bin/env python3
"""
fluxion-core Dependency-Budget Regression Gate (Issue #3467).

Verifies that the `fluxion-core` leaf crate's **default** dependency
manifest matches the documented "dependency-light" contract:

> "The dependency-light *leaf* (`weather/`, `assembly/`, `construction/`,
>  `multi_node/`, `per_surface_conduction/`, `physics_constants/`,
>  `ashrae_cases/`, `parser_limits/`, `earth_tube/`, `tensor/`,
>  `urban_radiation/`) must NOT compile reqwest / hyper / tokio / rustls
>  at default features. Only consumers that need the TMY3 network-
>  download / on-disk-cache path opt into the `tmy3-download` feature."
>
> — ARCHITECTURE.md §"Workspace Layout (#3467)",
>   AGENTS.md §"Read Before Changing Boundaries"

Two enforcement passes:

1. **Static manifest scan** — parses `fluxion-core/Cargo.toml` and fails
   when any *non-optional*, non-`dep:`-gated dependency in
   `[dependencies]` is on the heavyweight allow-list below. Catches the
   common regression "someone added `reqwest = ...` to the default
   block without gating it behind `tmy3-download`".

2. **Dynamic tree scan** — runs `cargo tree -p fluxion-core
   --no-default-features` and fails if any of the heavyweight crates
   (or their transitive dependencies in the same family) appear in the
   resulting tree. Catches the subtler "the crate entry is feature-
   gated but its workspace dependency declaration sneaks it in via
   `[workspace.dependencies]`" case (see #3467 §"Source" for the
   pre-fix tree).

The script also verifies the inverse: with `--features tmy3-download`,
the heavyweight deps MUST appear, so a future regression that
inadvertently breaks the feature gating (e.g. removes
`dep:reqwest` from the feature list) is caught.

Usage:
  python3 scripts/check_fluxion_core_dep_budget.py

Exit codes:
  0 — Default-feature dep tree is clean; opt-in tree has the heavy deps.
  1 — Regression: default features pull in a heavyweight dep, OR the
      opt-in feature does not pull it in (gating broken in either
      direction).
  2 — Script error (manifest missing, cargo unavailable, etc.).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FLUXION_CORE_MANIFEST = REPO_ROOT / "fluxion-core" / "Cargo.toml"

# Crates that MUST NOT appear in `cargo tree -p fluxion-core` at
# default features (Issue #3467). The list mirrors what `cargo tree
# --depth 2` showed before the fix — reqwest pulls in hyper / tokio /
# rustls / tower etc., and that whole family is what the gate forbids.
# Adding a new heavyweight crate to this set requires updating
# ARCHITECTURE.md §"Workspace Layout (#3467)" so the documented budget
# stays aligned with the enforced budget.
HEAVYWEIGHT_CRATES = frozenset(
    {
        # HTTP client + its TLS stack (the #3467 headline offender).
        "reqwest",
        "hyper",
        "hyper-rustls",
        "hyper-util",
        "h2",
        # Async runtime + TLS that reqwest/rustls bring in.
        "tokio",
        "tokio-rustls",
        "rustls",
        "rustls-pki-types",
        "rustls-webpki",
        "webpki-roots",
        # Project-directory / cache-locator helper (only meaningful for
        # the TMY3 cache path).
        "directories",
        "dirs",
        "dirs-sys",
        # Mock HTTP server used by the TMY3 download tests.
        "mockito",
        "httpmock",
        "wiremock",
    }
)

# Crates that MUST appear in `cargo tree -p fluxion-core
# --features tmy3-download`. This is the positive check: if a future
# change accidentally breaks the feature gating, this list detects it.
FEATURED_CRATES = frozenset({"reqwest", "directories", "sha2"})


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------

_TOML_TARGET_RE = re.compile(r"^\[([^\]]+)\]\s*$")
_TOML_KV_RE = re.compile(r"^([A-Za-z0-9_.\-]+)\s*=\s*(.+?)\s*$", re.DOTALL)


def parse_manifest(path: Path) -> dict[str, list[dict[str, str]]]:
    """Tiny TOML parser scoped to the dependencies tables we need.

    Cargo's manifest has a few shapes we must accept for a dep entry:

      foo = "1.0"
      foo = { version = "1.0", optional = true }
      foo.workspace = true
      foo = { workspace = true, optional = true }

    Rather than depend on `toml` (which is not stdlib), we parse the
    sections we care about and extract: (crate_name, dep_string,
    is_optional). The `is_optional` flag is what gates the default
    build.
    """
    sections: dict[str, list[dict[str, str]]] = {
        "dependencies": [],
        "dev-dependencies": [],
        "build-dependencies": [],
    }

    current_section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        section_match = _TOML_TARGET_RE.match(line)
        if section_match:
            name = section_match.group(1).strip()
            # Treat `[dependencies]`, `[dev-dependencies]`,
            # `[target.'cfg(...)'.dependencies]` all as the parent name.
            base = name.split(".", 1)[0].strip(" '\"")
            current_section = base if base in sections else None
            continue

        if current_section is None:
            continue

        kv_match = _TOML_KV_RE.match(line)
        if not kv_match:
            continue
        key = kv_match.group(1).strip()
        value = kv_match.group(2).strip()

        # Workspace-as-value shorthand: `foo.workspace = true`.
        if "." in key:
            continue
        # Multi-line continuation: skip — the file we read does not use it.
        if value.startswith("{") and not value.endswith("}"):
            continue

        is_optional = bool(
            re.search(r"\boptional\s*=\s*true\b", value)
        )
        sections[current_section].append(
            {"name": key, "value": value, "optional": is_optional}
        )

    return sections


def static_manifest_check(
    sections: dict[str, list[dict[str, str]]],
) -> tuple[list[str], list[str]]:
    """Pass 1: every non-optional [dependencies] entry must NOT be on
    the heavyweight list. [dev-dependencies] hits are reported as
    *advisory* warnings only — cargo has no per-feature dev-dep
    gating mechanism, so the prod build (`cargo build -p
    fluxion-core`) is unaffected by a heavy dev-dep, but every
    `cargo test -p fluxion-core` rebuild pulls it in.
    Returns (failures, warnings).
    """
    failures: list[str] = []
    warnings: list[str] = []
    for dep in sections["dependencies"]:
        if dep["optional"]:
            continue
        if dep["name"] in HEAVYWEIGHT_CRATES:
            failures.append(
                f"fluxion-core/Cargo.toml [dependencies]: `{dep['name']}` is "
                f"non-optional but is on the #3467 heavyweight allow-list. "
                f"Gate it behind an explicit cargo feature "
                f"(e.g. `tmy3-download`) and add `dep:{dep['name']}` to the "
                f"feature's dep list."
            )
    for dep in sections["dev-dependencies"]:
        if dep["optional"]:
            continue
        if dep["name"] in HEAVYWEIGHT_CRATES:
            warnings.append(
                f"fluxion-core/Cargo.toml [dev-dependencies]: `{dep['name']}` "
                f"is a heavyweight dep. Cargo has no per-feature dev-dep "
                f"gating (see issue #3467 §Caveats), so every "
                f"`cargo test -p fluxion-core` recompiles it. Acceptable "
                f"because the production build is unaffected, but worth "
                f"keeping an eye on."
            )
    return failures, warnings


# ---------------------------------------------------------------------------
# cargo metadata invocation
# ---------------------------------------------------------------------------


def cargo_tree_packages(
    manifest_path: Path, *, features: list[str] | None = None
) -> set[str]:
    """Return the set of crate names that appear in `cargo tree -p
    fluxion-core` for the given feature set.

    Uses `cargo tree` because it is the most stable surface for "what
    would actually be compiled" — `cargo metadata` reports resolved
    workspace-wide deps, which is too noisy for a per-package gate.
    Excludes dev-dependency edges (`--edges no-dev`) so the gate only
    triggers on the production build, matching the documented
    "dependency-light leaf" contract.
    """
    cmd = [
        "cargo",
        "tree",
        "-p",
        "fluxion-core",
        "--depth",
        "255",  # full transitive closure
        "--prefix",
        "none",
        "--edges",
        "no-dev",
        "--format",
        "{p}",
    ]
    if features is None:
        cmd.extend(["--no-default-features"])
    else:
        if features:
            cmd.extend(["--features", ",".join(features)])
    cmd.extend([f"--manifest-path={manifest_path}"])

    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        print(
            f"ERROR: cargo tree failed (exit {proc.returncode}):\n"
            f"  stdout: {proc.stdout[:2000]}\n"
            f"  stderr: {proc.stderr[:2000]}",
            file=sys.stderr,
        )
        return set()

    # Each line of `cargo tree --prefix none --format {p}` is the
    # package name (potentially with version, like `reqwest v0.12.28`).
    # We extract the leading crate name.
    names: set[str] = set()
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        name = line.split()[0]
        names.add(name)
    return names


def dynamic_tree_check(
    manifest_path: Path,
) -> tuple[list[str], list[str]]:
    """Pass 2: `cargo tree -p fluxion-core --no-default-features` must
    not contain any heavyweight crate. Also, `cargo tree -p fluxion-
    core --features tmy3-download` MUST contain `reqwest`,
    `directories`, `sha2` so the feature gating is not silently broken.
    """
    default_tree = cargo_tree_packages(manifest_path, features=None)
    if not default_tree:
        return (["cargo tree failed for default features"], [])

    featured_tree = cargo_tree_packages(manifest_path, features=["tmy3-download"])
    if not featured_tree:
        return ([], ["cargo tree failed for --features tmy3-download"])

    default_failures: list[str] = []
    for crate in sorted(HEAVYWEIGHT_CRATES):
        if crate in default_tree:
            default_failures.append(
                f"fluxion-core default-feature tree contains `{crate}` "
                f"(expected: only with `--features tmy3-download`)"
            )

    featured_failures: list[str] = []
    for crate in sorted(FEATURED_CRATES):
        if crate not in featured_tree:
            featured_failures.append(
                f"fluxion-core --features tmy3-download tree is missing "
                f"`{crate}` (expected: present — feature gating broken)"
            )

    return default_failures, featured_failures


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    print(
        "Checking fluxion-core dependency budget (Issue #3467) "
        f"(repo: {REPO_ROOT})"
    )
    print()

    if not FLUXION_CORE_MANIFEST.exists():
        print(f"ERROR: {FLUXION_CORE_MANIFEST} not found", file=sys.stderr)
        return 2

    # --- Pass 1: static manifest scan -------------------------------------
    print("[1/3] Static manifest scan (fluxion-core/Cargo.toml) ...")
    sections = parse_manifest(FLUXION_CORE_MANIFEST)
    static_failures, static_warnings = static_manifest_check(sections)
    if static_failures:
        for f in static_failures:
            print(f"    FAIL: {f}")
    elif static_warnings:
        for w in static_warnings:
            print(f"    WARN: {w}")
    else:
        prod_deps = [
            d["name"] for d in sections["dependencies"] if not d["optional"]
        ]
        dev_deps = [
            d["name"] for d in sections["dev-dependencies"] if not d["optional"]
        ]
        print(
            f"    OK: default [dependencies] = "
            f"{sorted(prod_deps)}; [dev-dependencies] = {sorted(dev_deps)}"
        )

    # --- Pass 2: dynamic tree scan (default features) ---------------------
    print("[2/3] Dynamic tree scan (cargo tree, --no-default-features) ...")
    default_failures, featured_failures = dynamic_tree_check(FLUXION_CORE_MANIFEST)
    if default_failures:
        for f in default_failures:
            print(f"    FAIL: {f}")
    else:
        print("    OK: no heavyweight crate in default-feature dep tree")

    # --- Pass 3: dynamic tree scan (opt-in feature) -----------------------
    print("[3/3] Dynamic tree scan (cargo tree, --features tmy3-download) ...")
    if featured_failures:
        for f in featured_failures:
            print(f"    FAIL: {f}")
    else:
        print(
            "    OK: tmy3-download feature pulls in "
            f"{sorted(c for c in FEATURED_CRATES)}"
        )

    all_failures = static_failures + default_failures + featured_failures
    print()
    if all_failures:
        print("DEPENDENCY-BUDGET REGRESSION DETECTED:")
        for f in all_failures:
            print(f"  {f}")
        print()
        print(
            "Issue #3467 forbids heavyweight crates (reqwest / hyper / tokio / "
            "rustls / directories / mockito / httpmock / wiremock) in the "
            "default fluxion-core build. Add a cargo feature, gate the new "
            "dep with `dep:<name>`, and update ARCHITECTURE.md "
            "§\"Workspace Layout (#3467)\"."
        )
        return 1

    print(
        "fluxion-core dependency budget is intact: default features stay "
        "dependency-light, and `tmy3-download` correctly pulls in the "
        "TMY3 network/cache deps."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
