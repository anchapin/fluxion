#!/usr/bin/env python3
"""
CI guard: every `[workspace] members` entry in the root `Cargo.toml`
must be present in the `Dockerfile` build context (Issue #4135).

`cargo build --bin fluxion-rest --no-default-features` resolves the
**whole workspace**, not just the requested package, so the build fails
if any member's manifest is missing from the image context — including an
*optional* member such as `fluxion-cfd`. `Dockerfile` therefore has to
mirror the member list exactly, and it silently drifted three times
(`fluxion-cfd` 2026-08-08 / #2469, `fluxion-tauri/src-tauri` 2026-08-25 /
#3196, `crates/fluxion-evaluator` 2026-09-04 / #3337). The resulting
image build had never succeeded on `develop`; it went unnoticed because
the digest-pin gate failed first and aborted the job (#4132).

Invariants enforced against the real `Dockerfile`:

  1. Every member has a `COPY` that places its manifest at the same
     relative path inside the image.
  2. Every `COPY` that targets a workspace member corresponds to a real
     member (a stale `COPY` for a removed member is dead context weight
     and usually signals a rename that was not propagated).

Exits 0 when the two lists agree, 1 on drift, 2 on script error.

Usage:
    python3 scripts/check_docker_workspace_members.py

Exit codes:
    0 — Dockerfile context covers every workspace member.
    1 — drift: a member is missing, or a COPY targets a non-member.
    2 — script error (e.g. `Cargo.toml` or `Dockerfile` missing).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
CARGO_TOML = REPO_ROOT / "Cargo.toml"
DOCKERFILE = REPO_ROOT / "Dockerfile"

# `COPY <src> <dest>` on a single line, the form this Dockerfile uses for
# the member manifests. Deliberately narrow: it excludes the multi-source
# form (`COPY a b c /dst/`), the `--from=` buildx form (runtime stage), and
# `.` as a catch-all whole-context copy, none of which this gate needs to
# reason about.
COPY_LINE = re.compile(r"^COPY\s+(\S+)\s+(\S+)\s*$")

# Source directory whose Cargo.toml is the member manifest, relative to the
# copied directory. e.g. member `fluxion-tauri/src-tauri` is satisfied by
# `COPY fluxion-tauri/src-tauri/ ./fluxion-tauri/src-tauri/`.
MEMBER_MANIFEST = "Cargo.toml"


def workspace_members(cargo_toml: Path) -> list[str]:
    """Return the `[workspace] members` list, normalised."""
    if not cargo_toml.is_file():
        print(f"ERROR: {cargo_toml} not found", file=sys.stderr)
        raise SystemExit(2)
    try:
        data = tomllib.loads(cargo_toml.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        print(f"ERROR: cannot parse {cargo_toml}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    members = data.get("workspace", {}).get("members")
    if not members:
        print(f"ERROR: no [workspace] members in {cargo_toml}", file=sys.stderr)
        raise SystemExit(2)
    return [str(m).rstrip("/") for m in members]


def dockerfile_copied_dirs(dockerfile: Path) -> set[str]:
    """Return the destination directories that a `COPY` populates."""
    if not dockerfile.is_file():
        print(f"ERROR: {dockerfile} not found", file=sys.stderr)
        raise SystemExit(2)
    copied: set[str] = set()
    for raw in dockerfile.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        match = COPY_LINE.match(line)
        if not match:
            continue
        dest = match.group(2)
        if dest.startswith("--from="):
            continue
        copied.add(dest.lstrip("./").rstrip("/"))
    return copied


def check(cargo_toml: Path, dockerfile: Path) -> list[str]:
    """Return a list of drift findings (empty when the lists agree)."""
    members = workspace_members(cargo_toml)
    copied = dockerfile_copied_dirs(dockerfile)
    findings: list[str] = []

    for member in members:
        # A member is satisfied when the context carries its directory
        # (or an ancestor of it that still contains the manifest).
        satisfied = any(
            member == c or member.startswith(f"{c}/") or c.startswith(f"{member}/")
            for c in copied
        )
        if not satisfied:
            findings.append(
                f"workspace member '{member}' is not COPYed into the Dockerfile "
                f"context — cargo resolves the whole workspace, so "
                f"`cargo build --bin fluxion-rest --no-default-features` fails "
                f"with 'failed to read /build/{member}/Cargo.toml'. Add: "
                f"COPY {member}/ ./{member}/"
            )

    member_set = set(members)
    for dest in sorted(copied):
        # Only consider destinations that look like workspace crates
        # (i.e. carry or are carried by a member path), so ordinary
        # `src/` / `benches/` copies are not flagged.
        if "src" in dest and dest not in member_set:
            continue
        if any(m.startswith(f"{dest}/") for m in members):
            continue
        if dest in {"", ".", "bin", "lib"}:
            continue
        if not (dockerfile.parent / dest / MEMBER_MANIFEST).is_file():
            continue
        if dest not in member_set:
            findings.append(
                f"Dockerfile COPYs '{dest}' which is not a [workspace] member "
                f"— stale context weight; if the member was renamed, update "
                f"both Cargo.toml and Dockerfile"
            )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that the Dockerfile context covers every workspace member."
    )
    parser.add_argument(
        "--cargo-toml",
        type=Path,
        default=CARGO_TOML,
        help="path to the root Cargo.toml (default: repo root)",
    )
    parser.add_argument(
        "--dockerfile",
        type=Path,
        default=DOCKERFILE,
        help="path to the Dockerfile (default: repo root)",
    )
    args = parser.parse_args()

    findings = check(args.cargo_toml, args.dockerfile)
    if findings:
        print("Docker/workspace member drift detected:", file=sys.stderr)
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        print(
            "\nAdd the missing COPY line(s) to the Dockerfile, then re-run this "
            "check. Members are parsed from [workspace] members in Cargo.toml.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: Dockerfile context covers all {len(workspace_members(args.cargo_toml))} "
        "workspace member(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
