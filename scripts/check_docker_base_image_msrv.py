#!/usr/bin/env python3
"""
CI guard: the `Dockerfile` base-image pins must agree with the workspace
MSRV and with the `docker.yml` env pins (Issue #4150).

The `Dockerfile` has asserted this control since the #4138 MSRV bump:

    #     2026-09-03 MSRV bump, so cargo could not compile the dependency
    #     graph at all (Issue #4138). scripts/check_docker_base_image_msrv.py
    #     now gates the tag against `rust-version`.

**The script named in that comment did not exist**, so the repository was
claiming a control it never enforced. The concrete cost was not
hypothetical: `scripts/pin_docker_base_images.sh` wrote
`FROM rust:1.98.0-bookworm@sha256:sha256:<hex>` (#4149) — a doubled
algorithm prefix and therefore an unparseable image reference — while CI
stayed green, because the only digest assertion in CI
(`.github/workflows/docker.yml`) reads the *env* pins and never compares
them against the `Dockerfile` `FROM` lines. This gate closes that gap.

Invariants enforced against the real tree:

  1. **MSRV parity.** Every `rust:` base image is tagged
     `rust:<rust-version>-bookworm`, where `rust-version` is read from the
     root `Cargo.toml`. A builder image older than the MSRV cannot compile
     the dependency graph, which is exactly the #4138 failure.
  2. **Digest pinning.** Every base `FROM` is pinned to
     `sha256:<64 lowercase hex>` — never a bare mutable tag, and never a
     doubly-prefixed or truncated digest. The documented `#   * Digest:`
     comment must name the same digest, so the audit trail a reviewer
     reads cannot drift from what is actually pulled.
  3. **Dockerfile/docker.yml parity.** Each `FROM` pin equals the
     `RUST_BUILDER_DIGEST` / `DEBIAN_RUNTIME_DIGEST` env entry the CI
     assertion consumes. Without this, the two files can disagree and CI
     validates the wrong one.

Exits 0 when all three hold, 1 on drift, 2 on script error.

Usage:
    python3 scripts/check_docker_base_image_msrv.py

Exit codes:
    0 — MSRV parity, digest pinning, and Dockerfile/docker.yml parity hold.
    1 — drift: see the reported findings.
    2 — script error (e.g. `Cargo.toml` or `Dockerfile` missing).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 — tomllib is stdlib only from 3.11
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
CARGO_TOML = REPO_ROOT / "Cargo.toml"
DOCKERFILE = REPO_ROOT / "Dockerfile"
WORKFLOW = REPO_ROOT / ".github/workflows" / "docker.yml"

# `FROM <image>[@sha256:<hex>] [AS <stage>]`. The stage suffix and the
# optional digest are captured separately so a bare mutable tag is a
# findable state rather than a silent no-match.
FROM_LINE = re.compile(r"^FROM\s+(?P<image>\S+)(?:\s+AS\s+(?P<stage>\S+))?\s*$")

# The documented pin comment, e.g. `#   * Digest: sha256:<hex>`.
DIGEST_COMMENT = re.compile(r"^\s*#\s*\*\s*Digest:\s*(?P<digest>\S+)\s*$")

# Env pins consumed by the CI digest assertion in `docker.yml`.
ENV_PINS = {
    "rust:": "RUST_BUILDER_DIGEST",
    "debian:": "DEBIAN_RUNTIME_DIGEST",
}

VALID_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")


def workspace_msrv(cargo_toml: Path) -> str:
    """Return the workspace `rust-version` as a `major.minor.patch` string."""
    if not cargo_toml.is_file():
        print(f"ERROR: {cargo_toml} not found", file=sys.stderr)
        raise SystemExit(2)
    try:
        data = tomllib.loads(cargo_toml.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        print(f"ERROR: cannot parse {cargo_toml}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    msrv = data.get("package", {}).get("rust-version")
    if not msrv:
        print(f"ERROR: no [package] rust-version in {cargo_toml}", file=sys.stderr)
        raise SystemExit(2)
    return str(msrv)


def base_from_lines(dockerfile: Path) -> list[dict[str, str | int]]:
    """Return the non-stage `FROM` lines as `{lineno, image, digest, stage}`.

    A `FROM` that names a stage declared earlier in the file (e.g.
    `COPY --from=builder`, or `FROM builder AS final`) refers to already
    built layers, not to a registry base image, so it carries no digest
    and is excluded.
    """
    if not dockerfile.is_file():
        print(f"ERROR: {dockerfile} not found", file=sys.stderr)
        raise SystemExit(2)
    lines: list[dict[str, str | int]] = []
    declared_stages: set[str] = set()
    for lineno, raw in enumerate(
        dockerfile.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = FROM_LINE.match(line)
        if match is None:
            continue
        image = match.group("image")
        stage = match.group("stage")
        if stage:
            declared_stages.add(stage)
        # A reference to an already-declared stage is not a base image.
        if image in declared_stages:
            continue
        digest = image.split("@", 1)[1] if "@" in image else ""
        lines.append(
            {
                "lineno": lineno,
                "image": image.split("@", 1)[0],
                "digest": digest,
                "stage": stage or "",
            }
        )
    return lines


def workflow_env_pins(workflow: Path) -> dict[str, str]:
    """Return the digest env pins declared in `docker.yml`."""
    if not workflow.is_file():
        print(f"ERROR: {workflow} not found", file=sys.stderr)
        raise SystemExit(2)
    pins: dict[str, str] = {}
    for raw in workflow.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s+([A-Z_]+_DIGEST):\s*(\S+)\s*$", raw)
        if match:
            pins[match.group(1)] = match.group(2)
    return pins


def check(cargo_toml: Path, dockerfile: Path, workflow: Path) -> list[str]:
    """Return drift findings (empty when every invariant holds)."""
    msrv = workspace_msrv(cargo_toml)
    expected_builder_tag = f"rust:{msrv}-bookworm"
    env_pins = workflow_env_pins(workflow)
    findings: list[str] = []

    bases = base_from_lines(dockerfile)
    # The digests the file documents in its `* Digest:` comments, read once.
    documented = {
        m.group("digest")
        for m in (
            DIGEST_COMMENT.match(raw.strip())
            for raw in dockerfile.read_text(encoding="utf-8").splitlines()
        )
        if m
    }

    for entry in bases:
        lineno = entry["lineno"]
        image = str(entry["image"])
        digest = str(entry["digest"])

        # Invariant 1 — MSRV parity.
        if image.startswith("rust:") and image != expected_builder_tag:
            findings.append(
                f"Dockerfile:{lineno} builds on '{image}' but the workspace "
                f"rust-version is {msrv}, so the expected builder image is "
                f"'{expected_builder_tag}'. An older builder cannot compile "
                f"the dependency graph (Issue #4138). Re-run "
                f"scripts/pin_docker_base_images.sh after bumping rust-version."
            )

        # Invariant 2 — digest pinning, and the audit comment agreeing.
        if not digest:
            findings.append(
                f"Dockerfile:{lineno} pulls '{image}' by mutable tag with no "
                f"@sha256: digest. Pin it (fail-closed supply-chain control, "
                f"Issue #3580) — a tag rotation upstream must not silently "
                f"substitute the base layer."
            )
        elif not VALID_DIGEST.match(digest):
            # Catches both the `sha256:sha256:` corruption (#4149) and any
            # truncated or upper-case digest.
            findings.append(
                f"Dockerfile:{lineno} pins '{image}' with a malformed digest "
                f"'{digest}' (expected sha256: followed by 64 lowercase hex "
                f"chars). A doubled prefix such as 'sha256:sha256:…' is not a "
                f"parseable image reference and `docker build` will reject it."
            )

    # Invariant 2 (continued) — the documented digest must match the pin.
    for entry in bases:
        digest = str(entry["digest"])
        if not digest:
            continue
        if digest not in documented:
            findings.append(
                f"Dockerfile:{entry['lineno']} pins '{entry['image']}' to "
                f"{digest} but no '#   * Digest:' comment documents that value. "
                f"Re-run scripts/pin_docker_base_images.sh so the audit "
                f"trail a reviewer reads matches what is pulled."
            )

    # Invariant 3 — Dockerfile/docker.yml parity.
    for prefix, env_key in ENV_PINS.items():
        env_digest = env_pins.get(env_key)
        matches = [e for e in bases if str(e["image"]).startswith(prefix)]
        if not matches:
            continue
        if env_digest is None:
            findings.append(
                f"{workflow.name} declares no {env_key}, so the CI digest "
                f"assertion cannot validate the '{prefix}' base image. Add the "
                f"env entry next to the matching FROM line in the Dockerfile."
            )
            continue
        for entry in matches:
            if entry["digest"] != env_digest:
                findings.append(
                    f"Dockerfile:{entry['lineno']} pins '{entry['image']}' to "
                    f"'{entry['digest'] or '<unpinned>'}' but {env_key} in "
                    f"{workflow.name} is '{env_digest}'. CI validates the env "
                    f"pin, so a mismatch here means the build pulls a base "
                    f"layer the audit never checked. Re-run "
                    f"scripts/pin_docker_base_images.sh."
                )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check that the Dockerfile base-image pins match the workspace MSRV "
            "and the docker.yml env pins."
        )
    )
    parser.add_argument(
        "--cargo-toml", type=Path, default=CARGO_TOML, help="path to the root Cargo.toml"
    )
    parser.add_argument(
        "--dockerfile", type=Path, default=DOCKERFILE, help="path to the Dockerfile"
    )
    parser.add_argument(
        "--workflow",
        type=Path,
        default=WORKFLOW,
        help="path to .github/workflows/docker.yml",
    )
    args = parser.parse_args()

    findings = check(args.cargo_toml, args.dockerfile, args.workflow)
    if findings:
        print("Docker base-image pin drift detected:", file=sys.stderr)
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        print(
            "\nRun ./scripts/pin_docker_base_images.sh to re-resolve the digests, "
            "then re-check.",
            file=sys.stderr,
        )
        return 1

    count = len(base_from_lines(args.dockerfile))
    print(
        f"OK: {count} base image pin(s) match rust-version "
        f"{workspace_msrv(args.cargo_toml)} and the docker.yml env pins."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
