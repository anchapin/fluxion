"""Tests for ``scripts/check_docker_base_image_msrv.py`` -- Issue #4150.

The `Dockerfile` has cited `scripts/check_docker_base_image_msrv.py`
since the #4138 MSRV bump, but the script did not exist: the repository
asserted a control it never enforced. What that cost was concrete.
`pin_docker_base_images.sh` wrote `FROM rust:1.98.0-bookworm@sha256:sha256:<hex>`
(#4149) -- an unparseable reference -- while CI stayed green, because
the only digest assertion in CI reads the `docker.yml` env pins and never
compares them to the `Dockerfile` `FROM` lines.

These tests drive the three invariants against hermetic ``tmp_path``
fixtures in the ``load_script``/explicit-argument style of
``test_check_docker_workspace_members.py``, so each failure mode is
covered without a docker daemon. The end-to-end run against the real
tree is the direct gate invocation wired into CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_NAME = "check_docker_base_image_msrv"

RUST_HEX = "a" * 64
DEBIAN_HEX = "b" * 64
RUST_DIGEST = f"sha256:{RUST_HEX}"
DEBIAN_DIGEST = f"sha256:{DEBIAN_HEX}"


@pytest.fixture
def gate(load_script):
    return load_script(SCRIPT_NAME)


def _cargo_toml(rust_version: str) -> str:
    return f'[package]\nname = "fluxion"\nrust-version = "{rust_version}"\n'


def _dockerfile(
    rust_digest: str | None = RUST_DIGEST,
    debian_digest: str | None = DEBIAN_DIGEST,
    rust_tag: str = "rust:1.98.0-bookworm",
    document: bool = True,
) -> str:
    """A Dockerfile with the two pin comment blocks the real one carries."""
    parts = (
        [f"FROM debian:bookworm-slim@{debian_digest} AS runtime"]
        if debian_digest
        else ["FROM debian:bookworm-slim AS runtime"]
    )
    out = []
    if document:
        out.append(f"#   * Tag:    {rust_tag}")
        if rust_digest:
            out.append(f"#   * Digest: {rust_digest}")
        out.append("#   * Pinned: 2026-09-28")
    out.append(
        f"FROM {rust_tag}@{rust_digest} AS builder" if rust_digest else f"FROM {rust_tag} AS builder"
    )
    out.append("RUN cargo build --release")
    if document:
        out.append("#   * Tag:    debian:bookworm-slim")
        if debian_digest:
            out.append(f"#   * Digest: {debian_digest}")
        out.append("#   * Pinned: 2026-09-28")
    out.extend(parts)
    return "\n".join(out) + "\n"


def _workflow(rust_digest: str = RUST_DIGEST, debian_digest: str = DEBIAN_DIGEST) -> str:
    return f"""env:
  RUST_BUILDER_IMAGE: rust:1.98.0-bookworm
  RUST_BUILDER_DIGEST: {rust_digest}
  DEBIAN_RUNTIME_IMAGE: debian:bookworm-slim
  DEBIAN_RUNTIME_DIGEST: {debian_digest}
"""


@pytest.fixture
def tree(tmp_path: Path):
    """Return a factory building a (cargo, dockerfile, workflow) triple."""

    def _build(
        msrv: str = "1.98.0",
        dockerfile: str | None = None,
        workflow: str | None = None,
    ):
        cargo = tmp_path / "Cargo.toml"
        dockerfile_path = tmp_path / "Dockerfile"
        workflow_path = tmp_path / "docker.yml"
        cargo.write_text(_cargo_toml(msrv), encoding="utf-8")
        dockerfile_path.write_text(
            dockerfile if dockerfile is not None else _dockerfile(), encoding="utf-8"
        )
        workflow_path.write_text(
            workflow if workflow is not None else _workflow(), encoding="utf-8"
        )
        return cargo, dockerfile_path, workflow_path

    return _build


# ---------------------------------------------------------------------------
# workspace_msrv
# ---------------------------------------------------------------------------


def test_workspace_msrv_parsed(gate, tree) -> None:
    cargo, _, _ = tree(msrv="1.98.0")
    assert gate.workspace_msrv(cargo) == "1.98.0"


def test_workspace_msrv_missing_raises_exit_2(gate, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        gate.workspace_msrv(tmp_path / "absent" / "Cargo.toml")
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# base_from_lines
# ---------------------------------------------------------------------------


def test_base_from_lines_captures_digest_and_stage(gate, tree) -> None:
    _, dockerfile, _ = tree()
    entries = gate.base_from_lines(dockerfile)
    assert [e["image"] for e in entries] == ["rust:1.98.0-bookworm", "debian:bookworm-slim"]
    assert [e["digest"] for e in entries] == [RUST_DIGEST, DEBIAN_DIGEST]
    assert [e["stage"] for e in entries] == ["builder", "runtime"]


def test_base_from_lines_skips_stage_references(gate, tmp_path: Path) -> None:
    """`FROM builder AS final` names built layers, not a registry image."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        f"FROM {RUST_DIGEST and 'rust:1.98.0-bookworm@' + RUST_DIGEST} AS builder\n"
        "FROM builder AS final\n",
        encoding="utf-8",
    )
    entries = gate.base_from_lines(dockerfile)
    assert [e["image"] for e in entries] == ["rust:1.98.0-bookworm"]


def test_base_from_lines_ignores_comments(gate, tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        f"# FROM rust:1.90.0 AS stale\nFROM rust:1.98.0-bookworm@{RUST_DIGEST} AS builder\n",
        encoding="utf-8",
    )
    entries = gate.base_from_lines(dockerfile)
    assert len(entries) == 1
    assert entries[0]["image"] == "rust:1.98.0-bookworm"


# ---------------------------------------------------------------------------
# Invariant 1 — MSRV parity
# ---------------------------------------------------------------------------


def test_clean_tree_has_no_findings(gate, tree) -> None:
    cargo, dockerfile, workflow = tree()
    assert gate.check(cargo, dockerfile, workflow) == []


def test_stale_builder_tag_is_reported(gate, tree) -> None:
    """A builder older than the MSRV cannot compile the graph (#4138)."""
    cargo, dockerfile, workflow = tree(
        msrv="1.98.0", dockerfile=_dockerfile(rust_tag="rust:1.89.0-bookworm")
    )
    findings = gate.check(cargo, dockerfile, workflow)
    assert any("rust-version is 1.98.0" in f for f in findings)


def test_builder_tag_matching_msrv_is_accepted(gate, tree) -> None:
    cargo, dockerfile, workflow = tree(msrv="1.99.1", dockerfile=_dockerfile(rust_tag="rust:1.99.1-bookworm"))
    assert gate.check(cargo, dockerfile, workflow) == []


def test_non_rust_images_are_exempt_from_msrv_parity(gate, tree) -> None:
    """The debian runtime base is not gated on the Rust MSRV."""
    cargo, dockerfile, workflow = tree()
    findings = gate.check(cargo, dockerfile, workflow)
    assert not any("debian" in f and "rust-version" in f for f in findings)


# ---------------------------------------------------------------------------
# Invariant 2 — digest pinning
# ---------------------------------------------------------------------------


def test_unpinned_from_line_is_reported(gate, tree) -> None:
    cargo, dockerfile, workflow = tree(dockerfile=_dockerfile(rust_digest=None))
    findings = gate.check(cargo, dockerfile, workflow)
    assert any("mutable tag" in f for f in findings)


def test_doubled_sha256_prefix_is_reported(gate, tree) -> None:
    """The #4149 corruption must be caught structurally, not by CI luck."""
    cargo, dockerfile, workflow = tree(
        dockerfile=_dockerfile(rust_digest=f"sha256:sha256:{RUST_HEX}")
    )
    findings = gate.check(cargo, dockerfile, workflow)
    assert any("malformed digest" in f and "sha256:sha256" in f for f in findings)


def test_truncated_digest_is_reported(gate, tree) -> None:
    cargo, dockerfile, workflow = tree(dockerfile=_dockerfile(rust_digest="sha256:abc123"))
    findings = gate.check(cargo, dockerfile, workflow)
    assert any("malformed digest" in f for f in findings)


def test_uppercase_digest_is_reported(gate, tree) -> None:
    cargo, dockerfile, workflow = tree(
        dockerfile=_dockerfile(rust_digest=f"sha256:{RUST_HEX.upper()}")
    )
    findings = gate.check(cargo, dockerfile, workflow)
    assert any("malformed digest" in f for f in findings)


def test_undocumented_digest_is_reported(gate, tree) -> None:
    """The audit comment must not drift from what is actually pulled."""
    cargo, dockerfile, workflow = tree(dockerfile=_dockerfile(document=False))
    findings = gate.check(cargo, dockerfile, workflow)
    assert any("documents that value" in f for f in findings)


# ---------------------------------------------------------------------------
# Invariant 3 — Dockerfile / docker.yml parity
# ---------------------------------------------------------------------------


def test_dockerfile_workflow_digest_mismatch_is_reported(gate, tree) -> None:
    """The gap that let #4149 hide: CI only validates the env pin."""
    cargo, dockerfile, workflow = tree(
        dockerfile=_dockerfile(debian_digest=DEBIAN_DIGEST),
        workflow=_workflow(debian_digest=f"sha256:{'c' * 64}"),
    )
    findings = gate.check(cargo, dockerfile, workflow)
    assert any("DEBIAN_RUNTIME_DIGEST" in f for f in findings)


def test_matching_parity_produces_no_finding(gate, tree) -> None:
    cargo, dockerfile, workflow = tree()
    assert not [f for f in gate.check(cargo, dockerfile, workflow) if "DEBIAN" in f]


def test_missing_env_pin_is_reported(gate, tree) -> None:
    cargo, dockerfile, workflow = tree(workflow="env:\n  RUST_BUILDER_IMAGE: rust:1.98.0-bookworm\n")
    findings = gate.check(cargo, dockerfile, workflow)
    assert any("declares no" in f for f in findings)


# ---------------------------------------------------------------------------
# Real tree
# ---------------------------------------------------------------------------


def test_real_tree_is_clean(gate, repo_root: Path) -> None:
    """Pinned against the actual checkout.

    A regression in the scanner would flip the real repo from clean to
    failing, which is the whole point of a supply-chain gate.
    """
    findings = gate.check(repo_root / "Cargo.toml", repo_root / "Dockerfile", repo_root / ".github" / "workflows" / "docker.yml")
    assert findings == [], "\n".join(findings)
