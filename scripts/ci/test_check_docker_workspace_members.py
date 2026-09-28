"""Tests for ``scripts/check_docker_workspace_members.py`` -- Issue #4135.

`cargo build --bin fluxion-rest --no-default-features` resolves the whole
workspace, so the image build context must carry the manifest of every
``[workspace] members`` entry -- including optional ones such as
``fluxion-cfd``. The Dockerfile's member list silently drifted three times
(``fluxion-cfd`` #2469, ``fluxion-tauri/src-tauri`` #3196,
``crates/fluxion-evaluator`` #3337) and the image build never succeeded on
``develop``.

These tests drive the matcher against hermetic ``tmp_path`` fixtures
(``Cargo.toml`` + ``Dockerfile`` pairs), following the
``load_script``/``write_file`` pattern of the other ``scripts/ci`` gate
tests, so the matching logic is covered without needing a real docker
daemon. The end-to-end check against the real tree is the direct
``check_docker_workspace_members.py`` invocation in ``scripts-tests.yml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_NAME = "check_docker_workspace_members"


@pytest.fixture
def gate(load_script):
    """Freshly-loaded copy of ``scripts/check_docker_workspace_members.py``."""
    return load_script(SCRIPT_NAME)


def _cargo_toml(members: list[str]) -> str:
    listed = ", ".join(f'"{m}"' for m in members)
    return f'[package]\nname = "fluxion"\n\n[workspace]\nmembers = [{listed}]\n'


DOCKERFILE_ALL = """FROM rust:1.87-bookworm AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY fluxion-core/ ./fluxion-core/
COPY fluxion-cfd/ ./fluxion-cfd/
COPY crates/fluxion-evaluator/ ./crates/fluxion-evaluator/
COPY fluxion-tauri/src-tauri/ ./fluxion-tauri/src-tauri/
COPY src/ ./src/
"""


# ---------------------------------------------------------------------------
# workspace_members
# ---------------------------------------------------------------------------


def test_members_parsed_from_cargo_toml(gate, tmp_path: Path) -> None:
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(_cargo_toml(["a", "b/c"]), encoding="utf-8")
    assert gate.workspace_members(cargo) == ["a", "b/c"]


def test_trailing_slash_in_member_is_normalised(gate, tmp_path: Path) -> None:
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(_cargo_toml(["fluxion-cfd/"]), encoding="utf-8")
    assert gate.workspace_members(cargo) == ["fluxion-cfd"]


def test_missing_cargo_toml_is_a_script_error(gate, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        gate.workspace_members(tmp_path / "nope.toml")
    assert exc.value.code == 2


def test_missing_members_key_is_a_script_error(gate, tmp_path: Path) -> None:
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text('[package]\nname = "x"\n', encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        gate.workspace_members(cargo)
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# dockerfile_copied_dirs
# ---------------------------------------------------------------------------


def test_copy_destinations_are_collected(gate, tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(DOCKERFILE_ALL, encoding="utf-8")
    copied = gate.dockerfile_copied_dirs(dockerfile)
    assert "fluxion-cfd" in copied
    assert "crates/fluxion-evaluator" in copied
    # `./x/` leading-dot form is normalised away.
    assert "fluxion-core" in copied


def test_commented_copy_lines_are_ignored(gate, tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "# COPY fluxion-cfd/ ./fluxion-cfd/\nCOPY src/ ./src/\n", encoding="utf-8"
    )
    assert "fluxion-cfd" not in gate.dockerfile_copied_dirs(dockerfile)


def test_multiline_and_from_form_copies_are_skipped(gate, tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "COPY a/ b/ c/ /dst/\n"
        "COPY --from=builder /build/target/release/fluxion-rest /usr/local/bin/\n",
        encoding="utf-8",
    )
    assert gate.dockerfile_copied_dirs(dockerfile) == set()


# ---------------------------------------------------------------------------
# check -- the actual drift contract
# ---------------------------------------------------------------------------


def test_fully_covered_context_has_no_findings(gate, tmp_path: Path) -> None:
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(
        _cargo_toml(
            [
                "fluxion-core",
                "fluxion-cfd",
                "crates/fluxion-evaluator",
                "fluxion-tauri/src-tauri",
            ]
        ),
        encoding="utf-8",
    )
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(DOCKERFILE_ALL, encoding="utf-8")
    assert gate.check(cargo, dockerfile) == []


def test_missing_member_is_reported_with_a_copy_hint(gate, tmp_path: Path) -> None:
    """The exact #4135 regression: a member absent from the Dockerfile."""
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(_cargo_toml(["fluxion-core", "fluxion-cfd"]), encoding="utf-8")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "COPY Cargo.toml Cargo.lock ./\nCOPY fluxion-core/ ./fluxion-core/\n",
        encoding="utf-8",
    )
    findings = gate.check(cargo, dockerfile)
    assert len(findings) == 1
    assert "fluxion-cfd" in findings[0]
    # The message must be actionable, not just a flag.
    assert "COPY fluxion-cfd/ ./fluxion-cfd/" in findings[0]


def test_ancestor_directory_copy_satisfies_a_nested_member(
    gate, tmp_path: Path
) -> None:
    """`COPY fluxion-tauri/` covers member `fluxion-tauri/src-tauri`."""
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(_cargo_toml(["fluxion-tauri/src-tauri"]), encoding="utf-8")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("COPY fluxion-tauri/ ./fluxion-tauri/\n", encoding="utf-8")
    assert gate.check(cargo, dockerfile) == []


def test_exact_member_copy_satisfies_a_nested_member(gate, tmp_path: Path) -> None:
    """`COPY fluxion-tauri/src-tauri/` covers the same member exactly."""
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(_cargo_toml(["fluxion-tauri/src-tauri"]), encoding="utf-8")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "COPY fluxion-tauri/src-tauri/ ./fluxion-tauri/src-tauri/\n", encoding="utf-8"
    )
    assert gate.check(cargo, dockerfile) == []


def test_stale_copy_for_a_removed_member_is_reported(gate, tmp_path: Path) -> None:
    """A COPY of a real crate dir that is no longer a member is drift."""
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(_cargo_toml(["fluxion-core"]), encoding="utf-8")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "COPY fluxion-core/ ./fluxion-core/\nCOPY gone-crate/ ./gone-crate/\n",
        encoding="utf-8",
    )
    # `gone-crate` must look like a real crate to trip the stale check.
    (tmp_path / "gone-crate").mkdir()
    (tmp_path / "gone-crate" / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    findings = gate.check(cargo, dockerfile)
    assert any("gone-crate" in f for f in findings)


def test_ordinary_source_copies_are_not_flagged(gate, tmp_path: Path) -> None:
    """`src/`, `benches/`, and single-file COPYs must not be drift."""
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(_cargo_toml(["fluxion-core"]), encoding="utf-8")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "COPY fluxion-core/ ./fluxion-core/\n"
        "COPY src/ ./src/\n"
        "COPY benches/ ./benches/\n"
        "COPY examples/grid_coupling_demo.rs ./examples/grid_coupling_demo.rs\n",
        encoding="utf-8",
    )
    assert gate.check(cargo, dockerfile) == []
