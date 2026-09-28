"""Regression tests for the docker.yml base-image digest-pin gate.

Issue #3580 pinned the `fluxion-rest` build/runtime base images by
``@sha256:`` digest; Issue #3815 then added a defence-in-depth
``docker inspect`` cross-check. That cross-check was **structurally
unsatisfiable**: it compared ``.RepoDigests[0]`` against the full
``<image:tag>@<digest>`` pin, but the daemon normalises ``RepoDigests``
to ``<repo-without-tag>@<digest>``. For ``rust:1.87-bookworm@sha256:…``
it returns ``rust@sha256:…``, so the string compare could never match a
tagged image.

The gate therefore false-failed **every** push to ``main``/``develop``
(10+ consecutive ``develop`` runs) while the digests matched
byte-for-byte -- a fail-closed control that was failing open on signal
and blocking the whole Docker build, ``security`` (Trivy) job, and
multi-arch publish.

These tests extract the *real* ``run:`` block from
``.github/workflows/docker.yml`` and execute it under bash against a
stubbed ``docker`` that reproduces the daemon's tag-stripping
normalisation. Running the committed bytes -- rather than a copy of the
logic -- means a re-introduced full-string comparison fails here
instead of on a real PR.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

STEP_NAME_PREFIX = "Verify base image digest pins"

RUST_IMAGE = "rust:1.87-bookworm"
RUST_DIGEST = "sha256:251cec8da4689d180f124ef00024c2f83f79d9bf984e43c180a598119e326b84"
DEBIAN_IMAGE = "debian:bookworm-slim"
DEBIAN_DIGEST = "sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171"

# What the daemon actually reports in `.RepoDigests[0]` after a
# pull-by-digest: the tag is stripped from the repository.
RUST_REPO_DIGEST = f"rust@{RUST_DIGEST}"
DEBIAN_REPO_DIGEST = f"debian@{DEBIAN_DIGEST}"

ENV_SUBS = {
    "RUST_BUILDER_IMAGE": RUST_IMAGE,
    "RUST_BUILDER_DIGEST": RUST_DIGEST,
    "DEBIAN_RUNTIME_IMAGE": DEBIAN_IMAGE,
    "DEBIAN_RUNTIME_DIGEST": DEBIAN_DIGEST,
}

STUB = """#!/usr/bin/env bash
# Stub `docker`. Emulates a successful pull-by-digest, then reports
# `.RepoDigests[0]` the way the real daemon does: tag stripped.
# The parameter expansions below use the un-defaulted form (no colon) so
# an explicitly-empty override stays empty and models `docker inspect`
# returning nothing.
if [[ "$1" == "pull" ]]; then exit 0; fi
if [[ "$1" == "inspect" ]]; then
  ref="${*: -1}"
  img="${ref%@*}"
  case "$img" in
    @RUST_IMAGE@)   printf '%s\\n' "${FLUXION_STUB_RUST-@RUST_REPO@}" ;;
    @DEBIAN_IMAGE@) printf '%s\\n' "${FLUXION_STUB_DEBIAN-@DEBIAN_REPO@}" ;;
  esac
  exit 0
fi
exit 0
"""


def _stub_text() -> str:
    text = STUB
    for token, value in (
        ("@RUST_IMAGE@", RUST_IMAGE),
        ("@RUST_REPO@", RUST_REPO_DIGEST),
        ("@DEBIAN_IMAGE@", DEBIAN_IMAGE),
        ("@DEBIAN_REPO@", DEBIAN_REPO_DIGEST),
    ):
        text = text.replace(token, value)
    return text


def _gate_steps(repo_root: Path) -> dict[str, str]:
    """Map job name -> the ``run:`` script of the digest-pin gate step."""
    wf = yaml.safe_load((repo_root / ".github/workflows/docker.yml").read_text())
    steps = {}
    for job_name, job in wf["jobs"].items():
        for step in job.get("steps", []):
            # NB: the declared name is `Verify base image digest pins
            # (Issue #3580)`, but an unquoted `#` starts a YAML comment,
            # so it parses as `Verify base image digest pins (Issue`.
            # Match on prefix.
            if str(step.get("name", "")).startswith(STEP_NAME_PREFIX):
                assert "run" in step, f"{job_name}: gate step has no run: block"
                steps[job_name] = step["run"]
    return steps


def _executable_body(script: str) -> str:
    """The script with comment lines removed, for cross-job comparison."""
    return "\n".join(l for l in script.splitlines() if not l.strip().startswith("#"))


def _write_stub(bin_dir: Path) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "docker"
    stub.write_text(_stub_text())
    stub.chmod(0o755)
    return stub


def _run_gate(script: str, bin_dir: Path, tmp_path: Path, **stub_env: str) -> int:
    """Execute the real gate script with the stubbed `docker` on PATH."""
    for key, value in ENV_SUBS.items():
        script = script.replace("${{ env.%s }}" % key, value)
    assert "${{" not in script, "unsubstituted ${{ ... }} expression remains"
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", **stub_env}
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env
    ).returncode


@pytest.fixture(scope="module")
def gate_script(repo_root_module: Path) -> str:
    """The gate ``run:`` block from the first job that carries it."""
    steps = _gate_steps(repo_root_module)
    assert steps, "no digest-pin gate step found in docker.yml"
    return steps[next(iter(steps))]


@pytest.fixture(scope="module")
def stub_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    bin_dir = tmp_path_factory.mktemp("stub_bin")
    _write_stub(bin_dir)
    return bin_dir


@pytest.fixture(scope="module")
def repo_root_module() -> Path:
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Drift guard: the gate exists in both build jobs with one shared body
# ---------------------------------------------------------------------------


def test_gate_present_in_both_build_jobs(repo_root_module: Path) -> None:
    steps = _gate_steps(repo_root_module)
    assert set(steps) == {"build-and-test", "build-platform"}, (
        f"digest-pin gate missing from a build job; found in {sorted(steps)}"
    )


def test_gate_bodies_identical_across_jobs(repo_root_module: Path) -> None:
    """A fix applied to one job must land in the other."""
    steps = _gate_steps(repo_root_module)
    bodies = {j: _executable_body(s) for j, s in steps.items()}
    assert len(set(bodies.values())) == 1, (
        "the two digest-pin gate steps have drifted apart; they must share "
        "one executable body"
    )


# ---------------------------------------------------------------------------
# Behaviour: the gate accepts the real (tag-stripped) daemon output …
# ---------------------------------------------------------------------------


def test_accepts_real_tag_stripped_repodigest(
    gate_script: str, stub_bin: Path, tmp_path: Path
) -> None:
    """The exact output that false-failed every push must now pass."""
    assert _run_gate(gate_script, stub_bin, tmp_path) == 0


def test_accepts_debian_tag_stripped_repodigest(
    gate_script: str, stub_bin: Path, tmp_path: Path
) -> None:
    assert _run_gate(gate_script, stub_bin, tmp_path, FLUXION_STUB_DEBIAN=DEBIAN_REPO_DIGEST) == 0


# ---------------------------------------------------------------------------
# … and stays fail-closed on the three real attack / drift modes
# ---------------------------------------------------------------------------


def test_rejects_genuine_digest_rotation(
    gate_script: str, stub_bin: Path, tmp_path: Path
) -> None:
    """Upstream tag rebase -> manifest digest changes -> must fail."""
    assert _run_gate(gate_script, stub_bin, tmp_path, FLUXION_STUB_RUST=f"rust@sha256:{'0' * 64}") == 1


def test_rejects_substituted_repository(
    gate_script: str, stub_bin: Path, tmp_path: Path
) -> None:
    """Mirror/attacker serves a different repository at the same digest."""
    evil = f"evilmirror.example.com/rust@{RUST_DIGEST}"
    assert _run_gate(gate_script, stub_bin, tmp_path, FLUXION_STUB_RUST=evil) == 1


def test_rejects_unpopulated_repodigest(
    gate_script: str, stub_bin: Path, tmp_path: Path
) -> None:
    """The original #3815 symptom: inspect yields nothing."""
    assert _run_gate(gate_script, stub_bin, tmp_path, FLUXION_STUB_RUST="") == 1
