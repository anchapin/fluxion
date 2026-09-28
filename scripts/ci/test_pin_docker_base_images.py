"""Tests for ``scripts/pin_docker_base_images.sh`` -- Issues #4149 / #4150.

The pin script is the *only* thing that writes the fail-closed
supply-chain digests asserted by `.github/workflows/docker.yml`
(Issue #3580). Before #4149 it had two latent defects that no test
covered, and the first one silently corrupted the files it was
supposed to protect:

1. **Doubled algorithm prefix.** ``NEW_DIGESTS`` holds the
   ``sha256:<hex>`` form, but every write site re-prepended
   ``sha256:``, producing
   ``FROM rust:1.98.0-bookworm@sha256:sha256:<hex>``. That is not a
   parseable image reference -- ``docker build`` rejects it -- yet the
   CI assertion reads the ``docker.yml`` env pins, not the Dockerfile,
   so CI stayed green and only the (main-only) image build surfaced it.
2. **Non-idempotent comment block.** The Digest/Pinned pair was
   appended after the ``* Tag:`` line without swallowing the previous
   pair, so every run grew the comment block by two lines.

These tests source the script (its ``main`` is guarded by
``BASH_SOURCE``) and drive ``apply_pins`` / ``apply_workflow_pins``
against ``tmp_path`` fixtures. No docker daemon, no network, and no
``docker` stub on ``$PATH`` -- a PATH stub is exactly the trick that
produced a vacuously green harness in the prior session.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "pin_docker_base_images.sh"

HEX = "0123456789abcdef" * 4  # 64 lowercase hex chars
RUST_TAG = "rust:1.98.0-bookworm"
DEBIAN_TAG = "debian:bookworm-slim"
RUST_DIGEST = f"sha256:{HEX}"
DEBIAN_DIGEST = "sha256:" + ("fedcba9876543210" * 4)
DATE = "2026-09-28"

DOCKERFILE = f"""# Pinned base image block
#   * Tag:    {RUST_TAG}
#   * Digest: sha256:{"a" * 64}
#   * Pinned: 2026-01-01
FROM {RUST_TAG}@sha256:{"a" * 64} AS builder

RUN cargo build --release

# Pinned base image block
#   * Tag:    {DEBIAN_TAG}
#   * Digest: sha256:{"b" * 64}
#   * Pinned: 2026-01-01
FROM {DEBIAN_TAG}@sha256:{"b" * 64} AS runtime

ENTRYPOINT ["./fluxion-rest"]
"""

WORKFLOW = f"""env:
  RUST_BUILDER_IMAGE: {RUST_TAG}
  RUST_BUILDER_DIGEST: sha256:{"a" * 64}
  DEBIAN_RUNTIME_IMAGE: {DEBIAN_TAG}
  DEBIAN_RUNTIME_DIGEST: sha256:{"b" * 64}
"""


def _source_and_run(shell: str, tmp_path: Path) -> subprocess.CompletedProcess:
    """Source the pin script and run ``shell`` against fixture paths.

    The script is sourced (not executed) so ``main`` never runs and the
    docker daemon is never contacted; ``$DOCKERFILE`` / ``$WORKFLOW`` are
    then pointed at the fixtures.
    """
    dockerfile = tmp_path / "Dockerfile"
    workflow = tmp_path / "docker.yml"
    dockerfile.write_text(DOCKERFILE, encoding="utf-8")
    workflow.write_text(WORKFLOW, encoding="utf-8")

    return subprocess.run(
        [
            "bash",
            "-c",
            (
                f'set -euo pipefail; source "{SCRIPT}"; '
                f'DOCKERFILE="{dockerfile}"; WORKFLOW="{workflow}"; {shell}'
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# normalize_digest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        f"rust@sha256:{HEX}",  # docker inspect RepoDigests[0] form
        f"sha256:{HEX}",  # already canonical
        HEX,  # bare hex
    ],
)
def test_normalize_digest_canonicalises_every_input_form(raw: str) -> None:
    """The three spellings must converge on one canonical form.

    Accepting all three is what keeps a prefix from being double-applied:
    the bug was a write site that prepended `sha256:` to a value that
    already had it.
    """
    out = subprocess.run(
        ["bash", "-c", f'set -euo pipefail; source "{SCRIPT}"; normalize_digest "{raw}"'],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == f"sha256:{HEX}"


@pytest.mark.parametrize("raw", ["", "sha256:deadbeef", "not-a-digest", f"sha256:{'z' * 64}"])
def test_normalize_digest_rejects_malformed_input(raw: str) -> None:
    """A malformed digest must fail loudly instead of reaching a Dockerfile."""
    out = subprocess.run(
        ["bash", "-c", f'set -euo pipefail; source "{SCRIPT}"; normalize_digest "{raw}"'],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode != 0
    assert "malformed digest" in out.stderr


# ---------------------------------------------------------------------------
# apply_pins -- the #4149 corruption
# ---------------------------------------------------------------------------


def test_apply_pins_never_doubles_the_algorithm_prefix(tmp_path: Path) -> None:
    """Every emitted reference must have exactly one `sha256:`.

    This is the direct regression test for the corruption: before the
    fix, all four write sites produced `sha256:sha256:`.
    """
    out = _source_and_run(
        f"apply_pins \"$DOCKERFILE\" '{RUST_TAG}' '{RUST_DIGEST}' '{DATE}'", tmp_path
    )
    assert out.returncode == 0, out.stderr
    text = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
    assert "sha256:sha256:" not in text


def test_from_lines_are_parseable_image_references(tmp_path: Path) -> None:
    """A `FROM` line must be `<tag>@sha256:<64 hex> [AS <stage>]`."""
    _source_and_run(
        f"apply_pins \"$DOCKERFILE\" '{RUST_TAG}' '{RUST_DIGEST}' '{DATE}'", tmp_path
    )
    text = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
    from_lines = [ln for ln in text.splitlines() if ln.startswith("FROM ")]
    assert from_lines, "fixture lost its FROM lines"
    for line in from_lines:
        assert re.fullmatch(
            rf"FROM {re.escape(RUST_TAG)}@sha256:[a-f0-9]{{64}} AS builder"
            rf"|FROM {re.escape(DEBIAN_TAG)}@sha256:[a-f0-9]{{64}} AS runtime",
            line,
        ), f"unparseable image reference: {line}"


def test_apply_pins_preserves_the_as_stage_suffix(tmp_path: Path) -> None:
    """Rebuilding the `FROM` line must not drop the build stage name."""
    _source_and_run(
        f"apply_pins \"$DOCKERFILE\" '{RUST_TAG}' '{RUST_DIGEST}' '{DATE}'", tmp_path
    )
    assert f"FROM {RUST_TAG}@{RUST_DIGEST} AS builder" in (
        tmp_path / "Dockerfile"
    ).read_text(encoding="utf-8")


def test_apply_pins_updates_the_documented_digest(tmp_path: Path) -> None:
    """The comment block must report the digest that was actually pinned."""
    _source_and_run(
        f"apply_pins \"$DOCKERFILE\" '{RUST_TAG}' '{RUST_DIGEST}' '{DATE}'", tmp_path
    )
    text = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
    assert f"#   * Digest: {RUST_DIGEST}" in text
    assert f"#   * Pinned: {DATE}" in text


def test_apply_pins_is_idempotent(tmp_path: Path) -> None:
    """Two consecutive runs must produce byte-identical files.

    Guards the accumulation defect: the old implementation appended a
    fresh Digest/Pinned pair on every run instead of replacing it.
    """
    shell = f"apply_pins \"$DOCKERFILE\" '{RUST_TAG}' '{RUST_DIGEST}' '{DATE}'"
    assert _source_and_run(shell, tmp_path).returncode == 0
    once = (tmp_path / "Dockerfile").read_text(encoding="utf-8")

    assert _source_and_run(shell, tmp_path).returncode == 0
    twice = (tmp_path / "Dockerfile").read_text(encoding="utf-8")

    assert once == twice
    assert once.count("#   * Digest:") == 2, "comment block accumulated Digest lines"
    assert once.count("#   * Pinned:") == 2, "comment block accumulated Pinned lines"


def test_apply_pins_repairs_a_doubly_prefixed_from_line(tmp_path: Path) -> None:
    """A previously corrupted `FROM` is rewritten, not skipped.

    The rewrite rebuilds the line from tag + digest rather than
    substituting in place, so it heals the `sha256:sha256:` state the
    old script could itself produce.
    """
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        f"FROM {RUST_TAG}@sha256:sha256:{HEX} AS builder\n", encoding="utf-8"
    )
    subprocess.run(
        [
            "bash",
            "-c",
            (
                f'set -euo pipefail; source "{SCRIPT}"; '
                f'apply_pins "{dockerfile}" "{RUST_TAG}" "{RUST_DIGEST}" "{DATE}"'
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert dockerfile.read_text(encoding="utf-8") == f"FROM {RUST_TAG}@{RUST_DIGEST} AS builder\n"


def test_apply_pins_pins_an_unpinned_from_line(tmp_path: Path) -> None:
    """A tag-only `FROM` must be given its digest, not left unpinned."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f"FROM {RUST_TAG} AS builder\n", encoding="utf-8")
    subprocess.run(
        [
            "bash",
            "-c",
            (
                f'set -euo pipefail; source "{SCRIPT}"; '
                f'apply_pins "{dockerfile}" "{RUST_TAG}" "{RUST_DIGEST}" "{DATE}"'
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert dockerfile.read_text(encoding="utf-8") == f"FROM {RUST_TAG}@{RUST_DIGEST} AS builder\n"


def test_apply_pins_leaves_other_images_untouched(tmp_path: Path) -> None:
    """Rewriting one pin must not disturb a different base image."""
    out = _source_and_run(
        f"apply_pins \"$DOCKERFILE\" '{RUST_TAG}' '{RUST_DIGEST}' '{DATE}'", tmp_path
    )
    assert out.returncode == 0, out.stderr
    text = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
    assert f"FROM {DEBIAN_TAG}@sha256:{'b' * 64} AS runtime" in text


# ---------------------------------------------------------------------------
# dockerfile_digest
# ---------------------------------------------------------------------------


def test_dockerfile_digest_returns_the_bare_digest(tmp_path: Path) -> None:
    """The ` AS <stage>` suffix must not ride along into the digest.

    Regression test for a false-failing `--check`: the helper used to
    return `sha256:<hex> AS builder`, so the parity assertion compared
    that against `sha256:<hex>` and reported a mismatch between two
    byte-identical values -- the signature of an unsatisfiable gate.
    """
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        f"FROM {RUST_TAG}@{RUST_DIGEST} AS builder\n", encoding="utf-8"
    )
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'set -euo pipefail; source "{SCRIPT}"; dockerfile_digest "{dockerfile}" "{RUST_TAG}"',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == RUST_DIGEST


def test_dockerfile_digest_round_trips_through_normalize(tmp_path: Path) -> None:
    """What the helper returns must be accepted by `normalize_digest`.

    The two functions meet only in `--check`, so a shape mismatch
    between them is invisible to the rewrite tests.
    """
    out = _source_and_run(
        f'normalize_digest "$(dockerfile_digest "$DOCKERFILE" \'{RUST_TAG}\')"',
        tmp_path,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == f"sha256:{'a' * 64}"


def test_dockerfile_digest_empty_for_unpinned_from_line(tmp_path: Path) -> None:
    """An unpinned `FROM` must report empty, not the tag or a partial."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f"FROM {RUST_TAG} AS builder\n", encoding="utf-8")
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'set -euo pipefail; source "{SCRIPT}"; dockerfile_digest "{dockerfile}" "{RUST_TAG}"',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


# ---------------------------------------------------------------------------
# apply_workflow_pins
# ---------------------------------------------------------------------------


def test_apply_workflow_pins_updates_both_env_entries(tmp_path: Path) -> None:
    out = _source_and_run(
        f"apply_workflow_pins \"$WORKFLOW\" '{RUST_DIGEST}' '{DEBIAN_DIGEST}'", tmp_path
    )
    assert out.returncode == 0, out.stderr
    text = (tmp_path / "docker.yml").read_text(encoding="utf-8")
    assert f"  RUST_BUILDER_DIGEST: {RUST_DIGEST}" in text
    assert f"  DEBIAN_RUNTIME_DIGEST: {DEBIAN_DIGEST}" in text
    assert "sha256:sha256:" not in text


def test_apply_workflow_pins_preserves_surrounding_lines(tmp_path: Path) -> None:
    out = _source_and_run(
        f"apply_workflow_pins \"$WORKFLOW\" '{RUST_DIGEST}' '{DEBIAN_DIGEST}'", tmp_path
    )
    assert out.returncode == 0, out.stderr
    lines = (tmp_path / "docker.yml").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "env:"
    assert f"  RUST_BUILDER_IMAGE: {RUST_TAG}" in lines
    assert f"  DEBIAN_RUNTIME_IMAGE: {DEBIAN_TAG}" in lines


def test_workflow_and_dockerfile_agree_after_a_full_rewrite(tmp_path: Path) -> None:
    """The two files must end up pinning byte-identical digests.

    This is the invariant the CI assertion depends on but does not
    itself check: the workflow only validates the env pins, so a
    Dockerfile that drifted from them keeps CI green.
    """
    shell = (
        f"apply_pins \"$DOCKERFILE\" '{RUST_TAG}' '{RUST_DIGEST}' '{DATE}'; "
        f"apply_pins \"$DOCKERFILE\" '{DEBIAN_TAG}' '{DEBIAN_DIGEST}' '{DATE}'; "
        f"apply_workflow_pins \"$WORKFLOW\" '{RUST_DIGEST}' '{DEBIAN_DIGEST}'"
    )
    assert _source_and_run(shell, tmp_path).returncode == 0

    dockerfile = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
    workflow = (tmp_path / "docker.yml").read_text(encoding="utf-8")
    for tag, digest in ((RUST_TAG, RUST_DIGEST), (DEBIAN_TAG, DEBIAN_DIGEST)):
        from_digest = re.search(rf"^FROM {re.escape(tag)}@(\S+)", dockerfile, re.MULTILINE)
        env_key = "RUST_BUILDER_DIGEST" if tag == RUST_TAG else "DEBIAN_RUNTIME_DIGEST"
        env_digest = re.search(rf"^  {env_key}: (\S+)", workflow, re.MULTILINE)
        assert from_digest and env_digest
        assert from_digest.group(1) == env_digest.group(1) == digest
