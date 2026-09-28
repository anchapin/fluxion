"""
Tests for ``scripts/generate_topology_diagrams.py`` — Issue #4194.

Generation must be transactional (fail-closed: no partial output on
failure), idempotent, deterministic, SHA-consistent, and produce SVGs
with unique element IDs. A fake ``fluxion`` binary stands in for the
real CLI so the tests run without building the Rust workspace.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

SCRIPT_NAME = "generate_topology_diagrams"

FAKE_EXPORT_DOC = {
    "metadata": {
        "model_name": "test-case",
        "model_source": "fake-fluxion",
        "node_count": 2,
        "edge_count": 1,
    },
    "nodes": [
        {"id": "n1", "kind": "outdoor_ambient", "label": "Outdoor"},
        {"id": "n2", "kind": "exterior_surface", "label": "Wall"},
    ],
    "edges": [
        {"source_id": "n1", "target_id": "n2", "coupling_type": "convection", "conductance_w_per_k": 5.0}
    ],
}

FAKE_LINT_DOC = {
    "clean": True,
    "summary": {"errors": 0, "warnings": 0, "total": 0},
}


@pytest.fixture
def generator(load_script):
    return load_script(SCRIPT_NAME)


def _fake_fluxion(tmp_path: Path, *, fail_export: bool = False, bad_json: bool = False) -> Path:
    """Write an executable fake ``fluxion`` CLI."""
    script = tmp_path / "fake-fluxion"
    export_doc = dict(FAKE_EXPORT_DOC)
    lint_doc = dict(FAKE_LINT_DOC)
    if bad_json:
        lint_payload = "not json{{"
    else:
        lint_payload = json.dumps(lint_doc)

    script.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"topology\" ] && [ \"$2\" = \"export\" ]; then\n"
        f"  {'exit 1' if fail_export else 'true'}\n"
        "  out=\"\"\n"
        "  prev=\"\"\n"
        "  for a in \"$@\"; do\n"
        "    if [ \"$prev\" = \"--output\" ]; then out=\"$a\"; fi\n"
        "    prev=\"$a\"\n"
        "  done\n"
        f"  printf '%s' '{json.dumps(export_doc)}' > \"$out\"\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = \"topology\" ] && [ \"$2\" = \"lint\" ]; then\n"
        f"  printf '%s' '{lint_payload}'\n"
        "  exit 0\n"
        "fi\n"
        "echo \"unknown command: $@\" >&2\n"
        "exit 2\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _run_generate(gen, dest: Path, fluxion_bin: Path) -> int:
    # Restrict to a single case for speed; the transactional logic is
    # case-independent.
    gen.TOPOLOGY_CASES = ["600"]
    return gen.generate(dest, str(fluxion_bin))


def _tree_files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_generate_succeeds_and_writes_expected_layout(generator, tmp_path):
    dest = tmp_path / "dest"
    fluxion = _fake_fluxion(tmp_path)
    rc = _run_generate(generator, dest, fluxion)
    assert rc == 0

    files = _tree_files(dest)
    assert "tests/reference_data/topology/case-600.json" in files
    assert "tests/reference_data/topology/lint/case-600.lint.json" in files
    assert "tests/reference_data/topology/index.json" in files
    assert "docs/architecture/topology/case-600.mmd" in files
    assert "docs/architecture/topology/case-600.svg" in files
    assert "docs/architecture/topology/payloads/case-600.json" in files
    assert "docs/architecture/topology/index.md" in files


def test_index_sha256_matches_files(generator, tmp_path):
    """Issue #4194: index/file SHA consistency."""
    dest = tmp_path / "dest"
    fluxion = _fake_fluxion(tmp_path)
    assert _run_generate(generator, dest, fluxion) == 0

    index = json.loads((dest / "tests/reference_data/topology/index.json").read_text())
    assert len(index["cases"]) == 1
    entry = index["cases"][0]
    for key, rel in entry["files"].items():
        if key == "reference":
            sha_key = "reference"
        else:
            sha_key = key
        expected = entry["sha256"][sha_key]
        actual = hashlib.sha256((dest / rel).read_bytes()).hexdigest()
        assert actual == expected, f"SHA mismatch for {rel}"


def test_idempotent_second_run_is_byte_identical(generator, tmp_path):
    """Issue #4194: idempotence — two runs produce identical trees."""
    dest = tmp_path / "dest"
    fluxion = _fake_fluxion(tmp_path)
    assert _run_generate(generator, dest, fluxion) == 0
    first = _tree_files(dest)

    assert _run_generate(generator, dest, fluxion) == 0
    second = _tree_files(dest)

    assert first.keys() == second.keys()
    for rel in first:
        assert first[rel] == second[rel], f"non-deterministic output: {rel}"


def test_svg_ids_are_unique(generator, tmp_path):
    """Issue #4194: SVG ID uniqueness."""
    dest = tmp_path / "dest"
    fluxion = _fake_fluxion(tmp_path)
    assert _run_generate(generator, dest, fluxion) == 0

    svg = (dest / "docs/architecture/topology/case-600.svg").read_text()
    ids = re.findall(r'id="([^"]+)"', svg)
    assert len(ids) > 0, "expected at least one id in the SVG"
    assert len(ids) == len(set(ids)), f"duplicate SVG ids: {sorted(set(i for i in ids if ids.count(i) > 1))}"


def test_no_minus_one_sentinels_in_index(generator, tmp_path):
    """Issue #4194: the -1 lint-count sentinels must be gone."""
    dest = tmp_path / "dest"
    fluxion = _fake_fluxion(tmp_path)
    assert _run_generate(generator, dest, fluxion) == 0

    index = json.loads((dest / "tests/reference_data/topology/index.json").read_text())
    lint = index["cases"][0]["lint"]
    assert lint["errors"] == 0
    assert lint["warnings"] == 0
    assert lint["total"] == 0


# ---------------------------------------------------------------------------
# Failure modes — destination must stay untouched
# ---------------------------------------------------------------------------


def test_failed_export_leaves_destination_untouched(generator, tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    sentinel = dest / "sentinel.txt"
    sentinel.write_text("untouched", encoding="utf-8")

    fluxion = _fake_fluxion(tmp_path, fail_export=True)
    rc = _run_generate(generator, dest, fluxion)
    assert rc != 0
    # Fail-closed: nothing written, sentinel intact.
    assert _tree_files(dest) == {"sentinel.txt": b"untouched"}


def test_unparseable_lint_leaves_destination_untouched(generator, tmp_path):
    dest = tmp_path / "dest"
    fluxion = _fake_fluxion(tmp_path, bad_json=True)
    rc = _run_generate(generator, dest, fluxion)
    assert rc != 0
    assert not dest.exists() or _tree_files(dest) == {}
