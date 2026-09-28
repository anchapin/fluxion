"""
Tests for the ``<model>.sha256`` digest sidecar (Issue #4191).

``scripts/export_onnx.py::write_sha256_sidecar`` must write the sidecar
atomically in standard ``sha256sum`` format so the Rust runtime verifier
(``verify_onnx_signature``) accepts it. The format contract is verified
here; the Rust side is covered by
``src/ai/surrogate/integrity.rs::exported_sidecar_is_accepted``.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

SCRIPT_NAME = "export_onnx"


@pytest.fixture
def exporter(load_script):
    return load_script(SCRIPT_NAME)


def _make_model(tmp_path: Path, name: str = "model.onnx") -> Path:
    model = tmp_path / name
    model.write_bytes(b"fake-onnx-bytes-for-digest-test" * 64)
    return model


def test_sidecar_written_in_sha256sum_format(exporter, tmp_path):
    model = _make_model(tmp_path)
    sidecar = exporter.write_sha256_sidecar(model)
    assert sidecar == tmp_path / "model.onnx.sha256"
    assert sidecar.exists()

    expected_digest = hashlib.sha256(model.read_bytes()).hexdigest()
    content = sidecar.read_text(encoding="utf-8")
    assert content == f"{expected_digest}  {model.name}\n"


def test_sidecar_passes_sha256sum_check(exporter, tmp_path):
    """The system ``sha256sum -c`` must accept the generated sidecar."""
    model = _make_model(tmp_path)
    sidecar = exporter.write_sha256_sidecar(model)
    result = subprocess.run(
        ["sha256sum", "-c", str(sidecar)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_sidecar_is_atomic_no_tmp_left_behind(exporter, tmp_path):
    model = _make_model(tmp_path)
    exporter.write_sha256_sidecar(model)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_sidecar_digest_matches_compute_model_hash(exporter, tmp_path):
    model = _make_model(tmp_path)
    digest = exporter.compute_model_hash(model)
    sidecar = exporter.write_sha256_sidecar(model)
    assert sidecar.read_text(encoding="utf-8").startswith(digest)
