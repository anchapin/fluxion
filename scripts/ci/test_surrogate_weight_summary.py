"""Regression tests for ``scripts/surrogate_weight_summary.py``.

Issue #4670 / PR #4783 -- the path-weighted first-layer summary crashed (or
silently mis-computed) for a **single-hidden-layer** surrogate, i.e. a model
whose exported ONNX graph carries exactly two weight matrices::

    W_layer0 : (n_features, n_hidden)
    W_layer1 : (n_hidden, n_outputs)

``compute_summary`` special-cased the single-matrix model but assumed at
least three matrices in the ``else`` branch::

    w1 = weights.weights[1]
    w_last = weights.weights[-1]
    path_score = w1 @ w_last          # <- W1 @ W1 for two matrices!

With two matrices ``w1 is w_last``, so the path score became ``W1 @ W1``:
numpy raised ``ValueError`` when ``n_hidden != n_outputs`` (the JiTTest H1
case) and silently returned nonsense when ``n_hidden == n_outputs``.

The correct path score is the chain product over ``weights[1:]`` -- ``W1``
alone for a single hidden layer -- which is what the script now computes.

The tests below build synthetic weights in memory (no network, no fixture
files outside ``tmp_path``) and check the numbers against the ground truth
of the synthetic weights.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_NAME = "surrogate_weight_summary"

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / f"{SCRIPT_NAME}.py"


@pytest.fixture
def summary_mod(load_script):
    """The script under test, loaded fresh from ``scripts/``."""
    return load_script(SCRIPT_NAME)


# ---------------------------------------------------------------------------
# Synthetic weight builders
# ---------------------------------------------------------------------------


def _weights(mod, matrices):
    """Build a ``SurrogateWeights`` from a list of weight matrices."""
    return mod.SurrogateWeights(
        weights=[np.asarray(m, dtype=np.float64) for m in matrices],
        biases=[
            np.zeros(np.asarray(m).shape[1], dtype=np.float64) for m in matrices
        ],
    )


def _deterministic_matrix(rows: int, cols: int, seed: int) -> np.ndarray:
    """A reproducible, well-conditioned matrix with mixed signs."""
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(rows, cols))
    # Guarantee no all-zero rows/columns so signs and norms are meaningful.
    for i in range(rows):
        if not np.any(mat[i] != 0.0):
            mat[i, 0] = 1.0
    return mat


def _feature_names(n: int):
    return [f"feature_{i}" for i in range(n)]


def _feature_std(n: int):
    return [1.0 + 0.25 * i for i in range(n)]


# ---------------------------------------------------------------------------
# 1. Two matrices, n_hidden != n_outputs  (the JiTTest H1 case)
# ---------------------------------------------------------------------------


def test_two_matrix_model_does_not_raise(summary_mod):
    """A single-hidden-layer surrogate must not crash the summary."""
    n_features, n_hidden, n_outputs = 5, 64, 1
    weights = _weights(
        summary_mod,
        [
            _deterministic_matrix(n_features, n_hidden, seed=1),
            _deterministic_matrix(n_hidden, n_outputs, seed=2),
        ],
    )

    summary = summary_mod.compute_summary(
        weights,
        _feature_names(n_features),
        _feature_std(n_features),
    )

    assert summary.n_features == n_features
    assert summary.n_outputs == n_outputs
    assert summary.layer_shapes == [(n_features, n_hidden), (n_hidden, n_outputs)]


def test_two_matrix_model_signs_match_ground_truth(summary_mod):
    """Per-input signs match the ground truth ``W0 @ W1`` of the weights."""
    n_features, n_hidden, n_outputs = 6, 64, 1
    w0 = _deterministic_matrix(n_features, n_hidden, seed=11)
    w1 = _deterministic_matrix(n_hidden, n_outputs, seed=12)
    weights = _weights(summary_mod, [w0, w1])

    summary = summary_mod.compute_summary(
        weights,
        _feature_names(n_features),
        _feature_std(n_features),
    )

    # Ground truth: the effective input influence of a single-hidden-layer
    # model is the plain matrix product W0 @ W1.
    expected = w0 @ w1
    assert expected.shape == (n_features, n_outputs)

    for j, effect in enumerate(summary.effects):
        dominant = int(np.argmax(np.abs(expected[j])))
        value = float(expected[j][dominant])
        assert effect.sign == int(np.sign(value)), f"feature {j} sign mismatch"
        assert effect.per_std_effect == pytest.approx(value, rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Two matrices, n_hidden == n_outputs  (the silent mis-computation)
# ---------------------------------------------------------------------------


def test_two_matrix_square_model_influence_is_w0_times_w1(summary_mod):
    """The square case must yield exactly ``W0 @ W1`` -- not ``W0 @ (W1 @ W1)``."""
    n_features = 4
    n_hidden = n_outputs = 5
    w0 = _deterministic_matrix(n_features, n_hidden, seed=21)
    w1 = _deterministic_matrix(n_hidden, n_outputs, seed=22)
    weights = _weights(summary_mod, [w0, w1])

    summary = summary_mod.compute_summary(
        weights,
        _feature_names(n_features),
        _feature_std(n_features),
    )

    expected = w0 @ w1
    for j, effect in enumerate(summary.effects):
        dominant = int(np.argmax(np.abs(expected[j])))
        assert effect.per_std_effect == pytest.approx(
            float(expected[j][dominant]), rel=1e-9, abs=1e-12
        )

    # The buggy ``W1 @ W1`` path score would give a different answer here;
    # guard against a silent regression to it.
    buggy = w0 @ (w1 @ w1)
    assert not np.allclose(expected, buggy)


# ---------------------------------------------------------------------------
# 3. Unchanged behaviour: one matrix and three matrices
# ---------------------------------------------------------------------------


def test_single_matrix_model_unchanged(summary_mod):
    """No hidden layer: the path score is the identity, so eff == W0."""
    n_features, n_outputs = 5, 3
    w0 = _deterministic_matrix(n_features, n_outputs, seed=31)
    weights = _weights(summary_mod, [w0])

    summary = summary_mod.compute_summary(
        weights,
        _feature_names(n_features),
        _feature_std(n_features),
    )

    assert summary.top_units == []
    for j, effect in enumerate(summary.effects):
        dominant = int(np.argmax(np.abs(w0[j])))
        value = float(w0[j][dominant])
        assert effect.sign == int(np.sign(value))
        assert effect.per_std_effect == pytest.approx(value, rel=1e-9, abs=1e-12)


def test_three_matrix_model_unchanged(summary_mod):
    """Two hidden layers: the path score stays ``W1 @ W2``."""
    n_features, h1, h2, n_outputs = 5, 8, 4, 2
    w0 = _deterministic_matrix(n_features, h1, seed=41)
    w1 = _deterministic_matrix(h1, h2, seed=42)
    w2 = _deterministic_matrix(h2, n_outputs, seed=43)
    weights = _weights(summary_mod, [w0, w1, w2])

    summary = summary_mod.compute_summary(
        weights,
        _feature_names(n_features),
        _feature_std(n_features),
    )

    expected = w0 @ (w1 @ w2)
    for j, effect in enumerate(summary.effects):
        dominant = int(np.argmax(np.abs(expected[j])))
        value = float(expected[j][dominant])
        assert effect.sign == int(np.sign(value))
        assert effect.per_std_effect == pytest.approx(value, rel=1e-9, abs=1e-12)


def test_four_matrix_model_uses_the_full_chain_product(summary_mod):
    """Three hidden layers: the path score is ``W1 @ W2 @ W3``.

    The chain product over ``weights[1:]`` is the general form; the old code
    took ``W1 @ W_last``, which silently skipped the intermediate layer.
    """
    n_features, h1, h2, h3, n_outputs = 4, 6, 5, 3, 2
    w0 = _deterministic_matrix(n_features, h1, seed=45)
    w1 = _deterministic_matrix(h1, h2, seed=46)
    w2 = _deterministic_matrix(h2, h3, seed=47)
    w3 = _deterministic_matrix(h3, n_outputs, seed=48)
    weights = _weights(summary_mod, [w0, w1, w2, w3])

    summary = summary_mod.compute_summary(
        weights,
        _feature_names(n_features),
        _feature_std(n_features),
    )

    expected = w0 @ (w1 @ (w2 @ w3))
    for j, effect in enumerate(summary.effects):
        dominant = int(np.argmax(np.abs(expected[j])))
        value = float(expected[j][dominant])
        assert effect.sign == int(np.sign(value))
        assert effect.per_std_effect == pytest.approx(value, rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# 4. Hidden-unit profile of the single-hidden-layer model
# ---------------------------------------------------------------------------


def test_two_matrix_model_hidden_unit_profile_is_consistent(summary_mod):
    """The top-unit profile of a single-hidden-layer model stays meaningful.

    The last hidden layer is the one hidden layer; its outgoing weights are
    the rows of ``W1`` and its incoming weights are the columns of ``W0``.
    Path strengths therefore follow the row norms of ``W1`` and the per-input
    profile of a unit follows the corresponding column of ``W0`` -- exactly
    the convention the three-or-more-matrix case uses.
    """
    n_features, n_hidden, n_outputs = 4, 6, 1
    w0 = _deterministic_matrix(n_features, n_hidden, seed=51)
    w1 = _deterministic_matrix(n_hidden, n_outputs, seed=52)
    weights = _weights(summary_mod, [w0, w1])

    summary = summary_mod.compute_summary(
        weights,
        _feature_names(n_features),
        _feature_std(n_features),
        top_k=3,
    )

    assert len(summary.top_units) == 3
    strengths = np.linalg.norm(w1, axis=1)
    order = np.argsort(-strengths)[:3]
    for unit, profile in zip(order, summary.top_units):
        assert profile.unit == int(unit)
        assert profile.path_strength == pytest.approx(
            float(strengths[int(unit)]), rel=1e-9, abs=1e-12
        )
        # The per-input contributions of the unit are the column of W0.
        expected_profile = w0[:, int(unit)]
        for name, value in profile.inputs:
            j = int(name.split("_")[1])
            assert value == pytest.approx(
                float(expected_profile[j]), rel=1e-9, abs=1e-12
            )


# ---------------------------------------------------------------------------
# 5. End-to-end CLI check on a synthetic two-matrix ONNX
# ---------------------------------------------------------------------------


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _varint_field(field: int, value: int) -> bytes:
    return _varint((field << 3) | 0) + _varint(value)


def _bytes_field(field: int, payload: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _tensor_proto(name: str, array: np.ndarray) -> bytes:
    """A minimal ONNX ``TensorProto`` (float32, raw_data) for ``array``."""
    payload = b""
    for dim in array.shape:
        payload += _varint_field(1, int(dim))  # dims
    payload += _varint_field(2, 1)  # data_type = FLOAT
    payload += _bytes_field(8, name.encode("utf-8"))  # name
    payload += _bytes_field(
        9, np.asarray(array, dtype=np.float32).tobytes()
    )  # raw_data
    return payload


def _onnx_bytes(matrices) -> bytes:
    """A minimal ONNX model carrying ``W_layer{i}`` / ``b_layer{i}``."""
    graph = b""
    for i, mat in enumerate(matrices):
        graph += _bytes_field(
            5, _tensor_proto(f"W_layer{i}", np.asarray(mat, dtype=np.float32))
        )
        bias = np.zeros(np.asarray(mat).shape[1], dtype=np.float32)
        graph += _bytes_field(5, _tensor_proto(f"b_layer{i}", bias))

    model = _varint_field(1, 8)  # ir_version
    model += _bytes_field(2, b"fluxion.surrogate")  # producer_name
    model += _bytes_field(3, b"0.1.0")  # producer_version
    model += _bytes_field(7, graph)  # graph
    opset = _bytes_field(1, b"") + _varint_field(2, 17)
    model += _bytes_field(8, opset)  # opset_import
    return model


def test_cli_end_to_end_on_synthetic_two_matrix_onnx(tmp_path):
    """The CLI prints a summary for a synthetic single-hidden-layer model."""
    n_features, n_hidden, n_outputs = 4, 8, 1
    matrices = [
        _deterministic_matrix(n_features, n_hidden, seed=61),
        _deterministic_matrix(n_hidden, n_outputs, seed=62),
    ]
    model_path = tmp_path / "surrogate_two_matrix.onnx"
    model_path.write_bytes(_onnx_bytes(matrices))

    stds = ",".join(str(s) for s in _feature_std(n_features))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(model_path),
            "--std",
            stds,
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "SURROGATE PATH-WEIGHTED FIRST-LAYER SUMMARY" in result.stdout
    # The report-only caveat block is still printed.
    assert "CAVEAT" in result.stdout
    # The effective influence printed per feature is the ground truth W0 @ W1.
    expected = matrices[0] @ matrices[1]
    lines = result.stdout.splitlines()
    for j in range(n_features):
        dominant = int(np.argmax(np.abs(expected[j])))
        value = float(expected[j][dominant])
        sign = {1: "+", -1: "-", 0: "0"}[int(np.sign(value))]
        # The per-input influence row is "<name> <sign> <per +1 std> ...";
        # the hidden-unit profile rows start with the same names but carry a
        # "contribution" column instead of a sign column.
        rows = [
            parts
            for parts in (line.split() for line in lines)
            if len(parts) >= 3
            and parts[0] == f"feature_{j}"
            and parts[1] in ("+", "-", "0")
        ]
        assert len(rows) == 1, result.stdout
        assert rows[0][1] == sign


def test_hand_built_onnx_bytes_are_readable(summary_mod, tmp_path):
    """Sanity check that the hand-built ONNX bytes are parseable.

    The script ships a dependency-free protobuf reader, so the synthetic
    model can be read back without the optional ``onnx`` package.
    """
    matrices = [
        _deterministic_matrix(3, 4, seed=71),
        _deterministic_matrix(4, 1, seed=72),
    ]
    model_path = tmp_path / "synthetic.onnx"
    model_path.write_bytes(_onnx_bytes(matrices))

    meta = summary_mod.read_onnx_without_onnx_package(model_path)
    assert "W_layer0" in meta.initializers
    assert meta.initializers["W_layer1"].shape == (4, 1)
    assert meta.producer_name == "fluxion.surrogate"
    assert meta.opset_version == 17

    weights = summary_mod.extract_weights(meta.initializers)
    assert len(weights.weights) == 2
    assert weights.weights[0].shape == (3, 4)
    assert weights.weights[1].shape == (4, 1)
