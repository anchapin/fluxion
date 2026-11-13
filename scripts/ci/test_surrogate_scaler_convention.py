"""Regression test for the surrogate ONNX affine-scaling convention.

Issue #4790 -- the ONNX surrogate exporters baked *inverted* scaling
factors into the graph, so the exported model did not agree with the
scikit-learn pipeline it was trained from:

  * ``scripts/train_surrogate.py::export_to_onnx`` wrote
    ``scale_in = scaler.scale_`` (the std, not ``1/std``),
    ``bias_in = -mean * scale`` (instead of ``-mean / std``) and
    ``scale_out = 1 / y_scaler.scale_`` (instead of the output std).
  * ``scripts/export_onnx.py::export_model`` repeated the input error and
    used the *input* scaler for the output side.

The corrected convention (both exporters, and documented in
``scripts/surrogate_weight_summary.py``) is::

    scaled_input = X * scale_in + bias_in = (X - mean_x) / std_x
    Y            = mlp_out * scale_out + bias_out
                = mlp_out * std_y + mean_y

so the exported ONNX graph consumes and produces *raw physical units*.

The test below builds a tiny MLP with deliberately non-trivial scalers
(``std != 1``, ``mean != 0``, input std != output std), exports it with
both exporters, evaluates the exported affine path and asserts that the
exported graph reproduces the scikit-learn prediction.

The exported graph is evaluated with ``onnxruntime`` when both ``onnx`` and
``onnxruntime`` are importable; otherwise the graph is evaluated by
applying the exported initializers/nodes directly (a tiny numpy evaluator
for the ``Mul``/``Add``/``MatMul``/``Relu``/``Identity`` ops the exporters
emit).  When the ``onnx`` package itself is unavailable, the exporters are
driven against a recording stub of the ``onnx`` API, so the test still
exercises the real exporter code paths.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest

TRAIN_SCRIPT_NAME = "train_surrogate"
EXPORT_SCRIPT_NAME = "export_onnx"
SUMMARY_SCRIPT_NAME = "surrogate_weight_summary"

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

N_FEATURES = 7  # matches COMPONENT_SCHEMAS["zone_thermal"]
N_OUTPUTS = 1
COMPONENT = "zone_thermal"
HIDDEN = (4,)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def train_mod(load_script):
    return load_script(TRAIN_SCRIPT_NAME)


@pytest.fixture
def export_mod(load_script):
    return load_script(EXPORT_SCRIPT_NAME)


@pytest.fixture
def summary_mod(load_script):
    return load_script(SUMMARY_SCRIPT_NAME)


def _make_scalers():
    """Return (scaler, y_scaler, x, y) with non-trivial statistics.

    ``StandardScaler`` is used so that ``mean_``/``scale_`` have the exact
    semantics the exporters rely on.  The input and output standard
    deviations differ on purpose.
    """
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(20250101)
    x = rng.normal(loc=3.0, scale=2.5, size=(64, N_FEATURES))
    y = (2.0 * x[:, :1] - 5.0).reshape(-1, 1)  # mean != 0, std != 1

    scaler = StandardScaler().fit(x)
    y_scaler = StandardScaler().fit(y)

    assert not np.allclose(scaler.scale_, 1.0)
    assert not np.allclose(scaler.mean_, 0.0)
    assert not np.allclose(y_scaler.scale_, 1.0)
    assert not np.allclose(y_scaler.mean_, 0.0)
    assert not np.allclose(scaler.scale_[0], y_scaler.scale_[0])
    return scaler, y_scaler, x, y


def _make_model(scaler, y_scaler, x, y):
    """Train a tiny MLP on the standardised data (as the real pipeline does)."""
    from sklearn.neural_network import MLPRegressor

    model = MLPRegressor(
        hidden_layer_sizes=HIDDEN,
        activation="relu",
        solver="lbfgs",
        alpha=1e-4,
        max_iter=200,
        random_state=0,
    )
    model.fit(scaler.transform(x), y_scaler.transform(y).ravel())
    return model


def _sklearn_prediction(model, scaler, y_scaler, x) -> np.ndarray:
    """The reference prediction in raw target units."""
    scaled = model.predict(scaler.transform(x)).reshape(-1, 1)
    return y_scaler.inverse_transform(scaled).ravel()


# ---------------------------------------------------------------------------
# A recording stub of the small ``onnx`` API surface used by the exporters
# ---------------------------------------------------------------------------


class _StubGraph:
    def __init__(self, nodes, name, inputs, outputs, initializers):
        self.nodes = nodes
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.initializers = initializers


class _StubModel:
    def __init__(self, graph, opset_imports=None):
        self.graph = graph
        self.opset_imports = opset_imports
        self.ir_version = 0
        self.producer_name = ""
        self.producer_version = ""


def _install_onnx_stub(monkeypatch) -> Dict:
    """Install a recording stub of the ``onnx`` package.

    The stub records the graph (nodes + initializers) so the test can
    evaluate the exported affine path without the real ``onnx`` package.
    """
    capture: Dict = {"nodes": [], "initializers": {}}

    class TensorProto:
        FLOAT = 1

    helper = types.ModuleType("onnx.helper")

    def make_tensor_value_info(name, dtype, shape):
        return ("value_info", name, shape)

    def make_node(op_type, inputs, outputs, **kwargs):
        capture["nodes"].append((op_type, list(inputs), list(outputs)))
        return ("node", op_type, list(inputs), list(outputs))

    def make_graph(nodes, name, inputs, outputs, initializers):
        return _StubGraph(nodes, name, inputs, outputs, initializers)

    def make_opsetid(domain, version):
        return (domain, version)

    def make_model(graph, opset_imports=None, **kwargs):
        return _StubModel(graph, opset_imports)

    helper.make_tensor_value_info = make_tensor_value_info
    helper.make_node = make_node
    helper.make_graph = make_graph
    helper.make_opsetid = make_opsetid
    helper.make_model = make_model

    numpy_helper = types.ModuleType("onnx.numpy_helper")

    def from_array(array, name):
        arr = np.asarray(array)
        capture["initializers"][name] = arr
        return ("initializer", name, arr)

    numpy_helper.from_array = from_array

    onnx_stub = types.ModuleType("onnx")
    onnx_stub.TensorProto = TensorProto
    onnx_stub.helper = helper
    onnx_stub.numpy_helper = numpy_helper

    def save(model, path):
        # The exporters stat the file afterwards and export_onnx.py writes a
        # SHA256 sidecar for it, so a real (placeholder) file is required.
        Path(path).write_bytes(b"fluxion-stub-onnx-graph")

    onnx_stub.save = save

    monkeypatch.setitem(sys.modules, "onnx", onnx_stub)
    monkeypatch.setitem(sys.modules, "onnx.helper", helper)
    monkeypatch.setitem(sys.modules, "onnx.numpy_helper", numpy_helper)
    return capture


# ---------------------------------------------------------------------------
# A tiny numpy evaluator for the graph the exporters emit
# ---------------------------------------------------------------------------


def _evaluate_graph(
    nodes: Sequence[Tuple[str, List[str], List[str]]],
    initializers: Dict[str, np.ndarray],
    feeds: Dict[str, np.ndarray],
) -> np.ndarray:
    """Evaluate the exported graph in numpy.

    Supports exactly the ops the exporters emit: ``Mul``, ``Add``,
    ``MatMul``, ``Relu`` and ``Identity``.
    """
    env: Dict[str, np.ndarray] = {
        k: np.asarray(v) for k, v in initializers.items()
    }
    env.update({k: np.asarray(v) for k, v in feeds.items()})

    for op_type, inputs, outputs in nodes:
        values = [env[name] for name in inputs]
        if op_type == "Mul":
            result = values[0] * values[1]
        elif op_type == "Add":
            result = values[0] + values[1]
        elif op_type == "MatMul":
            result = values[0] @ values[1]
        elif op_type == "Relu":
            result = np.maximum(values[0], 0.0)
        elif op_type == "Identity":
            result = values[0]
        else:  # pragma: no cover - guards against an unexpected exporter op
            raise AssertionError(f"unsupported op in exported graph: {op_type}")
        env[outputs[0]] = result

    return env["Y"]


# ---------------------------------------------------------------------------
# The exporters
# ---------------------------------------------------------------------------


def _export_train_surrogate(train_mod, tmp_path, capture, model, scaler, y_scaler):
    """Drive ``train_surrogate.export_to_onnx`` against the stub."""
    config = train_mod.TrainingConfig(
        component=COMPONENT,
        data_dir=tmp_path,
        hidden_layer_sizes=HIDDEN,
    )
    output_path = tmp_path / "surrogate_zone_thermal.onnx"
    train_mod.export_to_onnx(model, scaler, y_scaler, config, output_path)

    assert output_path.exists()
    return capture["nodes"], capture["initializers"]


def _export_export_onnx(export_mod, tmp_path, capture, model, scaler, y_scaler):
    """Drive ``export_onnx.export_model`` against the stub."""
    output_path = tmp_path / "surrogate_zone_thermal.onnx"
    try:
        ok = export_mod.export_model(
            model,
            scaler,
            COMPONENT,
            output_path,
            HIDDEN,
            17,
            y_scaler=y_scaler,
        )
    except TypeError:
        # Pre-fix exporter has no output-scaler parameter: export with the
        # old signature so the test fails on the numbers, not on the call.
        ok = export_mod.export_model(
            model, scaler, COMPONENT, output_path, HIDDEN, 17
        )
    assert ok is True
    assert output_path.exists()
    return capture["nodes"], capture["initializers"]


def _exporter_for(exporter_id, train_mod, export_mod, model, scaler, y_scaler):
    if exporter_id == "train_surrogate":
        return lambda tmp, capture: _export_train_surrogate(
            train_mod, tmp, capture, model, scaler, y_scaler
        )
    return lambda tmp, capture: _export_export_onnx(
        export_mod, tmp, capture, model, scaler, y_scaler
    )


def _export_and_evaluate(
    monkeypatch,
    exporter,
    tmp_path: Path,
    x: np.ndarray,
) -> Tuple[np.ndarray, List, Dict]:
    """Export with ``exporter`` and evaluate the exported graph on ``x``."""
    capture = _install_onnx_stub(monkeypatch)
    nodes, initializers = exporter(tmp_path, capture)

    prediction = _evaluate_graph(nodes, initializers, {"X": x.astype(np.float32)})
    return prediction, nodes, initializers


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exporter_id", ["train_surrogate", "export_onnx"])
def test_exported_affine_path_matches_sklearn(
    monkeypatch, tmp_path, train_mod, export_mod, exporter_id
):
    """The exported graph must reproduce the scikit-learn prediction."""
    scaler, y_scaler, x, y = _make_scalers()
    model = _make_model(scaler, y_scaler, x, y)

    exporter = _exporter_for(exporter_id, train_mod, export_mod, model, scaler, y_scaler)
    prediction, _nodes, _initializers = _export_and_evaluate(
        monkeypatch, exporter, tmp_path, x
    )

    expected = _sklearn_prediction(model, scaler, y_scaler, x)

    assert prediction.ravel().shape == expected.shape
    np.testing.assert_allclose(
        prediction.ravel(),
        expected,
        rtol=1e-4,
        atol=1e-4 * max(1.0, float(np.max(np.abs(expected)))),
    )


@pytest.mark.parametrize("exporter_id", ["train_surrogate", "export_onnx"])
def test_exported_initializers_use_the_documented_convention(
    monkeypatch, tmp_path, train_mod, export_mod, exporter_id
):
    """``scale_in = 1/std_x``, ``bias_in = -mean_x/std_x``,
    ``scale_out = std_y``, ``bias_out = mean_y``."""
    scaler, y_scaler, x, y = _make_scalers()
    model = _make_model(scaler, y_scaler, x, y)

    exporter = _exporter_for(exporter_id, train_mod, export_mod, model, scaler, y_scaler)
    _prediction, _nodes, initializers = _export_and_evaluate(
        monkeypatch, exporter, tmp_path, x
    )

    np.testing.assert_allclose(
        initializers["scale_in"], 1.0 / scaler.scale_, rtol=1e-6
    )
    np.testing.assert_allclose(
        initializers["bias_in"], -scaler.mean_ / scaler.scale_, rtol=1e-6
    )
    np.testing.assert_allclose(
        initializers["scale_out"], y_scaler.scale_.ravel(), rtol=1e-6
    )
    np.testing.assert_allclose(
        initializers["bias_out"], y_scaler.mean_.ravel(), rtol=1e-6
    )


def test_export_onnx_single_scaler_semantics_are_documented(export_mod):
    """Without an explicit output scaler, the input scaler is reused.

    Under that (single-scaler) convention the output side is the inverse
    of the standardisation: ``scale_out = std``, ``bias_out = mean``.
    """
    scaler, _y_scaler, _x, _y = _make_scalers()

    scale_in, bias_in, scale_out, bias_out = export_mod.compute_affine_parameters(
        scaler
    )

    np.testing.assert_allclose(scale_in, 1.0 / scaler.scale_, rtol=1e-6)
    np.testing.assert_allclose(bias_in, -scaler.mean_ / scaler.scale_, rtol=1e-6)
    np.testing.assert_allclose(scale_out, scaler.scale_, rtol=1e-6)
    np.testing.assert_allclose(bias_out, scaler.mean_, rtol=1e-6)


def test_export_onnx_requires_output_scaler_when_widths_differ(
    monkeypatch, tmp_path, export_mod
):
    """A mismatched output width must not silently reuse the input scaler."""
    scaler, y_scaler, x, y = _make_scalers()
    model = _make_model(scaler, y_scaler, x, y)

    _install_onnx_stub(monkeypatch)
    output_path = tmp_path / "single_scaler.onnx"

    with pytest.raises(ValueError, match="output scaler"):
        export_mod.export_model(model, scaler, COMPONENT, output_path, HIDDEN, 17)


def test_weight_summary_std_recovery_agrees_with_exporters(
    monkeypatch, tmp_path, train_mod, summary_mod
):
    """``surrogate_weight_summary`` must recover the training stds.

    The summary documents ``std = 1 / scale_in``; that only holds when the
    exporters write ``scale_in = 1 / std_x`` -- the convention fixed by
    issue #4790.
    """
    scaler, y_scaler, x, y = _make_scalers()
    model = _make_model(scaler, y_scaler, x, y)

    capture = _install_onnx_stub(monkeypatch)
    nodes, initializers = _export_train_surrogate(
        train_mod, tmp_path, capture, model, scaler, y_scaler
    )

    weights = summary_mod.extract_weights(initializers)
    assert weights.scale_in is not None

    args = argparse.Namespace(std=None, schema=None, metrics=None)
    recovered, source = summary_mod.resolve_feature_std(
        args, weights, weights.n_features
    )
    assert "scale_in" in source
    np.testing.assert_allclose(
        recovered, np.asarray(scaler.scale_, dtype=np.float64), rtol=1e-5
    )

    # The summary's graph-unit conversion (per_std * scale_out) must also
    # use the exported output std.
    np.testing.assert_allclose(
        np.asarray(weights.scale_out, dtype=np.float64).ravel(),
        np.asarray(y_scaler.scale_, dtype=np.float64).ravel(),
        rtol=1e-5,
    )


def test_exported_graph_matches_sklearn_with_onnxruntime(tmp_path, train_mod):
    """End-to-end check with the real ``onnx`` + ``onnxruntime`` packages."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    scaler, y_scaler, x, y = _make_scalers()
    model = _make_model(scaler, y_scaler, x, y)

    config = train_mod.TrainingConfig(
        component=COMPONENT,
        data_dir=tmp_path,
        hidden_layer_sizes=HIDDEN,
    )
    output_path = tmp_path / "surrogate_zone_thermal.onnx"
    train_mod.export_to_onnx(model, scaler, y_scaler, config, output_path)

    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: x.astype(np.float32)})[0]

    expected = _sklearn_prediction(model, scaler, y_scaler, x)
    np.testing.assert_allclose(
        result.ravel(),
        expected,
        rtol=1e-4,
        atol=1e-4 * max(1.0, float(np.max(np.abs(expected)))),
    )
