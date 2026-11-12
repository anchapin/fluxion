#!/usr/bin/env python3
"""Report-only path-weighted first-layer summary of a trained surrogate ONNX.

Issue #4670.

WHAT THIS IS
------------
A small diagnostic that reads the *weights* of an already-trained surrogate
ONNX model (the kind produced by ``scripts/train_surrogate.py::export_to_onnx``:
opset 17, weights stored as ONNX initializers, producer ``fluxion.surrogate``)
and prints a **path-weighted first-layer summary**:

  * the path score ``s1 = W1 @ W2 @ ...`` (hidden layer -> output),
  * the effective input influence ``eff = W0 @ s1`` (input -> output),
  * per-input sign, per-std effect and first-layer norm,
  * the input profiles of the top path-strength hidden units.

WHAT THIS IS NOT
----------------
* It is **report-only**: it never gates, never claims pass/fail, never edits
  weights, and never writes any file (no model, no registry, no sidecar).
  Everything it produces goes to stdout.
* The magnitudes it prints are **linearised (path-weighted) upper estimates,
  not derivatives** -- see the caveat printed with every summary.

THE GATING CAVEAT (printed with every run)
------------------------------------------
* Linearised magnitudes are upper estimates, not derivatives. The measured
  spike was 1.9-4.6x too large, explained by the ignored ReLU gating (a
  hidden unit only contributes when its pre-activation is positive, and the
  linearisation credits it with its full weight regardless).
* Because of that, the **ordering of mid-effect features is unreliable**:
  features with comparable |effect| can swap rank between this summary and
  the true (finite-difference / permutation-importance) response.
* Any behavioural claim drawn from this summary must be verified by finite
  differences or permutation importance first.

Usage
-----
    python3 scripts/surrogate_weight_summary.py models/surrogate_conduction.onnx
    python3 scripts/surrogate_weight_summary.py model.onnx \
        --metrics models/surrogate_conduction_metrics.json
    python3 scripts/surrogate_weight_summary.py model.onnx \
        --component conduction --top-k 3

The input schema (feature names + per-feature standard deviation) is taken
from whatever the training pipeline emits:

  * ``--metrics PATH``  -- the ``surrogate_<component>_metrics.json`` written by
    ``scripts/train_surrogate.py`` (``input_features``, ``scaler_mean``,
    ``scaler_std``);
  * ``--schema PATH``   -- a small JSON with ``input_features`` and
    ``feature_std``;
  * ``--component NAME``-- feature names from ``train_surrogate.COMPONENT_SCHEMAS``;
  * ``--std a,b,c``     -- explicit per-feature standard deviations.

When no standard deviations are supplied they are recovered from the ONNX
``scale_in`` initializer (``std = 1 / scale_in``), which is exactly the
inverse of the input standardisation that ``export_to_onnx`` bakes into the
graph.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Caveat text -- printed verbatim (in substance) with every summary.
# ---------------------------------------------------------------------------

GATING_CAVEAT_LINES = (
    "CAVEAT -- read before quoting any number above:",
    "  * The magnitudes above are LINEARISED (path-weighted) magnitudes: upper",
    "    estimates, NOT derivatives. The measured spike came out 1.9-4.6x too",
    "    large, explained by the ignored ReLU gating: a hidden unit only",
    "    contributes when its pre-activation is positive, while the linearised",
    "    path credits it with its full weight either way.",
    "  * Consequently the ORDERING of mid-effect features is unreliable:",
    "    features with comparable |effect| can swap rank between this summary",
    "    and the true (finite-difference / permutation-importance) response.",
    "  * This script is REPORT-ONLY: it is not a gate, it makes no pass/fail",
    "    claim, it edits nothing, and it writes no file (no model, no",
    "    registry, no sidecar). Verify any behavioural claim with finite",
    "    differences or permutation importance before relying on it.",
)

# ---------------------------------------------------------------------------
# Minimal ONNX reading.
#
# The ``onnx`` package is used when it is importable.  When it is not (a
# lean CI runner, a read-only container), a small self-contained protobuf
# reader is used instead: it extracts the graph initializers (the trained
# weights) and the model metadata directly from the .onnx bytes.  Nothing
# here writes to disk.
# ---------------------------------------------------------------------------

_TENSOR_PROTO_FLOATS = 1
_TENSOR_PROTO_INT64 = 7
_TENSOR_PROTO_INT32 = 6
_TENSOR_PROTO_DOUBLE = 11

_DATA_TYPE_NAMES = {
    1: "float32",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "float64",
}


def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint in ONNX protobuf")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _iter_fields(buf: bytes):
    """Yield ``(field_number, wire_type, payload)`` for a protobuf message."""
    pos = 0
    end = len(buf)
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field_no = key >> 3
        wire = key & 0x07
        if wire == 0:
            value, pos = _read_varint(buf, pos)
            yield field_no, wire, value
        elif wire == 1:
            yield field_no, wire, buf[pos : pos + 8]
            pos += 8
        elif wire == 2:
            length, pos = _read_varint(buf, pos)
            yield field_no, wire, buf[pos : pos + length]
            pos += length
        elif wire == 5:
            yield field_no, wire, buf[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire} in ONNX file")


def _packed_varints(payload: bytes) -> List[int]:
    values: List[int] = []
    pos = 0
    while pos < len(payload):
        value, pos = _read_varint(payload, pos)
        values.append(value)
    return values


def _decode_tensor_proto(payload: bytes) -> Tuple[str, List[int], np.ndarray]:
    """Decode an ONNX ``TensorProto`` into ``(name, dims, array)``."""
    name = ""
    dims: List[int] = []
    data_type = 0
    float_data: List[float] = []
    double_data: List[float] = []
    int32_data: List[int] = []
    int64_data: List[int] = []
    raw_data: Optional[bytes] = None

    for field_no, wire, value in _iter_fields(payload):
        if field_no == 1:  # dims
            if wire == 0:
                dims.append(int(value))
            else:
                dims.extend(int(d) for d in _packed_varints(value))
        elif field_no == 2:  # data_type
            data_type = int(value)
        elif field_no == 4:  # float_data
            if wire == 5:
                float_data.extend(struct.unpack("<f", value))
            else:
                float_data.extend(struct.unpack(f"<{len(value) // 4}f", value))
        elif field_no == 5:  # int32_data
            if wire == 0:
                int32_data.append(int(value))
            else:
                int32_data.extend(int(v) for v in _packed_varints(value))
        elif field_no == 7:  # int64_data
            if wire == 0:
                int64_data.append(int(value))
            else:
                int64_data.extend(int(v) for v in _packed_varints(value))
        elif field_no == 8:  # name
            name = value.decode("utf-8")
        elif field_no == 9:  # raw_data
            raw_data = bytes(value)
        elif field_no == 10:  # double_data
            if wire == 1:
                double_data.extend(struct.unpack("<d", value))
            else:
                double_data.extend(struct.unpack(f"<{len(value) // 8}d", value))

    if raw_data is not None:
        if data_type == _TENSOR_PROTO_FLOATS:
            array = np.frombuffer(raw_data, dtype=np.float32).copy()
        elif data_type == _TENSOR_PROTO_DOUBLE:
            array = np.frombuffer(raw_data, dtype=np.float64).copy()
        elif data_type == _TENSOR_PROTO_INT64:
            array = np.frombuffer(raw_data, dtype=np.int64).copy()
        elif data_type == _TENSOR_PROTO_INT32:
            array = np.frombuffer(raw_data, dtype=np.int32).copy()
        else:
            raise ValueError(
                f"unsupported tensor data_type {data_type} for initializer {name!r}"
            )
    elif data_type == _TENSOR_PROTO_FLOATS:
        array = np.asarray(float_data, dtype=np.float32)
    elif data_type == _TENSOR_PROTO_DOUBLE:
        array = np.asarray(double_data, dtype=np.float64)
    elif data_type == _TENSOR_PROTO_INT64:
        array = np.asarray(int64_data, dtype=np.int64)
    elif data_type == _TENSOR_PROTO_INT32:
        array = np.asarray(int32_data, dtype=np.int32)
    else:
        raise ValueError(
            f"unsupported tensor data_type {data_type} "
            f"({_DATA_TYPE_NAMES.get(data_type, 'unknown')}) for initializer {name!r}"
        )

    if dims:
        array = array.reshape(tuple(dims))

    return name, dims, array


@dataclass
class OnnxWeights:
    """Initializers + metadata extracted from a surrogate ONNX file."""

    initializers: Dict[str, np.ndarray] = field(default_factory=dict)
    producer_name: str = ""
    producer_version: str = ""
    ir_version: int = 0
    opset_version: int = 0
    reader: str = ""


def read_onnx_with_onnx_package(path: Path) -> OnnxWeights:
    """Read initializers with the ``onnx`` package."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    initializers = {
        init.name: numpy_helper.to_array(init) for init in model.graph.initializer
    }
    opset = 0
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            opset = int(entry.version)
    return OnnxWeights(
        initializers=initializers,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        ir_version=int(model.ir_version),
        opset_version=opset,
        reader="onnx",
    )


def read_onnx_without_onnx_package(path: Path) -> OnnxWeights:
    """Read initializers with the built-in minimal protobuf reader."""
    data = path.read_bytes()

    initializers: Dict[str, np.ndarray] = {}
    producer_name = ""
    producer_version = ""
    ir_version = 0
    opset_version = 0

    for field_no, wire, value in _iter_fields(data):
        if field_no == 1 and wire == 0:  # ModelProto.ir_version
            ir_version = int(value)
        elif field_no == 2 and wire == 2:  # ModelProto.producer_name
            producer_name = value.decode("utf-8")
        elif field_no == 3 and wire == 2:  # ModelProto.producer_version
            producer_version = value.decode("utf-8")
        elif field_no == 7 and wire == 2:  # ModelProto.graph
            for g_field, g_wire, g_value in _iter_fields(value):
                if g_field == 5 and g_wire == 2:  # GraphProto.initializer
                    name, _dims, array = _decode_tensor_proto(g_value)
                    initializers[name] = array
        elif field_no == 8 and wire == 2:  # ModelProto.opset_import
            domain = ""
            version = 0
            for o_field, o_wire, o_value in _iter_fields(value):
                if o_field == 1 and o_wire == 2:
                    domain = o_value.decode("utf-8")
                elif o_field == 2 and o_wire == 0:
                    version = int(o_value)
            if domain in ("", "ai.onnx"):
                opset_version = version

    if not initializers:
        raise ValueError(
            f"no initializers found in {path} -- is this a fluxion surrogate ONNX?"
        )

    return OnnxWeights(
        initializers=initializers,
        producer_name=producer_name,
        producer_version=producer_version,
        ir_version=ir_version,
        opset_version=opset_version,
        reader="builtin-protobuf",
    )


def read_onnx_weights(path: Path) -> OnnxWeights:
    """Read initializers + metadata from a surrogate ONNX file."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        return read_onnx_without_onnx_package(path)
    return read_onnx_with_onnx_package(path)


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------


@dataclass
class SurrogateWeights:
    """The trained matrices of a surrogate.

    Layout follows the exported graph: ``W_layer{i}`` has shape
    ``(n_in_i, n_out_i)`` (the scikit-learn ``coefs_`` layout, i.e. the
    forward pass is ``x @ W + b``), and ``b_layer{i}`` has shape
    ``(n_out_i,)``.
    """

    weights: List[np.ndarray]  # coefs_[i]: shape (n_in_i, n_out_i)
    biases: List[np.ndarray]  # intercepts_[i]: shape (n_out_i,)
    scale_in: Optional[np.ndarray] = None
    bias_in: Optional[np.ndarray] = None
    scale_out: Optional[np.ndarray] = None
    bias_out: Optional[np.ndarray] = None

    @property
    def n_features(self) -> int:
        return int(self.weights[0].shape[0])

    @property
    def n_outputs(self) -> int:
        return int(self.weights[-1].shape[1])


def _sorted_weight_names(initializers: Dict[str, np.ndarray]) -> List[str]:
    def key(name: str) -> Tuple[int, str]:
        if name.startswith("W_layer"):
            return (int(name[len("W_layer") :]), name)
        return (10_000, name)

    return sorted(initializers.keys(), key=key)


def extract_weights(initializers: Dict[str, np.ndarray]) -> SurrogateWeights:
    """Pull the ``W_layer*`` / ``b_layer*`` initializers out of an ONNX graph."""
    weight_names = [n for n in _sorted_weight_names(initializers) if n.startswith("W_layer")]
    if not weight_names:
        raise ValueError(
            "no 'W_layer*' initializers found -- this does not look like a "
            "surrogate exported by scripts/train_surrogate.py"
        )

    weights: List[np.ndarray] = []
    biases: List[np.ndarray] = []
    for name in weight_names:
        idx = name[len("W_layer") :]
        weights.append(np.asarray(initializers[name], dtype=np.float64))
        bias_name = f"b_layer{idx}"
        if bias_name in initializers:
            biases.append(np.asarray(initializers[bias_name], dtype=np.float64))
        else:
            biases.append(np.zeros(weights[-1].shape[1], dtype=np.float64))

    def _opt(name: str) -> Optional[np.ndarray]:
        if name in initializers:
            return np.asarray(initializers[name], dtype=np.float64)
        return None

    return SurrogateWeights(
        weights=weights,
        biases=biases,
        scale_in=_opt("scale_in"),
        bias_in=_opt("bias_in"),
        scale_out=_opt("scale_out"),
        bias_out=_opt("bias_out"),
    )


# ---------------------------------------------------------------------------
# The path-weighted summary
# ---------------------------------------------------------------------------


@dataclass
class FeatureEffect:
    """Per-input summary row."""

    name: str
    std: float
    sign: int  # +1 / -1 / 0 -- sign of the effective influence
    per_std_effect: float  # output change per +1 std, model (scaled) output units
    per_std_effect_graph: float  # same, expressed in the ONNX graph's output units
    per_raw_unit_effect: float  # output change per +1 raw unit, model output units
    first_layer_norm_std: float  # ||W0[j, :]||_2 (standardised input space)
    first_layer_norm_raw: float  # ||W0[j, :]||_2 / std (raw input space)


@dataclass
class HiddenUnitProfile:
    """Input profile of one path-strength hidden unit."""

    unit: int
    path_strength: float  # ||W_last[:, unit]||_2 -- strength of the path to output
    inputs: List[Tuple[str, float]]  # (feature name, contribution per +1 std)


@dataclass
class WeightSummary:
    """Everything the report prints."""

    n_features: int
    n_outputs: int
    layer_shapes: List[Tuple[int, int]]
    effects: List[FeatureEffect]
    top_units: List[HiddenUnitProfile]
    std_source: str
    feature_source: str


def compute_summary(
    weights: SurrogateWeights,
    feature_names: Sequence[str],
    feature_std: Sequence[float],
    top_k: int = 3,
    std_source: str = "caller",
    feature_source: str = "caller",
) -> WeightSummary:
    """Compute the path-weighted first-layer summary.

    Conventions (exported-ONNX / scikit-learn layout, ``W_i`` has shape
    ``(n_in_i, n_out_i)``, forward pass ``x @ W_i + b_i``):

      * path score              ``s1  = W1 @ W2 @ ...`` (hidden -> output)
      * effective input influence ``eff = W0 @ s1``       (input  -> output)

    The path score is the chain product over the hidden weight matrices
    ``weights[1:]``: with a single hidden layer (two matrices) it is ``W1``
    alone; with two or more hidden layers it is ``W1 @ W2 @ ...``.

    ``eff`` is expressed in the *training-standardised* input units: the output
    change caused by a +1 standard-deviation move of that input when the ReLU gating
    is ignored.  The raw-unit effect divides by the feature std.
    """
    if len(feature_names) != weights.n_features:
        raise ValueError(
            f"feature name count {len(feature_names)} does not match the model "
            f"input width {weights.n_features}"
        )
    if len(feature_std) != weights.n_features:
        raise ValueError(
            f"feature std count {len(feature_std)} does not match the model "
            f"input width {weights.n_features}"
        )

    w0 = weights.weights[0]
    if len(weights.weights) == 1:
        # No hidden layer: the "path" is the single weight matrix.
        path_score = np.eye(weights.n_outputs)
        eff = w0 @ path_score
        last_hidden = None
    else:
        # The path score is the chain product over the *hidden* weights
        # weights[1:], i.e. the hidden->output path:
        #
        #   2 matrices (one hidden layer): s1 = W1              (n_hidden, n_outputs)
        #   3+ matrices:                   s1 = W1 @ W2 @ ...  (n_hidden, n_outputs)
        #
        # For the two-matrix case W1 *is* the last weight matrix, so the
        # chain product is W1 alone -- computing W1 @ W1 would either raise
        # (n_hidden != n_outputs) or silently return nonsense
        # (n_hidden == n_outputs).
        path_score = weights.weights[1]
        for w in weights.weights[2:]:
            path_score = path_score @ w
        # eff = W0 @ s1 : (n_features, n_outputs)
        eff = w0 @ path_score
        # The last hidden layer is the one feeding the output layer; its
        # weight matrix is weights[-1] in every case (for a single hidden
        # layer that is W1 itself), and its rows index the hidden units whose
        # input profile is computed below.
        last_hidden = weights.weights[-1]

    std_arr = np.asarray(feature_std, dtype=np.float64).reshape(-1)
    std_arr = np.where(std_arr > 0, std_arr, 1.0)

    # The exported graph emits scaled_output * scale_out (per output channel);
    # without a scale_out initializer the graph output IS the model output.
    scale_out_arr: Optional[np.ndarray] = None
    if weights.scale_out is not None:
        scale_out_arr = np.asarray(weights.scale_out, dtype=np.float64).reshape(-1)

    effects: List[FeatureEffect] = []
    for j, name in enumerate(feature_names):
        eff_j = eff[j]
        # Effective sign is taken from the dominant output channel.
        dominant = int(np.argmax(np.abs(eff_j)))
        value = float(eff_j[dominant])
        sign = int(np.sign(value))
        per_std = value
        if scale_out_arr is None:
            per_std_graph = per_std
        elif scale_out_arr.size == 1:
            per_std_graph = per_std * float(scale_out_arr[0])
        elif scale_out_arr.size > dominant:
            per_std_graph = per_std * float(scale_out_arr[dominant])
        else:
            per_std_graph = per_std
        # The weights act on the *training-standardised* input, so a raw-unit
        # effect divides by the feature std.
        per_raw = per_std / float(std_arr[j])
        # W0 rows are indexed by input feature (export layout: x @ W0), so the
        # first-layer norm for feature j is the ROW norm ||W0[j, :]||.
        col = w0[j, :]
        norm_std = float(np.linalg.norm(col))
        norm_raw = norm_std / float(std_arr[j])
        effects.append(
            FeatureEffect(
                name=name,
                std=float(std_arr[j]),
                sign=sign,
                per_std_effect=per_std,
                per_std_effect_graph=per_std_graph,
                per_raw_unit_effect=per_raw,
                first_layer_norm_std=norm_std,
                first_layer_norm_raw=norm_raw,
            )
        )

    # Top path-strength hidden units of the last hidden layer, with the
    # per-input profile that feeds them.
    top_units: List[HiddenUnitProfile] = []
    if last_hidden is not None:
        # last_hidden: (n_hidden, n_outputs) -- path strength of unit k is
        # the norm of its row towards the output.
        strengths = np.linalg.norm(last_hidden, axis=1)
        order = np.argsort(-strengths)[: max(0, top_k)]
        # Contribution of each input to each last-hidden unit, per +1 std
        # (i.e. ignoring ReLU gating -- see the caveat).
        feed = w0
        for w in weights.weights[1:-1]:
            feed = feed @ w
        for unit in order:
            profile = feed[:, int(unit)]
            ranked = sorted(
                ((feature_names[i], float(profile[i])) for i in range(len(profile))),
                key=lambda kv: -abs(kv[1]),
            )
            top_units.append(
                HiddenUnitProfile(
                    unit=int(unit),
                    path_strength=float(strengths[int(unit)]),
                    inputs=ranked[: min(4, len(ranked))],
                )
            )

    layer_shapes = [(int(w.shape[0]), int(w.shape[1])) for w in weights.weights]

    return WeightSummary(
        n_features=weights.n_features,
        n_outputs=weights.n_outputs,
        layer_shapes=layer_shapes,
        effects=effects,
        top_units=top_units,
        std_source=std_source,
        feature_source=feature_source,
    )


# ---------------------------------------------------------------------------
# Schema resolution (feature names + per-feature std)
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_feature_names(
    args: argparse.Namespace, n_features: int
) -> Tuple[List[str], str]:
    """Resolve the feature names from whatever the caller supplied."""
    if args.schema:
        data = _load_json(Path(args.schema))
        names = list(data.get("input_features") or data.get("features") or [])
        if names:
            return names, f"schema:{args.schema}"
    if args.metrics:
        data = _load_json(Path(args.metrics))
        names = list(data.get("input_features") or [])
        if names:
            return names, f"metrics:{args.metrics}"
    if args.component:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from train_surrogate import COMPONENT_SCHEMAS  # type: ignore

            names = list(COMPONENT_SCHEMAS[args.component]["input_features"])
        except (ImportError, KeyError) as exc:
            raise SystemExit(f"cannot resolve features for component: {exc}")
        return names, f"train_surrogate.COMPONENT_SCHEMAS[{args.component}]"
    return [f"feature_{i}" for i in range(n_features)], "positional (no schema given)"


def resolve_feature_std(
    args: argparse.Namespace, weights: SurrogateWeights, n_features: int
) -> Tuple[List[float], str]:
    """Resolve the per-feature standard deviations."""
    if args.std:
        values = [float(v) for v in str(args.std).split(",") if v.strip()]
        if len(values) != n_features:
            raise SystemExit(
                f"--std has {len(values)} entries but the model takes {n_features} inputs"
            )
        return values, "caller (--std)"
    if args.schema:
        data = _load_json(Path(args.schema))
        values = data.get("feature_std") or data.get("scaler_std")
        if values:
            return [float(v) for v in values], f"schema:{args.schema}"
    if args.metrics:
        data = _load_json(Path(args.metrics))
        values = data.get("scaler_std")
        if values:
            return [float(v) for v in values], f"metrics:{args.metrics}"
    scale_in = weights.scale_in
    if scale_in is not None:
        scale_in = np.asarray(scale_in, dtype=np.float64).reshape(-1)
        if scale_in.shape[0] == n_features and np.all(scale_in > 0):
            return [float(1.0 / s) for s in scale_in], "onnx:scale_in (1/scale_in)"
    raise SystemExit(
        "no per-feature standard deviations available: pass --metrics, --schema "
        "or --std (the ONNX file carries no usable scale_in initializer)"
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value: float, width: int = 12, digits: int = 6) -> str:
    return f"{value:>{width}.{digits}e}"


def render_summary(
    summary: WeightSummary,
    weights: SurrogateWeights,
    meta: OnnxWeights,
    model_path: Path,
) -> str:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("SURROGATE PATH-WEIGHTED FIRST-LAYER SUMMARY (report-only)")
    lines.append("=" * 78)
    lines.append(f"model            : {model_path}")
    lines.append(
        f"producer         : {meta.producer_name or 'unknown'} "
        f"{meta.producer_version or ''}".rstrip()
    )
    lines.append(
        f"opset / ir       : {meta.opset_version or '?'} / {meta.ir_version or '?'}"
    )
    lines.append(f"weights read via : {meta.reader}")
    lines.append(
        "layer shapes     : "
        + " -> ".join(f"{out}x{inp}" for out, inp in summary.layer_shapes)
    )
    lines.append(
        f"feature names    : {summary.feature_source}"
    )
    lines.append(
        f"feature std      : {summary.std_source}"
    )
    scale_in = weights.scale_in
    if scale_in is not None:
        scale_in = np.asarray(scale_in, dtype=np.float64).reshape(-1)
        stds = np.asarray(
            [eff.std for eff in summary.effects], dtype=np.float64
        )
        expected = np.where(stds > 0, 1.0 / np.where(stds > 0, stds, 1.0), 1.0)
        lines.append(
            "graph input scal : scale_in="
            + ", ".join(f"{v:.4g}" for v in scale_in)
            + "  (the exported graph maps x -> x*scale_in + bias_in)"
        )
        if scale_in.shape == expected.shape and not np.allclose(scale_in, expected):
            lines.append(
                "                   NOTE: scale_in differs from 1/std, so the exported"
            )
            lines.append(
                "                   graph's input space differs from the training"
            )
            lines.append(
                "                   standardisation; the effects above are in the"
            )
            lines.append(
                "                   training-standardised space (per +1 std)."
            )
    lines.append("")
    lines.append("Per-input effective influence (ReLU gating ignored -- see caveat):")
    lines.append(
        f"  {'feature':<24}{'sign':>5}{'per +1 std':>16}{'per +1 raw unit':>18}"
        f"{'||W0[j,:]|| (std)':>18}{'||W0[j,:]||/std (raw)':>21}"
    )
    for eff in summary.effects:
        sign_text = {1: "+", -1: "-", 0: "0"}[eff.sign]
        lines.append(
            f"  {eff.name:<24}{sign_text:>5}"
            f"{_fmt(eff.per_std_effect, 16)}"
            f"{_fmt(eff.per_raw_unit_effect, 18)}"
            f"{_fmt(eff.first_layer_norm_std, 18)}"
            f"{_fmt(eff.first_layer_norm_raw, 21)}"
        )
    lines.append("")
    lines.append(
        "Top path-strength hidden units (path strength = ||W_last[:, unit]||) "
        "and their input profiles:"
    )
    if not summary.top_units:
        lines.append("  (none -- the model has no hidden layer)")
    for unit in summary.top_units:
        lines.append(
            f"  unit {unit.unit:<4} path_strength={_fmt(unit.path_strength, 12)}"
        )
        for name, value in unit.inputs:
            lines.append(
                f"      {name:<24}contribution per +1 std = {_fmt(value, 14)}"
            )
    lines.append("")
    for line in GATING_CAVEAT_LINES:
        lines.append(line)
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print a report-only path-weighted first-layer summary of a trained "
            "surrogate ONNX model. Writes nothing."
        )
    )
    parser.add_argument("model", help="path to the trained surrogate .onnx file")
    parser.add_argument(
        "--metrics",
        help=(
            "training metrics JSON written by scripts/train_surrogate.py "
            "(input_features, scaler_std)"
        ),
    )
    parser.add_argument(
        "--schema",
        help="small JSON with 'input_features' and 'feature_std'",
    )
    parser.add_argument(
        "--component",
        help="component name (feature names from train_surrogate.COMPONENT_SCHEMAS)",
    )
    parser.add_argument(
        "--std",
        help="comma-separated per-feature standard deviations",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="how many top path-strength hidden units to profile (default: 3)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"error: model not found: {model_path}", file=sys.stderr)
        return 2

    meta = read_onnx_weights(model_path)
    weights = extract_weights(meta.initializers)

    names, name_source = resolve_feature_names(args, weights.n_features)
    stds, std_source = resolve_feature_std(args, weights, weights.n_features)

    summary = compute_summary(
        weights,
        names,
        stds,
        top_k=args.top_k,
        std_source=std_source,
        feature_source=name_source,
    )
    print(render_summary(summary, weights, meta, model_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
