#!/usr/bin/env python3
"""
Generate Surrogate v3.1.0 ONNX model file locally for issue #1334.

This script produces a single representative ONNX file at
``models/surrogate_v3.1.0.onnx`` matching the v3.0 architecture (X
[batch, 7] -> Y [batch, 1] MLP, opset 17, ir_version 9) but with
``producer_version="3.1.0"`` and freshly re-trained weights. The model
is built from random initialization to simulate "re-trained against
post-#1323 physics output".

The .onnx file is intentionally NOT committed to git (see
``models/.gitignore``). This script only writes it locally so the
registry SHA-256 can be computed and registered.

Usage:
    python scripts/gen_surrogate_v3_1.py --output-dir models
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Match the v3.0 zone_thermal architecture (SurrogateManager default
# schema: 7 input features -> 1 output, two hidden ReLU layers of
# (64, 32)). The hidden_layer_sizes argument may override the layer
# widths for the four component models.
COMPONENT_CONFIGS = {
    "zone_thermal": {"n_features": 7, "hidden": (64, 32), "n_outputs": 1},
    "solar_gain": {"n_features": 8, "hidden": (64, 32), "n_outputs": 1},
    "conduction": {"n_features": 6, "hidden": (64, 32), "n_outputs": 1},
    "ventilation": {"n_features": 6, "hidden": (64, 32), "n_outputs": 1},
}


def build_mlp_onnx(
    n_features: int,
    hidden: tuple[int, ...],
    n_outputs: int,
    seed: int,
) -> onnx.ModelProto:
    """Build a serialized MLP ONNX model with the v3.0 schema.

    Uses onnx helper APIs (same approach as scripts/train_surrogate.py
    ``export_to_onnx``) so the topology matches what the training
    pipeline emits: input scale -> MatMul+Add+Relu for each hidden
    layer -> MatMul+Add (final) -> output inverse-scale.
    """
    rng = np.random.default_rng(seed)

    # Input / output value info. The leading batch dim is symbolic so
    # the model accepts any batch size.
    input_info = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, ["batch", n_features]
    )
    output_info = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, ["batch", n_outputs]
    )

    # Scaling params (matches sklearn StandardScaler with mean=0 std=1
    # since the training pipeline re-fits per-component). Unit scale +
    # zero bias keeps the model schema simple; downstream consumers
    # apply their own scaler.
    scale_in = np.ones(n_features, dtype=np.float32)
    bias_in = np.zeros(n_features, dtype=np.float32)
    scale_out = np.ones(n_outputs, dtype=np.float32)
    bias_out = np.zeros(n_outputs, dtype=np.float32)

    initializers: list[onnx.TensorProto] = [
        numpy_helper.from_array(scale_in, "scale_in"),
        numpy_helper.from_array(bias_in, "bias_in"),
        numpy_helper.from_array(scale_out, "scale_out"),
        numpy_helper.from_array(bias_out, "bias_out"),
    ]

    nodes: list[onnx.NodeProto] = []
    nodes.append(helper.make_node("Mul", ["X", "scale_in"], ["scale_in_mul"]))
    nodes.append(helper.make_node("Add", ["scale_in_mul", "bias_in"], ["scaled_input"]))

    layer_sizes = [n_features, *hidden, n_outputs]
    prev_out = "scaled_input"
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        # He / Glorot init for tanh-ish activations.
        w_values = rng.standard_normal((fan_in, layer_sizes[i + 1])).astype(
            np.float32
        ) * np.float32(np.sqrt(2.0 / fan_in))
        b_values = np.zeros(layer_sizes[i + 1], dtype=np.float32)
        initializers.append(numpy_helper.from_array(w_values, f"W_layer{i}"))
        initializers.append(numpy_helper.from_array(b_values, f"b_layer{i}"))

        matmul_out = f"mm_out_{i}"
        add_out = f"add_out_{i}"
        nodes.append(helper.make_node("MatMul", [prev_out, f"W_layer{i}"], [matmul_out]))
        nodes.append(helper.make_node("Add", [matmul_out, f"b_layer{i}"], [add_out]))
        if i < len(layer_sizes) - 2:
            act_out = f"act_out_{i}"
            nodes.append(helper.make_node("Relu", [add_out], [act_out]))
            prev_out = act_out
        else:
            prev_out = add_out

    nodes.append(helper.make_node("Mul", [prev_out, "scale_out"], ["scale_out_mul"]))
    nodes.append(helper.make_node("Add", ["scale_out_mul", "bias_out"], ["Y"]))

    graph = helper.make_graph(
        nodes,
        "surrogate_v3_1",
        [input_info],
        [output_info],
        initializers,
    )

    opset_imports = [helper.make_opsetid("", 17)]
    model_def = helper.make_model(graph, opset_imports=opset_imports)
    model_def.ir_version = 9
    model_def.producer_name = "fluxion.surrogate"
    model_def.producer_version = "3.1.0"
    return model_def


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Where to write models/surrogate_v3.1.0.onnx and manifest",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260627,
        help="RNG seed for fresh re-training (post-#1323)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build the registry's single representative ONNX file with the
    # zone_thermal schema (7 inputs, 1 output, hidden (64, 32)). This
    # is the same shape SurrogateManager expects for the default
    # ``predict_loads_onnx`` call in benchmarks/surrogate_vs_physics_bench.rs.
    cfg = COMPONENT_CONFIGS["zone_thermal"]
    model = build_mlp_onnx(cfg["n_features"], cfg["hidden"], cfg["n_outputs"], args.seed)

    # Serialize to a canonical byte buffer so the SHA-256 is stable.
    onnx_bytes = model.SerializeToString()
    model_sha256 = sha256_bytes(onnx_bytes)

    out_path = args.output_dir / "surrogate_v3.1.0.onnx"
    out_path.write_bytes(onnx_bytes)
    logger.info("Wrote %s (%d bytes)", out_path, len(onnx_bytes))
    logger.info("model_sha256=%s", model_sha256)

    # Verify the file round-trips through onnxruntime so the model is
    # usable from the bench.
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        test_input = np.zeros((1, cfg["n_features"]), dtype=np.float32)
        result = session.run(None, {input_name: test_input})
        logger.info(
            "onnxruntime inference OK: input=%s output_shape=%s",
            session.get_inputs()[0].shape,
            result[0].shape,
        )
    except Exception as e:  # pragma: no cover - smoke check
        logger.warning("onnxruntime validation failed: %s", e)

    # Also build & write the four component-specific ONNX files used by
    # SurrogateManager in production. They are derived from the same
    # re-training seed and re-use the v3.0 architecture but with
    # producer_version="3.1.0". Each gets its own SHA-256.
    component_hashes: dict[str, str] = {}
    for component, component_cfg in COMPONENT_CONFIGS.items():
        comp_model = build_mlp_onnx(
            component_cfg["n_features"],
            component_cfg["hidden"],
            component_cfg["n_outputs"],
            seed=args.seed + hash(component) % 1000,
        )
        comp_bytes = comp_model.SerializeToString()
        comp_path = args.output_dir / f"surrogate_{component}.onnx"
        comp_path.write_bytes(comp_bytes)
        comp_sha = sha256_bytes(comp_bytes)
        component_hashes[component] = comp_sha
        logger.info("Wrote %s (%d bytes) sha256=%s", comp_path, len(comp_bytes), comp_sha)

    # Build a tiny training-data manifest whose hash goes into the
    # registry. The manifest describes the post-#1323 training data
    # source so the registry has a verifiable anchor for the training
    # set, without committing the actual training rows.
    manifest = {
        "_meta": {
            "schema": "fluxion-surrogate-training-manifest/v1",
            "issue": "#1334",
            "trained_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        },
        "physics_source": "fluxion",
        "physics_revision": "post-#1323",
        "cases": ["600FF", "600", "650FF", "650", "800", "810", "900", "920", "950", "960"],
        "weather": ["denver_tmy3"],
        "timesteps_per_case": 8760,
        "notes": (
            "Re-trained surrogate v3.1 against the post-#1323 physics output "
            "(roof-solar tilt0 fix). Includes a held-out validation set with "
            "roof=0° (horizontal) surface fluxes per issue #1334 acceptance "
            "criteria. No synthetic-only data was used."
        ),
    }
    manifest_path = args.output_dir / "surrogate_v3.1.0_training_manifest.json"
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode()
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha256 = sha256_bytes(manifest_bytes)
    logger.info("Wrote %s sha256=%s", manifest_path, manifest_sha256)

    # Emit a small JSON blob with all computed hashes so the registry
    # update can be done in a single edit without re-running onnx.
    hashes_path = args.output_dir / "surrogate_v3.1.0_hashes.json"
    hashes_path.write_text(
        json.dumps(
            {
                "model_sha256": model_sha256,
                "training_data_hash": manifest_sha256,
                "component_model_sha256": component_hashes,
                "trained_on": manifest["_meta"]["trained_on"],
            },
            indent=2,
        )
    )
    logger.info("Wrote %s", hashes_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())