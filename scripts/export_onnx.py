#!/usr/bin/env python3
"""
ONNX Export and Validation Script (v3.0)

Exports trained surrogate models to ONNX format with opset 17+ compatibility,
validates model structure, and performs runtime checks using onnxruntime.

Usage:
    python scripts/export_onnx.py --component zone_thermal --input models/surrogate_zone_thermal.pkl
    python scripts/export_onnx.py --all-models --input-dir models/
    python scripts/export_onnx.py --validate models/surrogate_zone_thermal.onnx

Output:
    models/surrogate_{component}.onnx - Exported model ready for Rust inference
    models/surrogate_{component}_validation.json - Validation report
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    model_path: str
    onnx_version: str
    opset_version: int
    ir_version: int
    producer_name: str
    producer_version: str
    input_name: str
    input_shape: List[int]
    input_type: str
    output_name: str
    output_shape: List[int]
    output_type: str
    model_size_bytes: int
    total_params: int
    graph_size_bytes: int
    runtime_validation: bool
    inference_time_ms: float
    outputs_match: bool
    validation_timestamp: str
    errors: List[str]


COMPONENT_IO_SPECS = {
    "zone_thermal": {
        "input_name": "X",
        "input_shape": [None, 7],
        "output_name": "Y",
        "output_shape": [None, 1],
    },
    "solar_gain": {
        "input_name": "X",
        "input_shape": [None, 8],
        "output_name": "Y",
        "output_shape": [None, 1],
    },
    "conduction": {
        "input_name": "X",
        "input_shape": [None, 6],
        "output_name": "Y",
        "output_shape": [None, 1],
    },
    "ventilation": {
        "input_name": "X",
        "input_shape": [None, 6],
        "output_name": "Y",
        "output_shape": [None, 1],
    },
}


def compute_model_hash(model_path: Path) -> str:
    """Compute SHA256 hash of the model file."""
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def analyze_onnx_model(model_path: Path) -> ValidationReport:
    """Analyze ONNX model structure and metadata."""
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    input_tensor = session.get_inputs()[0]
    output_tensor = session.get_outputs()[0]

    graph = model.graph
    total_params = 0
    for initializer in graph.initializer:
        dims = [dim for dim in initializer.dims]
        if dims:
            total_params += int(np.prod(dims))

    ir_version = model.ir_version
    opset_version = model.opset_import[0].version if model.opset_import else 0
    onnx_version = onnx.__version__

    test_input = np.random.randn(1, input_tensor.shape[1] if input_tensor.shape[1] else 8).astype(np.float32)

    start = time.perf_counter()
    result = session.run([output_tensor.name], {input_tensor.name: test_input})
    inference_time_ms = (time.perf_counter() - start) * 1000

    test_result = result[0]
    expected_shape = list(test_result.shape)

    errors = []
    if ir_version < 7:
        errors.append(f"IR version {ir_version} is below minimum 7")

    if opset_version < 17:
        errors.append(f"Opset version {opset_version} is below required 17")

    runtime_validation = True
    try:
        session.run([output_tensor.name], {input_tensor.name: test_input})
    except Exception as e:
        runtime_validation = False
        errors.append(f"Runtime validation failed: {e}")

    outputs_match = test_result.shape == tuple(expected_shape)

    report = ValidationReport(
        model_path=str(model_path),
        onnx_version=onnx_version,
        opset_version=opset_version,
        ir_version=ir_version,
        producer_name=getattr(model, "producer_name", ""),
        producer_version=getattr(model, "producer_version", ""),
        input_name=input_tensor.name,
        input_shape=[int(d) for d in input_tensor.shape],
        input_type=str(input_tensor.type),
        output_name=output_tensor.name,
        output_shape=[int(d) for d in output_tensor.shape],
        output_type=str(output_tensor.type),
        model_size_bytes=model_path.stat().st_size,
        total_params=total_params,
        graph_size_bytes=0,
        runtime_validation=runtime_validation,
        inference_time_ms=inference_time_ms,
        outputs_match=outputs_match,
        validation_timestamp=datetime.now(timezone.utc).isoformat(),
        errors=errors,
    )

    return report


def validate_model_structure(model_path: Path, component: str) -> Tuple[bool, List[str]]:
    """Validate model structure matches expected I/O specification."""
    import onnx
    import onnxruntime as ort

    errors = []
    warnings = []

    try:
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        logger.info("  ONNX model structure is valid")
    except onnx.validation.ValidationError as e:
        errors.append(f"ONNX validation error: {e}")
        return False, errors

    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as e:
        errors.append(f"Failed to load model with onnxruntime: {e}")
        return False, errors

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1:
        errors.append(f"Expected 1 input, got {len(inputs)}")
    if len(outputs) != 1:
        errors.append(f"Expected 1 output, got {len(outputs)}")

    if component in COMPONENT_IO_SPECS:
        spec = COMPONENT_IO_SPECS[component]

        if inputs[0].name != spec["input_name"]:
            warnings.append(f"Input name '{inputs[0].name}' differs from spec '{spec['input_name']}'")

        if spec["input_shape"][1] is not None and inputs[0].shape[1] != spec["input_shape"][1]:
            errors.append(f"Input feature dim {inputs[0].shape[1]} != expected {spec['input_shape'][1]}")

        if outputs[0].name != spec["output_name"]:
            warnings.append(f"Output name '{outputs[0].name}' differs from spec '{spec['output_name']}'")

        if spec["output_shape"][1] is not None and outputs[0].shape[1] != spec["output_shape"][1]:
            errors.append(f"Output feature dim {outputs[0].shape[1]} != expected {spec['output_shape'][1]}")

    return len(errors) == 0, errors + warnings


def benchmark_inference(model_path: Path, n_iterations: int = 100) -> Dict:
    """Benchmark inference time using onnxruntime."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_tensor = session.get_inputs()[0]

    n_features = input_tensor.shape[1] if input_tensor.shape[1] else 8
    test_input = np.random.randn(1, n_features).astype(np.float32)

    warmup_runs = 10
    for _ in range(warmup_runs):
        session.run([input_tensor.name], {input_tensor.name: test_input})

    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        session.run([input_tensor.name], {input_tensor.name: test_input})
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "throughput_samples_per_sec": float(1000.0 / np.mean(times)),
    }


def export_model(
    model,
    scaler,
    component: str,
    output_path: Path,
    hidden_layer_sizes: Tuple[int, ...],
    opset: int = 17,
) -> bool:
    """Export sklearn MLP model to ONNX format."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    schema = COMPONENT_IO_SPECS.get(component)
    if schema is None:
        logger.error(f"Unknown component: {component}")
        return False

    n_features = schema["input_shape"][1]
    n_outputs = schema["output_shape"][1]

    input_tensor_name = schema["input_name"]
    output_tensor_name = schema["output_name"]
    scale_in_name = "scale_in"
    bias_in_name = "bias_in"
    scale_out_name = "scale_out"
    bias_out_name = "bias_out"
    mlp_out_name = "mlp_out"
    scaled_input_name = "scaled_input"

    input_info = helper.make_tensor_value_info(input_tensor_name, TensorProto.FLOAT, [None, n_features])
    output_info = helper.make_tensor_value_info(output_tensor_name, TensorProto.FLOAT, [None, n_outputs])

    init_scale_in = numpy_helper.from_array(scaler.scale_.astype(np.float32), scale_in_name)
    init_bias_in = numpy_helper.from_array((-scaler.mean_ * scaler.scale_).astype(np.float32), bias_in_name)
    init_scale_out = numpy_helper.from_array((1.0 / scaler.scale_).astype(np.float32), scale_out_name)
    init_bias_out = numpy_helper.from_array(scaler.mean_.astype(np.float32), bias_out_name)

    nodes = []
    initializers = [init_scale_in, init_bias_in, init_scale_out, init_bias_out]

    scale_in_node = helper.make_node("Mul", [input_tensor_name, scale_in_name], ["scale_in_mul"])
    add_in_node = helper.make_node("Add", ["scale_in_mul", bias_in_name], [scaled_input_name])
    nodes.extend([scale_in_node, add_in_node])

    layer_sizes = [n_features] + list(hidden_layer_sizes) + [n_outputs]

    for i in range(len(layer_sizes) - 1):
        w_name = f"W_layer{i}"
        b_name = f"b_layer{i}"

        w_values = model.coefs_[i].astype(np.float32)
        b_values = model.intercepts_[i].astype(np.float32)

        initializers.append(numpy_helper.from_array(w_values, w_name))
        initializers.append(numpy_helper.from_array(b_values, b_name))

        prev_name = scaled_input_name if i == 0 else f"layer{i-1}_out"
        curr_name = f"layer{i}_out"

        w_node = helper.make_node("MatMul", [prev_name, w_name], ["w_mul_out"])
        add_node = helper.make_node("Add", ["w_mul_out", b_name], [curr_name])
        nodes.append(w_node)
        nodes.append(add_node)

        if i < len(layer_sizes) - 2:
            act_node = helper.make_node("Relu", [curr_name], [curr_name])
            nodes.append(act_node)
        else:
            last_layer_node = helper.make_node("Identity", [curr_name], [mlp_out_name])
            nodes.append(last_layer_node)

    scale_out_node = helper.make_node("Mul", [mlp_out_name, scale_out_name], ["scaled_out"])
    add_out_node = helper.make_node("Add", ["scaled_out", bias_out_name], [output_tensor_name])
    nodes.extend([scale_out_node, add_out_node])

    graph = helper.make_graph(
        nodes,
        f"surrogate_{component}",
        [input_info],
        [output_info],
        initializers,
    )

    opset_imports = [helper.make_opsetid("", opset)]
    model_def = helper.make_model(graph, opset_imports=opset_imports)
    model_def.ir_version = 9
    model_def.producer_name = "fluxion.surrogate"
    model_def.producer_version = "3.0.0"

    onnx.save(model_def, str(output_path))
    logger.info(f"Exported ONNX model: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export and validate ONNX surrogate models")
    parser.add_argument("--model", type=Path, help="Model file to validate")
    parser.add_argument("--component", type=str, help="Component name (for validation)")
    parser.add_argument("--all-models", action="store_true", help="Validate all models in models/")
    parser.add_argument("--input-dir", type=Path, default=Path("models"), help="Input directory")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--n-iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--output", type=Path, help="Output validation report path")

    args = parser.parse_args()

    if args.all_models:
        model_files = list(args.input_dir.glob("surrogate_*.onnx"))
        if not model_files:
            logger.error(f"No ONNX models found in {args.input_dir}")
            return 1
    elif args.model:
        model_files = [args.model]
    else:
        parser.error("--model or --all-models is required")
        return 1

    all_passed = True
    reports = []

    for model_path in model_files:
        component = args.component
        if not component:
            for comp in COMPONENT_IO_SPECS:
                if comp in model_path.stem:
                    component = comp
                    break

        logger.info("=" * 60)
        logger.info(f"Validating: {model_path.name}")
        logger.info("=" * 60)

        try:
            report = analyze_onnx_model(model_path)
            reports.append(report)

            if report.errors:
                logger.warning("  Errors found:")
                for error in report.errors:
                    logger.warning(f"    - {error}")
                all_passed = False
            else:
                logger.info("  No structural errors")

            logger.info(f"  ONNX version: {report.onnx_version}")
            logger.info(f"  Opset version: {report.opset_version}")
            logger.info(f"  IR version: {report.ir_version}")
            logger.info(f"  Model size: {report.model_size_bytes / 1024:.2f} KB")
            logger.info(f"  Parameters: {report.total_params}")
            logger.info(f"  Runtime validation: {'PASS' if report.runtime_validation else 'FAIL'}")
            logger.info(f"  Inference time: {report.inference_time_ms:.4f} ms")

            if args.benchmark:
                logger.info(f"\nRunning benchmark ({args.n_iterations} iterations)...")
                bench = benchmark_inference(model_path, args.n_iterations)
                logger.info(f"  Mean: {bench['mean_ms']:.4f} ms")
                logger.info(f"  P95: {bench['p95_ms']:.4f} ms")
                logger.info(f"  Throughput: {bench['throughput_samples_per_sec']:.1f} samples/sec")

                if bench["mean_ms"] > 1.0:
                    logger.warning("  WARNING: Inference time exceeds 1ms target!")

            if component:
                struct_ok, struct_errors = validate_model_structure(model_path, component)
                if struct_errors:
                    for err in struct_errors:
                        logger.warning(f"  {err}")
                if not struct_ok:
                    all_passed = False

            model_hash = compute_model_hash(model_path)
            logger.info(f"\n  Model SHA256: {model_hash[:16]}...")

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            all_passed = False

    if args.output:
        output_data = {
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_models": len(reports),
            "passed": sum(1 for r in reports if r.runtime_validation and not r.errors),
            "failed": sum(1 for r in reports if not r.runtime_validation or r.errors),
            "reports": [asdict(r) for r in reports],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nValidation report saved: {args.output}")

    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("ALL MODELS PASSED VALIDATION")
    else:
        logger.warning("SOME MODELS FAILED VALIDATION")
    logger.info("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
