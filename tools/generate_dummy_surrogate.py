"""Generate a trivial ONNX surrogate model for examples/demo.

This script creates a tiny ONNX model that ignores inputs and outputs a constant vector.
It requires `onnx` and `numpy` to be installed in the target Python environment.

Usage:
    python tools/generate_dummy_surrogate.py --zones 10 \\
        --out examples/dummy_surrogate.onnx
"""

import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper


def build_constant_model(zones: int):
    # Model has a single input (temperatures) of shape [zones] and outputs a constant
    # vector of length `zones` with value 1.2
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, zones])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [zones])

    const_name = "const_loads"
    const_vals = np.array([1.2] * zones, dtype=np.float32)
    const_tensor = helper.make_tensor(
        const_name, TensorProto.FLOAT, [zones], const_vals
    )

    node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output"],
        value=const_tensor,
    )

    graph = helper.make_graph(
        nodes=[node],
        name="dummy_surrogate_graph",
        inputs=[input],
        outputs=[output],
        initializer=[],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 11)],
        ir_version=7,  # Compatible with opset 11
    )
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zones", type=int, default=10)
    p.add_argument("--out", type=str, default="examples/dummy_surrogate.onnx")
    args = p.parse_args()

    model = build_constant_model(args.zones)
    onnx.save(model, args.out)
    print(f"Wrote dummy ONNX surrogate to {args.out}")


if __name__ == "__main__":
    main()
