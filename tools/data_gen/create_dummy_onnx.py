import torch
import torch.nn as nn
import torch.onnx


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

    def forward(self, x):
        # y = x + 10
        return x + 10.0


def generate_model():
    model = DummyModel()
    model.eval()

    # Input to the model (batch_size=1, input_features=2)
    # The test code uses input [20.0, 21.0] which is a slice/vector.
    # The SurrogateManager::predict_loads uses `&[f64]`.
    # If the rust code passes a slice of length 2, the ONNX model should expect
    # an input of shape [2] or [1, 2].
    # Looking at the rust code: `let loads = m.predict_loads(&temps);`
    # And `predict_loads` calls `self.session.run(...)`.
    # Usually `ort` handles the shape. If we pass a 1D array, it expects 1D input.

    # Let's assume shape [2] for simplicity, or [1, 2] if batching is implied.
    # The batched test uses `vec![vec![20.0, 21.0], vec![50.0, 60.0]]`.
    # This implies the model should handle a batch dimension.
    # So input shape should be [batch_size, 2].

    dummy_input = torch.tensor([[20.0, 21.0]], dtype=torch.float32)

    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        "tests_tmp_dummy.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    generate_model()
    print("Model generated: tests_tmp_dummy.onnx")
