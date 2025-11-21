import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import argparse
import os

def quantize(model_path: str, output_path: str):
    """
    Quantize an ONNX model to Int8 using dynamic quantization.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Quantizing model: {model_path}")
    print(f"Output path: {output_path}")

    try:
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=QuantType.QInt8
        )
        print("Quantization complete.")

        # Compare sizes
        original_size = os.path.getsize(model_path)
        quantized_size = os.path.getsize(output_path)
        print(f"Original size: {original_size/1024:.2f} KB")
        print(f"Quantized size: {quantized_size/1024:.2f} KB")
        print(f"Reduction: {(1 - quantized_size/original_size)*100:.1f}%")

    except Exception as e:
        print(f"Quantization failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantize ONNX model to Int8.')
    parser.add_argument('--model', type=str, required=True, help='Path to input ONNX model')
    parser.add_argument('--output', type=str, required=True, help='Path to output quantized model')

    args = parser.parse_args()
    quantize(args.model, args.output)
