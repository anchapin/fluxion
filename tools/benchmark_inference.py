import time
import numpy as np
import onnxruntime as ort
import argparse
import os

def benchmark_batch_sizes(model_path: str, max_batch: int = 1000):
    """Find optimal batch size for throughput."""

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    # Check available providers
    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")

    # Use CUDA if available, otherwise CPU
    providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
    print(f"Using providers: {providers}")

    try:
        sess = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        print(f"Failed to create inference session: {e}")
        return

    # Inspect model input to determine shape
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    # Assuming input shape is [batch_size, input_dim] or similar.
    # If shape has symbolic dimension (like None or 'batch'), we can use variable batch size.
    # If dimensions are fixed, we might have issues with batching if the model doesn't support it.

    # Try to infer input dimension from shape
    input_dim = 2 # Default fallback
    if len(input_shape) == 2:
        if isinstance(input_shape[1], int):
            input_dim = input_shape[1]

    print(f"Input name: {input_name}, Input shape: {input_shape}, inferred dim: {input_dim}")

    for batch_size in [1, 10, 100, 1000, 10000]:
        if batch_size > max_batch:
            break

        # Generate dummy batch
        X = np.random.randn(batch_size, input_dim).astype(np.float32)

        # Measure inference time
        t0 = time.time()
        iterations = 100
        for _ in range(iterations):
            outputs = sess.run(None, {input_name: X})
        t1 = time.time()

        total_time = t1 - t0
        throughput = (batch_size * iterations) / total_time
        print(f"Batch {batch_size:5d}: {throughput:10.0f} configs/sec ({total_time/iterations*1000:.2f} ms/batch)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark ONNX inference throughput.')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--max-batch', type=int, default=10000, help='Maximum batch size to test')

    args = parser.parse_args()
    benchmark_batch_sizes(args.model, args.max_batch)
