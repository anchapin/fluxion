# Tools

This directory contains utility scripts for Fluxion development, including training AI surrogates, benchmarking, and data analysis.

## Available Tools

- `train_surrogate.py`: Main script for training neural network surrogate models.
- `benchmark_inference.py`: Measure inference latency of ONNX models.
- `benchmark_throughput.py`: Measure system throughput.
- `generate_dummy_surrogate.py`: Create placeholder models for testing.
- `quantize_model.py`: Optimize ONNX models using quantization.

## Training Surrogate Models

The `train_surrogate.py` tool automates the training of neural network surrogates that approximate the physics-based calculations of Fluxion.

### How it works

1.  **Data Generation**: If no input file is provided, it generates synthetic data using a simplified internal physics model (or can be adapted to use the analytical `fluxion` engine).
2.  **Training**: Trains a configurable PyTorch Multi-Layer Perceptron (MLP) on the data.
3.  **Validation**: Splits data into train/validation sets and tracks metrics (MAE, MSE, R²).
4.  **Export**: Saves the best model as an ONNX file for integration with the Rust engine.

### Usage Examples

**Basic training (generates data):**
```bash
python tools/train_surrogate.py
```

**Training with custom hyperparameters:**
```bash
python tools/train_surrogate.py --num-samples 50000 --epochs 50 --hidden-dims 128 128 64 --learning-rate 0.0005
```

**Training from existing data file:**
```bash
python tools/train_surrogate.py --input-file assets/training_data.npz --output-dir models/experiment_1
```

### Command-Line Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--input-file` | None | Path to .npz or .csv file containing X and y data. If omitted, data is generated. |
| `--num-samples` | 10000 | Number of samples to generate (if input-file is not provided). |
| `--num-zones` | 10 | Number of thermal zones (output dimension) for generation. |
| `--epochs` | 100 | Number of training epochs. |
| `--batch-size` | 32 | Batch size for training. |
| `--learning-rate` | 0.001 | Initial learning rate. |
| `--hidden-dims` | [64, 64] | List of hidden layer sizes (e.g., `128 64`). |
| `--seed` | 42 | Random seed for reproducibility. |
| `--output-dir` | models | Directory to save model (.onnx), checkpoints (.pt), and metrics (.json). |

### Outputs

The script saves the following files in the `output-dir`:
- `surrogate.onnx`: The exported ONNX model ready for Rust integration.
- `best_model.pt`: PyTorch state dictionary of the best performing epoch.
- `metrics.json`: Final validation metrics (MAE, R²).
- `generated_data.npz`: The data used for training (if generated).
- `prediction_plot.png`: A plot comparing actual vs. predicted values (if matplotlib is available).
