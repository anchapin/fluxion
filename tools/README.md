# Tools

This directory contains utility scripts for Fluxion development, including training AI surrogates, benchmarking, and data analysis.

## Available Tools

- `train_surrogate.py`: Main script for training neural network surrogate models.
- `benchmark_inference.py`: Measure inference latency of ONNX models.
- `benchmark_throughput.py`: Measure system throughput.
- `generate_dummy_surrogate.py`: Create placeholder models for testing.
- `quantize_model.py`: Optimize ONNX models using quantization.
- `data_gen/`: Data generation tools for creating training datasets.
- `compliance_agent/`: Automated code compliance checking for building energy models (ASHRAE 90.1 / IECC).

## Code Compliance Agent (`compliance_agent/`)

The compliance agent provides automated code compliance checking for building energy models against ASHRAE 90.1 and IECC standards using LLMs.

### Features

- **Multiple LLM Backends**: Support for Mock, Ollama, and OpenAI backends
- **ASHRAE 90.1 Support**: 2019 and 2022 standards
- **IEC Support**: 2021 and 2024 standards
- **Rules Engine**: Built-in compliance rules for quick checks
- **LLM-Powered Analysis**: Uses LLMs for detailed compliance analysis

### Quick Start

```python
from tools.compliance_agent import CodeComplianceAgent

# Create agent with mock backend
agent = CodeComplianceAgent(backend="mock")

# Building model data
model_data = {
    "model_name": "Office Building",
    "wall_r_value": 15.0,
    "window_u_factor": 0.35,
    "hvac_cop": 3.5,
}

# Check compliance
report = agent.check_compliance(
    model_data=model_data,
    standard="ASHRAE90.1-2019"
)

print(report.print_summary())
```

### Demo

```bash
# Run the demo script
python tools/compliance_agent/demo.py
```

## Data Generation (`data_gen/`)

The `data_gen/` subdirectory contains tools for generating large-scale training data for AI surrogate models.

### Monte Carlo Data Generator (`monte_carlo.py`)

The Monte Carlo data generator creates diverse training datasets by sampling random building configurations and running physics simulations.

#### Features

- **Diverse Sampling**: Samples building parameters using Latin Hypercube Sampling (LHS) or random sampling
- **Physics Simulation**: Runs simulations using the Fluxion engine for accurate energy modeling
- **Batch Processing**: Optimized for large-scale data generation with batch processing
- **Multiple Output Formats**: Supports Parquet, NumPy (.npz), and HDF5 output formats
- **Reproducibility**: Supports random seeds for reproducible datasets

#### Usage

```bash
# Generate 1000 training samples with default settings
python -m tools.data_gen.monte_carlo generate --count 1000

# Generate with custom output directory and seed
python -m tools.data_gen.monte_carlo generate --count 5000 --output /tmp/train_data --seed 42

# Generate with specific sampling method
python -m tools.data_gen.monte_carlo generate --count 1000 --sampling-method LHS

# Generate in batch mode for large datasets
python -m tools.data_gen.monte_carlo generate --count 10000 --batch-size 1000

# List available options
python -m tools.data_gen.monte_carlo --help
```

#### Command-Line Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--count` | 1000 | Number of samples to generate |
| `--output` | ./training_data | Output directory for generated data |
| `--seed` | 42 | Random seed for reproducibility |
| `--sampling-method` | LHS | Sampling method (RANDOM, LHS) |
| `--batch-size` | 100 | Number of samples per batch |
| `--weather-dir` | assets/weather | Directory containing weather files |
| `--format` | parquet | Output format (parquet, npz, hdf5) |

#### Output Format

The generated training data is in (State_t, Action_t) → (State_t+1, Energy) format:

| Column | Description |
| :--- | :--- |
| `outdoor_temp_t` | Outdoor temperature at time t (°C) |
| `indoor_temp_t` | Indoor temperature at time t (°C) |
| `solar_t` | Solar radiation at time t (W/m²) |
| `hvac_mode` | HVAC mode (0=off, 1=heating, 2=cooling) |
| `hvac_power` | HVAC power at time t (W) |
| `indoor_temp_t1` | Indoor temperature at time t+1 (°C) |
| `energy_consumed` | Energy consumed (Wh) |
| `run_id` | Sample identifier |
| `timestep` | Timestep index |

#### Python API

```python
from tools.data_gen.monte_carlo import MonteCarloDataGenerator

# Create generator
gen = MonteCarloDataGenerator(
    output_dir="./output",
    num_samples=1000,
    seed=42,
    sampling_method="LHS",
    batch_size=100,
)

# Setup (loads weather files, configures sampler)
gen.setup()

# Generate data
df = gen.generate()

# Save to specific format
gen.save_output(df, format="parquet")
```

#### Performance

The generator is optimized for large-scale data generation:
- Batch processing reduces memory usage
- Parallel simulation support for multi-core systems
- Efficient Parquet format for fast I/O
- Target throughput: >1000 samples/second

## PIML Surrogate Training (`piml_loss.py`)

Physics-Informed Machine Learning (PIML) embeds thermodynamic RC (Resistance-Capacitance) models into neural network loss functions for surrogate training. This addresses Issue #552 for improving peak load accuracy.

### Key Features

- **RC Thermal Network**: Embedded simplified resistance-capacitance thermal model in loss function
- **Physics-Informed Penalty Terms**: Thermodynamic violations penalized during backpropagation
- **Comparison Mode**: Train with PIML loss vs standard MSE for validation

### Usage

```bash
# Train with PIML loss (physics-informed)
python tools/piml_loss.py --epochs 500 --piml-weight 0.5

# Train with comparison between PIML and standard loss
python tools/piml_loss.py --compare-loss --epochs 500

# Custom PIML weights
python tools/piml_loss.py --piml-weight 0.3 --energy-balance-weight 0.5
```

### Command-Line Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--n-samples` | 5000 | Number of training samples |
| `--epochs` | 500 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--learning-rate` | 1e-3 | Learning rate |
| `--piml-weight` | 0.5 | Weight for physics-informed loss term |
| `--compare-loss` | False | Compare PIML vs standard MSE loss |
| `--output-dir` | models/piml | Output directory |

### RC Physics Configuration

The RC thermal model uses:
- `thermal_capacity`: Thermal mass (kJ/K)
- `h_transmission`: Transmission heat transfer coefficient (W/K)
- `h_ventilation`: Ventilation heat transfer coefficient (W/K)

The loss combines:
- **MSE loss**: Standard mean squared error
- **PIML loss**: RC model-based physics residual
- **Energy balance**: Heating/cooling thermodynamic constraints
- **Boundary loss**: Non-negative load constraints

### Validation

PIML-trained surrogates target:
- CvRMSE < 30% for peak load accuracy (Issue #552 requirement)
- Comparison with standard loss for improvement measurement

## Training Surrogate Models

The `train_surrogate.py` tool automates the training of neural network surrogates that approximate the physics-based calculations of Fluxion.

### How it works

1.  **Data Loading**: Loads physics-extracted samples from `--data-dir` (default `data/training/`). The directory must contain one or more `samples_*.csv` files produced by the EnergyPlus extraction pipeline described in `data/training/README.md`.
2.  **Hard error on missing physics samples**: Per Issue #1338, the script raises `SystemExit` (exit code 1) when `data/training/` is empty or missing — there is no silent fallback to random data. CI / benchmark harnesses that explicitly need a synthetic backfill must pass `--allow-synthetic-for-benchmark-only`.
3.  **Training**: Trains a configurable PyTorch Multi-Layer Perceptron (MLP) on the data.
4.  **Validation**: Splits data into train/validation sets and tracks metrics (MAE, MSE, R²).
5.  **Export**: Saves the best model as an ONNX file for integration with the Rust engine.

### Usage Examples

**Production training (requires physics samples in `data/training/`):**
```bash
python tools/train_surrogate.py --data-dir data/training --output-dir models
```

**Benchmark-only run (CI harnesses; random synthetic data, NOT for production):**
```bash
python tools/train_surrogate.py --allow-synthetic-for-benchmark-only \
    --run-benchmark --shap-analysis --ensemble
```

**Training with custom hyperparameters:**
```bash
python tools/train_surrogate.py --data-dir data/training --num-samples 50000 \
    --epochs 50 --hidden-dims 128 128 64 --learning-rate 0.0005
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
