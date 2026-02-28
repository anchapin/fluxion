# ML Surrogate FDD Pipeline Integration (Issue #383)

## Overview

This document describes the integration of the ML Surrogate FDD Pipeline to connect Rust data collection hooks to ONNX model training for surrogate models.

## Components

### 1. Python Bridge Script (tools/integrate_ml_surrogate_pipeline.py)

The Python script provides:

- Full pipeline orchestration (validate, collect, train, benchmark)
- Individual step execution (--validate, --collect, --train, --benchmark)
- Physics-informed loss training
- ONNX model export
- Model benchmarking with R² score calculation

## Usage

### Running the Full Pipeline

\`\`\`bash
# Run complete ML surrogate pipeline
python tools/integrate_ml_surrogate_pipeline.py --all

# This will:
# 1. Run ASHRAE 140 validation tests
# 2. Collect training data from successful runs
# 3. Train surrogate model with physics-informed loss
# 4. Export ONNX model and benchmark performance
\`\`\`

### Individual Steps

\`\`\`bash
# Run validation only
python tools/integrate_ml_surrogate_pipeline.py --validate

# Collect training data only
python tools/integrate_ml_surrogate_pipeline.py --collect

# Train model only
python tools/integrate_ml_surrogate_pipeline.py --train

# Benchmark existing model
python tools/integrate_ml_surrogate_pipeline.py --benchmark models/surrogate/surrogate.onnx
\`\`\`

### Custom Options

\`\`\`bash
# Custom training parameters
python tools/integrate_ml_surrogate_pipeline.py \\
    --all \\
    --data-dir custom/data/path \\
    --model-dir custom/model/path \\
    --use-physics-loss \\
    --lambda-physics 0.1 \\
    --epochs 200
\`\`\`

## Physics-Informed Training

The training pipeline uses a physics-informed loss function:

\`\`\`
L_total = L_data + λ_physics * L_physics
\`\`\`

Where:
- \`L_data\`: Mean squared error between predicted and actual loads
- \`L_physics\`: Deviation from theoretical steady-state heat balance: Q = U * (T_setpoint - T_outdoor)
- \`λ_physics\`: Weight for physics loss term (default: 0.1)

This ensures the surrogate model maintains thermodynamic fidelity.

## Success Metric

The pipeline achieves success when:

1. **R² > 0.98**: Model explains >98% of variance in test data
2. **Physics fidelity**: Model respects thermodynamic constraints
3. **Validated against ASHRAE 140**: Matches benchmark cases

## Testing

### Run ASHRAE 140 Validation

\`\`\`bash
cargo test --test ashrae_140_validation --release -- --nocapture
\`\`\`

### Run Python Integration

\`\`\`bash
# Install dependencies
pip install torch numpy pandas onnxruntime

# Run full pipeline
python tools/integrate_ml_surrogate_pipeline.py --all
\`\`\`

### Benchmark Trained Model

\`\`\`bash
# Benchmark inference performance
python tools/benchmark_inference.py --model models/surrogate/surrogate.onnx

# Validate surrogate against analytical engine
python examples/validate_surrogate.py
\`\`\`

## Related Issues

- Issue #383: Integrate ML Surrogate FDD Pipeline
- Issue #217: Implement fault detection and diagnostics (FDD)
- ASHRAE 140 Standard Test Cases
- Phase 4 of HVAC Systems & ML Surrogate Validation
