# Scripts Catalog

> **TL;DR**: Complete catalog of all tool and shell scripts in the Fluxion repository.
> **Key decisions**: Python scripts in tools/ | Shell scripts in scripts/ | Tests in tools/tests/
> **Owned by**: Wave orchestrator
> **Reviewed**: 2026-07-13

---

## Python Scripts (tools/)

### Core Benchmarking

| Script | Purpose | Key Dependencies |
|--------|---------|-----------------|
| `ashrae_140_test_harness.py` | Run ASHRAE 140 validation suite | energyplus, pandas |
| `ashrae_140_reference.py` | Generate ASHRAE 140 reference data | energyplus, pandas |
| `benchmark_throughput.py` | Throughput benchmarking | asyncio |
| `benchmark_batch_inference.py` | Batch inference benchmarking | onnxruntime |
| `benchmark_inference.py` | Single-case inference benchmarking | onnxruntime |
| `benchmark_throughput_gpu.py` | GPU throughput benchmarking | onnxruntime-gpu |

### Surrogate Training

| Script | Purpose | Key Dependencies |
|--------|---------|-----------------|
| `train_surrogate.py` | Train ML surrogate model | pytorch, onnx |
| `train_ml_surrogate.py` | Train ML surrogate (alternative) | pytorch, onnx |
| `train_pinn.py` | Train physics-informed neural network | pytorch |
| `train_ensemble_surrogates.py` | Train ensemble surrogate models | pytorch |
| `physics_informed_loss.py` | Physics-informed loss functions | pytorch |
| `physics_loss.py` | Physics loss computations | pytorch |
| `piml_loss.py` | Physics-informed machine learning loss | pytorch |
| `integrate_ml_surrogate_pipeline.py` | Integrate ML into physics pipeline | onnx, pandas |
| `validate_surrogate.py` | Validate surrogate model accuracy | onnxruntime, pandas |
| `generate_dummy_surrogate.py` | Generate dummy ONNX for testing | onnx |

### Data Generation

| Script | Purpose | Key Dependencies |
|--------|---------|-----------------|
| `generate_reference_data.py` | Generate E+ reference data | energyplus |
| `generate_ep_reference.py` | Generate E+ reference outputs | energyplus |
| `generate_diagnostic_data.py` | Generate diagnostic test data | pandas |
| `generate_case_900_idf.py` | Generate Case 900 IDF files | - |
| `generate_regression_tests.py` | Generate regression test cases | pandas |
| `geometry_extraction.py` | Extract geometry from IDF files | - |
| `data_collection.py` | Collect training data | pandas |
| `data_gen/main.py` | Main data generation entry point | - |
| `data_gen/ashrae_140_generator.py` | Generate ASHRAE 140 cases | - |
| `data_gen/monte_carlo.py` | Monte Carlo sampling | numpy |
| `data_gen/sampler.py` | Parameter sampling | numpy |
| `data_gen/simulation.py` | Simulation wrapper | - |
| `data_gen/geometry.py` | Geometry generation | - |
| `data_gen/weather.py` | Weather data handling | - |

### Optimization

| Script | Purpose | Key Dependencies |
|--------|---------|-----------------|
| `optimization.py` | General optimization | scipy, pandas |
| `multi_case_optimization.py` | Multi-case optimization | scipy, pandas |
| `online_learning.py` | Online learning for surrogates | pytorch |
| `parameter_validation.py` | Validate simulation parameters | pydantic |
| `performance_metrics.py` | Calculate performance metrics | pandas, numpy |
| `grid_search_h_si.py` | Grid search for H_si coefficients | numpy |
| `sweep_h_ms_coeff.py` | Sweep H_ms coefficients | numpy |

### Analysis & Comparison

| Script | Purpose | Key Dependencies |
|--------|---------|-----------------|
| `flux_diff.py` | Compare flux vs E+ for a case | pandas |
| `ep_oracle.py` | E+ oracle for ML training | energyplus |
| `export_xdt.py` | Export XDT format | onnx |
| `export_onnx.py` | Export ONNX models | onnx |
| `fluxion_delta.rs` | Delta computation in Rust | rust |
| `ensemble_training.py` | Ensemble training utilities | pytorch |
| `topsis.py` | TOPSIS multi-criteria analysis | pandas, numpy |

### Infrastructure

| Script | Purpose | Key Dependencies |
|--------|---------|-----------------|
| `cloud_campaign_manager.py` | Manage cloud campaigns | boto3 |
| `s3_aggregator.py` | Aggregate S3 results | boto3, pandas |
| `s3_worker.py` | S3 worker process | boto3 |
| `sync_planning.py` | Sync planning documents | - |

### Testing

| Script | Purpose | Key Dependencies |
|--------|---------|-----------------|
| `test_synthetic_fallback_disabled.py` | Test synthetic fallback disabled | pytest |

### Compliance

| Script | Purpose | Key Dependencies |
|--------|---------|-----------------|
| `compliance_agent/agent.py` | Compliance checking agent | openai |
| `compliance_agent/demo.py` | Compliance demo | openai |
| `compliance_agent/llm_backend.py` | LLM backend for compliance | openai |

---

## Shell Scripts (scripts/)

| Script | Purpose | Key Dependencies |
|--------|---------|-----------------|
| `annual_ashrae_revalidation.sh` | Full ASHRAE revalidation | energyplus, python |
| `run_mutation_testing.sh` | Run cargo-mutants | cargo-mutants |
| `check_pr_closing_refs.sh` | Check PR closing references | gh, bash |
| `install_ripr.sh` | Install RIPR dependencies | - |
| `provision-hetzner-runner.sh` | Provision Hetzner runner | terraform |
| `release_v0.8.0.sh` | Release v0.8.0 | git, github cli |
| `setup_parallel_worktrees.sh` | Setup parallel worktrees (race-safe, #2489) | git, xargs |
| `wt-add.sh` | Race-safe single-worktree creation (`--check` self-test, #2489) | git, bash |

---

## Test Scripts (tools/tests/)

| Script | Purpose |
|--------|---------|
| `test_ashrae_140_reference.py` | Test ASHRAE 140 reference data |
| `test_data_collection.py` | Test data collection |
| `test_data_gen.py` | Test data generation |
| `test_ensemble_training.py` | Test ensemble training |
| `test_gymnasium_env.py` | Test Gymnasium environment |
| `test_multi_case_optimization.py` | Test multi-case optimization |
| `test_online_learning.py` | Test online learning |
| `test_optimization.py` | Test optimization |
| `test_parameter_validation.py` | Test parameter validation |

---

## Usage Notes

### Prerequisites
- Python 3.10+
- EnergyPlus (for ASHRAE 140 tests)
- Rust toolchain (for fluxion_delta.rs)
- AWS credentials (for cloud scripts)

### Running Scripts
```bash
# Python scripts
python tools/ashrae_140_test_harness.py

# Shell scripts
bash scripts/annual_ashrae_revalidation.sh

# With arguments
python tools/benchmark_throughput.py --cases 100 --workers 4
```

---

## Wave 2 Scripts Issues

| Issue | Title | Status |
|-------|-------|--------|
| 1534 | docs/scripts/README.md | in-progress |
