# Examples Overview

This folder contains small, self-contained examples that demonstrate the primary user-facing APIs in Fluxion:

- `run_model.py` — Example using the single-`Model` API for detailed, single-configuration simulation.
- `run_oracle.py` — Example using `BatchOracle` to evaluate a small population in parallel (the hot loop for optimization).
- `simple_config.json` — Minimal building configuration consumed by `run_model.py` (the current `Model` constructor ignores details, but the file shows where configs belong).
- `quick_start.sh` — Helper script to build local Python bindings (`maturin develop`) and run `run_oracle.py`.

Purpose
- Provide reproducible examples for new users to run locally after building the Python bindings with `maturin develop`.
- Demonstrate the expected input formats (population vectors) and show simple output interpretation.

Running the examples

From the repository root, after building/installing the Python bindings locally:

```bash
# Optional: create and activate a venv
python3 -m venv .venv
source .venv/bin/activate

# Build & install local Python bindings
pip install --upgrade pip
pip install maturin
maturin develop

# Run the oracle example
python examples/run_oracle.py

# Or run the model example
python examples/run_model.py
```

Quick start script
- `examples/quick_start.sh` automates installation of `maturin` (if missing), builds the local bindings and runs `examples/run_oracle.py`. It is a convenience helper only — inspect it before running on CI environments.

Modifying examples
- `run_oracle.py` uses a small default population size (20). To stress-test throughput, increase the number passed to `make_population(n)` and observe `Elapsed` time.
- `run_model.py` runs a 1-year simulation by calling `Model.simulate(1, use_surrogates)`. Adjust the `years` argument to run multi-year checks.

Notes on determinism
- The repository's `SurrogateManager` currently returns deterministic mock loads when no ONNX model is loaded. To see different populations' behaviour, supply a different random seed or modify `run_oracle.py` to use NumPy with a seed.

Where to go next
- See `../docs/EXAMPLES.md` for detailed input/output semantics and example calculations.
- See `../docs/THEORY_AND_STRATEGY.md` for the underlying physics model description and recommended strategies for using surrogates safely in optimization workflows.
