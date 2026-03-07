# Agent's `gh` CLI Usage Notes

This document logs common issues encountered while using the `gh` CLI tool and their resolutions, serving as a future reference to avoid repeated mistakes.

## Issue #326: PINN (Physics-Informed Neural Network) Training Implementation

**Implementation Summary:**
- Extended neural network module using PyTorch to support PINNs
- Implemented custom loss function: L_total = L_data + λ * L_physics
- Uses PyTorch's autograd to calculate temperature gradients with respect to time
- Penalizes network for violating thermodynamic principles (q=mcΔT)

**Files Modified:**
- `tools/train_pinn.py` - Main PINN training pipeline with ThermalPINN, PINNLoss, PINNConfig, PhysicsConfig classes
- `tools/physics_informed_loss.py` - Physics-informed loss functions module

**Key Classes:**
- `ThermalPINN`: PyTorch neural network for thermal prediction
- `PINNLoss`: Custom loss combining data loss + physics loss + initial/boundary conditions
- `PhysicsConfig`: Configuration for thermal physics parameters (thermal_capacity, h_transmission, h_ventilation)
- `PINNConfig`: Training configuration (weights for data, physics, initial_condition, boundary, energy_balance)
- `ThermalDataGenerator`: Generate training data using 5R1C thermal model

**Verification:**
- Python compilation: PASSED
- Import tests: PASSED
- Forward pass: PASSED
- Loss computation: PASSED
- Training loop: PASSED

**Notes:**
- Physics weight should be small initially (e.g., 0.0001) to allow the network to learn basic patterns before enforcing physics constraints
- Unit conversion: thermal capacity in kWh/K, heat transfer in W/K, time in hours → convert to seconds for proper energy balance

## Issue #448: Automated Geometry Ingestion Pipeline (PDF/CAD-to-BEM) via Vision-Language Models

**Implementation Summary:**
- Created automated pipeline for extracting building geometry from PDF/CAD files
- Uses VLM (Vision-Language Models) to parse architectural drawings
- Converts extracted geometry to CTA (Combined Thermal and Airflow) tensor format
- Provides zero-copy handoff to Rust core via PyO3 bindings

**Files Created/Modified:**
- `tools/geometry_extraction.py` - Main geometry extraction pipeline module
- `src/physics/geometry_tensor.rs` - New Rust module for geometry tensors
- `src/physics/mod.rs` - Added geometry_tensor module
- `src/lib.rs` - Added PyGeometryTensor PyO3 bindings
- `demo_geometry_pipeline.py` - Demo script

**Key Components:**

1. **Python Pipeline** (`tools/geometry_extraction.py`):
   - `GeometryExtractor`: VLM-based geometry extraction from images/PDFs/DXFs
   - `GeometryToCTATensorConverter`: Converts geometry to CTA tensors
   - `GeometryIngestionPipeline`: High-level pipeline combining extraction + conversion
   - Supports VLM providers: mock (testing), Ollama, OpenAI Vision

2. **Rust Module** (`src/physics/geometry_tensor.rs`):
   - `GeometryTensor`: Container for CTA geometry tensors
   - `WallData`: Wall geometry structure
   - Constants: MAX_ZONES=100, MAX_WALLS=500

3. **PyO3 Bindings** (`src/lib.rs`):
   - `PyGeometryTensor`: Zero-copy Python bindings
   - Supports numpy array interop via `from_numpy()` and `to_numpy()`

**CTA Tensor Formats:**
- `zone_coords`: (100, 20) - Zone coordinates, heights, area, volume
- `wall_matrix`: (500, 6) - Wall geometry [x1, y1, x2, y2, height, thickness]
- `window_matrix`: (500, 6) - Window geometry [x1, y1, x2, y2, height, sill_height]
- `adjacency_matrix`: (100, 100) - Zone adjacency (0/1)
- `zone_properties`: (100, 5) - Zone thermal properties
- `summary`: (6,) - Summary statistics

**Verification:**
- Rust compilation: PASSED (`cargo check --features python-bindings`)
- Python import: PASSED
- Demo script: PASSED
- Tensor validation: PASSED

**Usage:**
```python
from tools.geometry_extraction import GeometryIngestionPipeline

# Create pipeline with VLM
pipeline = GeometryIngestionPipeline(vlm_provider='ollama')
geometry, tensors = pipeline.ingest('floor_plan.png')

# Pass to Rust (zero-copy)
import fluxion
geo_tensor = fluxion.GeometryTensor.from_numpy(
    tensors['zone_coords'],
    tensors['wall_matrix'],
    tensors['window_matrix'],
    tensors['adjacency_matrix'],
    tensors['zone_properties'],
    tensors['summary']
)
```

**Notes:**
- Mock VLM provider available for testing without external dependencies
- Supports DXF (CAD) direct parsing via ezdxf library
- PDF support via PyMuPDF (converts to image first)
- Tensor validation ensures data integrity

## Issue 1: Retrieving Job Logs for a Specific GitHub Actions Run

**Problem:**
Attempting to fetch logs for a specific job within a GitHub Actions workflow run using `gh run view <run-id> --job <job-id> --log` or `gh run view <run-id> --job <job-name> --log` consistently resulted in "HTTP 404: Not Found" errors or "unknown command 'jobs'". This was despite having the correct run ID and job ID/name extracted from GitHub Actions URLs.

**Mistakes Made:**
- Misunderstanding the exact syntax and capabilities of `gh run view` for job-specific log retrieval.
- Incorrectly assuming that `--job <job-name>` or `--job <job-id>` would work directly with `gh run view`.
- Relying on potentially outdated `gh pr checks` output for job IDs without verifying the correct command structure for `gh run view`.

**Solution:**
The correct approach to get the *full log* for a specific run is `gh run view <run-id> --log`. To specifically get the output of a *single job* within that run, it seems the `gh` CLI doesn't offer a direct filtered log view via `run view`. Instead, one must:
1.  Identify the `run-id` associated with the PR, potentially using `gh run list --workflow "CI" --branch <branch-name>`.
2.  Use `gh run view <run-id> --log` to fetch the *entire log* for that run.
3.  Manually parse the large log file to find the output of the specific job, or resort to manual inspection on the GitHub Actions website.

**Example of correct usage discovered:**
- `gh run list --workflow "CI" --branch "feature/validate-oracle-inputs" --json databaseId,status,conclusion,event,name,url` (to find `run-id`)
- `gh run view 19713997663 --log > /path/to/local_log.txt` (to get full run log)

**Lesson Learned:**
Always consult `gh <command> --help` or official documentation for precise syntax, especially when encountering "unknown command" or unexpected HTTP 404 errors. The structure of commands and available flags can be subtle. In cases where direct programmatic access is difficult, a hybrid approach of fetching full logs and then parsing, or resorting to manual web UI inspection, might be necessary.
