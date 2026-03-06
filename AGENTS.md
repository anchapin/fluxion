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
