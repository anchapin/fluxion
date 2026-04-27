# Session 89 Continuation: CTF-Primary Surface Temperature Coupling

## Context
Session 89 implemented CTF-primary surface temperature coupling for ASHRAE 140 Case 900FF in `src/sim/engine.rs`. The structural implementation is complete and all 2361 tests pass, but the thermal results need improvement.

## Current State of Results

| Configuration | 900FF Max (°C) | Target Max (°C) | Gap |
|---|---|---|---|
| Baseline (no CTF) | 53.31 | 41.8–46.4 | +6.9 above upper bound |
| CTF-primary (mass update only) | 53.75 | 41.8–46.4 | +7.4 |
| CTF-primary (direct T_si) | 52.96 | 41.8–46.4 | +6.6 |
| CTF additive mode | 56.72 | 41.8–46.4 | +10.3 |
| **Session 84 baseline** (user-stated) | **47.45** | 41.8–46.4 | **+1.05** |

**Critical discrepancy**: The user reported a baseline of 47.45°C, but current code gives 53.31°C — a 5.86°C regression. Fixing this regression likely closes most of the gap.

## What Was Implemented

### Structural Changes in `src/sim/engine.rs`
1. **`ctf_primary: bool` field** (~line 530) — flag for CTF-primary mode
2. **`prepare_solvers_and_sol_air`** — now returns 4-tuple `(t_sol_air, ctf_flux, fd_flux, ctf_surface_temps)`
3. **CTF enabled for 900FF/950FF** in `from_spec` (~line 2230) with wall layers:
   ```
   Concrete Block: k=0.51, ρ=1400, cp=1000, d=0.100m
   Foam Insulation: k=0.04, ρ=10, cp=1400, d=0.0615m
   Wood Siding: k=0.14, ρ=500, cp=1300, d=0.009m
   ```
4. **CTF-primary heat balance** in `step_physics_6r2c` — standard 6R2C path with CTF flux skipped when `ctf_primary`
5. **CTF T_si for mass update** — `t_s_act` uses CTF T_si (with HVAC offset) when `ctf_primary`
6. **Clone/copy** — `ctf_primary` added to ThermalModel clone block (~line 806)
7. **Design doc** created at `docs/design/ctf_primary_implementation_guide.md`

## Priority Issues to Investigate

### 1. Baseline Regression (HIGHEST PRIORITY)
The baseline jumped from 47.45°C to 53.31°C. Investigate:
- Check `git log` for changes to `configure_6r2c_model` parameters between sessions
- Compare current call `model.configure_6r2c_model(0.75, 100.0, None)` vs what session 84 used
- Check if `solar_beam_to_mass_fraction`, `solar_distribution_to_air`, or other 6R2C parameters changed
- Run `git diff HEAD~20..HEAD -- src/sim/engine.rs | grep -A5 -B5 "configure_6r2c\|solar_beam\|6r2c\|900FF"`

### 2. CTF Coefficient Accuracy
The CTF calculator uses a simplified model. Verify:
- Check if `sum(X)` approximates U-value = 0.556 W/m²K for the wall
- Look at `src/physics/ctf_coefficients.rs` — tests marked `#[ignore]` for convergence
- Consider implementing exact state-space method instead of simplified model

### 3. Per-Surface CTF Dispatching
Current CTF only models the wall. The 5R1C `h_tr_em` aggregates ALL opaque surfaces (walls + roof + floor). This mismatch causes the CTF net flux to diverge. Fix:
- Create separate CTF solvers for walls, roof, and floor
- Subtract per-surface 5R1C flux and add per-surface CTF flux

### 4. CTF Warmup
`CTFSolver::new()` initializes history at 20°C uniform. Use `CTFSolver::with_warmup()` for realistic initial conditions.

## Key Files
- `src/sim/engine.rs` — Main thermal model (ThermalModel struct, from_spec, step_physics_6r2c)
- `src/physics/ctf_solver.rs` — CTF solver (history buffers, step method)
- `src/physics/ctf_zone_coupling.rs` — Iterative T_si solver (Newton-Raphson)
- `src/physics/ctf_coefficients.rs` — CTF coefficient computation (simplified model)
- `src/sim/construction.rs` — Wall construction definitions (Materials::high_mass_wall)
- `tests/ashrae_140_free_floating.rs` — Free-floating test cases including 900FF
- `docs/design/ctf_primary_implementation_guide.md` — Session 89 design doc

## How to Test
```bash
# Build
cargo build

# Run all tests (should pass: 2361)
cargo test --lib

# Run 900FF diagnostic
cargo test test_free_floating_diagnostic_summary -- --nocapture 2>&1 | grep 900FF

# Run 900FF specific test
cargo test test_case_900ff_free_floating_high_mass -- --nocapture

# Check baseline without CTF (set ctf_primary = false and comment out enable_ctf in from_spec)
```

## Recommended Approach
1. **First**: Investigate and fix the baseline regression (47.45 → 53.31°C). This alone may close the gap.
2. **Second**: If baseline is still outside range, improve CTF coefficient accuracy.
3. **Third**: Implement per-surface CTF dispatching for proper flux correction.
