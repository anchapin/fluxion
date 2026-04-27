# CTF-Primary Surface Temperature Coupling

## Session 89 Implementation Report

### Problem Statement
Session 84 proved that multi-layer RC splitting does NOT fix the thermal time constant problem.
Eigenvalue analysis: tau = 4.6h with 2 layers vs 26h lumped -- concrete's h_12 = 771 W/K is too high.
The fix: enable the existing CTF solver and use its T_si as primary surface temperature.

### Implementation (SESSION 89)

#### Files Modified
- `src/sim/engine.rs` -- All changes in ThermalModel

#### Structural Changes

1. **`ctf_primary: bool` field** (line ~530): Flag for CTF-primary mode
2. **`prepare_solvers_and_sol_air`** returns 4-tuple (added `ctf_surface_temps`)
3. **CTF enabled for 900FF/950FF** in `from_spec` with wall layers matching `Materials::high_mass_wall()`
4. **CTF-primary heat balance** in `step_physics_6r2c`: uses standard 6R2C flow with CTF T_si for mass update
5. **Envelope mass update** uses CTF T_si instead of lumped T_s when `ctf_primary = true`

#### Wall Construction (matches `src/sim/construction.rs:1472-1478`)
```
Concrete Block: k=0.51, rho=1400, cp=1000, d=0.100m
Foam Insulation: k=0.04, rho=10, cp=1400, d=0.0615m
Wood Siding: k=0.14, rho=500, cp=1300, d=0.009m
```

### Results

| Configuration | 900FF Max (C) | 900FF Min (C) | Target Max |
|---|---|---|---|
| Baseline (no CTF) | 53.31 | -9.94 | 41.8-46.4 |
| CTF-primary (T_si direct) | 52.96 | -4.09 | 41.8-46.4 |
| CTF additive (flux correction) | 56.72 | -5.98 | 41.8-46.4 |
| CTF mass-update only | 53.75 | -9.69 | 41.8-46.4 |

### Analysis

CTF-primary provides marginal improvement (-0.35C vs baseline) because:

1. **Thermal mass buffering is essential**: Bypassing the mass node entirely (direct T_si in air balance) removes thermal inertia, causing overshoot. Keeping the 6R2C structure with CTF-corrected mass boundary is more stable.

2. **CTF coefficient accuracy**: The simplified CTF coefficient model has convergence issues (noted in tests with `#[ignore]` for convergence checks). The X, Y, Z coefficients may not accurately represent the multi-layer wall dynamics.

3. **Surface coverage mismatch**: CTF models only the wall construction, while the 5R1C `h_tr_em` aggregates ALL opaque surfaces (walls + roof). Per-surface CTF dispatching would resolve this.

### Recommendations for Next Steps

1. **Fix CTF coefficient computation**: Replace the simplified model with exact state-space method for accurate X, Y, Z, Phi coefficients.

2. **Per-surface CTF**: Create separate CTF solvers for walls, roof, and floor with their actual constructions.

3. **CTF warmup**: Use `CTFSolver::with_warmup()` to initialize history buffers with realistic diurnal cycles.

4. **Hybrid coupling**: Use CTF T_si as a correction signal rather than replacement:
   ```
   T_s_corrected = T_s_lumped + alpha * (T_si_ctf - T_s_lumped)
   ```
   Start with alpha = 0.3 and tune.

5. **Investigate baseline regression**: Current baseline (53.31C) differs from session 84 baseline (47.45C). Check if 6R2C parameters changed between sessions.
