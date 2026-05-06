## Problem Description

CTF solver for high-mass constructions (900FF, 950FF) is initialized with zero flux history instead of being properly warmed up, causing unphysical heat flux values at simulation start.

## Root Cause

**`src/physics/ctf_solver.rs:112-135`** - `CTFSolver::new()` initializes ALL history to 20°C uniform with zero heat flux:

```rust
t_interior_history: vec![20.0; history_size],   // <- UNIFORM 20°C
t_exterior_history: vec![20.0; history_size],   // <- UNIFORM 20°C
q_interior_history: vec![0.0; history_size],   // <- ZERO FLUX
q_exterior_history: vec![0.0; history_size],   // <- ZERO FLUX
```

**`src/physics/ctf_solver_wrapper.rs:148`** - `with_warmup()` exists but is NOT called during initialization:

```rust
self.solver = Some(CTFSolver::new(coeffs.clone(), config));  // Should use with_warmup()
```

## Impact

At simulation start for free-floating cases:
1. History buffers = [20°C, 20°C, ...] with q = 0
2. Exterior temperature swings widely (e.g., Denver winter -10°C to +35°C)
3. CTF calculates heat flux using unphysical initial conditions
4. No warmup period to establish realistic flux history

## Proposed Fix

Change `src/physics/ctf_solver_wrapper.rs:148` to call `with_warmup()` instead of `new()`:

```rust
self.solver = Some(CTFSolver::with_warmup(
    coeffs.clone(),
    config,
    20.0,  // t_interior_initial
    20.0,  // t_exterior_initial
    7,     // warmup_days
));
```

Or modify the caller to pass actual exterior temperature for the warmup period.