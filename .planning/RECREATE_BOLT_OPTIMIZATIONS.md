# Plan: Recreate Performance Optimizations as Fresh PRs

## Context

Six performance optimization PRs were closed due to merge conflicts and code issues. The core optimizations should be recreated as fresh PRs from current main.

### Optimizations to Recreate

1. **HVAC Power Demand Optimization (from PR #493)**
   - Short-circuit calculations when HVAC is disabled (`enabled == 0.0`)
   - Merge two-loop approach into single-pass

2. **t_sol_air VectorField Allocation Removal (from PR #492)**
   - Remove unnecessary `VectorField::new()` allocation in `step_physics_5r1c()`

---

## Implementation Plan

### PR 1: HVAC Power Demand Optimization

**File:** `src/sim/engine.rs`

**Location:** `hvac_power_demand` method (~lines 3019-3066)

**Current State:**
```rust
fn hvac_power_demand(&self, hour: usize, t_i_free: &T, sensitivity: &T) -> T {
    // ... setpoint retrieval ...

    let mut demand_vec = Vec::with_capacity(self.num_zones);
    for i in 0..self.num_zones {
        let t = t_vec[i];
        // Calculate power...
        demand_vec.push(power);
    }

    // Second loop to multiply by enabled flag
    let enabled_vec = self.hvac_enabled.as_ref();
    for (power, &enabled) in demand_vec.iter_mut().zip(enabled_vec.iter()) {
        *power *= enabled;
    }
    // ...
}
```

**Changes:**
1. Move `enabled_vec` retrieval before the loop
2. Inside loop: check `enabled == 0.0` and `continue` to skip expensive calculations
3. Multiply `power` by `enabled` before pushing (handles partial enablement)
4. Remove second-pass loop entirely

**Implementation:**
```rust
fn hvac_power_demand(&self, _hour: usize, t_i_free: &T, sensitivity: &T) -> T {
    let heating_setpoint = self.heating_setpoint;
    let cooling_setpoint = self.cooling_setpoint;

    let t_vec = t_i_free.as_ref();
    let sens_vec = sensitivity.as_ref();
    let enabled_vec = self.hvac_enabled.as_ref();

    let mut demand_vec = Vec::with_capacity(self.num_zones);
    for i in 0..self.num_zones {
        let enabled = enabled_vec[i];

        if enabled == 0.0 {
            demand_vec.push(0.0);
            continue;
        }

        let t = t_vec[i];
        let power = if t < heating_setpoint {
            ((heating_setpoint - t) / sens_vec[i]).clamp(0.0, self.hvac_heating_capacity) * enabled
        } else if t >= cooling_setpoint {
            ((cooling_setpoint - t) / sens_vec[i]).clamp(-self.hvac_cooling_capacity, 0.0) * enabled
        } else {
            0.0
        };
        demand_vec.push(power);
    }
    // Removed second-pass loop

    T::from(VectorField::new(demand_vec))
}
```

**Tests to Verify:**
- `cargo test --lib hvac_power_demand` (if exists)
- `cargo test --lib test_hvac_control_comprehensive`
- ASHRAE validation should still pass

**Additional Cleanup:**
- Remove the debug print statement (lines 3056-3063) for Case 600 that was in the original code

**Benchmarks:**
- Run `cargo bench --bench engine_bench` to verify no regression

---

### PR 2: Remove Unused t_sol_air VectorField Allocation

**File:** `src/sim/engine.rs`

**Location:** `step_physics_5r1c()` method (~lines 3497-3504)

**Current State:**
```rust
let mut t_sol_air_data = Vec::with_capacity(self.num_zones);
for i in 0..self.num_zones {
    let i_sol = solar_ref[i];
    let t_sol_air_zone = outdoor_temp + (alpha * i_sol / h_se);
    t_sol_air_data.push(t_sol_air_zone);
}
let t_sol_air = VectorField::new(t_sol_air_data.clone());  // <-- unnecessary clone
```

**Problem:** `t_sol_air` VectorField is created but never indexed. The code uses `t_sol_air_data.get(i).copied()` instead.

**Change:** Remove the unnecessary VectorField allocation:
```rust
let mut t_sol_air_data = Vec::with_capacity(self.num_zones);
for i in 0..self.num_zones {
    let i_sol = solar_ref[i];
    let t_sol_air_zone = outdoor_temp + (alpha * i_sol / h_se);
    t_sol_air_data.push(t_sol_air_zone);
}
// Remove: let t_sol_air = VectorField::new(t_sol_air_data.clone());
```

**Note:** The `step_physics_6r2c()` version at ~line 4452 DOES use `t_sol_air[i]` for indexing, so keep that allocation.

**Tests to Verify:**
- `cargo test --lib step_physics_5r1c` (if exists)
- ASHRAE 140 validation should pass

---

## Critical Files

| File | Purpose |
|------|---------|
| `src/sim/engine.rs` | ThermalModel with HVAC and physics methods |
| `src/sim/hvac/equipment.rs` | Equipment power calculations |
| `benches/engine_bench.rs` | Performance benchmarks |

---

## Verification

### HVAC Optimization Verification
```bash
# 1. Run unit tests
cargo test --lib test_hvac_control_comprehensive

# 2. Run ASHRAE validation
cargo test --test ashrae_140_validation

# 3. Run benchmarks
cargo bench --bench engine_bench -- --sample-size 3
```

### t_sol_air Optimization Verification
```bash
# 1. Run physics tests
cargo test --lib step_physics

# 2. Run ASHRAE validation
cargo test --test ashrae_140_validation
```

---

## Workflow

1. **Create branch for PR 1** (HVAC optimization)
2. Implement changes to `hvac_power_demand()`
3. Verify with tests and benchmarks
4. Open PR against main
5. **After PR 1 merges, create branch for PR 2** (t_sol_air)
6. Implement allocation removal in `step_physics_5r1c()`
7. Verify with tests
8. Open PR against main

---

## Notes

- Both optimizations are in `src/sim/engine.rs` but should be separate PRs
- HVAC optimization is higher priority (bigger performance impact in hot loop)
- The `_hour` parameter was already added in the original PR - keep it to avoid unused parameter warnings
