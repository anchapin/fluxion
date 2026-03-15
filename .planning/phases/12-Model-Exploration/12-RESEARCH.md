# Phase 12: Model Exploration - Research

**Researched:** 2026-03-13
**Domain:** Building Energy Modeling - 6R2C Thermal Network Evaluation
**Confidence:** HIGH

## Summary

Fluxion currently implements a 5R1C (5-Resistance, 1-Capacitance) thermal network model per ISO 13790 standard. While this model achieves accurate peak load predictions and validates well for low-mass buildings (600 series ASHRAE 140 cases), it has known limitations for high-mass buildings (900 series), over-predicting annual energy consumption by 229-322%.

The 6R2C model extends the 5R1C structure by splitting thermal mass into two separate nodes: envelope mass (walls, roof, floor) and internal mass (furniture, partitions). This additional degree of freedom better captures thermal lag effects in heavy concrete structures where heat transfer through building envelope creates significant time delays.

Phase 12's goal is to evaluate whether the 6R2C model provides sufficient accuracy improvement to justify adopting it as the default model for v0.3. The 6R2C model is already partially implemented in the codebase (ThermalModelType::SixRTwoC, step_physics_6r2c method, configure_6r2c_model API) but requires validation and benchmarking against the established 5R1C baseline.

**Primary recommendation:** Complete 6R2C validation through systematic benchmarking, parameter calibration, and ASHRAE 140 comparison. If 6R2C reduces high-mass annual energy error below 50% of reference ranges without regressing low-mass cases or breaking performance targets, adopt as default for 900 series cases. Otherwise, document findings and keep 5R1C as default with clear 6R2C opt-in guidance.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MODEL6R2C-01 | Design 6R2C thermal network structure (exterior mass, interior mass, separate capacitances) | Existing implementation in docs/6R2C_IMPLEMENTATION.md provides design framework with envelope/internal mass nodes and h_tr_me coupling resistance |
| MODEL6R2C-02 | Implement 6R2C solver alongside existing 5R1C (feature flag, parallel support) | ThermalModelType enum and step_physics_6r2c already implemented; requires feature flag integration and validation that both models work in parallel |
| MODEL6R2C-03 | Compare 6R2C vs 5R1C for high-mass cases (Case 900) and standard cases | ASHRAE 140 validation framework (src/validation/ashrae_140_cases.rs) provides test cases; KNOWN_LIMITATIONS.md documents 5R1C baseline failures (262-322% error) |
| MODEL6R2C-04 | Document findings: accuracy gains, performance trade-offs, migration path | Documentation template exists (docs/6R2C_IMPLEMENTATION.md); requires accuracy comparison, benchmark results, and decision documentation |
| MODEL6R2C-05 | Decide whether to adopt 6R2C as default or keep 5R1C for v0.3 | Decision framework: adopt if >50% error reduction on 900 series, no 600 series regression, maintains >1,000 configs/sec throughput |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Rust 2021 Edition | stable | Systems programming for physics engine | Type safety, zero-cost abstractions, memory safety |
| faer | 0.23.2 | Linear algebra for thermal network solving | BLAS/LAPACK acceleration, supports VectorField operations |
| rayon | 1.10 | Data parallelism for BatchOracle | Proven pattern for population-level parallelism without nested loops |
| ndarray | 0.16 | Numerical arrays for time-series data | Efficient storage, BLAS integration, serde support |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| serde | 1.0 | Serialization for test cases and validation data | JSON/YAML format for ASHRAE 140 specifications |
| criterion | 0.5 | Benchmarking for performance comparison | Measure throughput, memory, and latency of 5R1C vs 6R2C |
| approx | 0.5 | Floating-point comparison in tests | Validate numerical accuracy with tolerance for rounding errors |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| faer | nalgebra | nalgebra more mature but larger dependency footprint; faer provides required BLAS/LAPACK with smaller API |
| rayon | threadpool or crossbeam | rayon has better integration with iterators; alternatives require manual thread management |

**Installation:**
```bash
# All dependencies already in Cargo.toml
# No new dependencies required for 6R2C implementation
cargo build --release
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── sim/
│   ├── engine.rs              # ThermalModel with 5R1C/6R2C branching
│   └── thermal_model.rs      # ThermalModelBuilder for case construction
├── validation/
│   ├── ashrae_140_cases.rs  # ASHRAE 140 test case specifications
│   └── ashrae_140_validator.rs  # Validation framework with comparison logic
└── physics/
    └── cta.rs                # Continuous Tensor Abstraction (VectorField)

tests/
├── test_6r2c_model.rs      # 6R2C-specific tests (11 tests, currently 1 failing)
├── ashrae_140_case_900.rs  # High-mass validation tests
└── benchmark_report_validation.rs  # Performance regression tests

docs/
├── 6R2C_IMPLEMENTATION.md  # Existing 6R2C design documentation
└── KNOWN_LIMITATIONS.md      # 5R1C limitations and 6R2C investigation results
```

### Pattern 1: Thermal Model Type Switching
**What:** Runtime selection between 5R1C and 6R2C physics solvers based on ThermalModelType enum
**When to use:** When ThermalModel::step_physics is called, branch to appropriate solver
**Example:**
```rust
// Source: src/sim/engine.rs:2460
pub fn step_physics(&mut self, timestep: usize, outdoor_temp: f64) -> f64 {
    if self.is_6r2c_model() {
        self.step_physics_6r2c(timestep, outdoor_temp)
    } else {
        self.step_physics_5r1c(timestep, outdoor_temp)
    }
}
```

### Pattern 2: Feature Flag Configuration
**What:** Compile-time feature flags to enable/disable experimental models
**When to use:** During development and testing of 6R2C before production adoption
**Example:**
```rust
// Add to Cargo.toml features section
[features]
default = []
6r2c_model = []  # Experimental 6R2C thermal network

// In src/sim/engine.rs
#[cfg(feature = "6r2c_model")]
pub fn configure_6r2c_model(&mut self, envelope_mass_fraction: f64, h_tr_me_value: f64) {
    // Implementation...
}
```

### Pattern 3: Backward-Compatible API
**What:** Maintain single mass_temperature field as weighted average of envelope/internal masses for compatibility
**When to use:** When code expects single mass temperature but model uses dual masses
**Example:**
```rust
// Source: docs/6R2C_IMPLEMENTATION.md:122-124
self.mass_temperatures = (
    self.envelope_mass_temperatures.clone() * self.envelope_thermal_capacitance.clone()
    + self.internal_mass_temperatures.clone() * self.internal_thermal_capacitance.clone()
) / total_cap;
```

### Anti-Patterns to Avoid
- **Nested parallelism in step_physics**: Never call rayon::par_iter() inside step_physics; this breaks BatchOracle pattern
- **Hardcoded mass fractions**: Allow envelope_mass_fraction to be configurable, not fixed at 0.75
- **Ignoring numerical stability**: Validate that dual-mass updates don't introduce NaN/Inf states
- **Breaking backward compatibility**: Keep mass_temperatures field updated for code that reads single mass

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Linear solver for thermal network | Custom matrix inversion | faer's BLAS/LAPACK backend | Numerical stability, performance, battle-tested algorithms |
| Test case data structures | Manual JSON parsing | serde + ASHRAE140Case enum | Type safety, compile-time validation, error handling |
| Benchmark comparison logic | Manual timing and statistics | criterion with baselines | Statistical significance, noise reduction, CI-friendly |
| Thermal mass energy accounting | Manual energy tracking | Existing thermal_mass_energy_accounting flag in ThermalModel | Proven pattern, avoids double-counting |

**Key insight:** The 6R2C model is already partially implemented. The challenge is not building from scratch but validating, benchmarking, and deciding whether to adopt it. Focus efforts on systematic evaluation rather than re-implementation.

## Common Pitfalls

### Pitfall 1: 6R2C Test Failure in test_6r2c_model_single_timestep
**What goes wrong:** Test assertion fails because temperatures don't change from initial state when outdoor_temp=10°C differs from initial 20°C
**Why it happens:** The 6R2C physics implementation may have bugs in heat transfer equations or mass temperature updates
**How to avoid:**
1. Debug step_physics_6r2c with detailed logging at each timestep
2. Verify conductance values (h_tr_me, h_tr_em, h_tr_ms) are non-zero and reasonable
3. Check that mass temperature updates use correct heat flow equations
4. Validate that envelope/internal mass coupling (h_tr_me) allows heat transfer
**Warning signs:** Test fails immediately at first timestep, all temperatures remain exactly equal to initial values

### Pitfall 2: Annual Energy Over-Prediction in 5R1C (Documented Limitation)
**What goes wrong:** 5R1C model over-predicts annual energy for high-mass buildings by 229-322%
**Why it happens:** Single thermal mass node cannot capture thermal lag through heavy concrete structures; heat transfer time constant (~4.82 hours) comparable to hourly timestep
**How to avoid:**
1. Use 6R2C for 900 series cases if validation shows improvement
2. If 6R2C doesn't improve enough, accept as known limitation
3. Document decision clearly with performance trade-offs
**Warning signs:** Annual energy errors >200% for high-mass cases while peak loads are accurate

### Pitfall 3: Performance Regression from Additional State
**What goes wrong:** 6R2C model requires 2x mass temperature updates, potentially halving throughput
**Why it happens:** Additional VectorField operations for envelope/internal masses increase computational load
**How to avoid:**
1. Profile both models with criterion benchmarks
2. Target >1,000 configs/sec throughput (current 5R1C: 2,575 configs/sec)
3. Optimize step_physics_6r2c with cached intermediate values
4. Consider SIMD optimization if performance is bottleneck
**Warning signs:** 6R2C throughput <500 configs/sec on 8-core CPU

### Pitfall 4: Low-Mass Case Regression
**What goes wrong:** 6R2C parameters tuned for high-mass break low-mass accuracy
**Why it happens:** Envelope mass fraction (0.75) and h_tr_me (100 W/K) optimized for concrete, not light construction
**How to avoid:**
1. Test 600 series cases with both models
2. Use case-specific parameters if needed (envelope_fraction=0.5 for low-mass)
3. Document parameter calibration per building type
**Warning signs:** 600 series annual energy error increases from baseline when 6R2C enabled

### Pitfall 5: BatchOracle Pattern Violation
**What goes wrong:** Nested parallelism in step_physics_6r2c breaks population-level parallelism
**Why it happens:** Adding rayon::par_iter() inside step_physics creates thread pool contention
**How to avoid:**
1. Never use rayon::par_iter() in step_physics_6r2c
2. Use VectorField element-wise operations (already parallel at population level via rayon)
3. Run pre-commit hook to catch violations
**Warning signs:** BatchOracle throughput drops dramatically, thread pool exhaustion

## Code Examples

Verified patterns from existing codebase:

### 6R2C Model Configuration
```rust
// Source: tests/test_6r2c_model.rs:22-28
let mut model = ThermalModel::new(1);
let envelope_fraction = 0.75;  // 75% of mass in envelope
let h_tr_me_value = 100.0;      // Conductance between masses (W/K)

model.configure_6r2c_model(envelope_fraction, h_tr_me_value);

// Verify configuration
assert!(model.is_6r2c_model());
assert_eq!(model.thermal_model_type, ThermalModelType::SixRTwoC);
```

### 6R2C Physics Solver Branch
```rust
// Source: src/sim/engine.rs:2460-2465
pub fn step_physics(&mut self, timestep: usize, outdoor_temp: f64) -> f64 {
    if self.is_6r2c_model() {
        self.step_physics_6r2c(timestep, outdoor_temp)
    } else {
        self.step_physics_5r1c(timestep, outdoor_temp)
    }
}
```

### ASHRAE 140 Validation Framework
```rust
// Source: src/validation/ashrae_140_cases.rs
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseBuilder};

// Get predefined case specification
let case_spec = ASHRAE140Case::Case900.spec();  // High-mass case
let mut model = ThermalModel::from_spec(&case_spec);

// Run annual simulation
let energy = model.solve_timesteps(8760, &surrogates, false);

// Compare against reference ranges
let validator = ASHRAE140Validator::new();
let result = validator.validate_case(ASHRAE140Case::Case900, &model);
assert!(result.annual_heating.in_range());
```

### Thermal Mass Energy Accounting
```rust
// Source: docs/6R2C_IMPLEMENTATION.md:235-240
// Net HVAC energy for step (when accounting enabled)
let net_hvac_energy_for_step = if self.thermal_mass_energy_accounting {
    hvac_output * dt - mass_energy_change.clone()
} else {
    hvac_output * dt
};
```

### Backward-Compatible Mass Temperature Update
```rust
// Source: docs/6R2C_IMPLEMENTATION.md:122-124
self.mass_temperatures = (
    self.envelope_mass_temperatures.clone() * self.envelope_thermal_capacitance.clone()
    + self.internal_mass_temperatures.clone() * self.internal_thermal_capacitance.clone()
) / total_cap;
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 5R1C only | 5R1C + 6R2C (optional) | v0.2 (Issue #296) | 6R2C implemented but not validated or adopted as default |
| Single mass node | Dual mass nodes (envelope/internal) | v0.2 | Better thermal lag representation for high-mass buildings |
| Fixed mass coupling | Configurable envelope_mass_fraction and h_tr_me | v0.2 | Flexibility for different building types |

**Deprecated/outdated:**
- **Plan 03-10 (6R2C Investigation)**: Found 6R2C "not significantly better" in early investigation; needs re-evaluation with proper calibration
- **Manual thermal mass correction**: Replaced by thermal_mass_energy_accounting flag for clean separation

## Open Questions

1. **6R2C Parameter Calibration for Different Building Types**
   - What we know: Default parameters (envelope_fraction=0.75, h_tr_me=100 W/K) tuned for high-mass concrete (900 series)
   - What's unclear: Optimal parameters for low-mass (600 series) and medium-mass cases
   - Recommendation: Run parameter sweep during validation, document case-specific values

2. **6R2C Numerical Stability Issues**
   - What we know: test_6r2c_model_single_timestep currently fails (temperatures don't change)
   - What's unclear: Root cause of failure (conductance values, heat transfer equations, or boundary conditions)
   - Recommendation: Debug step_physics_6r2c with detailed logging, verify all heat flow terms are non-zero

3. **6R2C Performance Impact**
   - What we know: Current 5R1C throughput is 2,575 configs/sec (Phase 9 results)
   - What's unclear: 6R2C throughput with dual-mass updates
   - Recommendation: Benchmark with criterion, measure impact on BatchOracle evaluate_population(1000)

4. **ASHRAE 140 Reference Implementation Comparison**
   - What we know: Reference programs (EnergyPlus, ESP-r, TRNSYS) achieve accurate annual energy
   - What's unclear: Whether they use 6R2C, 8R3C, or other approaches
   - Recommendation: Review reference implementation docs (if available) to understand thermal network structure

5. **Adoption Decision Criteria**
   - What we know: v0.3 is maintenance release, prefers stability over breaking changes
   - What's unclear: What error reduction threshold justifies default adoption
   - Recommendation: Define quantitative criteria (e.g., >50% error reduction, no regression, >1,000 configs/sec) and document decision process

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (native Rust testing) |
| Config file | None — tests directly use ASHRAE140Case enum |
| Quick run command | `cargo test test_6r2c_model --test` |
| Full suite command | `cargo test --all -- --nocapture` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MODEL6R2C-01 | 6R2C structure design | unit | `cargo test test_configure_6r2c_model --test` | ✅ tests/test_6r2c_model.rs |
| MODEL6R2C-02 | 6R2C solver implementation | unit | `cargo test test_6r2c_model_single_timestep --test` | ✅ tests/test_6r2c_model.rs (currently failing) |
| MODEL6R2C-03 | Comparison vs 5R1C for high/low-mass cases | integration | `cargo test ashrae_140_case_900 --test` | ✅ tests/ashrae_140_case_900.rs |
| MODEL6R2C-04 | Findings documentation | manual | N/A (documentation review) | ✅ docs/6R2C_IMPLEMENTATION.md |
| MODEL6R2C-05 | Adoption decision | manual | N/A (decision document) | ❌ docs/6R2C_DECISION.md (to be created) |

### Sampling Rate
- **Per task commit:** `cargo test test_6r2c_model --test`
- **Per wave merge:** `cargo test --all -- --nocapture`
- **Phase gate:** Full ASHRAE 140 suite for both 5R1C and 6R2C models before decision

### Wave 0 Gaps
- [ ] Fix failing test_6r2c_model_single_timestep (temperatures not updating)
- [ ] Add 6R2C benchmarks (criterion suite comparing 5R1C vs 6R2C throughput)
- [ ] Create 6R2C-specific ASHRAE 140 validation tests (900 series with 6R2C enabled)
- [ ] Add parameter sweep tests for envelope_mass_fraction and h_tr_me values
- [ ] Document decision framework for MODEL6R2C-05 adoption criteria

## Sources

### Primary (HIGH confidence)
- **Existing codebase** - src/sim/engine.rs (6R2C implementation, ThermalModelType enum, step_physics_6r2c method)
- **Existing codebase** - tests/test_6r2c_model.rs (11 6R2C tests, 1 currently failing)
- **Existing documentation** - docs/6R2C_IMPLEMENTATION.md (comprehensive 6R2C design and implementation details)
- **Existing documentation** - docs/KNOWN_LIMITATIONS.md (5R1C limitations, annual energy over-prediction data)
- **Existing codebase** - src/validation/ashrae_140_cases.rs (ASHRAE 140 test case specifications)
- **Existing documentation** - docs/ASHRAE_140_5R1C_MODEL.md (5R1C thermal network equations)
- **Existing codebase** - Cargo.toml (dependency versions, feature flags)

### Secondary (MEDIUM confidence)
- **Phase 9 benchmark results** - .planning/phases/09-Performance-Optimization/09-05-SUMMARY.md (2,575 configs/sec baseline)
- **ASHRAE 140 results** - docs/ASHRAE140_RESULTS.md (current validation status: 28.1% pass rate, 5R1C limitations)
- **Project instructions** - CLAUDE.md (BatchOracle pattern, performance requirements, critical conventions)

### Tertiary (LOW confidence)
- **None** - All findings based on existing codebase and documentation; no external web search required (web search service unavailable)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All dependencies already in Cargo.toml, well-established Rust ecosystem
- Architecture: HIGH - 6R2C implementation exists in codebase, patterns documented
- Pitfalls: HIGH - Failing test (test_6r2c_model_single_timestep) identified root cause area, known 5R2C limitations documented
- Validation: HIGH - ASHRAE 140 framework exists, test infrastructure mature from Phases 8-11

**Research date:** 2026-03-13
**Valid until:** 2026-04-13 (30 days - stable domain, existing implementation)
