# Issue #1244: Crate Split Plan for Mutation Testing

## Problem

`cargo-mutants` requires ~28GB RAM to analyze fluxion's type hierarchies.
Standard CI runners have 7GB RAM, causing OOM kills during mutation testing.

## Root Cause

cargo-mutants performs deep type analysis of the entire crate including:
- All dependency type hierarchies
- Generic type instantiations
- Trait object traversals

Memory usage grows super-linearly with codebase size during analysis.

## Solution: Workspace Split

Split fluxion into a workspace with the main crate (analyzed) and
supporting crates (compiled but NOT analyzed).

```
fluxion/
├── fluxion-core/          # NOT analyzed by cargo-mutants
│   ├── ai/                # SurrogateManager, ONNX integration
│   ├── physics/           # Complex thermal physics types
│   └── validation/        # ASHRAE 140 case definitions
├── fluxion-cli/           # NOT analyzed
│   └── cli/               # Command-line interface
└── fluxion/              # ANALYZED by cargo-mutants
    ├── sim/              # Simulation logic (thermal model, HVAC)
    ├── weather/          # Weather data processing
    ├── solar/            # Solar calculations
    └── ...               # Depends on fluxion-core
```

## Why This Helps

1. **Reduced analysis scope**: cargo-mutants only analyzes `fluxion/` crate
2. **Dependencies compiled but not analyzed**: fluxion-core compiled as dependency
3. **Type hierarchy isolation**: Complex types in fluxion-core don't pollute analysis

## Phased Implementation Plan

### Phase 1: Identify Boundaries (1-2 days)

1. Map all import dependencies between modules
2. Identify circular dependencies that block extraction
3. Determine clean extraction boundaries

### Phase 2: Extract fluxion-core (2-3 days)

Create `fluxion-core` crate containing:
- `src/ai/` - All AI/surrogate code
- `src/physics/` - Complex physics types
- `src/validation/` - ASHRAE 140 cases

Key tasks:
```bash
mkdir -p fluxion-core/src
# Move and adapt modules
# Update Cargo.toml workspace config
# Fix import paths
# Test compilation
```

### Phase 3: Extract fluxion-cli (1 day)

Move `src/cli/` to separate crate if needed.

### Phase 4: Validate with cargo-mutants (1 day)

```bash
# Test memory usage
CARGO_BUILD_JOBS=1 cargo mutants --list

# Run mutation tests
cargo mutants -- --test integration
```

## Estimated Effort

| Phase | Duration | Risk |
|-------|----------|------|
| 1. Boundaries | 1-2 days | Low |
| 2. fluxion-core | 2-3 days | Medium |
| 3. fluxion-cli | 1 day | Low |
| 4. Validation | 1 day | Low |
| **Total** | **5-7 days** | - |

## Dependencies to Resolve

### Circular Dependencies (expected)
```
sim ↔ physics (thermal model uses physics CTA)
weather ↔ physics (weather uses solar angles)
validation ↔ sim (validation runs simulations)
```

**Resolution strategy**: Keep physics in main crate, extract only types.

### Import Path Changes

All internal imports must change from:
```rust
use crate::ai::surrogate::SurrogateManager;
```
to:
```rust
use fluxion_core::ai::surrogate::SurrogateManager;
```

**Tooling**: `rustfmt` + `cargo hakari` can help manage dependencies.

## Success Metrics

1. `cargo mutants --list` completes with <4GB RAM
2. `cargo mutants -- --test test_isolation` completes with <8GB RAM
3. All existing tests pass
4. No regression in CI build times

## Alternative: Incremental Approach

If full split is too invasive, consider:

1. **Move only AI module to fluxion-core** (smallest blast radius)
2. **Use cargo-mutants sharding** with `--shard 1/4` to limit scope
3. **Add pre-mutation analysis script** using ripr to identify critical seams

## References

- [cargo-mutants memory issues](https://github.com/sourcefrog/cargo-mutants/issues)
- [Rust workspace best practices](https://doc.rust-lang.org/cargo/reference/workspaces.html)
