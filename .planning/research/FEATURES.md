# Feature Landscape

**Domain:** Building Energy Modeling - ASHRAE 140 Validation Expansion
**Researched:** 2026-04-07

## Table Stakes

Features users expect. Missing = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Additional ASHRAE 140 Cases** | Complete validation coverage required for compliance | Medium | Cases 800-810 (HVAC equipment), 195-470 (diagnostics) |
| **Cross-Validation Framework** | Industry standard to compare against EnergyPlus/TRNSYS/ESP-r | High | Adapter pattern for external tool integration |
| **High-Mass Accuracy Improvements** | Current 229-322% error makes high-mass buildings unusable | High | Conditional physics enhancements for concrete construction |
| **Performance Optimization** | Maintain <50ms/timestep for CI/CD viability | Medium | CTA optimizations, Rayon parallelism tuning |

## Differentiators

Features that set product apart. Not expected, but valued.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Multi-Reference Validation** | Per-program tolerance ranges (EnergyPlus vs TRNSYS vs ESP-r) | Medium | Enhanced MultiReferenceDB with program-specific criteria |
| **Automated Cross-Validation** | One-click comparison with all reference tools | High | External tool adapters with standardized output parsing |
| **Surrogate-Assisted Validation** | AI acceleration for complex HVAC cases | High | ONNX models for 800-810 series cases |
| **Thermal Mass Diagnostics** | Detailed breakdown of thermal mass contributions | Medium | Energy separation (5R1C vs CTF) with visualization |

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Direct EnergyPlus FFI** | Complex build dependencies, licensing issues | Use file-based exchange (EPW, IDF) via epw-rs |
| **TRNSYS Python API** | Python dependency conflicts, performance overhead | Use trnsys-deck-parser-rs for Rust-native parsing |
| **Global Physics Changes** | Could break existing low-mass validation | Use conditional logic based on ConstructionType |
| **Full 6R2C/8R3C Models** | Research shows no accuracy improvement | Focus on improving 5R1C with CTF corrections |

## Feature Dependencies

```
Additional ASHRAE 140 Cases → Cross-Validation Framework (need cases to validate)
High-Mass Accuracy → Additional Cases (Case 900 series depends on improvements)
Cross-Validation Framework → Multi-Reference Validation (enhanced validation needs framework)
Performance Optimization → All Features (everything must maintain performance targets)
```

## MVP Recommendation

Prioritize:
1. **Additional ASHRAE 140 Cases** - Foundation for all other features
2. **Cross-Validation Framework** - Core validation capability
3. **High-Mass Accuracy Improvements** - Critical for compliance

Defer: **Surrogate-Assisted Validation**: High complexity, can be added after core validation works

## Sources

- ASHRAE Standard 140-2017 requirements (HIGH confidence)
- EnergyPlus validation methodology (MEDIUM confidence)
- Existing Fluxion validation framework analysis (HIGH confidence)
- Performance profiling data from current implementation (HIGH confidence)
