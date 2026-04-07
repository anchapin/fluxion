# Phase 40: Case Expansion Foundation

## Phase Overview

**Phase:** 40-case-expansion-foundation
**Milestone:** v1.1 (ASHRAE 140 Completion)
**Goal:** Extend ASHRAE 140 case support and establish cross-validation framework

**Depends on:** Phase 39 (v1.0 completion)
**Requirements:** [CASE-01, CASE-02, CASE-03, CROSS-01, CROSS-02]

## What This Phase Accomplishes

This phase establishes the foundation for comprehensive ASHRAE 140 validation by:

1. **Expanding Case Coverage:** Adding support for ASHRAE 140 Cases 800-810 (HVAC equipment validation) and 195-470 (diagnostic validation)
2. **Enhancing Reference Data:** Extending the reference database to include new validation cases
3. **Enabling Cross-Validation:** Implementing framework for comparing Fluxion results against EnergyPlus and TRNSYS references
4. **Maintaining Performance:** Ensuring validation suite maintains <50ms/timestep performance targets

## Success Criteria (What Must Be TRUE)

### Case Expansion
1. **User can run ASHRAE 140 Cases 800-810** (HVAC equipment validation)
2. **User can run ASHRAE 140 Cases 195-470** (diagnostic validation)
3. **User can access extended reference database** for new cases

### Cross-Validation Framework
4. **User can compare Fluxion results against EnergyPlus references**
5. **User can compare Fluxion results against TRNSYS references**

### Performance
6. **Validation suite maintains <50ms/timestep performance** for expanded case set

## Key Components

### ASHRAE 140 Case Expansion
- **Cases 800-810:** HVAC equipment validation cases
- **Cases 195-470:** Diagnostic validation cases
- **Reference Database:** Extended CSV/JSON reference data files
- **Case Builder:** Modular case definition system

### Cross-Validation Framework
- **EnergyPlus Adapter:** File-based comparison with EnergyPlus results
- **TRNSYS Adapter:** File-based comparison with TRNSYS results
- **Standardized Output:** Common format for cross-validation results
- **Tolerance Configuration:** Per-tool tolerance bands

## Implementation Approach

### Case Expansion
1. **Extend ASHRAE140Case enum** with new case variants
2. **Implement CaseBuilder methods** for new case series
3. **Add reference data loading** for expanded cases
4. **Create validation tests** for new cases

### Cross-Validation Framework
1. **Design adapter interface** for external tools
2. **Implement EnergyPlus adapter** with file-based comparison
3. **Implement TRNSYS adapter** with file-based comparison
4. **Add cross-validation CLI commands**
5. **Create comparison reports**

### Performance Optimization
1. **Profile new validation cases** individually
2. **Apply targeted optimizations** where needed
3. **Monitor CI/CD performance impact**

## Technical Requirements

### Case Expansion
- **Case Definition:** Enum extension with proper serialization
- **Reference Data:** CSV/JSON format with hourly values
- **Test Coverage:** Comprehensive validation tests for all new cases
- **Documentation:** Case-specific documentation and examples

### Cross-Validation
- **Adapter Interface:** Trait-based design for extensibility
- **File Formats:** Support EnergyPlus/TRNSYS output formats
- **Comparison Methods:** Statistical analysis (RMSE, percentage difference)
- **Reporting:** Standardized validation report format

## Research Context

From `.planning/research/SUMMARY.md`:
- **Stack:** Rust 1.70+, PyO3 0.20+, Rayon 1.8+, ONNX Runtime 2.0.0-rc.10
- **Architecture:** ASHRAE140CaseExpansion, CrossValidationFramework components
- **Pitfalls:** Avoid monolithic case integration, tight coupling with external tools
- **Patterns:** Modular case integration, adapter-based cross-validation

From `.planning/research/ARCHITECTURE.md`:
- **ASHRAE140CaseExpansion:** Extends ASHRAE140Case enum with new variants
- **CrossValidationFramework:** Adapter pattern for external tool integration
- **PerformanceOptimizer:** Maintains <50ms/timestep through targeted optimizations

## Known Challenges

1. **Case Integration Complexity:** Ensuring modular integration of 20+ new cases
2. **External Tool Compatibility:** Handling different output formats from EnergyPlus/TRNSYS
3. **Performance Maintenance:** Preventing regressions with expanded validation suite
4. **Reference Data Quality:** Ensuring accurate reference data for new cases
5. **Validation Consistency:** Maintaining consistent validation approach across all cases

## Out of Scope

- **ESP-r Integration:** Deferred to Phase 42
- **Surrogate-Assisted Validation:** Deferred to Phase 43
- **Thermal Mass Diagnostics:** Deferred to Phase 41
- **Comprehensive Automation:** Deferred to Phase 42
- **Advanced Performance Optimization:** Deferred to Phase 43

## Decisions Made

### From Research Phase
- **Modular Integration:** Cases integrated by series (800-810, 195-470) with separate validation
- **Adapter Pattern:** Cross-validation uses file-based exchange with mock adapters
- **Conditional Optimization:** Performance optimizations applied based on profiling data
- **Standardized Reporting:** Common validation report format across all tools

### Technology Stack
- **Rust 1.70+:** Core implementation language
- **PyO3 0.20+:** Python bindings for external tool integration
- **Rayon 1.8+:** Parallel execution for validation suite
- **serde 1.0+:** Validation report serialization

## Next Steps

1. **Planning:** Break down implementation into executable tasks
2. **Implementation:** Build case expansion and cross-validation framework
3. **Testing:** Validate all new cases and cross-validation functionality
4. **Performance:** Ensure <50ms/timestep targets are maintained
5. **Documentation:** Document new cases and cross-validation process

## Resources

- ASHRAE Standard 140-2017 specification
- EnergyPlus/TRNSYS reference documentation
- Existing Fluxion validation framework
- Research documentation in `.planning/research/`
- Phase 40 research summary in `.planning/research/SUMMARY.md`
