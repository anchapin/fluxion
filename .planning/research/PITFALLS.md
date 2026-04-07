# Domain Pitfalls

**Domain:** Building Energy Modeling - ASHRAE 140 Validation Expansion
**Researched:** 2026-04-07

## Critical Pitfalls

Mistakes that cause rewrites or major issues.

### Pitfall 1: Monolithic Case Integration

**What goes wrong:** Adding all new ASHRAE 140 cases in a single large commit without modular organization

**Why it happens:** Developer tries to implement all 800-810 and 195-470 cases at once

**Consequences:**
- Difficult to debug which specific case is failing
- Hard to isolate performance regressions
- CI/CD pipelines become unstable
- Code review complexity increases exponentially

**Prevention:**
- Organize cases by series (800-810 HVAC cases, 195-470 diagnostic variants)
- Implement each series with separate validation
- Use feature flags for incremental rollout
- Profile each new case individually

**Detection:**
- Large commits with multiple case implementations
- CI/CD pipeline time increases significantly
- Difficulty isolating test failures to specific cases

### Pitfall 2: Tight Coupling with External Tools

**What goes wrong:** Direct integration with EnergyPlus/TRNSYS/ESP-r binaries in validation code

**Why it happens:** Developer wants to automate cross-validation by calling external tools directly

**Consequences:**
- Creates platform-dependent build requirements
- Makes validation fragile (external tool versions, paths, licenses)
- Violates separation of concerns
- Difficult to test in CI/CD environments

**Prevention:**
- Use adapter pattern with clear interfaces
- Treat external tool outputs as inputs (file-based exchange)
- Implement mock adapters for testing
- Document external tool requirements separately

**Detection:**
- Build scripts with complex external tool dependencies
- Validation tests that require specific software installations
- Platform-specific code paths in validation framework

### Pitfall 3: Global Physics Changes for High-Mass Buildings

**What goes wrong:** Modifying core physics that affects all building types when trying to fix high-mass accuracy

**Why it happens:** Developer doesn't use conditional logic based on ConstructionType

**Consequences:**
- Breaks existing low-mass validation (regression)
- Violates ASHRAE 140 compliance for validated cases
- Creates unpredictable behavior across building types
- Requires complete re-validation of all cases

**Prevention:**
- Use conditional logic based on ConstructionType enum
- Maintain separate code paths for low-mass vs high-mass
- Implement feature flags for physics enhancements
- Test both building types in CI/CD

**Detection:**
- Changes to core ThermalModel::step_physics without ConstructionType checks
- Low-mass validation tests start failing
- Performance characteristics change across all cases

### Pitfall 4: Performance Regression in Validation Suite

**What goes wrong:** Adding new cases without considering performance impact, making CI/CD pipelines too slow

**Why it happens:** Developer focuses on functionality without profiling

**Consequences:**
- CI/CD pipelines exceed time limits
- Developer productivity decreases (waiting for validation)
- Reduced test coverage due to time constraints
- Difficult to maintain <50ms/timestep target

**Prevention:**
- Profile each new case individually
- Apply targeted optimizations (surrogates, CTA, Rayon)
- Monitor CI/CD impact with each addition
- Set performance budgets per case type

**Detection:**
- CI/CD pipeline time increases significantly
- Individual case execution exceeds 50ms/timestep
- Validation suite takes >30 minutes to complete

## Moderate Pitfalls

### Pitfall 5: Incomplete Cross-Validation Implementation

**What goes wrong:** Implementing cross-validation framework but only supporting one external tool

**Why it happens:** Developer focuses on EnergyPlus integration first, plans to add others later

**Consequences:**
- Limited comparison capabilities
- Biased validation results
- Incomplete compliance verification
- Difficult to add new tools later

**Prevention:**
- Design adapter interface to support multiple tools from start
- Implement mock adapters for all target tools
- Use dependency injection for tool selection
- Document extension points clearly

### Pitfall 6: Over-Optimizing Before Validation

**What goes wrong:** Applying aggressive optimizations before validating correctness

**Why it happens:** Developer prioritizes performance over accuracy

**Consequences:**
- Optimized but incorrect results
- Difficult to debug physics issues
- Wasted optimization effort
- Validation failures that are hard to trace

**Prevention:**
- Validate correctness first, then optimize
- Use reference implementations for comparison
- Profile to identify actual bottlenecks
- Apply optimizations incrementally

### Pitfall 7: Ignoring Thermal Mass Separation

**What goes wrong:** Not separating 5R1C and CTF energy contributions in high-mass buildings

**Why it happens:** Developer applies global corrections without considering different physics models

**Consequences:**
- Over-correction of CTF contributions
- Under-correction of 5R1C limitations
- Inconsistent behavior across building types
- Difficult to tune corrections properly

**Prevention:**
- Separate energy contributions by physics model
- Apply corrections only to appropriate components
- Document correction rationale clearly
- Test with and without corrections

## Minor Pitfalls

### Pitfall 8: Inconsistent Case Naming

**What goes wrong:** Using inconsistent naming conventions for new ASHRAE 140 cases

**Why it happens:** Different developers implement different case series

**Consequences:**
- Difficult to find and organize cases
- Inconsistent documentation
- Confusing for users
- Hard to maintain

**Prevention:**
- Follow existing naming patterns (Case800, Case801, etc.)
- Document naming conventions
- Use consistent prefixes and numbering
- Review case names in PR process

### Pitfall 9: Missing Case Documentation

**What goes wrong:** Implementing new cases without proper documentation

**Why it happens:** Developer focuses on implementation, plans to document later

**Consequences:**
- Users don't understand case purpose
- Difficult to debug failures
- Incomplete validation reports
- Knowledge loss over time

**Prevention:**
- Document each case purpose and expected results
- Include reference values and tolerance ranges
- Add examples of proper usage
- Link to ASHRAE 140 specifications

### Pitfall 10: Hardcoded Reference Values

**What goes wrong:** Embedding reference values directly in test assertions

**Why it happens:** Developer copies values from ASHRAE 140 documentation

**Consequences:**
- Difficult to update when standards change
- Inconsistent with centralized benchmark database
- Violates DRY principles
- Error-prone maintenance

**Prevention:**
- Centralize all reference values in benchmark database
- Use descriptive constants for magic numbers
- Document source of each reference value
- Automate reference value updates

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| **Case Expansion** | Monolithic integration, inconsistent naming | Modular implementation, naming conventions |
| **Cross-Validation** | Tight coupling, incomplete implementation | Adapter pattern, comprehensive design |
| **High-Mass Physics** | Global changes, ignoring separation | Conditional logic, energy contribution separation |
| **Performance Optimization** | Premature optimization, regressions | Validate first, profile-based optimization |

## Sources

- ASHRAE 140 validation best practices (MEDIUM confidence)
- EnergyPlus integration patterns (MEDIUM confidence)
- Existing Fluxion architecture analysis (HIGH confidence)
- Performance optimization case studies (HIGH confidence)
