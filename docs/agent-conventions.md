# Agent Conventions

> **TL;DR**: Standard conventions for physics code, Rust code, tests, and APIs in the Fluxion project.
> **Key decisions**: Physics first | Rust idioms | Test rigor | API consistency
> **Owned by**: Wave orchestrator
> **Reviewed**: 2026-07-13

## Overview

This document establishes conventions that all agents must follow when implementing issues for Wave 2 and beyond.

---

## 1. Physics Code Conventions

When working on building energy modeling (BEM) physics code:

### Mathematical Reasoning
- **Always use Python to verify calculations** — never attempt mental arithmetic for physics/math
- Use the `ctx_execute` tool with `language: "python"` for all numerical verification
- Reference ASHRAE standards with formula citations, not just the result

### Unit Conversions
- Explicitly convert units at module boundaries
- Document conversion factors in comments at point of use
- Use Python to verify conversion factors before implementing

### Validation
- Each physics module must match EnergyPlus within 1% tolerance on isolated scenarios
- Test against reference data in `tests/reference_data/` before integration
- No ASHRAE 140 system-level tests until individual modules pass

### ASHRAE Standards Reference
| Standard | Application | Citation |
|----------|-------------|----------|
| 90.1 | Envelope and equipment | Chapter 5 |
| 62.1 | Ventilation | Section 6.2 |
| 55 | Thermal comfort | Section 5.2 |
| 14 | Measurement and verification | Section 2 |

---

## 2. Rust Conventions

### Code Style
- Follow `rustfmt` defaults
- Use `clippy` lints — no warnings allowed
- Prefer idiomatic Rust patterns over Java/Python patterns translated to Rust

### Error Handling
- Use `Result<T, E>` for fallible operations
- Use `anyhow::Result<T>` for application-level errors
- Use `thiserror` for library-level errors with distinct variants
- Never use `unwrap()` in production code
- Never use `expect()` with generic messages — provide context

### Naming
- `snake_case` for functions and variables
- `PascalCase` for types and enums
- `SCREAMING_SNAKE_CASE` for constants
- Prefix traits with `Async` if they contain async methods

### Module Organization
```
src/
  physics/       # Domain logic (solver traits, physics modules)
  sim/           # Simulation orchestration
  cli/           # Command-line interface
  api/           # HTTP API (if applicable)
tests/
  reference_data/  # EnergyPlus CSV reference data
  unit/            # Module-level unit tests
  integration/     # Cross-module integration tests
```

### Documentation
- Document all public APIs with doc comments
- Include examples in doc comments for complex functions
- Document数学 formulas when implementing physics

---

## 3. Test Conventions

### Test File Organization
- Unit tests in `tests/unit/` or alongside source files (`{module}_test.rs`)
- Integration tests in `tests/integration/`
- Reference data tests in `tests/reference_data/`

### Naming
- Test functions: `test_{what_is_being_tested}_{scenario}`
- Example: `test_solar_position_at_noon_summer_solstice`

### Test Structure
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_under_condition() {
        // Arrange
        let input = setup_test_input();
        
        // Act
        let result = function_under_test(input);
        
        // Assert
        assert_eq!(result, expected_value);
    }
}
```

### Reference Data Tests
- CSV files in `tests/reference_data/` for E+ comparison
- Run E+ to generate reference before committing new physics
- Tolerances: 1% for most cases, 5% for weather-dependent extremes

### Test Data
- Use realistic but synthetic data for unit tests
- Do not hardcode magic numbers — use named constants
- Include comments explaining why specific values were chosen

---

## 4. API Conventions

### REST API (if applicable)
- Use `POST` for creation, `GET` for retrieval, `PUT` for update, `DELETE` for deletion
- Return appropriate HTTP status codes
- Use JSON for request/response bodies
- Document with OpenAPI 3.0

### Error Responses
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human-readable description",
    "details": [
      {"field": "temperature", "issue": "must be positive"}
    ]
  }
}
```

### Versioning
- Version in URL path: `/api/v1/...`
- Maintain backward compatibility within major versions
- Deprecate endpoints with 12-month notice

### CLI Interface
- Use `clap` for CLI argument parsing
- Provide `--help` with clear usage examples
- Use exit codes: 0 for success, 1 for general error, 2 for usage error

---

## Convention Compliance

Before committing any code:
- [ ] Run `cargo fmt` and `cargo clippy -- -D warnings`
- [ ] Run `cargo test --lib` for unit tests
- [ ] Verify physics calculations with Python
- [ ] Check reference data tests pass
- [ ] Review `ARCHITECTURE.md` matches implementation

---

## Wave 2 Issues

| Issue | Convention Area | Status |
|-------|-----------------|--------|
| 1531 | agent-workflow.md | in-progress |
| 1533 | agent-conventions.md | in-progress |
| 1534 | scripts/README.md | in-progress |
