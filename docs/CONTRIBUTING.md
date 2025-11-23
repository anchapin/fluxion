````markdown
# Contributing to Fluxion

Thank you for your interest in contributing to Fluxion! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive environment.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check existing issues to avoid duplicates
2. Provide a clear, descriptive title
3. Include steps to reproduce (for bugs)
4. Specify your environment (OS, Rust version, Python version)
5. Add relevant labels

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `develop`
2. **Follow the development workflow** (see Development Setup)
3. **Write or update tests** for your changes
4. **Ensure all checks pass**:
   ```bash
   cargo fmt && cargo clippy && cargo test
   ```
5. **Update documentation** if needed
6. **Write a clear PR description** explaining the "why" behind your changes
7. **Clean up temporary files** before committing (see Repository Hygiene)

**Note**: All PRs should be created against the `develop` branch. The `main` branch is reserved for releases.

## Development Setup

### Prerequisites

- **Rust**: Install via `rustup` (latest stable)
  ```bash
  rustup update
  rustup component add rustfmt clippy
  ```
- **Python**: 3.10+ required
- **maturin**: `pip install maturin`

### Local Development

```bash
# Clone and setup
git clone https://github.com/yourusername/fluxion.git
cd fluxion

# Ensure you're on the develop branch
git checkout develop

# First-time setup
cargo build
maturin develop

# Typical iteration
cargo fmt && cargo clippy
cargo test
```

## Code Style & Quality

### Formatting
- **Rust**: Run `cargo fmt` before committing
- **Python**: Follow PEP 8 style guide

### Linting
- Address all `cargo clippy` warnings
- Use `#[allow(...)]` sparingly with documentation

### Documentation
- Add doc comments to public functions/structs:
  ```rust
  /// Predicts thermal loads using the neural network surrogate.
  ///
  /// # Arguments
  /// * `current_temps` - Zone temperatures in Celsius
  ///
  /// # Returns
  /// Vector of predicted loads (W/m²) per zone
  pub fn predict_loads(&self, current_temps: &[f64]) -> Vec<f64> { ... }
  ```

## Testing Strategy

### Writing Tests

Place unit tests in the same file as implementation using `#[cfg(test)]` modules:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_model_energy_conservation() {
        let mut model = ThermalModel::new(10);
        // Test logic...
        assert!(condition, "failure message");
    }
}
```

### Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_name

# With output
cargo test -- --nocapture

# Single-threaded (for debugging)
cargo test -- --test-threads=1
```

### Physics Validation

- Use `Model::simulate(1, use_surrogates=false)` for single-year analytical validation
- Compare against baseline energies documented in `docs/Fluxion_PRD.md`
- Test batch operations with realistic population sizes (1000+)

## Commit Message Convention

Use semantic commit messages:

```
<type>(<scope>): <subject>

<body (optional)>
```

**Types**: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`

**Examples**:
- `feat(surrogate): integrate ONNX runtime session initialization`
- `perf(engine): reduce memory allocations in solve_timesteps`
- `test(batch-oracle): add population scaling validation`
- `fix(physics): correct window U-value calculation units`

## Repository Hygiene

### Before Committing

1. **Delete temporary files** generated during development
2. **Keep root directory clean** — only `README.md` should be in root (besides config files)
3. **Move development artifacts to `tmp/`** if they need to persist

Examples of files to clean up:
- `PRECOMMIT_*.md`, planning documents
- `.azure/plan.copilotmd`, deployment plans
- Temporary scripts or debug files

### Pre-commit Checklist

- [ ] Code formatted: `cargo fmt`
- [ ] No clippy warnings: `cargo clippy`
- [ ] All tests pass: `cargo test`
- [ ] Temporary files removed or moved to `tmp/`
- [ ] Root directory clean (only `README.md` and config files)
- [ ] Commit message follows convention
- [ ] Documentation updated (if applicable)

## Pull Request Checklist

- [ ] Tests added/updated for new functionality
- [ ] All tests pass: `cargo test`
- [ ] Code formatted: `cargo fmt`
- [ ] No clippy warnings: `cargo clippy`
- [ ] Documentation updated (doc comments, README if applicable)
- [ ] Commit messages follow convention
- [ ] PR description explains "why" not just "what"
- [ ] Temporary files cleaned up
- [ ] No unnecessary `.md` files in root

## Architecture Overview

See `docs/Fluxion_PRD.md` for:
- System architecture (BatchOracle pattern)
- Physics engine details (ThermalModel)
- AI surrogate integration (SurrogateManager)
- API reference

## Performance Considerations

### Critical Metrics
- **Per-configuration latency**: <100ms for single `solve_timesteps(8760)`
- **Throughput**: <100ms total for `evaluate_population(1000)`

### Optimization Guidelines
- Use `rayon::par_iter()` only at population level, not nested
- Minimize Python-Rust boundary crossings
- Avoid allocations in inner loops
- Test with `--release` profile

## Getting Help

- Check existing documentation in `docs/`
- Review the Copilot instructions file for architecture details
- Ask questions in PR comments or open a discussion issue

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

Thank you for contributing to Fluxion!
````