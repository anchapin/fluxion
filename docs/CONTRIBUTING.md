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

### Running CI locally with `act`

- **Purpose:** Run GitHub Actions workflows locally using the `act` CLI to reproduce CI jobs (useful for fast iterations and debugging).
- **Install:** Follow `act` installation instructions: https://github.com/nektos/act#installation
- **Example (macOS on Apple Silicon / ARM):**

```bash
act -j coverage \
  -W .github/workflows/code-coverage.yml \
  --container-architecture linux/arm64 \
  -P ubuntu-latest=catthehacker/ubuntu:act-latest
```

- **Notes:**
  - Use `--container-architecture linux/arm64` on Apple Silicon (M1/M2) to match the ARM environment.
  - The `-P ubuntu-latest=...` mapping sets the docker image `act` will use for the `ubuntu-latest` runner label; the `catthehacker/ubuntu:act-latest` image is commonly used and includes required tooling.
  - If you run into permission or sandboxing issues, ensure Docker Desktop is running and you have enough resources allocated.
  - For other jobs replace `-j coverage` with the job id from the workflow file or omit `-j` to run the default workflow.

#### Running Other Jobs

- **Run a different job by id:** find the job id under `jobs:` in the workflow YAML and pass it with `-j`:

```bash
act -j lint \
  -W .github/workflows/ci.yml \
  --container-architecture linux/arm64 \
  -P ubuntu-latest=catthehacker/ubuntu:act-latest
```

- **Run the entire workflow (all jobs):** omit `-j` and specify the workflow file:

```bash
act -W .github/workflows/code-coverage.yml \
  --container-architecture linux/arm64 \
  -P ubuntu-latest=catthehacker/ubuntu:act-latest
```

- **Provide an event payload:** simulate a specific event (e.g., `push`) with `-e` and a JSON file:

```bash
act -e tests/fixtures/push-event.json -W .github/workflows/ci.yml
```

- **Pass secrets and environment variables:** use `-s NAME=VALUE` for secrets or `--env-file .env` for environment variables:

```bash
act -j coverage -s GITHUB_TOKEN=ghp_xxx --env-file .secrets.env -W .github/workflows/code-coverage.yml
```

#### Troubleshooting `act`

- **Docker not running / connection errors:** ensure Docker Desktop is running and responsive. Restart Docker if mounts or networking fail.
- **Image/platform mismatches on Apple Silicon:** prefer `--container-architecture linux/arm64` and an ARM-compatible image (`-P ubuntu-latest=catthehacker/ubuntu:act-latest`). If an image is missing for ARM, pre-pull or choose an image that supports `linux/arm64`.
- **Permission / volume mount issues on macOS:** some actions rely on bind mounts that require Docker permissions — try enabling `--privileged` (note: increases privileges) or adjust Docker Desktop file sharing settings.
- **Slow or resource-heavy workflows:** increase Docker Desktop resources (CPUs, memory) or limit concurrent jobs. For heavy builds prefer running in the real CI runner.
- **Missing secrets or auth failures:** `act` does not automatically provide GitHub secrets. Supply required secrets with `-s` or `--env-file`, or create an `.actrc`/`.secrets.env` used locally.
- **Actions that rely on GitHub-hosted services (e.g., `actions/cache`, `setup-remote-docker`) may behave differently locally:** treat `act` as a debugging tool — final verification should still run in GitHub Actions.
- **Verbose logs:** add `-v` or `--verbose` to `act` to see more output and help diagnose failures.

If you hit an error you can't resolve locally, capture the `act` output and open an issue or include it in your PR so maintainers can reproduce it.


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
