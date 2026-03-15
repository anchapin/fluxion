# Versioned Test Data Directory

This directory contains versioned test data for integration tests and regression testing.

## Structure

```
tests/data/
├── v0.4/          # v0.4 baseline data (reference results)
├── v0.5/          # v0.5 current data (work in progress)
└── latest/         # Symlink to current version (v0.5)
```

## Usage

Tests should specify version explicitly:

```rust
let weather_path = "tests/data/v0.5/denver.epw";
```

This ensures tests use consistent data even when new versions are added.

## Contents

- **v0.4/**: ASHRAE 140 reference results from v0.4 (baseline)
- **v0.5/**: Current test data for v0.5 development
- **latest/**: Symlink to `v0.5` for convenience

## Adding New Data

1. Create new version subdirectory (e.g., `v0.6/`)
2. Add test data files
3. Update `latest/` symlink to point to new version
4. Update tests to use new version if needed

## Environment Variable

For CI or local dev with custom data location:

```bash
export FLUXION_TEST_DATA_DIR=/path/to/custom/data
```

Default fallback: `tests/data/`
