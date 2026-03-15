# Versioned Test Data Directory

This directory contains versioned test data for integration tests and regression testing.

## Overview

Fluxion uses versioned test data to ensure reproducibility and prevent breaking old tests when adding new data. Each version represents a stable snapshot of reference results and test fixtures at a specific release.

## Structure

```
tests/data/
├── v0.4/                    # v0.4 baseline data (18/18 cases passing)
│   ├── ashrae_140/          # ASHRAE 140 test case data
│   ├── weather/             # EPW weather files
│   └── reference_results.yaml
├── v0.5/                    # v0.5 current data (work in progress)
│   ├── ashrae_140/
│   ├── weather/
│   └── reference_results.yaml
├── latest/                  # Symlink to current version (v0.5)
└── README.md                # This file
```

## Usage

### Referencing Versioned Data

Tests should specify version explicitly to ensure reproducibility:

```rust
// Weather data
let weather_path = "tests/data/v0.5/denver.epw";

// ASHRAE 140 reference results
let ref_path = "tests/data/v0.4/reference_results.yaml";
```

**Always specify the version** (e.g., `v0.4`, `v0.5`) to prevent tests from breaking when new versions are added.

### Python Test Usage

```python
# Weather data
weather_file = "tests/data/v0.5/denver.epw"

# Reference results
import yaml
with open("tests/data/v0.4/reference_results.yaml") as f:
    refs = yaml.safe_load(f)
```

### Using the `latest` Symlink

For convenience, use the `latest/` symlink when you always want the current version:

```rust
let weather_path = "tests/data/latest/weather/denver.epw";
```

**Note:** The `latest` symlink points to the current development version (v0.5 during v0.5 development). Use explicit versions for regression tests that should not change.

## Versioning Strategy

### Why Versioned Subdirectories?

1. **Reproducibility:** Tests always use the same reference data, even after adding new versions
2. **Regression Testing:** Can run tests against old reference data to catch regressions
3. **CI/CD Stability:** CI pipelines use pinned versions, avoiding breaking changes
4. **Parallel Development:** Multiple versions can coexist during development

### Version Lifecycle

- **v0.4:** Baseline release (18/18 ASHRAE 140 cases passing)
- **v0.5:** Current development (work in progress)
- **v0.6+:** Future versions (to be created)

When a version is released:
1. Finalize reference_results.yaml with actual validation results
2. Update `latest/` symlink to point to new version
3. Optional: Mark previous versions as `stable` or `baseline`

## Adding New Data

### Creating a New Version

1. Create new version subdirectory:
   ```bash
   mkdir -p tests/data/v0.6/ashrae_140 tests/data/v0.6/weather
   ```

2. Add `.gitkeep` files to preserve structure:
   ```bash
   touch tests/data/v0.6/ashrae_140/.gitkeep
   touch tests/data/v0.6/weather/.gitkeep
   ```

3. Populate reference_results.yaml:
   - Copy from previous version as template
   - Update `version` and `status` fields
   - Add actual validation results

4. Add test data files (EPW, YAML, etc.)

5. Update `latest/` symlink:
   ```bash
   rm tests/data/latest
   ln -s v0.6 tests/data/latest
   ```

6. Update tests to use new version if needed:
   - Search: `tests/data/v0.5/`
   - Replace: `tests/data/v0.6/`

### Updating Existing Version

For in-progress versions (e.g., v0.5), you can update files directly:

```bash
# Add new EPW file
cp path/to/weather_file.epw tests/data/v0.5/weather/

# Update reference results
edit tests/data/v0.5/reference_results.yaml
```

**Never modify released versions** (e.g., v0.4) - create a new version instead.

## Environment Variable

### Custom Data Location

For CI or local development with custom data location:

```bash
export FLUXION_TEST_DATA_DIR=/path/to/custom/data
```

Tests should use this variable when present:

```rust
let data_dir = std::env::var("FLUXION_TEST_DATA_DIR")
    .unwrap_or_else(|_| "tests/data".to_string());

let weather_path = format!("{}/v0.5/denver.epw", data_dir);
```

### Fallback Behavior

- If `FLUXION_TEST_DATA_DIR` is set: Use custom location
- If not set: Use default `tests/data/`
- If custom location doesn't exist: Fall back to `tests/data/`

## File Contents

### EPW Weather Files

Location: `tests/data/v{version}/weather/`

- **Format:** EnergyPlus Weather (EPW) format
- **Content:** Hourly weather data (temperature, humidity, solar radiation, etc.)
- **Naming:** `{location}.epw` (e.g., `denver.epw`, `chicago.epw`)
- **Size:** ~1.7 MB per file (8760 hours of data)

**Example locations:**
- `denver.epw` - Denver, CO (continental climate)
- `chicago.epw` - Chicago, IL (cold climate)
- `miami.epw` - Miami, FL (hot-humid climate)
- `san_francisco.epw` - San Francisco, CA (marine climate)

### ASHRAE 140 Reference Results

Location: `tests/data/v{version}/ashrae_140/reference_results.yaml`

- **Format:** YAML
- **Content:** Reference ranges for ASHRAE 140 validation cases
- **Structure:** Case IDs → Annual heating/cooling, peak loads, free-floating temps
- **Status:** `pass`, `fail`, `pending`, `diagnostic`

**YAML Structure:**

```yaml
version: "0.4"
status: "baseline"
cases:
  "600":
    description: "Low mass baseline"
    annual_heating:
      min: 5.5  # MWh
      max: 7.5
    annual_cooling:
      min: 8.0  # MWh
      max: 10.5
    status: "pass"
summary:
  total_cases: 18
  passing_cases: 18
  pass_rate: 1.0
```

### Test Fixtures

Location: `tests/data/v{version}/fixtures/` (future)

- **Purpose:** Reusable test scenarios and building configurations
- **Format:** YAML, JSON, or TOML
- **Usage:** Integration tests load fixtures for building scenarios

## Directory Patterns

### ASHRAE 140 Directory

```
tests/data/v{version}/ashrae_140/
├── reference_results.yaml      # Main reference data file
├── case_600_data.yaml         # Per-case detailed data (optional)
├── case_610_data.yaml
└── ...
```

### Weather Directory

```
tests/data/v{version}/weather/
├── denver.epw                 # Continental climate
├── chicago.epw                # Cold climate
├── miami.epw                  # Hot-humid climate
└── san_francisco.epw          # Marine climate
```

### Fixtures Directory (Future)

```
tests/data/v{version}/fixtures/
├── buildings/                 # Building configurations
│   ├── small_office.yaml
│   ├── medium_office.yaml
│   └── large_office.yaml
├── hvac/                      # HVAC system configurations
│   ├── vav.yaml
│   ├── cav.yaml
│   └── heat_pump.yaml
└── schedules/                 # Occupancy and operation schedules
    ├── office.yaml
    └── residential.yaml
```

## Troubleshooting

### Common Issues

**Issue: "File not found" error**

- Check version number: `ls tests/data/v0.5/weather/`
- Verify file exists: `find tests/data/ -name "denver.epw"`
- Check symlink: `readlink tests/data/latest`

**Issue: "Symlink broken"**

```bash
# Remove broken symlink
rm tests/data/latest

# Recreate with correct target
ln -s v0.5 tests/data/latest

# Verify
ls -la tests/data/latest
```

**Issue: Tests fail on CI but pass locally**

- Check CI environment: `echo $FLUXION_TEST_DATA_DIR`
- Verify data directory exists on CI: `ls tests/data/`
- Check file permissions: `ls -la tests/data/v0.5/`

**Issue: Reference results outdated**

- Check version status: `grep "status:" tests/data/v0.5/reference_results.yaml`
- If `work_in_progress`: Update with actual validation results
- If `baseline`: Don't modify - create new version

### Debug Commands

```bash
# Check directory structure
find tests/data/ -type d

# Count files per version
find tests/data/v0.4/ -type f | wc -l
find tests/data/v0.5/ -type f | wc -l

# Check symlink target
readlink tests/data/latest

# Verify YAML syntax
python -c "import yaml; yaml.safe_load(open('tests/data/v0.4/reference_results.yaml'))"

# Check EPW file integrity
head -n 1 tests/data/v0.5/weather/denver.epw
```

## Best Practices

1. **Always specify version:** Use `tests/data/v0.5/` not `tests/data/latest/` for regression tests
2. **Never modify released versions:** Create new version instead of editing v0.4
3. **Document changes:** Update README when adding new versions or changing structure
4. **Use `.gitkeep` files:** Preserve empty directories in git
5. **Validate YAML:** Use Python or online validator before committing
6. **Test symlink:** Verify `latest/` works on all platforms (Linux, macOS, Windows)

## Integration with Test Framework

### Rust Integration Tests

```rust
use std::path::Path;

#[test]
fn test_with_versioned_data() {
    let version = "v0.4";
    let weather_path = format!("tests/data/{}/weather/denver.epw", version);

    assert!(Path::new(&weather_path).exists());

    // Load and use weather data
    let weather = WeatherSource::from_epw(&weather_path).unwrap();
}
```

### Python Integration Tests

```python
import os
import yaml

def test_with_versioned_data():
    version = "v0.4"
    ref_path = f"tests/data/{version}/reference_results.yaml"

    with open(ref_path) as f:
        refs = yaml.safe_load(f)

    assert refs["status"] == "baseline"
    assert refs["summary"]["passing_cases"] == 18
```

## Related Documentation

- **ASHRAE 140 Validation:** `docs/ASHRAE140_RESULTS.md`
- **Integration Tests:** `tests/integration/`
- **Test Framework:** `src/testing/`
- **Validation Module:** `src/validation/`

## Maintenance

### Version Release Checklist

- [ ] All validation results updated in `reference_results.yaml`
- [ ] `status` field set to `baseline` or `stable`
- [ ] All required data files present (EPW, YAML, etc.)
- [ ] `.gitkeep` files in empty directories
- [ ] README updated with version info
- [ ] `latest/` symlink updated to point to new version
- [ ] Tests updated to use new version (if required)
- [ ] Documentation updated (ASHRAE140_RESULTS.md, etc.)

## Version History

| Version | Status | Date | Notes |
|---------|--------|------|-------|
| v0.4    | Baseline | 2026-03-15 | 18/18 ASHRAE 140 cases passing |
| v0.5    | Work in progress | 2026-03-15 | Current development version |

## Contributing

When adding test data:

1. Follow directory structure conventions
2. Use versioned subdirectories for new data
3. Update this README with new data descriptions
4. Validate YAML files before committing
5. Test on multiple platforms if using symlinks
6. Document data sources and provenance

---

**Questions?** See `docs/CONTRIBUTING.md` or open an issue on GitHub.
