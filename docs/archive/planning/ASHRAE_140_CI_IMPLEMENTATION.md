# Issue #241: Add ASHRAE 140 Validation to CI Pipeline

## Overview

This document details the implementation of comprehensive ASHRAE 140 validation in the CI pipeline to ensure ongoing validation coverage and prevent regressions.

## Implementation Status

### ✅ Completed

1. **Enhanced CI Workflow** (`.github/workflows/ashrae_140_validation.yml`)
   - Added support for both push and PR events
   - Added nightly scheduled validation (2 AM UTC)
   - Implemented concurrency controls to prevent duplicate runs
   - Added 30-minute timeout protection

2. **Result Parsing & Extraction**
   - Extracts case-by-case results from test output
   - Parses summary statistics (pass rate, MAE, failed count)
   - Stores results in JSON for further processing

3. **Markdown Report Generation**
   - Creates human-readable validation report
   - Includes summary table with pass/fail indicators
   - Shows individual case results with reference ranges
   - Includes metadata about test configuration

4. **Validation Pass Criteria**
   - Minimum 25% pass rate required for CI success
   - Allows warnings and failures during development phase
   - Prevents complete regressions

5. **PR Integration**
   - Automatically comments on PRs with validation results
   - Uses GitHub artifacts for result storage
   - 30-day retention policy for historical tracking

6. **Artifact Management**
   - Uploads validation output for manual inspection
   - Stores JSON results for dashboard/metrics integration
   - Keeps generated markdown report as comment

## Files Modified

### `.github/workflows/ashrae_140_validation.yml`

**Key Changes:**
- Enhanced to run on `main` and `develop` branches
- Added nightly schedule via cron
- Improved error handling and reporting
- Uses inline Python scripts for result extraction (no external dependencies)
- Generates comprehensive markdown report
- Implements fail conditions based on pass rate threshold

**Workflow Steps:**

1. **Checkout & Setup**
   - Clones repository
   - Installs Rust toolchain
   - Sets up caching for faster builds

2. **Run Validation Tests**
   - Executes ASHRAE 140 test suite in release mode
   - Captures all output to text file
   - Continues on test failure (to generate report)

3. **Extract Results** 
   - Parses test output with regex patterns
   - Extracts case-by-case results
   - Summarizes statistics (pass rate, MAE, etc.)
   - Saves to JSON for processing

4. **Generate Report**
   - Creates markdown report with summary table
   - Formats case results with pass/fail indicators
   - Includes reference ranges for comparison

5. **Validate Pass Criteria**
   - Checks if pass rate meets minimum threshold (25%)
   - Reports validation status
   - Exits with appropriate code

6. **Upload Artifacts**
   - Stores raw test output
   - Saves JSON results
   - Keeps markdown report for future access

7. **Comment on PR**
   - Posts validation report as PR comment
   - Uses gh CLI for GitHub interaction
   - Allows manual review of results

## Pass Criteria Logic

The workflow uses a phased approach to allow development while preventing regressions:

```
Phase 1 (Development): 25% pass rate minimum
Phase 2 (Validation): 50% pass rate target
Phase 3 (Production): 90%+ pass rate required
```

Current phase: **Development** (25% minimum)

To update phase, modify `MIN_PASS_RATE` in the "Check Validation Results" step.

## Testing Locally

### Run validation tests locally:
```bash
cargo test --test ashrae_140_validation --release -- --nocapture
```

### Simulate CI extraction:
```bash
cargo test --test ashrae_140_validation --release -- --nocapture > validation_output.txt
python3 << 'EOF'
import re
import json

with open('validation_output.txt', 'r') as f:
    output = f.read()

# Extract results using same patterns as CI
cases = {}
case_pattern = r"Case (\d+[A-Z]*): Heating=([\d.]+) \(Ref: ([\d.-]+)-([\d.-]+)\), Cooling=([\d.]+) \(Ref: ([\d.-]+)-([\d.-]+)\)"
for match in re.finditer(case_pattern, output):
    case_id, heating, heating_min, heating_max, cooling, cooling_min, cooling_max = match.groups()
    cases[case_id] = {
        'heating': float(heating),
        'heating_min': float(heating_min),
        'heating_max': float(heating_max),
        'cooling': float(cooling),
        'cooling_min': float(cooling_min),
        'cooling_max': float(cooling_max)
    }

print(json.dumps(cases, indent=2))
EOF
```

## Integration Points

### With Issue #240 (Peak Load Tracking)
- Peak load fields are available in validation output
- Ready for inclusion in validation criteria when implemented

### With Issue #235 (Case 600 Fix)
- Validates the fix by monitoring case results over time
- Pass rate improvements indicate successful fixes

### With Documentation (#243)
- ASHRAE 140 reference data embedded in workflow
- Results can feed into developer documentation

## Future Enhancements

1. **Dashboard Integration**
   - Post results to GitHub Pages
   - Create historical trend charts
   - Show regression detection

2. **Notification System**
   - Slack/email alerts on failures
   - Performance regression detection
   - Nightly report summaries

3. **Extended Metrics**
   - Peak load validation once issue #240 complete
   - Per-construction-type pass rates
   - Seasonal breakdown analysis

4. **Performance Tracking**
   - CI runtime monitoring
   - Build time trends
   - Cache effectiveness metrics

## Related Issues

- **Issue #235**: Case 600 baseline validation fix (BLOCKING)
- **Issue #240**: Peak load tracking (enhances validation)
- **Issue #243**: ASHRAE 140 reference manual documentation

## Notes

- The 25% pass rate threshold is intentionally permissive during development
- This allows incremental fixes without blocking PRs
- As physics improvements are made, the threshold should be gradually increased
- The workflow will fail hard if validation infrastructure breaks (test doesn't run)
- Report generation is robust and continues even if tests fail

## Maintenance

### CI Failure Troubleshooting

If the workflow fails in GitHub Actions:

1. Check the workflow run logs for Python/regex errors
2. Verify test output format matches expected patterns
3. Check for timeout issues (30 minute limit)
4. Look for cache corruption (clear cache if needed)

### Updating Pass Criteria

To adjust the pass rate threshold:

1. Edit `.github/workflows/ashrae_140_validation.yml`
2. Find the "Check Validation Results" step
3. Update `MIN_PASS_RATE` variable
4. Test locally first

### Adding New Test Cases

When new ASHRAE 140 cases are added:

1. Update `tests/ashrae_140_validation.rs` with new case
2. Update case pattern regex if output format changes
3. Add benchmark data to `src/validation/benchmark.rs`
4. Test extraction script with new cases

---

**Created**: 2026-02-17  
**Branch**: feat/issue-241-ashrae-140-ci-integration  
**Status**: Ready for PR
