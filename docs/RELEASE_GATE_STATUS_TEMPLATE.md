# Release Gate Status

**Version:** TODO:VERSION
**Release Date:** TODO:DATE
**Status:** TODO:OVERALL_STATUS

---

## Summary

| Category | Gate | Status | Value | Threshold |
|----------|------|--------|-------|-----------|
| Validation | Overall Pass Rate | TODO:PASS_RATE_STATUS | TODO:PASS_RATE% | ≥4.0% |
| Validation | Mean Absolute Error | TODO:MAE_STATUS | TODO:MAE% | ≤30.0% |
| Validation | Extreme Deviations | TODO:EXTREME_STATUS | TODO:EXTREME_COUNT | ≤2 |
| Benchmark | Throughput | TODO:THROUGHPUT_STATUS | TODO:THROUGHPUT configs/sec | ≥800 |
| Benchmark | Latency | TODO:LATENCY_STATUS | TODO:LATENCY ms/config | ≤10.0 |
| Benchmark | Multi-Zone | TODO:MULTIZONE_STATUS | TODO:MULTIZONE configs/sec | ≥500 |
| Drift | Pass Rate Change | TODO:DRIFT_PR_STATUS | TODO:DRIFT_PR_CHANGE pp | ≤2.0 pp |
| Drift | MAE Change | TODO:DRIFT_MAE_STATUS | TODO:DRIFT_MAE_CHANGE pp | ≤5.0 pp |

---

## Validation Gates

### Overall Pass Rate

**Status:** TODO:PASS_RATE_STATUS

The ASHRAE 140 validation pass rate measures the percentage of test cases that fall within acceptable tolerance bands (NMABE and CV(RMSE) ≤ certain thresholds).

- **Current:** TODO:PASS_RATE%
- **Required:** ≥4.0%
- **Target:** ≥12.5%

> Note: Issue #497 temporarily lowered threshold to 4% while root cause of high-mass thermal modeling issues is investigated.

### Mean Absolute Error (MAE)

**Status:** TODO:MAE_STATUS

Mean Absolute Error across all validation cases.

- **Current:** TODO:MAE%
- **Required:** ≤30.0%

### Extreme Deviations

**Status:** TODO:EXTREME_STATUS

Cases with deviations exceeding 150%.

- **Current:** TODO:EXTREME_COUNT cases
- **Limit:** ≤2 cases

---

## Benchmark Gates

### Throughput

**Status:** TODO:THROUGHPUT_STATUS

Batch simulation throughput (configs/second).

- **Current:** TODO:THROUGHPUT configs/sec
- **Required:** ≥800 configs/sec
- **Absolute Minimum:** 100 configs/sec

### Latency

**Status:** TODO:LATENCY_STATUS

Single configuration evaluation latency.

- **Current:** TODO:LATENCY ms/config
- **Required:** ≤10.0 ms/config

### Multi-Zone Scaling

**Status:** TODO:MULTIZONE_STATUS

Throughput for 10-zone simulation.

- **Current:** TODO:MULTIZONE configs/sec
- **Required:** ≥500 configs/sec

---

## Drift Gates

Drift gates detect unexpected changes in validation results compared to the baseline.

### Pass Rate Change

**Status:** TODO:DRIFT_PR_STATUS

Change in pass rate compared to baseline.

- **Current Change:** TODO:DRIFT_PR_CHANGE percentage points
- **Allowed:** ≤2.0 percentage points
- **Baseline:** TODO:BASELINE_PR%

### MAE Change

**Status:** TODO:DRIFT_MAE_STATUS

Change in MAE compared to baseline.

- **Current Change:** TODO:DRIFT_MAE_CHANGE percentage points
- **Allowed:** ≤5.0 percentage points
- **Baseline:** TODO:BASELINE_MAE%

---

## Release Requirements

| Release Type | Validation | Benchmark | Drift |
|--------------|-----------|-----------|-------|
| Major | ✅ Required | ✅ Required | ✅ Required |
| Minor | ✅ Required | ✅ Required | ✅ Required |
| Patch | ⚠️ Relaxed (≥2%) | ✅ Required | ✅ Required |

---

## Actions

If any gate shows ❌:

1. **Validation gates failed:**
   - Review `validation_results.json` for failing cases
   - Check for recent changes affecting thermal modeling
   - Run `cargo test --test ashrae_140_validation --release -- --nocapture` locally
   - Consider whether a new baseline is appropriate (use `--update-baseline`)

2. **Benchmark gates failed:**
   - Review `benchmark_results.json` for performance regression
   - Check for recent changes affecting simulation performance
   - Run `cargo bench --bench performance` locally

3. **Drift gates failed:**
   - Review changes since last baseline
   - Update baseline if changes are intentional: `python scripts/release_gate_checker.py --update-baseline`
   - Investigate unexpected drift

---

## Verification Command

To verify gate status locally:

```bash
# Check all gates
python scripts/release_gate_checker.py

# Check specific results
python scripts/release_gate_checker.py --validation-results validation_results.json --benchmark-results benchmark_results.json

# Update baseline after confirming changes are intentional
python scripts/release_gate_checker.py --update-baseline

# Generate markdown report
python scripts/release_gate_checker.py --markdown > RELEASE_GATE_STATUS.md
```

---

*Generated from `release_gates.yaml` configuration (Issue #505)*
