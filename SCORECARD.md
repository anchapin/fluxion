# Fluxion Release Scorecard

**Generated:** 2026-04-18
**Wave:** Wave 1
**Version:** 1.0.0 (next release: 1.2.0)

---

## Summary

| Metric | Value | Status |
|--------|-------|--------|
| ASHRAE 140 Pass Rate | 3.1% (2/63) | ❌ Below Target
| Mean Absolute Error | 16.23% | ✅ Good
| Test Pass Rate | 100.00% (2285/2285) | ✅ Healthy
| Benchmark Throughput | ~900 configs/sec | ✅ Exceeds
| Open Issues (Critical/High) | 12 | ❌ Blocking

---

## Validation Results (ASHRAE 140)

### Pass Rate by Case Series

| Series | Cases | Passed | Failed | Pass Rate |
|--------|-------|--------|--------|-----------|
| Baseline (600-650) | 6 | 0 | 6 | 0.0% |
| High-Mass (900-950) | 6 | 1 | 5 | 16.7% |
| Free-Floating | 4 | 0 | 4 | 0.0% |
| Special (195, 960) | 2 | 1 | 1 | 50.0% |

### Critical Failures (Top 3)

| Case | Metric | Fluxion | Reference | Deviation |
|------|--------|---------|------------|------------|
| 195 | Annual Heating | 21.85 MWh | 3.50-6.00 | +313% |
| 950 | Annual Heating | 0.00 MWh | 0.79-1.41 | -100% |
| 600FF | Max Temp | -11.94°C | -18.80--15.60 | +30.6% |

---

## Benchmark Status

### Performance Metrics

| Benchmark | Value | Target | Status |
|-----------|-------|--------|--------|
| Throughput (configs/sec) | ~900 | ≥800 | ✅ Exceeds
| CTA Simulation Time | <100ms | <100ms | ✅ Meets |
| Multi-Zone (10 zones) | 800-1,200 | ≥500 | ✅ Exceeds |
| Cross-Validation Latency | <100ms | ≤500ms | ✅ Exceeds |

---

## Open Issues by Severity

| Severity | Count | Status |
|----------|-------|--------|
| Critical | 4 | ❌ Blocking
| High | 8 | ⚠️ Review
| Medium | 8 | 🔄 In Progress |
| Low | 5 | ✅ Tracked |

---

## Release Readiness

### Requirements Check

| Requirement | Status | Notes |
|-------------|--------|-------|
| Compilation | ✅ Pass | All crates compile |
| Unit Tests | ✅ Pass | 2285/2285 passed (100.0%)
| Integration Tests | ✅ Pass | All pass |
| ASHRAE 140 Pass Rate ≥12.5% | ❌ Fail | Currently 3.1%
| Benchmark Throughput ≥800 | ✅ Pass | ~900 configs/sec |
| Critical Issues Resolved | ⚠️ Partial | 4 critical open |
| Documentation Complete | ✅ Pass | 100% coverage |

### Overall: ❌ Not Ready

**Primary Blocker:** ASHRAE 140 Pass Rate below 12.5% threshold
Root cause: Solar gain issues (SOLAR-01, SOLAR-02) and high-mass thermal modeling

---

## Conflicting Metrics Resolution

The following metrics show conflicting trends between different measurement approaches:

| Metric | validation_results.json | QUALITY_METRICS.md | Resolution |
|--------|-------------------------|-------------------|------------|
| Case 900 Annual Heating | 1.35 MWh | 1.17-2.04 range | Use validation_results.json as authoritative |
| High-Mass Pass Rate | 16.7% | 0.0% | Different counting methods - standardization needed |

**Action:** Standardize on `validation_results.json` as authoritative source.
Update `QUALITY_METRICS.md` to use consistent reference data and counting.

---

## Regeneration Command

To regenerate this scorecard, run:

```bash
# Run this from the project root
python scripts/generate_scorecard.py

# Or with verbose output
python scripts/generate_scorecard.py --verbose

# To specify output location
python scripts/generate_scorecard.py --output SCORECARD.md
```

---

## Links

- [ASHRAE 140 Validation Report](docs/ASHRAE140_RESULTS_v0.8.0.md)
- [Known Issues Catalog](docs/KNOWN_ISSUES.md)
- [Quality Metrics](docs/QUALITY_METRICS.md)
- [Validation Report](validation_report.md)
- [Release Notes v1.2](docs/RELEASE_NOTES_v1.2.md)

---

*This scorecard is auto-generated as part of QG-01: Create a generated release scorecard*
