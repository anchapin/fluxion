# Issue #509: Benchmark & Throughput Verification Report

**Issue:** QG-02: Normalize benchmark and throughput claims
**Repository Focus:** README.md, docs/ARCHITECTURE.md, throughput tests
**Status:** Work in progress (benchmark infrastructure fixed, docs partially updated)

---

## Summary

Issue #509 requires that "All public throughput numbers come from a named benchmark and are consistent across docs and tests."

After investigation, several inconsistencies were found and corrected.

---

## Findings

### 1. Benchmark Harness Issue

**Problem:** All benchmarks in `Cargo.toml` had `harness = true` which caused Criterion benchmarks to be treated as test harnesses, resulting in "0 tests" instead of running benchmarks.

**Fix Applied:** Changed `harness = true` to `harness = false` for all benchmark targets in `Cargo.toml`.

### 2. Documentation vs. Reality Inconsistencies

| Location | Original Claim | Actual Measured | Status |
|----------|---------------|-----------------|--------|
| README.md | 10,000+ configurations/sec | ~900 configs/sec (release) | **Fixed** |
| README.md | 1,237 configs/sec | ~900 configs/sec (release) | **Fixed** |
| SCORECARD.md | 1237 configs/sec | ~900 configs/sec (release) | **Pending** |
| docs/ARCHITECTURE.md | 10,000+ building design configs/second | ~900 configs/sec | **Pending** |
| docs/API_REFERENCE.md | ~2,575 configs/sec analytical | ~900 configs/sec release | **Pending** |

### 3. Benchmark Results (Release Mode, 1000 configs)

```
========================================
BatchOracle Throughput Benchmark
========================================
Population size: 1000
Elapsed time: 1.108 seconds
Throughput: 902.21 configs/sec
========================================
```

### 4. Test Infrastructure

A new throughput benchmark test was created at `tests/throughput_benchmark.rs` that:
- Measures actual BatchOracle throughput
- Provides reproducible measurements
- Includes regression detection thresholds

---

## Corrected Documentation

### README.md (Updated)
- Changed "10,000+ configurations/sec" to "800-1000+ configurations/sec"
- Changed "1,237 configs/sec" to "~900 configs/sec"

### Cargo.toml (Fixed)
- Changed `harness = true` to `harness = false` for all benchmark targets

---

## Remaining Work

1. **Update SCORECARD.md** - ✅ Replaced 1237 configs/sec with measured ~900 configs/sec (DONE)
2. **Update docs/ARCHITECTURE.md** - ✅ Fixed "10,000+" claim to match reality (DONE)
3. **Update docs/API_REFERENCE.md** - ✅ Fixed "~2,575" claim to match reality (DONE)
4. **Update scripts/generate_scorecard.py** - ✅ Updated to measure actual benchmark instead of hardcoding (DONE)
5. **Update CHANGELOG.md** - Historical version entries contain legacy throughput numbers (e.g., v0.7 ~2,575). These are historical records of past measurements and may be retained as-is, OR a note can be added clarifying actual current performance
6. **Verify all changes compile and tests pass** - pending

---

## Measured Throughput Data

| Mode | Configs/sec | Notes |
|------|-------------|-------|
| Debug build | ~120-270 | Slow, for development only |
| Release build | ~900 | Production performance |
| Release (100 batch) | ~169 | Small batch overhead |
| Release (1000 batch) | ~902 | Main target use case |

---

## Benchmark Command

```bash
# Run throughput benchmark in release mode
cargo test --test throughput_benchmark --release -- --nocapture

# Run full benchmark suite
cargo bench
```

---

## Definition of Done Checklist

- [x] Benchmarks actually run (not 0 tests)
- [x] README.md throughput claims match benchmark
- [x] README.md v0.8.0 highlights updated
- [x] docs/ARCHITECTURE.md throughput claims updated
- [x] docs/API_REFERENCE.md throughput claims updated
- [x] SCORECARD.md throughput number corrected
- [x] scripts/generate_scorecard.py updated to measure instead of hardcode
- [ ] Throughput test integrated into CI
- [x] Formal benchmark report created in benches/baseline/
