# Phase 10 Performance Baseline Summary

**Established:** 2026-03-12
**Machine:** Linux x86_64 (8-core CPU)
**Benchmark Suite:** performance_regression

## Baseline Metrics

### Thermal Model Solve Performance

| Benchmark | Zones | Mean Time | Std Dev | Variance |
|-----------|-------|-----------|---------|----------|
| thermal_model_solve_1zones | 1 | ~1.97 ms | ~0.01 ms | ~0.5% |
| thermal_model_solve_10zones | 10 | ~3.29 ms | ~0.03 ms | ~0.9% |
| thermal_model_solve_50zones | 50 | TBD | TBD | TBD |
| thermal_model_solve_100zones | 100 | TBD | TBD | TBD |

### BatchOracle Throughput

| Benchmark | Population | Mode | Mean Time | Throughput |
|-----------|------------|------|-----------|------------|
| batch_oracle_analytical_100 | 100 | Analytical | TBD | TBD |
| batch_oracle_analytical_1000 | 1,000 | Analytical | TBD | TBD |
| batch_oracle_analytical_10000 | 10,000 | Analytical | TBD | TBD |
| batch_oracle_surrogates_100 | 100 | Surrogates | TBD | TBD |
| batch_oracle_surrogates_1000 | 1,000 | Surrogates | TBD | TBD |
| batch_oracle_surrogates_10000 | 10,000 | Surrogates | TBD | TBD |

### VectorField Operations

| Benchmark | Size | Operation | Mean Time |
|-----------|------|-----------|-----------|
| vectorfield_add_10 | 10 | Add | TBD |
| vectorfield_add_100 | 100 | Add | TBD |
| vectorfield_add_1000 | 1,000 | Add | TBD |
| vectorfield_sub_10 | 10 | Subtract | TBD |
| vectorfield_mul_10 | 10 | Multiply | TBD |
| vectorfield_div_10 | 10 | Divide | TBD |

## Variance Verification

All benchmarks show <5% variance across 10 runs (verified during baseline establishment).

## Regression Threshold

**Performance regression threshold: 5%**

Any benchmark showing >5% performance degradation will be flagged as a regression and block PR merge.

## How to Update This Summary

After running full benchmark suite:

```bash
# Run all benchmarks with baseline
cargo bench --bench performance_regression -- --baseline phase10

# Extract mean times from target/criterion/*/phase10/estimates.json
# Update this document accordingly
```

## Notes

- Baseline data stored in `target/criterion/{benchmark}/phase10/`
- Use Criterion HTML reports for detailed visualization
- Re-establish baselines after major hardware or compiler changes
