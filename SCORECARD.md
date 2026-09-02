# Fluxion Release Scorecard

> Consolidated view of release-readiness metrics. Generated from committed sources so it is fully reproducible.
>
> **Do not edit by hand** — regenerate with `python scripts/generate_scorecard.py`. CI fails on drift (`scorecard-drift` workflow).

**Last Updated:** 2026-08-16  
**Data source as of:** 2026-08-16 18:24 UTC  
**Sources:** `docs/ASHRAE140_RESULTS.md`, `release_gates.yaml`, `README.md`

---

## Headline

| Metric | Current | Budget (gate) | Status | Source |
|--------|---------|---------------|--------|--------|
| ASHRAE 140 pass rate | **14.3%** (12/84 metrics) | ≥ 60% (`validation.min_pass_rate`) | ❌ Fail | `docs/ASHRAE140_RESULTS.md` |
| Mean Absolute Error (MAE) | **51.03%** | ≤ 50% (`validation.max_mae`) | ❌ Fail | `docs/ASHRAE140_RESULTS.md` |
| BatchOracle throughput | **157 (CI) / 900 (release)** configs/sec | ≥ 150 (`benchmark.throughput.min_configs_per_sec`) | ✅ Pass | `release_gates.yaml` comment + `README.md` |
| Validation-suite throughput | 35.36 cases/sec | (informational) | ℹ️ | `docs/ASHRAE140_RESULTS.md` |
| Max single-case deviation | 470.11% | (ref: `individual.max_deviation` = 100%) | ℹ️ | `docs/ASHRAE140_RESULTS.md` |

## ASHRAE 140 Pass Rate

- **Overall (metric-level):** 14.3% — 12 PASS / 8 WARN / 64 FAIL of 84 results. Below the 60% gate.
- **Case-level:** 0/18 cases fully PASS (0.0%).

### Per-Series Breakdown (case-level)

| Series | Cases | PASS | WARN | FAIL | Pass rate |
|--------|-------|------|------|------|-----------|
| Baseline Cases (600 Series) | 6 | 0 | 0 | 6 | 0.0% |
| High-Mass Cases (900 Series) | 6 | 0 | 0 | 6 | 0.0% |
| Free-Floating Cases | 4 | 0 | 0 | 4 | 0.0% |
| Special Cases | 2 | 0 | 0 | 2 | 0.0% |

*Case-level = a case is PASS only if its aggregate row is ✅. Metric-level headline (20.3%) counts each reported metric individually; see `docs/ASHRAE140_RESULTS.md` Summary.*

## Throughput vs Budget

- **Gate:** ≥ **150** configs/sec (`benchmark.throughput.min_configs_per_sec`); absolute floor 100; latency ≤ 10 ms/config.
- **CI runner (Wave 1+1.5):** ~157 configs/sec — ✅ Pass (narrow margin; source: `release_gates.yaml` comment).
- **Release mode (BatchOracle, rayon):** ~900 configs/sec — ✅ Pass (source: `README.md`).
- **Validation-suite throughput:** 35.36 cases/sec — informational only; this is the test-runner cadence, not the BatchOracle benchmark (source: `docs/ASHRAE140_RESULTS.md`).

## MAE vs Budget

- **Gate:** ≤ **50%** (`validation.max_mae`).
- **Current:** **51.03%** — Over budget by +1.03 pp. Max single-case deviation 470.11%.
- *Driver:* high-mass annual-energy deviation (5R1C/CTF thermal-mass limitation; see Known Structural Failures).

## Known Structural Failures

Cases excluded from the strict ±15% annual-energy gate and from the `extreme_deviation_limit` count (`release_gates.yaml` → `validation.individual.known_failures`):

| Case | Series | Reason |
|------|--------|--------|
| **600** | Baseline (low-mass) | Multiple low-mass baseline tests — simplified envelope model (`AGENTS.md`). |
| **900** | High-mass | Heating deviation ~200% — high-mass thermal-mass model limitation (`release_gates.yaml` comment). |

Per `AGENTS.md`: cases **600** and **900** are documented structural failures. Fix path = underlying physics (no parameter tuning — `RULES.md`).

## CI Gate Status

Required branch-protection checks (`release_gates.yaml` → `ci.required_checks`):

| Required check | Issue |
|----------------|-------|
| ASHRAE 140 Strict Energy Gate (Issue #1333) | #1333 |
| Surrogate ASHRAE 140 MAE Gate (Issue #2924) | #2924 |
| Surrogate Drift Tolerance Gate (Issue #1784) | #1784 |
| Fluxion Determinism Gate (Issue #1351) | #1351 |
| Fluxion Performance Gate (Issue #1618) | #1618 |
| Code Coverage Gate (Issue #1932) | #1932 |
| Docs Hygiene Gate (Issue #2466) | #2466 |
| Physics-Sim-Cycle-Check (GH) | — |
| Workspace Check (GH) | — |
| Absolute Perf Gate (Issue #2693) | #2693 |
| Multi-Zone Perf Gate (Issue #2772) | #2772 |
| Multi-Zone Cold Start Gate (Issue #2919) | #2919 |
| Hybrid Perf Gate (Issue #2922) | #2922 |
| Energy Conservation (GH) | — |
| Rustfmt (GH) | — |
| Clippy (GH) | — |
| Known Issues Stale Check (GH) | — |
| Ashrae Cases Cycle Check (GH) | — |
| Cycle Downward Trend Guard (Issue #2768) | #2768 |
| CUDA Smoke Test (Issue #1603) | #1603 |
| Architecture Drift Detection | — |
| Cargo Deny | — |
| Audit Ignore Freshness (Issue #2912) | #2912 |
| MSRV Check (Issue #2934) | #2934 |
| Crate Size Gate (Issue #2930) | #2930 |
| fluxion-grid Integration Tests (GH) | — |
| h_tr_em Regression Gate (LIMIT-13) | — |
| FFI Feature Check (GH) | — |
| ASHRAE 140 Strict Energy Gate (Issue #1333) | #1333 |
| Surrogate ASHRAE 140 MAE Gate (Issue #2924) | #2924 |
| Surrogate Drift Tolerance Gate (Issue #1784) | #1784 |
| Fluxion Determinism Gate (Issue #1351) | #1351 |
| Fluxion Performance Gate (Issue #1618) | #1618 |
| Code Coverage Gate (Issue #1932) | #1932 |
| Physics-Sim-Cycle-Check (GH) | — |
| Workspace Check (GH) | — |
| Absolute Perf Gate (Issue #2693) | #2693 |
| Multi-Zone Perf Gate (Issue #2772) | #2772 |
| Multi-Zone Cold Start Gate (Issue #2919) | #2919 |
| Hybrid Perf Gate (Issue #2922) | #2922 |
| Energy Conservation (GH) | — |
| Rustfmt (GH) | — |
| Clippy (GH) | — |
| Known Issues Stale Check (GH) | — |
| Ashrae Cases Cycle Check (GH) | — |
| Cycle Downward Trend Guard (Issue #2768) | #2768 |
| CUDA Smoke Test (Issue #1603) | #1603 |
| Cargo Deny | — |
| Audit Ignore Freshness (Issue #2912) | #2912 |
| fluxion-grid Integration Tests (GH) | — |
| h_tr_em Regression Gate (LIMIT-13) | — |
| FFI Feature Check (GH) | — |

- **Live status** is intentionally not baked in here (it is non-deterministic and would break scorecard diff stability). Run:

  ```bash
  gh run list --repo anchapin/fluxion --branch develop --limit 10
  ```

- **Validation gate policy** (`release_gates.yaml`): major/minor releases require validation + benchmark + drift gates; patches relax validation to 40% pass (see `release_requirements.patch`).
- **Drift guard** (`drift.*`): max ±2.0 pp pass-rate change, ±5.0 pp MAE change, ≤1 pass→fail flip vs `validation_baseline.json`.

## Regenerate

```bash
# Regenerate the scorecard from committed sources
python scripts/generate_scorecard.py

# CI uses this to fail on drift (exit 1 if SCORECARD.md is stale)
python scripts/generate_scorecard.py --check

# Verbose: print every parsed value
python scripts/generate_scorecard.py --verbose
```

The scorecard is regenerated whenever `docs/ASHRAE140_RESULTS.md`, `release_gates.yaml`, or `README.md` changes. The `scorecard-drift` workflow enforces this on every PR.

---

*Auto-generated by `scripts/generate_scorecard.py` (issue #2496). Edit the generator, not this file.*
