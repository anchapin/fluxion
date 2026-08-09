# Fluxion Release Scorecard

**Generated:** 2026-06-19
**Wave:** Wave 1
**Version:** 1.0.0 (next release: 1.2.0)
**Data sources:** `validation_report.md` (ASHRAE 140), `cargo test` (tests), `scripts/generate_scorecard.py` (benchmark), `gh issue list` (issues)

---

## Summary

| Metric | Value | Status |
|--------|-------|--------|
| ASHRAE 140 Pass Rate | 6.2% (4/64 metric checks) | ❌ Below Target |
| Mean Absolute Error | 35.35% | ❌ High |
| Test Pass Rate (unit/lib) | 100.00% (2664/2664) | ✅ Healthy |
| Full Test Suite | 99.96% (2706/2707) | ⚠️ 1 known ASHRAE failure |
| Benchmark Throughput | 735 configs/sec | ❌ Below (machine-dependent) |
| Open Issues (GitHub) | 9 | ⚠️ Review |

---

## Validation Results (ASHRAE 140)

Source: `validation_report.md` (regenerated 2026-06-19). Authoritative per issue #1144
(`validation_results.json` is absent — see [Conflicting Metrics Resolution](#conflicting-metrics-resolution)).

### Aggregate

| Metric | Value |
|--------|-------|
| Total metric checks | 64 |
| Passed | 4 |
| Warnings | 2 |
| Failed | 58 |
| Pass Rate | 6.2% |
| Mean Absolute Error | 35.35% |
| Max Deviation | 346.87% |

### Pass Rate by Case Series

Case-level status — all 18 cases currently FAIL at the case level:

| Series | Cases | Passed | Failed | Pass Rate |
|--------|-------|--------|--------|-----------|
| Baseline (600-650) | 6 | 0 | 6 | 0.0% |
| High-Mass (900-950) | 6 | 0 | 6 | 0.0% |
| Free-Floating | 4 | 0 | 4 | 0.0% |
| Special (195, 960) | 2 | 0 | 2 | 0.0% |
| **Total (cases)** | **18** | **0** | **18** | **0.0%** |

*The 6.2% headline pass rate is measured at the **metric-check level** (4 of 64 individual
checks pass): Case 600 Annual Cooling, Case 600 Peak Cooling, Case 900 Peak Heating, and
Case 940 Annual Cooling. The previous scorecard's per-series "1/6" (High-Mass) and "1/2"
(Special) case passes were stale and are corrected here.*

### Critical Failures (Top 3, by deviation from reference midpoint)

| Case | Metric | Fluxion | Reference | Deviation |
|------|--------|---------|------------|------------|
| 950 | Annual Cooling | 3.80 MWh | 0.39-0.92 | +480% |
| 940 | Annual Heating | 5.29 MWh | 0.79-1.41 | +381% |
| 900 | Annual Heating | 7.17 MWh | 1.17-2.04 | +347% |

*Note: The previous scorecard listed "195 Annual Heating 21.85 MWh (+313%)" as a critical
failure. The current `validation_report.md` shows Case 195 Annual Heating at 5.00 MWh, which
is **within** the 3.50-6.00 reference range and is no longer a top failure.*

---

## Benchmark Status

### Performance Metrics

| Benchmark | Value | Target | Status |
|-----------|-------|--------|--------|
| Throughput (configs/sec) | 735 | ≥800 | ❌ Below |
| CTA Simulation Time | <100ms | <100ms | ✅ Meets |
| Multi-Zone (10 zones) | 800-1,200 | ≥500 | ✅ Exceeds |
| Cross-Validation Latency | <100ms | ≤500ms | ✅ Exceeds |

*Throughput measured at 735 configs/sec on the development worktree
(`fix/issue-1144-scorecard-md`). This is marginally below the 800 target; throughput is
machine/load-dependent (the prior scorecard reported 892 from a different environment).
Re-measure on the CI runner before treating this as a hard release blocker.*

---

## Open Issues (GitHub)

Source: `gh issue list --state open` (2026-06-19). GitHub does not apply
Critical/High/Medium/Low severity labels, so issues are grouped by area.

| Area | Count | Issues |
|------|-------|--------|
| Physics (module isolation) | 4 | #1145, #1146, #1147, #1152 |
| Validation / ASHRAE 140 | 1 | #1148 |
| v1.3 milestone / epics | 2 | #668, #672 |
| Documentation | 2 | #1143, #1144 |
| **Total open** | **9** | |

*The previous scorecard's "Open Issues by Severity" table (Critical 4 / High 8 / Medium 8 /
Low 5 = 25) was derived from `docs/KNOWN_ISSUES.md`, which is stale and does not reflect the
9 issues currently open on GitHub. Reconciling `KNOWN_ISSUES.md` is tracked separately.*

---

## Release Readiness

### Requirements Check

| Requirement | Status | Notes |
|-------------|--------|-------|
| Compilation | ✅ Pass | All crates compile (`cargo test` builds clean) |
| Unit Tests (lib) | ✅ Pass | 2664/2664 passed (100.0%) |
| Integration Tests | ⚠️ 1 expected failure | 2706/2707 passed (99.96%); `test_case_195_temperature_range` fails — ASHRAE Case 195 physics, tracked in #1148, not a regression |
| ASHRAE 140 Pass Rate ≥12.5% | ❌ Fail | Currently 6.2% (4/64) |
| Benchmark Throughput ≥800 | ❌ Fail | 735 configs/sec (machine-dependent; see Benchmark Status) |
| Open Issues | ⚠️ Review | 9 open on GitHub |
| Documentation Complete | ✅ Pass | 100% coverage |

### Overall: ❌ Not Ready

**Primary Blockers:**
1. ASHRAE 140 Pass Rate (6.2%) below the 12.5% threshold — high-mass thermal modeling and
   solar gains (tracked in #1148, #1146, #1147).
2. Benchmark throughput marginally below target on this environment (re-measure on CI).

---

## Conflicting Metrics Resolution

Issue #1144 flagged a conflict between `validation_results.json` and `QUALITY_METRICS.md`
for **Case 900 Annual Heating**.

| Metric | Reported value | Source status |
|--------|----------------|---------------|
| Case 900 Annual Heating | 1.35 MWh | `validation_results.json` — **file does not exist** in the repo (verified via `find`); 1.35 MWh is unverifiable |
| Case 900 Annual Heating | 7.17 MWh | `validation_report.md` (regenerated 2026-06-19) — **authoritative** |
| Case 900 reference range | 1.17-2.04 MWh | Both sources agree (this is the EnergyPlus reference band, not a Fluxion measurement) |

**Resolution:** `validation_results.json` is absent, so its 1.35 MWh figure cannot be used.
The authoritative source is `validation_report.md`, which reports
**Case 900 Annual Heating = 7.17 MWh** — still a FAIL against the 1.17-2.04 reference range
(+347% deviation, computed in Python against the reference midpoint).

**Action items:**
- Standardize on `validation_report.md` as the authoritative ASHRAE 140 source (regenerated
  by the validation harness).
- Recreate `validation_results.json` from the harness output, or update
  `scripts/generate_scorecard.py` to read `validation_report.md` directly. The generator
  currently falls back to the stale `QUALITY_METRICS.md` (which reports 0.0% / -inf% MAE)
  when `validation_results.json` is missing.
- Reconcile `docs/KNOWN_ISSUES.md` severity counts with GitHub's actual 9 open issues.

---

## Data Sources & Verification (2026-06-19)

| Field | Source | How verified |
|-------|--------|--------------|
| ASHRAE 140 results | `validation_report.md` | regenerated by validation harness; aggregate summary + detailed tables |
| Test counts | `cargo test` | 2664 lib passed (0 failed, 2 ignored); 2706/2707 full suite |
| Benchmark | `scripts/generate_scorecard.py` | measured 735 configs/sec (throughput benchmark test) |
| Open issues | GitHub | `gh issue list --state open` → 9 |
| Case 900 deviation | computed | Python, vs reference midpoint (1.17, 2.04) |
| `validation_results.json` absent | filesystem | `find . -name validation_results.json` → no results |

---

## Regeneration Command

To regenerate this scorecard, run:

```bash
# Run this from the project root
python3 scripts/generate_scorecard.py

# Or with verbose output
python3 scripts/generate_scorecard.py --verbose

# To specify output location
python3 scripts/generate_scorecard.py --output SCORECARD.md
```

*Note: `generate_scorecard.py` relies on `validation_results.json` (absent) and
`docs/QUALITY_METRICS.md` (stale, reports 0.0% pass / -inf% MAE). After running it, the
ASHRAE metrics, test count, and issue counts in this file were manually corrected to the
verified sources listed in [Data Sources & Verification](#data-sources--verification-2026-06-19).
Fixing the generator's data sources is tracked as a follow-up.*

---

## Links

- [ASHRAE 140 Validation Report](docs/ASHRAE140_RESULTS_v0.8.0.md)
- [Validation Report (authoritative)](validation_report.md)
- [Known Issues Catalog](docs/KNOWN_ISSUES.md)
- [Quality Metrics](docs/QUALITY_METRICS.md)
- [Release Notes v1.2](docs/RELEASE_NOTES_v1.2.md)

---

*This scorecard is auto-generated as part of QG-01: Create a generated release scorecard,
with manual verification of metrics against authoritative sources for issue #1144.*
