# ASHRAE 140 Blind Validation Results — Issue #1148

**Date:** 2026-06-19
**Branch:** `fix/issue-1148-blind-validation` (based on `main` @ `8f90bdc`)
**Suite:** `tests/ashrae_140_blind_validation.rs` (`cargo test --test ashrae_140_blind_validation -- --nocapture`)
**Methodology:** Blind execution — `ThermalModel::<VectorField>::from_spec(&spec)` with no case-id hint, no correction factors, no empirical multipliers. Case definitions and benchmark ranges from `validation::benchmark::get_all_benchmark_data()`.

---

## 1. Headline Result

| Metric | Value |
|--------|-------|
| **Total metric checks** | 58 |
| **Passed** | 10 |
| **Failed** | 48 |
| **Pass rate** | **17.24%** |
| **Mean Absolute Error (MAE)** | 50.39% |
| **Cases fully passing** | 0 / 18 |
| **Determinism** | ✅ Verified (two consecutive runs produced identical numbers) |
| **Phase D criterion (≥80%)** | ❌ NOT MET |

### Recommendation

**Do NOT unfreeze v1.3 (epic #672) and do NOT close #668.**

The pass rate (17.24%) is **far below** the Phase D criterion of 80%+. Epic #672 remains blocked on physics fixes, not on documentation or plumbing. Sections 4–6 below explain the systematic failure pattern and root causes.

This is an improvement over the previously-documented baseline of 13.79% (8/58), but only **+3.45 percentage points**. The Wave 1–3 module isolation work fixed the `inf` regression on 900FF MaxFreeFloat (now a real `35.45 °C`) and produced two new passes on heating metrics, but the dominant failure mode (systematic load underestimation by the steady-state 5R1C solver) is unchanged.

---

## 2. Run Command and Reproducibility

```bash
cargo test --test ashrae_140_blind_validation -- --nocapture 2>&1 | tee /tmp/blind_validation_run1.txt
```

**Run 1:** Pass rate 17.24%, MAE 50.39%, 10/58 pass, finished in 22.48s.
**Run 2:** Identical numbers (deterministic).

The test emits `ok` (not `FAILED`) because the harness is a *measurement* test — it reports the baseline failure state without asserting a pass-rate threshold. The threshold gate lives elsewhere (Phase D acceptance, #668).

---

## 3. Per-Case Results

| Case   | Metrics | Pass | Fail | All Pass? |
|--------|--------:|-----:|-----:|:---------:|
| 195    | 2       | 0    | 2    | NO        |
| 600    | 4       | 0    | 4    | NO        |
| 600FF  | 2       | 0    | 2    | NO        |
| 610    | 4       | 1    | 3    | NO        |
| 620    | 4       | 0    | 4    | NO        |
| 630    | 4       | 0    | 4    | NO        |
| 640    | 4       | 1    | 3    | NO        |
| 650    | 2       | 0    | 2    | NO        |
| 650FF  | 2       | 0    | 2    | NO        |
| 900    | 4       | 1    | 3    | NO        |
| 900FF  | 2       | 1    | 1    | NO        |
| 910    | 4       | 2    | 2    | NO        |
| 920    | 4       | 0    | 4    | NO        |
| 930    | 4       | 1    | 3    | NO        |
| 940    | 4       | 1    | 3    | NO        |
| 950    | 2       | 1    | 1    | NO        |
| 950FF  | 2       | 0    | 2    | NO        |
| 960    | 4       | 1    | 3    | NO        |

### 3.1 Detailed Metric Table

| Case   | Metric          |    Simulated |    Ref Min |    Ref Max |   % Error | Status |
|--------|-----------------|-------------:|-----------:|-----------:|----------:|:------:|
| 600    | AnnualHeating   |       4.3068 |     5.5000 |     7.5000 |     33.74 | FAIL   |
| 600    | AnnualCooling   |       0.5488 |     8.0000 |    10.5000 |     94.07 | FAIL   |
| 600    | PeakHeating     |       1.9112 |     2.8000 |     3.8000 |     42.08 | FAIL   |
| 600    | PeakCooling     |       0.6671 |     4.8000 |     6.2000 |     87.87 | FAIL   |
| 610    | AnnualHeating   |       4.3727 |     4.3600 |     5.7900 |     13.84 | **PASS** |
| 610    | AnnualCooling   |       0.5434 |     3.9200 |     6.1400 |     89.20 | FAIL   |
| 610    | PeakHeating     |       1.9183 |     4.3000 |     5.7000 |     61.63 | FAIL   |
| 610    | PeakCooling     |       0.7185 |     2.2000 |     2.9000 |     71.82 | FAIL   |
| 620    | AnnualHeating   |       4.0525 |     4.5000 |     6.5000 |     26.32 | FAIL   |
| 620    | AnnualCooling   |       0.8170 |     3.2000 |     5.0000 |     80.07 | FAIL   |
| 620    | PeakHeating     |       1.7152 |     2.8000 |     3.8000 |     48.02 | FAIL   |
| 620    | PeakCooling     |       0.9200 |     2.5000 |     3.5000 |     69.33 | FAIL   |
| 630    | AnnualHeating   |       4.1312 |     5.0500 |     6.4700 |     28.28 | FAIL   |
| 630    | AnnualCooling   |       0.6513 |     2.1300 |     3.7000 |     77.66 | FAIL   |
| 630    | PeakHeating     |       1.7171 |     4.7000 |     6.1000 |     68.20 | FAIL   |
| 630    | PeakCooling     |       0.8179 |     1.8000 |     2.4000 |     61.05 | FAIL   |
| 640    | AnnualHeating   |       3.1984 |     2.7500 |     3.8000 |      2.34 | **PASS** |
| 640    | AnnualCooling   |       0.5456 |     5.9500 |     8.1000 |     92.23 | FAIL   |
| 640    | PeakHeating     |       1.9837 |     4.3000 |     5.7000 |     60.33 | FAIL   |
| 640    | PeakCooling     |       0.6671 |     2.8000 |     3.7000 |     79.47 | FAIL   |
| 650    | AnnualCooling   |       0.2618 |     4.8200 |     7.0600 |     95.59 | FAIL   |
| 650    | PeakCooling     |       0.6394 |     1.9000 |     2.5000 |     70.94 | FAIL   |
| 600FF  | MinFreeFloat    |      -5.5370 |   -18.8000 |   -15.6000 |     67.81 | FAIL   |
| 600FF  | MaxFreeFloat    |      54.5931 |    64.9000 |    75.1000 |     22.01 | FAIL   |
| 650FF  | MinFreeFloat    |     -11.1449 |   -23.0000 |   -21.0000 |     49.34 | FAIL   |
| 650FF  | MaxFreeFloat    |      54.0265 |    63.2000 |    73.5000 |     20.96 | FAIL   |
| 900    | AnnualHeating   |       1.5400 |     1.1700 |     2.0400 |      4.05 | **PASS** |
| 900    | AnnualCooling   |       1.3667 |     2.1300 |     3.6700 |     52.87 | FAIL   |
| 900    | PeakHeating     |       0.9684 |     1.8000 |     2.4000 |     53.89 | FAIL   |
| 900    | PeakCooling     |       0.8672 |     1.6000 |     2.1000 |     53.13 | FAIL   |
| 910    | AnnualHeating   |       1.6183 |     1.5100 |     2.2800 |     14.60 | **PASS** |
| 910    | AnnualCooling   |       1.0064 |     0.8200 |     1.8800 |     25.45 | **PASS** |
| 910    | PeakHeating     |       0.9778 |     1.9000 |     2.5000 |     55.55 | FAIL   |
| 910    | PeakCooling     |       0.6906 |     1.2000 |     1.6000 |     50.67 | FAIL   |
| 920    | AnnualHeating   |       1.8891 |     3.2600 |     4.3000 |     50.02 | FAIL   |
| 920    | AnnualCooling   |       1.3038 |     1.8400 |     3.3100 |     49.37 | FAIL   |
| 920    | PeakHeating     |       1.0767 |     2.1000 |     2.8000 |     56.05 | FAIL   |
| 920    | PeakCooling     |       0.8838 |     1.4000 |     1.9000 |     46.44 | FAIL   |
| 930    | AnnualHeating   |       1.9618 |     4.1400 |     5.3400 |     58.61 | FAIL   |
| 930    | AnnualCooling   |       1.0549 |     1.0400 |     2.2400 |     35.68 | **PASS** |
| 930    | PeakHeating     |       1.0803 |     2.3000 |     3.0000 |     59.24 | FAIL   |
| 930    | PeakCooling     |       0.7645 |     1.1000 |     1.5000 |     41.19 | FAIL   |
| 940    | AnnualHeating   |       0.8366 |     0.7900 |     1.4100 |     23.95 | **PASS** |
| 940    | AnnualCooling   |       1.2921 |     2.0800 |     3.5500 |     54.10 | FAIL   |
| 940    | PeakHeating     |       1.1861 |     1.9000 |     2.5000 |     46.08 | FAIL   |
| 940    | PeakCooling     |       0.8672 |     1.7000 |     2.3000 |     56.64 | FAIL   |
| 950    | AnnualCooling   |       1.1468 |     0.3900 |     0.9200 |     75.08 | FAIL   |
| 950    | PeakCooling     |       0.8785 |     0.7000 |     0.9000 |      9.81 | **PASS** |
| 900FF  | MinFreeFloat    |      -2.4063 |    -6.4000 |    -1.6000 |     39.84 | **PASS** |
| 900FF  | MaxFreeFloat    |      35.4542 |    41.8000 |    46.4000 |     19.60 | FAIL   |
| 950FF  | MinFreeFloat    |      -9.5701 |   -20.2000 |   -17.8000 |     49.63 | FAIL   |
| 950FF  | MaxFreeFloat    |      35.4732 |    35.5000 |    38.5000 |      4.13 | FAIL   |
| 960    | AnnualHeating   |       2.4790 |     1.6500 |     2.4500 |     20.93 | FAIL   |
| 960    | AnnualCooling   |       0.4344 |     1.5500 |     2.7800 |     79.94 | FAIL   |
| 960    | PeakHeating     |       1.2193 |     2.0000 |     8.0000 |     75.61 | FAIL   |
| 960    | PeakCooling     |       0.3912 |     0.0000 |     4.0000 |     80.44 | **PASS** |

> Note: The `% Error` column uses the absolute deviation from `(ref_min + ref_max)/2`, expressed as a percentage of that midpoint. `within_tolerance` is true only when `ref_min ≤ simulated ≤ ref_max`.

---

## 4. Aggregate Statistics

### 4.1 By Metric Type

| Metric          | Total | Pass | Fail | Pass Rate | MAE    |
|-----------------|------:|-----:|-----:|----------:|-------:|
| AnnualHeating   |    12 |    5 |    7 |    41.7 % | 25.4 % |
| AnnualCooling   |    13 |    2 |   11 |    15.4 % | 69.3 % |
| PeakHeating     |    12 |    0 |   12 |     0.0 % | 55.4 % |
| PeakCooling     |    13 |    2 |   11 |    15.4 % | 59.9 % |
| MinFreeFloat    |     4 |    1 |    3 |    25.0 % | 51.7 % |
| MaxFreeFloat    |     4 |    0 |    4 |     0.0 % | 16.7 % |

### 4.2 By Case Category

| Category   | Total | Pass | Fail | Pass Rate | MAE    |
|------------|------:|-----:|-----:|----------:|-------:|
| low-mass   |    22 |    2 |   20 |     9.1 % | 61.5 % |
| high-mass  |    26 |    7 |   19 |    26.9 % | 47.3 % |
| free-float |     8 |    1 |    7 |    12.5 % | 34.2 % |
| special    |     2 |    0 |    2 |     0.0 % | 33.0 % |

### 4.3 Direction of Error

| Metric          | Sims Below Ref Mid | Sims Above Ref Mid |
|-----------------|-------------------:|-------------------:|
| AnnualHeating   |                 11 |                  1 |
| AnnualCooling   |                 12 |                  1 |
| PeakHeating     |                 12 |                  0 |
| PeakCooling     |                 12 |                  1 |
| MinFreeFloat    |                  0 |                  4 |
| MaxFreeFloat    |                  4 |                  0 |

**Every failing load metric is an *underestimate***. Cooling metrics average `sim / ref_mid ≈ 0.42` — the model predicts roughly **42 % of the reference cooling load**. Heating metrics average `sim / ref_mid ≈ 0.61`. Free-float extremes are damped toward the mean (night minima are too warm, day maxima are too cool).

### 4.4 Regression / Improvement vs Prior Baselines

| Baseline                         | Source             | Pass Rate     | Note |
|----------------------------------|--------------------|---------------|------|
| Issue #1148 body (pre-Wave 1–3)  | `8/58 = 13.79 %`   | 13.79 %       | `900FF MaxFreeFloat` reported `inf` |
| SCORECARD.md (regenerated)       | `4/64 = 6.2 %`     | 6.2 %         | Different metric set, includes more checks |
| **This run (post-Wave 3)**       | This report        | **17.24 %**   | `900FF MaxFreeFloat` now real (`35.45 °C`) |

- The `inf` value for `900FF MaxFreeFloat` mentioned in the issue body is **fixed** — the test now produces a real number (`35.45 °C`), reflecting the CTF stability work in #1154.
- Net change vs the issue's quoted 13.79 % baseline: **+3.45 pp** (two additional heating passes on cases 900 and 940, plus one free-float pass on 900FF MinFreeFloat).
- Heating pass rate rose from ~17 % to 41.7 %. Cooling pass rate is essentially unchanged (~15 %).

---

## 5. Failure Patterns

The 48 failures cluster into three coherent physical signatures:

### Pattern A — Systematic Underestimation of HVAC Loads (44/48 failures)

- All `AnnualCooling`, `PeakCooling`, `AnnualHeating`, `PeakHeating` failures are underestimates.
- Cooling is worse than heating: `sim/ref_mid ≈ 0.42` for cooling vs `≈ 0.61` for heating.
- The worst 10 errors are all `AnnualCooling` or `PeakCooling`, ranging 78–96 % error.
- Low-mass cases (600-series) are worse than high-mass (900-series): 9.1 % vs 26.9 % pass rate.

**Interpretation.** The 5R1C steady-state solver, as wired in `simulate_case_blind`, transfers too little heat across the envelope in both directions. The same model under-transfers solar gain in summer (under-cooling) and envelope loss in winter (under-heating). Cooling suffers an additional ~20 pp gap vs heating, which points to a cooling-specific term (window solar gain distribution, radiant split, or HVAC capacity logic) on top of the steady-state conduction floor.

### Pattern B — Over-Damped Free-Floating Extremes (7/8 FF failures)

- `MinFreeFloat`: 4/4 simulated values are **warmer** than the reference midpoint (e.g. `600FF` sim `-5.5 °C` vs ref `-17.2 °C`).
- `MaxFreeFloat`: 4/4 simulated values are **cooler** than the reference midpoint (e.g. `600FF` sim `54.6 °C` vs ref `70.0 °C`).
- The diurnal swing is too small — the model's thermal mass damps extremes too aggressively.

**Interpretation.** Two competing causes are consistent with the data:
1. The same under-transfer of envelope/solar heat that causes Pattern A also flattens the diurnal swing in free-floating mode.
2. The effective thermal capacitance in the 5R1C network is too high, absorbing heat that should appear as a peak.

### Pattern C — Peak Loads Never Pass (0/12 PeakHeating, 11/13 PeakCooling Fail)

- Peak loads are systematically underestimated by 40–90 %.
- Even annual-energy passes on the same case usually have failing peaks (e.g. case 900 AnnualHeating passes at 4.05 % but PeakHeating fails at 53.89 %).

**Interpretation.** Peak load is the worst-case timestep, so any steady-state averaging bias is amplified. This pattern is structurally the same as Pattern A — it confirms the model is not just biased on totals, it under-resolves every transient peak.

---

## 6. Root Cause Assessment

| # | Cause | Category | Evidence | Affected Failures |
|---|-------|----------|----------|-------------------|
| 1 | Steady-state-only 5R1C solver cannot resolve transient heat flow | **Architectural limitation** | ARCHITECTURE.md §Module Status explicitly attributes the ~90 % cooling underestimation to "the steady-state-only 5R1C solver"; this run reproduces that signature (cooling ratio ≈ 0.42) | ~40 of 48 |
| 2 | Cooling-specific physics gap (window solar gain, radiant exchange, HVAC capacity) | **Fixable (physics bug)** | Cooling MAE 69 % vs heating MAE 25 % — the extra 44 pp cannot be explained by the steady-state floor alone; specific to cooling | ~12 of 48 (the *additional* cooling gap above the heating floor) |
| 3 | Over-damped free-float extremes from combined Pattern A + mass tuning | **Mix of fixable and architectural** | 7/8 FF failures, all on the wrong side of midpoint (mins too warm, maxes too cool) | 7 of 48 |
| 4 | Reference data quality | **Not a cause** | Benchmark ranges come from the standard ASHRAE 140 EnergyPlus/ESP-r/TRNSYS envelope (`validation/benchmark.rs`); the `inf` regression noted in the issue is fixed, and the `ref_min`/`ref_max` values are the published ANSI/ASHRAE 140-2007 envelope, not "calibrated for 5R1C" | 0 |
| 5 | Parameter tuning opportunity | **Explicitly rejected** | AGENTS.md forbids parameter tuning to lift pass rate; per the "no tuning, fix the math" rule, no empirical multipliers were introduced by this run | 0 |

### Categorization Summary

- **Fixable (physics bug):** the *additional* cooling-specific underestimation beyond the steady-state floor. This is the next highest-leverage target — closing it would lift AnnualCooling/PeakingCooling toward the heating pass rate (~40 %).
- **Architectural limitation:** the 5R1C steady-state conduction floor. This is the dominant gap and is consistent with prior ARCHITECTURE.md findings. Options: (a) promote the 6R2C/8R3C dynamic solver to default for HVAC cases, (b) complete CTF solver adoption for ASHRAE 140, (c) accept that 5R1C cannot meet Phase D and re-scope the v1.3 epic around the dynamic solver path.
- **Reference data:** no issues found. The benchmark envelope is the standard published ASHRAE 140 range.

---

## 7. Status Against Phase D Acceptance (Issue #668)

| Tolerance                              | This Run                            | Met? |
|----------------------------------------|-------------------------------------|:----:|
| Annual energy: ±15 % of reference mean | AnnualHeating 41.7 % pass; AnnualCooling 15.4 % pass | ❌ |
| Monthly energy: ±10 % of reference mean| Not measured by this suite          | — (out of scope; suite measures annual/peak/free-float) |
| Peak loads: ±15 % of reference mean    | PeakHeating 0 % pass; PeakCooling 15.4 % pass | ❌ |
| Free-floating temp: ±1.0 °C of ref mean| 1/8 pass; mean abs error ~34 % of midpoint | ❌ |
| Suite coverage ≥ 80 %                  | 17.24 %                             | ❌ |

Phase D acceptance is not met on any measured criterion. Issue #668 cannot be closed by this run.

---

## 8. Recommendation

1. **Do NOT unfreeze v1.3 (epic #672).** Pass rate is 17.24 %, Phase D requires ≥80 %.
2. **Do NOT close #668 (Phase D acceptance).** No tolerance band is met.
3. **Prioritize the next wave of work on the cooling-specific gap** (Pattern A, cooling term): this is the most fixable category and would plausibly double the pass rate (lifting cooling from ~15 % to ~40 % would add ~5 metric passes, bringing total to ~15/58 ≈ 26 %).
4. **Open a follow-up design discussion on the 5R1C architectural limitation** (Pattern A, steady-state floor). The 9R4C multi-node model and CTF solver are already in the codebase (`sim/multi_node_thermal.rs`, `physics/ctf_*`); promoting one of them to be the default for ASHRAE 140 cases is a separate architectural decision and should not be smuggled into a validation-only PR.
5. **Keep the blind validation suite as-is.** It is deterministic, runs in ~23 s, and correctly reports the baseline state. The `ok` test result is correct (it is a *measurement*, not a gate).
6. **The `inf` regression on 900FF MaxFreeFloat noted in the issue body is resolved** by the CTF stability work in #1154; the test now emits a real number (`35.45 °C`).

---

## 9. Next Actions (Outside This PR)

- [ ] File follow-up issue for "cooling-specific underestimation beyond steady-state floor" with the Pattern A evidence above.
- [ ] Re-open the architectural discussion ("5R1C default vs dynamic solver for ASHRAE 140") referenced in ARCHITECTURE.md §Module Status.
- [ ] Re-run this suite after each physics fix; the determinism check makes regressions immediately visible.

---

## Appendix A — Reproducibility Provenance

- **Worktree:** `/home/alex/Projects/worktrees/issue-1148-blind-validation`
- **Branch HEAD:** `8f90bdc test(weather): complete Weather module isolation with psychrometrics fix (#1145)`
- **Test file:** `tests/ashrae_140_blind_validation.rs`
- **Run command:** `cargo test --test ashrae_140_blind_validation -- --nocapture`
- **Raw output saved to:** `/tmp/blind_validation_run1.txt` (run 1) and a grep over run 2 (determinism check).
- **Parsed machine-readable results:** `/tmp/blind_validation_results.json` (58 rows, columns: `case_id, metric, sim, ref_min, ref_max, ref_mid, pct_err, pass`).

## Appendix B — Tolerance Check Semantics

`within_tolerance` is `true` iff `ref_min ≤ simulated ≤ ref_max`. The reference bands are the ASHRAE 140 standard published envelopes (min/max across EnergyPlus/ESP-r/TRNSYS/DOE-2.1E etc.), **not** the ±15 % Phase D tolerance. The Phase D tolerance is stricter on the midpoint (`±15 % of mean`) and is reported separately in §7. The pass rate of 17.24 % is therefore a *lower bound* on the "inside the reference envelope" criterion and an *upper bound* on what Phase D acceptance would actually require.
