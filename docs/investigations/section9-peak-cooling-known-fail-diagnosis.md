# ASHRAE 140 §9a `peak_cooling` known-fail diagnosis

**Issue:** #4271 (narrow diagnosis slice; Refs — does NOT close #4271).
**Status:** diagnosis complete — test module + this note only; no engine code, no band changes, no results re-recorded.
**Audience:** the §9a test-set owner (#4271) and the §9 case-layer owner (#4296).
**Date:** 2026-11-01
**Verdict:** **no candidate reading explains the 9 known-fail gaps** — the Task-22 comparative band sits between the power and load readings (or above both); only CE170's load reading lands inside, and the 329 W fan-heat basis only narrows the gaps.
**Sources:** committed artefacts on `develop` at `ed2916b6` (CE case table, CE100 performance map, recorded engine-run results, comparative bounds CSV) and the live engine via the public `fluxion::sim::hvac` API — every number below is computed live and pinned by test.

---

## 1. Question

The §9a engine-run harness (`section9_engine_run.rs`, #4296) scores every
CE case's `peak_cooling` band row against the engine's full-load cooling
power (compressor power + the map's 230 W actual indoor fan heat,
adjusted-net basis). Nine of the fourteen CE cases land outside their
Task-22 comparative `peak_cooling` evidence band and are recorded as
known-fail rows (`docs/KNOWN_ISSUES.md` §LIMIT-42): CE100, CE110, CE120,
CE130, CE140, CE150, CE160, CE170, CE200. The ensemble's own metric
semantics — peak cooling *load* (kW thermal) vs peak cooling *power*
(kW electrical) — are not verifiable from this tree: the bounds CSV
records `source_page` as `pending` for every row.

This slice asks, with numbers: does any other defensible reading of the
metric explain the nine gaps? If the answer were yes, the harness's
scoring basis — not the engine — would be the suspect. The answer is no
(§4).

## 2. Method

The diagnosis module
(`tests/validation/hvac_equipment_rated_performance/section9_peak_cooling_diagnosis.rs`)
re-derives the 9 known-fail cases from the committed results JSON
(nothing hard-coded), dispatches each case through the same steady-state
path as the harness (`dispatch_ce_steady_state` via
`CePerformanceMap::evaluate`) at every entering wet-bulb (EWB) column the
map tabulates (15.0 / 17.2 / 19.4 / 21.7 °C), and computes six candidate
readings per (case, EWB column), each scored with the harness's
`gap_pct_of_mid` definition (0 inside the band, else distance outside as
a percentage of the band midpoint):

| candidate | definition |
|---|---|
| `power_adjnet_230W` | recorded reading: full-load compressor power (adjusted-net table) + 230 W actual fan heat |
| `load_required` | required cooling load (case sensible + latent gains) — the *load* reading |
| `power_fan_default_329W` | full-load compressor power + 329 W manufacturer-default fan heat |
| `power_gross_table` | full-load compressor power from the gross table (Table 26c) + 230 W fan |
| `capacity_gross` | full-load total capacity on the gross table, read as the metric |
| `power_partload` | delivered compressor power at the dispatch operating point + 230 W fan |

Verdicts, paired per EWB column: `explains` (inside the band at every
column), `partially_explains` (gap strictly smaller than the recorded
reading's gap at every column), `explains_nothing` (otherwise). The full
216-row candidate table (9 cases × 4 EWB columns × 6 candidates) and the
verdict matrix are pinned in the test module; the ranges in §3 are the
min–max across the 4 EWB columns from that live table. Reproduce with:

```bash
cargo test -p fluxion section9_peak_cooling -- --nocapture
```

No `src/` code was changed, no band was widened or fitted, and the
recorded results JSON was not re-recorded. A Python replica of the map
interpolation (`replica.py`) was used during the work as a cross-check
and deleted before commit; the pinned values are asserted by the Rust
test, not by the replica.

## 3. Results — gap ranges (% of band midpoint) across the 4 EWB columns

Every value below is a live-computed range; `inside` counts columns
inside the band. The recorded reading is below the band in all 36
(case, EWB) cells.

| case | band [kW] | power_adjnet_230W | load_required | power_fan_default_329W | power_gross_table | capacity_gross | power_partload |
|---|---|---|---|---|---|---|---|
| CE100 | 2.70–3.20 | 8.81–12.20 | 74.58 | 5.46–8.85 | 8.81–12.20 | 103.39–152.88 | 18.02–25.97 |
| CE110 | 2.65–3.15 | 22.76–27.59 | 77.59 | 19.34–24.17 | 22.76–27.59 | 147.24–211.72 | 38.96–43.59 |
| CE120 | 2.80–3.30 | 26.56–31.15 | 68.85 | 23.31–27.90 | 26.56–31.15 | 150.49–196.39 | 43.64–46.36 |
| CE130 | 2.60–3.10 | 5.61–9.12 | 81.75 | 2.14–5.65 | 5.61–9.12 | 110.53–161.75 | 78.91–79.57 |
| CE140 | 2.75–3.25 | 25.33–30.00 | 82.67 | 22.03–26.70 | 25.33–30.00 | 139.00–201.33 | 81.40–81.75 |
| CE150 | 3.10–3.70 | 32.65–36.76 | 82.35 | 29.74–33.85 | 32.65–36.76 | 109.41–164.41 | 40.37–44.70 |
| CE160 | 3.05–3.65 | 31.64–35.82 | 85.07 | 28.69–32.87 | 31.64–35.82 | 126.57–168.36 | 41.16–43.88 |
| CE170 | 3.15–3.75 | 33.62–37.68 | 0.00 (4/4 inside) | 30.75–34.81 | 33.62–37.68 | 106.38–160.58 | 60.70–63.49 |
| CE200 | 3.20–3.80 | 30.29–34.29 | 118.20 | 27.46–31.46 | 30.29–34.29 | 107.71–142.86 | 31.78–34.29 |

Verdict matrix (pinned in the test module):

| case | power_adjnet_230W | load_required | power_fan_default_329W | power_gross_table | capacity_gross | power_partload |
|---|---|---|---|---|---|---|
| CE100–CE160, CE200 | explains_nothing | explains_nothing | partially_explains | explains_nothing | explains_nothing | explains_nothing |
| CE170 | explains_nothing | **explains** | partially_explains | explains_nothing | explains_nothing | explains_nothing |

Worked examples (full rows, from the pinned table):

- CE100 @ EWB 15.0 °C, band [2.70, 3.20]: power 2.34 kW (gap 12.20%),
  load 5.40 kW (74.58%), fan-default 2.439 kW (8.85%), gross-table
  2.34 kW (12.20%), capacity 6.25 kW (103.39%), part-load 2.1684 kW
  (18.02%). Load/power ratio 2.31, map COP 2.57.
- CE170 @ EWB 15.0 °C, band [3.15, 3.75]: power 1.85 kW (37.68%),
  load 3.20 kW (0.00%, inside), fan-default 1.949 kW (34.81%),
  gross-table 1.85 kW (37.68%), capacity 7.42 kW (106.38%),
  part-load 1.056 kW (60.70%).

## 4. Conclusion

1. **Metric semantics alone do not explain the gaps.** For CE100, CE110,
   CE120, CE150, CE160 and CE200 the band sits *between* the power
   reading (below the band) and the load reading (far above it) — the
   load/power ratio is 2.2–2.9 (tracking the map COP), so swapping the
   reading overshoots to the other side. For CE130 and CE140 *both*
   readings are below the band (the load is only 0.27 kW — those cases
   have minimal gains — against bands of 2.60–3.10 and 2.75–3.25 kW),
   so no load/power relabelling can reach the band at all.
2. **CE170 is the single exception, and it proves the rule is not a
   relabelling.** Its load reading (3.20 kW) sits inside [3.15, 3.75]
   at all four EWB columns while its power reading is 34–38% below the
   band. One case landing inside under the load reading, with the other
   eight missing by 69–118%, is not a consistent metric semantics.
3. **Fan-heat basis helps everywhere but closes nothing.** The 329 W
   manufacturer-default basis narrows every recorded gap (e.g. CE100
   12.20% → 8.85% at EWB 15.0 °C) and is the only variant that
   `partially_explains` all nine cases — but it never reaches any band.
   The gross-table basis is numerically identical to the recorded
   reading at every cell (same compressor power, same fan heat), and
   the part-load reading is worse than the recorded reading everywhere
   (equal only at CE200's rating point, where PLR = 1).
4. **No engine change is indicated by this diagnosis, and none was
   made.** The gaps are a property of the comparison (ensemble evidence
   band of unverifiable metric semantics, `source_page: pending`,
   against a steady-state equipment dispatch), not a demonstrated
   engine defect. In particular the CE100 band bounds (2.70–3.20 kW)
   coincide numerically with the AE300 band bounds in the AE bounds
   file — noted here, not interpreted: the rows differ in case/table
   fields and neither file documents the metric semantics.

## 5. What remains

- The §9.2 quasi-analytical criteria stay `null` for every §9 case,
  blocked on #4285 (licensed 140-2023 §9 normative materials: per-case
  entering wet-bulb, quasi-analytical slots, normative CE weather
  files). This diagnosis does not unblock that gate and claims nothing
  about it.
- Whether the Task-22 ensemble band's metric is load, power, or a
  third quantity (e.g. a whole-system or annualised figure) cannot be
  settled from this tree. Resolving it needs the ensemble publication's
  own definition (`source_page` is `pending` in the bounds CSV) — a
  documentation/provenance task, not an engine task.
- If a future harness re-scores `peak_cooling` on a different basis,
  this module's pinned table is the regression guard: it fails if the
  engine, map, case table, or recorded results move under it.
