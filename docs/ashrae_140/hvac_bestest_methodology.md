# ASHRAE 140 §9a HVAC BESTEST Methodology Evaluation (OpenBSE)

**Issue:** #4239 — research spike: evaluate OpenBSE's ASHRAE 140 §9a HVAC BESTEST approach against fluxion's strict gate.
**Status:** evaluation complete — research/spike only; no engine code, no new validation cases, no test changes.
**Audience:** the §9a test-set owner (#4271) and the §9 case-layer owner (#4296).
**Date:** 2026-10-23
**Verdict:** **adopt-partially** — adopt the *quasi-analytical acceptance model* for §9a; keep inter-program/comparative bands as secondary context, not as the criterion.
**Sources:** issue #4239 text for the OpenBSE claims (not independently verified — see §1.4) and this tree at `develop` (commit `9e61b840`).

---

## 1. What OpenBSE did (per issue #4239)

This section records what the issue text reports about OpenBSE's implementation. It is a
*report of the issue's claims*, not an independent verification.

### 1.1 The OpenBSE deliverable as described in #4239

The issue text describes an OpenBSE change that adds a §9a HVAC BESTEST track comprising:

1. **14 §9a cases, CE100–CE200.** Space-cooling equipment cases, split between the dry-coil
   series (CE100–CE140) and the wet-coil / latent series (CE150–CE200), each with fixed
   internal gains, a cooling setpoint and an outdoor dry-bulb condition, driven against the
   CE100 performance map.
2. **112/112 monthly totals within ±5 %** versus *quasi-analytical* solutions — i.e. 14 cases
   × 8 reporting months = 112 monthly totals, every one of them inside ±5 % of the
   standard's own analytic reference. The February min/max columns are the specific output
   columns at risk in this family; a sibling review of OpenBSE's submission pipeline
   (`docs/ashrae_140/submission_pipeline_lessons.md:28-31`) records that OpenBSE's October CE
   submission failed a portal missing-data check and carried defective February min/max
   columns and a defective program-identity header, which is why this evaluation cares about
   the *shape* of the claim (per-case monthly totals, February columns) and not only the
   headline number.
3. **A reproducible case generator.** The case table is produced by a script, so the cases are
   regenerable rather than hand-authored.
4. **Normative weather CSVs.** The §9a case set is bound to the standard's normative weather
   inputs, shipped as CSV, rather than to synthetic or self-authored weather.

### 1.2 Why this matters for fluxion

The issue's framing is that OpenBSE reached 112/112 against *quasi-analytical* references —
a stronger claim than "fluxion has no §9 coverage at all" (fluxion's own status:
`docs/compliance/section_coverage.md:59` — §9a "Not implemented as a test set";
`data/reference/ashrae140/section9_ce_cases.json:12` — `"cases_run": 0`). This document asks
whether fluxion should adopt the same *acceptance methodology* (quasi-analytical references
where they exist), not whether it should copy OpenBSE's code.

### 1.3 Relevance to fluxion's own §9 artifacts

Fluxion already holds a *case-definition* layer for §9a:
`scripts/generate_section9_ce_cases.py` emits
`data/reference/ashrae140/section9_ce_cases.json` with the 14 CE100–CE200 case definitions
(`scripts/generate_section9_ce_cases.py:1-3`, `:19`, `:98`). The gap the issue exposes is not
the case table; it is the *acceptance criterion* and the *engine run*.

### 1.4 What could NOT be inspected

- The OpenBSE commit `e7b016a3c8d00b96d2a7915861fc3f587bc96be2` was **not fetched** during this
  evaluation — network access to `github.com` was not available to this worker. Every OpenBSE
  claim in §1 is transcribed from the issue text and is labelled as such.
- The ASHRAE 140-2023 normative materials (§9 tables, the per-case quasi-analytical solution
  values, and the normative weather files) are **licensed** and are not present in this tree;
  fluxion tracks that licensing gate as issue #4285
  (`scripts/generate_section9_ce_cases.py:144-148`;
  `fluxion-core/src/weather/tm2.rs:12-20`).
- Consequently this document cannot verify OpenBSE's ±5 % numbers, cannot diff OpenBSE's case
  generator against `scripts/generate_section9_ce_cases.py`, and cannot confirm which specific
  weather files OpenBSE shipped or their format.
- One further structural caveat: **Standard 140 defines no acceptance criteria at all.** Per
  `data/reference/ashrae140/analytical_acceptance_criteria.json`
  (`_standard_structure.finding`), §4.4 "Comparing Output to Other Results" is *informative*,
  and every Annex B statistics table carries "These statistics do not have any substantial
  importance and are not to be interpreted as acceptance criteria". OpenBSE's ±5 % is therefore
  OpenBSE's own engineering tolerance against an analytic reference — a good, defensible
  practice, but not a threshold the standard imposes.

---

## 2. Fluxion's current state

### 2.1 The strict gate today

Fluxion's PR-blocking strict gate is the CI job **"ASHRAE 140 Strict Energy Gate (Issue #1333)"**
(`.github/workflows/physics-pr.yml:300-301`, with an unconditional listener job at
`:464-482`). What it covers:

| Property | Value | Citation |
|---|---|---|
| Cases observed (annual energy) | 600, 800, 810, 900, 920, 950, 960, 970 | `scripts/check_strict_energy_gate_regression.py:60-68` |
| Cases added by #4170 | 610, 620, 630, 640, 650, 910, 930, 940, 195 | `scripts/check_strict_energy_gate_regression.py:69-78` |
| Free-float cases observed | 600FF, 650FF, 900FF, 950FF (Tmin/Tmax) | `scripts/check_strict_energy_gate_regression.py:104` |
| Verdict model | PASS / KNOWN-FAIL / REGRESSION against a recorded baseline; baselines must never be raised to hide a regression | `scripts/check_strict_energy_gate_regression.py:26-33`, `:37-40` |
| Test surface | `#[ignore]`-d strict tests that the workflow runs with `--include-ignored` | `tests/all_tests/zone_balance_eplus_isolation.rs:976-992` (Case 600), `:1119-1133` (800), `:1266-1274` (970); `tests/QUARANTINE.md:96-176` lists the quarantined cohort |
| Weather binding | `FLUXION_ASHRAE140_WEATHER` env, default `assets/weather/ashrae140/DRYCOLD_TMY725650TY.epw` | `src/validation/ashrae_140_cases.rs:76-81` |

**All of those cases are the envelope-and-fabric Class I family** — the §5.2 (pre-2023
numbering) / §7 (140-2023 numbering) test set. The 140-2023 source map in
`data/reference/ashrae140/analytical_acceptance_criteria.json`
(`_standard_structure.section_numbering_2020_to_2023`) maps "7 Envelope and Fabric (Class I)"
to old inputs "5.2 up to 5.2.4"; fluxion's committed coverage table calls the same family §7
(`docs/compliance/section_coverage.md:56`).

**No §9/CE case is in the strict gate.** The gate's case list contains no `CE*` identifier,
and the §9 case artifact itself records `"cases_run": 0`
(`data/reference/ashrae140/section9_ce_cases.json:11-14`). The §9 status is tracked as
`LIMIT-42` in `docs/KNOWN_ISSUES.md:145`.

### 2.2 Acceptance bands in use today

| Family | Band in use | Nature | Citation |
|---|---|---|---|
| §7 / §5.2 envelope (600…970) | **Inter-program min/max** (EnergyPlus, ESP-r, TRNSYS, DOE2) from ASHRAE 140-2023 Annex B8, widened to a ±15 %-of-midpoint gate | comparative, diagnostic | `src/validation/benchmark.rs:115-118` ("Reference data is from ASHRAE Standard 140-2023 and EnergyPlus BESTEST reports. Values are the min/max across reference programs"); per-case CSV mirrors at `tests/reference_data/zone_balance/case_*_energy_reference.csv`, parsed at `src/validation/benchmark.rs:1808-1820`; in-band flag at `src/validation/ashrae_140_validator/mod.rs:1384-1400` |
| §8 GC10a | **Closed-form analytical** 2433 W, proposed tolerance ≤ 0.5 % | true analytical | `data/reference/ashrae140/analytical_acceptance_criteria.json` → `section_8_ground_coupled.analytical_solution`; engine oracle `fluxion-core/src/ground_coupling.rs` |
| §11 air-side | **Quasi-analytical** QASv2 reimplementation | true analytical | `fluxion-core/src/qas_airside.rs:1-3`; targets `data/reference/ashrae140/section11_qas_targets.json`; checked by `scripts/release_gate_checker.py:1076-1088` |
| §9a CE100–CE200 | **Comparative Task-22 ensemble bands only**; the analytical slot is `null` | comparative, explicitly "evidence range (ensemble min/max), not a pass/fail criterion" | `data/reference/ashrae140/section9_ce_cases.json` per case (`acceptance.comparative.role`); bands in `tests/validation/hvac_equipment_rated_performance/data/comparative_bounds_ce100_ce200.csv`; analytical slot `null` at `:49`, `:86`, `:123`, `:160`, `:197`, `:239`, `:276`, `:298`, `:335`, `:357`, `:379`, `:401`, `:423`, `:460`; generator writes `None` + note at `scripts/generate_section9_ce_cases.py:143-148` |

So fluxion already runs the *dual* model it is being asked to extend: comparative bands where
nothing better exists (§7), analytical/quasi-analytical targets where they do exist (§8 GC10a,
§11 QASv2). §9a is currently the only 140 test set that has **neither**: it has comparative
bands for 10 of 14 cases and no criterion at all.

### 2.3 §9 / CE material already in this tree

| Artifact | What it is | Citation |
|---|---|---|
| Case generator | Python generator for the CE100–CE200 table; `--check` mode verifies the committed JSON is up to date | `scripts/generate_section9_ce_cases.py:1-3`, `:36`, `:197-208` |
| Case table | 14 cases: CE100, CE110, CE120, CE130, CE140 (dry), CE150, CE160, CE165, CE170, CE180, CE185, CE190, CE195 (wet), CE200 (full-load) | `data/reference/ashrae140/section9_ce_cases.json:15-460` |
| Performance map | CE100 map (Tables 9-5…9-10) transcribed from `CE100MAP.xls` | `data/reference/ashrae140/section9_ce100_performance_map.json:2-3` |
| Engine-side map | `CePerformanceMap` / `CeTable` / `DxMapEvaluation` with `evaluate()`, ARI rating-point accessors | `src/sim/hvac/dx_performance_map.rs:2`, `:56`, `:120`, `:131`, `:273`, `:324-349` |
| Steady-state dispatch | `dispatch_ce_steady_state()` — GSR cycling dispatch, PLR from load balance, adjusted-net fan-heat basis; **refuses** the 13 cases with no transcribed entering wet-bulb and runs only CE200 | `tests/validation/hvac_equipment_rated_performance/section9_ce.rs:1`, `:9-20`, `:73-80` |
| Comparative bands | Task-22 ensemble min/max for 10 of 14 cases: `annual_cooling` (MWh) and `peak_cooling` (kW) for all 10, plus `max_zone_temp` (°C) for CE150 | `tests/validation/hvac_equipment_rated_performance/data/comparative_bounds_ce100_ce200.csv:29-47` |
| Weather binding | Each case names its normative weather file (`CE100A`, `CE110A`, `CE165A`, `CE200A`) | `data/reference/ashrae140/section9_ce_cases.json:26`, `:63`, `:290`, `:437` |
| Weather reader | TMY2 fixed-width reader written for exactly this family (`CE1xxA.TM2`) | `fluxion-core/src/weather/tm2.rs:1-10`, `:507`, `:513`, `:583` |
| Non-§9 look-alikes | `src/validation/hvac_bestest/` (RP-865 stub) and `tests/validation/hvac_equipment_rated_performance/` (rated-performance harness) — both explicitly *not* §9 | `docs/KNOWN_ISSUES.md:145`; `tests/validation/hvac_equipment_rated_performance/mod.rs:10`; `tests/validation/hvac_equipment_rated_performance/analytical.rs:18-19`; `src/validation/hvac_bestest/runner.rs:8` |
| Synthetic "E100/E200/E300" labels | A synthetic equipment taxonomy in `analytical.rs`, renamed-aware but **not** the §9 cases | `tests/validation/hvac_equipment_rated_performance/README.md:30`; `docs/KNOWN_ISSUES.md:145` |

### 2.4 Gaps relative to the OpenBSE claim

| # | Gap | Evidence |
|---|---|---|
| G1 | Zero §9 cases are executed by the engine | `data/reference/ashrae140/section9_ce_cases.json:12`; `docs/compliance/section_coverage.md:59` |
| G2 | No acceptance criterion for §9a: comparative bands are labelled evidence, not criteria; analytical slots are `null` | `scripts/generate_section9_ce_cases.py:138`, `:143-148`; `tests/validation/hvac_equipment_rated_performance/section9_ce.rs:33-37` |
| G3 | 4 of 14 cases have no comparative band either (CE165, CE180, CE185, CE195 — the wet-coil high-latent corners) | `data/reference/ashrae140/section9_ce_cases.json` — no `acceptance.comparative` key for the CE165 entry (`:281`), CE180 (`:340`), CE185 (`:362`) and CE195 (`:406`); the bounds CSV lists only CE100–CE140, CE150, CE160, CE170, CE200 (`.../comparative_bounds_ce100_ce200.csv:29-47`) |
| G4 | The §9 weather inputs are neither committed nor fetchable by the current tooling | `fluxion-core/src/weather/tm2.rs:12-20`; `scripts/fetch_ashrae140_weather.py:46-55` (only two TF members are extracted) |
| G5 | 13 of 14 cases cannot even be dispatched because their entering wet-bulb is not transcribed | `tests/validation/hvac_equipment_rated_performance/section9_ce.rs:13-20` |
| G6 | §9a is not in the strict gate and cannot be until it produces per-case monthly outputs | `scripts/check_strict_energy_gate_regression.py:60-104` |

---

## 3. Acceptance-band methodology assessment

**Question.** Should fluxion adopt quasi-analytical references where they exist, alongside or
instead of inter-program bands?

**Answer: neither blanket "yes" nor blanket "no" — it is per case type, and the axis that
decides it is *whether the standard publishes an analytic solution for that case*.** The
governing principle already in the tree is that Standard 140 has no acceptance criteria
(`data/reference/ashrae140/analytical_acceptance_criteria.json`, `_standard_structure.finding`),
so every band fluxion uses is fluxion's own engineering criterion. That makes the choice of
criterion a *methodology* decision to be made per case family and documented, exactly as
fluxion did for §8 (closed-form → criterion) and §11 (QASv2 → criterion).

| Case type | Cases | Analytic reference exists? | **Recommendation** | Rationale |
|---|---|---|---|---|
| **§9a dry-coil analytical verification** | CE100, CE110, CE120, CE130, CE140 | Yes — §9.2.1 is by definition the *analytical* verification series; the per-case quasi-analytical solution ships in the licensed 140-2023 §9 normative materials (`scripts/generate_section9_ce_cases.py:145-147`) | **Adopt quasi-analytical as the primary criterion** (proposed ±5 % on the quasi-analytical monthly/annual totals, mirroring OpenBSE's tolerance *per issue #4239*), and demote the Task-22 comparative bands to a secondary diagnostic | A dry-coil CE case has a closed-form-ish steady-state answer that a correct DX model must reproduce; where an analytic reference exists, a comparative band is strictly weaker and would hide a real defect |
| **§9a wet-coil / latent series** | CE150, CE160, CE165, CE170, CE180, CE185, CE190, CE195 | Yes for the §9.2.2 series (same licensed materials), but the latent component is where a *simulation* (psychrometrics, coil latent capacity, dehumidification) can legitimately differ | **Adopt quasi-analytical as primary on sensible+total cooling energy** (per the analytic solution's own metric definition); keep the comparative band as a **co-gate** for the cases that have one, and for CE150's `max_zone_temp` use the comparative range as the *only* available criterion (flag it as such) | Latent handling is the part most likely to differ across implementations, so dual criteria are defensible here in a way they are not for the dry series |
| **§9a cases with no comparative band** | CE165, CE180, CE185, CE195 | Yes (same licensed materials) | **Quasi-analytical only.** Do not let a case enter the suite without *any* criterion; the comparative gap is a transcription gap, not a licence to invent a band | Precedent: fluxion refuses to invent bands — see the "nothing invented" rule quoted in `tests/validation/hvac_equipment_rated_performance/section9_ce.rs:20` and the `null` analytical slots |
| **§9a "full-load rating-point" case** | CE200 | Yes (but the case's own conditions *are* the map's ARI rating point — the assertion is cross-provenance, not numeric) | **Keep the cross-provenance assertion** (case conditions vs map rating point) as the criterion for CE200; do not wrap it in a ±5 % band against a reference that is the map itself | `tests/validation/hvac_equipment_rated_performance/section9_ce.rs:13-18` |
| **§9b dynamic comparative series** | CE300–CE545 (not in fluxion's tree) | No — §9b is the *comparative* series; no analytic solution exists | **Comparative bands only**, labelled evidence, with conformance = "the required cases ran and produced the required outputs" | Already fluxion's §9b position: `docs/compliance/section_coverage.md:158` |
| **§7 / §5.2 envelope family (the current strict gate)** | 600…970, FF variants | No analytic target is committed in this tree (no §5.2/§7 entry exists in `analytical_acceptance_criteria.json`; the file has only `section_8_ground_coupled` and `section_11_airside` keys) | **Keep the inter-program ±15 % bands.** If an in-series analytic solution for 600/900 is ever transcribed, treat it as an *additional* hard anchor (like GC10a's 2433 W), not as a replacement for the inter-program band, because the inter-program band is what protects against climate/annual-drift regressions | `src/validation/benchmark.rs:115-118`; `scripts/check_strict_energy_gate_regression.py:26-33` |
| **§8 / §11 (already analytical)** | GC10a; §11 cases 1–6 | Yes | **No change.** These already use analytical targets and are the precedent that justifies the §9a recommendation | `fluxion-core/src/ground_coupling.rs`; `fluxion-core/src/qas_airside.rs:1-3` |

**The one thing fluxion should *not* adopt from OpenBSE** is the *presentation* of a tolerance
as a compliance verdict. OpenBSE's "112/112 within ±5 %" is (per issue #4239) an internal
acceptance criterion; fluxion's own doctrine, settled in #4277, is that Standard 140 issues no
certificate and that the published example-result ranges are diagnostic context
(`docs/compliance/section_coverage.md:32-38`). So fluxion's §9a criterion should be phrased as
*"fluxion's quasi-analytical agreement criterion"*, published with its tolerance and its
provenance — never as "ASHRAE 140 §9a compliant".

---

## 4. Case-generator pattern

Fluxion already has the right pattern for the *case table*; it does not yet have it for the
*case runs*.

### 4.1 What exists

- `scripts/generate_section9_ce_cases.py` is a single-source generator: it holds the transcribed
  case parameters (`:60+`, `CASES` list), reads the performance map
  (`data/reference/ashrae140/section9_ce100_performance_map.json`), reads the comparative bounds
  CSV (`tests/validation/hvac_equipment_rated_performance/data/comparative_bounds_ce100_ce200.csv`),
  and writes the machine-readable table (`scripts/generate_section9_ce_cases.py:41-58`).
- It ships a **`--check` mode** that fails when the committed JSON drifts from the generator
  (`scripts/generate_section9_ce_cases.py:36`, `:197-208`; the file's own header states
  "hand edits to the output file are prohibited — regenerate instead").
- The output file is self-describing: `_purpose`, `_standard_scope`, `_parameter_source`,
  `_units`, and `engine_coverage` (`data/reference/ashrae140/section9_ce_cases.json:1-14`).

This is exactly the "reproducible case generator" property OpenBSE is credited with
(per issue #4239), so **fluxion does not need to copy OpenBSE's generator** — it needs to
extend its own to the next layer.

### 4.2 What fluxion should do when the §9a harness work starts

1. **Keep one generator per artefact class, never hand-authored JSON.** The §9a *expected
   outputs* (per-case monthly totals, the §6.3-style output rows) must be produced by a
   generator in the same shape as `generate_section9_ce_cases.py`: parameters in code, output
   written to `data/reference/ashrae140/`, provenance recorded in the file, plus a `--check`
   mode so CI can prove the committed artefact matches its generator. A hand-typed expected-value
   table is exactly the failure mode OpenBSE hit (regenerated columns carrying the wrong
   quantity — `docs/ashrae_140/submission_pipeline_lessons.md:35-52`).
2. **Generate the criterion, not just the case.** The generator should emit, per case, the
   `acceptance` block in three slots — `analytical` (target value + tolerance + provenance),
   `comparative` (evidence range), `diagnostics` (physics identities: e.g. the
   load-balance identity the current dispatch already asserts). Today only the middle slot is
   populated, and the first slot is hard-coded `None` (`scripts/generate_section9_ce_cases.py:143`).
3. **Emit the per-case run manifest from the same generator** so the strict-gate checker can
   discover cases, tolerances and provenance the way it does today for the envelope cohort
   (`scripts/check_strict_energy_gate_regression.py:60-104` reads a fixed case tuple plus a
   baseline JSON).
4. **Make the generator the single place where the CE identifiers live.** The E1xx→CE1xx
   rename (#4296) had to touch the CSV, the manifest, the loaders and the docs — a generator
   with an identifier mapping table (`scripts/generate_section9_ce_cases.py:130-136` already
   emits the `"identifier E" + c["id"][1:]` cross-reference) removes that class of drift.

### 4.3 Consumers of this recommendation

- **#4296** (the §9 case layer: generator, CE100–CE200 table, E1xx→CE1xx rename, DX performance
  map, steady-state dispatch) is the owner of the *artefact* pattern this section extends —
  see `src/sim/hvac/dx_performance_map.rs:2` ("the §9 CE100–CE200 equipment layer (Issue #4296
  part 2)") and `tests/validation/hvac_equipment_rated_performance/section9_ce.rs:1` ("Issue
  #4296 part 3").
- **#4271** is the owner of the §9a *test set* itself — the tracker lists §9a as owned by
  #4271 with status "Not implemented as a test set"
  (`docs/compliance/section_coverage.md:59`) — and is therefore the consumer of the
  acceptance-methodology decision in §3 and of the generator extension in §4.2.
- The sibling §9b owner is #4272 (`docs/compliance/section_coverage.md:158`); it inherits the
  comparative-bands-only conclusion from §3 but not the quasi-analytical work.

---

## 5. Weather inputs

### 5.1 What §9a needs

Each §9a case is bound to a normative TMY2 weather file: the case table records
`"weather_file": "CE100A"`, `"CE110A"`, `"CE165A"`, `"CE200A"`
(`data/reference/ashrae140/section9_ce_cases.json:26`, `:63`, `:290`, `:437`), and the
generator documents that this is "the ASHRAE supplemental TMY2 file the standard binds the case
to (never committed to this repo … see `fluxion-core/src/weather/tm2.rs` for the reader and the
licensing note)" (`scripts/generate_section9_ce_cases.py:56-58`). Four distinct weather files
cover the 14 cases.

### 5.2 What fluxion's weather pipeline already supports

| Capability | Status | Citation |
|---|---|---|
| TMY2 (`.TM2`) fixed-width reader, written *for* the §9 CE weather set | **Present** | `fluxion-core/src/weather/tm2.rs:1-10` ("Parses fixed-width TMY2 files of the kind the ASHRAE 140 §9 space-cooling equipment cases are driven by (`CE1xxA.TM2` and friends)"); module registered at `fluxion-core/src/weather/mod.rs:27-31` |
| Header parse (WBAN, city, time zone, lat/lon, elevation) | Present | `fluxion-core/src/weather/tm2.rs:77-180` |
| Hourly record parse, unit scalings, missing-data policy, 8,760-record contract | Present | `fluxion-core/src/weather/tm2.rs:182-490`, `:492-540` |
| `WeatherSource` implementation (engine-facing hourly data) | Present | `fluxion-core/src/weather/tm2.rs:583` |
| Fetch-and-verify-plus-hash-pinning policy for the CE files | **Policy stated, not implemented** — no fetch/verify step exists for the CE files | `fluxion-core/src/weather/tm2.rs:12-20` ("The CE weather files are ASHRAE supplemental material. **They are never committed to this repository** … This module reads a user-supplied licensed copy from local disk; it ships no data.") |
| Fetch tooling for ASHRAE's free 140-2023 supplemental package | Present but **narrow**: it extracts exactly two members, both from `Std140_TF_Files` | `scripts/fetch_ashrae140_weather.py:44-55` (`PACKAGE_URL` = `https://xp20.ashrae.org/supplementalfiles/140-2023-AccompanyingFiles.zip`; `ZIP_MEMBERS` = `DRYCOLD.TMY725650TY.CSV`, `Tsky-Informative.xlsx`) |
| SHA-256 provenance guard for the fetched files | Present, but pins only the two TF files | `tests/reference_data/weather/ashrae140_canonical_checksums.json` |
| Canonical-weather → EPW conversion + verification for the §5.2/§7 strict gate | Present | `scripts/convert_tmy3_to_epw.py`, `scripts/check_ashrae140_weather.py`, wired in `.github/workflows/physics-pr.yml:357-359`; the gate's weather binding is `src/validation/ashrae_140_cases.rs:76-81` |
| §9 weather files present in the tree or in the fetch set | **No** | `scripts/fetch_ashrae140_weather.py:46-55`; `assets/weather/ashrae140/` is git-ignored (fetch script docstring, `:9-21`) |

### 5.3 Assessment

- **The reader side is done.** fluxion can already read the exact format §9a binds to, and it
  already knows the file is licensed supplemental material that must never be committed
  (`fluxion-core/src/weather/tm2.rs:12-20`).
- **The supply side is missing.** Nothing in the tree enumerates, extracts, hashes or verifies
  the CE TMY2 files, and the only fetch tooling is hard-wired to the two Class I files. If the
  CE_a normative materials live in the same free package
  (`140-2023-AccompanyingFiles.zip` — the §9 map transcription cites
  `Std140_CE_a_Files/Normative Materials`, `data/reference/ashrae140/section9_ce100_performance_map.json:2`,
  and the generator cites the same package, `scripts/generate_section9_ce_cases.py:19`), then the
  §9 weather problem is a *fetch-and-verify extension*, not a licensing blocker. **That is an
  assumption this evaluation could not verify**: the zip's member list was not enumerated
  offline.
- **No conversion is needed.** §9a weather is TMY2, not TMY3, so the
  `convert_tmy3_to_epw.py` step used by the §7 strict gate does not apply; the TM2 reader feeds
  the engine directly through the `WeatherSource` trait.
- **Licensing caveat.** `fluxion-core/src/weather/tm2.rs:12-20` calls the CE files "ASHRAE
  supplemental material" subject to the fetch-and-verify-plus-hash-pinning policy, while
  `scripts/fetch_ashrae140_weather.py:12-15` notes the *Class I* package is distributed freely
  with "no membership gate". Whether the CE_a folder in the same zip is equally free is
  **unknown from this tree** and must be established before any fetch script is extended; the
  separate #4285 gate covers the *licensed §9 normative analytical materials*, which is a
  different thing from the weather files.

---

## 6. Proposed follow-up issues

> **Listed here only — no issue was filed by this task.**

| # | Title | One-line scope |
|---|---|---|
| F1 | Enumerate and hash-verify the §9 CE_a normative weather files from the 140-2023 supplemental package | List the `Std140_CE_a_Files/Normative Materials` members of `140-2023-AccompanyingFiles.zip`, extract `CE100A/CE110A/CE165A/CE200A` TM2 files, and pin their SHA-256 in `tests/reference_data/weather/ashrae140_canonical_checksums.json` (extending `scripts/fetch_ashrae140_weather.py:46-55`) — no licensed analytical values involved. |
| F2 | Transcribe the §9a quasi-analytical acceptance targets into `data/reference/ashrae140/section9_analytical_targets.json` | Blocked on #4285: transcribe the 140-2023 §9.2.1/§9.2.2 per-case analytic solutions with provenance, in the shape of `data/reference/ashrae140/section11_qas_targets.json` (owned by #4271). |
| F3 | Extend `scripts/generate_section9_ce_cases.py` to emit the three-slot acceptance block | Generate `analytical` (value + tolerance + provenance), `comparative` (evidence range) and `diagnostics` per case instead of the hard-coded `analytical: None` at `scripts/generate_section9_ce_cases.py:143`, and keep `--check` green in CI. |
| F4 | Transcribe the entering wet-bulb (EWB) condition for the 13 un-dispatchable CE cases | Add the EWB slot per case with provenance so `dispatch_ce_steady_state()` can evaluate CE100–CE195 instead of refusing them (`tests/validation/hvac_equipment_rated_performance/section9_ce.rs:13-20`); source is the licensed §9 materials (#4285). |
| F5 | §9a monthly-output emitter on the BESTEST-GSR output contract | Emit the per-case monthly totals (and the February min/max columns) in the GSR CSV contract shape used by §7 (`data/reference/ashrae140/bestest_gsr_csv_contract.json`), with program identity in the header — the failure mode OpenBSE hit (`docs/ashrae_140/submission_pipeline_lessons.md:35-52`). |
| F6 | Extend the strict-gate regression checker to a second cohort (CE cases) | Generalise `scripts/check_strict_energy_gate_regression.py:60-104` so a §9a cohort can be observed with the same PASS / KNOWN-FAIL / REGRESSION verdict model and a non-raisable baseline, once F2/F4/F5 land. |
| F7 | Document the §9a acceptance methodology in the section-coverage tracker | Update `docs/compliance/section_coverage.md:59` (§9a row) and `docs/KNOWN_ISSUES.md:145` (LIMIT-42) once the §3 recommendation is adopted, keeping the "conformance = coverage, not a score" wording. |
| F8 | Comparative bounds for CE165 / CE180 / CE185 / CE195 | Re-verify the Task-22 ensemble ranges for the four cases with no comparative band and extend `tests/validation/hvac_equipment_rated_performance/data/comparative_bounds_ce100_ce200.csv` (currently covering only CE100–CE140, CE150, CE160, CE170, CE200). |

---

## 7. Verdict

### **ADOPT-PARTIALLY**

**Adopt:** the *acceptance model* — for §9a cases where the standard publishes an analytic
solution (§9.2.1 dry-coil series CE100–CE140, and the §9.2.2 wet-coil series CE150–CE195 where
the analytic metric is defined), the **quasi-analytical reference is the criterion**, with a
documented tolerance (proposed ±5 % on the analytic monthly/annual totals, the OpenBSE
tolerance *per issue #4239*, to be justified against fluxion's own resolution), and the
Task-22 comparative bands demoted to **secondary diagnostic context** — the role the tree
already assigns them (`scripts/generate_section9_ce_cases.py:138`). This is not a new idea for
fluxion: it is the §11 QASv2 pattern (`fluxion-core/src/qas_airside.rs:1-3`) applied to §9a,
and it is the same conclusion the repo already reached for §8 GC10a
(`data/reference/ashrae140/analytical_acceptance_criteria.json`).

**Do not adopt:** (a) any framing of a tolerance as an ASHRAE compliance verdict — Standard 140
issues no certificate (`docs/compliance/section_coverage.md:32-38`); (b) inter-program bands as
the *primary* criterion for the analytical-verification cases; (c) OpenBSE's case generator
(fluxion already has one at `scripts/generate_section9_ce_cases.py` and should extend it, not
replace it).

**Rejected parts of the OpenBSE claim:** nothing — but the claim is unverified (§1.4), so no
part of it may be cited as evidence until the commit is read.

### First concrete step

**F1 — enumerate and hash-verify the §9 CE_a weather files** (`CE100A`, `CE110A`, `CE165A`,
`CE200A`) from the 140-2023 supplemental package, and extend
`scripts/fetch_ashrae140_weather.py` (`:46-55`) plus
`tests/reference_data/weather/ashrae140_canonical_checksums.json` to install them under the
existing git-ignored `assets/weather/ashrae140/` root.

Rationale for this step first:

- It is the **only** step on the critical path that is **not** blocked by the #4285 licensing
  gate — the weather files may ship in the same freely distributed package that already supplies
  the Class I weather (`scripts/fetch_ashrae140_weather.py:12-15`); if they turn out to be
  licensed, F1's output is the evidence that decides that question rather than an assumption.
- It is **cheap and verifiable**: the reader already exists and is tested
  (`fluxion-core/src/weather/tm2.rs:1-10`, `:507`, `:583`), so the risk is entirely in the
  supply step.
- It **unblocks the reader-based check** of F2: without weather in place, no §9a harness
  (#4271) can produce a monthly total for *any* case, so neither the analytic targets nor the
  comparative bands can be exercised.
- It is **data-only and reversible**: it adds a fetch member list and hash pins, touches no
  engine code and no test.

Sequencing: F1 → (F3 generator three-slot acceptance block, data-only) → F2/F4 (both blocked on
#4285) → F5 emitter → F6 strict-gate cohort → F7 tracker update. F8 runs in parallel and
unblocks the comparative context for the four uncovered cases.

---

*Research/spike only. No engine code, no validation cases and no test changes were made by the
issue that produced this document; the only artefacts touched are this document and the doc
inventory registration required by `AGENTS.md`.*
