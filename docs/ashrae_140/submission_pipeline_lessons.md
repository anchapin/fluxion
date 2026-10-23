# ASHRAE 140 Submission-Pipeline Lessons — OpenBSE v0.7.0 and the Fluxion Gap Audit

**Issue:** #4554 (review OpenBSE v0.7.0's ASHRAE 140 submission pipeline)
**Status:** review complete — analysis only; no engine, emitter or gate code changed.
**Audience:** the Fluxion results story (#3988) and the strict-gate hardening work.
**Date:** 2026-10-23

---

## 1. Scope and sources

Two kinds of source are used, and they are kept strictly apart:

| Source | What it gives | How it is cited |
|---|---|---|
| Issue #4554 body (the OpenBSE v0.7.0 release summary) | The OpenBSE-side lessons: February max/min columns, program-identity header, portal missing-data failure, engine features added for §9B | "per issue #4554" |
| This repository (fluxion, base `develop` at `7b5d92bb`) | Everything about Fluxion's own output generation, gates and contracts | `file:line` citations plus quoted command output |

No external facts about OpenBSE beyond the issue body are asserted here; the release
notes were not fetched (offline review). Every OpenBSE-side statement below is
marked *per issue #4554* and must be re-checked against the release page before it
is repeated in the results story.

---

## 2. OpenBSE v0.7.0 lessons (per issue #4554)

OpenBSE v0.7.0 (2026-10-12) implements the full ASHRAE 140-2023 §9B (CE_b) dynamic
HVAC BESTEST suite — all 20 CE_b cases (CE300–CE545) gated and inside the §9B
comparative ranges — and raised its validation gate to 550 checks, with portal
submissions regenerated for all five categories (TF, CE_a 14/14, CE_b 20/20,
HE 11/11, AE) *(per issue #4554)*.

The reason this matters to Fluxion is the failure mode, not the feature list.
OpenBSE's **Oct 2 CE submission had data-integrity failures — February max/min
columns and a program-identity header — that surfaced only after a portal
missing-data failure** *(per issue #4554)*. That is the classic submission-pipeline
shape:

1. a portal rejects a submission because data is *missing*;
2. the submitter regenerates the affected outputs under time pressure;
3. the regenerated outputs are *structurally* valid but *semantically* wrong
   (a column carries the wrong quantity — February's max/min instead of the
   intended one);
4. the submission carries a header that does not identify the program, so the
   reviewer cannot tell which program/version produced the rows.

### 2.1 The four transferable lessons

| # | Lesson (per issue #4554) | Why it bites | Fluxion analogue checked here |
|---|---|---|---|
| L1 | **Column semantics drift under regeneration.** The February max/min columns were wrong in the regenerated CE submission. | A column can be *present*, *well-formed* and *the wrong quantity*. Structural validators pass; only a value-level check catches it. | §4.1 — the §7 gate checks shape only; a min/max swap passes it (§4.1, experiment E1). |
| L2 | **Program identity must live in the header, not in prose.** The regenerated submission's program-identity header was defective. | A portal/reviewer must attribute every row to a program *and version* without asking. | §4.4 — the §7 CSV carries the bare name "Fluxion" and no version; §6/§8/§11 artifacts carry no identity at all. |
| L3 | **Missing data is a pipeline failure, not a per-case failure.** The portal missing-data failure was the trigger; the data-integrity failures were the consequence. | Per-case gates ("did this case run?") do not catch "this category was not submitted at all". | §4.2 — the §7 gate accepts a 3-row CSV; there is no category-level manifest. |
| L4 | **Date-format conventions differ between sections and must be pinned per section.** (S6 vs S7 date-format conventions.) | A §6-style `MM-DD-HH` token read with §7's `DD-MMM-HH:MM` convention silently mis-dates the data (February read as September) with no error raised. | §4.3 — Fluxion's §6 and §7 outputs really do use different conventions, and §6's daily CSV uses a third. |

### 2.2 Engine features OpenBSE had to add for §9B (context, not a Fluxion gap list)

Per issue #4554, OpenBSE had to add: seasonal (date-range) schedules; performance-map
DX outdoor-air coupling on the PSZ cycling path; economizer extensions
(compressor lockout, fixed + differential high limits, WARMEST load reset);
scheduled minimum outdoor air + seasonal infiltration; and cooling setback.

Fluxion's §9 work today is steady-state equipment only (§9 CE100–CE200, §4.2), so
none of the dynamic §9B features exist for Fluxion. The transferable lesson is the
pipeline shape (regenerate → verify → submit), not the feature list.

---

## 3. What Fluxion generates today (submission-pipeline inventory)

| Section | Artifact | Emitter | Gate / consumer |
|---|---|---|---|
| §6 Weather Driver | `data/reference/section6/section6_workflow_results.csv`, `data/reference/section6/section6_table61_daily_hourly.csv` | `tests/ashrae_140_section6_weather_driver.rs` | `scripts/check_ashrae140_weather.py`, `scripts/check_ashrae_140_fabric_regression.py` |
| §7 Envelope & Fabric Load | `data/reference/section7/workflow_results.csv` (BESTEST-GSR contract form), `data/reference/section7/section621_outputs.csv` | `tests/ashrae_140_section7_gsr_emitter.rs` | `scripts/check_bestest_csv.py` + `data/reference/ashrae140/bestest_gsr_csv_contract.json` |
| §8 In-depth | metric spec `docs/ashrae_140/section_8_output.md`; §8.2.4 compliance CSV (`src/validation/diagnostic.rs:751`) | `src/validation/diagnostic.rs` | GC10a analytic pins, `fluxion-core/src/ground_coupling.rs:151` |
| §9 Equipment | `data/reference/ashrae140/section9_ce_cases.json` (CE100–CE200) | `scripts/generate_section9_ce_cases.py` | `tests/validation/hvac_equipment_rated_performance/section9_ce.rs` |
| §11 Air-side (AE/QASv2) | §11 results JSON (`fluxion-core/src/qas_airside.rs:719`) | `fluxion-core/src/qas_airside.rs` | QASv2 case-1 anchor pins (`fluxion-core/src/qas_airside.rs:664`) |

There is **no** single "submission manifest" listing which categories and cases a
submission contains — that absence is the root of L3 (gap G3 below).

---

## 4. Audit of Fluxion's output generation against the OpenBSE failure classes

### 4.1 (a) Does `scripts/check_bestest_csv.py` check column *semantics* and *case coverage*, or only columns?

**Finding: only columns, and only their shape. No value semantics, no case coverage.**

`scripts/check_bestest_csv.py` is the §7 submission gate. What it actually does:

| Check | Where |
|---|---|
| Column 1 header must be `bestest.case_num` | `scripts/check_bestest_csv.py:81-83` |
| Every contract column present (error, or warning under `--allow-missing-fields`) | `scripts/check_bestest_csv.py:85-91` |
| No "Ruby symbol form" columns (prefix without the dot) | `scripts/check_bestest_csv.py:92-95` |
| Key column non-empty; duplicate case keys rejected (the consumer keeps the last one silently) | `scripts/check_bestest_csv.py:97-107` |
| Timestamp **syntax**: `DD-MMM-HH:MM`, uppercase 3-letter month, day 1..31, hour ≤ 24, minute ≤ 59, no nonzero minutes at hour 24 (the consumer would emit hour 25) | `scripts/check_bestest_csv.py:26-52` |
| Array **lengths**: monthly arrays exactly 12, day profiles ≥ 24 | `scripts/check_bestest_csv.py:51-63`, applied at `scripts/check_bestest_csv.py:109-123` |

What it does **not** do — precisely the OpenBSE L1/L3 exposure:

1. **No value semantics.** Nothing checks that `min_temp ≤ max_temp`, that a peak
   is non-negative, that `annual_heating` equals the sum of the twelve
   `monthly_htg_cons` entries, or that a day profile is plausible for its case. A
   February max/min mix-up is exactly this class of defect: present, correctly
   shaped, wrong quantity. The gate passes it (experiment **E1** below).
2. **No case coverage.** The script counts case keys but never compares them to an
   expected roster: there is no `expected_cases` field in
   `data/reference/ashrae140/bestest_gsr_csv_contract.json` (the contract has
   `key_column`, `field_prefix`, `fields`, `timestamp_format`, `array_format`,
   `template_map`, `counts` — and the `counts` block records *totals*, not the
   roster). A submission containing 3 of 53 cases exits 0 (experiment **E2**).
3. **No category-level check.** §6/§7/§8/§9/§11 artifacts are validated by
   different scripts, and nothing verifies that *all* categories were produced in
   a submission run.

**Experiments (run in this worktree, quoted verbatim).**

```
$ python3 scripts/check_bestest_csv.py data/reference/section7/workflow_results.csv
18 rows, 18 case keys, 0 errors, 0 warnings
(exit 0)

E1 — semantic corruption (min/max swapped on case 600):
$ python3 - <<'PY'   # copy the committed CSV, swap min_temp/max_temp on row 1
...                    # -> min=28.60, max=18.77
PY
$ python3 scripts/check_bestest_csv.py /tmp/mutated_bestest.csv
18 rows, 18 case keys, 0 errors, 0 warnings
(exit 0)                       <-- min > max is NOT caught

E2 — missing-data blindness (keep header + 3 case rows only):
$ python3 scripts/check_bestest_csv.py /tmp/partial_bestest.csv
3 rows, 3 case keys, 0 errors, 0 warnings
(exit 0)                       <-- 3 of 53 cases is NOT caught
```

E1 is the OpenBSE L1 class reproduced locally on Fluxion's own gate; E2 is the
OpenBSE L3 class (portal missing data) reproduced locally.

**Counter-evidence (so the finding is not overstated):** the committed §7 data
*does* satisfy the semantics the gate omits — an ad-hoc pass over
`data/reference/section7/workflow_results.csv` found 0 issues for `min ≤ max`,
non-negative peaks, and `annual == Σ monthly` (both heating and cooling, 2 %
tolerance). So G1 is implementable without false positives on today's data.

**Mitigation already in place:** the emitter's Rust tests pin the case set and
headline values — `tests/ashrae_140_section7_gsr_emitter.rs:752`
(`assert_eq!(results.len(), 18, ...)`), the contract shape test at
`tests/ashrae_140_section7_gsr_emitter.rs:1073`, and the value pins at
`tests/ashrae_140_section7_gsr_emitter.rs:1205` — plus a committed-artifact
byte-identity test at `tests/ashrae_140_section7_gsr_emitter.rs:1176`. These catch
drift in CI for the *committed* §7 set; they do not cover a submission assembled
from artifacts the gate has never seen.

### 4.2 (b) How many cases does the emitter emit vs how many exist?

**Finding: 18 emitted against a 53-case BESTEST-GSR roster (~34%); 45 implemented
cases in total across §6/§7/§8/§9/§11, with no §9B (CE_b dynamic) and no §10 (HE)
coverage at all.**

* Emitted §7: `pub const SECTION7_CASES: [Section7CaseDef; 18]`
  (`tests/ashrae_140_section7_gsr_emitter.rs:74`); the committed CSV has 18 data
  rows + header (`data/reference/section7/workflow_results.csv` = 19 lines); the
  case ids are `600, 610, 620, 630, 640, 650, 900, 910, 920, 930, 940, 950, 960,
  970, 600FF, 650FF, 900FF, 950FF`.
* Expected by the consumer: the contract's `counts` block records
  `main_block_cases: 46` and `free_float_cases: 7`
  (`data/reference/ashrae140/bestest_gsr_csv_contract.json`) — 53 case rows the
  populate script expects, against Fluxion's 18.

The emitter is explicit that this is a scope decision, not an accident:

> "Only implemented cases are emitted (contract rule: the consumer reads its case
> list off the template, so omitted cases fail one lookup rather than corrupt
> layout)."
> — `tests/ashrae_140_section7_gsr_emitter.rs:28-31`

**Severity:** High as a *submission* risk (a portal submission with 18 of 53 rows
is a missing-data submission, the exact class that triggered OpenBSE's failure),
but **not** a code defect. The correct response is a coverage report that states
the ratio before submission, not a change to the emitter.

Coverage by section:

| Section | Cases implemented | Evidence |
|---|---|---|
| §6 | 6 (WD100–WD600) | `tests/ashrae_140_section6_weather_driver.rs:264` — `SECTION6_CASES: [Section6CaseDef; 6]` |
| §7 | 18 | `tests/ashrae_140_section7_gsr_emitter.rs:74` |
| §8 | 1 analytic case (GC10a) | `fluxion-core/src/ground_coupling.rs:151` — `gc10a_closed_form_watts` |
| §9 | 14 (CE100–CE200 — §9 CE_a steady-state; **no CE_b / dynamic §9B cases**) | `data/reference/ashrae140/section9_ce_cases.json` (`case_id` entries CE100…CE200) |
| §11 | 6 ambient cases (AE/QASv2) | `fluxion-core/src/qas_airside.rs:79` — `TEST_CASES: [TestCaseInputs; 6]` |
| **Total** | **45** | vs OpenBSE's fully gated TF / CE_a 14 / CE_b 20 / HE 11 / AE *(per issue #4554)* |

Compare the *strict energy* gate, which already enforces case-level completeness
in the same spirit: `scripts/check_strict_energy_gate_regression.py` requires 17
annual cases (`SUPPORTED_CASES`, `scripts/check_strict_energy_gate_regression.py:60-80`)
plus the 4 free-float cases (`FF_CASES`, `scripts/check_strict_energy_gate_regression.py:104`)
and "fails the gate closed on missing coverage" (module docstring,
`scripts/check_strict_energy_gate_regression.py:34-37`). The submission-artifact
gate has no equivalent — that asymmetry is gap G2.

### 4.3 (c) Date-format conventions in §6 vs §7 outputs

**Finding: the sections use genuinely different conventions, §6's is ambiguous, and
§6's companion daily CSV uses a third.**

* **§7** uses `DD-MMM-HH:MM`, hour-ending (`HH = hour0 + 1`), minutes always `00`,
  because the BESTEST-GSR consumer parses timestamps by character offset:
  `fn format_dd_mmm_hhmm(at: (u32, u32, u32))` at
  `tests/ashrae_140_section7_gsr_emitter.rs:309-327`; the hour convention is
  documented at `tests/ashrae_140_section7_gsr_emitter.rs:34-37`.
* **§6** uses a numeric `MM-DD-HH` with **no month abbreviation** and **no
  separator between date and hour**:
  `fn format_mdh(mdh: &(u32, u32, u32)) -> String { format!("{:02}-{:02}-{:02}", mdh.0, mdh.1, mdh.2) }`
  at `tests/ashrae_140_section6_weather_driver.rs:749-751`; used for
  `peak_heating_occurrence`, `peak_cooling_occurrence`,
  `sky_temp_min_occurrence`, `sky_temp_max_occurrence`
  (`tests/ashrae_140_section6_weather_driver.rs:734-744`).

Observed committed data:

```
$ head -2 data/reference/section6/section6_workflow_results.csv
case,description,weather_file,...,peak_heating_occurrence,peak_cooling_kw,peak_cooling_occurrence,...
WD100,"Denver Intl AP, CO — baseline, high elevation",WD100.epw,...,4.7108,02-09-05,3.7685,10-01-13,...
```

Three hazards follow:

1. **Ambiguity.** `02-09-05` in §6 means *month 02, day 09, hour 05*; the same
   token under §7's convention means *day 02, month SEP, hour 05*. A reviewer or
   downstream script applying one convention to both sections mis-dates the other —
   including, per L4, moving a February row into a September row with no error
   raised.
2. **Hour basis is invisible.** §6 emits the raw 0-based engine hour, §7 emits
   hour-ending. Same quantity, two conventions, one artifact family.
3. **A third convention in §6's daily CSV.** `section6_table61_daily_hourly.csv`
   splits the timestamp into separate numeric columns with a 0-based hour:
   header `case,month,day,hour,...` (`tests/ashrae_140_section6_weather_driver.rs:385-389`),
   rows like `WD100,1,4,0,...` — again a different shape from §7's single
   `DD-MMM-HH:MM` field.

A fourth, unrelated, packing divergence compounds this: §7 packs its monthly
arrays as comma-separated values inside one quoted cell (contract
`array_format.encoding`), while §6 packs its monthly arrays **slash**-separated
inside one cell (`0.9986/1.0142/...`, `tests/ashrae_140_section6_weather_driver.rs:447-452`).
Two packing conventions and three date conventions across two adjacent sections of
one submission is the drift surface OpenBSE tripped over.

### 4.4 (d) Program-identity / header fields in generated reports

**Finding: the §7 CSV carries a program name but no version; the §6, §8 and §11
artifacts carry no program identity at all.**

* The §7 CSV's three trailing unprefixed columns are `building_name`, `dir_name`,
  `status` (`data/reference/ashrae140/bestest_gsr_csv_contract.json` →
  `fields.trailing_unprefixed`); emitted by `section7_gsr_csv_header()` at
  `tests/ashrae_140_section7_gsr_emitter.rs:809-876` (trailing columns added at
  `tests/ashrae_140_section7_gsr_emitter.rs:871-873`).
* The row writer hard-codes the program name and **reuses the case id as the
  "directory name"**:
  `format!("{},\"{}\",{},{},{}", key, fields.join("\",\""), "Fluxion", r.case_id, status)`
  — `tests/ashrae_140_section7_gsr_emitter.rs:963-970`.

Committed values confirm the shape:

```
$ python3 -c "...csv.DictReader(...)..."
building_name values: {'Fluxion'}
dir_name sample: ['600', '610', '620', '630', '640']
status: {'emitted', 'emitted_free_float'}
```

Consequences:

1. **No version, no build, no generation timestamp.** "Fluxion" alone is the whole
   identity. OpenBSE's program-identity-header failure was a defect in exactly
   this slot; a bare program name with no version is the same weak shape.
2. **`dir_name` is not a directory.** It is the case id (`600`, `610`, …), so the
   field name promises provenance the value does not carry.
3. **The §7 conformance Markdown report carries no identity either** — its header
   is a prose blob beginning `"# §7 conformance table (#4276)"`
   (`tests/ashrae_140_section7_conformance.rs:619-660`): no program, no version,
   no generation date, no input-weather provenance.
4. **The §6 CSV has no identity columns at all** — its header starts
   `case,description,weather_file,city_state,latitude_deg,...`
   (`tests/ashrae_140_section6_weather_driver.rs:433-443`).
5. **The §8 §8.2.4 compliance CSV has no identity** — its header is three comment
   lines (`# Hourly Zone Temperature Profiles (°C)`,
   `# ASHRAE 140-2023 Section 8.2.4 Compliance Data`, `# Case ID, Hour,
   Temperature_C`), `src/validation/diagnostic.rs:748-753`.
6. **The §11 JSON carries a method statement but no program/version** — it emits
   `"method": "QASv2 quasi-analytical solution reimplementation (Issue #4274); …"`
   and `"case_1_anchor_check": …` but no program identity block
   (`fluxion-core/src/qas_airside.rs:719-742`).

### 4.5 Adjacent observations (same failure class)

* **The consumer itself has a date bug Fluxion already documented.** The contract
  records that the consumer derives dates from the raw 8760 index rather than from
  the timestamp fields, rendering e.g. `DEC 32` at year end
  (`_consumer_date_quirk` in
  `data/reference/ashrae140/bestest_gsr_csv_contract.json`). That is an upstream
  bug, but it means Fluxion's `min_index_position`/`max_index_position` columns
  must be read with the consumer's off-by-one in mind — a Fluxion-side value
  check cannot trust the rendered date.
* **The gate is wired into CI, but on the committed artifact only.**
  `.github/workflows/docs-hygiene.yml:138` runs
  `python3 scripts/check_bestest_csv.py data/reference/section7/workflow_results.csv`.
  There is no invocation from the emitter itself, and the emitter lives in a test
  file (`tests/ashrae_140_section7_gsr_emitter.rs`), so "the CSV a submission
  would carry" and "the CSV the gate checks" are the same file only by
  convention.
* **Two doc claims about that gate are inaccurate.** `AGENTS.md:139` and
  `docs/scripts/README.md:43` both say the CSV is "the CSV `fluxion validate
  --report` emits". No CLI code emits it:
  `grep -rn "workflow_results\|gsr" --include=*.rs src` → no matches; and
  `grep -rn '"report"' src/cli/validation.rs src/cli/mod.rs` → no matches.
  The CSV is produced by the §7 emitter test harness and committed at
  `data/reference/section7/workflow_results.csv`. (Reported as a gap, G12; not
  fixed here, since these are hand-maintained tables with their own gates.)

---

## 5. Gap checklist

Severity: **H** = can cause a portal rejection or a wrong submission;
**M** = can cause a review question or a late catch; **L** = hygiene.

| # | Gap | Where | Severity | Suggested follow-up |
|---|---|---|---|---|
| G1 | Submission gate checks column *shape* only; no value semantics (L1). Proven: a min/max swap passes with 0 errors (E1). | `scripts/check_bestest_csv.py:85-123` | H | Add semantic checks: `min ≤ max` for temp/peak pairs, non-negative consumption and peaks, `annual == Σ monthly` (2 %), monthly array January-first, day-profile length/range sanity. Verified implementable: today's committed CSV yields 0 issues under these checks. |
| G2 | No case-coverage check: a 3-of-53 submission passes the gate silently (L3). Proven by E2. | `scripts/check_bestest_csv.py` (no expected roster); contract `counts` block | H | Add `expected_cases` to `data/reference/ashrae140/bestest_gsr_csv_contract.json` and fail the gate when a case is missing; mirror the precedent in `scripts/check_strict_energy_gate_regression.py` (coverage fails closed). |
| G3 | No category-level completeness gate across §6/§7/§8/§9/§11, and no submission manifest. | repo-wide; no such script exists | H | Add a `submission_manifest` (category → case → artifact path → hash) and a gate that refuses sign-off when a category is empty — this is the direct analogue of OpenBSE's portal missing-data failure. |
| G4 | Program identity lacks a version; `dir_name` carries a case id, not a directory (L2). | `tests/ashrae_140_section7_gsr_emitter.rs:871-873`, `:963-970` | H | Emit `program_name`, `program_version`, `build_id`, `generated_utc` as explicit fields (contract + emitter + gate). |
| G5 | §6 uses numeric `MM-DD-HH`; §7 uses `DD-MMM-HH:MM`; hour basis differs (0-based vs hour-ending) (L4). | `tests/ashrae_140_section6_weather_driver.rs:749-751` vs `tests/ashrae_140_section7_gsr_emitter.rs:309-327` | H | Pick one canonical per-section convention, document it in the contract, and run §6 timestamps through the same shape checker so a §6 row can never be read as a §7 row. |
| G6 | §6 daily/hourly companion CSV uses a third timestamp shape (numeric `month,day,hour` columns, 0-based hour). | `tests/ashrae_140_section6_weather_driver.rs:385-389`; `data/reference/section6/section6_table61_daily_hourly.csv` | M | Emit `DD-MMM-HH:MM` alongside (or instead of) the numeric columns and pin the format in a test. |
| G7 | §6/§7/§8/§11 reports carry no program identity block. | `tests/ashrae_140_section7_conformance.rs:619-660`; `tests/ashrae_140_section6_weather_driver.rs:433-443`; `src/validation/diagnostic.rs:748-753`; `fluxion-core/src/qas_airside.rs:719-742` | M | Add a standard identity header (program, version, generated-at, input weather + hash) to every generated report. |
| G8 | §9 coverage is CE_a only (CE100–CE200, 14 cases); no §9B/CE_b dynamic cases and no §10 (HE) coverage. | `data/reference/ashrae140/section9_ce_cases.json`; no HE artifacts in the repo | M (H at submission time) | Track as an explicit roadmap item; until then the manifest must mark §9B/§10 as *not submitted* rather than omit them. |
| G9 | The emitter lives in a test file, so "the submission CSV" is not produced by a shipped tool; the gate is wired only to the committed artifact. | `tests/ashrae_140_section7_gsr_emitter.rs`; `.github/workflows/docs-hygiene.yml:138` | M | Move the emitter into `src/` (or a bin) and have the test target run the gate script on the freshly produced CSV, so one shape definition exists. |
| G10 | Consumer-side date bug (`DEC 32`, index-derived dates) is recorded but not defended against. | `_consumer_date_quirk` in `data/reference/ashrae140/bestest_gsr_csv_contract.json` | L | Keep the note; when upstream fixes it, re-run the dry-run and re-record the affected `min/max_index_position` interpretation. |
| G11 | The §7 conformance table records honest `partial`/`not implemented` cells but there is no machine-readable submission-readiness roll-up. | `tests/ashrae_140_section7_conformance.rs:619-660` | M | Emit a per-category readiness summary (cases conformant / incomplete / missing) as part of the results story. |
| G12 | `AGENTS.md:139` and `docs/scripts/README.md:43` claim the §7 CSV is what `fluxion validate --report` emits; no CLI code emits it (grep evidence in §4.5). | `AGENTS.md:139`, `docs/scripts/README.md:43` | M | Correct both tables to name the emitter test harness and the committed artifact path (doc-only fix; left undone here because those tables are gated by `scripts/check_gate_inventory_sync.py`). |

---

## 6. Verification performed in this worktree

Commands and their verbatim output are recorded in `WORKER_REPORT.md`. Summary:

* `python3 scripts/check_doc_link_integrity.py` → `PASS: All doc references resolve.`
  (385 files scanned, 754 references, 0 failures).
* `python3 scripts/check_docs_summaries.py` → `PASS: All docs/**/*.md files have a 7-line summary.`
  (this document included: `OK: docs/ashrae_140/submission_pipeline_lessons.md`).
* `python3 scripts/check_bestest_csv.py data/reference/section7/workflow_results.csv`
  → `18 rows, 18 case keys, 0 errors, 0 warnings` (exit 0) — the baseline
  behaviour, and the evidence behind E1/E2.
* `python3 scripts/check_doc_inventory_fresh.py` → reports the inventory as stale
  **only** because the generator enumerates *git-tracked* docs
  (`scripts/generate_doc_inventory.py:39-56`) and this new document is untracked
  in a worktree where commits are not allowed. The row for this document was
  added to `docs/doc-inventory.md` in exactly the byte form the generator emits
  (verified by re-running the generator's table builder with this document
  included: the diff against the committed inventory is zero lines).
  After the document is staged/committed, re-run
  `python3 scripts/generate_doc_inventory.py` per `AGENTS.md:118`.
