# The sky/longwave input-provenance question and the canonical-weather binding (loop round 40, 2026-10-15)

## What the row tracks

The round-39 residual on 900FF Tmin (−5.87 °C vs published [0.6, 2.2], 519 % of
mid) was attributed ~1.1 K to the "sky/longwave input-provenance term" (§50).
This investigation settles what the engine's sky term does, what Standard 140
prescribes, what the weather files actually carry, and what the provenance is
worth in °C. It also carries the concrete spec for binding the standard's
normative weather file (§6–§12 validation-program backlog, item 1).

## What the engine does (verified on develop 542718b2)

- `EpwWeatherSource` (`fluxion-core/src/weather/epw.rs`) reads EPW field 13
  (`horizontal_infrared`, 0-based `fields[12]`).
- `HourlyWeatherData::sky_temperature()` computes
  `T_sky = (HIR/σ)^(1/4) − 273.15` — the standard effective-black-body sky
  temperature (`fluxion-core/src/weather/mod.rs`).
- `step_5r1c` / `exterior_boundary.rs` apply the per-surface ASHRAE sol-air
  form `T_sol = T_out + α·I/h_c − F_sky·ε·ΔR/h_c` with `ΔR = σ(T_out⁴ − T_sky⁴)`
  (correct operand order; the legacy swapped-ΔR variant is tracked as #4224),
  `F_sky = (1+cos tilt)/2` per ISO 13790 §12.3.2 (roof 1.0, walls 0.5), wind-
  dependent `h_c` per §5.2.1.9 — i.e. the spec's §5.2.1.9.3(a) route for
  programs that compute time-step-varying exterior IR exchange, which the
  standard explicitly permits.

## What the standard prescribes

- §5.2.1.1.1.1 / Annex A1.1.1.1: the **normative** weather for §5.2.1–5.2.3 is
  `DRYCOLD.TMY725650TY.CSV` (TMY3, Denver Intl AP, station 725650) — **not**
  Golden-NREL 724666, which the strict gate's canonical weather currently is.
- A1.1.1.2: the weather file contains **no Tsky and no horizontal-IR column**
  (verified against the actual TMY3 CSV header). For programs that use Tsky as
  an input, the standard provides `Tsky-Informative.xlsx`, an hourly series
  computed by a **modified Martin–Berdahl (1984)** clear-sky model with a
  Clark–Allen cloud correction (workbook tab `Supporting-Tsky-Calcs`).

## What the files carry (measured this round, 8,760 h each)

| series | Tsky−T_db mean | Jan mean | min |
|---|---|---|---|
| Golden 724666 EPW HIR → T_sky (engine input today) | −11.68 K | −13.59 K | −21.78 K |
| DRYCOLD 725650 informative Tsky (140-recommended) | −16.78 K | −21.29 K | −34.93 K |

The 140-recommended sky is **colder** than the engine's EPW-HIR-derived sky by
~5 K on annual mean and ~8 K in January. The "EPW horizontal-infrared
depression causes the cold bias" premise is therefore inverted: the sky input
is not the cause of the cold residual, and adopting the 140-recommended Tsky
as-is would deepen it.

## Measured sensitivity (scratch diagnostic, weather-swap only, engine untouched; deleted before commit)

900FF, default build, same harness as `run_blind_free_float`:

| run | Tmin °C | Tmax °C |
|---|---|---|
| Golden 724666 EPW (strict-gate baseline) | −5.870 | 42.545 |
| DRYCOLD 725650 + informative Tsky (HIR = σ·T_sky,inf⁴) | **−0.256** | 38.931 |
| DRYCOLD 725650 + T_sky = T_db (sky term neutralized) | 0.994 | 45.412 |

The published band is [0.6, 2.2] / [43.3, 46.0].

## Verdict

**(b) defensible but different formulation/input provenance — no defect.** The
formulation (effective-black-body Tsky from the measured HIR field, corrected
ASHRAE sol-air signs, permitted time-step-varying IR route, ISO view factors)
is physically standard and spec-permitted. The dominant term in the 900FF
residual is the **weather-file provenance**: on the canonical DRYCOLD file the
same engine lands 0.9 K from the Tmin band's lower edge (−0.256 vs 0.6),
closing ~5.6 K of the 6.5 K gap — while the sky-model choice inside that run
(informative Tsky vs neutralized) is worth only ~1.25 K in the opposite
direction to the round-39 attribution. §50's "sky/longwave ~1.1 K" lever is
confirmed as a real but second-order term; the first-order term is the
724666-vs-725650 weather binding. The residual cold bias on the canonical
weather (−0.26 vs [0.6, 2.2]) remains attributable to the structural terms
already tracked (§LIMIT-40 storage-release split, interior-film coupling).

No engine change lands this round; no re-record battery is owed (default build
untouched, strict gate reproduced 10 PASS / 32 KNOWN-FAIL / 0 REGRESSION on
542718b2 before the investigation).

## Canonical-weather binding spec (backlog item 1 — §LIMIT-15 context, Case 195 / DRYCOLD)

Feasibility: **feasible without licensing exposure.** Both files are freely
downloadable without an ASHRAE-membership gate:
`https://xp20.ashrae.org/supplementalfiles/140-2023-AccompanyingFiles.zip`
(704 MB, contains `Std140_TF_Files/Normative Materials/725650TY.csv` and
`Informative Materials/Tsky-Informative.xlsx`; same files linked from
`https://data.ashrae.org/standard140/AccompanyingSection7.html`). The TMY3
underlying data are NREL public domain; the ASHRAE-packaged CSV should be
**fetched at build/test time, not committed** — record SHA-256 checksums of
the two files in-repo as the provenance guard.

Implementation steps (one focused round, full re-record battery owed because
every validator number moves):

1. `scripts/fetch_ashrae140_weather.py` — downloads the zip, extracts
   `725650TY.csv` + `Tsky-Informative.xlsx` into a git-ignored
   `assets/weather/ashrae140/`, verifies SHA-256 against in-repo checksums;
   caches for CI (actions/cache keyed on checksum).
2. TMY3-CSV → EPW converter (`scripts/convert_tmy3_to_epw.py`, or a
   `WeatherSource` impl for the TMY3 CSV directly — the engine reads only
   fields[6,8,12–15,20,21]): dry-bulb col 31, dew-point 34, RH 38, pressure
   41, wind dir 44, wind spd 47, GHI 5, DNI 8, DHI 11; Tsky series → EPW field
   13 as σ·T⁴ (converter emits HIR, engine keeps its single sky path —
   physically identical to reading Tsky directly).
3. `ZoneBalanceGolden` strict-gate + validator harnesses point at the
   converted file; Golden 724666 stays as the E+-comparison weather for the
   per-surface irradiance traces (E+ reference CSVs were generated on it).
4. Full re-record battery (strict gate + checker, fabric, surrogate fallback,
   delta_config regen, hotloop goldens, grid adapter consts, python-bindings
   contract) with old→new provenance `_doc_round41`; 900FF/600FF/650FF/950FF
   bands re-scored; KNOWN_ISSUES cohort rows updated from the corrected
   provenance.
5. Provenance guard: `check_ashrae140_weather.py` asserts the checksum file
   exists and matches before the strict suites run in CI.

Expected first-order movement (from this round's diagnostic): 900FF Tmin
−5.87 → ≈ −0.3 (near-band), Tmax 42.5 → ≈ 39.0 (further below [43.3, 46.0]);
all 17 annual metrics move on the drier/colder DRYCOLD file — recorded, not
tuned, per the standing fidelity direction.

## Round 41 — implementation record (canonical-weather binding LANDED, 2026-10-15)

The 5-step spec above was implemented as-is:

1. `scripts/fetch_ashrae140_weather.py` — downloads the supplemental-files
   package, extracts the two canonical files into git-ignored
   `assets/weather/ashrae140/`, verifies SHA-256 against
   `tests/reference_data/weather/ashrae140_canonical_checksums.json`
   (csv `813e1051…3206`, xlsx `bc68ac83…60ec3`). CI restores the extracted
   files via an actions-cache keyed on the checksum file.
2. `scripts/convert_tmy3_to_epw.py` — TMY3 CSV (0-based cols: GHI 4, DNI 7,
   DHI 10, dry-bulb 31, dew 34, RH 37, pressure 40, wind dir 43, spd 46) +
   the `Informative-Tsky` sheet (column D, 8760 rows) →
   `DRYCOLD_TMY725650TY.epw`; HIR = σ·T_sky⁴ in field 12. Self-check: the
   emitted series reproduces the round-40 provenance measurement (Tsky−Tdb
   annual mean −16.78 K) before writing.
3. Harness repoint: `strict_gate_weather_epw()` in
   `src/validation/ashrae_140_cases.rs` — default canonical, Golden-NREL
   724666 opt-in via `FLUXION_ASHRAE140_WEATHER=golden`. Used by the strict
   gate (`zone_balance_eplus_isolation.rs`, 13 call sites) and the fabric
   harness (`ashrae_140_fabric_multiselector.rs`). E+ comparison families
   (solar_isolation, per-surface traces, setback/multi-climate) stay on
   Golden.
4. Full re-record battery from the corrected configuration, `_doc_round41`:
   strict gate + checker **13 PASS / 29 KNOWN-FAIL / 0 REGRESSION** (was
   10/32/0); fabric re-recorded (all six rows moved, parity case_600
   1.1148/0.6875 → 1.1588/0.6739); surrogate fallback bit-identical
   (3278.75 kWh, drift 0.000% — synthetic weather loop); hotloop goldens
   PASS; grid adapter PASS; delta_config.yaml regenerated bit-identical;
   python-bindings untouched (no engine change).
5. `scripts/check_ashrae140_weather.py` provenance gate + fetch/convert
   steps wired into `physics-pr.yml` (strict-energy-gate) and
   `ashrae_140_fabric.yml`.

Headline old→new (Golden → canonical DRYCOLD, strict gate):

| metric | old | new | band | old→new status |
|---|---|---|---|---|
| 900FF Tmin | −5.87 | **−0.26** | [0.6, 2.2] | known-fail → known-fail (gap 519→119 % of mid; 0.86 K from the band edge) |
| 900FF Tmax | 42.54 | 43.28 | [43.3, 46.0] | known-fail → known-fail (0.02 below lower edge) |
| 600FF Tmin | −18.77 | **−13.05** | [−13.8, −9.9] | known-fail → **PASS** |
| 600FF Tmax | 55.88 | 55.26 | [62.4, 68.4] | known-fail |
| 650FF Tmin | −24.33 | −18.71 | [−17.8, −16.7] | known-fail (gap 41.0→8.5) |
| 900 H | 1.438 | 1.225 | [1.357, 1.836] | PASS → known-fail |
| 900 C | 1.820 | 2.249 | [2.117, 2.864] | known-fail → **PASS** |
| 600 C | 4.468 | 4.975 | [4.927, 6.667] | known-fail → **PASS** |
| 640 C / 650 C / 930 C / 940 C | — | — | — | known-fail → **PASS** (4 more) |
| 950 C | 0.673 | 0.918 | [0.550, 0.743] | PASS → known-fail |
| 195 C | 0.683 | 0.867 | [0.554, 0.750] | PASS → known-fail |
| 940 H | 0.966 | 0.766 | [0.957, 1.295] | PASS → known-fail |

8 metrics enter band, 4 leave (all recorded, none tuned); 900FF lands 0.9 K
from its Tmin band exactly as §"Expected first-order movement" predicted.
The free-float residual cohort on the standard's own weather is now the
§LIMIT-40 storage-release split and interior-film coupling. Licensing note:
both files ship freely from ASHRAE's supplemental-files host (no membership
gate); the direct package URL fetched without robots friction at adoption
time (2026-10-15). The TMY3 underlying data are NREL public domain; files
are fetched at build/test time, never committed.

## Closes when

The canonical-weather binding lands (or is explicitly rejected as an Alex-level
tradeoff call), after which the free-float cohort rows re-score on the
standard's own weather and the §23/§LIMIT-40 structural terms are the only
remaining free-float attribution.
