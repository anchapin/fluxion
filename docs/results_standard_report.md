# Fluxion Standard Results Report — `results_v1` (Issue #4104)

<!-- 7-line summary for AI agents: lines 1-7 -->
<!-- 1: This document specifies the versioned standard results report (results_v1). -->
<!-- 2: Read it before consuming, generating, or extending the report. -->
<!-- 3: Key concepts: one constructor for REST/CLI/Python, byte-determinism, drift gate. -->
<!-- 4: Also covers the schema sections, units, and the versioning policy. -->
<!-- 5: Companion to schemas/results_v1.schema.json and src/api/report.rs. -->
<!-- 6: Stable as of issue #4104 — schema_version 1.0.0; minor = additive, major = breaking. -->
<!-- 7: After changes, regenerate the reference and run scripts/check_results_report_drift.py. -->

## TL;DR

The standard results report is Fluxion's EnergyPlus "Annual Building Utility
Performance Summary" equivalent: one documented, versioned, byte-deterministic
JSON document generated from a finished simulation — the hourly end-use series
(Issue #4101), the monthly summaries (Issue #4102), and the unmet hours
(Issue #4103). It closes the results story of Issue #3988.

Every surface produces the report through the **single constructor**
`StandardReport::from_simulation` in `src/api/report.rs`:

- **REST** — `POST /v1/simulate` with `options.report = true` carries the
  report object built by that constructor (the batch surface leaves it
  `None`; the report there is opt-in per simulation).
- **CLI** — `fluxion --report <model>` writes `{prefix}_report.json` next to
  `{prefix}_results.json`.
- **Python** — `MultiZoneThermalModel.get_standard_report(...)` returns the
  canonical JSON text.

There is no per-surface report logic, so the three surfaces cannot drift
apart.

## Schema

The normative schema is `schemas/results_v1.schema.json` (JSON Schema draft
2020-12, `$id: https://fluxion.dev/schemas/results_v1.schema.json`). The
document carries `schema_version: "1.0.0"` (semver).

### Sections

| Section | Contents |
|---|---|
| `metadata` | `schema_version`, `tool_version` (crate version), `model_name`, `model_source`, and `run_period` (`start`, `end`, `label`, `timestep_seconds`, `timesteps`, `years`). |
| `annual` | `end_use_kwh` (heating, cooling, lighting, equipment), `total_energy_kwh`, `eui_kwh_per_m2`, `floor_area_m2`, `peaks` (timestep-average kW per end use), `zone_count`. |
| `monthly` | Exactly 12 entries (January–December, `month` 1–12), each with `end_use_kwh` and `end_use_peak_kw`. Months are aggregated over the years of the run period (kWh summed, peak kW maximised), so the twelve bins add up to the annual totals. |
| `unmet_hours` | `occupied` and `all_hours` heating/cooling unmet-hour pairs, the setpoint `tolerance_c`, and a `by_zone` list in zone index order. |
| `hourly` | A **reference** to the hourly series — `series_id`, `end_uses`, `columns`, `timestep_seconds`, `timesteps`, and a `checksum` (`sha256:<hex>` over the canonical series text). The series itself is not embedded; it travels via the existing channels, and a consumer can recompute the checksum to detect a mismatch. |

### Units and conventions

- Energy is always **kWh**; demand is **timestep-average kW**; temperatures
  are **°C**; areas are **m²**; durations are **hours**.
- **Byte-determinism:** object keys are sorted, zones are in index order,
  and every float is rendered with exactly six decimals in plain (never
  scientific) notation; negative zero is normalised to `0.000000`. Identical
  inputs produce a byte-identical document.
- Non-finite values never appear: a non-finite sample counts as zero in
  every aggregate, so the document is always schema-valid JSON.

## Drift gate

The committed reference report at
`tests/reference_data/results_v1/standard_report.json` is regenerated from
deterministic fixture inputs (`examples/minimal.flux` + the Denver TMY EPW)
by `scripts/check_results_report_drift.py` and byte-compared against the
tree. Exit codes: 0 = no drift, 1 = drift, 2 = environment error. After an
intentional report change, regenerate the reference with
`python3 scripts/check_results_report_drift.py --update` and commit it in
the same PR.

## Versioning policy

- **Minor** versions are additive only: new optional fields or sections may
  appear; existing fields keep their meaning, units, and format.
- **Major** versions are breaking: a field is removed, renamed, re-typed, or
  changes meaning/units. A major bump ships a new schema file
  (`results_v2.schema.json`) alongside the old one.

## Non-goals (deferred, not in v1)

ESO/SQL parity, source-energy / emissions rollups, cost / tariff
computation, and sub-hourly series embedding.
