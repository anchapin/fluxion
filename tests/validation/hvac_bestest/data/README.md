# HVAC BESTEST (RP-865) reference bounds — data & provenance

Issue #1755 (Plan T1.2). This directory holds the typed, provenance-bearing
reference bounds consumed by the loader in
`tests/validation/hvac_bestest/reference_data.rs`.

## Files

| File | Contents |
|------|----------|
| `manifest.json` | File-level provenance: publications, reference programs + versions, recorded units, status legend. |
| `comparative_bounds_e100_e200.csv` | Comparative bounds for IEA SHC Task 22 Volume 1 unitary space-cooling cases (E100–E200). |
| `comparative_bounds_ae101_ae445.csv` | Comparative bounds for RP-865 airside HVAC cases (AE101–AE445). |

## Comparative vs analytical

These files hold **comparative** bounds — qualified-program ensemble (min/max)
ranges, not analytical truth. Per the parent `README.md`, comparative ranges are
*evidence, not physical constants*, and must retain source/version provenance on
every record. The loader (`reference_data.rs`) enforces that no bound reaches a
test without complete provenance.

## Schema (CSV)

```
case_id,metric,low,high,unit,source_program,program_version,source_table,source_page,status
```

- `metric` ∈ {`annual_heating`, `annual_cooling`, `peak_heating`,
  `peak_cooling`, `max_zone_temp`, `min_zone_temp`}.
- `low`,`high` — ensemble range endpoints, in `unit`.
- `unit` — the unit the value is **recorded** in (`MWh`, `kW`, `C`, …). The
  loader normalizes to SI (J, W, K); tests assert the conversion is correct.
- `source_program` — `ensemble` (multi-program range) or a single program name.
- `program_version` — version of the source program(s).
- `source_table`,`source_page` — citation locator.
- `status` — `published` | `transcribed` | `interim` (see `manifest.json`).

`mid` is **not** stored: the loader derives `mid = 0.5*(low+high)` to avoid a
redundant, drift-prone column (verified in `reference_data.rs` tests).

## Units & SI normalization

| Quantity | Recorded unit(s) | SI unit | Conversion |
|----------|------------------|---------|------------|
| Energy | MWh, kWh, GJ, MJ, kBtu, MMBtu, therm | J | see `to_si_energy` |
| Power | kW, W, Btu/h, ton_ref | W | see `to_si_power` |
| Temperature | C, K, F | K | C: +273.15; F: (F−32)·5/9+273.15 |
| Energy/area | kWh/m2, kBtu/ft2 | J/m2 | ft² = 0.09290304 m² (exact SI) |

All conversion factors are documented physical constants (IT Btu = 1055.05585 J,
1 ft² = 0.09290304 m²); none are tuning parameters.

## Status of current values

Records are marked `transcribed`: transcribed from the cited source and flagged
for independent re-verification against the primary publication PDF. This follows
the repo's established reference-data convention (cf.
`tests/reference_data/ashrae140/monthly/case_600_monthly_reference.csv`), which
requires every interim/transcribed artifact to carry full provenance and a clear
status so it can be replaced without touching test code. The loader and its unit
tests are independent of the absolute values: they verify parsing, provenance
completeness, and SI-conversion correctness.

## Sources

- Neymark et al., *Airside HVAC BESTEST … Cases AE101–AE445*, NREL/TP-5500-66000
  (2016), DOI 10.2172/1244668.
- IEA SHC Task 22, *HVAC BESTEST, Volume 1: Cases E100–E200*.
