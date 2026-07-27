# HVAC Equipment Reference Data

Analytically-derived HVAC equipment performance curves for equipment-level
cross-validation (Issue #1933). See `PROVENANCE.md` for the full citation
chain, coefficient sources, and the replacement path to direct EnergyPlus
output.

## Files

| File | Equipment | Rows | Columns |
|------|-----------|------|---------|
| `fan_affinity_laws.csv` | Centrifugal fan (VSD + VIV) | 11 | speed_ratio, flow_ratio, pressure_ratio, power_ratio_vsd, power_ratio_viv |
| `chiller_capacity_capft.csv` | Water-cooled centrifugal chiller | 40 | T_evap_C, T_cond_C, capft_raw, capft_normalized |
| `boiler_part_load_efficiency.csv` | Non-condensing hot-water boiler | 11 | plr, eta_norm_ratio, eta_absolute |
| `heat_pump_mode_transition.csv` | Air-source heat pump | 10 | T_odb_C, cop_heating, mode |

## Status

**ANALYTICAL** — derived from published engineering correlations (AHRI,
ASHRAE Fundamentals, ASHRAE 90.1, EnergyPlus TSD). Not EnergyPlus output.
Each CSV header carries a `Status:` line and full citation.

## Regeneration

```bash
python tests/reference_data/equipment/generate_equipment_reference.py
```

Deterministic; produces byte-identical output on every run.

## Consumers

- `src/validation/reference_catalog.rs` — catalogues these files under
  the `Equipment` category.
- `tests/reference_catalog_validation.rs` — validates loading, parsing,
  and analytical invariants (e.g. fan power ∝ N³, chiller CAPFT = 1.0 at
  the rated point, boiler curve monotonicity).
