# HVAC Equipment Reference Data — Provenance

This directory ships **analytically derived** HVAC equipment performance
curves added in **Issue #1933** (cross-validation reference data library
expansion). These datasets are NOT EnergyPlus simulation outputs — they
are computed from published engineering correlations so that
equipment-level validation can proceed while the EnergyPlus equipment
regeneration path is stood up.

## Regeneration

```bash
python tests/reference_data/equipment/generate_equipment_reference.py
```

The generator is deterministic (no random numbers, no network access) and
re-emits byte-identical CSVs on every run.

## Datasets

| File | Equipment | Correlation | Source |
|------|-----------|-------------|--------|
| `fan_affinity_laws.csv` | Centrifugal fan (VSD + VIV) | Affinity laws Q∝N, P∝N², W∝N³ | ASHRAE Fundamentals Ch. 21; ASHRAE 90.1-2022 §6.5.3.1 |
| `chiller_capacity_capft.csv` | Water-cooled centrifugal chiller | Biquadratic CAPFT(T_evap,T_cond) | AHRI 550/590; EnergyPlus TSD commercial reference |
| `boiler_part_load_efficiency.csv` | Non-condensing hot-water boiler | Normalized efficiency curve η(PLR) | ASHRAE 90.1-2022 Table 6.8.1; EnergyPlus TSD Boiler:HotWater |
| `heat_pump_mode_transition.csv` | Air-source heat pump | Linear COP(T_odb) + balance-temp switchover | AHRI 210/240; ISO 13256-2 |

## Why analytical, not EnergyPlus?

1. **Issue #1933 forbids fabrication.** The issue asks for "EnergyPlus CSV
   outputs for missing scenarios" but the issue comment (roomote-fluxion)
   explicitly asks whether published curves may substitute where E+ runs
   are unavailable. We cannot run EnergyPlus in this environment, so
   inventing E+ output would violate `RULES.md` ("must-never hardcode
   results").
2. **The curves are authoritative engineering references.** AHRI and
   ASHRAE publish these correlation forms and typical coefficients as the
   canonical models that EnergyPlus itself uses internally for its
   reference equipment. Deriving the tables from the same published
   coefficients is therefore a faithful, reproducible substitute.
3. **Clear labelling.** Every CSV header carries a `Status: ANALYTICAL`
   line and a full citation, so downstream tests know these are not E+
   outputs and the data can be replaced with direct E+ runs later without
   ambiguity.

## Coefficient provenance

### Fan affinity (`fan_affinity_laws.csv`)
The three affinity laws (flow ∝ N, pressure ∝ N², power ∝ N³) are
dimensional invariants for geometrically similar fans and appear in every
reference (ASHRAE Fundamentals 2021 Ch. 21.4; ASHRAE 90.1-2022 §6.5.3.1
"Fan Motor Efficiency"). The variable-inlet-vane (constant-speed) power
curve `0.13 + 0.35·N + 0.52·N²` is the typical quadratic published for
VIV-controlled fans (less efficient than VSD at part load).

### Chiller CAPFT (`chiller_capacity_capft.csv`)
The biquadratic form

```
CAPFT(T_evap, T_cond) = c0 + c1·Te + c2·Te² + c3·Tc + c4·Tc² + c5·Te·Tc
```

is the AHRI 550/590 standard capacity-correction curve used by
EnergyPlus, DOE-2, and TRNSYS for water-cooled chillers. The coefficients
`[0.958, 0.0179, -0.00037, -0.0010, -0.000007, 0.00021]` are the typical
water-cooled centrifugal chiller values tabulated in the EnergyPlus
Technical Support Document (commercial reference building set). The rated
point is AHRI 550/590 standard conditions: `T_evap = 6.67 °C` (44 °F
leaving chilled-water temperature) and `T_cond = 29.44 °C` (85 °F
entering condenser-water temperature).

### Boiler efficiency (`boiler_part_load_efficiency.csv`)
EnergyPlus `Boiler:HotWater` uses a Normalized Boiler Efficiency Curve of
the form `η_norm(PLR) = c0 + c1·PLR + c2·PLR²`, applied as
`η_actual = η_rated · η_norm`. The coefficients `[1.0229, 0.0256, -0.0458]`
are the non-condensing hot-water reference values from the EnergyPlus TSD;
the rated thermal efficiency `0.80` is the ASHRAE 90.1-2022 Table 6.8.1
minimum for non-condensing hot-water boilers. The curve peaks slightly
above unity at part load (≈ 1.0265 at PLR = 0.3) — the expected
behaviour for a non-condensing boiler with reduced flue losses at
moderate part load.

### Heat-pump mode transition (`heat_pump_mode_transition.csv`)
The linear model `COP(T_odb) = 2.0 + 0.07·T_odb` is anchored at the AHRI
210/240 heating rating point (8.33 °C / 47 °F dry-bulb, COP ≈ 2.58). The
balance temperature (18 °C) is the thermostat deadband centre above
which heating demand is satisfied passively and the unit switches to
cooling. Real ASHP performance is non-linear at low source temperatures
(crankcase-heater / defrost penalties); this linear model is the
first-order approximation used for sizing and mode-transition checks and
is clearly labelled as such.

## Replacement path

When an EnergyPlus ≥ 25.2 environment is available, these analytical
CSVs should be augmented (not deleted) with direct E+ hourly outputs for
the same equipment operating across the same envelope of conditions, so
that tests can cross-validate the analytical correlation against the
full simulation. Add the E+ outputs alongside these files with an
`_eplus` suffix and a matching provenance header.
