# Section 9 Standard Output Report -- conformance table

ASHRAE Standard 140 Section 9 (Space-Cooling Equipment Performance) case set CE100-CE200, reported in the Annex A2 Standard Output Report shape: one row per case, one column per reported value, and a per-requirement conformance state.

Case states: `conformant` (ran with the specified inputs and produced every required output), `incomplete` (ran, but at least one required output is missing or only partial), `not implemented` (did not run). Requirement cell states: `report` (required and produced), `partial` (required and produced for a subset), `missing` (required and not produced), `na` (not required for this case).

**No aggregate grade, rate or pass/fail verdict appears in this report.** Standard 140 is a method of test, not a rating or compliance threshold: it defines no such quantity, and each case is classified individually as conformant, incomplete or not implemented.

## How the required-output list was derived

The 140-2023 Section 9 output requirements live in the licensed normative materials (Issue #4285), so this repository carries no explicit Section 9 required-output list. The required-output list used here is therefore DERIVED from the two committed Section 9 source artefacts -- the case definitions (data/reference/ashrae140/section9_ce_cases.json) and the performance map (data/reference/ashrae140/section9_ce100_performance_map.json) -- by the following rule. (1) INPUTS: every case definition names the case's input conditions (internal sensible gains, internal latent gains, cooling setpoint, outdoor dry-bulb temperature) and the weather file the case binds, so the case inputs are required outputs of the case record. (2) ENTERING CONDITIONS: the performance map tabulates the entering dry-bulb and entering wet-bulb columns of the rating tables, so the entering conditions are required outputs. (3) RATED AND DISPATCHED PERFORMANCE: the performance map tabulates the rated total capacity, sensible capacity, latent capacity and compressor power at each (outdoor dry-bulb, entering wet-bulb, entering dry-bulb) point, and the steady-state dispatch of a case reports the delivered capacity and power, the coefficient of performance, the part load ratio, the sensible heat ratio, the peak cooling power, the required cooling load and the capacity shortfall at that point, so all of them are required outputs. (4) ANNUAL CASE OUTPUTS: every case definition binds an annual TMY2 weather file and the case table names the annual_cooling metric (MWh) for the case, and for CE150 also the max_zone_temp metric (degC), so those are required outputs as well. The derived requirement ids below are this report's own labels -- they are NOT clause numbers of the standard, which are not transcribed in this repository.

**The reference columns are diagnostic context and analytical reference, not criteria.** The 'Analytical reference' columns are the standard's own rating tables (Tables 9-5 through 9-10 of 140-2023) at the case condition; the engine evaluates the same map, so the reference and the reported value coincide. The 'Comparative evidence range' columns are the IEA SHC Task 22 ensemble min/max ranges carried verbatim from the case table; the case table labels them 'evidence range (ensemble min/max), not a pass/fail criterion'. Neither role is used to derive a verdict here, and no conclusion is drawn from either.

## Reported values and conformance

| Case | Title | Status | 9a.case_inputs | 9a.entering_dry_bulb | 9a.entering_wet_bulb | 9a.total_capacity | 9a.sensible_capacity | 9a.latent_capacity | 9a.compressor_power | 9a.delivered_capacity | 9a.delivered_power | 9a.cop | 9a.plr | 9a.shr | 9a.peak_cooling_power | 9a.required_load | 9a.capacity_shortfall | 9a.annual_cooling | 9a.max_zone_temp | Analytical reference total capacity at the case condition (kW) | Analytical reference sensible capacity at the case condition (kW) | Analytical reference compressor power at the case condition (kW) | Analytical reference apparatus dew point at the case condition (degC) | Comparative evidence range, peak cooling (kW) | Comparative evidence range, annual cooling (MWh) | Comparative evidence range, maximum zone temperature (degC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CE100 | Base-Case Building and Mechanical System | incomplete | report | report | partial | report | report | report | report | report | report | report | report | report | report | report | report | missing | na | 6.02 to 7.48 | 2.41 to 5.81 | 2.11 to 2.21 | 10 to 16.9 | 2.7 to 3.2 | 4.1 to 4.9 |  |
| CE110 | Reduced Outdoor Dry-Bulb Temperature | incomplete | report | report | partial | report | report | report | report | report | report | report | report | report | report | report | report | missing | na | 7.19 to 9.06 | 2.97 to 6.31 | 1.62 to 1.76 | 8.9 to 15.8 | 2.65 to 3.15 | 4 to 4.8 |  |
| CE120 | Increased Thermostat Setpoint | incomplete | report | report | partial | report | report | report | report | report | report | report | report | report | report | report | report | missing | na | 7.19 to 9.06 | 5.14 to 7.66 | 1.62 to 1.76 | 8.9 to 15.8 | 2.8 to 3.3 | 4.3 to 5.15 |  |
| CE130 | Low Part-Load Ratio | incomplete | report | report | partial | report | report | report | report | report | report | report | report | report | report | report | report | missing | na | 6.02 to 7.48 | 2.41 to 5.81 | 2.11 to 2.21 | 10 to 16.9 | 2.6 to 3.1 | 3.95 to 4.7 |  |
| CE140 | Reduced Outdoor Dry-Bulb Temperature at Low Part-Load Ratio | incomplete | report | report | partial | report | report | report | report | report | report | report | report | report | report | report | report | missing | na | 7.19 to 9.06 | 2.97 to 6.31 | 1.62 to 1.76 | 8.9 to 15.8 | 2.75 to 3.25 | 4.2 to 5 |  |
| CE150 | Latent Load at High Sensible Heat Ratio | incomplete | report | report | partial | report | report | report | report | report | report | report | report | report | report | report | report | missing | na | 7.19 to 9.06 | 2.97 to 6.31 | 1.62 to 1.76 | 8.9 to 15.8 | 3.1 to 3.7 | 5.4 to 6.45 | 25.5 to 27.5 |
| CE160 | Increased Thermostat Setpoint at High Sensible Heat Ratio | incomplete | report | report | partial | report | report | report | report | report | report | report | report | report | report | report | report | missing | na | 7.19 to 9.06 | 5.14 to 7.66 | 1.62 to 1.76 | 8.9 to 15.8 | 3.05 to 3.65 | 5.3 to 6.3 |  |
| CE165 | Variation of Thermostat Setpoint and Outdoor Dry-Bulb Temperature at High Sensible Heat Ratio | incomplete | report | report | partial | report | report | report | report | report | report | report | report | report | report | report | report | missing | na | 6.43 to 8.01 | 3.12 to 6.49 | 1.94 to 2.06 | 9.6 to 16.6 |  |  |  |
| CE170 | Reduced Sensible Load | incomplete | report | report | partial | report | report | report | report | report | report | report | report | report | report | report | report | missing | na | 7.19 to 9.06 | 2.97 to 6.31 | 1.62 to 1.76 | 8.9 to 15.8 | 3.15 to 3.75 | 5.55 to 6.6 |  |
| CE180 | Increased Latent Load | incomplete | report | report | partial | report | report | report | report | report | report | report | report | report | report | report | report | missing | na | 7.19 to 9.06 | 2.97 to 6.31 | 1.62 to 1.76 | 8.9 to 15.8 |  |  |  |
| CE185 | Increased Outdoor Dry-Bulb Temperature at Low Sensible Heat Ratio | incomplete | report | report | partial | report | report | report | report | report | report | report | report | report | report | report | report | missing | na | 6.02 to 7.48 | 2.41 to 5.81 | 2.11 to 2.21 | 10 to 16.9 |  |  |  |
| CE190 | Low Part-Load Ratio at Low Sensible Heat Ratio | incomplete | report | report | partial | report | report | report | report | report | report | report | report | report | report | report | report | missing | na | 7.19 to 9.06 | 2.97 to 6.31 | 1.62 to 1.76 | 8.9 to 15.8 |  |  |  |
| CE195 | Increased Outdoor Dry-Bulb Temperature at Low Sensible Heat Ratio and Low Part-Load Ratio | incomplete | report | report | partial | report | report | report | report | report | report | report | report | report | report | report | report | missing | na | 6.02 to 7.48 | 2.41 to 5.81 | 2.11 to 2.21 | 10 to 16.9 |  |  |  |
| CE200 | Full-Load Test at AHRI Conditions | incomplete | report | report | partial | report | report | report | report | report | report | report | report | report | report | report | report | missing | na | 6.81 to 8.57 | 4.96 to 7.34 | 1.77 to 1.91 | 9.3 to 16.2 | 3.2 to 3.8 | 5.6 to 6.7 |  |

## Coverage bookkeeping

The table below counts, per requirement, how many cases sit in each cell state. These are counts of cases, not a rate; no aggregate quantity is derived from them.

| Requirement | report | partial | missing | na |
|---|---|---|---|---|
| 9a.case_inputs | 14 | 0 | 0 | 0 |
| 9a.entering_dry_bulb | 14 | 0 | 0 | 0 |
| 9a.entering_wet_bulb | 0 | 14 | 0 | 0 |
| 9a.total_capacity | 14 | 0 | 0 | 0 |
| 9a.sensible_capacity | 14 | 0 | 0 | 0 |
| 9a.latent_capacity | 14 | 0 | 0 | 0 |
| 9a.compressor_power | 14 | 0 | 0 | 0 |
| 9a.delivered_capacity | 14 | 0 | 0 | 0 |
| 9a.delivered_power | 14 | 0 | 0 | 0 |
| 9a.cop | 14 | 0 | 0 | 0 |
| 9a.plr | 14 | 0 | 0 | 0 |
| 9a.shr | 14 | 0 | 0 | 0 |
| 9a.peak_cooling_power | 14 | 0 | 0 | 0 |
| 9a.required_load | 14 | 0 | 0 | 0 |
| 9a.capacity_shortfall | 14 | 0 | 0 | 0 |
| 9a.annual_cooling | 0 | 0 | 14 | 0 |
| 9a.max_zone_temp | 0 | 0 | 0 | 14 |

Case states: conformant 0, incomplete 14, not implemented 0. Every Section 9 case is dispatched by the engine harness, so no case is 'not implemented'; every case is 'incomplete' for the same documented reason -- the annual case outputs are not produced by this tree and the per-case entering wet-bulb is only partially transcribed. Every steady-state dispatch output is reported for every case.

## Notes

- Case states are conformant / incomplete / not implemented. No aggregate grade, rate or pass/fail verdict is reported anywhere in these artifacts: the standard defines no such quantity.
- Requirement cell states are report (required, produced), partial (required, produced for a subset), missing (required, not produced) and na (not required for this case).
- The 140-2023 Section 9 output requirements live in the licensed normative materials (Issue #4285), so this repository carries no explicit Section 9 required-output list. The required-output list used here is therefore DERIVED from the two committed Section 9 source artefacts -- the case definitions (data/reference/ashrae140/section9_ce_cases.json) and the performance map (data/reference/ashrae140/section9_ce100_performance_map.json) -- by the following rule. (1) INPUTS: every case definition names the case's input conditions (internal sensible gains, internal latent gains, cooling setpoint, outdoor dry-bulb temperature) and the weather file the case binds, so the case inputs are required outputs of the case record. (2) ENTERING CONDITIONS: the performance map tabulates the entering dry-bulb and entering wet-bulb columns of the rating tables, so the entering conditions are required outputs. (3) RATED AND DISPATCHED PERFORMANCE: the performance map tabulates the rated total capacity, sensible capacity, latent capacity and compressor power at each (outdoor dry-bulb, entering wet-bulb, entering dry-bulb) point, and the steady-state dispatch of a case reports the delivered capacity and power, the coefficient of performance, the part load ratio, the sensible heat ratio, the peak cooling power, the required cooling load and the capacity shortfall at that point, so all of them are required outputs. (4) ANNUAL CASE OUTPUTS: every case definition binds an annual TMY2 weather file and the case table names the annual_cooling metric (MWh) for the case, and for CE150 also the max_zone_temp metric (degC), so those are required outputs as well. The derived requirement ids below are this report's own labels -- they are NOT clause numbers of the standard, which are not transcribed in this repository.
- The per-case entering wet-bulb is not transcribed for any CE case: the 140-2023 Section 9 input specification is licensed (Issue #4285). The engine harness therefore dispatches every case at every entering wet-bulb column the performance map tabulates (15.0 / 17.2 / 19.4 / 21.7 degC) and reports the envelope; the entering wet-bulb requirement is therefore 'partial' for every case (the envelope is reported, the case's own entering condition is not transcribed).
- The annual case outputs (annual cooling energy, maximum zone temperature) are not produced by this tree: the normative CE weather files (CE100A, CE110A, CE165A, CE200A) are licensed ASHRAE supplemental material that is never committed, and the steady-state equipment dispatch has no duty cycle and no zone-temperature trajectory. The affected requirement cells are 'missing' and the corresponding report cells are empty -- no value is invented.
- The performance-map columns are the analytical reference: Tables 9-5 through 9-10 of 140-2023 are the standard's own rating tables, and the engine evaluates the same map, so the reference value and the engine value coincide. They are shown next to the reported values as the analytical reference; no acceptance criterion is derived from them.
- The comparative columns are the IEA SHC Task 22 ensemble evidence ranges, transcribed per case from tests/validation/hvac_equipment_rated_performance/data/comparative_bounds_ce100_ce200.csv and carried in the case table. They are diagnostic context, not a criterion; the five cases the bounds file does not cover (CE165, CE180, CE185, CE190, CE195) carry no range and no range is invented for them.
- The engine-run results file records its own per-case band bookkeeping (band kind, band edges, direction, gap, band status). That bookkeeping is deliberately NOT carried into this report: this report classifies conformance of the reported outputs only, and carries the reference ranges themselves, without any verdict attached to them.
- Powers and capacities are reported in kW; the engine records them in W, so the values are divided by 1000 rather than re-derived. The capacity shortfall is reported in W, as the engine records it. Reported values carry six decimal places.
- A case that did not run carries no value at all in the flat CSV -- not even its input conditions -- so an empty cell can never be read as a produced value.
- The Section 9b comparative test series is not implemented in this tree and is not part of the CE100-CE200 catalog reported here.

## Sources

- `data/reference/ashrae140/section9_ce_cases.json` -- the generated Section 9 CE100-CE200 case table (case definitions, input conditions, weather-file binding, per-case reference bands).
- `data/reference/ashrae140/section9_ce100_performance_map.json` -- the CE100 performance map (Tables 9-5 through 9-10 of 140-2023), the analytical reference.
- `tests/validation/hvac_equipment_rated_performance/data/section9_engine_run_results.json` -- the recorded Section 9 engine-run results (the steady-state dispatch of every case at every entering wet-bulb column the map tabulates).
- The flat results CSV for this case set: `data/reference/section9/section9_standard_output_report.csv`.
