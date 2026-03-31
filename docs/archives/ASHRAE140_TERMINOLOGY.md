# ASHRAE 140 Terminology & Reference Guide

**Purpose**: Provide a development reference to clarify key concepts, terminology, and conventions used in ASHRAE Standard 140 to prevent misunderstandings during implementation and validation.

**Related Issues**: [#243](https://github.com/anchapin/fluxion/issues/243), [#235](https://github.com/anchapin/fluxion/issues/235)

---

## Table of Contents

1. [Key Terminology](#key-terminology)
2. [Output Metrics](#output-metrics)
3. [Units and Conversions](#units-and-conversions)
4. [Case Categories](#case-categories)
5. [Validation Methodology](#validation-methodology)
6. [Common Pitfalls](#common-pitfalls)
7. [Reference Value Interpretation](#reference-value-interpretation)

---

## Key Terminology

### Thermal Load vs. Energy

**Thermal Load (Heating Load / Cooling Load)**:
- The instantaneous heat energy rate required to maintain setpoint temperatures
- Measured in kW (kilowatts) for peak values, or kWh (kilowatt-hours) for integrated values
- This is what the building *needs* to maintain temperature
- **Fluxion calculates this directly** from zone temperature differences and setpoint deviations

**HVAC System Energy (Electricity Consumption)**:
- The energy consumed by the HVAC equipment to deliver the thermal load
- Includes equipment efficiency, fan power, compressor work, etc.
- Measured in kWh (kilowatt-hours) or MWh (megawatt-hours)
- **ASHRAE 140 reference values typically represent this**

### Critical Distinction
In Fluxion's validation output:
- **Heating/Cooling values reported are THERMAL LOADS** (heat removed/added)
- **ASHRAE 140 reference values may be HVAC system energy** (electricity consumed)
- These are NOT directly comparable without accounting for system efficiency

### Example
- Building needs to remove 10 kWh of cooling thermal load (calculated by Fluxion)
- With a COP of 3.0, HVAC system consumes 3.33 kWh of electricity (ASHRAE reference)
- **Without efficiency factors, Fluxion will report ~3x higher cooling energy than reference**

---

## Output Metrics

### Annual Heating Energy
- **Definition**: Total thermal heating load required over the full year
- **Units**: MWh (megawatt-hours) per ASHRAE 140 conventions
- **Calculation**: Sum of all positive deviations from heating setpoint × mass × specific heat / efficiency
- **Fluxion Output**: Direct thermal load integration
- **ASHRAE Reference**: Often includes system efficiency (COP) adjustments

### Annual Cooling Energy
- **Definition**: Total thermal cooling load required over the full year
- **Units**: MWh (megawatt-hours) per ASHRAE 140 conventions
- **Calculation**: Sum of all negative deviations from cooling setpoint × mass × specific heat
- **Fluxion Output**: Direct thermal load (sensible cooling only)
- **ASHRAE Reference**: Often adjusted for system efficiency and latent cooling

### Peak Heating Load
- **Definition**: Maximum instantaneous heating power demand at any single timestep
- **Units**: kW (kilowatts)
- **Calculation**: Maximum value of hourly heating load
- **Timing**: Occurs during winter, typically during morning warm-up
- **ASHRAE 140 Context**: Tests sizing of HVAC equipment

### Peak Cooling Load
- **Definition**: Maximum instantaneous cooling power demand at any single timestep
- **Units**: kW (kilowatts)
- **Calculation**: Maximum value of hourly cooling load
- **Timing**: Occurs during summer, typically afternoon peak
- **ASHRAE 140 Context**: Tests seasonal variation and solar gain impacts

---

## Units and Conversions

### Standard ASHRAE 140 Units

| Metric | Unit | Notes |
|--------|------|-------|
| Annual Energy | MWh | Megawatt-hours (1 MWh = 1000 kWh) |
| Peak Load | kW | Kilowatts (instantaneous power) |
| Temperature | °C | Celsius |
| Area | m² | Square meters |
| Time | hour | Hour (simulation timestep = 1 hour) |
| Radiation | W/m² | Watts per square meter (solar intensity) |

### Common Conversion Errors to Avoid

1. **kWh vs Watts**
   - NOT interchangeable
   - kWh = Watts × Hours
   - Example: 10 kW load for 1 hour = 10 kWh energy

2. **MWh vs kWh**
   - 1 MWh = 1,000 kWh
   - Use MWh for annual values (cleaner numbers)

3. **Thermal Load vs Electricity**
   - Thermal Load (kW) ≠ Electricity (kW)
   - Relationship depends on HVAC efficiency (COP for cooling, resistance or furnace efficiency for heating)

4. **Per-Unit Area**
   - Reference values sometimes given as MWh/m² or kW/m²
   - Convert to building totals using gross floor area

---

## Case Categories

### Free-Floating Cases (FF suffix)
- **Definition**: HVAC systems disabled; zone temperature floats without control
- **Purpose**: Tests thermal response characteristics without control interference
- **Expected Values**:
  - Annual Heating: 0 MWh (no heating system)
  - Annual Cooling: 0 MWh (no cooling system)
  - Min Temperature: Typically negative (winter conditions)
  - Max Temperature: Typically > 35°C (summer conditions)
- **Validation Metric**: Min/max zone temperatures vs reference ranges
- **Cases**: 600FF, 650FF, 900FF, 950FF

### Controlled Cases (No Suffix)
- **Definition**: HVAC systems active with setpoint control
- **Heating Setpoint**: 20°C (default, 10°C during setback)
- **Cooling Setpoint**: 27°C (default)
- **Purpose**: Tests HVAC control logic and efficiency
- **Expected Values**: Annual energy in range specified per case
- **Cases**: 600, 610, 620, 630, 640, 650, 900, 910, 920, 930, 940, 950

### Special Cases
- **Case 195**: Pure conduction test (solid wall, no windows)
- **Case 960**: Multi-zone coupling test (sunspace with zone-to-zone heat transfer)

---

## Validation Methodology

### Reference Range Interpretation

ASHRAE 140 provides reference values from multiple programs (EnergyPlus, ESP-r, TRNSYS, etc.):

```
Reference Format: [Min, Max] MWh
Example: Case 600 Heating: [4.30, 5.71] MWh
```

**Pass Criteria**:
- Fluxion result falls within [Min, Max] range, OR
- Fluxion result within 5% of range midpoint, OR
- Implementation-specific criteria (Fluxion uses 25% tolerance during development)

### Example Validation

```
Case 600 - Annual Heating

Fluxion Output:        13.24 MWh
Reference Min:         4.30 MWh
Reference Max:         5.71 MWh
Midpoint:             5.01 MWh
Tolerance (±5%):      4.76 - 5.26 MWh

Status: FAIL (13.24 >> 5.71)
Variance: 2.64x reference maximum
```

### Debugging Process

When validation fails:

1. **Check units**: Ensure outputs are in MWh, not kWh
2. **Check control logic**: Verify setpoint heating/cooling is active
3. **Check efficiency assumptions**: Are you comparing loads to system energy?
4. **Check physics**: Are zone temperatures tracking reasonably?
5. **Check timesteps**: 1-hour timesteps per ASHRAE 140 spec

---

## Common Pitfalls

### Pitfall 1: Confusing Thermal Load with HVAC Energy
**Symptom**: Fluxion reports 3x higher heating/cooling than reference
**Cause**: Comparing thermal load against system electricity with efficiency factor
**Fix**: Either:
- Apply system efficiency to Fluxion outputs (divide by COP)
- Confirm reference values are thermal loads, not system energy
- Check ASHRAE 140 documentation for case-specific convention

**Example from Issue #235**:
> Cooling load (heat removed) was being compared directly against HVAC electricity consumption. These are not equivalent without efficiency factors.

### Pitfall 2: Wrong Timestep or Duration
**Symptom**: Peak loads are suspiciously low or high
**Cause**: Using wrong timestep duration (should be 1 hour) or partial year simulation
**Fix**: Verify simulation runs full year (8760 hours) with 1-hour timesteps

### Pitfall 3: Unit Inconsistencies
**Symptom**: Values are off by factor of 1000
**Cause**: MWh vs kWh confusion, or Watts vs Joules
**Fix**: Double-check unit conversions and be explicit in code (use type system when possible)

### Pitfall 4: HVAC Control Logic
**Symptom**: Free-floating cases show non-zero heating/cooling
**Cause**: HVAC not actually disabled or setpoint logic still active
**Fix**: Verify `free_floating` flag disables all HVAC energy calculation

### Pitfall 5: Setpoint Misinterpretation
**Symptom**: Thermostat setback cases don't show expected reduction
**Cause**: Schedule not implemented or setback temperature misread
**Fix**: Check `HvacSchedule` implementation and confirm setback setpoint is 10°C (not 20°C)

---

## Reference Value Interpretation

### Where Reference Values Come From

ASHRAE 140 publishes ranges from programs validated against the standard:
- **EnergyPlus**: DOE/NREL standard reference
- **ESP-r**: University of Strathclyde (UK)
- **TRNSYS**: University of Wisconsin
- **Other**: Additional programs validated per standard

These represent the **range of acceptable results** accounting for different modeling approaches and assumptions.

### What the Range Tells Us

**Narrow Range** (e.g., [5.0, 5.2]):
- Cases are well-defined, physics is straightforward
- All validated programs converge to similar results
- Fluxion should be very close to this range

**Wide Range** (e.g., [1.5, 3.5]):
- Case has complex physics or ambiguities
- Validated programs produce different results
- Fluxion within range is acceptable; outside requires investigation

### Example Interpretation

```
Case 600 (Low-Mass Baseline):
- Heating: [4.30, 5.71] MWh — Narrow range, straightforward case
- Cooling: [11.5, 17.8] MWh — Wider range, solar gains have more variance

Case 960 (Multi-Zone Sunspace):
- Heating: [1.65, 2.45] MWh — Complex inter-zone effects
- Cooling: [1.55, 2.78] MWh — Coupled thermal behavior

Fluxion Case 960 Results:
- Heating: 28.67 MWh (17x too high) — Major physics issue
- Cooling: 36.25 MWh (13x too high) — Likely same root cause
```

---

## Development Workflow

### When Adding a New Case

1. **Read the case specification** in ASHRAE 140 standard or Fluxion's `ashrae_140_cases.rs`
2. **Identify the category**:
   - Free-floating? → No HVAC control
   - Controlled with setback? → Check schedule implementation
   - Multi-zone? → Check inter-zone coupling
3. **Extract reference values** → Add to `benchmark.rs`
4. **Implement case configuration** → Geometry, materials, HVAC setpoints, schedules
5. **Run simulation** → Validate outputs
6. **Compare to reference**:
   - Peak loads: Should be within range
   - Annual energy: Should be within range or within 5% tolerance
7. **Debug if needed** → Use logging to identify physics issues

### When Debugging a Failing Case

1. **Check units first**: MWh not kWh?
2. **Verify control logic**: HVAC active/disabled as expected?
3. **Compare zone temperatures**: Are they reasonable (no -50°C or 80°C)?
4. **Check physics parameters**: Insulation, thermal mass, infiltration
5. **Add logging**: Hourly/daily values to see where divergence occurs
6. **Compare against EnergyPlus**: Run same case in reference program
7. **Document findings**: Add to case-specific issue or analysis file

---

## References

- **ASHRAE Standard 140-2023**: Standard Method of Test for the Evaluation of Building Energy Analysis Computer Programs
- **ASHRAE 140 User Manual**: `docs/140UsersManual-PartI-Final (050825).pdf`
- **Fluxion Implementation**: `src/validation/ashrae_140_validator.rs`
- **Test Cases**: `src/validation/ashrae_140_cases.rs`
- **Benchmarks**: `src/validation/benchmark.rs`

---

## Version History

| Version | Date | Author | Notes |
|---------|------|--------|-------|
| 1.0 | 2026-02-17 | Amp | Initial document created (Issue #243) |

---

**Last Updated**: February 17, 2026
