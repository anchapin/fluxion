# EnergyPlus Reference Data

This directory contains isolated reference data generated from EnergyPlus simulations for bottom-up validation of individual Fluxion physics modules.

## Data Generation Protocol

1. **Isolate**: Each CSV tests ONE physical phenomenon in isolation
2. **Simple geometry**: Single-zone box, no HVAC, known construction
3. **Denver TMY3**: Use USA_CO_Denver weather file (ASHRAE 140 standard)
4. **Hourly output**: 8760 rows (or subset for specific tests)
5. **Column names**: Must match Fluxion module output field names exactly

## Required Data Files

### Solar (`solar/`)
- `solar_position_denver.csv` — Hourly solar altitude, azimuth, zenith for Denver 2023
- `surface_irradiance_south.csv` — Beam, diffuse, ground-reflected on a vertical south surface

### Conduction (`conduction/`)
- `step_response_200mm_concrete.csv` — Inside surface heat flux for a step change in outdoor temp
- `annual_conduction_denver.csv` — Hourly conduction heat flux for a wall in Denver

### Ventilation (`ventilation/`)
- `infiltration_denver.csv` — Hourly infiltration ACH and ventilation heat loss

### Zone Balance (`zone_balance/`)
- `case_600_denver.csv` — Hourly zone temperatures for ASHRAE 140 Case 600 (low mass)
- `case_900_denver.csv` — Hourly zone temperatures for ASHRAE 140 Case 900 (high mass)

## EnergyPlus Models (`energyplus_models/`)
Source IDF files used to generate the reference data. These should be minimal single-zone models.

## Format

All CSVs use:
- UTF-8 encoding
- Header row with units in parentheses
- Comma delimiter
- Hour column is 1-8760 (ASHRAE standard hour numbering)
