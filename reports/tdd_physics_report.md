# Fluxion Physics TDD Report

Generated: 1774980168

## Overall Summary

| Metric | Value |
|--------|-------|
| Total Tests | 23 |
| Passed | 23 |
| Failed | 0 |
| Skipped | 0 |
| Pass Rate | 100.0% |

## Per-Domain Summary

| Domain | Total | Passed | Failed | Skipped | Pass Rate | Max Error |
|--------|-------|--------|--------|---------|-----------|-----------|
| HeatConduction | 3 | 3 | 0 | 0 | 100.0% | 0.00% |
| SolarRadiation | 3 | 3 | 0 | 0 | 100.0% | 0.00% |
| ThermalMass | 3 | 3 | 0 | 0 | 100.0% | 0.00% |
| HVACLoads | 2 | 2 | 0 | 0 | 100.0% | 0.00% |
| AirExchange | 2 | 2 | 0 | 0 | 100.0% | 0.00% |
| InterZoneTransfer | 2 | 2 | 0 | 0 | 100.0% | 0.00% |
| GroundCoupling | 2 | 2 | 0 | 0 | 100.0% | 0.00% |
| InternalGains | 2 | 2 | 0 | 0 | 100.0% | 0.00% |
| WindowHeatTransfer | 2 | 2 | 0 | 0 | 100.0% | 0.00% |
| LongwaveRadiation | 2 | 2 | 0 | 0 | 100.0% | 0.00% |

## HeatConduction

**Description:** Heat conduction through building envelope

Execution time: 0 ms


### Passed Tests (3)

- **Steady-state wall conduction**: 100.0000 W (ref: 100.0000 W, error: 0.00%)
- **Multi-layer wall U-value**: 0.6217 W/m²K (ref: 0.6217 W/m²K, error: 0.00%)
- **Thermal bridge linear conductance**: 1.5000 W/K (ref: 1.5000 W/K, error: 0.00%)

## SolarRadiation

**Description:** Solar radiation absorption and transmission

Execution time: 0 ms


### Passed Tests (3)

- **Solar constant**: 1361.0000 W/m² (ref: 1361.0000 W/m², error: 0.00%)
- **Solar altitude at noon (summer solstice, 40°N)**: 73.4500 degrees (ref: 73.4500 degrees, error: 0.00%)
- **Clear sky direct normal irradiance**: 906.9634 W/m² (ref: 906.9634 W/m², error: 0.00%)

## ThermalMass

**Description:** Thermal mass storage and release effects

Execution time: 0 ms


### Passed Tests (3)

- **Concrete wall thermal capacitance**: 3036000.0000 J/K (ref: 3036000.0000 J/K, error: 0.00%)
- **Thermal time constant**: 600000.0000 s (ref: 600000.0000 s, error: 0.00%)
- **ISO 13790 heavy mass classification**: 1.0000 boolean (ref: 1.0000 boolean, error: 0.00%)

## HVACLoads

**Description:** Heating and cooling load calculations

Execution time: 0 ms


### Passed Tests (2)

- **Sensible cooling load**: 2000.0000 W (ref: 2000.0000 W, error: 0.00%)
- **Latent cooling load**: 7350.0000 W (ref: 7350.0000 W, error: 0.00%)

## AirExchange

**Description:** Infiltration and ventilation heat transfer

Execution time: 0 ms


### Passed Tests (2)

- **Infiltration heat loss**: 837.5000 W (ref: 837.5000 W, error: 0.00%)
- **Stack effect pressure difference**: 1.4715 Pa (ref: 1.4715 Pa, error: 0.00%)

## InterZoneTransfer

**Description:** Heat transfer between thermal zones

Execution time: 0 ms


### Passed Tests (2)

- **Inter-zone conductive heat transfer**: 80.0000 W (ref: 80.0000 W, error: 0.00%)
- **Radiative heat transfer between surfaces**: 488.5086 W (ref: 488.5086 W, error: 0.00%)

## GroundCoupling

**Description:** Ground heat transfer and slab losses

Execution time: 0 ms


### Passed Tests (2)

- **Slab-on-grade perimeter heat loss**: 1200.0000 W (ref: 1200.0000 W, error: 0.00%)
- **Ground temperature amplitude damping**: 7.4390 °C (ref: 7.4390 °C, error: 0.00%)

## InternalGains

**Description:** Internal heat from occupants and equipment

Execution time: 0 ms


### Passed Tests (2)

- **Occupant sensible heat gain**: 600.0000 W (ref: 600.0000 W, error: 0.00%)
- **Lighting heat gain**: 400.0000 W (ref: 400.0000 W, error: 0.00%)

## WindowHeatTransfer

**Description:** Window conduction and solar gain

Execution time: 0 ms


### Passed Tests (2)

- **Window conduction heat loss**: 250.0000 W (ref: 250.0000 W, error: 0.00%)
- **Window solar heat gain**: 1500.0000 W (ref: 1500.0000 W, error: 0.00%)

## LongwaveRadiation

**Description:** Longwave radiation exchange between surfaces

Execution time: 0 ms


### Passed Tests (2)

- **Blackbody emissive power at 20°C**: 418.7383 W/m² (ref: 418.7383 W/m², error: 0.00%)
- **View factor reciprocity**: 0.1500 dimensionless (ref: 0.1500 dimensionless, error: 0.00%)
