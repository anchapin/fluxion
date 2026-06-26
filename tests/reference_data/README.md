# EnergyPlus Reference Data

This directory contains isolated reference data generated from EnergyPlus simulations
for bottom-up validation of individual Fluxion physics modules.

## Data Generation

Run the generation script from the repository root:

```bash
python tests/reference_data/generate_reference_data.py
```

Prerequisites:
- EnergyPlus 25.2.0 on PATH
- EPW: `USA_CO_Golden-NREL.724666_TMY3.epw` (bundled with E+)

## Generated Files

### Solar (`solar/`)

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `solar_position_denver.csv` | Hourly solar position for Denver TMY3 | 8760 | hour, altitude, azimuth, zenith |
| `surface_irradiance_south.csv` | Beam, diffuse, ground-reflected on south vertical wall | 8760 | hour, beam, diffuse, ground_reflected |

### Conduction (`conduction/`)

All conduction CSVs are generated from EnergyPlus 25.2.0 using the step-change
protocol (Jan 1-3, 15-min timesteps, free-floating zone, single test surface exposed
to outdoor weather).

| File | Description | Rows | Source |
|------|-------------|------|--------|
| `step_response_200mm_concrete.csv` | 200mm concrete south wall | ~288 | EnergyPlus |
| `step_response_composite.csv` | Composite wall (concrete + insulation + gypsum) south wall | ~288 | EnergyPlus |
| `step_response_floor.csv` | Floor slab on grade (carpet + concrete + insulation) | ~288 | EnergyPlus |
| `step_response_lightweight.csv` | Lightweight steel stud wall south wall | ~288 | EnergyPlus |
| `step_response_roof.csv` | Roof assembly (gravel + insulation + steel deck) | ~288 | EnergyPlus |
| `step_response_fixed_zone_20c.csv` | Fixed zone temperature (ASHRAE 140) | — | EnergyPlus |

### Ventilation (`ventilation/`)

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `infiltration_denver.csv` | Hourly outdoor temp, wind, infiltration ACH, vent conductance | 8760 | hour, T_out, wind_speed, ACH, C_vent |

### EnergyPlus Models (`energyplus_models/`)

| File | Description |
|------|-------------|
| `annual_solar_ventilation.idf` | Single-zone box (6×8×2.7m), lightweight walls, no HVAC, 0.5 ACH |
| `step_change_concrete.idf` | 200mm concrete south wall, free-floating, Jan 1-3 weather-driven |
| `step_change_composite.idf` | Composite wall (concrete + insulation + gypsum), south wall, Jan 1-3 |
| `step_change_floor.idf` | Floor slab on grade, Jan 1-3 |
| `step_change_lightweight.idf` | Lightweight steel stud wall, south wall, Jan 1-3 |
| `step_change_roof.idf` | Roof assembly, Jan 1-3 |

## Model Parameters

### Model 1: Annual Solar + Ventilation
- **Geometry**: 6m × 8m × 2.7m single zone
- **Volume**: 129.6 m³
- **Construction**: Lightweight (steel stud + 50mm insulation + gypsum)
- **Infiltration**: 0.5 ACH constant
- **HVAC**: None (free-floating)
- **Weather**: USA_CO_Golden-NREL TMY3 (39.74°N, 105.18°W)

### Models 2-6: Conduction Step-Change Tests
- **Geometry**: 6m × 8m × 2.7m single zone
- **Timestep**: 15 minutes (4 per hour)
- **Run period**: 72 hours (Jan 1-3)
- **HVAC**: None (free-floating)
- **Non-test surfaces**: Highly insulated (R-20, k=0.01 W/(m·K))
- **Ground temperature**: 18°C constant

| Model | Test Surface | Construction |
|-------|-------------|--------------|
| 2: 200mm Concrete | South wall | 200mm concrete (k=1.73, ρ=2300, cp=840) |
| 3: Composite | South wall | 100mm concrete + 100mm mineral wool + 13mm gypsum |
| 4: Floor Slab | Floor (slab on grade) | 10mm carpet + 150mm concrete + 100mm insulation |
| 5: Lightweight | South wall | 16mm ext gyp + 12mm OSB + 90mm cavity insulation + 13mm int gyp |
| 6: Roof | Roof | 50mm gravel + 150mm insulation + 1.5mm steel deck |

## CSV Format

### Conduction CSV columns

```
hour, T_outdoor, T_zone, T_surface_inside, T_surface_outside, q_inside_Wm2, q_outside_Wm2
```

- `hour`: elapsed hours from start (0 to 72)
- `T_outdoor`: outdoor air drybulb temperature (°C)
- `T_zone`: zone mean air temperature (°C)
- `T_surface_inside`: inside face temperature of test surface (°C)
- `T_surface_outside`: outside face temperature of test surface (°C)
- `q_inside_Wm2`: inside face conduction heat flux (W/m²)
- `q_outside_Wm2`: outside face conduction heat flux (W/m²)

## Ventilation Conductance Calculation

From ASHRAE Fundamentals, the ventilation conductance is:

```
C_vent = ACH × V × ρ × c_p / 3600  [W/K]
```

Where:
- ACH = infiltration air changes per hour [1/h]
- V = zone volume [m³]
- ρ = air density ≈ 1.2 kg/m³ (at standard conditions)
- c_p = specific heat of air = 1000 J/(kg·K)

For this model: C_vent = 0.5 × 129.6 × 1.2 × 1000 / 3600 = **21.6 W/K**
