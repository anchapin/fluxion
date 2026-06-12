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

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `step_response_200mm_concrete.csv` | Transient conduction through 200mm concrete wall | 288 | hour, T_ext, T_surf_in, T_surf_out, q_in, q_out |

### Ventilation (`ventilation/`)

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `infiltration_denver.csv` | Hourly outdoor temp, wind, infiltration ACH, vent conductance | 8760 | hour, T_out, wind_speed, ACH, C_vent |

### Zone Balance (`zone_balance/`)

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `fixed_inputs_zone_temp.csv` | Annual zone energy balance with fixed sub-module inputs | 8760 | hour, T_zone, T_out, Q_cond, Q_solar, Q_vent, Q_int, Q_heat, Q_cool |

### EnergyPlus Models (`energyplus_models/`)

| File | Description |
|------|-------------|
| `annual_solar_ventilation.idf` | Single-zone box (6×8×2.7m), lightweight walls, no HVAC, 0.5 ACH |
| `step_change_concrete.idf` | Single-zone, 200mm concrete south wall, free-floating, Jan 1-3 weather-driven |
| `fixed_inputs_zone_temp.idf` | Single-zone, ideal loads HVAC locked at 20°C, all surfaces NoSun/NoWind, annual run |

## Model Parameters

### Model 1: Annual Solar + Ventilation
- **Geometry**: 6m × 8m × 2.7m single zone
- **Volume**: 129.6 m³
- **Construction**: Lightweight (steel stud + 50mm insulation + gypsum)
- **Infiltration**: 0.5 ACH constant
- **HVAC**: None (free-floating)
- **Weather**: USA_CO_Golden-NREL TMY3 (39.74°N, 105.18°W)

### Model 2: Conduction Response
- **Geometry**: 6m × 8m × 2.7m single zone
- **Test wall (South)**: 200mm concrete (k=1.73 W/(m·K), ρ=2300 kg/m³, cp=840 J/(kg·K))
- **Other surfaces**: Highly insulated (R-20, k=0.01 W/(m·K), 200mm)
- **HVAC**: None (free-floating)
- **Timestep**: 15 minutes (4 per hour)
- **Run period**: 72 hours (Jan 1-3)
- **Driving force**: Real outdoor temperature from Golden-NREL TMY3
- **ZONE_EXT**: Step from 20°C to -10°C at hour 1 via schedule (ideal loads)
- **Other surfaces**: Adiabatic
- **Timestep**: 15 minutes (4 per hour)
- **Run period**: 72 hours (Jan 1-3)

## CSV Format

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
