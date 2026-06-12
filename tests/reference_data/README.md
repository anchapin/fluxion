# EnergyPlus Reference Data

This directory contains isolated reference data generated from EnergyPlus simulations
for bottom-up validation of individual Fluxion physics modules.

## Data Generation

Run the generation scripts from the repository root:

```bash
# Original annual data (solar + ventilation + conduction)
python tests/reference_data/generate_reference_data.py

# Expanded ventilation scenarios (Issue #967)
python tests/reference_data/generate_ventilation_scenarios.py
```

Prerequisites:
- EnergyPlus 25.2.0 on PATH
- EPW files bundled with E+ in `/usr/local/EnergyPlus-25-2-0/WeatherData/`

## Generated Files

### Solar (`solar/`)

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `solar_position_denver.csv` | Hourly solar position for Denver TMY3 | 8760 | hour, altitude, azimuth, zenith |
| `surface_irradiance_south.csv` | Beam, diffuse, ground-reflected on south vertical wall | 8760 | hour, beam, diffuse, ground_reflected |
| `solar_gain_distribution.csv` | Per-surface solar gain distribution for ASHRAE 140 box | 43800 | hour, surface, beam_W, diffuse_W, total_W |

### Conduction (`conduction/`)

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `step_response_200mm_concrete.csv` | Transient conduction through 200mm concrete wall | 288 | hour, T_ext, T_surf_in, T_surf_out, q_in, q_out |

### Ventilation (`ventilation/`)

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `infiltration_denver.csv` | Denver, 0.5 ACH — original single scenario | 8760 | hour, T_out, wind_speed, ACH, C_vent |
| `infiltration_denver_05ach.csv` | Denver, 0.5 ACH — expanded (verify) | 8760 | hour, T_out, wind_speed, ACH, C_vent, Q_vent |
| `infiltration_denver_10ach.csv` | Denver, 1.0 ACH | 8760 | hour, T_out, wind_speed, ACH, C_vent, Q_vent |
| `infiltration_denver_01ach.csv` | Denver, 0.1 ACH (tight envelope) | 8760 | hour, T_out, wind_speed, ACH, C_vent, Q_vent |
| `infiltration_tampa_05ach.csv` | Tampa FL (hot-humid), 0.5 ACH | 8760 | hour, T_out, wind_speed, ACH, C_vent, Q_vent |
| `infiltration_dulles_05ach.csv` | Dulles VA (cold), 0.5 ACH | 8760 | hour, T_out, wind_speed, ACH, C_vent, Q_vent |

### Zone Balance (`zone_balance/`)

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `fixed_inputs_zone_temp.csv` | Annual zone energy balance with fixed sub-module inputs | 8760 | hour, T_zone, T_out, Q_cond, Q_solar, Q_vent, Q_int, Q_heat, Q_cool |

### EnergyPlus Models (`energyplus_models/`)

| File | Description |
|------|-------------|
| `annual_solar_ventilation.idf` | Single-zone box (6×8×2.7m), lightweight walls, no HVAC, 0.5 ACH |
| `step_change_concrete.idf` | Single-zone, 200mm concrete south wall, free-floating, Jan 1-3 weather-driven |
| `ventilation_denver_05ach.idf` | Denver, 0.5 ACH — same geometry, ACH parameterised |
| `ventilation_denver_10ach.idf` | Denver, 1.0 ACH |
| `ventilation_denver_01ach.idf` | Denver, 0.1 ACH (tight) |
| `ventilation_tampa_05ach.idf` | Tampa, 0.5 ACH |
| `ventilation_dulles_05ach.idf` | Dulles, 0.5 ACH |
| `ashrae_140_solar_gain.idf` | ASHRAE 140 box for per-surface solar gain distribution, 5 surfaces, no HVAC |
| `fixed_inputs_zone_temp.idf` | Single-zone, ideal loads HVAC locked at 20°C, all surfaces NoSun/NoWind, annual run |

## Model Parameters

### Base Geometry (all ventilation scenarios)
- **Geometry**: 6m × 8m × 2.7m single zone
- **Volume**: 129.6 m³
- **Construction**: Lightweight (steel stud + 50mm insulation + gypsum)
- **HVAC**: None (free-floating)

### Ventilation Scenarios (Issue #967)

| Scenario | EPW | Climate Zone | ACH | C_vent (W/K) |
|----------|-----|-------------|-----|---------------|
| Denver 0.5 ACH | USA_CO_Golden-NREL.724666_TMY3 | 5B (Mixed-Dry) | 0.5 | 21.6 |
| Denver 1.0 ACH | USA_CO_Golden-NREL.724666_TMY3 | 5B (Mixed-Dry) | 1.0 | 43.2 |
| Denver 0.1 ACH (tight) | USA_CO_Golden-NREL.724666_TMY3 | 5B (Mixed-Dry) | 0.1 | 4.32 |
| Tampa 0.5 ACH | USA_FL_Tampa.Intl.AP.722110_TMY3 | 1A (Hot-Humid) | 0.5 | 21.6 |
| Dulles 0.5 ACH | USA_VA_Sterling-Washington.Dulles.Intl.AP.724030_TMY3 | 6A (Cold) | 0.5 | 21.6 |

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

### Ventilation CSV Columns

```
hour(1-8760), T_out(C), wind_speed(m/s), ACH(1/h), C_vent(W/K), Q_vent(W)
```

- **hour**: Hourly timestep (1–8760, representing Jan 1 hour 1 through Dec 31 hour 24)
- **T_out**: Outdoor dry-bulb temperature [°C] from EPW
- **wind_speed**: Site wind speed [m/s] from EPW
- **ACH**: Air changes per hour [1/h] — constant per scenario
- **C_vent**: Ventilation conductance [W/K] = ACH × V × ρ × c_p / 3600
- **Q_vent**: Ventilation heat loss [W] = C_vent × (T_zone − T_out)
  - T_zone = Zone Mean Air Temperature from E+ simulation (free-floating, no HVAC)

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

Per-scenario conductance:

| ACH | C_vent (W/K) |
|-----|-------------|
| 0.1 | 4.32 |
| 0.5 | 21.6 |
| 1.0 | 43.2 |
