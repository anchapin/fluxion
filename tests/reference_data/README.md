# EnergyPlus Reference Data

This directory contains isolated reference data generated from EnergyPlus simulations
for bottom-up validation of individual Fluxion physics modules.

## Data Generation

Run the generation script from the repository root:

```bash
# Single-climate reference data (Denver / Golden-NREL TMY3)
python tests/reference_data/generate_reference_data.py

# Multi-climate reference data (Miami / Phoenix / Chicago-as-6A-proxy)
python tests/reference_data/generate_multi_climate_reference.py

# Expanded ventilation scenarios (Denver / Tampa / Dulles)
python tests/reference_data/generate_ventilation_scenarios.py
```

Prerequisites:
- EnergyPlus 25.2.0 on PATH (`generate_reference_data.py`,
  `generate_ventilation_scenarios.py`)
- EPW: `USA_CO_Golden-NREL.724666_TMY3.epw` (bundled with E+)
- EPW: `USA_FL_Miami.Intl.AP.722020_TMY3.epw`,
  `USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.epw`,
  `USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw` (provided under
  `tests/test_data/`, downloaded from NREL/EnergyPlus, for
  `generate_multi_climate_reference.py`)

## Climate Coverage

ASHRAE 169 climate zones now covered by reference data after Issue #1427:

| ASHRAE Zone | City                | EPW File                                       | Reference Data                |
|-------------|---------------------|------------------------------------------------|-------------------------------|
| 1A          | Miami, FL           | `USA_FL_Miami.Intl.AP.722020_TMY3.epw`         | weather / solar / ventilation |
| 2A          | Tampa, FL           | `USA_FL_Tampa.Intl.AP.722110_TMY3.epw`         | ventilation only              |
| 2B          | Phoenix, AZ         | `USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.epw` | weather / solar / ventilation |
| 3C          | San Francisco, CA   | `USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw` | EPW bundled only              |
| 4A          | Dulles, VA          | `USA_VA_Sterling-Washington.Dulles.Intl.AP.724030_TMY3.epw` | ventilation only              |
| 5A          | Chicago, IL*        | `USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw` | weather / solar / ventilation |
| 5B          | Denver, CO          | `USA_CO_Golden-NREL.724666_TMY3.epw`           | weather / solar / ventilation / conduction |
| 6A target   | Minneapolis, MN†    | (not available)                                | n/a                           |
| 7, 8        | (none)              | (none)                                         | n/a                           |

*Chicago (5A) is the publicly-available cold-climate substitute for the
issue's stated Minneapolis 6A target (Minneapolis-St.Paul 6A TMY3 is not
bundled with EnergyPlus 25.2.0 and is not present on the canonical public
TMY3 mirrors we sampled). The substitution is documented in the
`CLIMATES` table of `generate_multi_climate_reference.py`.

†The Minneapolis 6A EPW was not found at any of:
- `https://raw.githubusercontent.com/NREL/EnergyPlus/develop/weather/`
- `https://raw.githubusercontent.com/NREL/openstudio-standards/`
- `https://www.energyplus.net/weatherdata/...`
- `https://climate.onebuilding.org/...`

After Issue #1427 the climate coverage ratio rises from 3/8 (38%) → **6/8
(75%)**: zones 1A, 2A, 2B, 4A, 5A, 5B have at least one reference CSV;
zone 6A has the EPW present but no reference CSVs until a public 6A EPW
is available.

## Generated Files

### Solar (`solar/`)

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `solar_position_denver.csv` | Hourly solar position for Denver TMY3 (E+ 25.2 reference) | 8760 | hour, altitude, azimuth, zenith |
| `surface_irradiance_south.csv` | Beam, ground-reflected on south vertical wall (Denver TMY3, E+ 25.2 reference) | 8760 | hour, beam, ground_reflected |
| `solar_position_miami.csv` | Hourly solar position for Miami TMY3 (Issue #1427) | 8760 | hour, altitude, azimuth, zenith |
| `solar_position_phoenix.csv` | Hourly solar position for Phoenix TMY3 (Issue #1427) | 8760 | hour, altitude, azimuth, zenith |
| `solar_position_minneapolis.csv` | Hourly solar position for Chicago 5A TMY3 (Issue #1427, 6A proxy) | 8760 | hour, altitude, azimuth, zenith |
| `surface_irradiance_south_miami.csv` | South-wall beam + ground, Miami (Issue #1427) | 8760 | hour, beam, ground_reflected |
| `surface_irradiance_south_phoenix.csv` | South-wall beam + ground, Phoenix (Issue #1427) | 8760 | hour, beam, ground_reflected |
| `surface_irradiance_south_minneapolis.csv` | South-wall beam + ground, Chicago/5A (Issue #1427, 6A proxy) | 8760 | hour, beam, ground_reflected |

### Weather (`weather/`)

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `denver_tmy3_reference.csv` | Hourly TMY3 weather for Denver (E+ reference) | 8760 | hour, dry_bulb_temp_c, humidity_rh_pct, dni_wm2, dhi_wm2, ghi_wm2, wind_speed_ms, humidity_ratio_kgkg |
| `miami_tmy3_reference.csv` | Hourly TMY3 weather for Miami (Issue #1427) | 8760 | (same as denver) |
| `phoenix_tmy3_reference.csv` | Hourly TMY3 weather for Phoenix (Issue #1427) | 8760 | (same as denver) |
| `minneapolis_tmy3_reference.csv` | Hourly TMY3 weather for Chicago 5A (Issue #1427, 6A proxy) | 8760 | (same as denver) |

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
| `infiltration_denver.csv` | Hourly outdoor temp, wind, infiltration ACH, vent conductance (Denver) | 8760 | hour, T_out, wind_speed, ACH, C_vent |
| `infiltration_denver_05ach.csv` | Denver 0.5 ACH scenario (E+) | 8760 | hour, T_out, wind_speed, ACH, C_vent, Q_vent |
| `infiltration_denver_10ach.csv` | Denver 1.0 ACH scenario (E+) | 8760 | hour, T_out, wind_speed, ACH, C_vent, Q_vent |
| `infiltration_denver_01ach.csv` | Denver 0.1 ACH tight-envelope scenario (E+) | 8760 | hour, T_out, wind_speed, ACH, C_vent, Q_vent |
| `infiltration_tampa_05ach.csv` | Tampa (2A) 0.5 ACH scenario (E+) | 8760 | hour, T_out, wind_speed, ACH, C_vent, Q_vent |
| `infiltration_dulles_05ach.csv` | Dulles (4A) 0.5 ACH scenario (E+) | 8760 | hour, T_out, wind_speed, ACH, C_vent, Q_vent |
| `infiltration_miami_05ach.csv` | Miami (1A) 0.5 ACH scenario (Issue #1427) | 8760 | hour, T_out, wind_speed, ACH, C_vent, Q_vent |
| `infiltration_phoenix_05ach.csv` | Phoenix (2B) 0.5 ACH scenario (Issue #1427) | 8760 | hour, T_out, wind_speed, ACH, C_vent, Q_vent |
| `infiltration_minneapolis_05ach.csv` | Chicago/5A (6A proxy) 0.5 ACH scenario (Issue #1427) | 8760 | hour, T_out, wind_speed, ACH, C_vent, Q_vent |

### EnergyPlus Models (`energyplus_models/`)

| File | Description |
|------|-------------|
| `annual_solar_ventilation.idf` | Single-zone box (6×8×2.7m), lightweight walls, no HVAC, 0.5 ACH |
| `step_change_concrete.idf` | 200mm concrete south wall, free-floating, Jan 1-3 weather-driven |
| `step_change_composite.idf` | Composite wall (concrete + insulation + gypsum), south wall, Jan 1-3 |
| `step_change_floor.idf` | Floor slab on grade, Jan 1-3 |
| `step_change_lightweight.idf` | Lightweight steel stud wall, south wall, Jan 1-3 |
| `step_change_roof.idf` | Roof assembly, Jan 1-3 |

### Test Files (`tests/reference_data/`)

| File | Description |
|------|-------------|
| `multi_climate_solar_invariant.rs` | Multi-climate solar + ventilation invariants across all 4 climates (Issue #1427). Runs in <10 s. |

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

### Multi-climate Solar + Ventilation (Issue #1427)
- **Geometry**: 6m × 8m × 2.7m single zone (same as Model 1)
- **Volume**: 129.6 m³
- **Infiltration**: 0.5 ACH constant (ZoneHVAC fixed at T_zone = 20°C
  for the Q_vent calculation; no HVAC, no internal gains)
- **Solar**: NOAA SPA simplified (matches `src/solar/solar_position.rs`)
  + Perez 1990 all-weather diffuse + isotropic ground reflection
  (matches `src/solar/surface_irradiance.rs`)
- **Tilt/Azimuth**: 90°/180° (south-facing vertical wall)
- **Weather sites**:
  - Miami, FL (25.82°N, 80.30°W, tz=-5, elev=11m) — Zone 1A
  - Phoenix, AZ (33.45°N, 111.98°W, tz=-7, elev=337m) — Zone 2B
  - Chicago, IL (41.98°N, 87.92°W, tz=-6, elev=201m) — Zone 5A
    (substituting for Minneapolis 6A per issue #1427)

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

### Weather CSV columns (denver/miami/phoenix/minneapolis)

```
hour, dry_bulb_temp_c, humidity_rh_pct, dni_wm2, dhi_wm2, ghi_wm2, wind_speed_ms, humidity_ratio_kgkg
```

- `hour`: hour-of-year index, 0-indexed (0..8759)
- `dry_bulb_temp_c`: outdoor air dry-bulb temperature (°C)
- `humidity_rh_pct`: relative humidity (%)
- `dni_wm2`, `dhi_wm2`, `ghi_wm2`: direct normal, diffuse horizontal, global horizontal irradiance (W/m²)
- `wind_speed_ms`: wind speed at 10 m (m/s)
- `humidity_ratio_kgkg`: kg_water_vapor / kg_dry_air, computed with the
  same Magnus-Tetens (T≥0) + Hyland-Wexler ice (T<0) formulas as
  `fluxion-core/src/weather/psychrometrics.rs`

### Solar position CSV columns (denver/miami/phoenix/minneapolis)

```
hour(1-8760), solar_altitude(deg), solar_azimuth(deg), solar_zenith(deg)
```

1-indexed hours; `solar_azimuth` is 0=N, 90=E, 180=S, 270=W (meteorological convention).

### Surface irradiance CSV columns

```
hour(1-8760), beam_irradiance(W/m2), ground_diffuse_irradiance(W/m2)
```

1-indexed hours; beam and ground-reflected irradiance on a south-facing
vertical wall (azimuth=180°, tilt=90°). Sky diffuse is captured by
fluxion's Perez model and not stored in the CSV (E+ 25.2 also does not
separate sky/ground diffuse in its standard output).

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

For Issue #1427 multi-climate CSVs, Q_vent is computed analytically as
`C_vent × (T_zone - T_out)` with a fixed T_zone = 20°C, identical to the
constant-ACH design value in `generate_ventilation_scenarios.py`.
