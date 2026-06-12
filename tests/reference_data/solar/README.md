# Solar Reference Data

Isolated solar reference data generated from EnergyPlus simulations for bottom-up
validation of the Fluxion solar module.

## Files

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `solar_position_denver.csv` | Hourly solar position for Denver TMY3 | 8760 | hour, altitude, azimuth, zenith |
| `surface_irradiance_south.csv` | Beam and ground-reflected irradiance on south vertical wall | 8760 | hour, beam, diffuse |
| `solar_gain_distribution.csv` | Per-surface solar gain distribution for ASHRAE 140 box | 43800 | hour, surface, beam_W, diffuse_W, total_W |

## solar_gain_distribution.csv

**Model**: ASHRAE 140 lightweight box (6×8×2.7m), no HVAC, Denver TMY3
**EPW**: USA_CO_Golden-NREL.724666_TMY3.epw
**Surfaces**: SouthWall (16.2 m²), NorthWall (16.2 m²), EastWall (21.6 m²), WestWall (21.6 m²), Roof (48.0 m²)

**Columns**:
- `hour(1-8760)`: Hour index (1–8760)
- `surface`: Surface name (SouthWall, NorthWall, EastWall, WestWall, Roof)
- `beam_W`: Beam solar gain [W] = beam irradiance (W/m²) × surface area (m²)
- `diffuse_W`: Diffuse solar gain [W] = (ground diffuse + sky diffuse) × surface area
- `total_W`: Total solar gain [W] = beam_W + diffuse_W

**Output Variables** (E+ 25.2):
- `Surface Outside Face Incident Beam Solar Radiation Rate per Area`
- `Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area`
- `Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area`

**Note**: Zone has no windows, so transmitted solar = 0 for all surfaces.

## Data Generation

Run from repository root:

```bash
python tests/reference_data/generate_reference_data.py
```

Prerequisites:
- EnergyPlus 25.2.0 on PATH
- EPW: `USA_CO_Golden-NREL.724666_TMY3.epw` (bundled with E+)
