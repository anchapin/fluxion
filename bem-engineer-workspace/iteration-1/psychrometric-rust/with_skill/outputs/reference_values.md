# Psychrometric Reference Values

## Conditions

| Parameter | Value |
|-----------|-------|
| Dry-bulb temperature (T_db) | 25.0 °C |
| Relative humidity (RH) | 50% (0.50) |
| Total pressure (P) | 101 325 Pa (sea level) |

## Derived Properties

All values computed using ASHRAE Handbook of Fundamentals, Chapter 1 correlations
(Hyland-Wexler saturation pressure formulation, eq. 5 for T ≥ 0°C).

| Property | Symbol | Value | Unit |
|----------|--------|-------|------|
| Saturation pressure | P_sat | 3 169.22 | Pa |
| Vapor pressure | P_v | 1 584.61 | Pa |
| Humidity ratio | W | 0.009 881 58 | kg_w / kg_da |
| Enthalpy | h | 50.3233 | kJ / kg_da |
| Specific volume | v | 0.858 088 | m³ / kg_da |
| Dew-point temperature | T_dp | 13.8640 | °C |
| Wet-bulb temperature | T_wb | 17.82 | °C |
| Density | ρ | 1.1769 | kg / m³ |

## Correlation Details

### Saturation Pressure (Hyland-Wexler, over liquid water)

ln(P_sat) = C₁/T + C₂ + C₃T + C₄T² + C₅T³ + 6.5459673·ln(T)

where T is in Kelvin and:
- C₁ = −5800.2206
- C₂ = 1.3914993
- C₃ = −0.048640239
- C₄ = 4.1764768×10⁻⁵
- C₅ = −1.4452093×10⁻⁸

### Humidity Ratio

W = 0.62198 · P_v / (P − P_v)

### Enthalpy

h = 1.006·T_db + W·(2501.0 + 1.86·T_db)  [kJ/kg_da]

Reference state: 0°C dry air + liquid water at 0°C.

### Specific Volume

v = R_da·(T_db + 273.15)·(1 + W/0.62198) / P  [m³/kg_da]

where R_da = 287.055 J/(kg·K).

### Dew Point

Obtained by Newton-Raphson inversion of the saturation pressure function:
find T_dp such that P_sat(T_dp) = P_v.

### Wet-Bulb Temperature

Obtained by bisection on the enthalpy balance:
find T_wb such that h_sat(T_wb, P) = h(T_db, W, P).
