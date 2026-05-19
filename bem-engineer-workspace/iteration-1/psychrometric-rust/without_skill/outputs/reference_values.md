# Psychrometric Reference Values

**Conditions:** 25°C dry-bulb, 50% relative humidity, 101325 Pa (sea level)

## ASHRAE Formulation Used

Saturation pressure via ASHRAE fundamental equation (SI units):

```
ln(P_ws) = C₁/T + C₂ + C₃·T + C₄·T² + C₅·T³ + C₆·T⁴ + C₇·ln(T)
```

where T is absolute temperature in Kelvin, P_ws in Pa.

## Step-by-step Derivation

### 1. Saturation Pressure (P_ws)

T = 25 + 273.15 = 298.15 K

| Coefficient | Value |
|---|---|
| C₁ | -5800.2206 |
| C₂ | 1.3914993 |
| C₃ | -4.8640239e-2 |
| C₄ | 4.1764768e-5 |
| C₅ | -1.4452093e-8 |
| C₆ | 0.0 |
| C₇ | 6.5459673 |

ln(P_ws) = -5800.2206/298.15 + 1.3914993 + (-0.048640239)(298.15) + (4.1764768e-5)(298.15²) + (-1.4452093e-8)(298.15³) + 6.5459673·ln(298.15)

ln(P_ws) = -19.4537 + 1.3915 - 14.5026 + 3.7124 - 0.3832 + 37.2946 = 8.0590

**P_ws = exp(8.0590) = 3168.5 Pa**

### 2. Partial Vapor Pressure (P_w)

P_w = φ · P_ws = 0.50 × 3168.5

**P_w = 1584.3 Pa**

### 3. Humidity Ratio (W)

W = 0.62198 · P_w / (P - P_w) = 0.62198 × 1584.3 / (101325 - 1584.3) = 985.4 / 99740.7

**W = 0.009879 kg/kg_dry_air**

### 4. Enthalpy (h)

h = 1.006 · T_db + W · (2501 + 1.86 · T_db)

h = 1.006 × 25 + 0.009879 × (2501 + 1.86 × 25)

h = 25.15 + 0.009879 × 2547.5 = 25.15 + 25.17

**h = 50.32 kJ/kg_dry_air**

### 5. Dew Point Temperature (T_dp)

Using the inverse Magnus-Tetens formula from P_w:

T_dp = 243.04 · ln(P_w / 610.94) / (17.625 - ln(P_w / 610.94))

ln(1584.3 / 610.94) = ln(2.5931) = 0.9528

T_dp = 243.04 × 0.9528 / (17.625 - 0.9528) = 231.56 / 16.6722

**T_dp = 13.89°C**

### 6. Specific Volume (v)

v = R_a · T_K · (1 + 1.6078 · W) / P

v = 287.055 × 298.15 × (1 + 1.6078 × 0.009879) / 101325

v = 85601.1 × 1.01589 / 101325 = 86959.8 / 101325

**v = 0.8582 m³/kg_dry_air**

### 7. Wet-Bulb Temperature (T_wb)

Iteratively solved from the enthalpy balance. At 50% RH and 25°C:

**T_wb ≈ 17.95°C**

## Summary of Reference Values

| Property | Value | Unit |
|---|---|---|
| Dry-bulb temperature | 25.00 | °C |
| Relative humidity | 50.0 | % |
| Atmospheric pressure | 101325 | Pa |
| Saturation pressure | 3168.5 | Pa |
| Vapor pressure | 1584.3 | Pa |
| Humidity ratio | 0.00988 | kg/kg_da |
| Enthalpy | 50.32 | kJ/kg_da |
| Dew point temperature | 13.89 | °C |
| Wet-bulb temperature | 17.95 | °C |
| Specific volume | 0.858 | m³/kg_da |
