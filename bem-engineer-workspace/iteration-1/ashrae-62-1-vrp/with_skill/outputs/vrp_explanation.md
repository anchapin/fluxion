# ASHRAE 62.1-2022 Ventilation Rate Procedure — Worked Example

## Problem Statement

Determine the minimum outdoor air rate for a **3,000 sq ft conference room** using
default occupant density per ASHRAE 62.1-2022.

## Table Citation

**Table 6.2.2.1** — "Minimum Ventilation Rates in Breathing Zone"
(ASHRAE Standard 62.1-2022, Section 6.2.2.1)

| Occupancy Category | People Rate Rp | Area Rate Ra | Default Density |
|---|---|---|---|
| Conference/Meeting | 5.0 cfm/person | 0.06 cfm/sq ft | 50 per 1,000 sq ft |

- **Rp** = 5.0 cfm/person (people outdoor air rate)
- **Ra** = 0.06 cfm/sq ft (area outdoor air rate)
- **Default occupant density** = 50 people per 1,000 sq ft

## Step 1 — Determine Zone Population (Pz)

Using the default density from Table 6.2.2.1:

```
Pz = (Az / 1000) × default_density
Pz = (3000 / 1000) × 50
Pz = 150 people
```

## Step 2 — Breathing Zone Outdoor Airflow (V_bz)

Per **Equation 6.2.3-1** (Section 6.2.3):

```
V_bz = Rp × Pz + Ra × Az
V_bz = 5.0 × 150 + 0.06 × 3000
V_bz = 750 + 180
V_bz = 930 cfm
```

## Step 3 — Zone Outdoor Airflow (V_oz)

Per **Equation 6.2.3-2** (Section 6.2.3):

```
V_oz = V_bz / (Ez × Ds × Ep)
```

Where:

- **Ez** = Zone air distribution effectiveness (Table 6.2.2.2). For a typical
  conference room with ceiling supply of cool air and floor-level return, **Ez = 1.0**.
- **Ds** = System diversity factor. Default = 1.0 (single-zone or no diversity).
- **Ep** = Plenum/transfer air effectiveness. Default = 1.0.

```
V_oz = 930 / (1.0 × 1.0 × 1.0)
V_oz = 930 cfm
```

## Result

| Parameter | Value |
|---|---|
| Zone area (Az) | 3,000 sq ft |
| Zone population (Pz) | 150 people |
| Rp (people rate) | 5.0 cfm/person |
| Ra (area rate) | 0.06 cfm/sq ft |
| **V_bz** | **930 cfm** |
| Ez | 1.0 |
| **V_oz** | **930 cfm** |

The minimum outdoor air rate for this conference room is **930 cfm**.

## Generalized VRP Equations

For any zone, the procedure is:

1. **Look up rates** from Table 6.2.2.1 for the occupancy category → get Rp, Ra, default density.
2. **Determine Pz** — either from design occupancy or: `Pz = (Az / 1000) × default_density`.
3. **V_bz** (Eq. 6.2.3-1): `V_bz = Rp × Pz + Ra × Az`
4. **V_oz** (Eq. 6.2.3-2): `V_oz = V_bz / (Ez × Ds × Ep)`

The Python function `compute_zone_ventilation()` in `vrp_calculator.py` implements
this procedure for any zone given area, population (or default density), and
zone category.
