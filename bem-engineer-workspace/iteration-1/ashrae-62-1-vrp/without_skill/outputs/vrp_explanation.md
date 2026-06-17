# ASHRAE 62.1-2022 Ventilation Rate Procedure — Conference Room Worked Example

## Table Citations

| Data | Source | Value |
|------|--------|-------|
| Default occupant density for Conference Rooms | **Table 6.4** (§6.4, "Minimum Ventilation Rates in Breathing Zone") | 50 people / 1,000 ft² |
| People OA rate (Rp) for Conference Rooms | **Table 6.4** | 5 cfm/person |
| Area OA rate (Ra) for Conference Rooms | **Table 6.4** | 0.06 cfm/ft² |
| Zone air distribution effectiveness (Ez), ceiling supply of cool air, ceiling return | **Table 6.3** (§6.4) | 1.0 |
| VRP equations | **§6.4**, Equations 6.4-1 and 6.4-2 | — |

## Given

- Zone category: **Conference Room**
- Zone floor area: **Az = 3,000 ft²**
- Occupant density (default from Table 6.4): **50 people per 1,000 ft²**
- Air distribution: ceiling supply (cool air), ceiling return → **Ez = 1.0**

## VRP Math

### Step 1 — Determine zone population (Pz) from default density

```
Pz = (50 people / 1,000 ft²) × 3,000 ft² = 150 people
```

### Step 2 — Breathing-zone outdoor airflow (Vbz), Equation 6.4-1

```
Vbz = Rp × Pz + Ra × Az

Vbz = (5 cfm/person × 150 people) + (0.06 cfm/ft² × 3,000 ft²)
Vbz = 750 cfm + 180 cfm
Vbz = 930 cfm
```

### Step 3 — Zone outdoor airflow (V_oz), Equation 6.4-2

```
V_oz = Vbz / Ez

V_oz = 930 cfm / 1.0
V_oz = 930 cfm
```

## Result

**The minimum outdoor air rate for a 3,000 ft² conference room with default occupant density per ASHRAE 62.1-2022 is 930 cfm.**

This assumes a single-zone system without an energy recovery ventilator. For systems with ERV or multiple zones recirculating air, apply the additional adjustments in §6.4 (system ventilation efficiency Ev per §6.4.4 / Normative Appendix A).

## Key References

- **ASHRAE 62.1-2022 §6.4** — Ventilation Rate Procedure
- **Table 6.3** — Zone Air Distribution Effectiveness (Ez)
- **Table 6.4** — Minimum Ventilation Rates in Breathing Zone (Rp, Ra, default occupant density)
- **Equation 6.4-1**: `Vbz = Rp × Pz + Ra × Az`
- **Equation 6.4-2**: `V_oz = Vbz / Ez`
