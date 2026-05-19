# ASHRAE 140 Section 8 Output Specification

## Overview

ASHRAE 140 Section 8 defines the required output metrics for inter-program comparison. This document describes each output metric, its units, format, and how Fluxion computes and reports it.

---

## Section 8 Output Metrics

### 8.1 Annual Heating Load

| Field | Value |
|---|---|
| **Metric** | Annual heating load |
| **Unit** | MWh |
| **Reference Table** | B8-1 — Annual Heating Loads |
| **Source** | `data/ashrae140_reference.json` → `annual_heating_MWh` |

**Definition**: Total annual sensible thermal energy delivered by the heating system to maintain the zone at the heating setpoint (20°C).

**Computation**:
```
Annual_Heating = Σ(hour=1 to 8760) max(0, Q_hvac(hour)) × Δt
```
where `Q_hvac(hour)` is the hourly ideal loads heating power in watts, and `Δt` = 1 hour.

**Conversion**: `J → MWh` using factor `1 / 3.6e9`

**Validation**: Result must fall within the inter-program min/max range from the six reference programs (BSIMAC, CSE, DeST, EnergyPlus, ESP-r, TRNSYS).

---

### 8.2 Annual Sensible Cooling Load

| Field | Value |
|---|---|
| **Metric** | Annual sensible cooling load |
| **Unit** | MWh |
| **Reference Table** | B8-2 — Annual Sensible Cooling Loads |
| **Source** | `data/ashrae140_reference.json` → `annual_cooling_MWh` |

**Definition**: Total annual sensible thermal energy removed by the cooling system to maintain the zone at the cooling setpoint (27°C).

**Computation**:
```
Annual_Cooling = Σ(hour=1 to 8760) max(0, -Q_hvac(hour)) × Δt
```

**Validation**: Same inter-program comparison as heating.

---

### 8.3 Peak Hourly Integrated Heating Load

| Field | Value |
|---|---|
| **Metric** | Peak hourly integrated heating load |
| **Unit** | kW |
| **Reference Table** | B8-3 — Annual Hourly Integrated Peak Heating Loads |
| **Source** | `data/ashrae140_reference.json` → `peak_heating_kW` |

**Definition**: Maximum single-hour heating power over the annual simulation.

**Computation**:
```
Peak_Heating = max(hour=1 to 8760) max(0, Q_hvac(hour)) / 1000
```

**Peak Timestamp**: The hour-of-year at which the peak occurs is recorded and reported. Format: `YYYY-MM-DD HH:MM` (Denver local time, UTC-7).

**Validation**: Must fall within inter-program range.

---

### 8.4 Peak Hourly Integrated Sensible Cooling Load

| Field | Value |
|---|---|
| **Metric** | Peak hourly integrated sensible cooling load |
| **Unit** | kW |
| **Reference Table** | B8-4 — Annual Hourly Integrated Peak Sensible Cooling Loads |
| **Source** | `data/ashrae140_reference.json` → `peak_cooling_kW` |

**Definition**: Maximum single-hour cooling power over the annual simulation.

**Computation**:
```
Peak_Cooling = max(hour=1 to 8760) max(0, -Q_hvac(hour)) / 1000
```

**Peak Timestamp**: Recorded similarly to peak heating.

---

### 8.5 Free-Float Zone Temperatures

| Field | Value |
|---|---|
| **Metrics** | Maximum, minimum, and mean annual zone temperature |
| **Unit** | °C |
| **Reference Table** | B8-5 — Free-Float Maximum/Minimum/Mean Annual Zone Temperature |
| **Source** | `data/ashrae140_reference.json` → `ff_max_zone_temp_C`, `ff_min_zone_temp_C`, `ff_mean_zone_temp_C` |

**Definition**: For free-floating cases (no HVAC), the zone temperature evolves freely based on weather, solar gains, internal gains, and thermal mass.

**Computation**:
```
T_max  = max(hour=1 to 8760) T_zone(hour)
T_min  = min(hour=1 to 8760) T_zone(hour)
T_mean = (1/8760) × Σ(hour=1 to 8760) T_zone(hour)
```

**Hourly Profiles**: The complete 8760-hour free-float temperature profile is stored for diagnostic analysis. This enables:
- Monthly minimum/maximum/mean breakdowns
- Identification of extreme temperature events
- Comparison with reference program hourly profiles

**Applicable Cases**: 600FF, 650FF, 900FF, 950FF (and other free-floating variants).

---

### 8.6 Incident Solar Radiation (Supplementary)

| Field | Value |
|---|---|
| **Metric** | Annual incident solar radiation on exterior surfaces |
| **Unit** | kWh/m² |
| **Reference Table** | Supplementary (not in B8 tables) |
| **Source** | Computed per-surface |

**Definition**: Total annual solar radiation incident on each building surface (walls, roof, glazing).

**Computation**:
```
I_annual = Σ(hour=1 to 8760) I_incident(hour, surface) × Δt
```

---

## Output Data Structures

### Validation Result (Rust)

The core validation result structure used internally:

```rust
// Representative fields — see src/validation/ for full definitions
struct CaseReference {
    case_id: &'static str,
    annual_heating_min: f64,   // MWh — inter-program minimum
    annual_heating_max: f64,   // MWh — inter-program maximum
    annual_cooling_min: f64,   // MWh
    annual_cooling_max: f64,   // MWh
    peak_heating_min: f64,     // kW
    peak_heating_max: f64,     // kW
    peak_cooling_min: f64,     // kW
    peak_cooling_max: f64,     // kW
    min_free_float_min: f64,   // °C
    min_free_float_max: f64,   // °C
    max_free_float_min: f64,   // °C
    max_free_float_max: f64,   // °C
}
```

### Reference Data Format (JSON)

Located at `data/ashrae140_reference.json`:

```json
{
  "_schema": {
    "version": "1.0",
    "source": {
      "standard": "ASHRAE 140-2023",
      "programs": ["BSIMAC 9.0.74", "CSE 0.861.1", "DeST 2.0",
                    "EnergyPlus 9.0.1", "ESP-r 13.3", "TRNSYS 18.01.0001"]
    }
  },
  "cases": {
    "600": {
      "annual_heating_MWh": { "min": X, "max": Y, "mean": Z },
      "annual_cooling_MWh": { "min": X, "max": Y, "mean": Z },
      "peak_heating_kW":    { "min": X, "max": Y, "mean": Z },
      "peak_cooling_kW":    { "min": X, "max": Y, "mean": Z }
    }
  }
}
```

Free-floating cases additionally include:
```json
"600FF": {
  "ff_max_zone_temp_C":  { "min": X, "max": Y, "mean": Z },
  "ff_min_zone_temp_C":  { "min": X, "max": Y, "mean": Z },
  "ff_mean_zone_temp_C": { "min": X, "max": Y, "mean": Z }
}
```

### Holdout Data

A separate holdout dataset (`data/ashrae140_holdout.json`) exists for blind validation — cases where the reference ranges are not exposed during development to prevent overfitting.

---

## Validation Pass Criteria

For each test case, Fluxion's result is compared against the inter-program reference range:

| Criterion | Pass Condition |
|---|---|
| Annual heating | `min ≤ result ≤ max` |
| Annual cooling | `min ≤ result ≤ max` |
| Peak heating | `min ≤ result ≤ max` |
| Peak cooling | `min ≤ result ≤ max` |
| FF max temperature | `min ≤ result ≤ max` |
| FF min temperature | `min ≤ result ≤ max` |

The `min` and `max` values represent the range of results from all six reference programs, providing a tolerance band that accounts for legitimate modeling differences (e.g., different sky models, slightly different solar geometry implementations).

---

## Report Generation

### Console Report

The CLI (`fluxion validate`) produces a summary report with:
- Per-case PASS/FAIL status for each metric
- Actual vs. reference range comparison
- Overall pass rate across all test cases

### CSV Export

The `export_csv` binary (`src/bin/export_csv.rs`) generates:
- Hourly time-series data for diagnostic analysis
- Annual summary statistics per test case
- Free-floating temperature profiles

### Markdown Results

Results are documented in `docs/ASHRAE140_RESULTS.md` with:
- Summary table of all cases and metrics
- Deviation from reference mean
- Historical comparison between versions

---

## Section 8 Compliance Summary

| Section 8 Requirement | How Fluxion Meets It |
|---|---|
| Annual heating loads (B8-1) | Hourly ideal loads accumulation, reported in MWh |
| Annual cooling loads (B8-2) | Hourly ideal loads accumulation, reported in MWh |
| Peak heating loads (B8-3) | Maximum hourly heating power with timestamp, reported in kW |
| Peak cooling loads (B8-4) | Maximum hourly cooling power with timestamp, reported in kW |
| Free-float temperatures (B8-5) | Min/max/mean of 8760-hour free-float profile, reported in °C |
| Weather data | Denver TMY (39.83°N, 104.65°W, 1655m) |
| Time step | 1 hour (8760 hours/year) |
| Ground temperature | 10°C constant (per ASHRAE 140 specification) |
| Internal gains | Per case specification (zero for base cases, specified for others) |
| Infiltration/ventilation | Per case specification (0.5 ACH for base cases) |

---

## References

1. ASHRAE Standard 140-2023 — Section 8: *Output Specifications*
2. Std140_TF_Results.pdf (TESS, 19-Aug-2024) — Annex B8 inter-program comparison tables
3. `data/ashrae140_reference.json` — Fluxion's encoded reference data from B8 tables
4. `docs/ASHRAE140_RESULTS.md` — Latest validation results
