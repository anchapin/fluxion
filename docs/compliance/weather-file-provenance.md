# Weather File Provenance Statement

**Product:** fluxion v0.8.0+
**Document Type:** Weather File Provenance Statement
**Maintained by:** Building Standards Engineer
**Last Updated:** 2026-06-16
**Related Issue:** #750, #732 (DEV-004)

---

## 1. Required Weather File

ASHRAE 140-2023 Annex C specifies the following weather file for all standard test cases:

| Field | Value |
|-------|-------|
| File Name | `USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw` |
| Location | Denver, Colorado (WMO# 724690) |
| Source | ASHRAE / NREL TMY (Typical Meteorological Year) |
| Latitude | 39.76°N |
| Longitude | 104.86°W |
| Elevation | 1611 m |
| Time Zone | UTC-7 (Mountain) |
| Standard | ASHRAE 140-2023 Annex C |
| Compliance Note | Using actual TMY (not TMY2) — 2023 edition Annex C update |

---

## 2. Required vs. Current Implementation

### 2.1 Required Implementation

Per ASHRAE 140-2023 Annex C, the simulation shall use the actual Denver Stapleton TMY file for all cases. The weather file provides:

- Hourly dry-bulb temperature (°C)
- Hourly wet-bulb temperature (°C)
- Hourly dew-point temperature (°C)
- Hourly relative humidity (%)
- Hourly atmospheric pressure (Pa)
- Hourly extraterrestrial horizontal radiation (Wh/m²)
- Hourly extraterrestrial direct normal radiation (Wh/m²)
- Hourly horizontal infrared radiation (Wh/m²)
- Hourly global horizontal radiation (Wh/m²)
- Hourly direct normal radiation (Wh/m²)
- Hourly diffuse horizontal radiation (Wh/m²)
- Hourly wind direction (0-360°)
- Hourly wind speed (m/s)
- Hourly total sky cover
- Hourly opaque sky cover
- Hourly visibility (km)
- Hourly ceiling height (m)
- Hourly precipitable water (mm)

### 2.2 Current Implementation (DEV-004 — OPEN)

**The current fluxion implementation does NOT use the actual ASHRAE 140 Annex C TMY file.**

The code generates **synthetic weather data** using sine/cosine approximations of outdoor temperature and solar radiation. This is a **P1 compliance blocker** — no ASHRAE 140 certification submission is possible until resolved.

**Current weather generation code location:** `tools/data_gen/weather.py`

**Issue reference:** DEV-004 in `deviations-register.md`; GitHub Issue #732

---

## 3. Weather File in Repository

### 3.1 File Inventory

The following EPW files exist in the fluxion repository:

| File Path | Description | ASHRAE 140 Annex C Compliant |
|-----------|-------------|------------------------------|
| `assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw` | Denver TMY (current weather file) | YES — correct file, but **not currently used** by simulation engine |
| `assets/weather/USA_CO_Golden-NREL.724666_TMY3.epw` | Golden CO TMY3 | No — different location |
| `assets/weather/USA_FL_Miami.Intl.AP.722020_TMY3.epw` | Miami TMY3 | No — different climate |
| `assets/weather/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw` | Chicago TMY3 | No — different climate |
| `assets/weather/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw` | San Francisco TMY3 | No — different climate |
| `assets/weather/WD100.epw` through `WD500.epw` | Design day weather files | No — design day only, not TMY |
| `tests/test_data/denver.epw` | Denver EPW for testing | May be identical to Annex C file |
| `tests/test_data/test_denver.epw` | Synthetic test data | No — generated for unit testing |

### 3.2 Denver TMY File Provenance

**File:** `assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw`

| Property | Value |
|----------|-------|
| File Size | ~280 KB |
| Line Count | 8,768 (1 header + 8,760 hourly records + 7 metadata) |
| SHA256 Hash (as of last verification) | `ce78fcda675cdde480025cda2fde5444b7346e81c1d521d05189dcbf70794224` |
| Source | NREL/ASHRAE TMY database |
| WMO Station | 724690 |
| TMY Generation Year | TMY (not TMY2 or TMY3) |
| Period of Record | Mixed annual periods typical of 1991-2005 TMY generation |

**Verify hash:**
```bash
sha256sum assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw
```

### 3.3 Header Verification

The EPW file header (first line) is:
```
LOCATION,Denver-Stapleton Intl,CO,USA,TMY--23062,724690,39.76000,-104.8600,-7.0,1611.0
```

This confirms:
- Location: Denver-Stapleton International Airport
- Country: USA
- TMY designation: TMY--23062 (mixed year TMY from NREL)
- WMO# 724690
- Coordinates: 39.76°N, 104.86°W
- Timezone: UTC-7
- Elevation: 1611 m

---

## 4. Required Code Changes (DEV-004)

To fix DEV-004, the following changes are required in the weather module:

### 4.1 Required Change: Use Actual EPW File Instead of Synthetic Data

**Files affected:**
- `tools/data_gen/weather.py` — remove synthetic weather generation
- `src/weather/epw.rs` — ensure EPW parser correctly reads the TMY file
- `src/weather/mod.rs` or `src/simulation.rs` — wire the TMY file path

**Implementation approach:**

1. Replace synthetic sine/cosine temperature generation with actual EPW reader
2. Replace synthetic solar radiation generation with EPW direct normal and diffuse horizontal values
3. Verify all 17 EPW fields are correctly parsed and used
4. Validate that fluxion simulation output matches EnergyPlus for identical weather input

### 4.2 Weather File Path Configuration

The ASHRAE 140 validation uses:
```rust
"assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw"
```

This path is hardcoded in `src/bin/fluxion.rs` and `src/validation/ashrae_140_validator.rs`. This is acceptable for ASHRAE 140 compliance — the path is versioned in git and the SHA256 hash is documented above.

---

## 5. Compliance Verification Checklist

| Check | Status | Notes |
|-------|--------|-------|
| Correct file (Denver Stapleton TMY) present in repo | PASS | `assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw` |
| File SHA256 hash documented | PASS | `ce78fcda...` |
| Source URL documented | **TODO** | NREL TMY URL to be added |
| Weather file actually used by simulation | **FAIL** | Synthetic weather in use (DEV-004) |
| EPW fields correctly parsed | **TODO** | Verify all 17 fields |
| EPW used in CI validation pipeline | **TODO** | Wire TMY file in `ashrae_140_validation.yml` |

### 5.1 Source URL

The official source for ASHRAE 140-2023 Annex C weather data is:

- **NREL TMY Database:** https://opendata.nrel.gov/files/tmy-hourly
- **Direct file reference:** ASHRAE 140-2023 Annex C references the Denver Stapleton TMY file specifically

For certification submission, the exact URL from which the file was obtained must be documented. If obtained from the EnergyPlus installation (which bundles ASHRAE weather files), note the EnergyPlus version and installation path.

**TODO:** Add exact source URL and retrieval date before submission.

---

## 6. Known Issues

### DEV-004 — Synthetic Weather Data (OPEN, P1)

As documented in `deviations-register.md`:
> "Current code generates weather data from sine/cosine approximations of temperature and solar. ASHRAE 140 requires the Denver Stapleton TMY file specified in Annex C. This affects all 64 validation metrics."

**Impact:** Compliance blocker — no certification submission possible until resolved.

**Fix owner:** Weather module owner
**Target resolution:** Wave 1 (before first compliance submission)

---

## 7. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-06-16 | Building Standards Engineer | Initial version for Issue #750 |

---

*End of Weather File Provenance Statement*
