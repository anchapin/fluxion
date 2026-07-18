# LBNL FLEXLAB Empirical Validation Dataset

This document identifies and licenses the primary empirical validation dataset
for Fluxion's empirical validation suite (T10.2 prerequisite for T10.3–T10.5
ingestion).

## 7-line summary

1. LBNL FLEXLAB (Facility for Low Energy eXperiments) is the primary empirical
   dataset for Fluxion validation.
2. The FLEXLAB-ASHRAE140 repository provides measured data for ASHRAE Standard
   140 empirical test cases.
3. Dataset includes interior temperature sensors, heat flux, HVAC flow/temperature
   measurements, and weather data.
4. Data covers multiple test scenarios with conventional mixing ventilation and
   radiant systems.
5. License: US government-funded work (DOE), publicly available for reuse.
6. Data access: GitHub repository `LBNL-ETA/FLEXLAB-ASHRAE140` (public).
7. Citation: Haves et al. (2020), DOI 10.20357/B7H88D.

## Dataset Overview

**Name:** FLEXLAB-ASHRAE140 Measured Data
**Facility:** LBNL FLEXLAB, Lawrence Berkeley National Laboratory, Berkeley, CA
**Climate Zone:** ASHRAE 3C (marine)
**Latitude/Longitude:** 37.87° N, 122.27° W

### What is FLEXLAB?

FLEXLAB (Facility for Low Energy eXperiments) is a DOE-funded building energy
research facility at Lawrence Berkeley National Laboratory. It provides
well-characterized, highly instrumented test cells for empirical validation of
building energy simulation tools. The facility was specifically designed to
generate validation-grade datasets for ASHRAE Standard 140.

### Data Contents

The FLEXLAB-ASHRAE140 repository provides:

- **Direct Validation Data:** Instantaneous measurements for comparing simulation
  output against measured values (zone temperatures, heat fluxes, HVAC flows).
- **Indirect Validation Data:** Longer-duration measurements for whole-building
  energy comparisons.
- **Input Data:** Building geometry, construction materials, HVAC specifications.
- **Auxiliary Data:** Weather data, material properties, construction details.
- **Architectural and Structural Drawings:** Detailed as-built documentation.
- **Window Models:** Manufacturer performance data for fenestration.
- **THERM Files:** Thermal bridging analysis for wall assemblies.

### Measurements Available

| Measurement Type | Resolution | Coverage |
|---|---|---|
| Zone air temperature | Sub-hourly | All test cells |
| Heat flux (walls, roof, floor) | Sub-hourly | All surfaces |
| HVAC air flow rates | Sub-hourly | Supply/return |
| HVAC water flow rates | Sub-hourly | Heating/cooling loops |
| HVAC supply/return temperatures | Sub-hourly | All streams |
| Outdoor weather (T, RH, wind, solar) | Hourly | On-site weather station |
| Interior sensible/cooling loads | Derived | Per test scenario |

### Test Scenarios

The Phase II dataset (DOE Lab RFP-2019) includes:

1. **Empty cells, conventional mixing ventilation** — baseline heating/cooling
   loads (Phase I, 7 scenarios)
2. **Furnished cells** — effect of thermal mass and internal gains
3. **Radiant slab and panel systems** — low-energy conditioning validation
4. **Varied airflow rates** — ventilation sensitivity
5. **Mixed-mode systems** — mechanical cooling + natural ventilation

## License

### Legal Status

The FLEXLAB-ASHRAE140 dataset was produced under US Department of Energy funding
(DOE Lab RFP-2019, Budget: $1,950,000). As a US government-funded work, the
data is in the **public domain** in the United States.

- **License:** Public Domain (US Government work)
- **ODbL or CC0:** Not explicitly stated; treated as public domain per
  17 U.S.C. § 105 (works of US Government employees are not subject to
  copyright in the US)
- **Restrictions:** None for research or commercial use
- **Attribution:** Required by good practice (see Citation below)

### Data Access

| Resource | URL |
|---|---|
| GitHub Repository | https://github.com/LBNL-ETA/FLEXLAB-ASHRAE140 |
| DOE Project Page | https://www.energy.gov/cmei/buildings/empirical-validation-energy-simulation-flexlab |
| ASHRAE 140 Data | https://data.ashrae.org/standard140 |
| OSTI Technical Report | https://www.osti.gov/biblio/1619175 |

### Accessing the Data

```bash
# Clone the repository
git clone https://github.com/LBNL-ETA/FLEXLAB-ASHRAE140.git

# Key directories:
# Direct Validation Data/   — instantaneous comparison data
# Indirect Validation Data/  — longer-duration energy data
# Input Data/                — building specs and construction
# Auxiliary Data/            — weather, materials, construction details
```

## Citation

### Primary Reference

Haves, P., Ravache, B., and Yazdanian, M. (2020). "Accuracy of HVAC Load
Predictions: Validation of EnergyPlus and DOE-2 using FLEXLAB Measurements."
Lawrence Berkeley National Laboratory. DOI: [10.20357/B7H88D](https://doi.org/10.20357/B7H88D)
OSTI: 1619175.

### BibTeX

```bibtex
@techreport{haves2020flexlab,
  author      = {Haves, Philip and Ravache, Baptiste and Yazdanian, Mehry},
  title       = {Accuracy of HVAC Load Predictions: Validation of EnergyPlus
                 and DOE-2 using FLEXLAB Measurements},
  institution = {Lawrence Berkeley National Laboratory},
  year        = {2020},
  doi         = {10.20357/B7H88D},
  osti_id     = {1619175},
}
```

### Related References

- Kohler, C. et al. (2021). "Empirical Validation and Uncertainty
  Characterization for Energy Simulation." DOE BTO Peer Review.
- ASHRAE Standard 140-2023. "Method of Test for Evaluating Building Performance
  Simulation Software." ASHRAE, Atlanta, GA.

## Dataset Metadata for Fluxion

```rust
MonitoredDataSource {
    id: "lbnl_flexlab_ashrae140",
    name: "LBNL FLEXLAB ASHRAE 140 Empirical Validation",
    source: "LBNL FLEXLAB-ASHRAE140 (DOE Lab RFP-2019)",
    building_type: BuildingType::Office,
    climate_zone: "3C",
    location: "Berkeley, CA",
    latitude: 37.87,
    longitude: -122.27,
    // Individual test cells vary; representative values below
    floor_area: 27.0,  // Single test cell ~4.5m x 6m
    num_floors: 1,
    zone_volume: 72.9,  // ~4.5m x 6m x 2.7m
    // Construction details vary by test case; see Input Data/ directory
    time_resolution_hours: 1.0,  // Hourly aggregated; raw is sub-hourly
    num_data_points: 8760,  // 1 year minimum
}
```

## Relationship to Fluxion Validation Phases

This dataset satisfies the prerequisites for:

- **T10.3:** Ingest weather + interior temperature data from FLEXLAB
- **T10.4:** Ingest energy use (HVAC loads) from FLEXLAB
- **T10.5:** Run Fluxion simulation and compare against FLEXLAB measurements

The FLEXLAB data is the **primary** empirical dataset. Secondary datasets
(ORNL FRP, NREL iUnit, NIST NZERTF) are documented in
`empirical_validation_datasets.md` and provide additional climate zones and
building types for broader validation coverage.

## Notes

- The dataset was specifically created for ASHRAE Standard 140 empirical test
  cases, making it directly comparable to Fluxion's existing ASHRAE 140
  validation suite.
- FLEXLAB test cells are guarded twin cells with known construction properties,
  minimizing model input uncertainty.
- Weather data is from an on-site station, not TMY, enabling direct comparison
  without weather file translation.
- The Phase II dataset adds furniture, radiant systems, and mixed-mode
  scenarios beyond the baseline empty-cell Phase I data.
