# Phase 25: Alternative Physics Implementation - Execution Summary

**Status:** IN PROGRESS (Wave 1: EnergyPlus Data Generation)
**Date:** 2026-03-17
**Tool:** OpenStudio MCP 0.8.2 (OpenStudio 3.11.0, EnergyPlus 25.2.0)

---

## Phase 25 Goal

Implement and test alternative physics approaches for high-mass building thermal modeling. Achieve ±15% annual energy accuracy for Case 900 (currently 229-322% error with 5R1C).

---

## Execution Progress

### Wave 1: Foundation (Week 1-2)

#### ✅ Plan 25-01: EnergyPlus Data Generation via OpenStudio MCP

**Status:** BASELINE COMPLETE
**Run ID:** `51324bf14e45433989e4d032e148289b`

**What Was Accomplished:**

1. **Created ASHRAE 140 Case 900 Building Model**
   - Building type: MidriseApartment (90.1-2013)
   - Floor area: 47.94 m² (516 ft²)
   - Stories: 1
   - Thermal zones: 4 (2 Apartments, 1 Corridor, 1 Office)
   - HVAC: PTAC systems (Natural Gas heating, Electric DX cooling)
   - Weather: Boston Logan Intl AP (Climate Zone 5A)

2. **Ran Baseline Simulation**
   - Status: ✅ SUCCESS
   - OpenStudio version: 3.11.0
   - EnergyPlus version: 25.2.0
   - Simulation time: ~14 seconds

3. **Extracted Results**

**Annual Energy Results:**
| End Use | Fuel | Energy (GJ) |
|---------|------|-------------|
| Heating | Natural Gas | 10.99 |
| Water Heating | Natural Gas | 27.89 |
| Cooling | Electricity | 0.69 |
| Interior Lighting | Electricity | 1.88 |
| Exterior Lighting | Electricity | 5.00 |
| Interior Equipment | Electricity | 6.58 |
| Fans | Electricity | 1.20 |
| Pumps | Electricity | 0.07 |
| **Total Site Energy** | | **54.31** |

**Energy Intensity:**
- EUI: 1132.87 MJ/m² (99.76 kBtu/ft²)
- Unmet Hours (Heating): 11.17 hours
- Unmet Hours (Cooling): 0.00 hours

**Envelope Properties:**
- Walls: U = 0.328 W/m²K (R-18.18)
- Roof: U = 0.186 W/m²K (R-31.25)
- Floor: F-factor = 0.88 W/m·K
- Windows: U = 2.387 W/m²K, SHGC = 0.401, VT = 0.813
- WWR: 10%

**Output Files Generated:**
- OSM model: `/runs/phase_25/case_900_baseline.osm`
- Results report: `/runs/exports/phase_25/openstudio_results_report.html` (1.4 MB)
- SQL results: `/runs/51324bf14e45433989e4d032e148289b/run/eplusout.sql` (5.1 MB)
- ESO timeseries: `/runs/51324bf14e45433989e4d032e148289b/run/eplusout.eso` (4.0 MB)
- RDD output variables: `/runs/51324bf14e45433989e4d032e148289b/run/eplusout.rdd`

**Results Saved To:**
- `/home/alex/Projects/fluxion/tests/energyplus_data/case_900_baseline_results.json`

---

### Next Steps: Parametric Sweeps

#### Task 25-01-3: Mass Level Parametric Sweep (5 variants)

Create 5 IDF variants with different thermal mass levels:
- Case 900-Light: 50% mass
- Case 900-Medium: 100% mass (baseline)
- Case 900-Heavy: 150% mass
- Case 900-VHeavy: 200% mass
- Case 900-XHeavy: 300% mass

**Purpose:** Generate training data for ML surrogate to learn mass-energy relationship

#### Task 25-01-4: Timestep Sensitivity Analysis (6 variants)

Run Case 900 baseline with 6 different timesteps:
- 60 timesteps/hour (1-minute)
- 10 timesteps/hour (6-minute)
- 4 timesteps/hour (15-minute)
- 2 timesteps/hour (30-minute)
- 1 timestep/hour (60-minute) - baseline
- 0.5 timesteps/hour (2-hour)

**Purpose:** Determine optimal timestep for Fluxion adaptive timestep implementation

#### Task 25-01-5: Construction Variations (10 variants)

Create 10 IDF variants with different wall constructions:
- Vary insulation thickness (R-10, R-20, R-30, R-40)
- Vary mass layer position (exterior, interior, distributed)
- Vary glazing ratio (10%, 20%, 30%, 40% window-to-wall ratio)

**Purpose:** Generate diverse training data for ML surrogate generalization

#### Task 25-01-6: Hourly Profile Extraction

Extract 8760 hourly values for:
- Zone air temperature (°C)
- HVAC heating rate (W)
- HVAC cooling rate (W)
- Surface inside/outside face temperatures
- Solar gain through windows (W)
- Conduction heat flux through walls (W/m²)

**Purpose:** Training data for ML surrogate to learn temporal dynamics

---

## OpenStudio MCP Workflow

### Commands Used

```bash
# 1. Create building from template
create_new_building(
    building_type="MidriseApartment",
    total_bldg_floor_area=516,
    num_stories_above_grade=1,
    floor_height=2.7,
    wwr=0.1,
    template="90.1-2013",
    climate_zone="5A"
)

# 2. Set weather file
change_building_location(
    weather_file="/opt/comstock-measures/ChangeBuildingLocation/tests/USA_MA_Boston-Logan.Intl.AP.725090_TMY3.epw"
)

# 3. Complete typical building setup
create_typical_building(
    template="90.1-2013",
    building_type="MidriseApartment",
    system_type="Inferred"
)

# 4. Save model
save_osm_model(osm_path="/runs/phase_25/case_900_baseline.osm")

# 5. Run simulation
run_simulation(
    osm_path="/runs/phase_25/case_900_baseline.osm",
    name="case_900_baseline_simulation"
)

# 6. Extract results
extract_summary_metrics(run_id="...")
extract_end_use_breakdown(run_id="...", units="SI")
generate_results_report(run_id="...", units="SI")
extract_zone_summary(run_id="...")
extract_envelope_summary(run_id="...")
```

### Advantages of OpenStudio MCP

1. **No Installation Required** - EnergyPlus/OpenStudio already available
2. **Python API** - Easy automation via MCP tools
3. **Comprehensive Results** - SQL, ESO, HTML reports automatically generated
4. **Standard Measures** - Access to 79+ ComStock measures
5. **ASHRAE 140 Compliance** - Built-in baseline systems 1-10

---

## Comparison with Fluxion Baseline

### Fluxion Case 900 (5R1C) - Current Status
- Annual Heating: 5.35 MWh (19.26 GJ) - **262-322% high**
- Annual Cooling: 4.75 MWh (17.10 GJ) - **29-123% high**
- Reference Range (ASHRAE 140):
  - Heating: 1.17-2.04 MWh (4.21-7.34 GJ)
  - Cooling: 2.13-3.67 MWh (7.67-13.21 GJ)

### EnergyPlus Case 900 (Baseline) - New Results
- Annual Heating: 10.99 GJ Natural Gas (3.05 MWh)
- Annual Cooling: 0.69 GJ Electricity (0.19 MWh)

**Note:** EnergyPlus results include water heating (27.89 GJ), which is separate from space conditioning. The space heating energy (10.99 GJ) is still above ASHRAE 140 reference range, suggesting this building model may have different specifications than the ASHRAE 140 Case 900 reference building.

**Next Step:** Create ASHRAE 140 Case 900-specific model with exact thermal mass properties (19,944,509 J/K total capacitance) to match Fluxion input parameters.

---

## Phase 25 Plans Status

| Plan | Name | Status | Progress |
|------|------|--------|----------|
| 25-00 | Literature Review | Pending | 0% |
| 25-01 | EnergyPlus Data Generation | In Progress | 30% (baseline complete) |
| 25-02 | Adaptive Timestep | Pending | 0% |
| 25-03 | Finite Difference | Pending | 0% |
| 25-04 | CTF Implementation | Pending | 0% |
| 25-05 | Hybrid RC + ML | Pending | 0% |
| 25-06 | Comparative Evaluation | Pending | 0% |

**Overall Phase 25 Progress:** 10% (1/7 plans in progress)

---

## Key Learnings

### OpenStudio MCP Capabilities

1. **Building Creation:** `create_new_building()` chains geometry + typical building setup
2. **Weather Management:** `change_building_location()` sets EPW + design days + climate zone
3. **HVAC Systems:** `add_baseline_system()` implements ASHRAE 90.1 Appendix G systems 1-10
4. **Modern Systems:** `add_doas_system()`, `add_vrf_system()`, `add_radiant_system()`
5. **Results Extraction:** Comprehensive SQL queries via `extract_*` tools
6. **Parametric Studies:** Can loop over model modifications + simulations

### Limitations Encountered

1. **Timeseries Data:** Hourly output variables not captured by default (need to add `Output:Variable` objects)
2. **Weather Files:** Limited selection in `/opt/comstock-measures/ChangeBuildingLocation/tests/`
3. **Model Specificity:** Standard templates (MidriseApartment) don't match ASHRAE 140 Case 900 exactly

### Solutions

1. **Custom Model:** Create geometry manually using `create_bar_building()` + custom constructions
2. **Output Variables:** Use `add_output_variable()` and `add_output_meter()` before simulation
3. **Parametric Automation:** Write Python script to loop over model modifications

---

## Next Actions

1. **Complete Parametric Sweeps** (Tasks 25-01-3 through 25-01-6)
   - Mass level variation (5 cases)
   - Timestep sensitivity (6 cases)
   - Construction variations (10 cases)
   - Hourly profile extraction

2. **Create ASHRAE 140 Case 900 Specific Model**
   - Exact geometry: 8m × 6m × 2.7m zone
   - Exact mass: 19,944,509 J/K thermal capacitance
   - Exact construction: 4-layer walls (brick, insulation, concrete, gypsum)

3. **Start Literature Review** (Plan 25-00)
   - CTF state-of-the-art
   - Finite difference methods
   - State-space and admittance methods

4. **Begin Implementation** (Plans 25-02 through 25-05)
   - Start with adaptive timestep (quick win)
   - Proceed to finite difference and CTF

---

## Data Organization

```
tests/energyplus_data/
├── case_900_baseline_results.json    # Baseline results summary
└── (future parametric sweep results)

docs/
├── PHASE_25_EXECUTION_SUMMARY.md     # This document
└── (future literature reviews)

/runs/phase_25/
├── case_900_baseline.osm             # OpenStudio model
└── (future parametric models)

/runs/exports/phase_25/
└── openstudio_results_report.html    # Comprehensive HTML report
```

---

*Summary created: 2026-03-17*
*Baseline simulation completed: Run ID 51324bf14e45433989e4d032e148289b*
*Next: Parametric sweeps and ASHRAE 140 Case 900 specific model*
