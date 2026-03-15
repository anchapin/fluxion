# Phase 20 Plan Breakdown

**Phase:** 20-data-quality-finalization
**Requirements:** 12 (PHYS-02, PHYS-03, PHYS-06, PHYS-07, WEATHER-01, WEATHER-03, WEATHER-04, WEATHER-05, DATA-02, DATA-03, DATA-04, DATA-05)
**Discovery Level:** Level 0 (Skip - all patterns established)

---

## Task Breakdown

### Wave 1: Foundation (Parallel Execution)

**Plan 20-01: Building Assembly System** (PHYS-02, PHYS-07)
- Task 1: Create MaterialLayer trait and concrete material implementations
- Task 2: Implement AssemblyBuilder for composing assemblies with validation
- Task 3: Load building assemblies from YAML configuration
- Task 4: Auto-calculate thermal mass classification (ISO 13790 Annex C)

**Plan 20-02: Constants Module** (PHYS-03)
- Task 1: Create domain-based constants module structure (thermal/, solar/, atmospheric.rs)
- Task 2: Implement ASHRAE 140 constants with version subfolders (v2021.rs, v2023.rs)
- Task 3: Document all constants with complete metadata (value, units, source, uncertainty, validity, assumptions)
- Task 4: Create derived constants for standard constructions (hybrid pre-calculated + computation functions)

**Plan 20-03: Extended Weather Parsing** (WEATHER-01, WEATHER-04)
- Task 1: Extend EPW parsing for v3, AMY, IWEC formats
- Task 2: Add missing weather fields (ground temp, illuminance, snow, present weather)
- Task 3: Implement TMY3 download infrastructure with caching
- Task 4: Create weather location metadata database

### Wave 2: Advanced Features (Parallel Execution)

**Plan 20-04: Weather Interpolation & Sky Model** (WEATHER-03, WEATHER-05)
- Task 1: Implement sub-hourly interpolation functions (linear, piecewise hermite, step)
- Task 2: Implement clearness index calculation (kt = GHI / GHI_clear)
- Task 3: Integrate cloud cover effects with sky emissivity
- Task 4: Validate interpolation accuracy against ASHRAE 140 cases

**Plan 20-05: 8R3C Thermal Network Evaluation** (PHYS-06)
- Task 1: Implement 8R3C thermal network structure (8 resistance, 3 capacitance nodes)
- Task 2: Create evaluation tests against ASHRAE 140 high-mass cases (Case 920, Case 960)
- Task 3: Compare accuracy vs 5R1C baseline
- Task 4: Document findings and recommendation (keep 5R1C if no improvement, per Phase 12 pattern)

### Wave 3: Integration & Validation (Sequential)

**Plan 20-06: Configuration Validation** (DATA-04)
- Task 1: Create validation error types and structures (Structured JSON)
- Task 2: Implement assembly validation (bounds, cross-field consistency)
- Task 3: Implement constants validation (units, ranges)
- Task 4: Integrate validation into ThermalModel initialization

**Plan 20-07: Mock Data Replacement** (DATA-02, DATA-03)
- Task 1: Replace mock predictions in AI modules (batch_inference.rs, distributed.rs, ensemble.rs)
- Task 2: Replace hardcoded physical constants with constants module references
- Task 3: Replace hardcoded material properties with building assembly system
- Task 4: Verify all production code paths use real data

**Plan 20-08: Documentation & Finalization** (DATA-05)
- Task 1: Complete docstring documentation for all physical parameters
- Task 2: Create PHYSICAL_CONSTANTS.md reference document
- Task 3: Validate all parameters against ASHRAE 140 and ISO 13790 sources
- Task 4: Run comprehensive validation suite and generate final report

---

## Dependency Graph

```
Wave 1 (Parallel):
  20-01 (Building Assembly) → Creates: MaterialLayer trait, AssemblyBuilder, assemblies.yaml
  20-02 (Constants) → Creates: Constants module structure
  20-03 (Weather Parsing) → Creates: Extended EPW parser, TMY3 downloader

Wave 2 (Parallel, depends on Wave 1):
  20-04 (Interpolation) → Needs: Weather data structure from 20-03
  20-05 (8R3C Evaluation) → Independent, can run in parallel

Wave 3 (Sequential, depends on Wave 1+2):
  20-06 (Validation) → Needs: Assembly system from 20-01, Constants from 20-02
  20-07 (Mock Replacement) → Needs: Assembly from 20-01, Constants from 20-02
  20-08 (Documentation) → Needs: All previous work complete
```

---

## Wave Structure

| Wave | Plans | Parallel? | Requirements |
|------|-------|-----------|--------------|
| 1 | 20-01, 20-02, 20-03 | Yes | PHYS-02, PHYS-03, PHYS-07, WEATHER-01, WEATHER-04 |
| 2 | 20-04, 20-05 | Yes | WEATHER-03, WEATHER-05, PHYS-06 |
| 3 | 20-06, 20-07, 20-08 | No (sequential) | DATA-02, DATA-03, DATA-04, DATA-05 |

---

## Plan Details

### Plan 20-01: Building Assembly System
**Requirements:** PHYS-02, PHYS-07
**Wave:** 1
**Dependencies:** None
**Files Modified:**
- src/sim/assembly.rs (new)
- data/assemblies.yaml (new)
- data/materials.yaml (new)
- src/sim/mod.rs

### Plan 20-02: Constants Module
**Requirements:** PHYS-03
**Wave:** 1
**Dependencies:** None
**Files Modified:**
- src/physics/constants/ (new directory)
- src/physics/constants/thermal/ashrae_140/v2021.rs
- src/physics/constants/thermal/ashrae_140/v2023.rs
- src/physics/constants/thermal/iso_13790/annex_c.rs
- src/physics/constants/solar/ashrae_140.rs
- src/physics/constants/atmospheric.rs
- src/physics/mod.rs

### Plan 20-03: Extended Weather Parsing
**Requirements:** WEATHER-01, WEATHER-04
**Wave:** 1
**Dependencies:** None
**Files Modified:**
- src/weather/epw.rs (extend)
- src/weather/tmy3.rs (new)
- data/weather_locations.json (new)
- Cargo.toml (add reqwest, chrono, directories dependencies)

### Plan 20-04: Weather Interpolation & Sky Model
**Requirements:** WEATHER-03, WEATHER-05
**Wave:** 2
**Dependencies:** 20-03
**Files Modified:**
- src/weather/interpolation.rs (new)
- src/weather/mod.rs
- src/sim/sky_radiation.rs (extend)

### Plan 20-05: 8R3C Thermal Network Evaluation
**Requirements:** PHYS-06
**Wave:** 2
**Dependencies:** None
**Files Modified:**
- src/sim/engine.rs (extend)
- tests/test_8r3c_evaluation.rs (new)

### Plan 20-06: Configuration Validation
**Requirements:** DATA-04
**Wave:** 3
**Dependencies:** 20-01, 20-02
**Files Modified:**
- src/validation/config.rs (new)
- src/validation/mod.rs
- src/sim/engine.rs (integrate validation)

### Plan 20-07: Mock Data Replacement
**Requirements:** DATA-02, DATA-03
**Wave:** 3
**Dependencies:** 20-01, 20-02
**Files Modified:**
- src/ai/surrogate/batch_inference.rs (remove mocks)
- src/ai/surrogate/distributed.rs (remove mocks)
- src/ai/surrogate/ensemble.rs (remove mocks)
- src/sim/engine.rs (replace hardcoded values)

### Plan 20-08: Documentation & Finalization
**Requirements:** DATA-05
**Wave:** 3
**Dependencies:** All previous plans
**Files Modified:**
- docs/PHYSICAL_CONSTANTS.md (new)
- Various source files (add docstrings)
- .planning/phases/20-data-quality-finalization/20-SUMMARY.md

---

## Context Budget Analysis

- **Wave 1 (3 plans, ~15% each):** ~45% total
  - 20-01: Building assembly system (complex trait + JSON config)
  - 20-02: Constants module (domain structure + documentation)
  - 20-03: Weather parsing (EPW extensions + TMY3 download)
- **Wave 2 (2 plans, ~20% each):** ~40% total
  - 20-04: Interpolation + sky model (mathematical functions)
  - 20-05: 8R3C evaluation (thermal network + testing)
- **Wave 3 (3 plans, ~5% each):** ~15% total
  - 20-06: Validation (error structures + functions)
  - 20-07: Mock replacement (straightforward replacements)
  - 20-08: Documentation (docstrings + reference doc)

**Total:** ~100% context across 3 waves, each wave stays within budget.

---

## Verification Strategy

**Per Wave:**
- Wave 1: Unit tests for each module (assembly, constants, weather)
- Wave 2: Integration tests (interpolation accuracy, 8R3C comparison)
- Wave 3: Full validation suite + mock data audit

**Phase Gate:**
- All 12 requirements addressed in at least one plan
- No mock data in production code paths
- All configurations validated at load time
- All physical parameters documented with source references
- Comprehensive validation report generated

---

## Next Steps

1. **Execute Wave 1:** Run plans 20-01, 20-02, 20-03 in parallel
2. **Execute Wave 2:** After Wave 1 complete, run plans 20-04, 20-05 in parallel
3. **Execute Wave 3:** After Wave 2 complete, run plans 20-06, 20-07, 20-08 sequentially
4. **Phase Complete:** All 8 plans executed, validation report generated, v0.4 milestone complete
