---
phase: 17-internal-loads
plan: 03
subsystem: profile-loading
tags: [profiles, internal-loads, json, caching]
dependency_graph:
  requires:
    - "17-01: Weekly schedule support"
    - "17-02: Equipment trait and implementations"
  provides:
    - "ProfileBundle for building profile bundling"
    - "load_building_profile() for JSON-based profile loading"
  affects:
    - "ASHRAE 140 Cases 600-960 (internal loads)"
tech_stack:
  added:
    - "ProfileBundle struct with lighting/equipment/occupancy"
    - "JSON-based profile loading with serde"
    - "OnceLock caching for performance"
  patterns:
    - "Trait object downcasting with as_any()"
    - "Manual Clone implementation for trait objects"
key_files:
  created:
    - "src/sim/profiles.rs: ProfileBundle, load_building_profile()"
    - "data/building_profiles.json: Office/Retail/School profiles"
  modified:
    - "src/sim/mod.rs: Added profiles module export"
    - "src/sim/equipment.rs: Extended Equipment trait with as_any()"
    - "src/sim/occupancy.rs: Added Hash to BuildingType"
key_decisions:
  - "Manual Clone implementation for ProfileBundle using trait object downcasting"
  - "Daily/constant schedule types for equipment from JSON"
  - "Hash trait added to BuildingType for HashMap usage in caching"
  - "Default daily schedule: 8am-6pm active hours"
metrics:
  duration: "2m 45s"
  completed_date: "2026-03-14"
  tasks_completed: 3
  files_modified: 4
  tests_added: 4
  tests_passing: 4
---

# Phase 17 Plan 03: Building Profile Loading Summary

Building profile loading with JSON-based defaults and caching implemented for Office, Retail, and School building types, enabling realistic internal load profiles with weekday/weekend patterns essential for ASHRAE 140 Cases 600-960 compliance.

## Implementation Overview

Successfully implemented ProfileBundle struct and load_building_profile() function with JSON-based profile loading, OnceLock caching to avoid repeated file I/O, and comprehensive test coverage. The implementation enables loading pre-configured building profiles without recompilation, supporting multiple equipment types with schedule-based time-varying power calculations.

## Key Components

### ProfileBundle Struct
- Bundles lighting, equipment, and occupancy profiles for a building type
- Implements manual Clone using trait object downcasting via as_any() method
- Provides Debug implementation showing equipment count instead of full equipment list

### JSON Profile Loading
- Deserializes from data/building_profiles.json with serde
- Supports Office, Retail, and School building types
- Maps JSON equipment types to concrete Rust types (ComputerEquipment, ServerRack, GenericEquipment)
- Applies schedule_type logic (daily: 8am-6pm, constant: 24/7)
- Error handling for missing profiles and unknown equipment types

### Caching Strategy
- Uses OnceLock<HashMap<BuildingType, ProfileBundle>> for thread-safe lazy initialization
- Checks cache before file I/O, loads once per building type
- Significant performance improvement for repeated profile loading

### Building Profiles
**Office Profile:**
- Lighting: 10 W/m² power density, 20% convective/80% radiative
- Equipment: 50 computers (150W each), 2 server racks (500W each)
- Occupancy: 100 max occupants with office schedule

**Retail Profile:**
- Lighting: 12 W/m² power density, 30% convective/70% radiative
- Equipment: 5 POS systems (200W each)
- Occupancy: 50 max occupants with retail schedule

**School Profile:**
- Lighting: 8 W/m² power density, 25% convective/75% radiative
- Equipment: 30 classroom computers (150W each)
- Occupancy: 200 max occupants with school schedule

## Technical Challenges Resolved

### Trait Object Clone Implementation
ProfileBundle contains `Vec<Box<dyn Equipment + Send + Sync>>` which cannot derive Clone. Solved by:
1. Extending Equipment trait with as_any() method returning &dyn Any
2. Implementing manual Clone for ProfileBundle
3. Downcasting to concrete types (ComputerEquipment, ServerRack, GenericEquipment) for cloning
4. Panic on unknown equipment type (should never occur with valid JSON)

### Thread-Safe Caching
ProfileBundle contains trait objects which require Send + Sync bounds. Resolved by:
1. Adding Send + Sync bounds to all trait object references
2. Using OnceLock for thread-safe lazy initialization
3. Ensuring all Equipment implementations are Send + Sync (automatic via derived traits)

### HashMap Key Requirement
BuildingType needed Hash trait for use as HashMap key. Added Hash to BuildingType enum derive attributes.

## Test Coverage

### test_profile_bundle_struct
Verifies ProfileBundle struct compiles correctly with all required fields.

### test_building_profile_loading
Validates loading of all three building types:
- Verifies lighting power density and heat fractions
- Checks equipment count per profile
- Confirms occupancy max_occupancy values

### test_profile_caching
Tests that OnceLock caching works correctly by loading same profile twice and verifying second load succeeds without file I/O errors.

### test_equipment_in_profile
Verifies equipment power calculations:
- Confirms zero power at midnight (off hours)
- Confirms positive power during active hours (8am-6pm)

All 4 tests passing.

## Deviations from Plan

None - plan executed exactly as written.

## Next Steps

Profile loading infrastructure is complete and ready for integration with ThermalModel for ASHRAE 140 Cases 600-960. Next plans will focus on:
- Profile 17-04: Integrate profiles with thermal model simulation
- Phase 18: Diagnostic cases (195-470, 800-810) using internal loads

## Files Modified

### Created
- src/sim/profiles.rs (308 lines): ProfileBundle, load_building_profile, tests
- data/building_profiles.json (60 lines): Office, Retail, School profiles

### Modified
- src/sim/mod.rs: Added profiles module export
- src/sim/equipment.rs: Extended Equipment trait with as_any() method
- src/sim/occupancy.rs: Added Hash to BuildingType enum

## Commits

1. b2e029a: feat(17-03): create ProfileBundle struct and profile loading module
   - ProfileBundle struct with lighting, equipment, occupancy
   - load_building_profile() with JSON deserialization
   - OnceLock caching for performance
   - Equipment trait extended with as_any() for downcasting
   - Manual Clone implementation for trait objects
   - Hash trait added to BuildingType

2. 81c399f: feat(17-03): create default building profiles JSON file
   - Office profile with computers and servers
   - Retail profile with POS systems
   - School profile with classroom computers
   - schedule_type logic (daily vs constant)
   - Default daily schedule: 8am-6pm active
   - All profile tests passing

## Success Criteria Verification

- [x] ProfileBundle struct contains lighting, equipment, and occupancy profiles
- [x] load_building_profile function loads from JSON file for Office, Retail, School
- [x] Profile caching uses OnceLock to avoid repeated file I/O
- [x] Building profiles JSON contains realistic defaults for three building types
- [x] Equipment profiles deserialize correctly with proper thermal characteristics
- [x] Lighting profiles use convective/radiative fractions from JSON
- [x] Occupancy profiles use max_occupancy and building-specific schedules
- [x] All tests pass with cargo test --package fluxion --lib profiles::

## Self-Check: PASSED

All implementation files exist and compile successfully. All commits verified in git log. All tests passing.
