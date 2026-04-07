# Phase M2: Zone-Level HVAC Controls

**Milestone:** v1.0 (Multi-Zone Support)
**Phase:** M2
**Status:** Planning
**Created:** 2026-04-07

---

## Overview

Phase M2 implements zone-level HVAC controls for the multi-zone thermal network established in Phase M1. This phase focuses on adding per-zone heating/cooling setpoints, independent HVAC control logic, and extending the Python API and CLI to support multi-zone HVAC operations.

---

## Goals

- Implement zone-specific heating/cooling setpoints with deadband control
- Develop independent HVAC control logic for each thermal zone
- Extend Python API bindings for multi-zone HVAC functionality
- Add CLI commands for multi-zone HVAC simulation and control
- Validate zone-level HVAC behavior and integration

---

## Requirements Addressed

- **MZ-03:** Zone-Specific HVAC Setpoints
- **MZ-04:** Zone-Level HVAC Control
- **MZ-09:** Python API Multi-Zone
- **MZ-10:** CLI Multi-Zone

---

## Research Context

Based on research in:
- `.planning/research/ARCHITECTURE_MULTI_ZONE.md`
- `.planning/research/FEATURES_MULTI_ZONE.md`

Key architectural patterns:
- Per-zone HVAC setpoints with deadband control
- Independent HVAC control loops for each zone
- Integration with existing multi-zone thermal model
- Extension of Python API and CLI interfaces

---

## Critical Pitfalls to Avoid

- **Pitfall 1:** Inconsistent setpoint application across zones
- **Pitfall 2:** HVAC control logic interfering with thermal calculations
- **Pitfall 3:** Python API binding mismatches with Rust implementation
- **Pitfall 4:** CLI command conflicts with existing single-zone commands

---

## Success Criteria

- Zone-specific setpoints can be configured and applied independently
- HVAC control logic maintains zone temperatures within deadband
- Python API exposes all multi-zone HVAC functionality
- CLI supports multi-zone HVAC simulation and control
- Integration tests validate end-to-end functionality

---

## Next Steps

1. Create executable plans for M2 implementation
2. Execute plans to build zone-level HVAC controls
3. Validate against integration tests
4. Document architecture and usage patterns
