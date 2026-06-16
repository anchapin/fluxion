# ASHRAE 140 Section 8 Output Gaps

**Standard:** ASHRAE Standard 140-2023 Section 8 (Output Specifications)
**Related Issue:** #762, #749
**Analysis Date:** 2026-06-16
**Status:** IncidentSolar metric type implemented

---

## Overview

ASHRAE 140 Section 8 defines required output metrics for inter-program comparison. This document tracks gaps between the standard's requirements and Fluxion's implementation, specifically focusing on per-surface solar radiation reporting.

---

## Section 8.2.3 — Per-Surface Solar Radiation

| Requirement | Status | Notes |
|------------|--------|-------|
| Annual incident solar per orientation | IMPLEMENTED | `MetricType::IncidentSolar` added |
| Peak incident solar per orientation | NOT STARTED | Future enhancement |
| Surface-specific breakdown (N/S/E/W/Roof) | IMPLEMENTED | Via `surface_id` and `Orientation` fields |

---

## Metric Type Implementation

### IncidentSolar Variant

```rust
// src/validation/report.rs
pub enum MetricType {
    // ... other variants ...
    /// Incident solar radiation per surface orientation (kWh/m²).
    /// Per ASHRAE 140-2023 Section 8.2.3, outputs annual and peak solar per orientation.
    IncidentSolar {
        /// Surface identifier (e.g., "roof", "N", "S", "E", "W")
        surface_id: String,
        /// Surface orientation
        orientation: crate::validation::ashrae_140_cases::Orientation,
    },
}
```

### ValidationResult Integration

The `IncidentSolar` metric type is integrated into the validation system:

- **Display name:** "Incident Solar Radiation (kWh/m²)"
- **Units:** kWh/m²
- **Reference range:** Returns `None` (per-surface solar has no inter-program reference)
- **Sorting:** Alphabetically after `MaxFreeFloat` variant

---

## Gap Analysis

### Completed Gaps

| Gap ID | Description | Resolution | Issue |
|--------|-------------|------------|-------|
| GAP-8.2.3-001 | No IncidentSolar metric type existed | Added `MetricType::IncidentSolar` variant | #762 |

### Open Gaps

| Gap ID | Description | Priority | Notes |
|--------|-------------|----------|-------|
| GAP-8.2.3-002 | Peak incident solar per orientation not tracked | P2 | Future enhancement |
| GAP-8.2.3-003 | No reference data for solar validation | N/A | Per-surface solar has no ASHRAE reference ranges |

---

## Cross-References

- **Section 8 Output Spec:** `docs/ashrae_140/section_8_output.md`
- **MetricType Definition:** `src/validation/report.rs:83-104`
- **ValidationResult Structure:** `src/validation/report.rs:471-530`
- **ASHRAE 140 Cases:** `src/validation/ashrae_140_cases.rs`

---

## Related Issues

- #762 — Add IncidentSolar metric type for per-surface solar radiation reporting (this issue)
- #749 — ASHRAE 140 Section 8 compliance overall tracking issue
