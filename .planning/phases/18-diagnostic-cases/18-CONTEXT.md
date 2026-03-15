# Phase 18: Diagnostic Cases - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement comprehensive diagnostic case coverage for in-depth validation, building on Phase 15 (HVAC equipment) and Phase 17 (internal loads) to validate full engine with ASHRAE 140 Cases 195-470, 800-810, and non-residential diagnostic variants.

**What this delivers:**
- ASHRAE 140 Cases 195-470 (in-depth diagnostics) for comprehensive testing of specific components and scenarios
- ASHRAE 140 Cases 800-810 (HVAC equipment) to validate equipment efficiency and control strategies
- Non-residential cases from ASHRAE 140 to extend validation beyond residential buildings
- Solid conduction and solar gain diagnostic variants to expose edge cases and validate specific physics components

This phase enhances validation completeness — covers diagnostic cases that test specific components and edge scenarios beyond the baseline Cases 600-960 residential suite.

</domain>

---

<decisions>
## Implementation Decisions

### Case File Organization

**Approach:** Hybrid structure with consolidated logic and public case specs

- Create `tests/ashrae_140/diagnostics.rs` module containing:
  - Consolidated validation logic for all diagnostic cases
  - Helper functions shared across diagnostic ranges
  - Integration with existing `ASHRAE140Validator` framework
- Keep case specification functions public in `src/validation/ashrae_140_cases.rs`:
  - Each diagnostic case (195-470, 800-810, variants) has its own spec function
  - Functions return `CaseSpec` or `CaseModel` for easy importing
  - Case spec data loaded from `docs/ashrae_140_references.json` (multi-reference DB)
- Balance: Centralized validation logic (easier to maintain) with accessible case spec functions (for direct use and testing)

**Rationale:** Hybrid approach balances organization (consolidated validation reduces file count and centralizes logic) with accessibility (public case spec functions allow direct importing without navigating large modules). This scales better than pure separation (many files) or pure consolidation (one giant file).

### Case Specification Format

**Approach:** Multi-reference database (docs/ashrae_140_references.json)

- Case parameters loaded from external JSON files via `MultiReferenceDB`
- Case specs reference: `docs/ashrae_140_references.json` (Phase 7 multi-reference integration)
- Diagnostic cases query multi-ref DB for reference ranges (when ASHRAE 140 official specs available)
- Fallback to sensible defaults when official specs not available
- Follows existing pattern from Phase 17 (building_profiles.json)

**Rationale:** Multi-reference DB (Phase 7) provides centralized, version-controlled ASHRAE 140 specifications when available. This is more maintainable than hardcoded constants and supports official ASHRAE data. Matches existing building profile pattern from Phase 17.

### Validation Strategy

**Approach:** Smart validation with diagnostic awareness

- Auto-discovery for baseline cases (600-960): `fluxion validate` auto-runs complete baseline suite
- Targeted re-run for affected case ranges when diagnostics added: `fluxion validate-case 195-470` or `fluxion validate-case 800-810`
- Diagnostic-aware validation: Validator tracks which cases have been added and only re-runs affected ranges
- Full re-run option available: `fluxion validate --full` for comprehensive validation

**Rationale:** Smart validation balances speed (baseline auto-runs, selective re-runs) with coverage (diagnostic awareness prevents gaps). Baseline Cases 600-960 remain auto-discovered (convenient for users), new diagnostic ranges are explicit (developer control). Full re-run option available for complete validation when needed.

### CLI Integration

**Approach:** Both auto-discovery and explicit invocation

- Auto-discovery for baseline Cases 600-960: `fluxion validate` automatically discovers and runs all diagnostic cases in ranges
- Explicit invocation for new diagnostic ranges: `fluxion validate-case` allows running specific cases (e.g., `fluxion validate-case 800`)
- Subcommands: `fluxion validate 195-470`, `fluxion validate 800-810` for specific diagnostic ranges
- Consistent with existing pattern: Matches `fluxion validate-600` pattern (baseline validation)

**Rationale:** Both approaches provide maximum flexibility. Auto-discovery for baseline cases (convenient for users), explicit invocation for new diagnostic cases (developer control, selective testing). Maintains consistency with existing validation CLI.

### Claude's Discretion

- Exact JSON schema for multi-reference DB structure (field names, types)
- Diagnostic module organization details (module structure, helper function naming)
- Smart validation re-run trigger thresholds (how many cases before auto-discovery re-runs baseline)
- CLI subcommand design (validate-case flags, range argument format)
- Test framework patterns (property tests, integration tests, validation structure)

</decisions>

---

<code_context>
## Existing Code Insights

### Reusable Assets

**ASHRAE140Validator (src/validation/ashrae_140_validator.rs):**
- Existing validation framework with tolerance checking, diagnostic output, multi-reference comparison
- Supports per-program validation with `DiagnosticConfig` (full/silent/summary)
- `ASHRAE140Validator::create_hvac_controller()` creates controllers from case specs
- Reusable: Extend for diagnostic cases by adding case spec models to `CaseSpec` enum

**Case600Model (src/validation/ashrae_140_cases.rs):**
- Baseline case structure with reference ranges (EnergyPlus, ESP-r, TRNSYS)
- Pattern: Case struct with reference constants + `simulate_year()` method
- Reusable: Extend for diagnostic cases as `CaseModel` trait

**DiagnosticCollector (src/validation/diagnostics/):**
- Existing diagnostic logging framework with simulation diagnostics, HVAC tracking
- Outputs detailed reports with HourlyData, PeakTiming, TemperatureProfile
- Reusable: Use for detailed diagnostic case output

**MultiReferenceDB (src/validation/multi_reference.rs):**
- Phase 7 integration already implemented
- Loads reference data from `docs/ashrae_140_references.json`
- Provides lookup by program and case number
- Reusable: Use for loading diagnostic case specifications

**Existing test stubs:**
- `tests/ashrae_140_cases_800_810.rs` has TODO markers (needs ASHRAE 140 specs)
- Baseline case validation already works (`tests/ashrae_140_case_600.rs`)

### Established Patterns

**Validation-driven development (Phase 14-17):**
- Address accuracy before optimization
- Validate against ASHRAE 140 reference ranges with strict tolerances
- Apply same principle: validate diagnostic case implementations against reference ranges

**Trait-based abstractions (from Phase 15-16):**
- Codebase uses traits for common behavior across implementations
- Apply same pattern to diagnostic case models (CaseModel, CaseSpec traits)
- Supports code reuse and consistent testing

**Test organization:**
- Tests organized in `tests/ashrae_140/` directory with dedicated files per case range
- Baseline cases 600-960: dedicated test files (e.g., `ashrae_140_case_600.rs`)
- Integration tests: `tests/validation/` directory for framework testing
- Modular approach allows selective execution and easy maintenance

**Multi-reference integration pattern (Phase 7):**
- External JSON data in `docs/` directory
- Loaded via `MultiReferenceDB::from_file()`
- Version-controlled, easy to update without code changes
- Apply same pattern to diagnostic case specifications

### Integration Points

**Where diagnostic module lives:**
- `tests/ashrae_140/diagnostics.rs` — New module for diagnostic test consolidation
- Add helper functions shared across diagnostic ranges
- Integrate with `ASHRAE140Validator` for smart re-run logic

**Where case spec functions live:**
- `src/validation/ashrae_140_cases.rs` — Extend with diagnostic case spec functions
- Add functions for Cases 195-470, Cases 800-810, and variants
- Each function returns `CaseSpec` or `CaseModel` implementing traits

**Where multi-reference data lives:**
- `docs/ashrae_140_references.json` — Add diagnostic case reference ranges
- Structure: Follow existing multi-reference DB pattern from Phase 7
- Extend with Cases 195-470, 800-810, and variant specifications

**Where CLI integration happens:**
- `src/bin/fluxion.rs` or CLI module (`src/cli.rs` if exists)
- Extend `validate` subcommand with diagnostic case options:
  - `fluxion validate` (auto-discover baseline + diagnostics)
  - `fluxion validate-case` (explicit case invocation)
  - `fluxion validate 195-470` (specific range)
  - `fluxion validate 800-810` (specific range)

**Where diagnostic tests live:**
- `tests/ashrae_140/diagnostics.rs` — Module-level tests for consolidated validation
- `tests/ashrae_140_case_195_470.rs` — Test for Cases 195-470 range
- `tests/ashrae_140_case_800_810.rs` — Test for Cases 800-810 range
- `tests/ashrae_140_case_195_solid_conduction.rs` — Test for solid conduction variant
- `tests/ashrae_140_case_600_960.rs` — Integration test for baseline + diagnostics

</code_context>

---

<specifics>
## Specific Ideas

**Diagnostic case ranges following ASHRAE 140 specification:**
- Cases 195-470: In-depth diagnostics for specific component testing (lighting, equipment, thermal mass behavior)
- Cases 800-810: HVAC equipment validation (heat pumps, chillers, boilers, variable capacity control)
- Non-residential cases: Extend validation beyond standard office/lightweight building assumptions
- Solid conduction variants: High-mass wall effects, inter-zone conduction, thermal bridge effects
- Solar gain variants: Zero solar, diffuse solar, different albedo values

**Smart validation re-run thresholds (example values):**
- After adding Cases 195-470: Auto-re-run baseline + cases 195-470 range (20-30 new cases)
- After adding Cases 800-810: Auto-re-run baseline + cases 800-810 range (11-20 new cases)
- Threshold tuning based on iteration feedback: Start conservative (re-run more), adjust based on typical development cycle

**CLI usage examples:**
- `fluxion validate` — Auto-discover and run all diagnostic cases
- `fluxion validate-case 800` — Run only Cases 800-810 range
- `fluxion validate-case 195-470` — Run only Cases 195-470 range
- `fluxion validate --full` — Full re-run of all cases (baseline + diagnostics)

</specifics>

---

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. All decisions relate to case organization, specification format, validation strategy, and CLI integration as defined in Phase 18 requirements (DIAG-01 through DIAG-05).

</deferred>

---

*Phase: 18-diagnostic-cases*
*Context gathered: 2026-03-14*
