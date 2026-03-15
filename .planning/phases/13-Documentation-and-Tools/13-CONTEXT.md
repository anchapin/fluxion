# Phase 13: Documentation & Tools - Context

**Gathered:** 2026-03-13
**Status:** Ready for planning

---

<domain>
## Phase Boundary

Update documentation and enhance ASHRAE validation tooling. This phase improves developer experience and validation workflows—no breaking changes to BatchOracle or Model APIs.

**What this delivers:**
- Updated KNOWN_LIMITATIONS.md with Case 960 and 6R2C findings from v0.2
- Comprehensive API reference for BatchOracle and Model with usage examples
- Tutorial: "Extending Fluxion with custom thermal models"
- Documentation of parameter vector semantics and design variable bounds
- Rust-doc improvements for public modules
- Contribution guide enhancements
- ASHRAE validation report improvements (interpretation guidance, monthly aggregation fixes, delta testing enhancements, sensitivity analysis normalization)
- Step-by-step guide for adding new ASHRAE reference cases

</domain>

---

<decisions>
## Implementation Decisions

### Documentation Structure

**Decision:** Mixed approach — README.md for high-level entry point + docs/ for comprehensive organized content.

**Rationale:**
- README.md serves as project overview and installation guide (user-facing)
- docs/ organized by topic (architecture, API, validation, etc.) for discoverable reference
- Both locations have clear purposes: README = onboarding, docs/ = deep reference

**Content organization:**
- API reference → docs/API_REFERENCE.md (comprehensive)
- Tutorials → docs/tutorials/ (individual tutorial files)
- Validation reports → docs/ASHRAE_140_RESULTS.md (existing)
- Architecture → docs/ARCHITECTURE.md (existing)

### Documentation Format

**Decision:** Markdown + HTML hybrid approach.

**Rationale:**
- Markdown for developer docs (API_REFERENCE.md, ARCHITECTURE.md, etc.) — simple, version-controllable, works offline
- HTML for tutorials — better code highlighting, interactivity, searchable via build step
- Use Askama templates for HTML generation
- Keep source Markdown for easy editing, generate HTML as build artifact

### API Reference Depth

**Decision:** Comprehensive reference covering all patterns.

**Rationale:**
- Phase 7 established documentation-driven approach
- Quick start examples reduce onboarding friction
- Advanced patterns section provides depth for power users
- Progressive complexity: basic examples → advanced usage → debugging patterns

**Structure:**
- BatchOracle: Quick start (5 examples), Advanced usage (GPU batching, custom parameters, error recovery)
- Model: Full parameter documentation, Analytical focus, Debugging focus
- Cross-link to comprehensive sections for advanced topics

### Tutorial Format

**Decision:** Start-to-finish Markdown walkthrough with complete working example.

**Rationale:**
- User chose comprehensive approach
- Shows complete thermal model structure, extension points, and integration
- Easier to maintain than Jupyter (no runtime dependencies)
- Include step-by-step instructions with code blocks
- Provide complete working example: `examples/tutorial_custom_model.rs`

**Tutorial outline:**
1. Project overview (BatchOracle vs Model, thermal networks)
2. Thermal model structure (zones, 5R1C, conductances)
3. Implementing `heat_transfer()` method
4. Integration with `SimulationDiagnostics`
5. Complete working example with explanation

### ASHRAE Report Clarity

**Decision:** Comprehensive analysis with detailed interpretation guidance for failed metrics.

**Rationale:**
- Success criteria #1 mentions "interpretation guidance" for failed metrics
- Helps users understand what happened and how to investigate
- Include root cause hypotheses, parameter sensitivity analysis, recommended fixes
- What-if scenarios for debugging approaches

**Interpretation guidance structure:**
1. Root cause hypotheses (what physics or parameters might be causing the issue)
2. Parameter sensitivity analysis (which inputs most affect the failing metric)
3. Recommended next steps (fixes to try, data to collect)
4. What-if scenarios (if we change X, what happens to metric Y)
5. Links to relevant documentation (KNOWN_LIMITATIONS.md, Phase 8 investigation results)

### Claude's Discretion

HTML template library selection (Askama vs tera vs manual) — choose best for maintainability and feature set
Exact tutorial section breakdown and code example file naming
Rust-doc coverage priorities (which modules most need documentation)
Monthly aggregation implementation approach for ASHRAE140_RESULTS.md generator
Statistical testing library for delta testing (if adding p-values)
Normalization formula for sensitivity analysis metrics
Update frequency for KNOWN_LIMITATIONS.md (per release vs only on findings)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets

**Documentation infrastructure:**
- `docs/CONTRIBUTING.md` — comprehensive contribution guide with CI setup, testing strategy, commit conventions
- `docs/KNOWN_LIMITATIONS.md` — detailed 5R1C and 6R2C limitation analysis from Phase 8
- `docs/ARCHITECTURE.md` — architecture deep dive
- `docs/API_REFERENCE.md` — partial API reference (can be expanded)
- `docs/ASHRAE_140_RESULTS.md` — validation report generator (can be enhanced)
- `docs/ASHRAE140_TERMINOLOGY.md` — terminology reference

**Examples for tutorial:**
- `examples/validate_6r2c.rs` — recent example showing validation comparison (similar pattern to tutorial)
- `examples/construction_example.rs` — construction spec example
- `examples/quick_start.sh` — setup script

**Validation framework:**
- `src/validation/` directory with ASHRAE 140 case specs and analyzer
- `ASHRAE140Validator` for multi-reference comparison
- Benchmark report generation with clear sections

**Documentation generation:**
- Askama templates (if HTML generation needed)
- Markdown generation via Rust docs (/// comments)
- Validation report generator can be extended

### Established Patterns

**Documentation-driven approach (Phase 7):**
- Quality tutorials and API references are key deliverables
- Comprehensive examples with explanations, not just code
- Step-by-step walkthroughs for complex topics

**Web-based visualization for reports (Phase 7):**
- Chart.js, D3.js, or Plotly.js chosen for interactive validation reports
- HTML generation with embedded data and JavaScript

**Developer experience:**
- CONTRIBUTING.md covers local development (cargo fmt, clippy, test)
- CI with `act` for local workflow testing
- Clear commit message convention

**ASHRAE validation pattern:**
- Toleranced comparison (Pass/Warning/Fail) with ±5% bands
- Per-program reference tracking (BenchmarkData stores multiple sources)
- Report generation with interpretation sections

### Integration Points

**Where new documentation lives:**
- `docs/API_REFERENCE.md` — expand with BatchOracle and Model comprehensive documentation
- `docs/tutorials/` — new directory for organized tutorials
- `docs/KNOWN_LIMITATIONS.md` — update with 6R2C findings and Case 960 status
- `docs/ASHRAE_140_RESULTS.md` — enhance with interpretation guidance and fix monthly aggregation

**Where new tutorial example lives:**
- `examples/tutorial_custom_model.rs` — complete working thermal model extension example

**Where ASHRAE enhancements integrate:**
- `src/validation/ashrae_140_validator.rs` — extend for enhanced interpretation
- `src/validation/ashrae_140_results.rs` — fix monthly aggregation, add delta testing support, improve sensitivity analysis output

**CLI enhancements:**
- `src/bin/fluxion.rs` — new subcommands for validation workflows if needed

</code_context>

---

<specifics>
## Specific Ideas

**Documentation examples:**
- Follow CONTRIBUTING.md patterns (commit messages, PR checklist)
- Use code blocks with syntax highlighting in Markdown
- Link between related docs (API_REFERENCE.md → tutorial → source code)

**Tutorial specific:**
- Include practical example: simulate a custom building and compare against Case 600 baseline
- Explain when to use BatchOracle vs Model (population evaluation vs single building analysis)
- Show complete error handling patterns (Result types, validate_parameters())

**API reference structure:**
- Group by class: BatchOracle section, Model section
- Each method: Description, Parameters, Returns, Example, Error cases
- Cross-link to Architecture and CLAUDE.md

**ASHRAE interpretation template:**
```
## Case 900 Annual Cooling

**Status:** FAIL
**Metric Value:** 4.75 MWh
**Reference Range:** 2.13 - 3.67 MWh
**Deviation:** +125% above reference maximum

### Interpretation

**Root Cause Hypothesis:**
High-mass annual energy over-prediction is a known 5R1C ISO 13790 limitation (documented in KNOWN_LIMITATIONS.md Section 1.1). The single thermal capacitance node cannot accurately represent complex thermal mass dynamics over 8760 simulation hours.

**Parameter Sensitivity:**
Annual cooling energy is most sensitive to:
- Thermal mass coupling ratio (h_tr_em / h_tr_ms) — values ~0.05 for high-mass buildings
- Solar gain parameters (SHGC, incidence angles) — higher gains in summer increase cooling demand
- HVAC setpoint — affects free-floating temperature and HVAC activation

**Recommended Next Steps:**
1. Review Phase 8 investigation for Case 960 inter-zone heat transfer issues
2. Check 6R2C_DECISION.md for alternative thermal network evaluation
3. Consider mode-specific coupling (h_tr_em_heating vs h_tr_em_cooling) effectiveness

**What-if Scenarios:**
- If we increased h_tr_em coupling ratio to 0.2: Would reduce heating/cooling energy by ~15% (test sensitivity analysis)
- If we added exterior surface area: Would increase solar gains, potentially worsening cooling
- If we used adaptive HVAC setpoint: Could reduce cooling demand by matching thermal mass temperature to comfort band
```

**Monthly aggregation approach:**
Sum monthly energies from hourly data, compare to reference monthly ranges, aggregate for annual total

**Sensitivity analysis normalization:**
Normalize coefficients by dividing by parameter range to enable fair parameter ranking

</specifics>

---

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. All decisions captured relate to documentation structure, API reference, tutorial format, and ASHRAE report clarity as defined in Phase 13 requirements.

</deferred>

---

*Phase: 13-documentation-tools*
*Context gathered: 2026-03-13*
