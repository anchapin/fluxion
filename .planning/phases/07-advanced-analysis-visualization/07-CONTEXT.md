# Phase 7: Advanced Analysis & Visualization - Context

**Gathered:** 2025-03-10 (Expected)
**Status:** Ready for planning

---

<domain>
## Phase Boundary

Implement sensitivity analysis, delta testing framework, interactive visualization, component-level energy breakdown, and extensible case specification to support building design optimization and diagnostic investigation.

**Key Requirements:**
- SENS-01..04: Parameter perturbation studies with ranked impact metrics
- DELTA-01..03: Case variant comparison with custom specifications
- VIZ-01..04: Real-time plotting, interactive zoom/pan, export to PNG/SVG, time series animation
- COMP-01..03: Component-level energy breakdown and CSV export
- EXT-01..04: Extensible framework for custom geometries, climate zones, building types
- MREF-01..03: Multi-reference comparison against EnergyPlus, ESP-r, TRNSYS

**Out of scope:**
- 3D building model rendering (deferred to future)
- Real-time parameter adjustment during simulation (optimization UI beyond scope)
- Automated statistical significance testing (simple comparisons sufficient)

</domain>

---

<decisions>
## Implementation Decisions

### Visualization Library & Architecture

**Approach:** Web-based visualization using Rust → WebAssembly + JavaScript charting library (e.g., Chart.js, D3.js, or Plotly.js).

**Rationale:**
- Best interactivity (zoom/pan, tooltips, responsive)
- Easy sharing (single HTML file)
- Vector graphics export (SVG) straightforward
- Consistent with modern web tooling pattern
- Allows future extension to browser-based dashboard

**Integration pattern:** Mixed approach
- Library API: `fluxion::visualization` module for custom Rust scripts
- CLI subcommand: `fluxion visualize <case_id>` for standard plots
- Generates standalone HTML files with embedded data and JavaScript

**Output priority:** SVG primary (vector for documentation), PNG fallback (raster for quick view), interactive HTML (exploratory)

**Interactivity level:** Advanced - real-time updating during simulation supported via streaming diagnostics, but no parameter adjustment during run (that's optimization UI, out of scope)

---

### Sensitivity Analysis Methodology

**Sampling strategy:** Combined OAT (One-At-a-Time) + random/Sobol for global coverage.

- Primary: OAT for clear main effects (vary one parameter, hold others at baseline)
- Secondary: Sobol quasi-random sequence for space-filling when interactions matter
- Users can choose via CLI flag: `--method oat|sobol|both`

**Sample count:** Default 10 levels per parameter (baseline ± 4 steps, total 10). Adaptive refinement optional for sensitive regions (user can specify follow-up sampling around interesting parameters).

**Output metrics:** Compute:
- Normalized sensitivity coefficient: `(max_output - min_output) / baseline × 100%`
- CVRMSE (Coefficient of Variation of RMSE) for response curve fit quality
- NMBE (Normalized Mean Bias Error) for systematic bias in response
- Linear regression slope (∂output/∂parameter) from least-squares fit through perturbation points

**Parallel evaluation:** Hybrid batching - construct full design matrix as population, call `BatchOracle.evaluate_population()` once. This maximizes GPU/rayon utilization. Implementation aggregates all parameter settings into single batch.

---

### Delta Testing Framework

**API surface:** Configuration-driven with CLI frontend.

- YAML configuration file defines base case and variants
- CLI: `fluxion delta --config comparisons.yaml`
- Also library API: `DeltaTester::compare(base, variants)` for programmatic use

**Comparison scope:** Both summary metrics and hourly time series.
- Generate Markdown delta report with side-by-side tables (annual energy, peaks, free-float temps)
- Include absolute and percent differences
- Export detailed hourly differences to CSV (long format) for external analysis
- Automated plots optional (can generate difference plots if visualization module available)

**Case specification:** Support modification patches + parameter sweeps.
```yaml
base: case600.yaml
variants:
  - name: "Low-E windows"
    patch: { window_u_value: 0.8, window_shgc: 0.7 }
  - name: "Sweep HVAC setpoint"
    sweep: { hvac_heating_setpoint: [18, 19, 20, 21, 22] }
```

**Statistical treatment:** Simple primary.
- Report mean difference (variant - base) with standard deviation across time series
- Example: "Annual cooling: 3.45 MWh (base: 3.12 MWh) Δ = +0.33 MWh (10.6%) σ = 0.15 MWh"
- No formal hypothesis testing (t-test, etc.) unless explicitly requested via advanced flag

---

### Multi-Reference Comparison

**Primary reference:** EnergyPlus (gold standard). Fluxion passes if within EnergyPlus ±15% tolerance.

**Fallback logic:**
- If EnergyPlus PASS → case status = PASS (regardless of others)
- If EnergyPlus FAIL but ESP-r or TRNSYS PASS → case status = WARN (reference disagreement)
- If all FAIL → case status = FAIL

**Per-program reporting:** Report pass/fail status separately for each reference program in the validation report. Transparent about disagreements.

**Missing reference handling:** Use available references with confidence flag.
- Single reference (e.g., only EnergyPlus) → mark as "Single reference [EnergyPlus]" in report
- Two references → normal comparison
- No references → cannot validate, mark as "No reference data"

**Reference versioning:** Versioned data files.
- References stored in `docs/ashrae_140_references.json` with fields: `version`, `source`, `cases: { case_id: { annual_heating: {min, max}, ... } }`
- Validation runs capture the reference version used in report header
- Provide `fluxion references update` command to fetch/validate new reference data from ASHRAE publications

---

### Extensible Case Framework

**Custom geometry support:**
- Programmatic: `CaseBuilder` API for Rust/Python users
- Simplified parameters: `.rectangular_zone(length, width, height)`, `.add_common_wall(zone1, zone2, area, r_value)` methods
- Standard file import: gbXML and EnergyPlus IDF formats (use existing parsers or add new readers)
- **Out of scope:** Graphical editor (that's a separate product)

**Custom weather:**
- EPW import: Allow custom `.epw` file path in `CaseSpec` (weather module already supports parsing)
- Formula-based synthetic: Provide `SyntheticWeather` builder with sinusoidal temperature, clear/cloudy patterns for testing
- Default: Denver TMY (only validated reference)

**Component library:**
- JSON/YAML library: `assemblies.yaml` with named constructions (wood_frame_R20, CMU_block, etc.)
- Builder extensions: `.with_wall("wood_frame_R20")` performs lookup
- Users can provide custom assembly files via config path

**Documentation:** Comprehensive examples-driven.
- Quickstart: "Your first custom case" (5 minutes)
- Tutorial: "Multi-zone office building from scratch" (step-by-step)
- Reference examples: ASHRAE 140 cases as live code examples
- API reference: all `CaseBuilder` methods documented with physics rationale

---

### Component Energy Breakdown

**Breakdown hierarchy (priority order):**
1. **By load source** (highest priority for diagnostic clarity)
   - Solar: beam vs diffuse vs ground-reflected
   - Internal: people vs equipment vs lighting (if data available)
   - Infiltration: air change rate contributions
   - Envelope conduction: per-surface type (walls, roof, windows, doors)
2. **By building element** (secondary, useful for retrofit)
   - Walls, roof, floor, windows (glazing vs frame), doors
3. **HVAC component** (low priority - 5R1C model has idealized HVAC)

**Temporal aggregation:**
- Default: Annual total + monthly breakdown (most useful for validation)
- Optional: Hourly time series already in diagnostics; can include component columns when detailed diagnostics enabled
- Derivable: Diurnal patterns from hourly data (average by hour-of-day)

**Validation approach:**
- Energy conservation: Sum(components) must equal total load ±1% (detect allocation bugs)
- Approximate ranges: Document expected proportion ranges for each ASHRAE case (e.g., "Case 600 cooling: solar 50-70%")
- No component-level reference comparison (reference programs may allocate differently)

**CSV export format:** Long (tidy) format for flexibility.
Columns: `Hour, Zone, Component, Value`
Example rows:
```
0, 0, solar_beam, 120.5
0, 0, solar_diffuse, 45.2
0, 0, internal_equipment, 80.0
```

**Alternative:** If wide format needed, provide `--wide` flag for pivot.

---

### Animation Implementation

**Content:** Combined 2D animated line chart.
- Upper panel: Zone temperatures (multiple lines, one per zone)
- Lower panel: HVAC output (heating positive, cooling negative) + solar gains
- Playhead scrolls through 8760 hours
- Optional: add internal loads and infiltration in separate panel if needed

**Output format:** Interactive HTML (primary) with MP4 fallback if HTML not feasible.
- HTML5 Canvas with Chart.js or similar
- Controls: play/pause, speed slider (0.5× to 100×), timeline scrubber
- Export frame as PNG button

**Integration:** Post-process only.
1. Run simulation with diagnostics: `fluxion simulate --diagnostics`
2. Generate animation: `fluxion animate --input diagnostics.csv --output animation.html`
- Keeps simulation fast (no rendering overhead)
- Allows re-animation without re-running simulation

**Speed defaults:** 1 hour per second (8760 sec = 2.4 hours). Fast-forward 100× = 87.6 seconds.

---

### Claude's Discretion

The following implementation choices are left to Claude's judgment during planning/implementation:

- Exact web visualization library (Chart.js vs D3.js vs Plotly.js) - choose based on ease of embedding and performance
- Sobol sequence generation crate and parameters (skip count, dimension)
- Adaptive refinement trigger criteria (when to follow up on sensitive parameters)
- Whether to merge animation into `fluxion visualize` or keep as separate `fluxion animate` subcommand
- Component-level breakdown exact column names and aggregation logic
- Names of builder extension methods for simplified geometry
- Default speed slider range and initial speed for animation

</decisions>

---

<code_context>
## Existing Code Insights

### Reusable Assets

**Diagnostic infrastructure:**
- `SimulationDiagnostics` with hourly data collection (temps, loads, energy accumulation)
- `CsvExporter` for per-zone CSV output with configurable delimiter
- `DiagnosticCollector` with energy breakdowns and peak timing
- `HourlyData` struct already contains most needed fields for animation and component breakdown

**Validation framework:**
- `ASHRAE140Validator` and `BenchmarkReport` for multi-reference comparison
- `BenchmarkData` stores reference ranges from EnergyPlus, ESP-r, TRNSYS
- `ValidationReportGenerator` can be extended to include sensitivity/delta reports
- Existing systematic issue classification can be reused for delta summaries

**Batch processing:**
- `BatchOracle::evaluate_population()` with rayon parallelism and GPU batching
- `DynamicBatchManager` for adaptive batching (from Phase 6)
- `SharedBatchInferenceService` for cross-case GPU utilization (if using surrogates)

**Case specification:**
- `CaseBuilder` with flexible construction pattern
- `CaseSpec` supports multiple zones, constructions, HVAC, weather
- `Assemblies` module for wall/roof/floor layer definitions

**Weather and file I/O:**
- EPW parser in `src/weather/epw.rs`
- CSV reading/writing with `csv` crate already used
- JSON/YAML config handling via serde (existing pattern in validation)

### Established Patterns

**Two-class API:**
- `BatchOracle` for high-throughput population evaluation
- `Model` for single-building detailed analysis
- Both used in validation framework

**Continuous Tensor Abstraction (CTA):**
- All state as `VectorField` with element-wise operations
- Enables GPU acceleration and rayon parallelism

**Builder pattern:**
- `CaseBuilder` for flexible case construction
- Extensible with new methods (`.with_custom_weather()`, `.rectangular_zone()`, etc.)

**Diagnostics-driven development:**
- Optional diagnostics with negligible overhead when disabled
- Diagnostic data drives reporting and analysis
- Export to CSV and Markdown is established pattern

**Validation-driven:
- Toleranced comparison (Pass/Warning/Fail) with ±5% band
- Per-program reference tracking possible (BenchmarkData already stores multiple sources)
- Report generation with sections and summary tables

### Integration Points

**Where new code connects:**

- `src/lib.rs` or `src/visualization/mod.rs` - New visualization module (library + subcommand registration)
- `src/bin/fluxion.rs` - Add subcommands: `visualize`, `sensitivity`, `delta`, `animate`
- `src/validation/ashrae_140_validator.rs` - Extend to support per-program comparison and versioned references
- `src/sim/construction.rs` / `CaseBuilder` - Add simplified geometry methods and assembly library
- `src/weather/epw.rs` - Expose custom EPW path in CaseSpec
- `Cargo.toml` - Add web visualization dependencies (wasm-bindgen, web-sys, chart library) OR desktop GUI deps
- `tools/` - Add configuration file validators, reference updater script

**New modules to create:**
- `src/visualization/mod.rs` - Core visualization library (HTML generation, SVG export)
- `src/analysis/sensitivity.rs` - Sensitivity analysis engine (sampling, evaluation, metrics)
- `src/analysis/delta.rs` - Delta testing framework (config parsing, comparison logic)
- `tools/reference_updater.rs` - Fetch and validate new ASHRAE reference data
- `src/sim/geometry_simplified.rs` - Simplified geometry builders (rectangular zones, etc.)
- `src/sim/assembly_library.rs` - Predefined construction assemblies (YAML-backed)

**Configuration files (new):**
- `config/assemblies.yaml` - Standard component library
- `docs/ashrae_140_references.json` - Versioned reference data
- User-provided: `comparisons.yaml` (delta config), custom assemblies files, custom EPW files

</code_context>

---

<specifics>
## Specific Ideas

### Visualization Library Selection

If choosing web-based approach:

- **Wasm target:** Compile visualization module to WebAssembly for potential future web app
- **HTML template:** Use `askama` or similar templating to embed data in HTML with `<script>` tags
- **JS library:** Chart.js for simplicity, D3.js for maximum customization, Plotly.js for built-in interactivity
- **Export to SVG:** Most JS libraries support SVG natively; can also use `svg` crate to generate server-side
- **Standalone HTML:** All dependencies either embedded or CDN-linked; single file portable

Example structure:
```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <canvas id="tempChart"></canvas>
  <script>
    const data = {{ embedded_json_data }};
    new Chart(ctx, { type: 'line', data: data, options: { responsive: true } });
  </script>
</body>
</html>
```

### Sensitivity Analysis Metrics Details

**CVRMSE (Coefficient of Variation of RMSE):**
```
CVRMSE = RMSE / mean(output) × 100%
RMSE = sqrt(mean((y_i - y_fit)^2))
```
Used to assess how well a linear (or nonlinear) model fits the perturbation response. Values <10% indicate near-linear response.

**NMBE (Normalized Mean Bias Error):**
```
NMBE = sum(y_i - y_fit) / (n × mean(output)) × 100%
```
Measures systematic bias in response curve. Positive = over-prediction on average.

**Linear regression slope:**
Fit `output = a + b × parameter_value`. The slope `b` is the sensitivity coefficient (units: output/parameter). Normalized: `b × (parameter_std / output_mean)`.

### Delta Testing Configuration Format

Example `comparisons.yaml`:

```yaml
base:
  spec: "cases/600.yaml"
  name: "Case 600 Baseline"

variants:
  - name: "Double-glazed windows"
    patch:
      window_u_value: 2.8  # W/m²K
      window_shgc: 0.65
    description: "Upgrade from single to double-pane clear glass"

  - name: "Increased thermal mass"
    patch:
      thermal_mass_capacitance: 15000  # J/K (from default 5000)
      thermal_mass_conductance: 15.0   # W/K

  - name: "HVAC setpoint sweep"
    sweep:
      hvac_heating_setpoint: [18, 19, 20, 21, 22]
      hvac_cooling_setpoint: [26, 27, 28, 29, 30]
    description: "Comfort setpoint variation study"

outputs:
  format: "markdown"  # or "json"
  include_hourly: true
  hourly_format: "long"  # or "wide"
  plots: true  # generate comparison plots if viz module available
```

### Component Breakdown CSV Long Format

```
Hour,Zone,Component,Value_Watts
0,0,solar_beam,125.3
0,0,solar_diffuse,42.1
0,0,solar_ground_reflected,8.7
0,0,internal_people,40.0
0,0,internal_equipment,80.0
0,0,internal_lighting,60.0
0,0,infiltration,-45.2
0,0,envelope_walls,-32.1
0,0,envelope_roof,-15.4
0,0,envelope_windows,-28.7
0,0,envelope_doors,-3.2
0,0,hvac_heating,0.0
0,0,hvac_cooling,0.0
...
```

Can be pivoted in R/Pandas:
```r
library(tidyr)
wide <- pivot_wider(data, names_from=Component, values_from=Value_Watts)
```

### Multi-Reference Data File Format

`docs/ashrae_140_references.json`:

```json
{
  "version": "2024-01",
  "source": "ASHRAE 140-2024 Standard, Table X1 through X5",
  "cases": {
    "600": {
      "annual_heating": {"min": 3.10, "max": 3.80, "programs": ["EnergyPlus", "ESP-r"]},
      "annual_cooling": {"min": 1.90, "max": 2.40, "programs": ["EnergyPlus"]},
      "peak_heating": {"min": 5.20, "max": 6.10, "programs": ["EnergyPlus", "ESP-r", "TRNSYS"]},
      "peak_cooling": {"min": 3.10, "max": 3.80, "programs": ["EnergyPlus"]}
    },
    "900": {
      ...
    }
  }
}
```

The `programs` array indicates which reference programs provided data for that metric.

### Animation HTML Template

Pre-generate an HTML file with embedded data array (8760 × num_metrics).

JavaScript controls:
```javascript
let currentHour = 0;
let speed = 1; // hours per second
let animationId;

function renderFrame(hour) {
  // Update all charts to show data up to 'hour'
  // Highlight current hour with vertical line
}

function play() {
  animationId = setInterval(() => {
    currentHour += speed;
    if (currentHour > 8760) currentHour = 0;
    renderFrame(currentHour);
  }, 1000);
}

function scrubTo(hour) {
  currentHour = hour;
  renderFrame(hour);
}
```

Data embedded as JSON in `<script>` tag; Chart.js datasets updated via `setData`.

</specifics>

---

<deferred>
## Deferred Ideas

None identified — discussion stayed within phase scope. All ideas captured within the 7 gray areas.

**Ideas explicitly rejected or out of scope:**
- 3D building model rendering (requires 3D graphics engine, heavy)
- Real-time parameter adjustment during simulation (optimization UI, Phase 8+)
- Automated hypothesis testing (t-tests, Wilcoxon) for delta comparisons (can be added later as advanced feature)
- Graphical building editor (separate product)

</deferred>

---

*Phase: 07-advanced-analysis-visualization*
*Context gathered: 2025-03-10 (expected)*
