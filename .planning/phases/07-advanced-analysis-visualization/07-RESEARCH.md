# Phase 7 Research: Advanced Analysis & Visualization

**Date:** 2026-03-10
**Status:** Ready for planning
**Context:** Based on 07-CONTEXT.md decisions and existing Fluxion architecture

---

## Executive Summary

Phase 7 requires implementing 20 requirements across sensitivity analysis, delta testing, interactive visualization, component-level breakdown, extensible case specification, and multi-reference comparison. The existing Fluxion codebase provides strong foundations: `DiagnosticCollector` with `HourlyData`, delta testing tool (`fluxion-delta`), `CaseBuilder` pattern, and CSV export infrastructure.

**Key architectural decisions from context:**
- Web-based visualization using embedded HTML/JS with Chart.js/D3/Plotly
- OAT + Sobol sampling for sensitivity analysis
- YAML-driven delta configuration with patch+sweep variants
- Long-format CSV for all exports
- EnergyPlus primary reference with per-program reporting
- Simplified geometry builders + gbXML/IDF import

---

## 1. Sensitivity Analysis (SENS-01..04)

### 1.1 Design Requirements

Implement parameter perturbation studies with:
- OAT (One-At-a-Time) for clear main effects
- Sobol quasi-random sequence for global coverage
- Metrics: normalized sensitivity coefficient, CVRMSE, NMBE, linear regression slope
- Hybrid batching to maximize `BatchOracle` utilization

### 1.2 Rust Ecosystem Assessment

**Sampling Strategies**

| Strategy | Recommended Crate | Rationale |
|----------|------------------|-----------|
| OAT (simple) | Manual implementation | Simple to implement: vary one parameter over N levels while holding others at baseline |
| Sobol sequence | `sobol` or `rand::rngs::StdRng` + `rand_distr::Uniform` | The `sobol` crate (if still maintained) or implement using Joe-Kuo Sloan directions; `rand` already in dependencies |
| Random sampling | `rand` (already present) | For comparison baseline |
| Latin Hypercube | `lhs` crate (optional) | Alternative space-filling design |

**Statistical Analysis**

| Metric | Recommended Crate | Implementation |
|--------|------------------|----------------|
| Linear regression | `linregress` or `ndarray` + manual OLS | `linregress` provides simple OLS with R², standard errors; `ndarray` already present for matrix operations |
| CVRMSE (RMSE/mean × 100) | Manual (sqrt + mean) | `rmse = sqrt(mean((y - y_fit).powi(2)))`; mean already available |
| NMBE (bias) | Manual (sum + mean) | `nmbe = sum(y - y_fit) / (n * mean(output)) * 100` |
| Sensitivity coefficient | Manual | `(max - min) / baseline * 100%` |

**Dependencies to add:**
- `linregress = "0.6"` (lightweight, no bloat)
- `sobol = "0.2"` OR implement custom Sobol generator using direction numbers (more control)

### 1.3 Implementation Strategy

**Module Structure:**
```
src/analysis/
  mod.rs
  sensitivity.rs        // SensitivityEngine, SamplingStrategy, SensitivityMetrics
  sampling/             // Samplers: OATSampler, SobolSampler, RandomSampler
```

**Core Algorithm:**
1. **Design Matrix Construction:**
   - Given parameter bounds (e.g., window U-value: [0.1, 5.0], setpoints: [15, 30])
   - For OAT: Create N × P matrix (N=10 levels, P=parameters)
     - For each parameter i: create N runs varying param[i] over levels, others = baseline
   - For Sobol: Generate N × P quasi-random points in [0,1]^P, scale to parameter bounds
   - Combine into single population matrix (Vec<Vec<f64>>)

2. **Batch Evaluation:**
   - Call `BatchOracle::evaluate_population(population, use_surrogates?)`
   - Returns fitness scores (EUI or annual energy)

3. **Metrics Computation:**
   - For each parameter, fit linear model: energy ~ parameter value
   - Compute CVRMSE, NMBE to assess fit quality (<10% desirable)
   - Extract slope as sensitivity coefficient (normalized version for ranking)

4. **Output:**
   - Ranked parameter table (Markdown or JSON)
   - Response curves (plot later with visualization module)
   - Detailed metrics per parameter

**Integration points:**
- `BatchOracle` for parallel evaluation (critical for speed)
- `CaseBuilder` to generate base case for perturbations
- Output to CSV/JSON for visualization module consumption

**Pseudocode:**
```rust
pub struct SensitivityEngine {
    sampler: Box<dyn Sampler>,
    oracle: BatchOracle,
}

impl SensitivityEngine {
    pub fn analyze(
        &self,
        base_params: &[f64],
        bounds: &[(f64, f64)],
        levels: usize
    ) -> Result<SensitivityReport> {
        // 1. Generate design matrix
        let population = self.sampler.sample(base_params, bounds, levels)?;

        // 2. Evaluate in batch
        let results = self.oracle.evaluate_population(population, true)?;

        // 3. Compute metrics per parameter
        let mut metrics = Vec::new();
        for (i, param_bounds) in bounds.iter().enumerate() {
            let mut param_values = Vec::new();
            let mut outputs = Vec::new();

            // Extract param[i] and corresponding output for OAT
            for (row, &output) in population.iter().zip(&results) {
                param_values.push(row[i]);
                outputs.push(output);
            }

            // Linear regression
            let regression = linregress(&param_values, &outputs)?;
            let cvrmse = compute_cvrmse(&param_values, &outputs, regression.intercept, regression.slope);
            let nmbe = compute_nmbe(&param_values, &outputs, regression.intercept, regression.slope);

            metrics.push(SensitivityMetric {
                parameter_index: i,
                slope: regression.slope,
                normalized_slope: (regression.slope * param_bounds.1) / mean(&outputs),
                cvrmse,
                nmbe,
                r_squared: regression.r_value * regression.r_value,
            });
        }

        // 4. Rank by absolute normalized slope
        metrics.sort_by(|a, b| b.normalized_slope.abs().partial_cmp(&a.normalized_slope.abs()).unwrap());

        Ok(SensitivityReport { metrics, design_matrix: population, outputs: results })
    }
}
```

### 1.4 Risks & Challenges

- **OAT misses interactions**: Document limitation; Sobol helps but requires more samples
- **Nonlinear response curves**: Linear fit may be poor; detect via high CVRMSE (>20%) and flag
- **Parameter bounds selection**: User must provide sensible bounds; document with examples
- **Computational cost**: 10 levels × P parameters; OAT with P=2 gives 20 runs; Sobol more efficient

---

## 2. Delta Testing Framework (DELTA-01..03)

### 2.1 Design Requirements

- YAML configuration file with base case + variants (patch and sweep)
- Comparison: summary metrics + hourly time series
- Export to Markdown report + long-format CSV
- CLI: `fluxion delta --config comparisons.yaml`
- Library API: `DeltaTester::compare(base, variants)`

### 2.2 Existing Infrastructure

**Already present:** `tools/fluxion_delta.rs` compares Fluxion vs EnergyPlus CSVs (hourly delta analysis). This provides:
- CSV parsing with flexible column mapping
- Hourly comparison structs (`HourlyComparison`, `DeltaAnalysisReport`)
- Markdown and CSV export
- Summary statistics (mean deltas, max errors)

**Gap:** This is a reference implementation that we can **extend** for general delta testing (any base vs variant, not just Fluxion vs EnergyPlus).

### 2.3 Implementation Strategy

**Reuse and Generalize `fluxion_delta.rs`:**

1. **Extend `DeltaAnalysisReport`** to support arbitrary case variants:
   - Current: compares Fluxion vs EnergyPlus
   - New: compare any two `CaseSpec` results
   - Store `Vec<CaseResult>` where `CaseResult` has case_id, annual_metrics, hourly_data

2. **YAML Configuration Parsing:**
   - Use `serde_yaml` (add to Cargo.toml if not present; likely needed)
   - Config structure:
     ```yaml
     base:
       spec: "cases/600.yaml"   # Could be CaseSpec file or built-in identifier
       name: "Case 600 Baseline"
     variants:
       - name: "Low-E windows"
         patch: { window_u_value: 0.8, window_shgc: 0.7 }
       - name: "HVAC sweep"
         sweep: { hvac_heating_setpoint: [18, 19, 20, 21, 22] }
     outputs:
       format: "markdown"
       include_hourly: true
       hourly_format: "long"
       plots: true
     ```

3. **DeltaTester API:**
   ```rust
   pub struct DeltaTester { config: DeltaConfig, output_dir: PathBuf }
   impl DeltaTester {
       pub fn new(config_path: &Path) -> Result<Self>;
       pub fn run(&self) -> Result<DeltaReport>;
   }
   ```

4. **Comparison Logic:**
   - For each variant (including sweep expansions):
     1. Load base `CaseSpec`, run simulation with diagnostics enabled
     2. Apply patch modifications to `CaseSpec` OR generate sweep populations
     3. Run simulation(s), collect `DiagnosticReport` (annual energy, hourly data)
     4. Compare metrics: absolute difference, percent difference
     5. Compute statistics across time series: mean(delta), std(delta)
   - For sweep variants: generate parameter sweep plots (later with viz)

5. **Report Generation:**
   - Markdown table: variant vs base (annual heating, cooling, peak loads, % diff)
   - Per-variant hourly delta CSV in long format: `Hour, Metric, Base, Variant, Delta`
   - Optional plots via visualization module

**Module location:**
- Library: `src/analysis/delta.rs`
- CLI binary: `src/bin/delta.rs` or `tools/delta.rs`

### 2.4 Integration with Existing Code

- **Re-use:** `DiagnosticCollector` for hourly data collection
- **Re-use:** `EnergyBreakdown` for component-level comparison (if detailed breakdown available)
- **Extend:** `CaseBuilder` to support YAML deserialization from file (load custom case specs)
- **Batch evaluation:** Sweeps use `BatchOracle` to evaluate multiple parameter sets in parallel

### 2.5 Risks & Challenges

- **YAML complexity**: Need robust schema validation; define clear error messages
- **Sweep expansion**: Sweep over multiple parameters creates combinatorial explosion; warn if >1000 runs
- **Missing diagnostics**: Ensure diagnostics enabled for all delta runs (even if user didn't explicitly request)
- **Reference versioning**: If comparing to external programs (EnergyPlus), need to track reference data version

---

## 3. Interactive Visualization (VIZ-01..04)

### 3.1 Design Requirements

- Real-time plotting with zoom/pan
- Export to PNG/SVG
- Time series animation (8760 hours)
- Web-based: single HTML file with embedded data
- Library API + CLI subcommand

### 3.2 Architecture Decision

**Chosen approach:** Generate standalone HTML with embedded JSON data + external JS library

**Why this approach:**
- No native graphics dependencies (no `plotters` SVG generation on Rust side, which produces static only)
- Full interactivity (zoom/pan/tooltips) via Chart.js/D3
- Single HTML file portable
- Future extension to WASM dashboard possible
- Works offline if we embed JS library (base64) or use CDN

**Library selection:**
- **Chart.js** (recommended): Simple API, good for time series, responsive, supports animation
- **Plotly.js**: More plot types, but larger bundle
- **D3.js**: Maximum flexibility but verbosity

We'll use Chart.js via CDN link; allows single HTML file without embedding 200KB of JS.

**Output formats:**
- SVG: Supported by Chart.js (via `chartjs-plugin-datalabels` or built-in export)
- PNG: Chart.js can export canvas to PNG
- Interactive HTML: Primary output

### 3.3 Rust Dependencies

**Core:**
- `askama = "0.12"` (templating) - not locked yet but stable
  - Alternative: Simple string formatting (no dependency) if templates are simple
  - We'll likely use simple format! with JSON embedding for zero dep

- **No heavy visualization deps in Rust library:** All HTML/JS generation is string manipulation

**Optional (if generating plots server-side without JS):**
- `plotters = "0.3"` - for static SVG/PNG export (fallback if HTML not desired)

**Decision:** Minimal deps - use `serde_json` (already present) to serialize data, embed in HTML template.

### 3.4 Implementation Strategy

**Module:** `src/visualization/mod.rs`

**Core types:**
```rust
pub struct PlotData {
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub datasets: Vec<Dataset>,
}

pub struct Dataset {
    pub label: String,
    pub data: Vec<f64>,
    pub color: Option<String>,
}

pub struct HTMLReport {
    plots: Vec<PlotData>,
    time_series: Option<TimeSeriesData>,
}

impl HTMLReport {
    pub fn to_html(&self) -> String {
        // Use Askama template or manual string building
        // Embed data as JSON in <script> tag
        // Include Chart.js from CDN
        // Generate Canvas elements with Chart.js config
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_html())
    }
}
```

**Time Series Animation:**
- Data: 8760 points per series (temperature, HVAC, solar)
- Chart.js streaming or animation API
- Controls: play/pause, speed slider, timeline scrubber
- Implementation: Pre-load all data, animate by updating chart visible range

**CLI Integration:**
- New subcommand in `src/bin/fluxion.rs`: `fluxion visualize <case_id>`
- Or separate binary: `fluxion-visualize`
- Accept diagnostic CSV or `DiagnosticReport` struct

**Example HTML structure:**
```html
<!DOCTYPE html>
<html>
<head>
  <title>Fluxion Analysis: {{ case_id }}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: sans-serif; margin: 20px; }
    .chart-container { margin: 20px 0; }
    .controls { margin: 10px 0; }
  </style>
</head>
<body>
  <h1>Case {{ case_id }}</h1>

  <div class="chart-container">
    <canvas id="tempChart"></canvas>
  </div>

  <div class="controls">
    <button onclick="play()">Play</button>
    <button onclick="pause()">Pause</button>
    <input type="range" min="1" max="100" value="1" onchange="setSpeed(this.value)">
  </div>

  <script>
    const hourlyData = {{ data_json | safe }};
    const ctx = document.getElementById('tempChart').getContext('2d');
    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array.from({length: 8760}, (_, i) => i),
        datasets: [
          { label: 'Zone Temp', data: hourlyData.temps, borderColor: 'red' },
          { label: 'Outdoor Temp', data: hourlyData.outdoor, borderColor: 'blue' }
        ]
      },
      options: {
        scales: { x: { type: 'linear' } },
        animation: false  // Disable for performance
      }
    });

    let currentHour = 0;
    let speed = 1;
    let playing = false;

    function renderFrame(hour) {
      // Show data up to hour
      chart.data.datasets.forEach(ds => {
        ds.data = hourlyData.slice(0, hour);
      });
      chart.update();
    }

    function play() {
      playing = true;
      const step = () => {
        if (!playing) return;
        currentHour += speed;
        if (currentHour > 8760) currentHour = 0;
        renderFrame(currentHour);
        setTimeout(step, 1000 / 30);  // 30 FPS
      };
      step();
    }
  </script>
</body>
</html>
```

### 3.5 Export to SVG/PNG

**SVG export:**
- Chart.js supports SVG rendering via configuration
- Or allow user to right-click → Save as SVG from browser

**PNG export:**
- JavaScript `canvas.toDataURL('image/png')` → download button
- Add to HTML template: `<button onclick="exportPNG()">Export PNG</button>`

**Alternative (server-side):** Use `plotters` crate to generate static charts without browser. Consider as fallback for headless environments.

### 3.6 Integration Points

- **Source data:** `DiagnosticCollector::hourly_data` or exported CSV
- **CLI:** `fluxion visualize --input diagnostics.csv --output report.html`
- **Library:** `VisualizationEngine::from_diagnostic_report(report) -> HTMLReport`

### 3.7 Risks & Challenges

- **Chart.js version:** Use CDN with pinned version to avoid breaking changes
- **Large datasets:** 8760 points × multiple series may slow down browser; consider downsampling option
- **Animation performance:** Re-rendering entire chart each frame is expensive; use incremental updates if possible
- **WASM size:** If we later compile to WASM, Chart.js dependency must be loaded externally

---

## 4. Component-Level Energy Breakdown (COMP-01..03)

### 4.1 Design Requirements

- Break down energy by load source: solar (beam/diffuse/ground), internal (people/equipment/lighting), infiltration, envelope conduction (walls/roof/windows/doors)
- Temporal aggregation: annual total + monthly breakdown
- CSV export in long format: `Hour,Zone,Component,Value_Watts`
- Energy conservation check: sum(components) ≈ total load ±1%

### 4.2 Existing Infrastructure

**Diagnostic module already tracks:**
- `solar_gains` per zone per hour
- `internal_loads` per zone per hour
- `infiltration_loss` per zone per hour
- `envelope_conduction` per zone per hour

**EnergyBreakdown struct** (`src/validation/diagnostic.rs`) aggregates to MWh totals:
- `solar_gains_mwh`, `internal_gains_mwh`, `infiltration_mwh`, `envelope_conduction_mwh`
- `heating_mwh`, `cooling_mwh`

**Gap:** The breakdown currently aggregates at annual level only and doesn't break down by component subtype (beam vs diffuse, walls vs windows). We need to extend to:
- Distinguish solar beam vs diffuse vs ground-reflected
- Distinguish envelope components (walls, roof, windows, doors)
- Support monthly aggregation
- Generate long-format CSV with component-level detail

### 4.3 Implementation Strategy

**Extend `HourlyData` struct:**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HourlyData {
    // ... existing fields ...
    /// Solar beam gains per zone (W)
    pub solar_beam: Vec<f64>,
    /// Solar diffuse gains per zone (W)
    pub solar_diffuse: Vec<f64>,
    /// Solar ground-reflected gains per zone (W)
    pub solar_ground_reflected: Vec<f64>,
    /// Envelope conduction by surface type per zone (W)
    pub surface_conduction: Vec<SurfaceConduction>,
    /// Infiltration breakdown (optional)
    pub infiltration_breakdown: Option<InfiltrationBreakdown>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceConduction {
    pub walls: f64,
    pub roof: f64,
    pub floor: f64,
    pub windows: f64,
    pub doors: f64,
}
```

**New reporting module:** `src/analysis/component_breakdown.rs`

```rust
pub struct ComponentBreakdown {
    // Annual totals by component
    pub total_heating_mwh: f64,
    pub total_cooling_mwh: f64,
    // Component-level energy (MWh)
    pub solar_beam_mwh: f64,
    pub solar_diffuse_mwh: f64,
    pub solar_ground_mwh: f64,
    pub internal_people_mwh: f64,
    pub internal_equipment_mwh: f64,
    pub internal_lighting_mwh: f64,
    pub envelope_walls_mwh: f64,
    pub envelope_roof_mwh: f64,
    pub envelope_floor_mwh: f64,
    pub envelope_windows_mwh: f64,
    pub envelope_doors_mwh: f64,
    pub infiltration_mwh: f64,
    // Monthly breakdowns
    pub monthly: Vec<MonthlyBreakdown>,
    // Conservation check
    pub balance_error_percent: f64,
}

impl ComponentBreakdown {
    pub fn from_hourly_data(data: &[HourlyData], num_zones: usize) -> Self {
        // Sum up all hours, convert Wh->MWh
        // Compute balance: (solar+internal) - (heating - cooling) - (envelope+infiltration) ≈ 0
    }

    pub fn to_long_csv(&self) -> String {
        // Format: Hour,Zone,Component,Value_Watts (for hourly export)
        // Or aggregate: Component,Annual_MWh,Monthly_[1..12]
    }

    pub fn to_markdown(&self) -> String { /* table */ }
}
```

**CSV Export Format (Long):**
```csv
Hour,Zone,Component,Value_Watts
0,0,solar_beam,125.3
0,0,solar_diffuse,42.1
0,0,internal_people,40.0
0,0,envelope_walls,-32.1
...
```

**API:**
- Extend `DiagnosticCollector` to collect detailed component data if enabled via config
- New method: `collector.get_component_breakdown(case_id) -> ComponentBreakdown`
- CLI: `fluxion breakdown --input diagnostics.csv --output breakdown.csv`

### 4.4 Integration Points

- **Physics engine** must populate detailed fields when diagnostics enabled:
  - Modify `ThermalModel::solve_timesteps()` or `record_timestep()` to capture solar breakdown
  - Surface conduction breakdown requires tracking per-surface heat transfer (already in `HourlyData`? Check)

- **Existing CSV export** (`export_hourly_csv`) extends to include component columns

### 4.5 Risks & Challenges

- **Requires tracking detailed sources** in physics engine: beam vs diffuse solar already computed; need to expose
- **Surface conduction**: Currently `envelope_conduction` is total per zone; may need to add per-surface tracking
- **Conservation check**: Sum of components should equal total load ±1%; this is validation of allocation logic
- **Data volume**: Long-format CSV with 8760 hours × ~10 components × zones = 87,600 rows; manageable

---

## 5. Time Series Animation (SWING-01..03)

### 5.1 Design Requirements

- Combined 2D animated line chart
- Upper panel: Zone temperatures (multiple lines)
- Lower panel: HVAC output (heating + cooling) + solar gains
- Playhead scrolls through 8760 hours
- Controls: play/pause, speed slider (0.5× to 100×), timeline scrubber
- Export frame as PNG
- Post-process only: `fluxion animate --input diagnostics.csv --output animation.html`

### 5.2 Implementation Strategy

**Separate binary or subcommand:**
- `src/bin/animate.rs` (new)
- Or `fluxion animate` subcommand in main

**Engine:** `src/animation/mod.rs`

```rust
pub struct AnimationData {
    pub timestamps: Vec<f64>,  // Hour indices
    pub zone_temps: Vec<Vec<f64>>,  // [zone][hour]
    pub hvac_heating: Vec<f64>,
    pub hvac_cooling: Vec<f64>,
    pub solar_gains: Vec<f64>,
}

impl AnimationData {
    pub fn from_diagnostic_csv(path: &Path) -> Result<Self> {
        // Parse existing hourly diagnostic CSV
        // Extract columns: Hour, Zone_Temp (multi-zone), HVAC_Heating, HVAC_Cooling, Solar_Gain
    }

    pub fn to_html(&self) -> String {
        // Use Chart.js animation with streaming
        // Dual-axis: temp chart (top), power chart (bottom)
    }
}
```

**Chart.js animation technique:**
- Load all data as arrays
- Two separate charts stacked vertically (or Chart.js subplots if using v3+)
- Update visible window based on `currentHour`
- `setInterval` updates `currentHour` by `speed` per tick
- Call `chart.update('none')` for efficient animation (no full re-render)

**Controls implementation:**
```javascript
let currentHour = 0;
let speed = 1;
let animationId = null;

function updateCharts() {
  const visibleEnd = currentHour + 168;  // Show one week ahead
  chart.options.scales.x.min = currentHour;
  chart.options.scales.x.max = visibleEnd;
  chart.update();
}

function play() {
  if (animationId) return;
  animationId = setInterval(() => {
    currentHour += speed;
    if (currentHour > 8760) currentHour = 0;
    updateCharts();
  }, 1000 / 30);
}

function scrubTo(hour) {
  currentHour = hour;
  updateCharts();
}
```

**Export PNG:**
```javascript
function exportPNG() {
  const link = document.createElement('a');
  link.download = 'frame.png';
  link.href = document.getElementById('tempChart').toDataURL('image/png');
  link.click();
}
```

**Speed control:** Speed = hours per second; default 1× = 8760 seconds = 2.43 hours runtime

### 5.3 Integration Points

- **Input:** Diagnostic CSV output from `DiagnosticCollector::export_hourly_csv()`
- **Alternative:** Ingest from `HourlyData` struct directly if using library API
- **CLI:** `fluxion animate --input case_600_hourly.csv --output animation.html`

### 5.4 Risks & Challenges

- **Chart.js animation performance:** Eager re-rendering can be slow; use `chart.update('none')` mode
- **Memory:** 8760 points × multiple series in browser memory is fine (<1MB)
- **Time zone handling:** Hour axis should be "Hour of Year" or convert to month/day ticks
- **Multiple zones:** If many zones (>10), consider checkbox to select visible zones

---

## 6. Extensible Case Specification (EXT-01..04)

### 6.1 Design Requirements

- **Custom geometry:**
  - Programmatic: `CaseBuilder` API with `.rectangular_zone()`, `.add_common_wall()`
  - File import: gbXML and EnergyPlus IDF formats
- **Custom weather:**
  - EPW import (custom `.epw` path)
  - Synthetic weather builder (sinusoidal patterns for testing)
- **Component library:**
  - JSON/YAML assembly definitions (`assemblies.yaml`)
  - `.with_wall("wood_frame_R20")` lookup
- **Documentation:** Examples-driven (quickstart, tutorial, reference)

### 6.2 Existing Infrastructure

**CaseBuilder** (`src/validation/ashrae_140_cases.rs`):
- Already has methods like `.with_dimensions()`, `.with_south_window()`, `.with_hvac_setpoints()`
- Returns `CaseSpec` which contains geometry, constructions, HVAC, weather

**Assemblies module** (`src/sim/construction.rs`):
- Already has `Construction`, `ConstructionLayer`, `MassClass`
- Functions to compute R-value, U-value, effective mass parameters

**Weather:** `src/weather/epw.rs` parses EPW files; `src/weather/mod.rs` defines `WeatherSource`

**Gap:** Need to extend `CaseBuilder` with:
- `.rectangular_zone()` convenience method
- `.add_wall()`, `.add_roof()` methods
- `.with_weather_from_epw(path)` method
- `.with_assembly(name)` lookup from YAML/JSON

### 6.3 Implementation Strategy

**1. Simplified Geometry Builders (`src/sim/geometry_simplified.rs`):**

```rust
impl CaseBuilder {
    /// Add a rectangular single-zone building.
    pub fn rectangular_zone(mut self, length: f64, width: f64, height: f64) -> Result<Self> {
        let floor_area = length * width;
        let volume = floor_area * height;
        // Compute surface areas: walls = 2*(L+W)*H, roof = floor_area, floor = floor_area
        // Add default constructions (need to specify)
        self.spec.num_zones = 1;
        self.spec.floor_area = floor_area;
        // ... populate geometry
        Ok(self)
    }

    /// Add a common wall between two zones.
    pub fn add_common_wall(&mut self, zone1: usize, zone2: usize, area: f64, r_value: f64) -> Result<()> {
        // Add to inter-zone conductance matrix
        Ok(())
    }
}
```

**2. Assembly Library (`src/sim/assembly_library.rs`):**

```rust
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
pub struct Assembly {
    pub name: String,
    pub layers: Vec<ConstructionLayer>,
    pub total_r_value: f64,
}

pub struct AssemblyLibrary {
    assemblies: HashMap<String, Assembly>,
}

impl AssemblyLibrary {
    pub fn from_yaml(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let lib: HashMap<String, Assembly> = serde_yaml::from_reader(file)?;
        Ok(Self { assemblies: lib })
    }

    pub fn get(&self, name: &str) -> Option<&Assembly> {
        self.assemblies.get(name)
    }
}

// In CaseBuilder:
impl CaseBuilder {
    pub fn with_assembly(mut self, assembly: &Assembly) -> Result<Self> {
        // Apply assembly to appropriate surface (need context: wall/roof/floor)
        Ok(self)
    }

    pub fn with_wall(mut self, assembly_name: &str, library: &AssemblyLibrary) -> Result<Self> {
        let assembly = library.get(assemblies_name).ok_or_else(|| ...)?;
        self.spec.wall_construction = Some(assembly.to_construction()?);
        Ok(self)
    }
}
```

**Example `assemblies.yaml`:**
```yaml
wood_frame_R20:
  layers:
    - name: "wood siding"
      thickness: 0.02
      conductivity: 0.15
      density: 600
      specific_heat: 1500
    - name: "fiberglass insulation"
      thickness: 0.14
      conductivity: 0.04
      density: 30
      specific_heat: 1000
    - name: "gypsum board"
      thickness: 0.016
      conductivity: 0.25
      density: 800
      specific_heat: 1000
  total_r_value: 5.0  # ms²K/W
```

**3. Custom Weather:**
- Extend `CaseSpec` to accept `Option<PathBuf>` for custom EPW
- `CaseBuilder::with_weather_epw(path)` loads from file instead of default Denver TMY
- `WeatherSource::from_epw_path(path)`

**Synthetic weather:**
```rust
impl CaseBuilder {
    pub fn with_synthetic_weather(self) -> Result<Self> {
        // Generate sinusoidal temperature: T_out(t) = T_mean + A * sin(2πt/8760)
        // Clear/cloudy patterns: step changes in solar radiation
        Ok(self)
    }
}
```

**4. gbXML/IDF Import:**
- **gbXML:** Use `quick-xml` crate to parse; map to `CaseSpec`
  - gbXML contains building geometry, surfaces, constructions
  - Need translator module: `src/io/gbxml_import.rs`
- **EnergyPlus IDF:** More complex; use `epr` crate or custom parser
  - Mapping: Objects → `CaseSpec` (constructions, zones, schedules)
  - Recommended: Start with gbXML (simpler schema)

**5. Documentation:**
- `docs/cases/quickstart.md`: "Your first custom case" (5 min)
- `docs/cases/tutorial.md`: Step-by-step multi-zone office building
- `docs/cases/reference.md`: ASHRAE 140 cases as code examples
- `docs/api/case_builder.md`: All builder methods with physics rationale

### 6.4 Integration Points

- **CaseBuilder** (already exists in `validation/ashrae_140_cases.rs`) is THE extension point
- Add methods to `CaseBuilder` as above
- `CaseSpec` needs fields for custom geometry, constructions
- Simulations use `CaseSpec` to build `ThermalModel`

### 6.5 Risks & Challenges

- **gbXML/IDF complexity:** Full parsers are large undertaking; consider starting with simplified custom format (YAML geometry)
- **Geometry validation:** Ensure user-provided surfaces are closed, consistent areas; may need geometry engine
- **Assembly library mapping:** Need clear mapping from YAML layers to Fluxion `Construction` parameters (mass area multiplier A_m)
- **Documentation burden:** Examples must be tested to avoid drift

---

## 7. Multi-Reference Comparison (MREF-01..03)

### 7.1 Design Requirements

- Compare Fluxion results against EnergyPlus, ESP-r, TRNSYS reference data
- Primary reference: EnergyPlus (gold standard)
- Fallback logic: EnergyPlus PASS → PASS; EnergyPlus FAIL but ESP-r/TRNSYS PASS → WARN; all FAIL → FAIL
- Per-program pass/fail status reporting
- Versioned reference data files (JSON)
- Command: `fluxion references update` to fetch new ASHRAE data

### 7.2 Existing Infrastructure

**Validation framework:**
- `ASHRAE140Validator` (in `src/validation/ashrae_140/`) runs cases and compares to reference ranges
- `BenchmarkReport` likely stores reference min/max values
- `ComparisonRow` with `status: PASS/FAIL`

**Gap:** Currently stores only single reference range (min/max). Need to expand to store per-program values and compute status per program.

### 7.3 Implementation Strategy

**1. Extend Benchmark Data Structure:**

```rust
// Current (simplified):
pub struct BenchmarkData {
    cases: HashMap<String, CaseBenchmark>,  // min/max
}

// New:
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceProgram {
    pub name: String,  // "EnergyPlus", "ESP-r", "TRNSYS"
    pub version: String,
    pub results: HashMap<String, MetricRange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiReferenceData {
    pub cases: HashMap<String, CaseReference>,
    pub source: String,  // "ASHRAE 140-2024"
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseReference {
    pub annual_heating: Option<ProgramRange>,
    pub annual_cooling: Option<ProgramRange>,
    pub peak_heating: Option<ProgramRange>,
    pub peak_cooling: Option<ProgramRange>,
    // ... other metrics
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramRange {
    pub min: f64,
    pub max: f64,
    pub program: String,
    // Could add uncertainty, confidence interval
}
```

**Example JSON structure (`docs/ashrae_140_references.json`):**
```json
{
  "version": "2024-01",
  "source": "ASHRAE 140-2024 Standard, Tables X1-X5",
  "cases": {
    "600": {
      "annual_heating": [
        { "program": "EnergyPlus", "min": 3.10, "max": 3.80 },
        { "program": "ESP-r", "min": 3.15, "max": 3.85 }
      ],
      "annual_cooling": [
        { "program": "EnergyPlus", "min": 1.90, "max": 2.40 }
      ]
    }
  }
}
```

**2. Multi-Reference Comparison Engine:**

```rust
pub struct MultiReferenceValidator {
    reference_data: MultiReferenceData,
}

impl MultiReferenceValidator {
    pub fn validate_case(&self, case_id: &str, fluxion_value: f64, metric: &str) -> ValidationResult {
        let program_ranges = self.reference_data.cases.get(case_id)
            .and_then(|c| match metric {
                "annual_heating" => c.annual_heating.as_ref(),
                "annual_cooling" => c.annual_cooling.as_ref(),
                _ => None,
            });

        match program_ranges {
            Some(ranges) => {
                let mut per_program_status = Vec::new();
                for range in ranges {
                    let status = if fluxion_value >= range.min && fluxion_value <= range.max {
                        "PASS"
                    } else {
                        "FAIL"
                    };
                    per_program_status.push(ProgramStatus {
                        program: range.program.clone(),
                        status: status.to_string(),
                        min: range.min,
                        max: range.max,
                    });
                }

                // Overall status logic
                let overall = if per_program_status.iter().any(|p| p.status == "PASS") {
                    "PASS"
                } else if per_program_status.iter().any(|p| p.status == "FAIL" && p.program != "EnergyPlus") {
                    "WARN"  // Failed EnergyPlus but passed others
                } else {
                    "FAIL"
                };

                ValidationResult {
                    case_id: case_id.to_string(),
                    metric: metric.to_string(),
                    fluxion_value,
                    per_program_status,
                    overall: overall.to_string(),
                }
            }
            None => ValidationResult::no_reference(case_id, metric),
        }
    }
}
```

**3. Report Generation:**

Extend `ValidationReport`:
```rust
pub struct ValidationReport {
    pub cases: Vec<CaseValidationReport>,
    pub summary: ValidationSummary,
}

pub struct CaseValidationReport {
    pub case_id: String,
    pub metrics: Vec<MetricResult>,
}

pub struct MetricResult {
    pub metric_name: String,
    pub fluxion_value: f64,
    pub per_program: Vec<ProgramStatus>,
    pub overall_status: String,  // PASS/WARN/FAIL/NOREF
}
```

**4. Reference Updater CLI:**

`fluxion references update`:
- Fetch latest ASHRAE reference data from official publications (likely PDF → need manual transcription)
- Or download from curated JSON in repo (update via PR)
- Validate JSON schema before replacing `docs/ashrae_140_references.json`
- Backup previous version

**Implementation:** Simple file copy with version bump, or prompt user to provide path to new reference file.

### 7.4 Integration Points

- **Extend** `ASHRAE140Validator` to use `MultiReferenceValidator`
- **Modify** `BenchmarkData` loading to parse multi-program JSON
- **Report generation** (`docs/ASHRAE140_RESULTS.md`) displays per-program status columns

### 7.5 Risks & Challenges

- **Reference data availability:** ASHRAE 140 reference values are published in standard document; must be manually transcribed. ESP-r and TRNSYS values may be harder to find.
- **Versioning:** When ASHRAE updates standard (e.g., 2024 → 2027), need migration path
- **Program disagreement:** If EnergyPlus says 3.5 MWh and ESP-r says 4.2 MWh, which is correct? Document as "reference discrepancy" and recommend EnergyPlus as primary.

---

## 8. Cross-Cutting Concerns

### 8.1 Common Data Structures

**Long-format CSV:**
```csv
Hour,Zone,Component,Value_Watts,Unit
0,0,solar_beam,125.3,W
0,0,solar_diffuse,42.1,W
...
```
Advantages:
- Tidy data format (R/Pandas friendly)
- Easy filtering: `df[df.Component == 'solar_beam']`
- Pivot to wide format if needed

**Helper functions:**
```rust
pub fn write_long_csv<W: Write>(
    writer: &mut W,
    records: &[(usize, usize, &str, f64)],  // (hour, zone, component, value)
) -> Result<()> { ... }
```

### 8.2 Configuration Management

**YAML parsing:**
- Add `serde_yaml = "0.9"` to Cargo.toml (if not present)
- Define config structs with `#[derive(Deserialize)]`
- Validation: use `validator` crate or manual checks with clear error messages

**Config locations:**
- Global: `$HOME/.fluxion/config.yaml` (user assemblies, default settings)
- Project: `./fluxion.yaml` or `./cases/` directory
- CLI: `--config` flag

### 8.3 CLI Structure

Proposed subcommands in `src/bin/fluxion.rs`:

```bash
fluxion validate --all                    # Existing
fluxion validate --case 600 --diagnostics  # Extend to output CSV

fluxion sensitivity \
  --base case600.yaml \
  --params window_u,hvac_setpoint \
  --bounds "0.1:5.0,15:30" \
  --method oat \
  --levels 10 \
  --output sensitivity.md

fluxion delta --config comparisons.yaml

fluxion visualize \
  --input diagnostics.csv \
  --output report.html \
  --type time_series

fluxion animate \
  --input diagnostics.csv \
  --output animation.html \
  --speed 10

fluxion breakdown \
  --input diagnostics.csv \
  --output breakdown.csv \
  --format long

fluxion references update \
  --source new_references.json
```

### 8.4 Testing Strategy

**Unit tests:**
- Sensitivity: Test OAT sampling produces correct matrix dimensions; Sobol generates in [0,1]
- Delta: Test YAML parsing, patch application to CaseSpec
- Visualization: Test HTML generation contains required JS hooks (regex check)
- Component breakdown: Test conservation law (sum == total ±1%)
- Multi-reference: Test fallback logic (EnergyPlus FAIL + ESP-r PASS → WARN)

**Integration tests:**
- Run sensitivity on simple Case 600 and verify output format
- Generate delta report between Case 600 and variant; compare to known diffs
- Produce animation from 1-week diagnostic data; open in browser sanity check
- Validate multi-reference with mock reference data

**Performance tests:**
- Sensitivity with 1000 samples should complete in < batch oracle target (100ms)
- Visualization: HTML file size < 1MB (including embedded data)
- Delta: 100 variant comparisons < 1 minute

### 8.5 Performance Considerations

- **Sensitivity sampling:** Generate design matrix as single population → one `BatchOracle` call (critical)
- **Delta testing:** Sweeps expand to populations → parallel evaluation via `BatchOracle`
- **Visualization data:** Embed raw data in HTML (8760 × N numbers). Compress with gzip (servers do automatically). Alternative: downsample for plotting, full data hidden.
- **Component breakdown:** Compute from already-collected diagnostics; negligible cost

---

## 9. Implementation Phasing (Recommendation)

Given 20 requirements, recommend splitting Phase 7 into 2-3 subphases:

**Subphase A: Core Analysis (SENS + DELTA + COMP)**
- SENS-01..04: Sensitivity analysis engine
- DELTA-01..03: Delta testing framework
- COMP-01..03: Component breakdown (extend existing diagnostics)
- MREF-01..03: Multi-reference comparison schema (can be separate but related)

**Subphase B: Visualization (VIZ + SWING)**
- VIZ-01..04: HTML report generation with Chart.js
- SWING-01..03: Animation

**Subphase C: Extensibility (EXT)**
- EXT-01..04: Extensible case framework
- Documentation

**Rationale:** Build analysis first (needs diagnostics), then visualization (plots the analysis), then extensibility (last because it's API polish). Diagrams/building editors deferred.

---

## 10. Dependencies Summary

**Add to Cargo.toml:**
```toml
# Sensitivity analysis
linregress = "0.6"
sobol = "0.2"   # OR implement manually

# YAML config
serde_yaml = "0.9"

# Optional HTML templating (if not using manual string format)
askama = { version = "0.12", features = ["with-axum"] }  # or just "runtime"

# Optional server-side plotting fallback
plotters = "0.3"

# gbXML import
quick-xml = { version = "0.34", features = ["serialize"] }
```

**Already present:**
- `serde`, `serde_json`, `csv`, `ndarray`, `rand`, `clap`

---

## 11. Gap Analysis & Uncertainties

**Open questions requiring planner discretion:**

1. **Animation merging:** Should `animate` be separate or merged into `visualize`? (Context leaves to Claude's judgment)
   - **Recommendation:** Separate subcommand for clarity; animation is specialized use case

2. **Sobel parameters:** Skip count, dimension parameters?
   - **Recommendation:** Use standard defaults: 1024 samples for 2-parameter study; direction numbers from Joe-Kuo Sloan up to 2^10

3. **Adaptive refinement trigger:** When to follow up on sensitive parameters?
   - **Recommendation:** Post-hoc analysis: If normalized sensitivity > threshold (e.g., 50%), suggest user to sample more densely in that region

4. **Component breakdown column names:** Exact naming?
   - **Recommendation:** Use snake_case: `solar_beam`, `solar_diffuse`, `envelope_walls`, etc.

5. **Simplified geometry method names:** (e.g., `rectangular_zone` vs `single_zone_rectangle`)
   - **Recommendation:** Follow builder fluent pattern: `.with_rectangular_zone(length, width, height)`

6. **Default animation speed range:** 0.5× to 100×? Initial value?
   - **Recommendation:** Default 1× = 1 hour/sec; slider range [0.5, 100]; step 0.5

**Potential gaps:**
- **gbXML/IDF import complexity:** Might be larger than anticipated; consider delegating to external tool or Phase 8
- **WASM compilation:** If we want to compile visualization to WASM, need `wasm-bindgen` and careful with JS interop; keep as native binary generating HTML for now
- **Statistical significance testing:** Context says no hypothesis testing; keep simple (mean ± std)
- **Real-time parameter adjustment:** Out of scope (Phase 8+)

---

## 12. Risks Summary

| Risk | Impact | Mitigation |
|------|--------|------------|
| Physics engine doesn't expose detailed component breakdowns (solar beam vs diffuse) | COMP-01 blocked | Modify `ThermalModel::record_timestep` to capture breakdown; low complexity |
| YAML config validation is brittle | DELTA-01 poor UX | Use `validator` crate, detailed error messages with line numbers |
| Sobol sequence implementation tricky | SENS implementation delay | Use existing `sobol` crate; fallback to LHS if needed |
| Chart.js animation performance on 8760 points | VIZ-04 slow | Downsample for display (show weekly/daily resolution), full data on demand |
| gbXML parser large undertaking | EXT-03 scope creep | Defer gbXML/IDF import to Phase 8; start with YAML geometry + EPW custom |
| Multi-reference data not publicly available | MREF-01 manual effort | Document data source, provide empty template; users fill in their own reference values |

---

## 13. Success Criteria Alignment

| Requirement | Research Coverage |
|-------------|-------------------|
| **SENS-01..04** | OAT + Sobol via `sobol`/`rand`, `linregress` for regression, CVRMSE/NMBE manual, hybrid batching strategy |
| **DELTA-01..03** | Extend `fluxion_delta.rs`, YAML via `serde_yaml`, patch/sweep application on `CaseSpec`, long CSV format |
| **VIZ-01..04** | HTML+Chart.js via templates, export via JS `toDataURL`, animation with `setInterval` |
| **COMP-01..03** | Extend `HourlyData` with component fields, `to_long_csv()` format, conservation check in `ComponentBreakdown::from_hourly_data()` |
| **SWING-01..03** | Separate `animate` subcommand, Chart.js two-panel chart, speed/scrub controls |
| **EXT-01..04** | `CaseBuilder` extensions, `AssemblyLibrary` from YAML, EPW custom path, gbXML via `quick-xml` (defer if needed) |
| **MREF-01..03** | Multi-reference JSON schema, `MultiReferenceValidator`, per-program status, fallback logic, `references update` command |

All requirements addressed with concrete crates and implementation patterns.

---

## 14. Recommended Next Steps for Planner

1. **Prioritize Subphases:** Order A (Analysis) → B (Viz) → C (Extensibility)
2. **Task breakdown:** Each requirement becomes 1-2 tasks; some share modules
3. **Start with:** Sensitivity analysis (SENS-01) as it's the most novel and uses BatchOracle pattern
4. **Parallel work:** Delta testing and component breakdown can proceed concurrently (both use diagnostics)
5. **Validation:** Ensure all new code integrates with existing `DiagnosticCollector` and CLI structure
6. **Documentation:** Write API docs alongside implementation; examples in `docs/cases/`

---

**Appendix: Code Location Map**

| Module | Path | Purpose |
|--------|------|---------|
| Sensitivity engine | `src/analysis/sensitivity.rs` | OAT/Sobol sampling, metrics computation |
| Delta testing | `src/analysis/delta.rs` + `src/bin/delta.rs` | YAML config, variant comparison |
| Visualization | `src/visualization/mod.rs` | HTML generation, Chart.js integration |
| Component breakdown | `src/analysis/component_breakdown.rs` | Detailed energy allocation, CSV export |
| Animation | `src/animation/mod.rs` + `src/bin/animate.rs` | Time series animation |
| Extensibility | Extend `CaseBuilder` in `src/validation/ashrae_140_cases.rs` + new `assembly_library.rs` | Custom geometries, assemblies |
| Multi-reference | Extend `src/validation/ashrae_140/` | Per-program comparison logic |

---

*End of Phase 7 Research Document*
