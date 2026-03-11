# Phase 7 Plan 03: Component-Level Energy Breakdown & Swing Analysis - Summary

## One-Liner

Implemented component-level energy breakdown and temperature swing analysis utilities that process validation framework data to diagnose energy path errors and evaluate thermal mass effectiveness and passive cooling/heating potential.

## Implementation Details

### 1. Component Energy Breakdown (`src/analysis/components.rs`)

#### Core Types

```rust
/// Component entry for aggregated energy breakdown.
#[derive(Debug, Clone, Serialize)]
pub struct ComponentEntry {
    pub case_id: String,
    pub component: String,
    pub energy_mwh: f64,
}
```

Components tracked:
- `envelope_conduction`: Heat transfer through building envelope
- `infiltration`: Air leakage losses
- `solar_gains`: Solar radiation contributions
- `internal_gains`: Internal heat sources (people, equipment)
- `heating`: HVAC heating energy consumption
- `cooling`: HVAC cooling energy consumption

#### Aggregation API

```rust
pub fn aggregate_from_validator<I>(iter: I) -> Vec<ComponentEntry>
where
    I: Iterator<Item = (String, EnergyBreakdown)>,
```

Accepts an iterator over `(case_id, EnergyBreakdown)` pairs and emits a long-format vector of component entries. This design is flexible—works with any source that can produce EnergyBreakdown structs (validation runs, batch evaluations, etc.).

#### CSV Export

```rust
pub fn export_component_csv(entries: &[ComponentEntry], path: &Path) -> Result<(), Box<dyn std::error::Error>>
```

Writes component breakdown to CSV with headers: `Case,Component,Energy_MWh`. Long format means each row is a single component for a single case. Supports downstream analysis in pandas/R/Excel.

#### Conservation Validation

```rust
pub struct ConservationResult {
    pub net_balance_mwh: f64,
    pub tolerance_pct: f64,
    pub is_valid: bool,
}

pub fn check_conservation(breakdown: &EnergyBreakdown, tolerance_pct: f64) -> ConservationResult
```

Verifies first law of thermodynamics: `(solar + internal) - (heating + cooling) ≈ net_balance` (should be near zero). Tolerance defaults to 1% of total input energy. Flags data quality issues or modeling errors.

#### Unit Tests (3 passing)

- `test_aggregate_and_export`: Creates mock EnergyBreakdown for single case, verifies 6 component entries generated with correct values.
- `test_conservation`: Balanced data (net ≈ 0) passes.
- `test_conservation_fail`: Unbalanced data (net exceeds tolerance) fails.

---

### 2. Temperature Swing Analysis (`src/analysis/swing.rs`)

#### Swing Metrics

```rust
#[derive(Debug, Clone, Serialize)]
pub struct SwingMetrics {
    pub case_id: String,
    pub min_temp: f64,
    pub max_temp: f64,
    pub avg_temp: f64,
    pub swing_range: f64,              // max - min
    pub comfort_hours: usize,
    pub comfort_hours_pct: f64,
}
```

#### Calculation

```rust
pub fn calculate_swing_metrics(
    profile: &TemperatureProfile,
    comfort_band_min: f64,
    comfort_band_max: f64,
) -> SwingMetrics
```

Processes a `TemperatureProfile` (hourly free-floating temperatures) and computes:
- Temperature extremes and mean
- Swing range (indicates thermal mass damping effectiveness)
- Comfort hours within configurable band (default 18–26°C) and percentage

Comfort band aligns with ASHRAE 55/ISO 7730 adaptive comfort standards for naturally ventilated buildings.

#### CSV Export

```rust
pub fn export_swing_csv(metrics: &[SwingMetrics], path: &Path) -> Result<(), Box<dyn std::error::Error>>
```

Exports swing metrics with headers: `Case,Min_Temp,Max_Temp,Avg_Temp,Swing_Range,Comfort_Hours,Comfort_Percent`.

---

### 3. Swing Interpretation & Diagnostics

#### Interpretation Structure

```rust
#[derive(Debug, Clone, Serialize)]
pub struct SwingInterpretation {
    pub case_id: String,
    pub thermal_mass_effectiveness: String, // "High", "Moderate", "Low"
    pub passive_cooling_potential: String,
    pub passive_heating_potential: String,
    pub recommendations: Vec<String>,
}
```

#### Classification Logic

**Thermal Mass Effectiveness:**
- **High**: swing_range < 5°C AND comfort_hours_pct ≥ 80% (mass stabilizes temperatures)
- **Moderate**: swing_range < 10°C AND comfort_hours_pct ≥ 50%
- **Low**: swing_range ≥ 10°C OR comfort_hours_pct < 50% (insufficient mass or extreme climate)

**Passive Cooling Potential:**
- **High**: comfort_hours_pct ≥ 70% AND avg_temp ∈ [18, 26]°C
- **Moderate**: comfort_hours_pct ∈ [40, 70%) AND avg_temp ≤ 28°C
- **Low**: comfort_hours_pct < 40% OR avg_temp > 28°C

**Passive Heating Potential:**
- **High**: comfort_hours_pct ≥ 70% AND avg_temp ≥ 18°C
- **Moderate**: comfort_hours_pct ∈ [40, 70%) AND avg_temp ∈ [15, 18)°C
- **Low**: comfort_hours_pct < 40% OR avg_temp < 15°C

#### Recommendations Engine

Generates actionable suggestions based on classifications:
- Low thermal mass effectiveness → "Consider increasing thermal mass or improving insulation"
- Low passive cooling potential with high avg temp → "Improve shading, increase ventilation, or reduce window-to-wall ratio"
- Low passive heating potential with low avg temp → "Increase solar gain (south-facing windows), add thermal mass, or improve insulation"

#### Report Generation

```rust
pub fn generate_swing_report(interpretations: &[SwingInterpretation]) -> String
```

Returns Markdown-formatted table summarizing cases and recommendations.

#### Unit Tests (4 passing)

- `test_swing_metrics_basic`: Synthetic profile (15, 25, 18) → min=15, max=25, avg=19.33, swing=10, comfort=66.7%
- `test_interpretation_high`: swing=4°C, comfort=91% → all "High"
- `test_interpretation_low`: swing=25°C, comfort=11.4% → all "Low" with recommendations
- `test_generate_swing_report`: Markdown contains expected headers and case IDs

---

## Artifacts Deliverable

**Files Modified/Created:**

| File | Lines | Purpose |
|------|-------|---------|
| `src/analysis/components.rs` | ~120 | Component aggregation, CSV export, conservation checks |
| `src/analysis/swing.rs` | ~220 | Swing metrics, interpretation, report generation |
| (Supporting fixes) `src/validation/assembly_library.rs` | ~98 | Fixed imports and Construction API usage |
| (Supporting fixes) `src/validation/ashrae_140_cases.rs` | +2 imports, 1 line change | Added ConstructionLayer import, fixed Materials::fiberglass call |

**Dependencies:** No new dependencies; uses existing `csv` and `serde` crates.

---

## Example Usage

### Component Breakdown

```rust
use fluxion::analysis::components::{aggregate_from_validator, export_component_csv};
use fluxion::validation::{ASHRAE140Validator, DiagnosticCollector};

// Run validation and collect EnergyBreakdowns
let validator = ASHRAE140Validator::new();
let report = validator.validate_analytical_engine();

// Convert report.energy_breakdowns HashMap to iterator
let entries = aggregate_from_validator(
    report.energy_breakdowns.iter()
        .map(|(id, breakdown)| (id.as_str(), breakdown.clone()))
);

// Export to CSV
export_component_csv(&entries, Path::new("component_breakdown.csv"))?;
```

### Swing Analysis

```rust
use fluxion::analysis::swing::{calculate_swing_metrics, interpret_swing_metrics, generate_swing_report};
use fluxion::validation::diagnostic::TemperatureProfile;

// For free-floating cases (no HVAC), collect temperature profile
let mut profile = TemperatureProfile::new("600FF");
for (hour, temp) in free_floating_temps.iter().enumerate() {
    profile.update(*temp);
}
profile.finalize();

// Compute metrics
let metrics = calculate_swing_metrics(&profile, 18.0, 26.0);

// Interpret
let interpretation = interpret_swing_metrics(&metrics);

// Generate report for multiple cases
let report = generate_swing_report(&interpretations_vec);
std::fs::write("swing_analysis.md", report)?;
```

---

## Verification

✅ **Build**: `cargo check` compiles without errors

✅ **Unit Tests**: All 7 tests in components and swing modules pass

```
cargo test --lib analysis::components  # 3 passed
cargo test --lib analysis::swing        # 4 passed
```

✅ **Integration**: Modules depend only on existing `validation::diagnostic` types (EnergyBreakdown, TemperatureProfile), which are already collected by the validation framework.

---

## Key Decisions

- **Long format CSV**: One row per component-case pair enables easy pivot/group in analysis tools.
- **Separate interpretation module**: Pure function `interpret_swing_metrics` enables programmatic use without generating Markdown.
- **Comfort band default 18–26°C**: Aligns with ASHRAE 55 adaptive comfort for naturally ventilated buildings; parameterized to allow customization.
- **Conservative thresholds**: Thermal mass effectiveness uses swing < 5°C for "High" based on typical residential building damping; can be tuned for commercial buildings with larger thermal mass.

---

## Next Steps

- **Plan 07-02** (Delta Testing Framework): Complete stub in `src/analysis/delta.rs` (Wave 2, parallel)
- **Plan 07-04** (Interactive Visualization): Build on `src/analysis/visualization.rs`
- **Plan 07-07** (Extended CaseBuilder API): Simplify custom case creation
- **Plan 07-06** (CLI Integration): Wire all analysis features to `fluxion` binary

The utilities implemented here provide diagnostic insights for validation engineers to understand energy component contributions and evaluate passive design strategies.
