# Inter-Program Comparison Charts

**Product:** fluxion v0.8.0+
**Document Type:** Inter-Program Comparison Charts
**Maintained by:** Building Standards Engineer
**Last Updated:** 2026-06-16
**Related Issue:** #750

---

## 1. Overview

ASHRAE 140-2023 Section 1.5 requires inter-program comparison as part of the compliance submission. This document describes:

1. The reference programs used for comparison
2. The chart generation implementation in fluxion
3. How to generate comparison charts
4. Current status and implementation notes

---

## 2. Reference Programs

fluxion's ASHRAE 140 validation compares against multiple reference building energy simulation programs:

| Program | Full Name | Organization | Role |
|---------|-----------|--------------|------|
| EnergyPlus | EnergyPlus | DOE/NREL | **Primary reference** — used for pass/fail determination |
| ESP-r | ESP-r | University of Strathclyde | Research-grade reference |
| TRNSYS | TRNSYS | TRANSSOLAR | Industry reference |
| DOE-2 | DOE-2 | LBL/DOE | Legacy reference |

**Pass/fail rule (per ASHRAE 140-2023 Section 1.5):**
- **PASS:** fluxion result within the inter-program envelope AND EnergyPlus is within its own envelope
- **WARN:** EnergyPlus outside its envelope but another program within
- **FAIL:** All reference programs outside the envelope

---

## 3. Chart Generation Implementation

### 3.1 Code Location

Chart generation is implemented in `src/validation/report.rs` using the `plotters` crate:

```rust
// src/validation/report.rs
use plotters::backend::BitMapBackend;
use plotters::drawing::IntoDrawingArea;
use plotters::prelude::*;
```

### 3.2 Implemented Chart Functions

The following chart generation methods exist in `BenchmarkReport`:

| Method | Purpose | Status |
|--------|---------|--------|
| `generate_temperature_profile_plot()` | Hourly temperature profile visualization | **Placeholder** — stub only |
| `generate_energy_comparison_chart()` | Bar chart: fluxion vs reference programs | **Partially implemented** — uses hardcoded 0-20 MWh range |
| `generate_heat_transfer_visualization()` | Line chart: inter-zone heat transfer | **Placeholder** — stub only |

### 3.3 `generate_energy_comparison_chart` — Current Implementation

```rust
// src/validation/report.rs:1847-1889
pub fn generate_energy_comparison_chart(
    &self,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "ASHRAE 140 Multi-Zone Energy Comparison",
            ("sans-serif", 50).into_font(),
        )
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..2, 0f64..20f64)?;

    chart.configure_mesh().draw()?;

    // Only plots Case 960 and Case 970 heating values
    // Hardcoded y-axis range (0-20 MWh) — not adaptive
    chart.draw_series(
        Histogram::vertical(&chart)
            .style(RED.filled())
            .data(vec![(0, case_960_heating), (1, case_970_heating)]),
    )?;

    Ok(())
}
```

**Issues with current implementation:**
1. Only plots Case 960 and Case 970 (hardcoded)
2. Y-axis range is hardcoded (0-20 MWh) — not adaptive to data range
3. Does not include reference program bars (EnergyPlus, ESP-r, TRNSYS, DOE-2)
4. No per-program color coding
5. X-axis labels not set

### 3.4 `generate_temperature_profile_plot` — Placeholder

```rust
// src/validation/report.rs:1829-1845
pub fn generate_temperature_profile_plot(
    &self,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Placeholder — writes a text file instead of generating a chart
    let mut file = std::fs::File::create(path)?;
    file.write_all(
        b"Temperature profile visualization would be generated here in a full implementation\n",
    )?;
    Ok(())
}
```

### 3.5 `generate_heat_transfer_visualization` — Placeholder

```rust
// src/validation/report.rs:1891-1915
pub fn generate_heat_transfer_visualization(
    &self,
    &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Plots a flat line at y=0 — placeholder
    chart.draw_series(LineSeries::new(vec![(0.0, 0.0), (8760.0, 0.0)], &BLACK))?;
    Ok(())
}
```

---

## 4. Required Chart Types for Compliance

ASHRAE 140-2023 compliance submission requires the following chart types:

### 4.1 Annual Energy Consumption Bar Charts

**Required:** Bar charts comparing fluxion annual energy (heating + cooling) against reference program ranges for all 14 standard cases.

**Data requirements:**
- Fluxion value (annual heating, annual cooling)
- Reference program ensemble minimum and maximum
- EnergyPlus specific value and range
- Color coding: fluxion (blue), EnergyPlus (green), other programs (gray)

**Chart per case:**
- Case 600 (Low-mass, No Solar)
- Case 610 (Low-mass, Fixed Solar)
- Case 620 (Low-mass, Movable Insulation)
- Case 630 (Low-mass, Exterior Shading)
- Case 640 (Low-mass, Overhangs)
- Case 650 (Low-mass, Combined)
- Case 900 (High-mass, No Solar)
- Case 910 (High-mass, Fixed Solar)
- Case 920 (High-mass, Movable Insulation)
- Case 930 (High-mass, Exterior Shading)
- Case 940 (High-mass, Overhangs)
- Case 950 (High-mass, Combined)
- Case 960 (Two-Zone)
- Case 970 (Multi-Zone Framework)

### 4.2 Peak Load Comparison

**Required:** Bar charts for peak heating and peak cooling loads.

### 4.3 Free-Float Temperature Profile

**Required:** Line chart showing hourly zone temperature for free-float cases (600FF, 650FF, 900FF, 950FF) vs reference range.

---

## 5. Multi-Reference Data

fluxion's `MultiReferenceDB` (`src/validation/multi_reference.rs`) provides per-program reference ranges:

```rust
// src/validation/multi_reference.rs
pub struct ProgramRange {
    pub min: f64,
    pub max: f64,
    pub programs: Vec<String>,
}

pub struct CaseReferences {
    pub annual_heating: Option<HashMap<String, ProgramRange>>,
    pub annual_cooling: Option<HashMap<String, ProgramRange>>,
    pub peak_heating: Option<HashMap<String, ProgramRange>>,
    pub peak_cooling: Option<HashMap<String, ProgramRange>>,
}
```

This data structure enables per-program bar charts — the infrastructure exists but the chart generation code does not use it.

---

## 6. How to Generate Charts

### 6.1 From Rust Code

```rust
use fluxion::validation::{ASHRAE140Validator, BenchmarkReport};

let mut report = BenchmarkReport::new();
// ... run validation and populate report ...

// Generate comparison chart
report.generate_energy_comparison_chart("output/energy_comparison.png")?;
report.generate_temperature_profile_plot("output/temperature_profile.png")?;
```

### 6.2 From CLI

```bash
# Run ASHRAE 140 validation and generate charts
cargo run --release --bin fluxion -- validate ashrae-140 --output-dir ./charts
```

Note: The `--output-dir` flag for chart generation may not yet be implemented — verify in `src/bin/fluxion.rs`.

### 6.3 From CI (GitHub Actions)

The `ashrae_benchmark_harness.yml` workflow generates structured JSON results but does not currently generate visual charts. The chart generation would need to be added as a separate step.

---

## 7. Current Gaps

| Gap | Severity | Description |
|-----|----------|-------------|
| Reference program bars not plotted | HIGH | Charts show only fluxion values; no reference programs |
| Y-axis hardcoded | HIGH | 0-20 MWh range may not fit all data |
| Per-case charts not generated | HIGH | Only Case 960/970 hardcoded |
| Free-float temperature chart | MEDIUM | Placeholder — not implemented |
| Heat transfer chart | MEDIUM | Placeholder — not implemented |
| CI chart generation | MEDIUM | No automated chart generation in workflow |
| Chart output format | LOW | PNG only — consider SVG for publication quality |

---

## 8. Implementation Requirements

To complete inter-program comparison charts for compliance:

1. **Extend `generate_energy_comparison_chart`:**
   - Add per-program reference bars (EnergyPlus, ESP-r, TRNSYS, DOE-2)
   - Make y-axis range adaptive from data
   - Support all 14 standard cases (not just 960/970)
   - Set proper x-axis labels
   - Use correct color coding

2. **Implement `generate_temperature_profile_plot`:**
   - Use plotters `LineSeries` for hourly temperature
   - Plot reference range as a shaded area (min-max band)
   - Add fluxion line
   - Support free-float cases (600FF, 650FF, 900FF, 950FF)

3. **Add chart generation to CI pipeline:**
   - Update `ashrae_benchmark_harness.yml` to generate charts as artifacts
   - Upload PNG artifacts alongside JSON results

4. **Add SVG output option:**
   - plotters supports SVG backend for publication-quality output

---

## 9. Related Documentation

- `src/validation/report.rs` — Chart generation implementation
- `src/validation/multi_reference.rs` — Per-program reference data
- `src/validation/ashrae_140_cases.rs` — Case specifications
- `deviations-register.md` — DEV-017 tracks this documentation task

---

## 10. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-06-16 | Building Standards Engineer | Initial version for Issue #750 |

---

*End of Inter-Program Comparison Charts*
