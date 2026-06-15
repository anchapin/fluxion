# ASHRAE 140 Program Description

## Software Identification

| Field | Value |
|---|---|
| **Software Name** | Fluxion |
| **Version** | 1.0.0 |
| **Description** | A differentiable, AI-accelerated Building Energy Modeling (BEM) engine with multi-zone support |
| **Language** | Rust (Edition 2021) |
| **License** | Apache-2.0 |
| **Repository** | https://github.com/anchapin/fluxion |

## Intended Use Cases

Fluxion is designed for:

1. **Building Thermal Envelope Analysis** — Annual heating and cooling load calculations for single-zone and multi-zone buildings per ISO 13790 and ASHRAE 140 methodologies.
2. **Energy Code Compliance** — Validation against ASHRAE Standard 140 inter-program comparison test cases to demonstrate modeling accuracy.
3. **Design Optimization** — Differentiable thermal models enabling gradient-based optimization of envelope parameters (U-values, glazing ratios, thermal mass).
4. **Rapid Annual Simulation** — Reduced-order models (ROM) and AI-accelerated surrogates for fast parametric studies and design-space exploration.
5. **Multi-Zone Thermal Analysis** — 6R2C coupled thermal networks for inter-zone heat transfer in multi-room buildings.

## Modeling Approach Overview

Fluxion implements a hierarchical solver architecture for building thermal simulation:

### Heat Conduction Solvers

| Solver | Fidelity | Speed | Use Case |
|---|---|---|---|
| **5R1C** (ISO 13790) | Standard | Fastest | Single-zone annual loads, ASHRAE 140 compliance |
| **CTF** (Conduction Transfer Functions) | High | Moderate | Detailed transient conduction, ASHRAE 140 high-mass cases |
| **Finite Difference** | Highest | Slowest | Research-grade multi-node conduction |

The default solver for ASHRAE 140 Section 7 test cases is the **5R1C thermal network model** (ISO 13790:2008, Section 7), which represents the building thermal envelope as a five-resistance, one-capacitance (5R1C) network.

### Supporting Models

- **Solar gain calculation**: Position-based solar geometry with incident angle-dependent transmittance
- **Shading**: Geometric overhang and vertical fin shadow projection with inclusion-exclusion overlap handling
- **Ground temperature**: Constant (10°C per ASHRAE 140) or dynamic (Kusuda-Achenbach)
- **HVAC**: Ideal loads system (100% efficiency, infinite capacity) per EnergyPlus terminology
- **Weather**: Embedded Denver TMY data (39.83°N, 104.65°W, 1655m elevation) or EPW file import
- **Sky radiation**: Long-wave radiation exchange model for opaque surface heat loss

### Solver Registry

The solver registry (`src/physics/solver_registry.rs`) dynamically selects the appropriate conduction solver based on building characteristics and required fidelity. For ASHRAE 140 compliance runs, the 5R1C solver is the baseline method.

## ASHRAE 140 Compliance Scope

Fluxion implements **52 test cases** from ASHRAE 140-2023 Section 7 — Building Thermal Envelope and Fabric Load Tests, validated against the inter-program comparison reference ranges from six approved programs (BSIMAC, CSE, DeST, EnergyPlus, ESP-r, TRNSYS).

### Test Case Coverage

| Series | Description | Cases |
|---|---|---|
| 195–220 | Solid conduction / low-mass base | 195, 200, 210, 220 |
| 600–650 | Low-mass with windows and shading | 600, 610, 620, 630, 640, 650 |
| 800–810 | Opaque envelope variations | 800, 810 |
| 900–970 | High-mass with windows and shading | 900, 910, 920, 930, 940, 950, 960, 970 |
| FF variants | Free-floating temperature cases | 600FF, 650FF, 900FF, 950FF |

### Validation Metrics

Per ASHRAE 140 Section 8, the following metrics are validated:

- **Annual heating load** (MWh) — Table B8-1
- **Annual sensible cooling load** (MWh) — Table B8-2
- **Peak hourly integrated heating load** (kW) — Table B8-3
- **Peak hourly integrated sensible cooling load** (kW) — Table B8-4
- **Free-floating maximum annual zone temperature** (°C) — Table B8-5
- **Free-floating minimum annual zone temperature** (°C) — Table B8-5
- **Free-floating mean annual zone temperature** (°C) — Table B8-5

## Technical Capabilities

### Differentiable Simulation

Fluxion's thermal models are implemented with automatic differentiation support, enabling:
- Sensitivity analysis of envelope parameters
- Gradient-based optimization of building design variables
- Integration with AI/ML surrogate training pipelines

### Multi-Zone Support

The 6R2C extension adds a second thermal capacitance node and sixth resistance path, enabling:
- Inter-zone heat transfer through internal walls
- Coupled zone temperature evolution
- Multi-room building simulation with shared boundary conditions

### Binding Layers

| Interface | Feature Flag | Status |
|---|---|---|
| Rust API | Default | Stable |
| Python (PyO3) | `python` | Available |
| Node.js (NAPI) | `nodejs` | Available |
| CLI | Default | Stable |
| FMI co-simulation | `fmi` | Experimental |

## References

- ASHRAE Standard 140-2023: Standard Method of Test for Building Energy Simulation Programs
- ISO 13790:2008: Energy performance of buildings — Calculation of energy use for space heating and cooling
- Std140_TF_Results.pdf (TESS, 19-Aug-2024): Inter-program comparison reference results
