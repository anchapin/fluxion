# FMI Interoperability Module

## Overview

The FMI (Functional Mock-up Interface) interoperability module provides building energy simulation interoperability capabilities for Fluxion, enabling integration with other energy modeling tools and workflows.

## Scope

This module generates FMI 2.0 Co-Simulation FMUs (`.fmu` archives) for Fluxion thermal models. The export pipeline is documented by `src/interop/fmi/mod.rs` and validated against the official FMI 2.0 XSD set (`fmi2ModelDescription.xsd`).

### Export Mode (Fluxion → FMU)

- **Multi-zone thermal network** — per-zone (outdoor_temperature,
  direct_normal_solar, diffuse_horizontal_solar, internal_gains)
  inputs and (zone_temperature, heating_load, cooling_load) outputs
  — i.e. **7 × N** ScalarVariables for **N** zones.
- **Configurable communication timestep** — set via
  [`FmiConfig::communication_timestep`]; accepts 60 s, 300 s, 600 s,
  or 3600 s.  Default 3600 s preserved for backward compatibility
  with the original single-zone spike (#1125).
- **Validation** — the emitted `modelDescription.xml` is validated
  against the official FMI 2.0 XSD set by the verification script
  at `.agents/results/issue-D1-multi-zone-fmi-verification.py`.
- **Standalone FMU** — declared with `needsExecutionTool="true"`, so
  no platform binary is shipped; the master tool drives the
  simulation by calling into Fluxion for each `fmi2DoStep`.

### Co-Simulation Mode

- **Master algorithm**: First-order Euler (simplified)
- **Communication timestep**: configurable (60 s / 300 s / 600 s / 3600 s)
- **Requires external FMU** for weather data or advanced controls

## Architecture

```
fluxion::interop
└── fmi
    ├── FmiConfig          - Configuration (timestep, model name, GUID, ...)
    ├── FmiExporter        - Export Fluxion models as FMU
    ├── FmiMode            - FMI execution mode (Export/Import/Co-simulation)
    ├── FmiVariables       - Per-zone variable name templates (4 inputs + 3 outputs)
    └── ZoneVariables      - Per-zone label (name) for the multi-zone interface
```

## Usage

### Single-zone FMU (legacy spike, #1125)

```rust,ignore
use fluxion::interop::fmi::{FmiExporter, FmiConfig};

let config = FmiConfig::default(); // 1 zone, 3600 s timestep
let exporter = FmiExporter::with_config(config)?;
exporter.export_fmu("fluxion_building.fmu")?;
```

The exported FMU exposes the 7 default variables: `outdoor_temperature`,
`direct_normal_solar`, `diffuse_horizontal_solar`, `internal_gains`,
`zone_temperature`, `heating_load`, `cooling_load`.

### Multi-zone FMU (#1339)

```rust,ignore
use fluxion::interop::fmi::{FmiExporter, FmiConfig, ZoneVariables};

let mut cfg = FmiConfig::default();
cfg.communication_timestep = 300.0; // 5-minute communication timestep

let exporter = FmiExporter::with_config(cfg)?
    .with_zones(vec![
        ZoneVariables::new("zone"),    // zone 0 — keeps bare template names
        ZoneVariables::new("bedroom"), // zone 1 — variables prefixed `bedroom_`
        ZoneVariables::new("kitchen"), // zone 2 — variables prefixed `kitchen_`
    ]);

exporter.export_fmu("fluxion_three_zone.fmu")?;
```

The exported FMU exposes **21** ScalarVariables (3 zones × 7), with
unique `valueReference` indices for every variable.  The master tool
addresses inputs and outputs either by `name` (e.g. `bedroom_zone_temperature`)
or by `valueReference`.

### Configurable timestep

```rust,ignore
use fluxion::interop::fmi::FmiConfig;

let cfg = FmiConfig { communication_timestep: 60.0, ..FmiConfig::default() };
```

The timestep is validated to be positive (see
[`FmiExporter::with_config`]) and is forwarded to:

* the `<DefaultExperiment stepSize="…">` element,
* the `<CoSimulation …>` capability flags
  (`canHandleVariableCommunicationStepSize="true"`).

## FMI Standard

Implements **FMI 2.0** for:

* **Co-Simulation** (export mode, `needsExecutionTool="true"`)

See: https://fmi-standard.org/

## Validation

The XML produced by `FmiExporter::generate_model_description_xml()` is
validated against the official FMI 2.0 XSD set by:

```text
python .agents/results/issue-D1-multi-zone-fmi-verification.py
```

The script:

1. Builds a 3-zone FMU via a temporary Cargo harness crate.
2. Extracts `modelDescription.xml` from the FMU archive.
3. Parses the XML and asserts the acceptance-criteria contracts
   (variable count, input/output split, valueReferences, default
   stepSize, ModelStructure.Outputs).
4. Validates the XML against `fmi2ModelDescription.xsd` (lxml).

If PyFMI 2.x or FMPy 0.3.x are installed locally, the script also
attempts to instantiate the FMU as a runtime acceptance gate.

## Limitations

1. **Co-Simulation only** — the FMU is exported as a tool-driven
   Co-Simulation; Model Exchange (`FmiMode::ModelExchange`) is not
   yet wired through `FmiExporter`.
2. **No FMU import** — `FmiMode::Import` is reserved for a future
   issue; out of scope for #1339.
3. **No FMI 3.0 features** — Hybrid Co-Simulation, terminals, and
   other FMI 3.0 capabilities are out of scope until upstream tooling
   (FMPy, PyFMI) stabilizes around them.

## Future Extensions

1. Wire `FmiMode::Import` for FMU loading.
2. Higher-order master algorithms (RK4, etc.).
3. Expanded input variable set (e.g. per-zone shading).
4. FMI 3.0 once upstream tooling supports it.

## API Reference

See `src/interop/fmi/mod.rs` for complete API documentation.