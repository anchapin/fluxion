# FMI Interoperability Module

## Overview

The FMI (Functional Mock-up Interface) interoperability module provides building energy simulation interoperability capabilities for Fluxion, enabling integration with other energy modeling tools and workflows.

## Scope

This is a **constrained initial implementation** (IO-01 spike) developed according to Issue #518 requirements. The implementation provides a foundation for FMI interoperability with the following known scope limits:

### Export Mode (Fluxion → FMU)

- **Single-zone thermal network only** - Multi-zone support is not included
- **Fixed timestep** - 3600s (1 hour) communication timestep only
- **Outputs**: Zone temperature (K), heating load (W), cooling load (W)
- **Inputs**: Outdoor temperature (K), direct normal solar (W/m²), diffuse horizontal solar (W/m²), internal gains (W)

### Co-Simulation Mode

- **Master algorithm**: First-order Euler (simplified)
- **Communication timestep**: 1 hour (3600s)
- **Requires external FMU** for weather data or advanced controls

## Architecture

```
fluxion::interop
└── fmi
    ├── FmiConfig      - Configuration for FMI operations
    ├── FmiExporter    - Export Fluxion models as FMU
    ├── FmiMode        - FMI execution mode (Export/Import/Co-simulation)
    └── FmiVariables   - FMI variable definitions
```

## Usage

### Creating an FMI Exporter

```rust
use fluxion::interop::fmi::{FmiExporter, FmiConfig};

let config = FmiConfig::default();
let exporter = FmiExporter::with_config(config)?;
```

### Exporting an FMU

```rust
use fluxion::interop::fmi::FmiExporter;

let exporter = FmiExporter::new();
exporter.export_fmu("fluxion_building.fmu")?;
```

## FMI Standard

Implements **FMI 2.0** for:
- **Co-Simulation** (export mode)
- **Model Exchange** (import mode)

See: https://fmi-standard.org/

## Known Limitations

1. **Single-zone only**: Multi-zone thermal networks not supported in initial release
2. **Fixed timestep**: Variable timestep not supported
3. **No multi-step solver**: Uses first-order Euler, not higher-order methods
4. **Limited inputs**: Solar position calculated internally, not from FMU
5. **No Windows binaries**: Initial release supports POSIX systems only

## Future Extensions

Planned improvements for post-spike development:

1. Multi-zone thermal network support
2. Variable communication timestep
3. Higher-order master algorithms (RK4, etc.)
4. Expanded input variable set
5. Cross-platform FMU binary generation
6. Real-time co-simulation capability

## API Reference

See `src/interop/fmi/mod.rs` for complete API documentation.
