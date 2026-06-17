# Fluxion CLI Interface Design for EnergyPlus/OpenStudio Compatibility

## Issue
**#784**: CLI interface compatibility with OpenStudio/EnergyPlus workflow patterns

## Overview

This document proposes a redesigned CLI interface for Fluxion that follows conventions familiar to EnergyPlus and OpenStudio users, while maintaining backward compatibility with existing commands.

## Current CLI Analysis

### Existing Command Structure

```
fluxion <COMMAND>

Commands:
  references     Manages reference data for validation
  validate       Validates the engine against ASHRAE Standard 140
  validate-case  Validate specific diagnostic case or range
  quantize       Quantize an ONNX model for optimized edge inference
  benchmark      Run inference benchmark on an ONNX model
  sensitivity    Run sensitivity analysis
  delta          Run delta testing comparison
  components     Generate component energy breakdown for a case
  swing          Calculate and display swing metrics for a free-floating case
  visualize      Generate interactive visualization from diagnostics CSV
  animate        Generate animated visualization from diagnostics CSV
```

### Current Issues

1. **No direct simulation capability**: Users must use subcommands for specific tasks
2. **Inconsistent with EnergyPlus patterns**: EnergyPlus uses positional input file
3. **No workflow file support**: OpenStudio uses `.osw` files
4. **Missing familiar options**: `-w` for weather, `-d` for output directory, etc.

## EnergyPlus/OpenStudio CLI Patterns

### EnergyPlus CLI Pattern

```bash
energyplus [options] [input-file]

Options:
  -a, --annual                 Force annual simulation
  -c, --convert                Output IDF->epJSON or epJSON->IDF
  -d, --output-directory ARG   Output directory path
  -D, --design-day             Force design-day-only simulation
  -h, --help                   Display help
  -i, --idd ARG                Input data dictionary path
  -j, --jobs ARG               Multi-thread with N threads
  -m, --epmacro                Run EPMacro prior to simulation
  -p, --output-prefix ARG      Prefix for output file names
  -r, --readvars               Run ReadVarsESO after simulation
  -s, --output-suffix ARG      Suffix style for output file names
  -v, --version                Display version
  -w, --weather ARG             Weather file path
  -x, --expandobjects          Run ExpandObjects prior to simulation

Example: energyplus -w weather.epw -r input.idf
```

### OpenStudio CLI Pattern

```bash
openstudio <program-options> <subcommand> [subcommand-options]

Subcommands:
  measure   Manage and query measures
  run       Run complete simulation workflow

Example: openstudio run -w workflow.osw
```

### OpenStudio Workflow (OSW) Structure

```json
{
  "seed_file": "baseline.osm",
  "weather_file": "USA_CO_Golden-NREL.724666_TMY3.epw",
  "steps": [
    {"measure_dir_name": "IncreaseWallRValue", "arguments": {}},
    {"measure_dir_name": "SetEplusInfiltration", "arguments": {"flowPerZoneFloorArea": 10.76}}
  ]
}
```

## Proposed Fluxion CLI Design

### Design Principles

1. **EnergyPlus compatibility**: Direct simulation mode with familiar options
2. **OpenStudio workflow support**: Fluxion Workflow (FWF) JSON format
3. **Backward compatibility**: All existing commands preserved
4. **Progressive disclosure**: Simple usage for beginners, advanced features for experts

### New Command Structure

```
fluxion [OPTIONS] [input-file]

Direct Simulation Mode (EnergyPlus-compatible):
  fluxion -w weather.epw input.flux
  fluxion -w weather.epw -d output/ input.flux
  fluxion --annual -w weather.epw input.flux

Workflow Mode (OpenStudio-compatible):
  fluxion run -w workflow.fwf
  fluxion measure --update /path/to/measures/

Analysis Commands (existing, preserved):
  fluxion validate [--case 600]
  fluxion sensitivity --config sens.yaml
  fluxion delta --config delta.yaml
```

### Option Mapping (EnergyPlus-compatible)

| Fluxion Option | EnergyPlus | Purpose |
|---------------|------------|---------|
| `-w, --weather <path>` | `-w` | Weather file (EPW) |
| `-d, --output-directory <path>` | `-d` | Output directory |
| `-p, --output-prefix <prefix>` | `-p` | Output file prefix |
| `-s, --output-suffix <style>` | `-s` | Output suffix style (L/C/D) |
| `-D, --design-day` | `-D` | Design day only simulation |
| `-a, --annual` | `-a` | Force annual simulation |
| `-i, --idd <path>` | `-i` | Input data dictionary (for validation) |
| `-j, --jobs <n>` | `-j` | Number of parallel jobs |
| `-r, --readvars` | `-r` | Run post-processing |
| `-x, --expandobjects` | `-x` | Pre-process input |
| `--convert` | `-c` | Convert between formats |
| `--version` | `-v` | Show version |
| `--help` | `-h` | Show help |

### Fluxion Workflow Format (FWF)

Inspired by OpenStudio's OSW:

```json
{
  "version": "1.0",
  "name": "Baseline Simulation",
  "description": "ASHRAE 140 Case 600",
  "seed_file": "case600.flux",
  "weather_file": "USA_CO_Denver.epw",
  "steps": [
    {
      "measure_type": "model",
      "measure_dir_name": "increase_wall_rvalue",
      "arguments": {"r_value": 45}
    },
    {
      "measure_type": "simulation",
      "measure_dir_name": "set_infiltration",
      "arguments": {"ach": 0.5}
    },
    {
      "measure_type": "reporting",
      "measure_dir_name": "monthly_summary",
      "arguments": {"output_format": "csv"}
    }
  ],
  "simulation_control": {
    "run_period": "annual",
    "timestep": 4,
    "convergence_tolerance": 0.001
  }
}
```

### Measure Types

1. **Model Measures**: Modify the building model before simulation
2. **Simulation Measures**: Modify EnergyPlus IDF directly
3. **Reporting Measures**: Generate reports from simulation results

## Implementation Phases

### Phase 1: EnergyPlus-compatible Direct Mode

Add support for:
- Positional input file argument
- `-w` weather file option
- `-d` output directory option
- `-p` output prefix option
- `-D` design-day only
- `-a` annual simulation (default)
- `-h` / `--help` and `-v` / `--version`

### Phase 2: Workflow Support

- Define FWF JSON schema
- Implement `fluxion run -w workflow.fwf`
- Support for measure steps

### Phase 3: Measure Management

- `fluxion measure --update <dir>`
- `fluxion measure --compute_arguments <model> <measure>`
- `fluxion measure --run_tests <dir>`

## Example Usage

### Simple Simulation (EnergyPlus-style)

```bash
# Run annual simulation with weather file
fluxion -w Denver_TMY.epw building.flux

# Run with custom output directory
fluxion -w Denver_TMY.epw -d results/ building.flux

# Design day only
fluxion -w Denver_TMY.epw -D building.flux
```

### Workflow-based (OpenStudio-style)

```bash
# Run complete workflow
fluxion run -w baseline.fwf

# Debug workflow (keep temp files)
fluxion run --debug -w baseline.fwf

# Measures only (don't run simulation)
fluxion run --measures_only -w baseline.fwf

# Post-process only (use existing results)
fluxion run --postprocess_only -w baseline.fwf
```

### Traditional Analysis Commands

```bash
# ASHRAE 140 validation
fluxion validate --case 600

# Sensitivity analysis
fluxion sensitivity --config sensitivity.yaml

# Delta comparison
fluxion delta --config delta.yaml
```

## File Extensions

| Format | Extension | Description |
|--------|-----------|-------------|
| Fluxion Model | `.flux` | Building model in Fluxion JSON format |
| Fluxion Workflow | `.fwf` | Workflow definition (JSON) |
| Weather Data | `.epw` | EnergyPlus Weather format |
| Diagnostics Output | `.csv` | Simulation diagnostics CSV |

## Backward Compatibility

All existing commands remain functional:

```bash
fluxion validate --case 600        # Unchanged
fluxion sensitivity --config x.yaml # Unchanged
fluxion components --case 900FF    # Unchanged
```

## Exit Codes

Follows EnergyPlus convention:
- `0`: Success
- `1`: Fatal error
- `2`: Severe error
- `3`: Warning (simulation completed with warnings)

## Error Output Format

```
** Severe  ** [file:line] Error message
** Warning  ** [file:line] Warning message
** Fatal  ** [file:line] Fatal error - simulation aborted
```

## Documentation Mapping

| Concept | EnergyPlus | OpenStudio | Fluxion |
|---------|------------|------------|---------|
| Input file | IDF | OSM | FUX |
| Weather file | EPW | EPW | EPW |
| Workflow | N/A | OSW | FWF |
| Output | eplusout.* | eplusout.* | fluxion_out.* |
| Error format | ** Severe/Warning/Fatal | Ruby exceptions | Rust Result |

## Migration Path

1. **Users new to BEM**: Start with simple `fluxion -w weather.epw input.flux`
2. **EnergyPlus users**: Familiar pattern with `-w`, `-d`, `-p` options
3. **OpenStudio users**: Adopt workflow files for complex automation
4. **Existing fluxion users**: No changes required to existing scripts

## References

- EnergyPlus CLI: `energyplus --help`
- OpenStudio CLI: `openstudio run --help`
- ASHRAE 140: Standard for BEM validation
