# Issue #785: OpenStudio Workflow / Ruby Script Adapter for Fluxion

## Status

**Created**: 2026-06-16  
**Tracking**: Issue #785  
**Labels**: `api`, `feature`, `tools`, `backlog`  
**Stage**: Design Document

---

## Executive Summary

This document outlines a strategy for integrating Fluxion with the OpenStudio Measures ecosystem, enabling reuse of the extensive OpenStudio Measure library (~5,000+ measures) while providing an alternative simulation engine pathway. The approach creates a **Ruby-first CLI adapter** that allows Fluxion to participate in OpenStudio workflows without requiring OpenStudio to be rewritten, and positions Fluxion as a high-performance co-simulation engine alongside EnergyPlus.

---

## 1. Background

### 1.1 OpenStudio Measures Ecosystem

OpenStudio (NREL) provides a scripting platform for building energy modeling via **OpenStudio Measures** — Ruby scripts that:

- Inspect and modify OpenStudio Models (`.osm` files)
- Transform models during translation to EnergyPlus IDF
- Generate custom reports from simulation results
- Are orchestrated via **OpenStudio Workflow** (`.osw`) JSON files

The ecosystem includes:
- **Building Component Library (BCL)**: 5,000+ curated measures
- **OpenStudio CLI**: Command-line workflow executor
- **Ruby API**: Full SDK access from Ruby scripts
- **Python support** (v3.5.0+): Phase 1 Python measure scripting

### 1.2 OpenStudio CLI Workflow

```
.osw (JSON workflow)
├── seed_file: "baseline.osm"
├── weather_file: "USA_CO_Denver.epw"
└── steps[]
    ├── { measure_dir_name: "AddWallInsulation", arguments: { r_value: 45 } }
    ├── { measure_dir_name: "SetHVACSystem", arguments: { system_type: "VAV" } }
    └── { measure_dir_name: "StandardReports", arguments: { output_format: "CSV" } }
```

The CLI executes steps sequentially: Model Measures → EnergyPlus Translation → EnergyPlus Measures → Simulation → Reporting Measures.

### 1.3 Fluxion Position

Fluxion is a Rust-based BEM engine with:
- Python (pyo3) and Node.js (napi-rs) bindings
- CLI for simulation workflows
- FMI 2.0 co-simulation export
- **ASHRAE 140 validation** (v0.8.0)
- BatchOracle for high-throughput parametric analysis

**Gap**: No native OpenStudio integration — Fluxion cannot participate in OpenStudio workflows or leverage OpenStudio Measures.

---

## 2. Integration Options Analysis

### 2.1 Option A: Ruby Shim (Recommended)

**Approach**: Create a Ruby gem (`fluxion-measure`) that wraps Fluxion's CLI as an OpenStudio Reporting Measure.

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenStudio Workflow                       │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │ Measures │ → │   IDF    │ → │EnergyPlus│ → │Reporting│ │
│  └──────────┘   └──────────┘   └──────────┘   │ Measures│ │
│                                                 └────▲────┘ │
│                                                       │      │
│                                              ┌────────┴────┐ │
│                                              │ fluxion     │ │
│                                              │ Measure     │ │
│                                              │ (Ruby gem)  │ │
│                                              └─────┬──────┘ │
│                                                    │        │
│                                              ┌─────▼────┐  │
│                                              │fluxion CLI│  │
│                                              │  (Rust)  │  │
│                                              └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Pros**:
- Native OpenStudio measure — appears alongside EnergyPlus measures
- Leverages OpenStudio workflow orchestration
- Can query OpenStudio model for geometry, constructions, schedules
- Full access to OpenStudio Ruby SDK from within the measure

**Cons**:
- Requires Ruby gem distribution
- Two simulation engines running (E+ for baseline steps, Fluxion for reporting)
- Increased workflow complexity

### 2.2 Option B: Fluxion-Centric OSW Runner

**Approach**: Extend Fluxion's CLI to accept `.osw` files and run OpenStudio Measures as pre/post-processors to Fluxion simulation.

**Pros**:
- Fluxion remains the central engine
- Can leverage OpenStudio Measures for model setup

**Cons**:
- Significant CLI development effort
- OpenStudio CLI dependencies
- Not idiomatic for OpenStudio users

### 2.3 Option C: Python PyO3 Bridge

**Approach**: Use OpenStudio's Python 3.5+ support + Fluxion's Python bindings to create a Python-based adapter.

```python
# OpenStudio measure (Python)
import openstudio
import fluxion

def run(model, runner, user_arguments):
    # Extract geometry from OpenStudio model
    # Call Fluxion Python API
    results = fluxion.simulate(...)
    # Register results as OpenStudio outputs
    runner.registerFinalCondition(...)
```

**Pros**:
- Modern Python-first approach
- Reuses existing PyO3 bindings
- OpenStudio 3.5+ native Python support

**Cons**:
- Python-only (no Ruby ecosystem)
- Less mature than Ruby measures
- Performance overhead of Python↔Rust FFI

### 2.4 Option D: FMU Co-Simulation Export

**Approach**: Extend existing FMI module to export Fluxion as an FMU, then import into OpenStudio via the FMI adapter.

**Pros**:
- Standardized interface (FMI 2.0)
- Reuses existing FMI work

**Cons**:
- OpenStudio FMI support is limited
- Single-zone only in current implementation
- Co-simulation master algorithm complexity

---

## 3. Recommended Implementation: Option A (Ruby Shim)

### 3.1 Architecture

```
fluxion/
├── fluxion-measure/              # Ruby gem
│   ├── lib/
│   │   ├── fluxion_measure.rb    # Main measure class
│   │   └── fluxion/
│   │       ├── runner.rb         # CLI invocation
│   │       └── parser.rb         # Result parsing
│   ├── measure.rb                # OpenStudio::Measure::ReportingMeasure
│   ├── measure.xml               # Measure metadata
│   └── Gemfile                   # Ruby dependencies
├── src/
│   ├── cli/                     # Existing CLI
│   └── bin/
│       └── fluxion_cli.rs        # Entrypoint
└── docs/
    └── OPENSTUDIO_INTEGRATION.md  # User guide
```

### 3.2 Ruby Measure Structure

**File**: `measure.rb`

```ruby
# frozen_string_literal: true

require 'openstudio'
require_relative 'lib/fluxion_measure'

class FluxionSimulation < OpenStudio::Measure::ReportingMeasure
  # Required OpenStudio Measure methods
  def name
    return "Fluxion Simulation"
  end

  def description
    return "Run Fluxion building energy simulation and import results into OpenStudio."
  end

  def modeler_description
    return "This measure invokes the Fluxion CLI to run a simulation. " \
           "Fluxion must be installed and in PATH."
  end

  # Define user-configurable arguments
  def arguments(_model)
    args = OpenStudio::Measure::OSArgumentVector.new

    # Weather file override (optional)
    weather_file = OpenStudio::Measure::OSArgument.makeStringArgument('weather_file', false)
    weather_file.setDisplayName('Weather File (EPW)')
    weather_file.setDescription('Leave blank to use model weather file')
    weather_file.setDefaultValue('')
    args << weather_file

    # Simulation duration
    duration = OpenStudio::Measure::OSArgument.makeIntegerArgument('duration', false)
    duration.setDisplayName('Simulation Duration (days)')
    duration.setDefaultValue(365)
    args << duration

    # Use surrogates
    use_surrogates = OpenStudio::Measure::OSArgument.makeBoolArgument('use_surrogates', false)
    use_surrogates.setDisplayName('Use AI Surrogates')
    use_surrogates.setDefaultValue(false)
    args << use_surrogates

    args
  end

  # Main execution
  def run(model, runner, user_arguments)
    super(model, runner, user_arguments)

    # Validate arguments
    if !runner.validateUserArguments(arguments(model), user_arguments)
      return false
    end

    # Parse arguments
    weather_file = runner.getStringArgumentValue('weather_file', user_arguments)
    duration = runner.getIntegerArgumentValue('duration', user_arguments)
    use_surrogates = runner.getBoolArgumentValue('use_surrogates', user_arguments)

    # Log initial condition
    runner.registerInitialCondition("Starting Fluxion simulation for #{duration} days")

    # Convert OpenStudio model to Fluxion schema
    fluxion_input = Fluxion::ModelConverter.from_openstudio(model, runner)

    # Invoke Fluxion CLI
    result = Fluxion::Runner.run(
      input: fluxion_input,
      weather: weather_file,
      days: duration,
      surrogates: use_surrogates
    )

    if result.success?
      # Register outputs with OpenStudio
      runner.registerInfo("Fluxion simulation completed in #{result.elapsed_ms}ms")
      runner.registerFinalCondition(
        "EUI: #{result.eui.round(2)} kWh/m², " \
        "Peak Heating: #{result.peak_heating.round(0)} W, " \
        "Peak Cooling: #{result.peak_cooling.round(0)} W"
      )
      return true
    else
      runner.registerError("Fluxion simulation failed: #{result.errors.join(', ')}")
      return false
    end
  end
end

# Register measure with OpenStudio
FluxionSimulation.new.registerWithApplication
```

### 3.3 Fluxion Schema Mapping

Key mappings from OpenStudio Model to Fluxion `SimulationSchema`:

| OpenStudio Object | Fluxion Schema Field |
|------------------|---------------------|
| `OS:Building` | `metadata.name` |
| `OS:ThermalZone` | `geometry.zones[].name` |
| `OS:Space` | `geometry.zones[].floor_area` |
| `OS:Construction` | `constructions.wall.layers[]` |
| `OS:Window` | `constructions.wall.window` |
| `OS:Schedule` | `schedules.occupancy`, `schedules.lighting` |
| `OS:WeatherFile` | `weather.epw.path` |

### 3.4 CLI Extension

Extend Fluxion CLI to support OpenStudio model input:

```bash
# Existing CLI
fluxion multi-zone simulate --schema input.json --weather epw/denver.epw

# New OpenStudio adapter commands
fluxion openstudio run-from-model model.osm --weather epw/denver.epw
fluxion openstudio run-from-osw workflow.osw
fluxion openstudio export-schema model.osm > fluxion_input.json
```

### 3.5 Ruby Gem Specification

**File**: `fluxion-measure/Gemfile`

```ruby
source 'https://rubygems.org'

gemspec name: 'fluxion-measure',
       version: '1.0.0',
       authors: ['Fluxion Team'],
       email: 'fluxion@example.org'
```

**File**: `fluxion-measure/fluxion-measure.gemspec`

```ruby
Gem::Specification.new do |s|
  s.name = 'fluxion-measure'
  s.version = '1.0.0'
  s.summary = 'OpenStudio Measure for Fluxion simulation'
  s.description = 'Integrates Fluxion BEM engine with OpenStudio workflows'
  s.authors = ['Fluxion Team']
  s.email = 'fluxion@example.org'
  s.files = Dir['{lib,measure.rb,measure.xml}']
  s.metadata = {
    'openstudio_version' => '>= 3.0.0',
    'intended_use_case' => 'Model Articulation'
  }
end
```

---

## 4. Implementation Phases

### Phase 1: Foundation (2-3 weeks)

**Goal**: Minimal viable OpenStudio Reporting Measure

1. Create `fluxion-measure/` Ruby gem structure
2. Implement `ModelConverter` — OpenStudio model → Fluxion `SimulationSchema`
3. Implement `Runner` — CLI invocation and result parsing
4. Create `measure.rb` with basic reporting
5. Add CLI subcommand: `fluxion openstudio run-from-model`

**Deliverables**:
- Ruby gem in `fluxion-measure/`
- CLI extension in `src/cli/commands/openstudio.rs`
- Working measure that reads `.osm` files and runs Fluxion

### Phase 2: Workflow Integration (2-3 weeks)

**Goal**: Full OSW workflow support

1. Implement `fluxion openstudio run-from-osw` CLI command
2. Support pre-processing model measures (translate OSM → Fluxion schema)
3. Support post-processing reporting measures
4. Add OpenStudio model → Fluxion schema conversion tests

**Deliverables**:
- Complete OSW runner
- Integration tests with OpenStudio CLI
- Example OSW files

### Phase 3: Measure Authoring (2 weeks)

**Goal**: Enable writing OpenStudio Measures in Ruby that call Fluxion

1. Provide helper classes for Fluxion API access from Ruby
2. Support Fluxion result objects as OpenStudio model attributes
3. Add continuous integration for Ruby gem

**Deliverables**:
- `Fluxion::ModelConverter`, `Fluxion::Runner`, `Fluxion::Result` Ruby classes
- CI/CD for gem release

### Phase 4: Ecosystem Expansion (3-4 weeks)

**Goal**: Deep OpenStudio ecosystem integration

1. Add Fluxion output as OpenStudio OutputAttribute
2. Support EnergyPlus measure compatibility (Fluxion as E+ alternative)
3. Publish gem to RubyGems.org and BCL
4. Create documentation and tutorials

**Deliverables**:
- Published Ruby gem
- BCL contribution
- User documentation

---

## 5. File Structure

```
fluxion/
├── src/
│   ├── cli/
│   │   └── commands/
│   │       └── openstudio.rs       # NEW: OpenStudio adapter CLI
│   └── api/
│       └── schema.rs               # EXISTING: SimulationSchema
├── fluxion-measure/                 # NEW: Ruby gem
│   ├── Gemfile
│   ├── fluxion-measure.gemspec
│   ├── measure.rb                   # OpenStudio ReportingMeasure
│   ├── measure.xml                   # Measure metadata
│   └── lib/
│       ├── fluxion_measure.rb        # Gem entry point
│       └── fluxion/
│           ├── runner.rb             # CLI invocation
│           ├── converter.rb          # OSM → Fluxion schema
│           └── parser.rb             # Result parsing
└── docs/
    └── OPENSTUDIO_INTEGRATION.md    # User guide
```

---

## 6. Dependencies

| Component | Dependency | Version |
|-----------|-----------|---------|
| OpenStudio CLI | NREL OpenStudio SDK | >= 3.0.0 |
| Ruby | MRI Ruby | >= 2.7 |
| Fluxion CLI | fluxion binary | >= 1.0.0 |
| Ruby gem | openstudio | >= 3.0.0 |

---

## 7. Open Questions

1. **How to handle HVAC?** OpenStudio HVAC systems are complex — should Fluxion's simplified HVAC map to OpenStudio systems, or should we require users to use OpenStudio HVAC and Fluxion only for envelope/thermal response?

2. **Multi-zone support?** Current Fluxion multi-zone implementation vs OpenStudio thermal zones — need to verify zone mapping completeness.

3. **Result compatibility?** EnergyPlus outputs (SQLite, CSV) vs Fluxion outputs — should Fluxion output mimic E+ format for compatibility?

4. **Version skew?** OpenStudio releases ~yearly; how to maintain compatibility across versions?

5. **Testing strategy?** OpenStudio CI infrastructure for measures — should Fluxion measure use same test framework (minitest)?

---

## 8. References

- [OpenStudio CLI Documentation](http://natlabrockies.github.io/OpenStudio-user-documentation/reference/command_line_interface/)
- [OpenStudio Measure Writer's Reference Guide](http://natlabrockies.github.io/OpenStudio-user-documentation/reference/measure_writing_guide/)
- [Fluxion SimulationSchema](../src/api/schema.rs)
- [FMI Module](../docs/FMI.md)
- [OpenStudio SDK](https://openstudio.net/)
- [BCL (Building Component Library)](https://bcl.nrel.gov/)

---

## 9. Success Criteria

1. **Functional**: An OpenStudio user can add the Fluxion Measure to an OSW workflow and run Fluxion simulation
2. **Compatible**: Measure works with OpenStudio CLI on Linux/macOS
3. **Tested**: Unit tests cover OSM→FluxionSchema conversion for common building types
4. **Documented**: User guide explains installation, configuration, and troubleshooting
5. **Maintained**: Gem is released on RubyGems.org with semantic versioning
