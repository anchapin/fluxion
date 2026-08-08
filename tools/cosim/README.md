# Co-Simulation Master for BES+FFD (Issue #2391)

This directory contains the infrastructure for running coupled Building Energy Simulation (BES) + Fast Fluid Dynamics (FFD) co-simulation using MasterSim as the FMI 2.0 co-simulation master.

## Overview

The co-simulation setup implements loose coupling (Issue #2390) between:
- **BES FMU**: Thermal envelope model (zone temperature, heating/cooling loads)
- **FFD FMU**: Airflow solver (zone air temperature stratification, CHTCs)

## Files

| File | Purpose |
|------|---------|
| `run_cosimulation.py` | Main test harness: generates dummy FMUs, creates MasterSim config, runs simulation, validates results |
| `mastersim_config.py` | MasterSim `.ums` configuration generator (standalone utility) |
| `README.md` | This file |

## Quick Start

### Prerequisites

Install MasterSim (FMI 2.0.3 co-simulation master):

```bash
git clone https://github.com/ghorwin/MasterSim.git
cd MasterSim
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

Verify installation:
```bash
mastersim --version
```

### Generate FMUs and Configuration Only

```bash
python tools/cosim/run_cosimulation.py --generate-only
```

This creates:
- `bes_dummy.fmu` — BES FMU archive
- `ffd_dummy.fmu` — FFD FMU archive
- `master_config_BES_FFD.ums` — MasterSim configuration

### Run Full 24-Hour Co-Simulation

```bash
python tools/cosim/run_cosimulation.py --run
```

Expected output:
- `cosim_results.csv` — Time series of all FMU variables
- Validation confirms:
  - Simulation completed to t=86400s (24 hours)
  - No synchronization deadlocks
  - Master clock advanced correctly

### Validate Existing Results

```bash
python tools/cosim/run_cosimulation.py --validate-only
```

## FMU Interface Summary

### BES FMU (FluxionBES)

**Inputs** (BES ← MasterSim):
- `outdoor_temperature` (K): Outdoor dry bulb temperature
- `direct_normal_solar` (W/m²): Direct normal solar radiation
- `diffuse_horizontal_solar` (W/m²): Diffuse horizontal solar
- `internal_gains` (W): Internal heat gains

**Outputs** (BES → MasterSim → FFD):
- `zone_temperature` (K): Zone air temperature
- `heating_load` (W): Heating load (≥ 0)
- `cooling_load` (W): Cooling load (≥ 0)

### FFD FMU (FluxionFFD)

**Inputs** (FFD ← MasterSim ← BES):
- `inlet_air_temperature` (K): Supply air temperature
- `mass_flow_rate_supply` (kg/s): HVAC supply air mass flow rate
- `mass_flow_rate_exhaust` (kg/s): HVAC exhaust air mass flow rate
- `wall_temperature_N` (K): Zone wall surface temperatures (N = 0..5)

**Outputs** (FFD → MasterSim → BES):
- `zone_air_temperature_N` (K): Stratified air temperature at heights (N = 0..3)
- `chtc_N` (W/m²K): Convective heat transfer coefficients (N = 0..5)
- `surface_heat_flux_N` (W/m²): Surface heat fluxes (N = 0..5)

## Coupling Data Flow

```
BES FMU                          FFD FMU
┌────────────────────────────┐   ┌─────────────────────────────┐
│ outputs:                   │   │ inputs:                      │
│   zone_temperature ────────►│──►│ wall_temperature_0           │
│                            │   │ inlet_air_temperature ◄──────│
│ inputs:                    │   │                             │
│   zone_air_temperature_0 ◄─│───│ (from FFD)                │
└────────────────────────────┘   └─────────────────────────────┘
      ▲                                   │
      │           MasterSim               │
      └───────────────────────────────────┘
```

At each BES timestep (3600s):
1. MasterSim reads `zone_temperature` from BES
2. MasterSim writes to FFD inputs (`wall_temperature_0`, `inlet_air_temperature`)
3. FFD computes airflow and heat transfer at 60s micro-steps
4. MasterSim reads FFD outputs (`zone_air_temperature_0`)
5. MasterSim advances to next BES timestep

## Time-Stepping

| FMU | Communication Step | Micro-Steps |
|-----|-------------------|-------------|
| BES | 3600 s (1 hour) | 1 per communication step |
| FFD | 60 s (1 minute) | 60 per BES step |

MasterSim uses 60s as the master step size and handles FFD's finer timestep internally.

## Validation Criteria (Issue #2391)

1. ✅ **FMU Loading**: MasterSim can load BES and FFD FMUs simultaneously
2. ✅ **24-Hour Run**: Simulation completes to t=86400s without hanging
3. ✅ **No Deadlocks**: All 1440 FFD micro-steps (24h × 60min) complete
4. ✅ **Clock Sync**: Master clock advances correctly with FFD micro-stepping

## Troubleshooting

### MasterSim not found

```bash
# Set path explicitly
MASTERSIM_BIN=/path/to/mastersim python tools/cosim/run_cosimulation.py --run

# Or add to PATH
export PATH="$HOME/MasterSim/build:$PATH"
```

### FMU validation fails

The dummy FMUs are structural placeholders. They declare the correct FMI 2.0 interface but don't implement the actual physics. MasterSim can load them but they won't produce physically meaningful results.

For full BES+FFD coupling with real physics, use the actual Fluxion-exported FMUs (after issue #2388 implementation).

### Permission denied

```bash
chmod +x tools/cosim/run_cosimulation.py
```

## References

- MasterSim: https://github.com/ghorwin/MasterSim
- FMI 2.0 Standard: https://fmi-standard.org/
- Loose Coupling Strategy: Issue #2390
- FFD FMU Export: Issue #2388
