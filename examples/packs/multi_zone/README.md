# Multi Zone Pack

This pack provides a multi-zone building configuration demonstrating zone-to-zone heat transfer.

## Files
- `manifest.json` - Pack configuration and metadata
- `config.yaml` - Multi-zone building parameters
- `run_simulation.sh` - Simulation script

## Use Case
Use this pack to verify multi-zone thermal modeling and inter-zone coupling effects.

## Expected Results
- Two zones with independent setpoints
- Heat flow between zones based on temperature differential
- Energy conservation validated across zones

## Quick Start
```bash
cd examples/packs/multi_zone
./run_simulation.sh
```

## Configuration
- Zone 1 (Living): 20°C heating, 24°C cooling
- Zone 2 (Sunspace): 15°C heating only
- Inter-zone conductance: 50 W/K

## References
- See `examples/multi_zone_demo.rs` for detailed implementation
- See `tests/ashrae_140_case_960.rs` for ASHRAE 140 Case 960 (sunspace)
