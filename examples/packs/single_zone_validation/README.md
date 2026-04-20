# Single Zone Validation Pack

This pack provides a minimal single-zone building configuration for quick validation testing.

## Files
- `manifest.json` - Pack configuration and metadata
- `config.yaml` - Single-zone building parameters
- `run_validation.sh` - Quick validation script

## Use Case
Use this pack to verify Fluxion installation and basic functionality with a simple single-zone building model.

## Expected Results
- Simulation completes in < 1 minute
- Energy balance within tolerance
- Temperature setpoint tracking verified

## Quick Start
```bash
cd examples/packs/single_zone_validation
./run_validation.sh
```

## Configuration
- Zone count: 1
- Floor area: 100 m²
- Heating setpoint: 20°C
- Cooling setpoint: 24°C
- Internal gains: 500 W (equipment) + lighting

## References
- See `examples/validate_6r2c.rs` for detailed validation code
- See `tests/ashrae_140_case_600.rs` for ASHRAE 140 Case 600 baseline
