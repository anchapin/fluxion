# Retrofit Pack

This pack provides building upgrade scenarios for evaluating energy savings from retrofit measures.

## Files
- `manifest.json` - Pack configuration and metadata
- `config.yaml` - Baseline and retrofit configurations
- `compare_scenarios.sh` - Script to compare baseline vs retrofit

## Use Case
Use this pack to evaluate energy savings from various retrofit measures:
- Wall insulation upgrades
- Window replacements
- HVAC efficiency improvements
- Lighting upgrades

## Expected Results
- Energy savings: 15-40% depending on measures
- Simple payback period: 5-20 years
- Carbon reduction assessment

## Quick Start
```bash
cd examples/packs/retrofit
./compare_scenarios.sh
```

## Configuration
Baseline (1980s construction) vs Retrofit (current code compliance):
- Wall U-value: 0.5 → 0.3 W/m²K
- Window U-value: 2.5 → 1.8 W/m²K
- HVAC efficiency: 0.80 → 0.95
- Lighting: 15 → 8 W/m²

## References
- See `data/assemblies.yaml` for construction definitions
- See `examples/construction_example.rs` for U-value calculations
