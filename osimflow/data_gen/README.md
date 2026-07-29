# osimflow.data_gen

Monte Carlo delta-file generator for OSimFlow (Fluxion Issue #1813, Phase 1 —
Declarative Deltas).

Generates the lightweight delta files that the Fluxion Rust worker entrypoint
consumes:

```
fluxion monte-carlo sweep --base-model base.yaml --delta-file delta.yaml --output ./out
```

## Quick start

```bash
# Emit a declarative delta file with the default 1000 samples.
python -m osimflow.data_gen.generate_monte_carlo_deltas -o delta.yaml

# Also pre-materialize one JSON patch per draw for distributed workers.
python -m osimflow.data_gen.generate_monte_carlo_deltas -n 100 \
    -o delta.yaml --materialize ./patches
```

## Parameters

The default parameter set covers the variables named in Issue #1813:

| Path                       | Distribution | Parameters            |
|----------------------------|--------------|-----------------------|
| `infiltration_ach`         | uniform      | min 0.3, max 1.5      |
| `window_properties.u_value`| normal       | mean 3.0, std 0.3     |
| `window_properties.shgc`   | triangular   | min 0.4, mode 0.7, max 0.9 |
| `opaque_absorptance`       | uniform      | min 0.6, max 0.9      |

## Tests

```
pytest osimflow/data_gen/test_generate_deltas.py
```
