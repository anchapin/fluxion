# Legacy examples

These JSON files pre-date the PyO3 / axum surface and are **not**
consumed by `fluxion.Model`, `fluxion.BatchOracle`, or `fluxion-rest`.
They are kept here for historical reference only.

- `simple_config.json` — old 10-zone `Model` config stub. The current
  `Model` constructor takes `num_zones: usize`, not a config path.
- `simulation_schema_v1.json` — old `SimulationSchemaV1` example with
  non-canonical schedule keys. The canonical REST fixture is
  [`../../tests/fixtures/single_zone.json`](../../tests/fixtures/single_zone.json),
  which is round-tripped by `tests/examples_smoke.rs` on every CI run.

See [`../README.md`](../README.md) for the live, supported examples and
[`../../docs/EXAMPLES.md`](../../docs/EXAMPLES.md) for input/output
semantics. (Moved in #2544.)
