# Fluxion FAQ

Frequently asked questions and consolidated user-facing caveats.
Single landing page for the nine caveats that used to be scattered across
EXAMPLES.md, QUICKSTART.md, KNOWN_ISSUES.md, README, and AGENTS.md.
Each entry links to the authoritative source for the latest status.
For developer-facing build/CI/physics-constant pitfalls, see
[docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md) instead.

*Last Updated: 2026-08-10*

## Q1. Why are `peak_heating_load` and `peak_cooling_load` always `0.0`?

In the current release, the per-hour peak-load tracking is **not wired
into the REST handler or the Python `Model.simulate()` path** — both
fields are emitted as `0.0` so the JSON shape stays stable across
releases.

```json
{
  "output": {
    "peak_heating_load": 0.0,
    "peak_cooling_load": 0.0,
    "heating_energy": 350.0,
    "cooling_energy": 242.32
  }
}
```

`heating_energy` and `cooling_energy` (both in kWh) **are** populated
and are the correct way to compare candidates today. Peak loads in
**W** are tracked internally by the 9R4C batch runner
(`src/ai/batch_runner_9r4c.rs`, exposed as `peak_heating_load_w` /
`peak_cooling_load_w` on its output struct) but are not yet surfaced
through `SimulationOutput`.

**Source:** [`docs/EXAMPLES.md` §3](EXAMPLES.md),
[`docs/SCHEMA.md`](SCHEMA.md) (`SimulationOutput` definition at line 144).

## Q2. Why are the `eui` values so large?

`ThermalModel::solve_timesteps` accumulates an **uncalibrated
temperature-departure metric**, not a physical kWh/m²/year EUI. At each
timestep it adds `|T_zone - T_setpoint|` across every zone; with
`num_zones = 10` and `timesteps = 8760` you get 87 600 contributions,
hence the large numbers.

The metric is intentionally uncalibrated — it exists for **algorithm
correctness and performance testing** of the optimisation harness, not
for absolute energy reporting. To normalise:

```python
normalized = raw_eui / (num_zones * 8760)   # avg hourly °C-gap per zone
```

To convert to physical kWh/m²/year you need additional data (zone heat
capacity in J/K, floor area in m², timestep duration in hours); see
[`docs/EXAMPLES.md` §4–§7](EXAMPLES.md) for the full pipeline.

**Source:** [`docs/EXAMPLES.md` §4](EXAMPLES.md).

## Q3. Why doesn't `fluxion run` accept `simple_config.json`?

The `fluxion` binary understands **OpenStudio-style `.fwf` workflow
files**, not arbitrary JSON. `examples/legacy/simple_config.json` and
`examples/legacy/simulation_schema_v1.json` are **stale historical
stubs** that pre-date the PyO3 / axum surface and are **not consumed by
`Model`, `BatchOracle`, the CLI, or `fluxion-rest`**. They were moved
under `examples/legacy/` in PR #2544 and are kept for historical
reference only.

The canonical REST request body — and the only JSON document the
`POST /v1/simulate` endpoint validates against — is
[`tests/fixtures/single_zone.json`](../tests/fixtures/single_zone.json),
which matches `fluxion::api::schema::SimulationSchemaV1` byte-for-byte
and is round-tripped by `tests/examples_smoke.rs` on every CI run.

**Source:** [`docs/QUICKSTART.md` §6](QUICKSTART.md),
[`docs/EXAMPLES.md` §2.3](EXAMPLES.md),
[`examples/legacy/README.md`](../examples/legacy/README.md).

## Q4. Why does Case 900 deviate from the ASHRAE 140 reference by ~47 %?

This is a **known structural limitation of the 5R1C thermal network**,
not a bug. After the #2227 (`derived_h_tr_3`) and #2229
(`h_ms_coeff` 9.1 → 13.4 W/(m²·K)) fixes, Case 900 still produces
heating **2.362 MWh** (ref [1.17, 2.04] MWh, +47 % above midpoint) and
cooling **1.330 MWh** (ref [2.13, 3.67] MWh, −54 % below midpoint).

The pattern — heating too high **and** cooling too low — is the
textbook signature of a single lumped thermal-mass node on a 1-hour
timestep: it cannot simultaneously release stored solar heat fast
enough during shoulder seasons **and** absorb enough daytime solar to
charge the mass for night-time cooling. No `h_ms_coeff`,
`f_furniture`, or `derived_h_tr_3` adjustment can move both metrics
into band simultaneously (proven by the #1522 air-node investigation).

Closing the gap by tuning constants would violate `RULES.md`
("must-never hardcode results"). The correct fix is the **GaugeSolver**
(#1465 / #1462 / #2304), which treats solar as geometric curvature
rather than per-timestep energy injection, or sub-hour air-node
sub-stepping. **Tracked by #1465.**

**Source:** [`docs/KNOWN_ISSUES.md` §SOLAR-02 UPDATE (Issue #2239)](KNOWN_ISSUES.md),
also §LIMIT-05 UPDATE (Issue #2453) for the 900-series bidirectional
over-prediction.

## Q5. How do I load a real ONNX surrogate? Why does it fall back to a mock?

The ONNX runtime is **opt-in** — default builds skip it. You must:

1. Build with `--features ort` (alias `onnx`).
2. Either set `FLUXION_ONNX_MODEL=/path/to/model.onnx`, place a model
   at `models/surrogate_zone_thermal.onnx`, or call
   `model.load_surrogate("path/to/model.onnx")` from Python.
3. Optionally select a backend with `FLUXION_ONNX_BACKEND`
   (`cpu` | `cuda` | `coreml` | `directml` | `openvino`). `cuda` is a
   no-op when the `cuda` cargo feature is not built; the manager falls
   back to CPU at runtime. Set `FLUXION_GPU=0` to force CPU inference.

If no model resolves, `SurrogateManager` is constructed in **mock
mode**: `model_loaded: false`, and inference falls back to
`deterministic_analytical_loads` (a pure function of the inputs — see
`src/ai/surrogate.rs:1853`). The mock is deterministic and intended
for **API validation and performance testing**, not production
accuracy. `SurrogateManager::new_with_auto_load()` is the constructor
that performs the resolution order above.

**Source:** [`docs/EXAMPLES.md` §2.1](EXAMPLES.md),
[`docs/FEATURES.md`](FEATURES.md), [`AGENTS.md` §Environment
Variables](../AGENTS.md), [`README.md`](../README.md).

## Q6. Why doesn't `BatchOracle` expose `load_surrogate`?

`BatchOracle` does **not** expose `load_surrogate` — only `Model` does.
The oracle always uses its **internal** `SurrogateManager`, constructed
at `BatchOracle::new()`. To use a real ONNX model with `BatchOracle`,
set `FLUXION_ONNX_MODEL` **before** constructing the oracle (so the
auto-load path picks it up), or rebuild with `--features ort` and place
the model at the default path. When no ONNX model is loaded, the
oracle's internal `SurrogateManager` falls back to the deterministic
analytical loads described in Q5.

```python
from fluxion import BatchOracle

oracle = BatchOracle()                              # internal SurrogateManager
results = oracle.evaluate_population(pop, use_surrogates=False)
```

**Source:** [`docs/EXAMPLES.md` §2.2](EXAMPLES.md),
[`src/lib.rs`](../src/lib.rs) `BatchOracle` impl.

## Q7. Are deterministic mock results reproducible across runs?

**Yes — given the same inputs.** `SurrogateManager::deterministic_analytical_loads`
is a pure function of `&[SurrogateInputs]` (verified by
`test_deterministic_analytical_loads_is_pure` in `src/ai/surrogate.rs`).
There is no hidden RNG state; the "seed" referred to elsewhere in the
docs is the parametric sweep seed that drives the **inputs**, not the
inference itself.

Two scenarios where reproducibility **can** break:

1. **Cross-platform FP drift** — IEEE 754 transcendental ordering is
   not guaranteed across OS/libm versions. The determinism gate
   (Q8 / [#1351](https://github.com/anchapin/fluxion/issues/1351))
   hashes Case 900 output across ubuntu/windows/macos and fails the PR
   if it diverges. Run locally with:
   ```bash
   RUSTFLAGS="-C opt-level=3 -C debug-assertions=no" \
     cargo test --test case_900_determinism --release -- --nocapture
   ```
2. **RNG-seeded inputs** — if you generate stochastic candidate
   populations in Python (NumPy RNG) you must re-seed the RNG to
   reproduce the *inputs*; the inference itself is already pure.

**Source:** [`docs/EXAMPLES.md` §8](EXAMPLES.md),
[`src/ai/surrogate.rs`](../src/ai/surrogate.rs)
(`deterministic_analytical_loads`, test at line 2887).

## Q8. Why did my PR fail the "Fluxion Determinism Gate (Issue #1351)"?

The cross-platform determinism gate is a **required** branch-protection
check. It runs `tests/case_900_determinism.rs` on ubuntu/windows/macos
and compares SHA-256 hashes of extracted values — byte-identical output
is required across all three. Common causes of failure (issue #1297
fix list):

- A new `HashMap` / `HashSet` was used where a deterministic `BTreeMap`
  is required (non-deterministic iteration order across platforms).
- A non-deterministic `f32` reduction path (SIMD reordering, parallel
  reduction with non-associative orderings) was added without an
  explicit `BTreeMap`/sorted-iterator wrapper.
- A new dependency pulls in non-portable FP code.

The listener workflow
(`.github/workflows/determinism_check.yml` +
`workflow_run` listener in `ashrae_validation.yml`) is the canonical
non-matrix required check that branch protection references. The full
list of required checks lives in
[`release_gates.yaml`](../release_gates.yaml) → `ci.required_checks`.

**Source:** [`docs/CONTRIBUTING.md` §Cross-Platform Determinism CI
Gate](CONTRIBUTING.md), [`AGENTS.md` §CI Gates](../AGENTS.md).

## Q9. Why are my `peak_cooling_load` results for the 9xx series off?

Cases 910/920/930/940/950/960 (high-mass shading / setback / night-
ventilation set) still show peak-cooling under-prediction tracked
under §LIMIT-05 of KNOWN_ISSUES — the same root cause as Q4
(roof-solar under-counting; see
[`docs/investigations/issue-1280-ctf-peak-load.md`](investigations/issue-1280-ctf-peak-load.md)
§4). Cases 600/650 and 900 peak cooling **are** within the post-#1270
±15 % reference envelope. Case 940 setback thermostat overshoots blind
by 6–8× in the CTF path (diagnostic test
`tests/diagnostics/case_940_setback_diagnostic.rs`, issue #2452); the structural
fix is routed to GaugeSolver #1465/#1462.

Do **not** cite the legacy "peak cooling 40–80 % under-predicted"
figure from §SOLAR-01 — it predates the #1323 baseline and is obsolete.
Refer to §LIMIT-05 and the per-case numbers in
[`docs/ASHRAE140_RESULTS.md`](ASHRAE140_RESULTS.md) instead.

**Source:** [`docs/KNOWN_ISSUES.md` §SOLAR-01, §LIMIT-05 UPDATE
(Issue #2453)](KNOWN_ISSUES.md), [`docs/ASHRAE140_RESULTS.md`](ASHRAE140_RESULTS.md).

## Q10. Where is the full list of known validation limitations?

[`docs/KNOWN_ISSUES.md`](KNOWN_ISSUES.md) is authoritative. It tracks
the BASE-0x foundation issues, SOLAR-0x solar issues, and LIMIT-0x
limit-cycle issues with severity, affected cases, GitHub-issue links,
and per-row `*Last Updated*` markers. A CI gate
(`scripts/check_known_issues_stale.py`, #1723) fails if the file's
top-level `*Last Updated*` line is more than 60 days old. See also
[`docs/ASHRAE140_RESULTS.md`](ASHRAE140_RESULTS.md) for the latest
per-case engine output and
[`docs/ASHRAE_REVALIDATION_SCHEDULE.md`](ASHRAE_REVALIDATION_SCHEDULE.md)
for the revalidation cadence.

## Q11. Where do I go next?

- [docs/QUICKSTART.md](QUICKSTART.md) — installation, first run
- [docs/EXAMPLES.md](EXAMPLES.md) — `Model`, `BatchOracle`, REST usage
- [docs/API_REFERENCE.md](API_REFERENCE.md) — full REST + Python API
- [docs/REST_API.md](REST_API.md) — every endpoint with curl examples
- [docs/FEATURES.md](FEATURES.md) — every `--features` flag
- [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md) — developer-facing
  build / CI / physics-constant pitfalls
- [ARCHITECTURE.md](../ARCHITECTURE.md) — module boundaries & data flow

## Getting Help

- GitHub Issues: <https://github.com/anchapin/fluxion/issues>
- Documentation: <https://fluxion.readthedocs.io>
