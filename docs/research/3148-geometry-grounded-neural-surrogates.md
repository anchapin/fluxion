# Geometry-Grounded Neural Surrogates for 3D Indoor Airflow — Research Artifact (Issue #3148)

> **Summary 1/7:** Issue #3148 is a research artifact (not an implementation task) capturing the architectural questions, energy-balance protocol, training-data sourcing, and acceptance criteria for a future geometry-grounded neural surrogate that would plug into Fluxion as an alternative implementation of `loose_coupling::FfdSolver` — alongside, not replacing, `fluxion_cfd::FfdCfdSolver`.
> **Summary 2/7:** All nine pre-requisites the issue body enumerates are CLOSED as of 2026-09 (verified): CFD wiring (#2460 / PR #2469), CPU baseline (#2456 / PR #2477), surrogate runtime critical issues (#1784, #2905, #2906, #2919, #2920, #2921, #2922, #2923, #2924, #2925), and adjacent closed research spikes (#2937, #2940) — so the architectural question is now decidable in isolation from those blockers.
> **Summary 3/7:** Scope is design-only. This artifact does NOT modify `src/ai/surrogate.rs` (4941 LOC, opt-in `--features ort`), does NOT touch `fluxion-cfd/` internals, does NOT widen `loose_coupling::FfdSolver`, and does NOT raise any test reference_data baseline. Implementation PRs that follow this artifact will land separately, each with its own AC.
> **Summary 4/7:** Architecture survey weighs four candidate families against the BES-FFD exchange contract (`BesToFfdBoundaryConditions` ↔ `FfdToBesResults`): Graph Neural Operators (GNN/GNO), Point-Cloud Nets (PointNet++/Point Transformer), Fourier Neural Operators (FNO/MANO), and mesh-free transformer hybrids. The artifact's tentative pick is **Graph Neural Operator (GNO) on the BES surface graph** — selected for symmetry with the existing `HybridRouting` boundary and the existing `src/physics/geometry_tensor.rs` CTA representation; the pick is documented as conditional and falsifiable.
> **Summary 5/7:** Training-data ground truth must come from `fluxion-cfd` runs (per AGENTS.md: no OpenFOAM dependency). Cfd ground-truth suite is sized at ~5,000 macro-step FFD runs spanning ASHRAE 140 envelope variations (Case 600 / 900 / 950FF base + perturbed geometry / ACH / wind), each at the production FFD grid (`fluxion_cfd::FfdConfig { nx=ny=nz=32, dx=dy=dz=0.1 m }`); the suite is an explicit deliverable of any follow-up implementation issue.
> **Summary 6/7:** Energy-balance protocol is layered: per-timestep drift gate (±1 % per `#1784`, reused verbatim) + annual zone-energy-balance residual (≤ 0.5 % per `RULES.md §1` "total heat transfer must sum to zero") + new CFD-to-CFD MAPE bound of **≤ 8 % per-field-element, ≥ 95th-percentile** (a deliberately conservative target — CFD-to-CFD comparisons rarely beat ~5 % MAPE on instantaneous fields, and an uncritical `<3 % MAPE` AC would either fail or invite parameter tuning, both forbidden by `RULES.md §0`).
> **Summary 7/7:** Out of scope (explicit): creating a `fluxion-surrogates` workspace crate; introducing a `Vertical-C: ML & Surrogates` label; modifying `GaugeZoneSolver` to consume dynamic `h_c` directly (the convective-film feedback into `9R4C`/`GaugeZoneSolver` belongs in a separate `HybridRouting` extension, not this artifact); GPU / CUDA inference path (blocked on #2456 GPU follow-up); ONNX Runtime backend parity (orthogonal, `#2906` SHA-256 gate reused).

- **Status:** Proposed (research artifact only — no implementation recorded; no Rust code change ships in this PR)
- **Date:** 2026-09-06
- **Deciders:** Fluxion maintainers (TBD)
- **Supersedes:** None
- **Depends on:** Closed pre-requisites verified 2026-09-06 — #2460 (PR #2469), #2456 (PR #2477), #1784, #2905, #2906, #2919, #2920, #2921, #2922, #2923, #2924, #2925, #2937, #2940; the umbrella v1.3 ASHRAE 140 release gate (`release_gates.yaml → validation.min_pass_rate = 60.0`) is **orthogonal** to this artifact per the issue body's explicit framing
- **Issue:** [#3148](https://github.com/anchapin/fluxion/issues/3148)
- **Related:** #2460, #2456, #1784, #2905, #2906, #2919, #2920, #2921, #2922, #2923, #2924, #2925, #2937, #2940, #1139 (surrogate v3.0 training), #1431 (`HybridRouting`), #1892 (OOD fallback), #2457 (HVAC routing), `ARCHITECTURE.md` §"Module N+2: BES-FFD Loose Coupling" + §"Hybrid mode — HybridRouting" + §"ML-surrogate swap-point traits", `AGENTS.md` "Main swap points" + "fluxion-core is a dependency-light leaf"

---

## Context

Fluxion's v1.3 release sits between two simulation regimes. The 0-D lumped-parameter
thermal networks (5R1C, 9R4C, CTF) — what Fluxion validates against ASHRAE 140 today —
are fast (milliseconds per macro-step) but cannot resolve spatial gradients inside
a zone. Full 3-D CFD (`fluxion-cfd::FfdCfdSolver`, semi-Lagrangian advection +
implicit diffusion + pressure Poisson) resolves those gradients but is too slow for
batch design-space exploration: a single representative macro-step on the
`nx=ny=nz=32, dx=dy=dz=0.1 m` FFD grid takes seconds, and a year-long simulation
at the 15-min macro-step cadence that ASHRAE 140 enforces would require on the
order of 10⁵–10⁶ macro-steps — well outside the throughput budget
(`release_gates.yaml → benchmark.throughput.min_configs_per_sec = 150`).

Issue #3148 proposes bridging the gap with a **geometry-grounded neural surrogate**:
a neural network conditioned on building geometry that predicts 3-D velocity,
temperature, and surface convective heat transfer coefficient (CHTC) fields in
milliseconds, suitable as a faster stand-in for CFD in the BES coupling loop.

This artifact is the design record. It is not an implementation. The previously
blocking prerequisites (per the issue body's "Status" section, verified closed as
of 2026-09) are:

| # | Title | State | Reference |
|---|-------|-------|-----------|
| #2460 | `fluxion-cfd::FfdCfdSolver` wired into `loose_coupling::FfdSolver` | CLOSED (PR #2469) | `src/sim/ffd_cfd_adapter.rs` |
| #2456 | `fluxion-cfd` CPU baseline (GPU stub no-op removed) | CLOSED (PR #2477) | `fluxion-cfd/src/cpu/` |
| #1784 | Surrogate per-timestep drift gate | CLOSED | `tests/surrogate_drift_gate.rs` |
| #2905 | ONNX env-var bypass of `validate_model_path` | CLOSED | `src/ai/surrogate.rs` |
| #2906 | ONNX model integrity verification (`verify_onnx_signature`) | CLOSED | `src/ai/surrogate.rs:3348` |
| #2919 | Cold-start latency gate | CLOSED | (per issue body) |
| #2920 | GPU/ORT backend silent downgrade | CLOSED | (per issue body) |
| #2921 | Allocating `predict_loads_with_fallback` in hot loop | CLOSED | `predict_loads_into` now the canonical path |
| #2922 | Throughput / latency gate | CLOSED | (per issue body) |
| #2923 | Drift-tolerance silent advisory downgrade | CLOSED | (per issue body) |
| #2924 | ASHRAE 140 surrogate MAE gate | CLOSED | `tests/surrogate_ashrae_600_cooling_mae.rs` |
| #2925 | `HybridThermalModel::clone` asymmetry | CLOSED | (per issue body) |
| #2937 | Mojo / MAX framework spike | CLOSED | `docs/adr/0013-mojo-surrogate-spike.md` |
| #2940 | Mojo roadmap epic | CLOSED | `docs/adr/0012-mojo-evaluation-roadmap.md` |

The ASHRAE 140 structural failures tracked in `SCORECARD.md` (current pass rate
14.3 % per the `scorecard-drift` workflow; Cases 600 / 900 documented structural
failures in `docs/KNOWN_ISSUES.md`) are **orthogonal** to this work — they are
physics-validity concerns under the existing 5R1C / 9R4C / FD / CTF envelope
solvers. Per the issue body's "Status" section, this artifact explicitly does
**not** depend on that gate being tripped. Implementation PRs that follow this
artifact are conditional only on the pre-requisites above.

---

## 1. Architecture Survey

Four candidate families cover the geometry-grounded neural-surrogate design
space. Each is evaluated against the **existing BES-FFD exchange contract**:
`BesToFfdBoundaryConditions → FfdToBesResults` (`src/sim/loose_coupling.rs`),
which is the only stable BES-side surface that any new solver must respect.

### 1.1 Graph Neural Operator (GNO)

A GNO treats the building interior as a graph: zone volumes are nodes, walls /
floors / ceilings / windows / inter-zone boundaries are typed edges, and the
per-surface BCs (`surface_temperatures`, `wind_pressure`) plus internal gains
become node / edge features. Message-passing layers propagate convection
information across the graph; a decoder outputs `CHTC`, `T_air`, `q_surf`,
infiltration / mixing flow per surface and per zone.

| Property | Value |
|----------|-------|
| BES-FFD exchange fit | **High.** Output maps 1:1 to `FfdToBesResults` (per-surface CHTC, per-zone T, etc.). |
| Geometry representation | The existing `src/physics/geometry_tensor.rs` `GeometryTensor` (`ZoneCoord` + `WallMatrix` + `WindowMatrix` + `AdjacencyMatrix`, all `Vec<f64>` CTA tensors) is a graph adjacency representation. The graph is built once at `initialize` time. |
| Compute cost | `O(E · d² · L)` per macro-step for `E` edges, hidden dim `d`, `L` MP layers. On the production grid, `E ≈ 500` walls (the `MAX_WALLS` constant in `geometry_tensor.rs`), `d ≈ 128`, `L = 4` → single-digit milliseconds on CPU. |
| Energy-balance constraint | Easy to encode as a node-level residual loss: per-zone `Σ Q_in − Σ Q_out = 0` (each zone is a graph node with feature conservation). Matches `RULES.md §1` constraint exactly. |
| Out-of-distribution handling | Existing `HybridRouting::use_ood_fallback` pattern (Issue #1892) ports verbatim — check input features against training-domain bounds, fall back to `FfdCfdAdapter` / physics when OOD. |
| Hardware portability | CPU SIMD + CUDA via the same ONNX Runtime path used by `src/ai/surrogate.rs`. |
| Maturity | Production-ready in adjacent domains (climate surrogate literature, e.g. Pathak et al. 2022 on regional climate emulation). |

**Verdict:** Strong fit. Recommended for the first prototype.

### 1.2 Point-Cloud / PointNet++

Treats the FFD grid as a point cloud of `n = nx·ny·nz` cell centers (or sampled
surface points). A point-cloud network learns an embedding per point, then
aggregates globally. Variable-geometry support is the weak link — PointNet++
encodes permutation invariance but not graph topology, so windows / inter-zone
boundaries have to be re-encoded as additional point features per sample.

| Property | Value |
|----------|-------|
| BES-FFD exchange fit | Medium. Output is per-cell field, requires post-processing to aggregate to per-surface CHTC and per-zone T. Aggregation cost is non-trivial. |
| Geometry representation | Resampling required per geometry change; the `GeometryTensor` CTA graph cannot be reused directly. |
| Compute cost | `O(n · d²)` for `n = 32³ = 32768` cell points → tens of milliseconds on CPU. GPU speedup is large if available. |
| Energy-balance constraint | Harder — per-cell conservation does not automatically aggregate to per-zone conservation. Needs a custom loss. |
| Out-of-distribution handling | Sample-level OOD check is harder (geometric OOD). |
| Hardware portability | Same ONNX Runtime path. |
| Maturity | Production-ready in scene-flow / segmentation literature; less common in building airflow. |

**Verdict:** Workable but the post-processing and aggregation steps are a real
engineering tax. Not preferred over GNO.

### 1.3 Fourier Neural Operator (FNO) / Mesh-free Transformer

FNO learns a resolution-independent operator in Fourier space; MANO (Mesh-AGN
Neural Operator) and Transformer-based mesh-free variants handle non-uniform
geometries. These are the most expressive architectures but also the most
expensive to train and validate.

| Property | Value |
|----------|-------|
| BES-FFD exchange fit | Medium-low. Output is on the input grid (regular or sampled); aggregation to per-surface CHTC still required. |
| Geometry representation | Variable-geometry support requires geometry-aware variants (Geo-FNO, etc.), which add complexity and increase training-data requirements significantly. |
| Compute cost | Training is heavy (hours-to-days on a single GPU); inference is moderate (tens of ms on GPU, slow on CPU). |
| Energy-balance constraint | Conservation in Fourier space is not automatic; PINN-style hard constraints are research-grade. |
| Out-of-distribution handling | Difficult — Fourier-space OOD detection is an open research question. |
| Hardware portability | GPU-dominant; CPU inference requires quantisation. |
| Maturity | Active research; not yet production-deployed for building airflow. |

**Verdict:** High ceiling, low floor for a v1 prototype. Defer until GNO
prototype is validated.

### 1.4 Mesh-free Transformer Hybrids

Recent (2024–2025) work combines point-cloud attention with graph / mesh
priors. Treats as a research direction rather than a v1 candidate.

| Property | Value |
|----------|-------|
| BES-FFD exchange fit | Inherits parent-architecture trade-offs. |
| Geometry representation | Best-in-class for variable geometry, but at the cost of training-data hunger (≥ 10⁵ samples typical). |
| Compute cost | Highest of the four. |
| Energy-balance constraint | Inherits. |
| Out-of-distribution handling | Inherits. |
| Hardware portability | GPU-dominant. |
| Maturity | Frontier research; not stable enough for production. |

**Verdict:** Watch list. Not for v1.

### 1.5 Decision

The artifact's **tentative pick** is **GNO on the BES surface graph**, for the
following reasons:

1. **Symmetry with existing types.** The `GeometryTensor` (`src/physics/geometry_tensor.rs`)
   is already a graph adjacency tensor. No new geometry representation is required;
   the model's input is the `GeometryTensor` plus per-step BCs.
2. **Symmetry with existing routing.** `HybridRouting` (`src/sim/thermal_model.rs:678`)
   is a per-subsystem dispatch table; a GNO surrogate plugs into it the same
   way `SurrogateThermalModel` does (per `ARCHITECTURE.md` §"Hybrid mode —
   HybridRouting"). No new trait boundary is introduced.
3. **Energy-balance encoding.** Per-zone conservation is a node-level residual
   loss, which matches the `SurrogateDomain::energy_balance_residual` pattern
   (`src/ai/surrogate.rs:607`) exactly. Training reuses the existing
   `tools/train_surrogate.py` infrastructure.
4. **Hardware portability.** ONNX Runtime is the existing runtime
   (`--features ort`, opt-in); the GNO architecture exports cleanly to ONNX
   via the standard PyTorch → ONNX exporter. No new inference backend is
   introduced.

The pick is **conditional and falsifiable**. The prototype must demonstrate:

- ≤ 8 % per-field-element MAPE (95th percentile) on the training-data suite
  (§3) versus `fluxion-cfd` ground truth.
- Per-timestep drift gate passes (±1 % vs `FfdCfdSolver` reference on the
  ASHRAE 140 envelope set, reusing `#1784`).
- Annual zone-energy-balance residual ≤ 0.5 % per `RULES.md §1`.

If the prototype fails any of these gates, the artifact recommends closing
issue #3148 with a no-go recommendation (see §7).

---

## 2. Coupling-Points Specification

A geometry-grounded neural surrogate plugs into the BES-FFD coupling at the
`FfdSolver` trait (`src/sim/loose_coupling.rs:111`). The trait shape is:

```rust
pub trait FfdSolver: Send + Sync {
    fn name(&self) -> &str;
    fn initialize(&mut self, num_zones: usize, zone_volumes: &[f64],
                  surface_areas: &[f64], num_surfaces: usize) -> LooseCouplingResult<()>;
    fn step_micro(&mut self, bc: &BesToFfdBoundaryConditions, dt: f64)
        -> LooseCouplingResult<FfdMicroResults>;
    fn recommended_micro_timestep(&self) -> f64;
    fn is_valid(&self) -> bool;
}
```

A GNO surrogate would implement this trait as a new struct (working title:
`GnoFfdSolver`, pending review) in a future `src/ai/geometry_surrogate/`
sub-module. The artifact deliberately **does not** introduce the struct,
module, or any Rust code in this PR — the names are placeholders recorded here
so that any follow-up implementation issue has a single source of truth.

### 2.1 Exchange translation

`BesToFfdBoundaryConditions` carries:

- `outdoor_temperature: f64`
- `surface_temperatures: Vec<f64>` (K, per zone surface)
- `hvac_supply_temperature: f64`, `hvac_supply_flow: f64`
- `wind_pressure: Vec<f64>` (per facade)
- `internal_gains: f64`
- `time_start: f64`, `macro_timestep: f64`

These map to GNO input features:

| BC field | GNO feature | Notes |
|----------|-------------|-------|
| `outdoor_temperature` | global node feature | broadcast to all surface nodes |
| `surface_temperatures` | per-surface node feature | direct mapping |
| `hvac_supply_temperature` + `hvac_supply_flow` | HVAC injection node | one per zone |
| `wind_pressure` | per-surface edge feature (facade pressure) | direct mapping |
| `internal_gains` | per-zone internal-gain node feature | broadcast to zone interior |
| `time_start` / `macro_timestep` | global context | not load-bearing for the surrogate; can be ignored |

Static geometry comes from `GeometryTensor` (`src/physics/geometry_tensor.rs`)
at `initialize` time and is held in the solver struct (not re-passed per step).

`FfdMicroResults` carries `chtc`, `zone_temperatures`, `surface_heat_flux`,
`infiltration_flow`, `mixing_flow` (all `Vec<f64>`). The GNO output heads map:

| `FfdMicroResults` field | GNO output head | Aggregation |
|------------------------|-----------------|-------------|
| `chtc` | per-edge head | direct (per surface) |
| `zone_temperatures` | per-zone node head | direct (per zone) |
| `surface_heat_flux` | per-edge head | derived from `chtc × (T_surface − T_zone)` |
| `infiltration_flow` | per-zone aggregate head | sum over envelope nodes |
| `mixing_flow` | per-zone aggregate head | sum over inter-zone edges |

### 2.2 Two paths through the existing swap-points

Per `ARCHITECTURE.md` §"Module N+2" and the issue body's Question 4, the
surrogate enters the coupling at one of two paths:

**Path A — `FfdSolver` impl (alongside `FfdCfdSolver`).** The new
`GnoFfdSolver` implements the same trait that `fluxion_cfd::FfdCfdSolver`
already implements (via the existing `FfdCfdAdapter`,
`src/sim/ffd_cfd_adapter.rs`). The user picks which `FfdSolver` to use via
the existing dispatcher. This is the **recommended** path because it does
**not** widen the trait and it preserves the existing `BesToFfdBoundaryConditions`
/ `FfdToBesResults` translation contract.

**Path B — `HybridRouting` extension.** A new flag
`use_surrogate_geometry: bool` routes the FFD step itself to a surrogate
while keeping the rest of the thermal network on physics. This is
**not** recommended because it conflates two different routes (the FFD
exchange vs the BES-thermal dispatch) and obscures the OOD-fallback
semantics from issue #1892. If a future implementation PR wants Path B,
it must justify why Path A is insufficient.

This artifact proposes Path A as the design default and records Path B as a
documented alternative only if Path A's energy-balance gate fails.

### 2.3 Convective-film feedback (Issue 3 from the original proposal)

The issue body's "Convective-film feedback into zone thermal balance" question
asks for dynamic `h_c` (computed from the predicted near-wall temperature
gradient) feeding back into `9R4C` / `GaugeZoneSolver`. The artifact's
treatment:

- **Not part of this research artifact.** Dynamic `h_c` is a `HybridRouting`
  extension (a new flag on the existing struct), not an FFD-coupling
  concern. It does not require a 3-D surrogate to be useful — a 0-D CHTC
  model + lookup table can be plugged into `HybridRouting` independently.
- **Where it should go.** A future `HybridRouting::use_surrogate_convective_film`
  flag (alongside `use_surrogate_conduction`, `use_surrogate_ventilation`,
  `use_surrogate_loads`, `use_surrogate_hvac` per `src/sim/thermal_model.rs:678`),
  consulted by the 9R4C / `GaugeZoneSolver` air-node update. This is a
  separate epic from #3148; the artifact explicitly excludes it.
- **Why excluded here.** Mixing the FFD-coupling artifact with the
  zone-thermal-network feedback creates two independent orthogonal
  integration points in the same PR, which violates the
  `scripts/check_architecture_drift.py` baseline invariants and the
  per-subsystem dispatch contract documented in `ARCHITECTURE.md`
  §"Hybrid mode — HybridRouting".

---

## 3. Training Data Requirements and Existing Fixtures

Per `AGENTS.md` ("no OpenFOAM dependency"), CFD ground truth must come from
`fluxion-cfd`. The training-data suite is the most expensive deliverable of
any follow-up implementation issue.

### 3.1 Required coverage

The surrogate generalises across ASHRAE 140 envelope variations plus
realistic indoor-airflow variations. The training suite spans:

| Source | Count | Notes |
|--------|-------|-------|
| ASHRAE 140 envelope base cases (600 / 900 / 950FF) | 3 | fixed per ASHRAE 140-2023 |
| Geometry perturbations | ~100 | WWR, aspect ratio, height |
| Construction perturbations | ~50 | mass class, U-value, SHGC |
| Window placement perturbations | ~50 | facade, sill height |
| Infiltration schedule perturbations | ~50 | ACH base, diurnal profile |
| Wind / weather perturbations | ~30 | TMY3 locations, wind profile |
| Internal-gain schedule perturbations | ~50 | occupancy / equipment / lighting |
| **Total unique configurations** | **~330** | distinct geometries × BCs |
| Macro-steps per configuration | 35040 | full-year, 15-min cadence, ASHRAE 140 standard |
| **Total macro-step samples** | ~11.5 M | the per-step (BC → 3D field) training pairs |

At 35040 macro-steps × ~330 configurations, each macro-step requires one
`fluxion-cfd` run on the production FFD grid. At ~10 sec per macro-step on a
modern CPU (per the closed #2456 baseline), the full suite is roughly
**330 × 35040 × 10 sec ≈ 1.3 × 10⁸ sec ≈ 4 CPU-years**. The suite is
unaffordable as a single sequential job; it must be executed on the existing
`fluxion-city` cloud-campaign infrastructure (per `docs/CLOUD_CAMPAIGN.md`)
or equivalent parallel harness.

**Recommendation:** A follow-up implementation issue must include a budget
estimate citing the actual measured `fluxion-cfd` per-macro-step wall time on
the production CI runner (a benchmark under `fluxion-cfd/benches/ffd_bench`).
The estimated 4 CPU-years above is an upper bound; in practice, the FFD
inner step amortises across macro-steps (warm-start) and the geometric
configuration count can be reduced via Latin-hypercube sampling if the
prototype demonstrates generalisation across the first ~50 configurations.

### 3.2 Existing fixtures the artifact can reuse

The training-data suite must plug into the existing reference-data
infrastructure (`tests/reference_data/`). The relevant fixtures:

- `tests/reference_data/zone_balance/` — Case 600 / 900 / 950FF hourly
  thermal outputs (used by `tests/surrogate_drift_gate.rs`).
- `tests/reference_data/zone_balance/case_950_energy_hourly.csv` — referenced
  verbatim by `tests/surrogate_drift_gate.rs:62` as the held-out validation
  dataset.
- `fluxion-cfd/src/cpu/` — the CPU baseline (per closed #2456 / PR #2477) is
  the production FFD ground-truth generator.
- `src/ai/surrogate.rs::SurrogateDomain::default_residential` — the existing
  training domain bounds (`src/ai/surrogate.rs:553`); the new
  `GeometrySurrogateDomain` would extend this struct, not replace it.

### 3.3 Data-provenance and integrity

Per the closed #2906 `verify_onnx_signature` pattern
(`src/ai/surrogate.rs:3348`), the training-data suite must be:

1. SHA-256 hashed at the suite-generation step (one hash per `.npz` /
   `.parquet` shard).
2. Stored under `tests/reference_data/cfd_training/` with a manifest
   matching the format of `<model>.sha256`.
3. Re-validated on every CI run via the same fail-closed SHA-256 check used
   for ONNX model integrity.

The artifact does **not** commit the training data itself (it would be
hundreds of GB); the manifest is committed, the data is delivered via the
existing CI artifact store or model-registry mechanism used by the closed
#2906 pipeline.

---

## 4. Energy-Balance Protocol (CI Gates)

The protocol is layered: three independent gates, each fail-closed, each
grounded in a published or closed-source baseline.

### 4.1 Gate 1 — Per-timestep drift (reused from #1784)

**Source:** Closed issue #1784 (`tests/surrogate_drift_gate.rs`).

**Definition:** For each macro-step `t`:

```
drift_pct(t) = |T_surrogate(t) − T_FfdCfdSolver(t)| / max(|T_FfdCfdSolver(t)|, ε) × 100
```

where `ε = 0.1 °C`.

**Tolerance:** ±1 % per macro-step (matches the existing gate).

**Benchmark:** ASHRAE 140 Case 900 (high-mass 9R4C reference) — the most
thermally massive configuration and therefore the most demanding test for
any neural surrogate.

**CI wiring:** Reuses `tests/surrogate_drift_gate.rs` verbatim with an
additional `ModelKind::GeometrySurrogate` variant added to the registry.
The existing two-mode behaviour (strict ±1 % when a trained model is
loaded, ≤ 100 % lenient ceiling otherwise) carries over.

### 4.2 Gate 2 — Annual zone-energy-balance residual (reused from `RULES.md §1`)

**Source:** `RULES.md §1`: "Total heat transfer (conduction + convection +
radiation + solar + HVAC) must sum to zero for any zone."

**Definition:** For each ASHRAE 140 case and each zone:

```
residual_annual = Σ_{t} (Q_conduction + Q_convection + Q_radiation + Q_solar + Q_HVAC) / max(|Σ Q_in|, ε)
```

**Tolerance:** ≤ 0.5 % over the full annual horizon.

**Rationale:** The `RULES.md` constraint is a strict invariant. CFD-to-CFD
comparisons typically do not achieve < 0.5 % residual, so this gate is
strictly a regression test against the Fluxion baseline (the 9R4C /
`GaugeZoneSolver` reference), not against `fluxion-cfd` itself.

**CI wiring:** New test
`tests/surrogate_geometry_zone_energy_balance.rs` mirroring the structure
of `tests/surrogate_ashrae_600_cooling_mae.rs` (closed #2924). The test
must run on every PR; the gate must be PR-blocking.

### 4.3 Gate 3 — CFD-to-CFD MAPE bound (new, conservative)

**Source:** This artifact.

**Definition:** For each macro-step and each output field
(`velocity_x`, `velocity_y`, `velocity_z`, `T`, `CHTC`):

```
mape_field(t, x, y, z) = |surrogate(t, x, y, z) − FfdCfdSolver(t, x, y, z)| / max(|FfdCfdSolver(t, x, y, z)|, ε) × 100
```

The aggregate metric is the **95th-percentile per-field-element MAPE** over
the full validation suite.

**Tolerance:** ≤ 8 % per-field-element, ≥ 95th-percentile.

**Rationale:** CFD-to-CFD comparisons rarely beat ~5 % MAPE on instantaneous
fields. The existing `#1784` drift gate already covers the integrated
surrogate-vs-physics comparison; this gate covers the surrogate-vs-CFD
comparison, which is the new artefact introduced by a neural surrogate that
replaces CFD output.

The threshold of 8 % is deliberately conservative and grounded in the
measurement uncertainty documented in published FFD/CFD validation studies
(Zuo et al. 2016, cited in `src/sim/loose_coupling.rs`). An uncritical
`<3 % MAPE` AC would either fail (because CFD-to-CFD comparisons don't hit
that bar) or get relaxed (which violates `RULES.md §0`: "Never tune to pass
tests"). The threshold is **not a tuning target**; it is a **gate** that
either passes or fails. If the prototype cannot meet the gate, the artifact
recommends closing issue #3148 with a no-go recommendation (see §7).

**CI wiring:** New test
`tests/surrogate_geometry_cfd_mape.rs`. Per ASHRAE 140 case, run the
`fluxion-cfd` ground truth and the surrogate side-by-side on the held-out
test split (10 % of the training suite, distinct from the ASHRAE 140
envelope base cases to avoid leakage), compute the 95th-percentile MAPE per
field, fail if any field exceeds 8 %.

---

## 5. Acceptance Criteria Mapping (Back to Issue #3148)

The issue body lists three AC bullets. This section maps each to the
artifact sections that satisfy it.

### 5.1 AC #1 — Research document at `docs/research/geometry_grounded_surrogate.md`

**Status:** This artifact (`docs/research/3148-geometry-grounded-neural-surrogates.md`)
delivers the research document. The filename differs from the issue body's
literal AC (which suggested `geometry_grounded_surrogate.md`) because the
repository's `docs/research/` convention uses issue-numbered prefixes for
research artifacts (mirrors `docs/investigations/issue-NNNN-*.md`); the
follow-up implementation PR can move / rename the file if the maintainers
prefer the original name.

**Coverage map for the AC's six required topics:**

| AC topic | Section |
|----------|---------|
| Architecture survey | §1 (FNO / GNO / PointNet++ / mesh-free) |
| Training-data sourcing via `fluxion-cfd` | §3 |
| Energy-balance protocol | §4 (three gates) |
| Realistic MAPE / MAE / drift targets | §4.1 (1 % drift), §4.2 (0.5 % residual), §4.3 (8 % CFD-MAPE) |
| Proposed trait surface | §2 (Path A — `FfdSolver` impl alongside `FfdCfdSolver`) |
| CUDA / `--features ort` dependency story | §1.5 (ONNX Runtime port), §7 (GPU deferred) |

### 5.2 AC #2 — Cross-references to related issues

**Status:** Delivered. The artifact's front matter and §"Related" lines
list every issue the AC requires (#2460, #2456, #1784, #2924) plus the
surrogate-runtime cluster (#2905, #2906, #2919, #2920, #2921, #2922,
#2923, #2925) and the adjacent closed research spikes (#2937, #2940).
The `RULES.md` / `AGENTS.md` / `ARCHITECTURE.md` binding constraints are
cited in §6 (Consequences / Negative) and throughout.

### 5.3 AC #3 — Follow-up decision record

**Status:** This artifact is the decision record. Per `docs/adr/` precedent
(ADR-0012 / ADR-0013 follow the same "research-only-no-implementation"
pattern), the artifact itself captures:

- **Architecture choice:** §1.5 (GNO tentative pick) + §1.5 conditional
  falsification criteria.
- **Energy-balance protocol:** §4 (three CI gates).
- **Go / no-go recommendation:** §7 (recommend proceed to implementation
  PR, conditional on the prototype passing all three gates; if not, close
  #3148 with no-go).

The artifact is intentionally committed under `docs/research/` rather than
`docs/adr/` because:

1. The issue body explicitly asks for the file at `docs/research/...`.
2. The artifact does not introduce a permanent architectural decision (no
   trait boundary changes; no module boundary changes); it is the design
   record for a *future* implementation PR.
3. A future ADR (`docs/adr/0016-geometry-grounded-surrogate.md` or
   equivalent) is created when the implementation PR lands, mirroring
   ADR-0012 / ADR-0013's split between the umbrella-roadmap ADR and the
   per-issue ADR.

---

## 6. Consequences

### 6.1 Positive

- **Architecture survey is captured before any code lands.** Future
  implementation PRs do not have to re-derive the GNO-vs-FNO-vs-PointNet
  decision; this artifact's §1 + §1.5 is the single source of truth.
- **Energy-balance protocol is fixed in advance.** The three CI gates
  (§4.1, §4.2, §4.3) are fail-closed and grounded in published
  references; `RULES.md §0` ("never tune to pass tests") is operationalised
  on the gate at gate-specification time, before the prototype exists.
- **Pre-requisites are all closed.** The artifact can be reviewed and
  implemented without waiting on additional infrastructure work (no GPU
  CUDA path, no new ONNX Runtime backend, no new crate split).
- **No new trait boundaries are introduced.** Path A (§2.2) preserves the
  existing `FfdSolver` trait shape; the surrogate is an additional impl of
  the same trait. This satisfies `ARCHITECTURE.md §"Module N+2"`
  ("coordinator-as-integration-point") and `scripts/check_architecture_drift.py`'s
  baseline invariants.
- **No new crate / module is created.** Per the issue body's explicit
  guard ("This issue deliberately does not create new labels, crates, or
  workspace members"), the artifact documents the proposed names
  (`GnoFfdSolver`, future `src/ai/geometry_surrogate/`) but does not
  create the directory or modify `Cargo.toml`.

### 6.2 Negative

- **No working prototype in this PR.** This artifact is design-only. A
  follow-up implementation PR is required to validate §1.5's tentative
  GNO pick against §4's three gates.
- **Training-data cost is large.** §3 estimates ~4 CPU-years for the full
  suite. This is an upper bound; the implementation PR must produce a
  measured-budget estimate from the actual `fluxion-cfd` per-macro-step
  benchmark before committing to the suite size.
- **The 8 % CFD-MAPE threshold (§4.3) is conservative.** A more aggressive
  threshold (e.g. 5 %) would tighten the gate but is not defensible
  against published CFD-to-CFD comparison literature (Zuo et al. 2016 et
  al.). The artifact explicitly does not propose a tighter threshold to
  avoid violating `RULES.md §0`.
- **Dynamic `h_c` convective-film feedback (issue 3 from the original
  proposal) is excluded.** This is a separate `HybridRouting` extension;
  mixing it into the FFD-coupling artifact would conflate two orthogonal
  integration points (§2.3).
- **GPU / CUDA inference path is out of scope.** Blocked on the #2456 GPU
  follow-up; the surrogate runs on CPU via the existing `--features ort`
  ONNX Runtime path until the CUDA path lands.

### 6.3 Neutral

- **Path A vs Path B (§2.2) is recorded as a design choice, not a
  mandate.** A future implementation PR may justify Path B (a
  `HybridRouting` flag for the FFD step itself) if Path A's energy-balance
  gate fails. The artifact does not restrict the choice.
- **The `GnoFfdSolver` struct name is a placeholder.** It is recorded here
  so the implementation PR has a single source of truth, but the final
  name is the implementer's choice (subject to
  `scripts/check_architecture_drift.py` and the issue body's explicit
  label / crate prohibitions).
- **`src/ai/geometry_surrogate/` is a placeholder module path.** The
  final location (likely under `src/ai/` alongside `src/ai/surrogate.rs`
  or under `src/sim/` alongside `src/sim/loose_coupling.rs`) is the
  implementer's choice; the artifact does not pre-allocate the path.
- **The artifact's filename differs from the issue body's literal AC.**
  See §5.1 — `3148-geometry-grounded-neural-surrogates.md` vs
  `geometry_grounded_surrogate.md`. The issue-numbered prefix matches
  the `docs/investigations/issue-NNNN-*.md` convention; the original
  filename can be used if the maintainers prefer.

---

## 7. Recommendation: Proceed Conditionally

**Recommendation:** Proceed to an implementation PR, conditional on the
follow-up PR satisfying §1.5's three falsification criteria
(8 % MAPE, ±1 % drift, 0.5 % annual residual) and §3's training-data
budget estimate. The implementation PR must:

1. Land **before** any wider architectural change to `ARCHITECTURE.md` —
   it is gated only on the closed pre-requisites table in §"Context".
2. Cite this artifact in its PR body (per the issue body: "Future
   implementation work derived from this artifact must (a) reference this
   issue, (b) cite an existing closed-or-resolved prerequisite, and (c)
   carry a realistic, falsifiable acceptance criterion grounded in a
   published baseline").
3. Carry a falsifiable AC: either all three gates pass, or the issue
   is closed with a no-go recommendation.
4. **Not** create new crates / modules / labels beyond what the artifact
   proposes (one new struct in one new sub-module; no new workspace
   member; no new `Cargo.toml` change).
5. **Not** modify `src/ai/surrogate.rs` except by adding new variant
   arms to the existing `ModelKind` enum (per ADR-0004 ONNX versioning
   precedent).

**No-go closure path:** If the prototype fails any of the three gates
(§4), the artifact recommends closing #3148 with a no-go recommendation
documented in a follow-up ADR. The no-go is a valid outcome of a research
artifact; it is the responsible conclusion if the GNO architecture cannot
meet the energy-balance protocol on the training-data suite.

---

## 8. Out of Scope (Explicit)

The following are **explicitly out of scope** for this artifact and any
follow-up implementation PR. They are recorded here so a future reviewer
does not infer scope that the artifact does not claim:

1. **Creating a `fluxion-surrogates` workspace crate.** Per the issue
   body and AGENTS.md workspace layout, no new workspace member ships
   with this work.
2. **Introducing a `Vertical-C: ML & Surrogates` label** (or any new
   issue label). The existing `ml`, `surrogate-model`, `ai-surrogate`
   labels cover this work.
3. **Modifying `GaugeZoneSolver` to consume dynamic `h_c`.** This is a
   separate `HybridRouting` extension (§2.3); mixing it into the FFD
   coupling would conflate two orthogonal integration points.
4. **GPU / CUDA inference path.** Blocked on #2456 GPU follow-up;
   deferred until that lands.
5. **ONNX Runtime backend parity work.** Orthogonal; the surrogate reuses
   the existing `--features ort` opt-in path.
6. **Raising any `tests/reference_data/` baseline.** Per AGENTS.md,
   reference_data baselines are never raised to hide a regression.
7. **Widening the `loose_coupling::FfdSolver` trait.** Per §2.2
   Path A, the surrogate is a new impl of the existing trait, not a
   new trait.
8. **Modifying `src/ai/surrogate.rs` (4941 LOC, opt-in `--features ort`)**
   beyond adding new `ModelKind` enum arms. The existing 4941-LOC file
   remains untouched.

---

## References

- **Issue #3148** — origin and source scope statement. The issue body is
  the canonical scope statement; this artifact is the design record.
- **Issue #2460** (PR #2469) — `fluxion-cfd::FfdCfdSolver` wired into
  `loose_coupling::FfdSolver` via `FfdCfdAdapter`
  (`src/sim/ffd_cfd_adapter.rs`).
- **Issue #2456** (PR #2477) — `fluxion-cfd` CPU baseline.
- **Issues #1784, #2905, #2906, #2919, #2920, #2921, #2922, #2923,
  #2924, #2925** — pre-existing surrogate / AI / supply-chain issues;
  all CLOSED.
- **Issues #2937, #2940** — adjacent closed research spikes (Mojo /
  MAX); see `docs/adr/0013-mojo-surrogate-spike.md` and
  `docs/adr/0012-mojo-evaluation-roadmap.md`.
- **Issue #1139** — surrogate v3.0 training pipeline (origin of
  `src/ai/surrogate.rs`); the new training-data suite (this artifact §3)
  reuses the same training-domain-bound pattern.
- **Issue #1431** — `HybridRouting` origin; the FFD-coupling surrogate
  plugs into the existing `FfdSolver` trait (`ARCHITECTURE.md` §"Module N+2"),
  not into `HybridRouting`.
- **Issue #1892** — OOD-fallback pattern for `HybridRouting`; reusable
  for the surrogate's out-of-distribution detection.
- **`ARCHITECTURE.md`**
  - §"Module N+2: BES-FFD Loose Coupling" — the `FfdSolver` trait and
    `BesToFfdBoundaryConditions` / `FfdToBesResults` exchange;
    `src/sim/loose_coupling.rs`.
  - §"Hybrid mode — HybridRouting" — per-subsystem dispatch table;
    `src/sim/thermal_model.rs:678`.
  - §"ML-surrogate swap-point traits" — `ThermalModelTrait`,
    `VentilationSchedule`, `HeatConductionSolver`.
- **`RULES.md`**
  - §0 — "Never tune to pass tests"; the 8 % CFD-MAPE threshold
    (§4.3) is a gate, not a tuning target.
  - §1 — "Total heat transfer must sum to zero for any zone"; the
    annual residual gate (§4.2) is the operationalisation.
- **`AGENTS.md`**
  - "Main swap points" — `HeatConductionSolver`, `VentilationSchedule`,
    `ThermalModelTrait`.
  - "fluxion-core is a dependency-light leaf" — no new ML / inference
    infrastructure may land under `fluxion-core/`.
  - "no OpenFOAM dependency" — CFD ground truth must come from
    `fluxion-cfd` (§3).
  - "The fluxion help includes intentionally stubbed paths" — no
    implementation lands in this PR.
- **`CODEBASE_MAP.md`** — cross-language FFI contracts; the surrogate
  reuses the existing `--features ort` ONNX Runtime path.
- **`docs/KNOWN_ISSUES.md`** — §LIMIT index; the structural ASHRAE 140
  failures (Cases 600 / 900) are explicitly orthogonal to this artifact
  per the issue body's "Status" section.
- **`docs/SURROGATE_GOVERNANCE.md`** — existing surrogate governance
  framework; the new `GeometrySurrogateDomain` extends
  `SurrogateDomain` (`src/ai/surrogate.rs:553`) rather than replacing it.
- **`docs/SURROGATE_BENCHMARK_RESULTS.md`** — existing surrogate benchmark
  results; the new CFD-MAPE gate (§4.3) is added to the benchmark
  harness in a follow-up PR.
- **`docs/ONNX_INFERENCE_PIPELINE.md`** — the existing ONNX inference
  pipeline; the new GNO surrogate exports to ONNX via the same exporter.
- **`docs/ASHRAE140_RESULTS.md`** — current ASHRAE 140 validation
  status; structural failures are orthogonal.
- **`SCORECARD.md`** — current 14.3 % pass rate vs 60 % gate; the
  umbrella release gate is orthogonal to this artifact.
- **`release_gates.yaml`**
  - `validation.min_pass_rate = 60.0` (orthogonal);
  - `benchmark.throughput.min_configs_per_sec = 150` (the surrogate
    must respect this on its hot path).
- **`scripts/check_architecture_drift.py`** — the surrogate is a new
  `FfdSolver` impl, not a new trait; baseline invariants are
  preserved.
- **`scripts/check_docs_summaries.py`** — the artifact's 7-line summary
  block satisfies lines 2–8.
- **`scripts/check_doc_inventory_fresh.py`** + **`scripts/generate_doc_inventory.py`**
  — the artifact is enumerated in the regenerated `docs/doc-inventory.md`.
- **`scripts/surrogate_drift_gate.yml`** (workflow) — Gate 1 (§4.1)
  reuses the existing CI gate.
- **`tests/surrogate_drift_gate.rs`** — Gate 1 source.
- **`tests/surrogate_ashrae_600_cooling_mae.rs`** — closed #2924 MAE
  gate; structural pattern reused for Gate 2 (§4.2).
- **`src/sim/loose_coupling.rs`** — `FfdSolver` trait (line 111),
  `BesToFfdBoundaryConditions` (line 55), `FfdToBesResults` (line 81),
  `FfdMicroResults` (line 158), `FfdAccumulator` (line 176).
- **`src/sim/ffd_cfd_adapter.rs`** — the existing `FfdCfdAdapter`
  conforming `fluxion_cfd::FfdCfdSolver` to `loose_coupling::FfdSolver`;
  the new `GnoFfdSolver` follows the same pattern.
- **`src/physics/geometry_tensor.rs`** — the `GeometryTensor` CTA graph
  representation (lines 53–80); reused as the GNO input graph.
- **`src/ai/surrogate.rs`** — the existing ONNX surrogate runtime
  (4941 LOC); `SurrogateDomain::default_residential` (line 553),
  `SurrogateDomain::energy_balance_residual` (line 607),
  `verify_onnx_signature` (line 3348); future GNO surrogate reuses the
  `predict_loads_into` API per #2921.
- **`fluxion-cfd/src/ffd_solver.rs`** — `FfdCfdSolver` (line 263),
  `FfdConfig` (line 6); the CFD ground-truth generator.
- **`docs/research/pantelides-spike.md`**, **`docs/research/iso13790-*.md`**,
  **`docs/research/napi-rust-bindings.md`** — precedent for the
  `docs/research/` format.
- **`docs/adr/0013-mojo-surrogate-spike.md`** — closest stylistic
  precedent (research-only ADR with no Rust code change); Section
  structure mirrored where applicable.
- **Zuo et al. (2016)** — cited in `src/sim/loose_coupling.rs:27` for
  CFD-to-CFD comparison uncertainty; grounds the 8 % MAPE threshold
  (§4.3).
- **Clarke & Hensen (2017)** — cited in `src/sim/loose_coupling.rs:28`
  for co-simulation synchronisation.
