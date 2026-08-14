# ADR-0005: Acausal HVAC / Fluid-System Port-Traits (fluxion-fluid)

- **Status:** Accepted
- **Date:** 2026-08-14
- **Deciders:** Fluxion maintainers
- **Supersedes:** None
- **Depends on:** None
- **Issue:** #1980

---

## Executive Summary

We adopt an **acausal, port-based** representation for HVAC plant loops, air
loops, and other fluid/thermal networks as the basis for the `fluxion-fluid`
crate. This document records the design decision that drives the `fluxion-fluid`
crate's public API (`ports/`, `graph/`, `mediums/`, `solvers/`) and the
separation between `fluxion-core/src/fluid/` (legacy leaf module — now deprecated
for new code) and `fluxion-fluid/` (the real acausal HVAC/fluid domain layer).

## Context

Fluxion's v1.0–v1.2 thermal core models envelope heat transfer (5R1C, 9R4C,
MultiNode) but lacks a first-class representation for **plant-side components**
(chillers, boilers, pumps, pipes, coils, valves, radiators). Engineers need to
assemble these into **systems** that solve as a coupled DAE — not as a
predefined directed acyclic graph.

Three options were considered:

1. **Causal/block-diagram:** each component exposes explicit input/output pins.
   Easy to implement, but the system assembly must be redrawn whenever the
   physical topology changes (e.g., reversing flow in a bypass loop).
2. **Equation-based / acausal port-based:** each component exposes ports whose
   flow direction is determined by the network solver. This is the Modelica /
   VHDL-AMS / FMI-2 convention. The same component can be reused in any
   topology without rewiring.
3. **Black-box co-simulation (FMI/FMU):** wrap an external simulator. Powerful
   but adds an FMI runtime dependency, licensing complexity, and a hard
   performance floor.

## Decision

We adopt option **(2) acausal port-based** modeling as the foundation of the
`fluxion-fluid` crate.

Specifically:

- Each component implements a `Component` trait with strongly-typed ports
  (e.g., `FluidPort { kind: PortKind, medium: Medium }`).
- Connections are formed by a `Network` graph that supports **bidirectional
  flow**; direction emerges from the solver, not the topology.
- The `Medium` trait covers Air, Water, Glycol mixtures, and a generic ideal-gas
  extension. Mediums are an open set — new fluids can be added by downstream
  crates without modifying `fluxion-fluid`.
- The default solver is a **Pantelides-index-reduced DAE solver** with
  Newton-Raphson iteration. A **WASM-compatible sequential fallback** is
  shipped as a feature-default for browser/CAD-bound deployments.
- `fluxion-fluid` is a **separate workspace member** (not a leaf in
  `fluxion-core`). The leaf `fluxion-core/src/fluid/` module predates this
  decision and is preserved only for backward-compatibility — new code must
  use `fluxion-fluid::ports`, `fluxion-fluid::graph`, `fluxion-fluid::mediums`,
  `fluxion-fluid::solvers`.
- Co-simulation (option 3, FMI/FMU) is deferred — see `docs/adr/0006-ffd-feasibility-study.md`
  for the related airflow co-simulation ADR.

## Consequences

### Positive

- Components are reusable across topologies; a Chiller is the same Chiller
  whether it's in a primary/secondary loop or a heat-recovery loop.
- The same model can target Rust, Python (`fluxion-fluid-py`), Node (napi),
  and WASM (`fluxion-wasm`) without rewrite.
- The Pantelides solver naturally handles stiff multi-domain systems (water
  + air + refrigerant) without manual decoupling.

### Negative

- The acausal API has a steeper learning curve than causal/block diagrams.
  New contributors must understand port-typing and graph assembly before they
  can add a component.
- The DAE solver is heavier than a causal time-marcher; the WASM fallback
  trades accuracy for portability. Benchmarked overhead per step on a
  6-component loop: ~3× a causal solver (see Issue #1980 follow-ups).

### Neutral

- `fluxion-core/src/fluid/` is now an orphan leaf module. It is retained
  because downstream crates may still reference it, but new code MUST use
  `fluxion-fluid`. A follow-up issue will either delete it or move its
  contents into `fluxion-fluid/src/graph.rs`.

## Alternatives Considered

- **Causal/block-diagram:** rejected for plant-side because it forces
  topology-aware component graphs.
- **FMI/FMU-only:** rejected for cost/latency; reserved as a future
  co-simulation option (ADR-0006).

## References

- Issue #1980 — original feature request
- `fluxion-fluid/README.md` — public surface
- `ARCHITECTURE.md` §"Module N: fluxion-fluid" — module boundaries
- ADR-0006 — Fast Fluid Dynamics feasibility study (separate concern)