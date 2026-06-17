# RFC-0001: Dynamic Boundary Agent

**Status:** DRAFT  
**Authors:** Research & Innovation Engineer  
**Date:** 2026-05-12  
**Related Issues:** #708 (MIRAI temporal forecasting), #718 (surrogate model), #719 (LHC training data)  
**Target Milestone:** v2.5 (post-v1.3 blind ASHRAE 140 validation)

---

## Abstract

This RFC proposes replacing fluxion's current static boundary condition inputs (ASHRAE 90.1 occupancy CSVs and hardcoded grid pricing schedules) with a **Dynamic Boundary Agent** — a MIRAI-style LLM agent that uses FAISS retrieval over historical sequences and chain-of-thought temporal reasoning to produce **probability distributions** over occupancy, demand-response events, and weather conditions. The BatchOracle simulation engine draws 32 Monte Carlo samples per simulation run from these distributions, enabling uncertainty-aware energy predictions without requiring changes to the physics core or surrogate model architecture.

---

## 1. Problem Statement

### 1.1 Current State

Fluxion's boundary conditions are sourced at simulation time from:

- **Occupancy**: Static schedule CSVs following ASHRAE 90.1 typical-day patterns
- **Grid/DR Events**: Hardcoded time-of-use electricity pricing schedules
- **Weather**: Static Typical Meteorological Year (TMY) files

These inputs are deterministic point estimates. Real buildings, however, operate under stochastic boundary conditions:

- Occupancy deviates from schedules during holidays, events, and behavioral shifts
- Demand-response events are discrete stochastic occurrences tied to grid stress signals
- Weather follows probabilistic short-range forecasts, not historical averages

### 1.2 Impact

Static boundary conditions produce:

1. **Overconfident energy predictions** — no uncertainty bounds on outputs
2. **Poor MPC performance** — control policies optimized against wrong inputs (e.g., pre-cooling against static TMY when a heatwave is forecast)
3. **Systematically biased validation** — ASHRAE 140 passes despite real-world prediction gaps driven by occupancy/weather mismatch

### 1.3 Scope Exclusion

This RFC does **not** propose changes to:
- The physics solver (CTF/FD) — see #726
- The surrogate model architecture (MLP/PINN) — see #718, #764
- ASHRAE 140 validation boundary conditions — those remain deterministic per test spec

---

## 2. Proposed Design

### 2.1 Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │         Dynamic Boundary Agent           │
                    │                                          │
Historical data ──► │  1. Retrieval (FAISS/Chroma)             │
Calendar context──► │     nearest-neighbor over historical     │
ISO price signal──► │     sequences                            │
NWP forecast    ──► │                                          │
                    │  2. Temporal Reasoning (CoT)             │
                    │     chain-of-thought over retrieved      │
                    │     context + current inputs             │
                    │                                          │
                    │  3. Structured Output                    │
                    │     probability distributions per        │
                    │     boundary variable, 72h horizon       │
                    └─────────────────────────────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          │    32 stochastic draws   │
                          │    (Monte Carlo)          │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │      BatchOracle          │
                          │  (existing, unchanged)    │
                          │  32 parallel sim runs     │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   Aggregated Outputs      │
                          │   mean ± σ per timestep   │
                          └───────────────────────────┘
```

### 2.2 Boundary Variable Specifications

#### 2.2.1 Occupancy Distribution

**Input sources:**
- Historical building sensor data (preferred) or Time Use Survey (TUS) data as prior
- Calendar context: day-of-week, public holidays, local events
- Building type metadata (office, residential, mixed-use)

**Retrieval strategy:** FAISS cosine-similarity search over embeddings of historical day-patterns. Match current context (weekday/holiday/weather) to nearest k=5 historical sequences.

**Output:** `OccupancyDistribution` — hourly probability distribution over occupancy levels (fraction of max occupancy), 72h horizon.

```
P(occ_t = x) for x ∈ [0.0, 1.0] for each t ∈ [now, now+72h]
```

**Fallback:** If no historical data available, use ASHRAE 90.1 schedule as prior mean with ±15% standard deviation.

#### 2.2.2 Demand-Response Event Probability

**Input sources:**
- ISO/RTO real-time price signals (ISO-NE, PJM, CAISO, MISO, SPP, ERCOT)
- Day-ahead price forecasts from utility API or NREL OpenEI
- Historical DR event log

**Output:** `DREventProbability` — hourly probability of active DR event, 72h horizon.

```
P(dr_active_t = 1) ∈ [0.0, 1.0] for each t ∈ [now, now+72h]
```

**Integration with RL policy:** `dr_event_probability` is passed as a state feature to the RL policy model, enabling pre-emptive load shifting.

#### 2.2.3 Weather Ensemble

**Input sources:**
- NOAA GFS short-range forecast (0.25° grid, 384h horizon, 3h resolution)
- NOAA HRRR where available (3km CONUS, 18h horizon, 1h resolution)
- Historical climate data for covariance estimation

**Output:** `WeatherEnsemble` — ensemble of N=32 weather realizations spanning the forecast uncertainty cone.

**Fallback:** If NWP unavailable, use TMY ± historical seasonal variance.

### 2.3 Monte Carlo Integration with BatchOracle

The existing `BatchOracle` API already supports parallel simulation runs via rayon. The Dynamic Boundary Agent supplies stochastic boundary inputs to the existing API:

```rust
// Proposed extension to BatchOracle input type (non-breaking)
pub struct SimulationBatch {
    pub building_config: BuildingConfig,
    pub boundary_conditions: BoundaryConditions,
    // NEW: optional stochastic override
    pub stochastic_boundaries: Option<Vec<StochasticBoundaryDraw>>,
}

pub struct StochasticBoundaryDraw {
    pub draw_index: usize,             // 0..31
    pub occupancy_schedule: Vec<f64>,  // hourly, 72h
    pub dr_events: Vec<bool>,          // hourly, 72h
    pub weather: WeatherTimeSeries,    // hourly, 72h
}
```

When `stochastic_boundaries` is `Some(draws)`, BatchOracle runs `draws.len()` simulations in parallel and returns `SimulationEnsemble` with per-timestep mean and standard deviation.

**Performance target:** 32 draws × ~900 configs/sec throughput = ~28K configs/sec ensemble throughput. Rayon work-stealing already handles this.

### 2.4 Agent Configuration

```yaml
# .sdd/agents/dynamic-boundary-agent.yaml (proposed)
agent: dynamic_boundary
version: 0.1.0-draft
description: >
  MIRAI-inspired temporal reasoning agent that replaces static ASHRAE boundary
  condition CSVs with stochastic occupancy, DR event, and weather distributions
  for uncertainty-aware building energy simulation.

inputs:
  historical_occupancy:
    type: TimeSeries
    required: false
    fallback: ashrae_90_1_schedule
    description: Building occupancy sensor data or TUS prior
  calendar_context:
    type: CalendarFeatures
    required: true
    fields: [day_of_week, is_holiday, local_event_flags]
  iso_price_signal:
    type: TimeSeries
    required: false
    description: ISO/RTO real-time and day-ahead price signal
  nwp_forecast:
    type: WeatherForecast
    required: false
    fallback: tmy_with_seasonal_variance
    description: NOAA GFS or HRRR short-range forecast

reasoning:
  type: chain_of_thought
  retrieval_backend: faiss               # or chroma
  retrieval_k: 5                         # nearest neighbor sequences
  horizon_hours: 72
  llm_model: gpt-4o-mini                 # or local llama for offline use

outputs:
  occupancy_distribution:
    type: StochasticSchedule
    description: Probability distribution over hourly occupancy, 72h horizon
  dr_event_probability:
    type: HourlyProbability
    description: P(dr_active) per hour, 72h horizon
  weather_ensemble:
    type: WeatherEnsemble
    description: N=32 weather realizations across forecast uncertainty cone

integration:
  batch_oracle:
    monte_carlo_samples: 32
    output_aggregation: [mean, std_dev, p10, p50, p90]
  rl_policy:
    pass_features: [dr_event_probability, occupancy_distribution_mean]
```

---

## 3. Dependencies

| Dependency | Issue | Required for | Status |
|------------|-------|--------------|--------|
| LHC parameter distributions | #719 | Boundary variable range calibration | In progress (LHC bounds provided) |
| BatchOracle surrogate API | #718, #764 | Stochastic draws must pass through surrogate | In progress (Phase 4a) |
| FD solver for high-mass | #726 | Accurate thermal response under stochastic weather | Planned |
| Sol-air LW correction | #741 | Accurate boundary condition processing | In progress |

**Pre-condition:** Dynamic Boundary Agent is additive — it can be implemented independently of the physics solver work. The BatchOracle API extension is non-breaking (optional field).

---

## 4. Open Questions

These questions require team input before moving from DRAFT to PROPOSED:

### Q1: Occupancy data source

**Question:** For the initial implementation, do we:
- (a) Ship with ASHRAE 90.1 as the prior ± variance, requiring no external data?
- (b) Require integration with a building sensor API (BACnet, Haystack, BRICK)?
- (c) Use public Time Use Survey (TUS) data as a demographic prior?

**Recommendation:** Start with (a) for v2.5 prototype; upgrade to (b) in v3.0 when fluxion has building management system (BMS) connectivity.

### Q2: FAISS index update cadence

**Question:** How frequently should the FAISS retrieval index be updated?
- Real-time (continuous append): highest accuracy, highest compute
- Daily batch update: reasonable trade-off
- One-time initialization with manual refresh: lowest complexity

**Recommendation:** Daily batch update for v2.5; evaluate real-time streaming in v3.0 based on usage patterns.

### Q3: LLM for temporal reasoning — hosted vs. local

**Question:** Chain-of-thought temporal reasoning requires an LLM. Options:
- Hosted (GPT-4o-mini via API): easier to deploy, API cost, data privacy concerns
- Local (llama3.2, Phi-4-mini via llama.cpp): offline-capable, no data egress, higher setup cost
- Fine-tuned (Time-R1 style): highest accuracy, significant training effort

**Recommendation:** Hosted (GPT-4o-mini) for prototype; add local model option for privacy-sensitive deployments.

### Q4: Monte Carlo sample count — 32 vs. configurable

**Question:** Is 32 fixed draws appropriate, or should this be user-configurable?

**Recommendation:** Configurable via `batch_oracle.monte_carlo_samples`, default 32, min 8, max 128.

---

## 5. Non-Goals

- Replacing ASHRAE 140 test case boundary conditions (those remain static per standard)
- Real-time streaming inference (targeted for v3.0+)
- Physics-layer changes (out of scope — see #726, #741)
- Integration with BMS/BACnet protocols (v3.0+)

---

## 6. References

1. MIRAI — Temporal Complex Event Forecasting with LLM Agents (Ye et al., NeurIPS LAW 2025). https://github.com/yecchen/MIRAI
2. Time-R1 — Slow-thinking temporal reasoning with GRIP RL (2024)
3. AutoLFM — Multi-agent LLM for building load forecasting, 12.3% R² improvement (2025)
4. LLM-driven generative agents for occupant behavior with RAG+FAISS (2025)
5. Stochastic differential equations for building occupancy modeling (2024)
6. #708 research comment: https://github.com/anchapin/fluxion/issues/708#issuecomment-4434722718

---

## Appendix A: Agent YAML (full)

See Section 2.4 above.

## Appendix B: Proposed BatchOracle API Extension

See Section 2.3 above.
