# LDLP Evaluation — Latent-Dynamics World Model + MCTS Planning for the Fluxion RL Gym

**Issue:** #4237 — research spike: evaluate the LDLP paper (latent-dynamics world model + MCTS planning) as an input to fluxion's future RL gym.
**Status:** evaluation complete — research/spike only; no engine code, no gym code, no test changes, no issue filed.
**Audience:** the future RL-gym owner; the maintainers of the evaluation protocol that will live in the gym's benchmark harness.
**Date:** 2026-10-24
**Verdict:** **adopt-partially** — adopt the paper's *evaluation protocol* (action diversity alongside energy/comfort, plus a planning-budget ablation); **reject** the "model-based agent in the reference suite from day one" proposal until fluxion has a gym, a stepping API and a cost model.
**Sources:** the paper read in full (DOI 10.3390/app16189131; `mdpi-res.com/d_attachment/applsci/applsci-16-09131/article_deploy/applsci-16-09131.pdf`) and this tree at `develop` (commit `7eaec380`).

---

## 0. Scope, and what this document is not

Issue #4237 is explicitly a research/spike issue: *"Research/spike issue only. No engine changes
until the RL gym exists."* This document therefore changes nothing but itself. It answers three
questions:

1. What did the LDLP paper actually do and show, with every number traceable to a section or
   table of the paper (§1)?
2. What is fluxion's RL/gym surface *today*, verified by reading the tree (§2)?
3. What should fluxion do about the two design inputs the issue proposes (§3), and what would a
   successor prototype need (§4, deferred)?

**Journal-name correction (contradiction with the issue text).** The issue cites the paper as
*Energies* 2026. The paper is **Applied Sciences** 2026, 16, 9131 (`Appl. Sci. 2026, 16, 9131`
in the running head of every page; DOI `10.3390/app16189131`). The URL in the issue
(`mdpi.com/2076-3417/16/18/9131`) is correct — ISSN 2076-3417 is *Applied Sciences*; *Energies*
is ISSN 1996-1073. Everything below cites the paper as Applied Sciences.

---

## 1. What the paper did

### 1.1 Testbed and protocol (§3.1, §5)

| Element | Value | Citation |
|---|---|---|
| Physics engine | EnergyPlus, via Sinergym | §3.1 |
| Building | Sinergym **5Zone** — single-storey commercial building split into east / west / south / north / core zones, each with its own facade exposure, glazing fraction and occupancy schedule; thermal coupling through shared walls and air mixing | §3.1 |
| Climates | Three, one weather file per experiment: hot, mixed, cool (Sinergym injects Ornstein–Uhlenbeck noise into the outdoor temperature series, so evaluation episodes differ from one another) | §3.1, §5 |
| Episode | One calendar year at 15-min control resolution = **35,039 timesteps** | §3.1, §3.2 |
| Action | 2-D continuous: cooling setpoint in [22.5, 30.0] °C, heating setpoint in [15.0, 22.5] °C, applied uniformly to every zone; the agent emits normalized [−1, 1] and a wrapper rescales | §3.1 |
| Observation | 6-D cosine–sine encoding of month/day/hour; dry-bulb, RH, wind speed and direction, diffuse and direct solar; per-zone cooling/heating setpoints, zone air and mean radiant temperatures, PPD comfort index, occupant count, total HVAC power demand | §3.1 |
| Reward (1), "standard" | `r_t^std = −W·λ_E·E_t − (1−W)·λ_T·T_t`, `W = 0.5`, `λ_E = 1e−4`, `λ_T = 1.0`; comfort band [23, 26] °C in summer, [20, 23.5] °C otherwise | §3.1, Equation (1) |
| Reward (2), "gated" | The comfort term is applied only when the occupant count `N_t > 0`; when `N_t = 0` the reward is a pure energy penalty | §3.1, Equation (2) |
| Metrics | Energy = annual sum of penalized HVAC electricity demand (units of 10⁷ W summed over control steps); Comfort Ratio = fraction of **occupied** steps (`N_t > 0`) inside the band; "Power @ 100% Comfort" = Σ E_t / Comfort Ratio, computed per episode then averaged | §5, Equation (3) |
| Model-free baselines | PPO, SAC, TD3 from Stable-Baselines3; TD3 differs from the other two in learning rate, discount and *no exploration noise* | §4, Table 2 |
| Model-based baseline | **Sampled EfficientZero** from LightZero (not the authors' code), run at LDLP's budget: 4 parallel collectors, 100 simulations/decision, 7 sampled root actions, batch 4096, 30,000 network updates | §5 |

### 1.2 The method (§3.2, §4, Table 1)

LDLP = **Latent Dynamics Learning and Planning**:

- A **representation function** `h_θ` maps the raw observation to a **64-dimensional latent state**
  (MLP with ELU activations; Table 1), min–max rescaled to [0, 1] at every forward pass so that
  many sequential applications during search stay numerically stable.
- A **dynamics function** `g_θ` maps (latent state, action) → (next latent state, immediate
  reward) without calling EnergyPlus. The reward head has parameters separate from the state
  transition head, because predicting a scalar reward and a high-dimensional latent transition are
  structurally different tasks and sharing parameters induces gradient interference (§4).
- A **prediction function** produces a value estimate (substituting for terminal rollouts at
  leaves) and a Gaussian policy distribution. The log standard deviation is clamped to
  (−20, 0) so the sampling distribution can neither collapse (premature convergence) nor blow up
  (wide but shallow trees) (§4, Table 1).
- **MCTS inside the latent model** (four stages: selection by UCB with `c₁ = 1.25`, `c₂ = 19,652`;
  expansion; rollout in latent space; backup). Continuous setpoints are handled by **progressive
  widening** with exponent `α = 0.4` plus **sampled-action expansion** (children drawn from the
  Gaussian policy), so the tree grows toward the directions the policy likes (§3.2).
- `α = 0.4` is the reference implementation's default, **not a value tuned for this problem**;
  with 100 simulations the widening rule admits **seven** candidate setpoint pairs at the root,
  i.e. each control decision evaluates seven candidates (§3.2, §5.4 Table 9 caption).
- Training (Table 1): 4 self-play workers, batch 4096, replay buffer 300 episodes, 30,000
  training steps, unroll 4, TD 4, discount 0.99, LR 1e−3 with ×0.9 decay per 10,000 steps,
  weight decay 1e−4, observation stacking 4.
- The three networks are trained jointly under one combined loss: cross-entropy against the
  categorical encoding of the observed reward (rewards/values on a discrete support, not scalar
  regression), cross-entropy against periodically refreshed bootstrapped n-step returns, and a
  policy term (§4).

### 1.3 Headline result: energy at a common comfort level (Table 3, standard reward)

Raw energy alone is not comparable across controllers because comfort differs; the paper's
primary column is **Power @ 100% Comfort** (Equation (3)). Table 3 (standard reward,
Eq. (1)) — energy in 10⁷ W (mean ± sample s.d. over evaluation episodes):

| Climate | Controller | Energy (10⁷ W) | Comfort Ratio | Power @ 100% Comfort (10⁷ W) |
|---|---|---|---|---|
| Hot | **LDLP** | 8.408 ± 0.039 | 0.8017 | **10.489** |
| Hot | PPO | 9.793 ± 0.046 | 0.7757 | 12.624 |
| Hot | SAC | 8.832 ± 0.979 | 0.8795 | 12.165 |
| Hot | Sampled EfficientZero | 7.034 ± 0.022 | 0.3154 | 22.305 |
| Hot | TD3 (grey — degenerate) | 7.703 ± 0.046 | 0.7450 | 10.339 |
| Mixed | **LDLP** | 7.441 ± 0.105 | 0.7226 | **10.299** |
| Mixed | PPO | 10.454 ± 0.059 | 0.8676 | 12.050 |
| Mixed | SAC | 8.732 ± 0.958 | 0.8521 | 11.228 |
| Mixed | Sampled EfficientZero | 7.783 ± 0.062 | 0.3644 | 21.358 |
| Mixed | TD3 (grey) | 4.456 ± 0.052 | 0.8303 | 5.367 |
| Cool | **LDLP** | 6.129 ± 0.054 | 0.7815 | **7.843** |
| Cool | PPO | 9.039 ± 0.049 | 0.8477 | 10.663 |
| Cool | SAC | 7.675 ± 0.104 | 0.9407 | 8.159 |
| Cool | Sampled EfficientZero | 6.005 ± 0.052 | 0.3524 | 17.044 |
| Cool | TD3 (grey) | 10.757 ± 0.060 | 0.7526 | 14.293 |

Derived from those cells, LDLP's normalized energy is **3.9 % below SAC and 16.9 % below PPO**
(hot), **8.3 % / 14.5 %** (mixed) and **3.9 % / 26.4 %** (cool). The paper's own abstract
summarizes this as *"LDLP consumes 4 % to 26 % less than PPO and SAC under the standard reward"*
(Abstract). Note what the table also shows: Sampled EfficientZero has the **lowest raw energy**
in the hot and cool climates (7.034 and 6.005 × 10⁷ W) while holding a comfort ratio of only
0.3154 / 0.3524 — it is last on the normalized column in every cell (§5, §6).

### 1.4 Under the occupancy-gated reward (Table 4)

Table 4 repeats the same five controllers under Equation (2): hot LDLP 6.607 ± 0.039 energy,
0.6604 comfort, **10.005** normalized; PPO 7.852 ± 0.044, 0.7745, 10.137; SAC 6.978 ± 0.543,
0.8568, 10.349; Sampled EfficientZero 7.041 ± 0.019, 0.3169, 22.225; TD3 7.407 ± 0.038, 0.7644,
9.691. Mixed: LDLP 5.542 ± 0.096, 0.6159, **8.999**; PPO 6.124 ± 0.301, 0.8063, 10.087;
SAC 4.948 ± 0.114, 0.7791, 9.818; Sampled EfficientZero 7.793 ± 0.070, 0.3646, 21.378;
TD3 4.363 ± 0.053, 0.8579, 5.086. Cool: LDLP 2.537 ± 0.043, 0.7121, **3.563**; PPO 3.542 ±
0.060, 0.8590, 4.123; SAC 2.483 ± 0.071, 0.8621, 4.405; Sampled EfficientZero 6.008 ± 0.054,
0.3529, 17.025; TD3 2.576 ± 0.042, 0.7369, 3.496.

The ordering is unchanged but the margins narrow: LDLP leads PPO by **1.3 %** (hot — the thinnest
margin anywhere), 10.8 % (mixed) and 13.6 % (cool), against **1 % to 19 %** across PPO and SAC
in the abstract. The gate costs LDLP comfort: 0.8017 / 0.7226 / 0.7815 → 0.6604 / 0.6159 /
0.7121 (§5, §6). The gate is not an LDLP-specific trick: unoccupied demand falls 44 % / 42 % / 83 %
for LDLP, 46 % / 57 % / 76 % for PPO and 50 % / 65 % / 84 % for SAC (§5.1, Table 5).

### 1.5 Action diversity (§5.2, Table 6, Table 7)

The paper reports diversity **alongside** the aggregate metrics, because aggregate metrics
cannot distinguish a controller that regulates from one that does not. Table 6 (occupancy-gated
reward): distinct setpoint pairs (counted after rounding each setpoint to two decimals), the
share held by the four most frequent pairs, and action entropy in bits:

| Climate | Controller | Distinct setpoints | Top-4 share | Entropy (bits) |
|---|---|---|---|---|
| Hot | LDLP | 30,713 | 0.0008 | 14.83 |
| Hot | PPO | 11,548 | 0.2000 | 11.06 |
| Hot | SAC | 26,976 | 0.0027 | 14.47 |
| Hot | Sampled EfficientZero | 33,871 | 0.0004 | 15.03 |
| Hot | TD3 | **4** | **1.0000** | 1.00 |
| Mixed | LDLP | 26,930 | 0.0023 | 14.49 |
| Mixed | PPO | 10,800 | 0.2846 | 9.71 |
| Mixed | SAC | 26,150 | 0.0013 | 14.44 |
| Mixed | Sampled EfficientZero | 33,860 | 0.0004 | 15.03 |
| Mixed | TD3 | **13** | 0.9997 | 1.69 |
| Cool | LDLP | 18,456 | 0.0031 | 13.71 |
| Cool | PPO | 15,073 | 0.2986 | 10.43 |
| Cool | SAC | 26,405 | 0.0012 | 14.47 |
| Cool | Sampled EfficientZero | 33,871 | 0.0004 | 15.03 |
| Cool | TD3 | **1** | 1.0000 | 0.00 |

The degenerate case is TD3: **4, 13 and 1** distinct setpoint pairs in the hot, mixed and cool
climates, with its four most frequent pairs covering at least **99.95 %** of all control steps
(§5.2). A five-seed sweep (Table 7) shows the collapse is not a single-run artifact: under the
gated reward **all fifteen** seed-climate cells degenerate (1–60 distinct pairs, top-4 share ≥
0.998); under the standard reward thirteen of fifteen do, the exceptions being hot/seed 0
(5,986.4 pairs) and mixed/seed 32 (14,312.8 pairs). Aligning TD3's learning rate, discount and
exploration noise with the other baselines **removes the degeneration entirely** (§5.5).

Diversity is necessary but not sufficient: Sampled EfficientZero has the *highest* entropy
(15.03 bits in every climate) because it scatters setpoints almost uniformly over the action
rectangle, yet it has the worst normalized energy in every cell (Tables 3 and 4) and a comfort
ratio of 0.32–0.36 (§5.2, §6). The paper's framing: the diversity measure separates "repeats one
setpoint" from "regulates"; the energy and comfort columns separate "regulates" from "scatters
at random".

### 1.6 Planning-budget ablation (§5.4, Table 9)

The same **trained** model is re-evaluated with only the number of MCTS simulations per decision
changed — no retraining — in the mixed climate under the gated reward:

| Simulations / decision | Energy (10⁷ W) | Comfort Ratio | Power @ 100% Comfort (10⁷ W) | Distinct setpoints | Entropy (bits) |
|---|---|---|---|---|---|
| 1 (no planning) | 7.132 | 0.1504 | **47.428** | 30,989 | 14.85 |
| 25 | 5.530 | 0.5525 | 10.008 | 27,337 | 14.53 |
| 50 | 5.513 | 0.5873 | 9.387 | 27,077 | 14.51 |
| 100 (default) | 5.584 | 0.6142 | 9.094 | 26,942 | 14.49 |

At one simulation the tree never grows and the controller acts on the policy network alone:
comfort collapses to 0.1504 and normalized energy rises from 9.094 to **47.428 × 10⁷ W** — worse
than any controller in Table 4, including Sampled EfficientZero at 21.378 (§5.4). That is the
abstract's *"one simulation per decision, which removes planning, multiplies its normalized
energy fivefold"* (47.428 / 9.094 ≈ 5.2). Because the widening rule admits 1, 4, 5 and 7 root
candidates at 1, 25, 50 and 100 simulations, the same sweep is also a widening ablation: the gain
arrives with the first few candidates and then flattens (§5.4). Note the direction of the effect:
the search buys **comfort at essentially the same energy** (5.513–7.132 × 10⁷ W across the
sweep) — and the number of *distinct* setpoints **falls** slightly as the budget grows, so the
search is selecting better among candidates, not producing more of them (§5.4).

### 1.7 The Pareto reading (§5.3, Table 8)

Dominance (no energy/comfort trade-off assumption) is used to qualify the normalized column:
under the standard reward LDLP is non-dominated in hot and cool but **dominated by TD3 in the
mixed** climate; under the gated reward LDLP is non-dominated in hot (6.607 vs SAC's 6.978 × 10⁷ W,
but SAC holds the higher comfort ratio), **dominated by SAC and TD3 in the mixed** climate and
**dominated by SAC in the cool** climate (Table 8). Sampled EfficientZero is dominated in every
climate under the gated reward (Table 8). The paper is explicit that dominance alone is not
sufficient either — it registers that nobody beats EfficientZero on both axes, "but not that the
comfort ratio it defends is only 0.32 to 0.36" (§5.3).

### 1.8 What the paper does *not* establish

- **One building.** "the experiments use only the 5Zone building" (§7).
- **Simulation only.** No hardware, no commissioning; actuator delay/deadband hysteresis, sensor
  noise/drift/dropout, EnergyPlus-to-building structural mismatch and occupant disturbances are
  each listed as ways the learned model and the search would fail (§7).
- **Compute cost.** "100 simulations per decision may exceed the budget of an embedded
  building-management controller" (§7).
- **Occupancy signal granularity.** `N_t` is read from a single zone while setpoints apply to all
  five, so "the building is empty" is an approximation (§7).
- **No cross-climate generalization.** Each climate is trained as a separate problem; whether one
  model generalizes across climates is open (§7).
- **Reproducibility.** No public code artifact: "the implementation, reward definitions, and
  per-timestep evaluation logs are available from the authors on reasonable request" (Data
  Availability Statement). Hyperparameters are fully tabulated (Tables 1 and 2), but the code,
  the trained models and the raw per-timestep logs are not.
- **Single policy per cell.** "A single policy per algorithm-climate-reward cell is trained"
  (§5); only TD3 gets a five-seed sweep (Table 7).

---

## 2. Verified current state of fluxion's RL/gym surface

Everything below was checked by reading this tree at `develop` (`7eaec380`). **Fluxion does not
have an RL gym, and has no RL code at all.**

### 2.1 The only gym artifact in fluxion's history was removed

| Fact | Evidence |
|---|---|
| A Gymnasium environment wrapper (`FluxionEnv`) was developed under issue #328 | `tools/tests/test_gymnasium_env.py:5` ("Issue: GitHub #328 - Develop Gymnasium Environment Wrapper (FluxionEnv)"); original commits `1b95645b` / `ba1e4472` ("feat(python): Develop Gymnasium environment wrapper for Fluxion (#328)") added `tools/gymnasium_env.py` + `tools/tests/test_gymnasium_env.py` |
| The wrapper was archived, then deleted | `a58f5895` ("docs: reorganize project documentation into archives…") moved `tools/gymnasium_env.py` → `tools/archive/gymnasium_env.py`; `9de56d75` (2026-04-02, "chore(cleanup): remove unused diagnostic scripts, test artifacts, and session histories") deleted it |
| **The module is not in the tree today** | `git ls-files` returns no `tools/gymnasium_env.py` and no `tools/archive/` entry; only the test file survives (`tools/tests/test_gymnasium_env.py`) |
| The surviving test imports a module that no longer exists | `tools/tests/test_gymnasium_env.py:15` — `from tools.gymnasium_env import GYMNASIUM_AVAILABLE, EnvConfig, FluxionEnv`; the Gymnasium-API assertions live at `:196-235` (`gym.make("Fluxion-v0", config=config)`) |
| The removed wrapper's design (for reference) | Its docstring specified state = indoor temperatures + weather forecast + time-of-use electricity prices + time of day; action = continuous HVAC setpoints (+ discrete equipment toggles); reward = energy-cost penalty + PMV comfort penalty (`git show 1b95645b:tools/gymnasium_env.py:1-25`) |

**Consequence for this issue:** fluxion's "gym" history is a *removed prototype*, not a surface
to extend. Nothing in the shipped documentation mentions a gym or RL at all — `grep -rn -i "gym"`
over `docs/**/*.md`, `README.md`, `AGENTS.md`, `CHANGELOG.md`, `CODEBASE_MAP.md` and
`docs/KNOWN_ISSUES.md` returns **no hits** outside this document. The post-1.0 roadmap in
`docs/RELEASE_NOTES_v1.2.md:451-471` (v1.3: thermal damping, ESP-r integration, validation
dashboard, cloud validation; v2.0: finite-volume solver, parallel zones, visualization,
validation API) contains **no RL or gym item**.

### 2.2 There is no RL code

`grep -rn -i "reinforcement|\bPPO\b|\bSAC\b|TD3|MCTS|monte carlo tree|policy gradient|Q-learning"`
over all `*.rs`, `*.py`, `*.ts`, `*.js`, `*.toml` and `*.yml` in the tree returns exactly **one**
hit, and it is unrelated: a comment in `tests/all_tests/conduction_5r1c_isolation.rs:1169`
("Bit-level reinforcement"). There is no policy-gradient library, no value-network training
loop, no tree search, and no learning-to-control entry point in the engine.

### 2.3 What a gym would have to sit on (the surfaces that do exist)

| Capability a gym needs | What fluxion has today | Evidence |
|---|---|---|
| Resettable, single-timestep stepping of a whole building | **Absent.** The engine runs a whole simulation in one call; the only `step()` functions are sub-component (grid adapter, airside coupling, DHW, sensor node, multi-node HVAC runner), not a building-level RL step | `fluxion.pyi:112-114` (`simulate`, `simulate_with_loads`, `simulate_numpy`); `src/sim/grid_adapter.rs:155`, `src/sim/hvac/airside_coupling.rs:119`, `src/sim/sensor_node.rs:195` |
| Per-timestep sensors + actuators (setpoint overrides) | **Present but "intentionally wired-but-dead"** — the EMS module is the planned closed-loop surface, parked until the simulation loop consumes it | `src/sim/ems.rs:1-13` ("Intentionally wired-but-dead: this module is the planned EMS integration surface … and is parked until the simulation loop consumes it. Tracked in `tests/reference_data/wired_but_dead_dispositions.json`"); sensors/actuators at `src/sim/ems.rs:59-91` (`EmsSensorType::ZoneTemperature`, `EmsActuatorType::HeatingSetpoint`, `…CoolingSetpoint`); per-timestep program hook at `src/sim/ems.rs:175-196` |
| Runtime setpoint actuation | Present at the HVAC-system level (not wired to the timestep loop) | `src/python/hvac_bindings.rs:49-56` (`set_heating_setpoint`, `set_cooling_setpoint`); `src/cli/hvac_commands.rs:186-191`; `src/napi/hvac_config.rs:534` |
| Setpoints supplied at simulate time | Present (one-shot, per-zone scalars) | `src/api/server/simulate.rs:228-229`, `:402-405` (`model.setpoints.heating_setpoint` / `cooling_setpoint` / per-zone vectors) |
| REST/streaming simulation | Present | `src/api/server/router.rs:163-164` (`POST /v1/simulate`, `POST /v1/simulate/stream`), `:181-182` |
| Energy accounting per run | Present: heating / cooling / latent-cooling / electrical kWh, and per-zone heating/cooling kWh | `src/sim/thermal_model_core/mod.rs:458-496` |
| Occupancy signal (needed for the paper's gate) | Present as a schedule type; occupancy-driven comfort accounting exists | `src/api/schema.rs:492` (`pub occupancy: DailySchedule`), `:979-996` (occupancy-gated unmet-hours); `src/sim/occupancy.rs` (`BuildingType` schedules) |
| Weather sources | Present (EPW, TMY2, TMY3, design-day, psychrometrics) | `fluxion-core/src/weather/mod.rs:20-31` (`denver`, `epw`, `interpolation`, `miami`, `minneapolis`, `psychrometrics`, `tm2`) |
| A learned surrogate of building response | Present, but it is a **load/parameter surrogate**, not a latent dynamics model | `src/ai/surrogate/manager.rs:563` (`SurrogateManager`, ONNX Runtime); `src/ai/modular_surrogate.rs`; `docs/ONNX_INFERENCE_PIPELINE.md:5` |
| A state estimator (not a planner) | Present: unscented Kalman filter | `src/ai/ukf.rs:16-30` |
| Search/optimization tooling (not RL) | Present: PSO, genetic algorithm, OpenEvolve adapter — parameter tuning, not sequential decision making | `tools/optimization.py:44` (`ParticleSwarmOptimizer`), `:309` (`GeneticAlgorithmOptimizer`); `tools/evolution/openevolve_adapter.py`; `tools/multi_case_optimization.py` |
| ML toolchain available to a prototype | Present (torch, onnx, onnxruntime, numpy) — but as **dev** dependencies, not runtime dependencies | `requirements-dev.txt:11-17`; `pyproject.toml:33-35` (runtime deps: numpy, onnxruntime, pandas) |

### 2.4 Assessment of fluxion's position

- fluxion has a **physics core** that is a plausible gym substrate: multi-zone thermal model,
  HVAC dispatch, weather, occupancy schedules, energy meters.
- fluxion has **no control-loop API**: no per-timestep building `step()`, no observation
  assembly, no reward, no episode control. The closest thing is `src/sim/ems.rs`, which is
  *by its own disposition note* parked until the simulation loop consumes it — i.e. the exact
  seam a gym would need, already identified and deliberately left unbuilt.
- fluxion has **no RL reference suite**, so there is no agent to compare a latent-dynamics agent
  against, and no benchmark harness in which an "evaluation protocol" could be enforced today.

**Therefore: the gym does not exist, and this issue's design inputs cannot be executed against
fluxion today.** The paper can still inform *how* the gym should be specified, because the
evaluation protocol is a property of the benchmark harness, not of the engine.

---

## 3. Verdict on the two design inputs

### 3.1 Input 1 — "include a model-based (latent dynamics + search) agent in the gym's reference agent suite from day one"

### **REJECT (for now), with a concrete re-entry condition.**

Reasons:

1. **There is no gym, no stepping API and no reference suite to join** (§2). "From day one"
   cannot be scheduled against a surface that does not exist; the first day of the gym has to
   spend its budget on `reset`/`step`/observation/reward semantics, not on a search agent.
2. **The paper's own evidence does not support a model-based agent as the *default* reference
   agent.** The honest reading of the paper is narrower than the headline:
   - The advantage is **relative to model-free baselines on a normalized metric in one
     building, and it is not uniform**: 3.9 %–26.4 % (standard reward, §1.3) and 1.3 %–13.6 %
     against PPO under the gated reward (§1.4).
   - LDLP is **dominated by SAC and TD3 in the mixed climate under the gated reward** and
     **dominated by TD3 in the mixed climate under the standard reward** (Table 8) — the
     Pareto reading is the paper's own preferred lens (§5.3).
   - The independent model-based baseline, **Sampled EfficientZero, transfers badly**: comfort
     0.3154–0.3646, worst normalized energy in every cell of Tables 3 and 4, dominated in every
     climate under the gated reward (§6). The paper's conclusion is that "a model-based
     architecture is not by itself sufficient on this problem" (§6). A model-based agent in
     fluxion's reference suite from day one would encode the *category*, which this paper's own
     strongest model-based arm contradicts.
3. **The cost profile does not fit fluxion's product.** 100 simulations per decision with a
   64-dim latent model is an inference-time cost the paper itself flags as possibly exceeding an
   embedded controller's budget (§7), and fluxion's engine is a physics engine, not an
   inference cluster.
4. **The comparison the paper makes is not the comparison fluxion would need.** The paper
   compares controllers *inside one simulator* (EnergyPlus via Sinergym). fluxion's value
   proposition is being the simulator. A fluxion reference suite would compare agents *on
   fluxion*, which reverses the roles: the paper's absolute numbers (10⁷ W per year on 5Zone)
   are not transferable to fluxion's models, and the paper's evidence says so itself
   ("the buildings, objectives and baselines differ too much for a like-for-like comparison",
   §6).

**Re-entry condition (state it in the gym's own design doc when the gym is specced):** the
model-based agent joins the reference suite **as an optional, clearly-labelled agent** once
(a) the gym has a documented `step()` with a fixed control interval, (b) a learned-latent
dynamics model can be trained against fluxion's own simulator, and (c) the harness can report
the planning budget as a first-class parameter — i.e. exactly the three things §5.4 shows are
what the result depends on.

### 3.2 Input 2 — "adopt their evaluation protocol: action-diversity alongside energy/cost, and planning-budget ablation"

### **ADOPT.**

This is the paper's most transferable contribution and it is *independent of the model-based
agent*: it is a property of any benchmark harness.

**(a) Action diversity alongside energy/comfort — adopt.**

Evidence for adopting it is strong and specific: TD3 records the **lowest** normalized energy in
five of the six cells of Tables 3 and 4 and is **non-dominated in the mixed climate under the
gated reward** — while issuing **1 to 13 distinct setpoint pairs for an entire year**, with its
four most frequent pairs covering ≥ 99.95 % of control steps (§5.2, Table 6). Under a
two-axis aggregate metric it is indistinguishable from a good controller; only a diversity
metric exposes it. The paper generalizes this in §5.3: *"A controller that stops responding to
the weather can therefore look as good on these two axes as one that regulates."*

What fluxion should adopt, concretely:
- report **distinct-action count** and **action-entropy (bits)** next to energy and comfort for
  every agent in the gym's benchmark harness;
- report a **top-k share** (the paper uses top-4) so a controller that repeats a handful of
  setpoints is visible without reading an entropy number;
- treat "energy + comfort" as a **Pareto pair**, not a single score, and publish the dominance
  relation per cell (the paper's Table 8) rather than only a normalized number;
- keep the **normalization caveat visible**: the paper states plainly that Power @ 100% Comfort
  "is a first-order device rather than a principled efficiency measure: it assumes energy trades
  against comfort linearly and at a constant rate, and it over-corrects as the comfort ratio
  falls" (§5, before Equation (3)). fluxion's harness must ship the raw pair next to any
  normalized figure.

**(b) Planning-budget ablation — adopt, as a harness capability.**

The paper's ablation (§5.4, Table 9) is methodologically the strongest thing it does: the
**same trained model**, only the search budget changed, no retraining, so the difference is
attributable to the search alone. Removing planning (1 simulation per decision) took comfort
from 0.6142 to 0.1504 and normalized energy from 9.094 to 47.428 × 10⁷ W — a fivefold change
from one knob.

What fluxion should adopt, concretely:
- the gym's agent interface must expose a **budget parameter** (simulations per decision for a
  search agent; gradient/rollout budget for a model-free agent) so the harness can sweep it;
- the harness must support **search-off** runs (the "1 simulation" row) as a first-class
  configuration, because that is the row that shows what the architecture contributes;
- any fluxion reference agent that uses any form of search or lookahead must ship **one
  budget-sweep row** in its own results — the paper's conclusion ("what the controller gains
  comes from evaluating several candidate setpoints per decision rather than from the
  architecture that surrounds the search", §5.4) is a claim every search-based agent in the
  suite has to earn.

**(c) A third protocol element worth adopting: multi-seed reporting where a baseline is
suspect.** The paper's TD3 result is only interpretable because of the five-seed sweep
(Table 7) that shows the collapse is systematic (15/15 cells under the gated reward) and that
aligning the baseline's hyperparameters removes it entirely (§5.5). The paper's own rule —
flag the anomaly, do not quietly replace the row — is the right one for a benchmark harness.

**(d) What *not* to adopt from the paper's protocol:** the occupancy-gated reward (Equation (2))
as a *default*. It is a legitimate objective, and it is well-motivated (§3.1), but it changes
the task: the gate lowers LDLP's comfort ratio from 0.8017 to 0.6604 in the hot climate (§5).
A benchmark harness should fix **one** objective and report agents against it; objective
variants belong in the ablation section, exactly as the paper does in §5.5.

### 3.3 Verdict summary

| # | Design input from #4237 | Verdict |
|---|---|---|
| 1 | Model-based (latent dynamics + search) agent in the gym's reference agent suite from day one | **Reject for now** — no gym exists; the paper's own model-based arm (Sampled EfficientZero) fails to transfer (§6), and the LDLP advantage is non-uniform and Pareto-inconsistent across cells (Tables 3, 4, 8). Re-enter as an *optional* agent once a stepping API, a trainable latent model and a budget-aware harness exist. |
| 2 | Adopt the evaluation protocol: action diversity alongside energy/cost, and planning-budget ablation | **Adopt** — transferable to any harness, independent of the agent category, and directly motivated by the paper's TD3 result (Table 6) and §5.4 / Table 9. With two refinements: report the (energy, comfort) Pareto relation and keep the normalization caveat visible; and adopt the objective *variants* as ablations, not as the default objective. |

---

## 4. Concrete successor scope — a prototype LDLP-style agent against fluxion's gym

> **Listed here only — no issue was filed, no code was written, nothing was scheduled.**
> Every item below is **deferred until the RL gym exists** (i.e. until `reset`/`step`,
> observation assembly, reward, and an episode loop are specified and implemented). Nothing in
> this list may be started against the current tree, because the required surfaces do not exist
> (§2).

| # | Item | Why it is needed / what it depends on |
|---|---|---|
| P1 | **Gym substrate, not an agent.** Specify `reset()` / `step(action)` / `observation_space` / `action_space` / `reward` over fluxion's multi-zone model, with a fixed control interval and an explicit occupancy signal | The paper's controller is defined by a 15-min control interval over a 5-zone building with per-zone setpoint actions (§3.1); fluxion has no building-level `step()` today (§2.3) and `src/sim/ems.rs:1-13` is the parked seam where the per-timestep actuator override would live |
| P2 | **Observation assembly matching the action space.** fluxion must expose zone air and mean radiant temperatures, per-zone setpoints, PPD/PMV-comparable comfort index, occupant count, HVAC power, and the weather vector; plus the trig time encoding the paper uses | The paper's observation set is what its model and reward are built around (§3.1); fluxion's comfort reporting is currently unmet-hours based (`src/api/schema.rs:979-996`), not a PPD/PMV index |
| P3 | **Action semantics: per-zone or uniform setpoint pairs.** The paper applies one setpoint pair to all five zones; fluxion's multi-zone model supports per-zone setpoints (`src/api/server/simulate.rs:402-405`) | Decide deliberately; the paper's "one pair for all zones" is a simplification that a multi-zone fluxion gym should expose as a config, not bake in |
| P4 | **Energy + comfort + diversity + Pareto reporting in the harness** (the adopted protocol of §3.2) | Needs P1's observation/action/reward contract; the diversity metric needs the action history the harness records per episode |
| P5 | **Budget-aware agent interface and a budget sweep** (planning-off / low / default / high) | §5.4 / Table 9 show the result lives in this knob; the harness must be able to run one agent at several budgets without retraining |
| P6 | **A latent-dynamics world model trained on fluxion's own simulator** (representation / dynamics / prediction heads, 64-dim latent, discrete reward support) | The paper's model is trained purely from simulator interaction (§4, Table 1); fluxion already has an ONNX surrogate path (`src/ai/surrogate/manager.rs:563`) and torch/onnx in the dev toolchain (`requirements-dev.txt:11-17`) that could host this — but a *dynamics* model is a different artifact from a *load* surrogate |
| P7 | **A search agent over that latent model** (MCTS with progressive widening and sampled-action expansion) or, cheaper, a one-step learned-policy lookahead | §3.2 specifies the design; P6 is its prerequisite |
| P8 | **An occupancy-gated reward as an *ablation variant*, not the default** | §5 / §5.5 (Table 10) — the gate is a genuine objective change and belongs in the ablation table |
| P9 | **Multi-seed baseline reporting** for any agent whose policy collapses | Table 7 and §5.5 — the mechanism that caught TD3 |
| P10 | **A clean-up of the stale Gymnasium test** `tools/tests/test_gymnasium_env.py` (imports a module deleted in `9de56d75`) — decide: restore a gym under `tools/`, or delete the orphan test | Not part of #4237 (no engine/test changes allowed here); it is a real dangling reference in the tree that any future gym work will collide with |

---

## 5. Risks and limits of the paper's evidence, as they bear on the verdict

| Risk | Evidence in the paper | Bearing on the verdict |
|---|---|---|
| **Single simulator / single building** | Only the Sinergym 5Zone building, EnergyPlus as the engine (§3.1, §7). No other Sinergym model tested | Caps the external validity of the 4–26 % headline. Does **not** bear on Input 2, which is a statement about how to *measure*, not about which controller wins. Reinforces the rejection of Input 1 (no basis for picking a model-based agent as a fluxion reference agent) |
| **Single climate file per experiment** | One weather file per climate; OU noise gives episode-to-episode variation but the climate is fixed within a cell (§3.1, §5) | The climate set (hot/mixed/cool) is a coarse axis; the paper itself flags cross-climate generalization as open (§7) |
| **Normalization is a first-order device** | "assumes energy trades against comfort linearly and at a constant rate, and it over-corrects as the comfort ratio falls" (§5, before Eq. (3)) | The headline "% less" numbers live on that normalized column. A gym must publish raw energy, comfort and the Pareto relation together — this is why §3.2(a) requires all three |
| **Baseline tuning / asymmetry** | TD3's configuration differs from PPO/SAC in learning rate, discount and absence of exploration noise (Table 2); its degeneracy disappears when those are aligned (§5.5); Sampled EfficientZero gets LDLP's exact budget but its search has no progressive widening and its latent state has no reward-prediction weighting (§6) | The paper is honest about this and reports it rather than hiding it — which strengthens the case that **baseline alignment is a protocol requirement**, and weakens any naive "LDLP > PPO/SAC" reading. Also means the *absolute* margins (4–26 %) are lower bounds in the paper's favour only for the configurations tested |
| **Single policy per cell** | "A single policy per algorithm-climate-reward cell is trained" (§5); only TD3 gets a seed sweep (Table 7) | The reported dispersion (± s.d. over *evaluation* episodes) is not a training-seed variance. Any fluxion harness must distinguish episode variance from seed variance in its own outputs |
| **Reproducibility** | No public code, models or logs; "available from the authors on reasonable request" (Data Availability Statement). Hyperparameters are tabulated (Tables 1, 2) | The *protocol* (what to measure, how to ablate) is reproducible from the paper; the *numbers* are not independently checkable without the authors. This is precisely why Input 2 is adoptable and Input 1 is not: a protocol transfers, a headline number does not |
| **Compute profile** | 100 simulations/decision, 30,000 network updates, 4 collectors, batch 4096 (Table 1); "may exceed the budget of an embedded building-management controller" (§7) | The fluxion gym must treat per-decision inference cost as a reported metric, not a hidden constant. The budget sweep (§5.4) is the mechanism |
| **Sim-to-real gap** | Actuator delay/hysteresis, sensor noise/drift/dropout, structural mismatch, occupant disturbances (§7) | Not a blocker for a simulation benchmark, but it means a fluxion gym's results describe *the simulator*, and the doc must say so |
| **Occupancy-signal approximation** | `N_t` from a single zone, setpoints applied to all five (§7) | Directly relevant to fluxion: fluxion's occupancy is a schedule (`src/api/schema.rs:492`), so any gated-reward ablation in a fluxion gym inherits the same approximation and must document it |
| **Dominance is not decisive either** | Dominance "registers that no controller beats Sampled EfficientZero on both axes at once, but not that the comfort ratio it defends is only 0.32 to 0.36" (§5.3) | Supports the protocol decision: report the Pareto relation **and** the diversity metric **and** the normalized figure; none of the three alone is sufficient |

---

## 6. First concrete step (when the gym is specced)

Write the gym's **benchmark-harness specification** with the four reporting obligations this
evaluation adopts — energy, comfort, action diversity (distinct-action count + entropy +
top-k share), and the Pareto relation — plus the budget-sweep requirement, and with **one**
fixed objective and the objective variants reserved for ablations. That specification is the
deliverable this evaluation feeds; the LDLP-style agent is a later, optional occupant of the
suite, not its first requirement.

---

*Companion research documents in this directory: `docs/research/night_ventilation_convection_audit.md`
(issue #4238), `docs/research/mazzeo-rt27-pcm-spike.md`, `docs/research/pantelides-spike.md`. A
sibling evaluation for issue #4239 lives at `docs/ashrae_140/hvac_bestest_methodology.md`.*
