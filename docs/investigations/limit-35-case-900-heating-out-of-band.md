# §LIMIT-05 companion — Case 900 annual heating out of band: reproduction, isolation, and a falsified mass-node hypothesis

**Status:** investigation record on develop. The measurement hooks described
in §5/§6 (`FLUXION_MASS_SUBSTEPS`, `FLUXION_CONDITIONED_SUBSTEPS`) live only
on the closed DO-NOT-MERGE branch `physics/limit35-case900-conditioned-subhourly`
(PR #4333); they are **not** in develop's tree. Develop's physics is unchanged
hourly stepping. §7 records the decision taken on this investigation.

**Scope note:** this is the Case 900 (high-mass) annual **heating**
over-prediction. It is pre-existing on develop `7d144c80`, unrelated to the
§LIMIT-33 fix (Case 900 is 9R4C-path ideal-loads-insensitive; variants of the
LIMIT-33 fix leave Case 900 bit-identical), and distinct from the §LIMIT-05
*peak-cooling under-prediction* symptom, though both live in the high-mass
path.

## 1. Reproduction (develop `7d144c80`, default build, Denver-Stapleton TMY EPW)

Harness: `ASHRAE140Validator::simulate_case_with_diagnostics` on
`ASHRAE140Case::Case900` (the production validator path; per
`step_dispatcher.rs` the high-mass spec is promoted selector-independently to
the 9R4C network, so `new_with_selector(NineRFourC)` returns identical
numbers to `new()`).

| Quantity | Value | Reference band |
|---|---|---|
| Annual heating | **3,477.88 kWh** | [1,170, 2,040] kWh → **+70.6% over the upper bound** |
| Annual cooling | 923.56 kWh | [2,130, 3,670] kWh → under |
| Peak heating | 3.304 kW | [1.80, 2.40] kW |

Standalone `ThermalModel` harnesses agree (3,249.6 kWh over 8,760 h with
14-day warmup via `DenverTmyWeather`); the ASHRAE140_RESULTS.md snapshot's
5,052.83 kWh is a stale/legacy build, not current develop.

Temperature trace: zone air never drops below the heating setpoint (min
exactly 20.00 °C, mean 23.32 °C) while heating energy is positive in
**3,722 of 8,760 hours** — the metered heating is the full net load of a
zone held at setpoint, and it is the *magnitude* of that net load that is
wrong.

Monthly split: Jan 934 / Feb 588 / Mar 187 / Apr 7 / Oct 85 / Nov 521 /
Dec 928 kWh — a physically plausible seasonal shape; the excess is spread
over the whole heating season, not a spike artifact.

Hourly split: nearly uniform across the day (63 kWh at 15:00 to 207 kWh at
06:00, annual totals per clock hour) — midday solar offsets only ~50–70% of
the noon-hour load, which for a high-mass single-zone building with large
south glazing is the signature of either under-captured solar gain or
over-counted envelope loss.

## 2. Decisive experiment: whole-step timestep refinement

Running the same annual loop with the physics step divided into N sub-steps
per hour (same hourly weather drivers, `step_physics(dt=3600/N)`):

| N | dt | Annual heating (kWh) | Annual cooling (kWh) |
|---|---|---|---|
| 1 | 3600 s | 3,249.6 | 1,602.4 |
| 2 | 1800 s | 4,717.9 | 555.5 |
| 4 | 900 s | **1,802.7** | 192.2 |
| 6 | 600 s | **1,305.7** | 126.9 |

At dt = 600 s the heating energy lands **inside the published band
[1,170, 2,040] kWh**, and the sequence converges toward it (the N=2
overshoot is the kind of non-monotone step seen when a stiff node crosses an
integration stability boundary). Refining the *whole* discretized hour
reproduces the reference behavior. This localizes the error to the
**hour-scale discretization of the coupled air-node/HVAC/gain system**, not
to any slow (building-mass-scale) drift.

## 3. Falsification: sub-stepping the mass node alone does NOT converge

§LIMIT-29 hypothesis 3 attributes the high-mass errors to the mass-node
integrator (dt/τ ≈ 0.81 at Case 900's τ ≈ 1.23 h on the 5R1C air
trajectory). Two targeted tests of that hypothesis, both now in the tree as
env-gated hooks:

1. **Lumped 9R4C mass block** (`step_9r4c.rs` mass-temperature backward
   Euler; hook applied temporarily, `FLUXION_MASS_SUBSTEPS=12`): validator
   Case 900 heating unchanged
   to 5 decimals (3,477.88 kWh). That block's output demonstrably does not
   drive the conditioned-air trajectory (consistent with its own comment:
   the multi-node network, not the lumped mass, sets the high-mass air
   temperature).
2. **Multi-node solver sub-step override** (`multi_node_solver/mod.rs`
   `step_with_gains`, same env var; the adaptive path there already
   sub-steps to 12 when dt > 4τ but τ for the 900 envelope exceeds dt/4, so
   it normally runs 1): N = 2/4/6/12 gives 3,484.2 / 3,487.4 / 3,488.5 /
   3,489.6 kWh — a +12 kWh drift **away** from the band.

Conclusion: **the mass-node backward Euler is not the root cause** of the
heating over-prediction. The convergence seen in §2 must come from refining
the air-node balance and the HVAC setpoint-feedback/metering loop at
sub-hour resolution (the same loop §LIMIT-33's state feedback closes at
hour resolution). The τ-based hypotheses in §LIMIT-29 remain valid for the
peak-cooling under-prediction (§LIMIT-05) but are not what pushes annual
heating out of band.

## 4. Proposed fix direction (not implemented — review first)

Refine the **conditioned-hour inner loop**: with the zone held at setpoint,
solve the air-node balance and meter the HVAC energy with sub-hour steps
(or in closed form for the piecewise-constant drivers), leaving the mass
node hour-stepped. Success criterion: Case 900 heating moves monotonically
toward [1,170, 2,040] kWh as the inner refinement increases, with Case 600
(NAPI/WD600 5,590.48 kWh in [4,314, 5,836]) and the recorded-value baselines
re-checked per the re-record discipline — never tuned to pass.

## 5. Measurement hooks in this branch

- `FLUXION_MASS_SUBSTEPS` (multi-node solver): forces N sub-steps in
  `step_with_gains`; unset/0 → adaptive logic bit-identical to develop.
- The same variable name in the 9R4C **lumped** mass block is intentionally
  not kept (zero observable effect; removed to avoid dead knobs).

Both hooks exist to make §3's numbers reproducible by anyone; neither is
wired into any gate.

## 6. Implementation (2026-10-08): conditioned-hour sub-hourly inner loop — measured outcome, DO NOT MERGE

Alex's direction (2026-10-07): implement sub-hourly timesteps for the
conditioned-hour air-node + HVAC-metering inner loop, "maybe 10 or 15
minutes". Implemented in `try_step_physics`
(`src/sim/thermal_model_physics/step_dispatcher.rs`): every hourly
(dt = 3600 s) physics step now runs as an inner loop of N = 4 equal
sub-steps (15-minute; `FLUXION_CONDITIONED_SUBSTEPS` overrides N for
investigation), with loads/solar computed once per hour and per-sub-step
HVAC metering summed to the hour. The `FLUXION_MASS_SUBSTEPS`
multi-node measurement hook from §5 is retained; the zero-effect lumped
9R4C hook was folded away as the doc suggested.

### Decisive finding: the production Case 900 path does not respond to outer-dt refinement

The §2 convergence was measured on the **standalone plain-`ThermalModel`
(5R1C, synthetic Denver) harness**. On the production validator/EPW path —
where the Case 900 HighMass spec is auto-promoted to the **9R4C** network —
refining the outer step is a near no-op:

| Path | N=1 | N=4 (15 min) | N=6 (10 min) |
|---|---|---|---|
| Case 900 spec → 9R4C (EPW, 14-day warmup, standalone) | 3,143.8 kWh | 3,147.7 kWh | 3,148.2 kWh |
| Validator Case 900 (default N=4 on this branch) | — | **3.47 MWh — unchanged, still out of band** | — |

The 9R4C air node is quasi-steady and its multi-node mass block already
sub-steps internally (τ-adaptive), so outer-dt refinement cannot move it.
Consistent with §3: the discretization error that puts Case 900 heating out
of band is not reachable through the timestep at all on this path.

### What the loop does move: the 5R1C path — Case 600 regresses

| Quantity (NAPI/WD600 harness) | develop (hourly) | this branch (N=4) | band |
|---|---|---|---|
| Case 600 heating | 5,590.48 kWh | **4,450.67 kWh** (in) | [4,314, 5,836] |
| Case 600 cooling | 4,675.48 kWh | **9,777.69 kWh (out, +109%)** | [4,275, 5,784] |

The golden-EUI hotloop gates (no weather) also move
([1.2518, 1.2522, 1.2526] analytical) — **not re-recorded**, per the
#4314 precedent: baselines are never re-recorded while the physics change
is unreviewed. Verification ladder on this branch: ASHRAE pipeline engine
stage 102 passed / 26 failed; strict ±15% annual-energy gate FAIL (same
reason); npm suite 53/54 (the Case 600 published-band test fails on
cooling); `cargo fmt` clean.

### Status and ask

Per Alex's standing rule this PR is DO-NOT-MERGE. The decision needed:
the sub-hourly conditioned-hour loop as directed (a) does not fix Case 900
(9R4C path is timestep-robust — the LIMIT-05-family roof-solar
under-counting is the live suspect there), and (b) buys Case 600 heating
margin at the cost of blowing Case 600 out of its cooling band. Options
for review: (1) revert this loop and attack Case 900 via the solar
delivery path; (2) keep the loop but scope it to conditioned 5R1C models
only and re-record the affected baselines; (3) drop it. The env var
`FLUXION_CONDITIONED_SUBSTEPS=1` restores hourly behavior exactly.

## 7. Decision (2026-10-08): sub-hourly stepping empirically ruled out for the production path; solar-delivery investigation opened

Alex chose option (3) from issue #4332 on 2026-10-08: drop the
sub-hourly conditioned-hour loop and attack Case 900 heating via the
solar-delivery path (the §LIMIT-05-family roof-solar under-counting
suspect).

Evidence for the rule-out, all measured (PR #4333, closed unmerged; branch
`physics/limit35-case900-conditioned-subhourly` deleted after this doc and
the issue preserved its numbers):

- The production Case 900 path (HighMass → 9R4C, EPW) is outer-dt-invariant:
  3,143.8 kWh (N=1) → 3,148.2 kWh (N=6) standalone; validator unchanged at
  3,477.88 kWh under N=4. The §2 convergence lives only on the standalone
  plain-5R1C harness and does not transfer to the 9R4C validator path.
- The loop does move the 5R1C path, and adversely: Case 600 NAPI/WD600
  cooling 4,675.48 → 9,777.69 kWh — +109% over the [4,275, 5,784] band —
  while heating moved 5,590.48 → 4,450.67 kWh (in band). A change that
  regresses cooling by +109% cannot be scoped around without baseline
  re-records under unreviewed physics.

Consequence: develop keeps hourly stepping exactly. No
`FLUXION_CONDITIONED_SUBSTEPS` code, env hook, or baseline re-record lands.

Next step: quantify roof solar delivery for Case 900 (incidence, absorbed
fraction, shading path on the 900 geometry vs 600) against the ASHRAE 140
reference assumptions, and measure what solar-delivery delta would bring
heating into [1,170, 2,040] kWh. Recorded in `docs/KNOWN_ISSUES.md`
§LIMIT-35.

## 8. Solar-delivery investigation (2026-10-08): roof-solar under-counting falsified; horizontal ground-reflection over-count found instead

Alex's option (3): attack Case 900 heating via the solar-delivery path.
Method: reproduce the recorded numbers first (validator Case 900
3,477.88 / 923.56 kWh H/C; Case 600 harness 5,590.48 / 4,675.48 kWh —
both exact), then isolate with split experiments (temporary in-place
edits, reverted after each measurement; never committed).

### Measurements

**Irradiance source.** With the engine's own solar-position and
`calculate_surface_irradiance` formulas over the Denver-Stapleton TMY
EPW, the horizontal (roof) plane receives **2,180.4 kWh/m²·yr** while the
EPW's own GHI is **1,831.9 kWh/m²·yr** — a **+19% over-count** (348
kWh/m²·yr ≈ ρ·GHI, ρ = 0.2). Cause: Issue #1326 pinned the isotropic
ground-reflected view factor at β = 0° to the FULL ρ·GHI, on the premise
that "a horizontal roof sees the full ground hemisphere". That premise is
physically backwards — an up-facing horizontal plane's normal points at
the zenith, so the ground hemisphere is entirely behind the plane and the
view factor (1 − cos β)/2 correctly gives **0** there. The repo's own
EnergyPlus 25.2 reference data confirms:
`tests/reference_data/solar/case_900_roof_solar_hourly.csv` (Case 900
roof, tilt = 0) records `ground_diffuse_irradiance = 0.0` for all 8,760
hours. Oct–Apr roof-plane: 952.3 vs GHI 802.0 kWh/m² (+18.7%).

**Geometry / shading / absorptance (900 vs 600).** Identical: 48 m² roof
(U ≈ 0.317), 12 m² south glazing, no overhangs or fins on either case,
α_roof = 0.7 / α_wall = 0.6, same site (lat 39.83, lon −104.65). The
cases differ only in mass and in `solar_distribution_to_air` (900 → 0.0,
600 → 0.3). No Case-900-specific solar under-delivery exists in geometry
or shading.

**Delivery routes and split experiments** (validator path, Denver EPW;
ΔH/ΔC in kWh/yr):

| Route | Experiment | ΔH | ΔC |
|---|---|---|---|
| Roof sol-air BC (`t_ext_roof`, step_9r4c) | drop ground term from roof irradiance | +110.4 | −122.5 |
| Roof sol-air BC | double roof irradiance | −572.6 | +913.8 |
| Conducted opaque gain → mass node (`phi_m`) | double the injection | **bit-identical** | **bit-identical** |

The `phi_m` opaque route has zero observable effect on Case 900 annual
H/C — the internal mass node it feeds does not drive the conditioned-air
trajectory (consistent with §3). The only effective roof-solar lever is
the sol-air boundary condition, at ≈ 0.26–0.32 kWh of heating per
kWh/m²·yr of roof-plane irradiance.

### Verdict

**The roof-solar under-counting hypothesis (§LIMIT-05 suspect) is
falsified.** Roof solar is if anything OVER-counted (+19% at the
irradiance source), and correcting the over-count moves Case 900 heating
*up* (3,477.88 → 3,588.28 under the full fix, i.e. further out of
band). Quantitatively, reaching the band [1,170, 2,040] kWh through roof
solar alone would require ΔH = −1,438 kWh (upper bound) to −1,873 kWh
(midpoint), i.e. +4,600–7,100 kWh/m²·yr of roof-plane irradiance on top
of the current 2,180 — 3–4.5× the physically correct value — while
cooling would swing +1,900 to +2,900 kWh toward its own band. No
physical roof-solar correction can close the heating gap.

The ground-reflection finding is still a genuine physics bug (E+-backed),
prepared as DO-NOT-MERGE PR: fix `fix/horizontal-ground-reflection` —
view factor used continuously at the endpoints (β=0° → 0, β=180° → ρ·GHI),
premise test `test_horizontal_ground_reflected` updated with the
corrected physics. **Do not merge and do not re-record baselines before
Alex's physics review** (#4314 precedent). Measured outcome of the fix:
validator Case 900 H 3,477.88 → 3,588.28, C 923.56 → 801.09; validator
Case 600 (Denver path) H 6,165.34 → 6,238.59, C 4,666.73 → 4,498.34;
engine roof-plane irradiance 2,180.4 → 1,814.2 kWh/m²·yr (ratio to GHI
0.990).

Next suspect for Case 900 heating: the air-node/HVAC setpoint-feedback
loop at hour resolution on the 9R4C path (§4's original direction), and
the winter storage-release coupling of the mass network — both outside
the solar-delivery path.

### UPDATE 2026-10-08 (same day): physics approved and merged; baselines re-recorded from the corrected engine

Alex approved the ground-reflection physics the same day. PR #4335 merged
with the conflict resolution against the post-#4334 doc (§7 kept, §8 kept)
and the baselines below re-recorded FROM the corrected engine, per the
#4314 discipline (code first, never the reverse). Old → new, measured:

| Gate | Quantity | develop | this fix |
|---|---|---|---|
| Strict ±15% gate | Case 900 H | 3.723 MWh (gap 116.95%) | 3.823 MWh (gap 123.18%) |
| Strict ±15% gate | Case 900 C | 0.788 MWh (gap 57.83%) | 0.689 MWh (gap 61.24%) |
| Strict ±15% gate | Case 600 H | 6.000 MWh (gap 3.23%) | 6.072 MWh (gap 4.65%) |
| Strict ±15% gate | Case 600 C | 4.023 MWh (gap 5.01%) | 3.877 MWh (gap 7.91%) |
| Fabric parity | case_600 ratio_H / ratio_C | 1.1218 / 0.7057 | 1.1246 / 0.6926 |
| Fabric | case_900_5r1c H / C | 3.723 / 0.788 | 3.823 / 0.689 |
| NAPI (npm) | Case 600 H / C | 5,590.48 / 4,675.48 kWh | 5,654.86 / 4,509.99 kWh (in published band) |
| Hotloop golden EUIs | — | unchanged (gate passes; workload does not exercise the horizontal ground term) | — |
| Grid thermal (Case 600 January) | — | unchanged (gate passes) | — |
| Surrogate drift (Case 900 fallback) | H / C / total | 4594.6127 / 9.0086 / 4603.6213 kWh | reproduces exactly (synthetic fallback, not irradiance-driven) |
| Validator `simulate_case` widened gate | Case 900 H | ~3.144 MWh (gate ≤ 3.162) | 3.2493 MWh; gate re-centered ≤ 3.6 per the #4156 precedent (regression ratchet around the corrected engine, not a band fix) |

Test premises updated honestly (not baselines, not the engine):
`solar_isolation::test_horizontal_ground_reflected` (in the original PR) and
`solar_horizontal_isolation::test_roof_solar_gain_ratio_to_vertical` (this
merge) — the latter asserted "roof receives more ground-reflected than the
vertical wall", which is exactly the removed #1326 endpoint pin; under the
corrected view factor the roof (β = 0°) sees NO ground and the assertion is
inverted with an exact-zero check plus the E+ reference citation. Both
premise flips verified to pass on origin/develop and fail with the fix
before rewriting.

Pre-existing reds untouched and re-confirmed on origin/develop via a
detached worktree: 12 `solar`-filter failures, the Case 600
energy-balance-conservation checks (zone_balance and cross-platform),
and the `cta_bench` clippy compile error.

The Case 900 heating gap remains, now slightly wider (3,477.88 → 3,588.28
kWh validator / 3.723 → 3.823 MWh strict path). The next suspect moves off
the solar-delivery path entirely: the 9R4C air-node/HVAC setpoint-feedback
loop at hour resolution and the winter mass storage-release coupling — see
§9.

## 9. 9R4C air-node/HVAC setpoint-feedback loop + winter mass storage-release coupling (2026-10-08, opened after the §8 merge)

With the solar-delivery path closed by §8 (the ground-reflection fix merged
as PR #4335 and the corrected baseline recorded), the next suspect named at
the end of §8 is attacked here with measurements. Method as before:
reproduce first, instrument the production path, quantify by split
experiments, never tune outputs.

### Reproduction (corrected engine, PR #4335 merged)

- Validator `simulate_case_with_diagnostics`, Denver EPW: Case 900
  **H = 3,588.28 kWh, C = 801.09 kWh** (exact match to the §8 record);
  Case 600 H = 6,238.59 / C = 4,498.34 kWh.
- Standalone 9R4C harness (Case 900 spec + Denver EPW, 14-day warmup,
  8760 hourly steps): H = 3,249.34 / C = 796.86 kWh — matches the
  validator's `simulate_case` path (3.2493 MWh) exactly.

### Instrumentation: where the mass network gets its heat

Reading the production path (`step_9r4c.rs` + `multi_node_solver/mod.rs`):

1. Each hour, the multi-node solver's `zone_temperature` (its internal air
   node) is set to `t_air_mn_pre` — the FREE-FLOAT air implied by the
   previous state (step_9r4c.rs ~line 539).
2. `solver.step_with_gains(...)` then advances the wall/roof/floor mass
   nodes by backward Euler **against that free-float air** — before the
   HVAC demand is computed (step_9r4c.rs ~line 694).
3. The HVAC demand `Q = h_coeff × (T_heat_sp − t_free)` is computed from
   the free-float air `t_i_free_mn` that the (cold) mass just produced, and
   `t_i_act = t_free + Q/h_coeff = T_setpoint` is committed to
   `setpoints.temperatures`.
4. **Nothing writes the conditioned air back into the multi-node solver.**
   The 9R4C mass network free-runs on the un-conditioned trajectory for the
   entire simulation. Only the legacy 5R1C lumped mass receives `t_i_act`
   (post-HVAC integrator, step_9r4c.rs ~line 1175), and per §8's `phi_m`
   split experiment that lumped mass has zero observable effect on the
   Case 900 air trajectory.

Measured consequence (harness, January, corrected engine): the 9R4C
envelope (conductance-weighted mass) sits at **11.4–14 °C overnight and
peaks at 17.0 °C at 15:00** — 3–9 K below the 20 °C setpoint-held air —
while `mass.air_temperatures[0]` reads exactly 20.00 °C for every winter
hour (the controlled state; the free state the feedback loop actually sees
lives inside the solver). The storage-release cycle seen by the setpoint
feedback is solar/ambient-driven storage in a phantom free-floating
building: the mass charges from sun by day and discharges overnight into
an air node nobody is heating, and every hour the cold mass drags
`t_free` down, inflating `Q`. The HVAC energy extracted to hold the
setpoint never charges the mass it will be measured against next hour.

### Seasonal shape vs the reference data

E+ 25.2 hourly reference `tests/reference_data/zone_balance/case_920_energy_hourly.csv`
(Golden EPW; no Case 900 hourly reference exists in-repo) vs the engine's
Case 920 on the same EPW (harness, corrected engine), monthly share of
annual heating:

| Month | E+ | engine |
|---|---|---|
| Jan | 19.0% | 19.3% |
| Feb | 18.9% | 19.5% |
| Mar | 11.6% | 13.0% |
| Apr | 4.4% | 4.1% |
| May | 2.3% | 1.9% |
| Jun–Aug | 0.2% | 0.0% |
| Sep | 0.4% | 0.1% |
| Oct | 6.1% | 5.1% |
| Nov | 14.7% | 14.6% |
| Dec | 22.5% | 22.4% |

The shapes agree within ≈1.5 pp per month (DJF: E+ 60.4% vs engine 61.2%).
The engine's error on Case 920 is a clean +24.6% scale factor (5,295 vs
4,251 kWh), and on Case 900 a +71% scale factor. **The seasonal profile is
not distorted — the gap is a magnitude problem, not a timing problem.** No
storage-release timing shift is hiding in the shape.

### Split experiment A: couple the mass to the conditioned air

Temporary in-place edit (measured, then reverted; never committed): before
`step_with_gains`, in HVAC mode, set the solver's `zone_temperature` to the
previous step's committed `t_act` (`setpoints.temperatures[0]`, i.e. the
setpoint on heating hours) instead of `t_air_mn_pre`; free-float mode keeps
the original behavior.

| Quantity | corrected engine | experiment A |
|---|---|---|
| Validator Case 900 H / C | 3,588.28 / 801.09 kWh | **2,322.94 / 552.83 kWh** |
| Harness Case 900 H / C | 3,249.34 / 796.86 kWh | 2,101.98 / 550.03 kWh |
| Strict-gate Case 900 H / C | 3.823 / 0.689 MWh | 2.455 / 0.478 MWh |
| Strict-gate Case 920 H | 5.252 MWh (out) | **3.302 MWh (IN band [3.213, 4.347])** |
| Strict-gate Case 960 H | 6.426 MWh | 4.055 MWh (still out of [1.742, 2.357]) |
| Strict-gate Case 810 H | 3.823 MWh (in) | 2.455 MWh (falls BELOW band [3.357, 4.543]) |
| Strict-gate Case 970 H | 8.822 MWh | 6.079 MWh (falls BELOW band [10.540, 14.260]) |
| Case 600 (both paths) | unchanged | unchanged |
| Case 950 C | 0.198 MWh | 0.173 MWh |

Coupling the mass to the conditioned air removes **1,265 kWh/yr (35%)** of
Case 900 heating — from 3,588.28 down to 2,322.94 kWh, with cooling moving
3,588.28-side too (801.09 → 552.83 kWh). The remaining gap to the published
upper bound (2,040 kWh) is ≈283 kWh (≈14% of the band midpoint), the same
order as the neighboring documented gaps.

### Verdict

**Genuine physics bug, high confidence: the 9R4C mass network is thermally
decoupled from the conditioned zone.** HVAC heat extraction never charges
the wall/roof/floor mass nodes that produce the free-float air temperature
driving the next hour's setpoint feedback, so (a) the mass free-runs 3–9 K
cold in winter, (b) the feedback loop over-requests heating against a
phantom cold mass every hour, and (c) the metered "heating" is the load of
holding a setpoint over a building whose thermal mass behaves as if
unheated. The winter storage charge/release timing itself is not the
distortion — the shape matches E+ — it is the missing charge route that
scales the whole heating season up. Quantitatively the decoupling accounts
for ≈1,265 kWh/yr of the Case 900 gap (3,588 → 2,323 kWh), i.e. the
majority of the 3,588 − 2,040 = 1,548 kWh excess over the published upper
bound.

Caveats for Alex's physics review:
- Experiment A is a deliberately minimal one-line coupling (previous-step
  `t_act`, zone 0 only). A production fix should couple within the same
  step (the mass still steps against the free air in step A) and handle
  multi-zone (per-zone index); expect the production numbers to differ
  somewhat from the experiment's.
- The side-effect footprint is mixed: Case 920 heating enters its band,
  Case 960 moves toward its band, but Case 810 and Case 970 heating move
  BELOW their bands, and Case 970 is a multi-zone case where the
  experiment's zone-0-only coupling is not meaningful. This needs a
  physics decision, not a mechanical merge.

Fix prepared as PR **DO NOT MERGE** (pending Alex's physics review, #4314
precedent): branch `investigation/9r4c-mass-hvac-coupling`, no baselines
re-recorded. Scratch harness preserved at
`~/OS3/fluxion-case900-setpoint/tests/zz_scratch_setpoint900.rs`.

### §9.1 Decision and merge record (2026-10-08)

Alex approved the coupling fix on 2026-10-08 ("I approve of the changes in
4336. Proceed to fix failing ci checks and merge it. Re-run baseline,
etc."). The PR branch was rebased onto develop `968e629d` (the §8
ground-reflection engine) before any numbers below were recorded, and the
probe reproduced the experiment-A table exactly (validator Case 900
2,322.94 / 552.83 kWh; harness 2,101.98 / 550.03; Case 600 6,238.59 /
4,498.34 unchanged).

**Variant choice (recorded per the review ask).** The production fix keeps
the **previous-step committed `t_act` coupling, per-zone**
(`setpoints.temperatures[zone_idx]`), not a within-step re-stepping of the
mass. Rationale: in HVAC mode the committed `t_act` *is* the active
setpoint on every unclamped conditioned hour, so the mass steps against
the same setpoint-held air a within-step coupling would use — the two
variants differ only in the metering basis of the clamped hours and the
first hour of simulation. Restructuring the 1,600-line step function for a
snapshot/re-step corrector (with its #864 gain-capture and night-vent
boost side effects) was judged not worth that second-order difference;
recorded here as the deliberate choice and left as a possible refinement.

**Honest side effects (recorded, not hidden):** Case 810 heating falls
below its band (3.823 → 2.455 MWh strict; the strict baseline row moves
pass → known_fail) and Case 970 heating drops further below (8.822 →
6.081 MWh; §LIMIT-23 updated). Case 920 heating re-enters its band
(5.252 → 3.302 MWh, now pass). Case 960 remains out (6.426 → 4.055).
Case 950 cooling 0.198 → 0.173 MWh. Case 600/800 (5R1C path) reproduce
unchanged.

**Re-record list (all measured from the corrected engine):**
strict-energy gate baseline (all 16 metrics), fabric harness baseline
(case_900 H/C 3.823/0.689 → 2.455/0.478 MWh; case_950 C 0.198 → 0.173;
Case 600 rows and parity ratios reproduce), surrogate fallback baseline
(4,594.61/9.01 → 3,160.02/95.02 kWh), KNOWN_ISSUES §LIMIT-35/§LIMIT-23
rows + regenerated summary. Hotloop golden EUIs, npm suite (54/54), grid
thermal, and the engine-keyed src unit gates reproduce unchanged. Local
all_tests reds were diffed against a develop worktree per module: every
failing module fails on the identical test names on develop (pre-existing)
except the two gates re-recorded above and the pre-existing
`cross_language_contract` python/node `BatchOracle` divergence
(0.01135 vs 168.77 kWh, bit-identical on develop when both surfaces are
built — previously masked locally by a missing surface).

## 10. Residual-gap thread after the #4336 merge: what still drives Case 900 heating (2026-10-08, opened post-merge)

Fresh thread per the merge follow-up. Method unchanged: reproduce the known
numbers first, then quantify each candidate lever by split experiment. No
engine change is made in this section; nothing below is tuned to pass.

### Reproduction (develop `654db3ca`, post-#4336)

Exact match to the §9.1 record on all four numbers:

- Validator `simulate_case_with_diagnostics`, Denver EPW: Case 900
  **H = 2,322.94 kWh, C = 552.83 kWh** (band [1,170, 2,040]; residual
  ≈283 kWh over the upper bound); Case 600 H = 6,238.59 / C = 4,498.34.
- Standalone 9R4C harness (Case 900 spec + Denver EPW, 14-day warmup,
  8,760 hourly steps): H = 2,101.98 / C = 550.03 kWh.
- Harness annual regime (from the committed `t_act` and metered energy):
  heating 3,667 h, deadband 3,263 h, cooling 1,830 h.

### Lever A — §9.1 metering basis on clamped hours: FALSIFIED

The §9.1 variant note says the committed previous-step `t_act` coupling
differs from a within-step coupling only on capacity-clamped hours and the
first hour. Measured over the full year:

- **Zero capacity-clamped hours.** The committed controlled air stays
  exactly within [20.00, 27.00] °C all year — the ideal-load demand is never
  capacity-clamped, so there is no clamped-hour population at all.
- Heating energy metered on hours *following* a `t_act` that deviated from
  both setpoints (the only hours where the two metering bases can differ):
  **5.11 kWh/yr** — two orders of magnitude below the 283 kWh residual.

The metering-basis refinement cannot close the residual. The §9.1 variant
choice is hereby exonerated as a Case 900 residual driver.

### Lever B — envelope/ventilation basis vs the reference: FALSIFIED

The engine's ventilation conductance is `h_ve = 21.708 W/K` on the Case 900
geometry (8 × 6 × 2.7 m = 129.6 m³), which is exactly the published
0.5 ACH ASHRAE 140 basis (0.5 ach × 129.6 m³ × 1.2 kg/m³ × 1005 J/kgK /
3600 s = 21.7 W/K); the spec confirms `infiltration = 0.500 ach`. The
assumption matches the reference — there is no mismatch to correct.

Sensitivity (harness, `h_ve` scaled, diagnostic only): −10 % ventilation →
H 2,101.98 → 1,929.23 kWh (−172.8 kWh, ≈ −17.3 kWh per 1 % of `h_ve`).
Closing the 283 kWh residual through ventilation would require ≈ −16 %
(≈ 0.42 ach), which contradicts the verified 0.5 ach basis. Falsified.

### Lever C — window-transmitted solar (9R4C-specific): OPEN SUSPECT #1

Split experiment (temporary in-place edit, reverted; never committed):
window solar gain into the 9R4C gain tensor scaled ×1.25.

| Quantity | baseline | ×1.25 window solar |
|---|---|---|
| Validator Case 900 H | 2,322.94 kWh (out) | **1,856.59 kWh (IN [1,170, 2,040])** |
| Validator Case 900 C | 552.83 kWh | 768.33 kWh |
| Validator Case 600 (both) | 6,238.59 / 4,498.34 | **bit-identical** (5R1C path — the edit is 9R4C-only, confirming Case 600's separate +6.9 %-over-band residual shares no mechanism with this one) |

Sensitivity ≈ −18.6 kWh heating per +1 % window solar: closing the residual
needs **+15.2 %** window-transmitted solar on the 9R4C path. The §8 record
falsified *roof* solar under-counting; the *window* transmission/angle
route has not been cross-checked against the E+ reference CSVs. No
reference mismatch is identified yet, so this stays an open suspect with a
measured lever, not a bug.

### Lever D — opaque-solar `phi_m` route: OBSERVABLY INERT (flagged)

The `phi_m` tensor route (`scratch.phi_m` → `phi_m_zone` →
`distribute_opaque_solar_gains` → per-surface mass gains,
step_9r4c.rs ~lines 150-156/631/672) shows **zero** observable effect on
the Case 900 trajectory: scaling the opaque contribution at the tensor
assembly by ×1.25 and again by ×2.0 leaves validator and harness numbers
bit-identical (2,322.94 / 552.83), despite the zone-0 opaque solar input
being nonzero (measured 1,901.82 kWh/yr). By contrast the same split on the
window solar (lever C) moves the numbers substantially, so the gain tensor
is live — only the `phi_m`-routed portion is inert on this path.

Two readings: (a) intentional double-count avoidance — opaque absorption
already reaches the surfaces through the sol-air boundary condition (§8
measured its lever as small), and window beam-to-mass through another
channel (the #864 gain capture), making the `phi_m` routing a legacy/no-op
path; or (b) a real unapplied-gain defect. Distinguishing them needs the
multi-node gain-injection path traced, which is its own thread. Recorded as
a flagged suspect; **no engine fix prepared** — it changes no current
number, and whether it is a bug at all is a physics question for Alex.

### Cooling-side asymmetry: SEPARATE CAUSE, UNCHANGED

Cooling 552.83 kWh vs band [2,130, 3,670] is 4× under the lower bound, and
was already 4–5× under before the coupling (801.09 pre-#4336, 923.56
pre-#4335). The lever-C split bounds any window-solar explanation: cooling
moves ≈ +0.86 kWh per +1 % window solar, so reaching the 2,130 lower bound
would need ≈ +1,800 % — impossible. The heating over-prediction and the
cooling under-prediction do not share a window-solar magnitude cause. The
asymmetry's mechanism (why the high-mass free-float air so rarely exceeds
the 27 °C cooling setpoint — 1,830 cooling hours vs 3,263 deadband hours)
is unexplained and remains open.

### Verdict

The 283 kWh residual is **not** the §9.1 metering basis (≤ 5 kWh/yr), not
a ventilation-assumption mismatch (engine matches 0.5 ach exactly), and not
the inert `phi_m` route (zero effect). The only quantified lever that
reaches into the band is window-transmitted solar on the 9R4C path
(+15.2 % closes it; ×1.25 lands at 1,856.59 kWh, in band), but no reference
mismatch is identified for that route. Case 600's +6.9 % over-band residual
is confirmed mechanically independent (5R1C path, bit-identical under the
9R4C split). Next thread: cross-check the 9R4C window solar
transmission/angle model against `tests/reference_data/solar/` E+ CSVs, and
trace the multi-node gain-injection path to settle lever D's two readings.
Scratch harness preserved at `~/OS3/fluxion-case900-residual/tests/`.

## §11 — Window-solar cross-check against the in-repo E+ 25.2 reference CSVs (lever C follow-up)

Date: 2026-10-08. Scratch harnesses: `~/OS3/fluxion-case900-windowsolar/tests/`
(`zz_scratch_wsolar.rs` — §10 baseline reproduction + per-hour window-gain and
south-irradiance trace; `zz_scratch_residual900.rs` copied from the §10 thread).
All engine numbers below re-measured on develop `c74fed92`; reproduction was
exact (validator Case 900 2,322.94 / 552.83 kWh; harness 2,101.98 / 550.03 kWh).

### Method

The §10 lever C asked whether the 9R4C window solar transmission is
under-counted. The engine window-solar pipeline is
`calculate_zone_solar_gain` → per-orientation `calculate_window_solar_gain`
(beam: `SHGC × ratio(θ)` on the ASHRAE-Fundamentals Ch.15 table; sky diffuse:
`0.9 × SHGC`; ground-reflected: `0.85 × SHGC`) → W/m²-normalized
`solar.solar_gains` → 9R4C tensor split
(`phi_ia = sol_to_air`, `phi_st = remaining × (1 − beam_to_mass)`,
`phi_m = remaining × beam_to_mass + opaque_sol_w`; step_9r4c.rs ~lines 129-156,
629-656). Cross-checked at three levels against
`tests/reference_data/solar/` (E+ 25.2, Golden TMY3 EPW), keeping the §10
caution: the `zone_balance` CSVs and `data/ashrae140_reference.json` were not
mixed in.

**Weather confound, handled.** The validator runs the Denver-Stapleton
(724690) EPW; the reference CSVs were generated from the Golden-NREL
(724666) EPW, which is also in-repo. Comparing the engine on Stapleton against
a Golden reference conflates weather with model. The cross-check therefore runs
the same harness on both EPWs; model-vs-reference statements are made on the
Golden run only.

### Results (Golden EPW, same weather as the reference)

Annual, south-facing 12 m² window:

| Quantity | Engine | E+ 25.2 reference implies | ratio |
|---|---|---|---|
| South incident (beam+sky+ground), kWh/m²·yr | 1,214.6 | 1,312.6 | **0.925** |
| Window transmitted (with SHGC 0.77 model applied to reference irradiance), kWh/yr | 9,519 | 10,315 | **0.923** |
| Transmission fraction (transmitted / incident×12 m²) | 0.653 | 0.655 | **0.996** |

Monthly incident ratios span 0.87-0.93 — every month low, no seasonal sign
flip. Component decomposition:

| Component (annual Wh/m² on south wall) | Engine | E+ | ratio |
|---|---|---|---|
| ground-reflected | 160,120 | 161,904 | **0.989** |
| beam + sky-diffuse | 1,054,522 | 1,150,655 | **0.916** |

Hour-level discriminator (beam-dominated hours `DNI·cosθ > DHI`, n=1,088 vs
diffuse-dominated n=7,672): engine/E+ = **1.160** on beam-dominated hours,
**0.887** on diffuse-dominated hours. So the gap sits in the tilted
sky-diffuse model — the engine's Perez tilted-diffuse implementation delivers
≈11 % less than E+ 25.2 on diffuse-heavy hours — while the ground-reflected
isotropic model matches within 1.1 % and the engine beam is if anything
higher. Both engines are Perez-form; this is an implementation-level
divergence, not a model-choice difference. The December Stapleton-vs-Golden
"anomaly" (+65 %) in the naive cross-check was pure weather-file difference:
on the same weather the engine is uniformly ≈7-13 % low, not high.

### Split experiments (temporary, assert-guarded, reverted; tree verified clean and baselines bit-identical after revert)

1. **Transmittance / incidence-angle model: no divergence.** The engine's
   transmitted/incident fraction (0.653) matches the ASHRAE-table
   reconstruction of the E+ reference irradiance to 0.4 %. The §10 "×1.25
   window solar" lever is not explained by the SHGC or angle model.
2. **Gain-injection path: live.** A null-control run scaling only the traced
   (harness-side) window gain left H bit-identical (2,101.98), while the §10
   in-engine ×1.25 split moved the validator to 1,856.59 — confirming
   `step_physics` consumes exactly the tensor route traced above.
   Observed on the `phi_m` route while mapping it: `phi_m` carries BOTH the
   window beam-to-mass share (`remaining_sol × m_sol_frac`) and the opaque
   term (`opaque_sol_w`); §10 scaled only the opaque term and saw no effect.
   Lever D's trace (which of the two channels is dead inside
   `step_with_gains`) remains the follow-up.
3. **Measured E+-matching diffuse correction (diagnostic, not a fix).**
   Temporary patch scaling the tilted Perez diffuse ×1.1286 (the measured
   diffuse-hour E+ ratio):

| Quantity | baseline | ×1.1286 tilted diffuse |
|---|---|---|
| Validator Case 900 H | 2,322.94 kWh (out) | 2,280.12 kWh (still out) |
| Validator Case 900 C | 552.83 kWh | 583.10 kWh |
| Validator Case 600 H / C | 6,238.59 / 4,498.34 | 6,148.74 / 4,665.82 (moves — the irradiance path is shared, unlike the 9R4C-only splits) |
| Harness Case 900 H / C | 2,101.98 / 550.03 | 2,060.48 / 580.14 |

### Verdict

The §10 +15.2 % window-solar under-delivery hypothesis is **falsified as a
transmission bug**: the window transmittance, incidence-angle model and
gain-injection path all match the E+ reference basis. What the cross-check
*did* find is a genuine, quantified divergence **upstream** of the window: the
tilted-surface Perez sky-diffuse implementation delivers ≈11 % less than
E+ 25.2 on diffuse-dominated hours (−8.4 % beam+sky, −7.5 % total incident on
the south wall annually, uniform across months). But the measured heating
sensitivity to fully correcting it is only **−42.8 kWh of the 283 kWh
residual (≈15 %)** — the corrected Case 900 (2,280.12 kWh) remains
above the 2,040 kWh upper bound, and the correction would also move Case 600
and every other solar-driven metric, requiring re-records. No engine change is
prepared: whether the Perez tilted-diffuse divergence is a defect to repair or
an accepted model-form difference is a physics decision, and no single
window-solar lever reaches into the band.

The remaining ≈240 kWh residual is not explained by window solar, the metering
basis (§10: ≤5 kWh/yr), ventilation (§10: falsified), or the opaque `phi_m`
route (§10: inert). Next thread, if pursued: the Perez tilted-diffuse
divergence on its own terms (it depresses solar gains in ALL cases, and
low-sun diffuse hours dominate the annual window budget), and lever D's
multi-node gain-injection trace.
