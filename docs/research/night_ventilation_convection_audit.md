# ACH-Dependent Interior Surface Convection Audit — Issue #4238

## 7-Line Summary

Audit of whether fluxion underpredicts night-ventilation cooling of thermal mass because its interior surface convection ignores air-change rate (Buildings 2026, "Conditional Effectiveness of Night Ventilation During Heatwaves"). Finding: fluxion already has an ACH-dependent interior-convection mechanism — `h_tr_is_ach_multiplier` (`src/sim/ventilation.rs:217`), the ASHRAE/EnergyPlus correlation `h_c = 3.45 + 0.84·ACH^0.8` — wired into the 9R4C step (`src/sim/thermal_model_physics/physics_impl/step_9r4c.rs:602`), but capped at 2.0× (`capped_h_tr_is_ach_multiplier`, `src/sim/ventilation.rs:269`) to suppress a morning-ramp peak-cooling artifact (Issue #2871), well below the 2.91× the correlation gives at the Case 650/950 night-vent rate of 13.14 ACH. A reduced-model replica of the production 9R4C parallel-resistance network shows the cap costs little at night: removing it entirely changes 8-hour wall-mass cooling by only ≈0.06 K, because at fan-driven ACH the mass→air path is series-limited by the mass-side conductance, not by the film. The 5R1C series path is more sensitive to the multiplier, and fluxion applies one aggregate multiplier rather than per-surface HTCs, which is the part of the paper's finding fluxion does not capture. **Verdict: the gap is real but narrow — it is a cap-and-granularity question, not a missing correlation. No constants were tuned and no engine code changed; a scoped follow-up is proposed in §6.**

## 1. Question

Paper: *Conditional Effectiveness of Night Ventilation During Heatwaves*, Buildings 16(17), 2026 (https://www.mdpi.com/2075-5309/16/17/3555), ~18,000 EnergyPlus runs over 36 Shanghai heatwaves. Its engine-modelling finding: interior surface convective heat-transfer coefficients vary with air-change rate and thermal-mass distribution, and existing correlations underpredict the effect — zone-level airflow alone does not characterise surface heat exchange.

Issue #4238 asks, as a research spike only:

1. Which interior-surface convection correlations does fluxion use, and does any depend on local air-change rate?
2. Is the effect material in fluxion's network?
3. If material, scope the model change as its own issue. Do not tune constants to match the paper.

## 2. Interior convection models found in the codebase

| # | Model | Where | Depends on |
|---|-------|-------|------------|
| 1 | ACH multiplier on `h_tr_is`: `h_c = 3.45 + 0.84·ACH^0.8` (ASHRAE HoF ch. 4 / EnergyPlus Eng. Ref.; Issues #1279, #1624) | `src/sim/ventilation.rs:217` (`h_tr_is_ach_multiplier`) | Night-vent ACH |
| 2 | Cap on #1 at 2.0× (Issue #2871: unbounded 2.91× at 13.14 ACH caused +48–92% peak-cooling overshoot on Cases 610–650 during the morning ramp) | `src/sim/ventilation.rs:269` (`capped_h_tr_is_ach_multiplier`) | — |
| 3 | ISO 13790 per-surface interior film coefficients: wall 7.69, ceiling 10.0, floor 5.88 W/m²K | `fluxion-core/src/construction.rs:59-67` | Surface orientation only |
| 4 | ASHRAE 140 fixed interior film 8.29 W/m²K (`ASHRAE140_H_INT`) | `src/physics/constants/thermal/ashrae_140/materials.rs:20` | Nothing (140 prescription) |
| 5 | CTF interior boundary film (`INTERIOR_FILM_COEFF`) applied via `SurfaceBC::new_interior` / `interior_heat_flux` | `src/sim/thermal_model_core/mod.rs:355,380` | Zone temperature only |
| 6 | FFD/CFD adapter CHTC: `h = max(2.5, 2.5 + k_v·v_t)` from local tangential velocity | `src/sim/ffd_cfd_adapter.rs:44-66` | Local air velocity — the only velocity-driven HTC, and it is **not** in the production thermal path |

Application of #1/#2: the capped multiplier is applied in the 9R4C physics step at `src/sim/thermal_model_physics/physics_impl/step_9r4c.rs:602` and reused at lines 803, 841, 976 of the same file; `src/sim/thermal_model_iterative.rs:1020-1083` discusses the same night-vent boost. Production 9R4C mass↔air coupling uses `MassAirCouplingMode::ParallelResistance` (`src/sim/thermal_model_core/mod.rs:2206-2216`, Issue #1859): per surface k, `T_s_k = (h_ms_k·T_m_k + h_is·T_air) / (h_ms_k + h_is)`.

## 3. The ventilation side

- The aggregate still-air value the multiplier scales is `h_tr_is = 3.45 W/m²K × floor area` (`src/sim/thermal_model_physics/dispatch_state.rs:70`).
- `WeatherDependentVentilation` exists (`src/sim/ventilation.rs`, referenced by `src/sim/thermal_model/hybrid.rs` and tests) but the production REST path hard-codes infiltration at 0.5 ACH: `src/api/server/simulate.rs:234` (`DEFAULT_INFILTRATION_ACH`), pushed unconditionally at lines 294 and 305; the schema's infiltration schedule is not consumed there. Night-vent ACH therefore reaches the convection multiplier through the case/spec path (e.g. the ASHRAE 140 night-vent cases), not through the weather-dependent ventilation model.

## 4. Materiality — reduced-model replica

Method: a unit-level Python replica of the production 9R4C parallel-resistance network (per-surface mass nodes with fluxion's Case-900-style solver parameters, series `h_ms`/`h_is` surface coupling, zone air coupled by `h_ve`), run for an 8-hour night, mass at 30 °C, outdoor 24 °C, no gains. This is a reduced model of fluxion's equations, not a full engine run (the VM cannot build the workspace); magnitudes, not absolutes, are the evidence.

Multiplier values from fluxion's own correlation (script §A): 0.5 ACH → 1.14×, 3 ACH → 1.59×, 7.5 ACH → 2.22× (cap engages), 13.14 ACH → 2.91× natural / 2.00× as capped.

9R4C, night fan at 13.14 ACH (`h_ve,total` ≈ 592 W/K), 8 h wall/roof/floor mass cooling:

| Multiplier | Wall Δ vs 1× | Roof Δ | Floor Δ | Effective mass→out conductance H_eff |
|---|---|---|---|---|
| 1.00× (no boost) | — (28.21 °C) | — | — | 82.6 W/K |
| 2.00× (current cap) | −0.151 K | −0.057 K | −0.026 K | 92.1 W/K (+12%) |
| 2.91× (uncapped correlation) | −0.208 K | −0.076 K | −0.034 K | 95.7 W/K (+16%) |
| 4.00× | −0.245 K | −0.087 K | −0.039 K | 98.0 W/K (+19%) |

Reading: the boost that exists matters (+12% H_eff at the cap). The cap's cost is small — going from the cap to the full correlation adds ≈0.06 K of wall-mass cooling over a full night — because at fan ACH the mass→air path (≈96–114 W/K) is far smaller than `h_ve` (≈592 W/K) and is series-limited by the mass-side conductance `h_ms`, not by the film. The film is not the bottleneck in the 9R4C night-vent regime.

5R1C series path (`H_tr_3 = h_is·h_ms / (h_is + h_ms)`, cf. `src/sim/thermal_model_physics/hvac.rs:113-120`) is structurally more sensitive because `h_is` enters a single series bottleneck for the whole zone: at `h_ms = 100 W/K` (the range `src/sim/diagnostics.rs` expects for `h_tr_ms`, 10–100 W/K), H_tr_3 rises 62.4 → 76.8 → 82.8 W/K for multipliers 1 → 2 → 2.91 (+33% across the full range). Larger `h_ms` assumptions inflate this further and are not used here as evidence.

## 5. What fluxion does not capture from the paper

1. **Per-surface / mass-distribution dependence.** The paper's second claim is that HTC variation interacts with where the mass is. Fluxion applies one zone-aggregate multiplier to `h_tr_is`; the per-surface film values (#3 above) are orientation constants that never see ACH or velocity.
2. **The cap is an artifact guard, not physics.** The 2.0× cap exists to stop a morning-ramp load pulse (Issue #2871). Whether the pulse is a solver/controls artifact that could be fixed at its source — which would let the correlation run uncapped — has not been investigated.
3. **Velocity-driven HTC** exists only in the FFD/CFD adapter (#6), disconnected from the production network.

## 6. Verdict and proposed follow-up

**Verdict: gap real but narrow.** Fluxion is not missing the ACH→HTC mechanism the paper's headline implies; it has it, capped, in aggregate form. The reduced-model evidence says uncapping alone would change night-vent mass cooling by ≈0.06 K in the 9R4C path — immaterial. The defensible follow-up is granularity and the cap's root cause, in this order:

1. Revisit the Issue #2871 morning-ramp overshoot at its source (controls/solver ramp), with Case 650/950 peak-cooling as the guard metric; if the artifact is fixed there, re-test the uncapped correlation.
2. Per-surface ACH multipliers (wall/roof/floor film values from `fluxion-core/src/construction.rs` scaled individually) behind the existing selector, validated on Cases 650/950 night-vent schedules — no new correlation, no tuned constants.
3. Only if (1)–(2) show material movement: evaluate a local-velocity HTC (the FFD adapter's form) for the production path.

No engine code was changed by this audit, and no constants were tuned to the paper, per the issue's scope.

## 7. Provenance

Analysis produced for Issue #4238 on develop `44716f5f`. Code citations verified against that tree. The reduced-model replica (§4) is a transcription of the cited production equations with Case-900-style parameters; it is analysis tooling for this audit, not a fluxion component, and is not committed.
