# ADR-0017: Equation-based DAE teacher as the single production thermal path — ending the two-solver limbo

> **Summary 1/7:** The repo is in a two-solver limbo: legacy 5R1C is declared structurally dead for 600-series peaks (#1522, #3983) yet still executes in every default build via a silent cfg fall-through under a Gauge-named default; the GaugeSolver prototype is quasi-steady (algebraic flux, no thermal-mass dynamics, setpoint-forcing bug, hardcoded ACH=0.5, h_vent=0 — #3982 evidence) and its β-soak (#3286) sits at 0/30 nights green.
> **Summary 2/7:** Decision D1: adopt an **equation-based DAE teacher** — envelope + zone air + HVAC in one system of differential-algebraic equations, adaptive-step implicit BDF/IDA-type integration with tolerance control (the IDA ICE / Modelica paradigm) — as the single production thermal path; teacher-class validation basis is ASHRAE 1052-RP analytical steady-periodic conduction plus PCM test-box experiments (Mazzeo et al.).
> **Summary 3/7:** Decision D2: legacy 5R1C/9R4C retire to a **documented legacy feature** — reachable only via explicit `ThermalSelector` opt-in, never a default and never a silent fall-through target. Decision D3: GaugeSolver is designated the **DAE-path prototype**: its boundary-condition translation / connection assembly is the seed of the DAE residual assembly; #3982 must replace its quasi-steady flux with genuine thermal-mass dynamics (demonstrated mass node) or the ADR directs retirement.
> **Summary 4/7:** Decision D4 (interim posture, merged with this ADR): the default thermal selector is **cfg-dependent and explicit in every build** — `Gauge` in `gauge-solver` builds (unchanged ADR-0007 posture; β-soak continues as nightly authority), the **explicit legacy `FiveROneC`** in default builds (HighMass auto-promotion to 9R4C preserved); an explicit `Gauge` selector in a default build now panics loudly at construction. The silent fall-through is gone.
> **Summary 5/7:** Decision D5 (migration sequence): **conduction first** (#3979 CTF coupling defect → #3980 Crank–Nicolson/BDF2 + CTF demoted to cross-check → #3981 1052-RP analytical regressions → #3983 per-surface multi-node conduction), then **zone air** (#3982 multi-node/zonal air in the same DAE + gauge-path resolution), then **HVAC** (#3984 pressure-driven equation-based air loop); cross-cutting: #3985 single trait-slot dispatch mechanism, #3986 teacher validation suite, #3987 teacher-to-surrogate training pipeline.
> **Summary 6/7:** The unconditional default flip to the teacher is gated on the **#3986 validation suite passing** — superseding the old "β-soak-trips → feature flips" plan (§LIMIT-21 / #3297). Rejected alternatives: flipping `default = ["gauge-solver"]` today (gauge's Case 650 blowup, 85.57 vs the 4.82–7.06 band, would break required ASHRAE gates — and RULES.md bans tuning to compensate); retiring GaugeSolver now (discards reusable connection/translation machinery before its replacement exists); keeping the silent fall-through (the limbo this ADR exists to end — every solver decision was being re-litigated per LIMIT and the surrogate pipeline #3987 had no stable teacher to train from).
> **Summary 7/7:** Constraints carried through the migration: RULES.md — fixes are architectural and diagnostic-driven, never parameter tuning to pass ASHRAE 140; energy balance preserved; the strict ±15 % energy gate (#1333/#3572) and existing ASHRAE tolerance bands are not relaxed at any step. Zero default-build physics drift was verified for the D4 code change (identical ASHRAE failure sets vs `develop` in both feature states).

- **Status:** Accepted (D4 interim posture implemented; D1–D3, D5 in progress via the #3979–#3987 series)
- **Date:** 2026-09-25 (record created)
- **Deciders:** Fluxion maintainers
- **Supersedes:** ADR-0007 §"Phase A8 production posture" (the `gauge-solver` feature, its β-soak-gated flip plan, and the default-build `Gauge`-named default with silent fall-through); ADR-0007's structural-work program continues unchanged.
- **Issues:** Parent #3978; children #3979, #3980, #3981, #3982, #3983, #3984, #3985, #3986, #3987; related #1522, #3286, #3290, #3291, #3297, #3305, #3724, #3817

## Context

The v1.3 line has two half-alive zone-solver paths and a validation record that punishes each for the other's sins:

1. **Legacy 5R1C is declared structurally dead for 600-series peaks** (#1522; the #3983 evidence: single lumped mass node cannot produce the measured peak-magnitude/swing pair). It is nevertheless *executed physics* in every default build, because `ThermalSelector::default()` named `ZoneSolverKind::Gauge` in both feature states (ADR-0007 Phase A8) and the default build's dispatcher silently fell through to 5R1C/9R4C (`step_dispatcher.rs`, pre-ADR-0017). A declared-dead solver as the *de facto* default under another solver's name is the limbo.
2. **GaugeSolver is not yet a teacher.** The gauge path is quasi-steady — algebraic fluxes with no thermal-mass dynamics (`C_mass` is telemetry-only), a setpoint-forcing bug, hardcoded infiltration ACH=0.5 and h_vent=0 (#3982 evidence). Its β-soak (#3286) is at 0/30 nights green; #3817 routes heavyweight HighMass specs to 9R4C because gauge cannot satisfy the swing-reduction sanity bound; Case 650 blows up (85.57 kWh vs the 4.82–7.06 band).
3. **The cost of limbo:** every solver decision is re-litigated per LIMIT; the surrogate pipeline (#3987) has no stable teacher to train from; two code paths double the validation surface and the doc-sync surface (AGENTS.md, FEATURES.md, Cargo.toml canonical wording all carry the two-axis explanation).

The research basis (`research_notes/building-simulation-solvers-20260925-1507`, §2, §8): the IDA ICE / Modelica class of tools solves envelope + zone air + HVAC as one DAE with adaptive-step implicit (BDF/IDA-type) integration and tolerance control; Mazzeo et al. validate PCM test boxes against exactly this class; Spawn-of-EnergyPlus adopts the equation-based paradigm for controls/loop-coupling fidelity. Fluxion already has the acausal building blocks (`fluxion-fluid`, ADR-005).

## Decision

### D1 — Adopt the equation-based DAE teacher as the single production thermal path

Envelope conduction, zone air (single- and multi-node), and HVAC air-side in **one** coupled system of differential-algebraic equations; adaptive-step implicit BDF/IDA-type time integration; solver tolerance as the accuracy control (replacing per-case timestep tuning). Teacher-class validation: ASHRAE 1052-RP analytical steady-periodic conduction solutions, ASHRAE 140 fabric cases, PCM test-box experiments (Mazzeo et al.), with the existing EnergyPlus cross-checks retained.

### D2 — Legacy 5R1C/9R4C retire to a documented legacy feature

Available via explicit `ThermalSelector` opt-in (`FiveROneC` / `NineRFourC` selectors and binding-layer strings `"5r1c"` / `"9r4c"`); never a default, never a silent fall-through target. The HighMass ⇒ 9R4C auto-promotion (ADR-0002) is preserved as construction-driven policy honored by the legacy dispatch arms.

### D3 — GaugeSolver is the DAE-path prototype

Its boundary-condition translation and connection assembly are the seed of the DAE residual assembly (the same "assemble connections, then integrate" shape). **#3982 is the decision executor:** it must replace the quasi-steady flux with genuine DAE mass-node dynamics — the bar is a *demonstrated thermal-mass node* passing the swing-reduction sanity bound — or GaugeSolver is retired and the DAE residual assembly is built fresh on the #3985 trait-slot mechanism.

### D4 — Interim posture: one explicit default per build; the silent fall-through is removed (implemented with this ADR)

- `ThermalSelector::default()` is **cfg-dependent**: `ZoneSolverKind::Gauge` in `gauge-solver` builds (ADR-0007 Phase A8 production posture, β-soak #3286 continues as the nightly authority on this arm); the **explicit legacy `FiveROneC`** in default builds — the selector name now matches the physics the default build actually executes.
- An **explicit `Gauge` selector in a default build panics loudly** at `ThermalModel::from_spec_with_selector` construction (naming this ADR and the feature flag); the dispatcher's `Gauge` arm keeps a defense-in-depth panic for hand-assembled models.
- The HighMass ⇒ 9R4C auto-promotion survives the selector flip: the `FiveROneC` dispatch arm honors the `is_nine_r4c_model` flag exactly as the removed fall-through arm did.
- Zero default-build physics drift was verified: the ASHRAE 140 test failure sets are byte-identical to `develop` in both feature states (24 known-LIMIT failures default / 32 gauge build; `tests/gauge_dispatcher_cases.rs` pins the new contract).

### D5 — Migration sequence

1. **Conduction first** (the envelope carries the 600/900-series error budget): #3979 (fix/bypass the 900-series CTF coupling defect) → #3980 (Crank–Nicolson/BDF2 FD time integration; CTF demoted to fast cross-check) → #3981 (ASHRAE 1052-RP analytical steady-periodic regressions) → #3983 (per-surface multi-node conduction with distributed solar).
2. **Zone air:** #3982 (multi-node/zonal air in the same DAE solver; resolve the gauge path per D3).
3. **HVAC:** #3984 (pressure-driven equation-based air loop replacing imposed-flow sequential coupling).
4. **Cross-cutting:** #3985 (single trait-slot component dispatch mechanism — the `HeatConductionSolver` / `ThermalModelTrait` swap-point consolidation), #3986 (teacher validation suite — the flip authority), #3987 (teacher-to-surrogate training pipeline with swap acceptance criteria).

## Consequences

**Positive**

- One validation surface, one doc story; the "two-axis" explanation is deleted everywhere it was maintained.
- Default-build users get an honest selector name (`FiveROneC` executes 5R1C) and a loud error instead of silent substitution (#3305 observability becomes trivial).
- The surrogate pipeline (#3987) gets a defined teacher and swap criteria.
- The default flip has a concrete, testable authority (#3986) instead of a calendar/soak race.

**Negative**

- Default-build consumers who *explicitly* requested `gauge` (CLI `--zone-solver gauge`, binding `zone_solver: "gauge"`) now fail loudly instead of silently running legacy — deliberate; the error names the feature flag and the legacy strings.
- The gauge prototype carries known physics defects until #3982 lands; the β-soak continues against a solver whose mass dynamics are pending.
- Two dispatch arms (legacy + prototype) coexist until the teacher flip; the difference is that the boundary is now explicit and loud rather than silent.

**Neutral**

- ADR-0007's structural-work program (case-spec coverage, multi-zone wiring, observability) carries over unchanged; only its production-posture section is superseded.
- The β-soak nightly (`nightly-ashrae-140-gauge.yml`) runs unmodified — it already passes `--zone-solver gauge` explicitly and does not rely on the default selector.

## Test acceptance criteria (D4, implemented)

- `tests/gauge_dispatcher_cases.rs`: default-build default selector is `FiveROneC`; explicit `Gauge` panics with the `gauge-solver` feature name; gauge-build default stays `Gauge`; HighMass + default ⇒ `effective_zone_solver == NineRFourC` (auto-promotion preserved).
- `src/sim/thermal_selector.rs` unit test: cfg-dependent default in both feature states.
- Binding truth tests (`StateMatrices::effective_solver`, Python `effective_zone_solver`): report the dispatch outcome per build — updated wording, unchanged values.

## References

- ADR-0002 (9R4C high-mass promotion), ADR-0005 (`fluxion-fluid` acausal HVAC), ADR-0007 (GaugeSolver structural work — partially superseded), RULES.md (no ASHRAE tuning; energy balance), `docs/KNOWN_ISSUES.md` §LIMIT-21, `research_notes/building-simulation-solvers-20260925-1507` §2, §8; Mazzeo et al. (PCM test-box DAE validation); ASHRAE Standard 140; ASHRAE 1052-RP.
