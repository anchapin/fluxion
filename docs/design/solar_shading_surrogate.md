# Solar/Shading Surrogate Feature Design — Average Shading as the Primary Obstruction Feature

<!-- 7-line summary -->
<!-- Line 1: Design finding for Fluxion's (post-1.0) solar/shading surrogate feature set. -->
<!-- Line 2: Evidence: occlusion-aware ML for climate-adaptive BIPV facade yield prediction, -->
<!-- Line 3: Results in Engineering (2026-10-07), DOI 10.1016/j.rineng.2026.113363. -->
<!-- Line 4: Conclusions: average shading is the primary obstruction feature; explicit -->
<!-- Line 5: obstruction geometry must earn its place on Fluxion's own cases; surrogates -->
<!-- Line 6: pay off only in batch screening (physics stays the single-query default); invest -->
<!-- Line 7: in geometry-representation fidelity before ML model tuning. -->

| Field | Value |
|---|---|
| Status | Design finding (documentation only — no code change ships with this artifact) |
| Scope | Surrogate feature design for solar/shading workloads; **post-1.0 scope (goals 3 & 4)** |
| Issue | #4557 |
| Source paper | *Occlusion-aware machine learning for climate-adaptive BIPV facade yield prediction*, Results in Engineering (2026-10-07), DOI [10.1016/j.rineng.2026.113363](https://doi.org/10.1016/j.rineng.2026.113363) |
| Related | `ARCHITECTURE.md` § "Module 2: Solar Position & Irradiance"; `docs/SURROGATE_GOVERNANCE.md`; `src/sim/shading.rs` |

---

## 1. Scope note

Surrogate work in Fluxion is **post-1.0 scope (goals 3 & 4)**. Nothing in this
artifact changes today's behaviour: `ARCHITECTURE.md` § "Module 2: Solar
Position & Irradiance" records that the solar module exposes standalone
functions with **no ML surrogate swap point at the solar calculation layer**,
because solar position/irradiance is deterministic physics. This document fixes
the *feature-design conclusions* that any future solar/shading surrogate must
honour; it is not a proposal to add a surrogate, a swap point, or a model.

The companion surrogate **validation protocol** is documented separately (issue
#4556) and is deliberately **not** restated here — this artifact covers feature
design only.

---

## 2. Evidence

Source: *Occlusion-aware machine learning for climate-adaptive BIPV facade
yield prediction*, **Results in Engineering**, published 2026-10-07, DOI
[10.1016/j.rineng.2026.113363](https://doi.org/10.1016/j.rineng.2026.113363).

The paper trains a **tree-ensemble surrogate** to predict facade PV yield under
weather and obstruction (occlusion) variation, and reports:

| Reported quantity | Value |
|---|---|
| Leave-one-city-out pooled error | **6.95 %** |
| Leave-one-climate-out error | **7.2 %** (tropical) to **~15 %** (continental) |
| Ablation — climate-only features | error **> 33 %** |
| Ablation — climate + average shading | error **~7.5 %** |
| Ablation — + full explicit obstruction geometry | adds **~nothing** (0.016 pp, confidence interval spans zero) |
| Batch screening of 1,000 candidates | **< 2 s** (5–20× faster than the physical reference) |
| Single-query latency | **slower** than direct physical calculation |
| Geometry-representation sensitivity | changing the geometric representation alone shifts reference yields by up to **65 %** for narrow obstructions |

Reading of the ablations: the *aggregate* obstruction value ("average
shading") carries essentially all of the learnable signal about obstructions —
climate alone leaves > 33 % error, and adding average shading collapses the
error to ~7.5 %. Layering the full explicit obstruction geometry on top of
average shading changes the error by 0.016 pp, with a confidence interval that
spans zero, i.e. the extra geometry buys nothing measurable for the surrogate.
Reading of the sensitivity result: how the obstruction is *represented
geometrically* moves the reference yields themselves by up to 65 % for narrow
obstructions — geometry fidelity dominates the error budget, ahead of the
choice of ML model.

---

## 3. Design conclusions for Fluxion

### D1 — Average shading is the primary obstruction feature

Any solar/shading surrogate feature set should lead with an **average (i.e.
aggregate, time- and area-aggregated) shading** feature and treat explicit
obstruction geometry as optional. In Fluxion terms, the natural instance of
this feature already exists in the physics layer: `src/sim/shading.rs`
`calculate_shaded_fraction(...)` returns the per-surface **effective shaded
fraction** (geometric blocked fraction, with the vegetation
`TransmittanceSchedule` folded in). The average-shading feature is the
time/area-weighted
aggregate of that quantity over the facade/period of interest — i.e. the
feature already describes the obstruction's *value*, not its *shape*.

Rationale: the paper's ablation shows climate-only features leave > 33 % error
while adding average shading drops it to ~7.5 %, and further explicit geometry
adds 0.016 pp (CI spans zero). The same ordering is the cheapest possible
starting point for Fluxion: one aggregated scalar per surface, no geometry
pipeline in the feature extractor.

### D2 — Explicit obstruction geometry must earn its place on Fluxion's own cases

Before any explicit obstruction geometry (overhang depth, fin width/height,
obstruction extents, orientation …) is added to a Fluxion surrogate feature
set, it must be shown to improve accuracy **on Fluxion's own cases** against
the D1 baseline (average shading only). The paper's result is evidence that
the expected gain is ~0 — the feature can be carried, but it must not be
assumed valuable. Any such ablation is measured against Fluxion's own
reference outputs; the paper's numbers are a prior, not a substitute.

### D3 — Surrogates pay off only in batch screening; physics stays the single-query default

Batch screening of 1,000 candidates completes in **< 2 s** (5–20× faster than
the physical reference), while **single queries are slower** than direct
physical calculation. Therefore:

- The surrogate's value proposition is **throughput on batch/design-space
  screening** (candidate ranking, sweep exploration), not per-call latency.
- For single-query use, the **physics path remains the default**. This matches
  Fluxion's swappable-surrogate architecture: `ARCHITECTURE.md` § "ML
  Surrogate Ready" and the `HybridRouting` / `UnifiedThermalModel` split mean
  physics and surrogate are interchangeable implementations behind a trait, and
  the routing policy — not the surrogate — decides which one runs. A solar/shading
  surrogate, when it exists, plugs into that same swappable arrangement; it
  does not displace physics for the ordinary single-query path.
- Any future solar/shading surrogate must therefore ship with a documented
  fallback to the physical path (per `docs/SURROGATE_GOVERNANCE.md` § 2), and
  the batching path is where its speed advantage is real.

### D4 — Invest in geometry-representation fidelity before ML model tuning

Changing only the geometric representation shifts reference yields by up to
**65 %** for narrow obstructions. The error contributed by the representation
of the geometry is therefore an order of magnitude larger than the difference
between ML model choices, and larger than the entire gain that explicit
geometry features add to the surrogate (D2). Design consequence for Fluxion:
improve how shading/obstruction geometry is *represented and resolved* (the
`src/sim/shading.rs` projection model, its discretization of narrow
obstructions, and the fidelity of the reference geometry) **before** spending
effort on surrogate model architecture, hyper-parameters, or ensemble
composition. Model tuning cannot recover a mis-represented ground truth.

---

## 4. What this artifact does not claim

- It does **not** claim the paper's error percentages transfer to Fluxion. The
  paper's target quantity is **facade PV yield** (BIPV); Fluxion today
  computes solar heat gain, not PV yield, and has no PV module. The paper's
  numbers are quoted as evidence for *feature-design ordering* (D1, D2) and
  *cost/benefit shape* (D3, D4), not as a Fluxion accuracy claim.
- It does **not** introduce a solar/shading surrogate, a swap point at the
  solar layer, or any feature-schema change to a trained model.
- It does **not** restate the surrogate validation protocol (issue #4556).

---

## 5. Related documents

- `ARCHITECTURE.md` § "Module 2: Solar Position & Irradiance" — solar module
  contract; the note that solar has no ML surrogate swap point today.
- `docs/SURROGATE_GOVERNANCE.md` — binding surrogate policy (domain of
  validity, fallback behaviour, versioning, validation artifacts).
- `docs/SURROGATE_BENCHMARK_RESULTS.md` — Fluxion's own surrogate-vs-physics
  latency measurements (the "surrogate overhead vs physics" question for the
  zone-thermal surrogate).
- `src/sim/shading.rs` — overhang/fin shading, vegetation transmittance, and
  the per-surface effective shaded fraction that D1's average-shading feature
  aggregates.
- `docs/specs/phase_4a_mlp_surrogate.md` — the current (zone-thermal) surrogate
  feature schema.
