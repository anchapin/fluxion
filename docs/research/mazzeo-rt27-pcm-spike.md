# Mazzeo RT27 PCM Test-Box Reference-Data Spike — Issue #3986 / #4118

## 7-Line Summary

Mazzeo et al. RT27 PCM test-box reference data is **NOT freely obtainable in the spike window**. Search across Google Scholar, ResearchGate, ScienceDirect, MDPI, IntechOpen, preprints.org, and direct author-homepage traces returned **no open-access Mazzeo-group PCM test-box paper** with tabulated solid-fraction-vs-time curves. Closest geometry match (Kraiem thesis — RT27 in rectangular cavity heated isothermally at left wall) is paywalled at the Tebessa university repository. **Decision: BLOCKER per issue #4118 scope → PR-B lands as skeleton-only** (PhaseChangeMaterial struct + pcm_test_box harness + empty reference data + behavior test asserting harness constructs). Real PCM physics deferred to PR-B+1 once a licensable RT27 reference curve is identified.

## 1. Goal

Find ONE experimental reference curve from Mazzeo et al. (paraffin RT27, solid fraction vs. time at constant wall temperature) that is freely obtainable (open-access, public repository, or licensed redistribution) within a ≤2h spike window. Tabulate as `tests/reference_data/pcm_test_box/rt27_solid_fraction.csv`; document provenance.

## 2. Sources Surveyed

### 2.1 Open-access candidates

| Source | URL | Status | Notes |
|---|---|---|---|
| Kumari & Ghosh 2025 (IntechOpen) | `intechopen.com/chapters/1006640` | **Reachable** (CC BY 4.0) | RT27 in double-pipe heat exchangers; **wrong geometry** — double-pipe vs. the rectangular test-box called for in the spike scope. Could be a fallback but not what ADR-0017 referenced. |
| Lo Brano et al. 2014 (preprint PDF mirrors) | ScienceDirect, ResearchGate | **403** | RT27 latent heat storage, ~101 citations; abstract accessible, full PDF paywalled. |
| Assis et al. 2009 (numerical/experimental) | `sciencedirect.com/...` | Behind paywall | Spherical shell geometry; not rectangular cavity. Not Mazzeo, not RT27. |
| Shmueli et al. 2010 (open numerical benchmark) | Multiple mirrors | Reachable | RT27 properties only; no experimental solid-fraction curve. |

### 2.2 Mazzeo-group publications

Searched `Domenico Mazzeo` (Politecnico di Milano) + `RT27`, `phase change material`, `paraffin`, `shell and tube`, `latent heat`, `solidification`. Results:

- ResearchGate author page `researchgate.net/profile/Domenico-Mazzeo` — **403 forbidden** (anti-scraping).
- Google Scholar "Mazzeo" + "RT27" — **0 direct hits** for a Mazzeo RT27 PCM test-box paper. Mazzeo's PCM publications appear to be on shell-and-tube heat exchangers and PV-coupled systems, **not** on the rectangular-cavity test-box geometry called for in the spike.
- `research_notes/building-simulation-solvers-20260925-1507 §2, §8` (referenced in `docs/adr/0017-equation-based-dae-teacher-architecture.md` §8) — **does not exist in this repo**; the file may have been in a previous research workspace that wasn't committed.

### 2.3 Kraiem thesis (closest geometry match)

`Kraiem 2019 (University of Tebessa, Algeria)` — "Numerical study of the melting process of a phase change material (RT-27) in a rectangular cavity heated by natural convection." Geometry: **rectangular cavity, isothermal wall**. Setup matches the spike scope exactly.

- `oldspace.univ-tebessa.dz` — server reachable, but the PDF bitstream URL returned **404** during the spike window.
- No open-access mirror found on arXiv, ResearchGate, or Google Scholar.

## 3. Conclusion

The Mazzeo RT27 PCM test-box curve referenced in the spike scope is **not freely obtainable** within the spike window. Two structural factors limit the search:

1. The reference data is in the Mazzeo group's paywalled publications (likely the source of ADR-0017's mention).
2. The closest geometry-matching alternative (Kraiem thesis) is hosted on an institutional repository whose PDF bitstreams are unreliable from outside Algeria.

## 4. Decision (per #4118 scope, BLOCKER branch)

The issue #4118 body explicitly defines this outcome:

> If Mazzeo RT27 data is not freely obtainable in the spike window, file this comment on #3986:
> ```
> BLOCKER: PCM test-box reference data not obtainable in spike window.
> Deferring PR-B to a follow-up; current PR-B stub will only contain
> the PhaseChangeMaterial skeleton + test-box harness with empty reference data.
> ```

**PR-B lands as a skeleton-only PR**:
- `src/physics/phase_change_material.rs` — `PhaseChangeMaterial` struct with name, melting range, latent_heat, density, sensible-heat coefficient. `enthalpy(T) -> J/kg` and `apparent_cp(T) -> J/(kg·K)` stub methods that return sensible defaults (no real phase-change transition yet).
- `src/physics/pcm_test_box.rs` — one-zone, one-PCM-layer harness with construction + a placeholder `solid_fraction_at_time(t) -> Option<f64>` method that returns `None` until real reference data lands.
- `tests/all_tests/teacher_validation_pcm_box.rs` — behavior test asserting (a) `PhaseChangeMaterial::rt27()` constructs, (b) `PCMTestBox::new(rt27())` constructs, (c) `solid_fraction_at_time(0.0)` returns the documented sentinel (`None` when no reference data is loaded), (d) `apparent_cp(t)` at the melting midpoint returns the latent-heat-band value per the apparent-heat-capacity method.
- `tests/reference_data/pcm_test_box/PROVENANCE.md` — documents this spike and the blocker.

The skeleton creates the public surface (`PhaseChangeMaterial`, `PCMTestBox`, `solid_fraction_at_time`, `apparent_cp`) that PR-B+1 will fill in once a licensable RT27 reference curve is identified.

## 5. Follow-up criteria (PR-B+1)

PR-B+1 unlocks when any of:
1. The Mazzeo group releases the RT27 reference data as a public supplement.
2. A free mirror of the Kraiem thesis surfaces (Tebessa repository, ResearchGate, academic torrents).
3. The user acquires a non-redistribution license and provides the raw data file.
4. The user substitutes a different open-access PCM test-box curve (Assis 2009 spherical shell, Bareiss&Lau 2014, etc.) — in which case `tests/reference_data/pcm_test_box/PROVENANCE.md` is updated to cite the substitute.

When that lands, the skeleton behavior test `solid_fraction_at_time_returns_some_after_reference_data_loaded` replaces the `returns_none_sentinel` test.
