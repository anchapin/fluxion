# PCM Test-Box Reference Data — Provenance

## Scope

This directory holds the reference data for the PCM (Phase Change Material) test-box
sub-suite of issue #3986 / PR-B (#4118). The test-box is a rectangular enclosure
with an isothermal wall, used to validate enthalpy-method thermal mass calculations
in `fluxion`.

## What lives here (planned)

- `rt27_solid_fraction.csv` — ONE experimental reference curve of solid fraction
  vs. time at constant wall temperature for paraffin Rubitherm RT27. **Not yet
  present**; see BLOCKER below.

## Nominal RT27 property values (used by `PhaseChangeMaterial::rt27()`)

These are the values hard-coded in `src/physics/phase_change_material.rs::rt27()`.
They are the **manufacturer-published nominal values** for Rubitherm RT27, the
paraffin identified in `docs/research/mazzeo-rt27-pcm-spike.md` as the closest
available open-access PCM with the test-box geometry in scope.

| Property | Value | Source |
|---|---|---|
| Melting range (T_solidus, T_liquidus) | (25.0, 28.0) °C | Kumari & Ghosh 2025 (IntechOpen, CC BY 4.0, DOI 10.5772/intechopen.1006640), §"Materials"; also cited in Portable Cold Storage Technologies Review (Scribd, ref_1 of `docs/research/mazzeo-rt27-pcm-spike.md` §2.1) |
| Latent heat of fusion | 184_000 J/kg | Rubitherm RT27 datasheet, the canonical value cited across the PCM literature (ref_1: 184 kJ/kg; ref_5: 25-28 °C range; ref_8 cites 179 kJ/kg as an alternative which would change the apparent_cp band height by 2.7%) |
| Specific heat capacity (representative c_sensible) | 2000 J/(kg·K) | Average of Rubitherm-published solid (1800 J/(kg·K)) and liquid (2400 J/(kg·K)) values; the apparent-heat-capacity method (`apparent_cp_j_per_kg_k`) uses this constant outside the melting band |
| Density (representative ρ) | 800 kg/m³ | Average of Rubitherm-published solid (880 kg/m³ at 15 °C) and liquid (760 kg/m³ at 40 °C) values |
| Thermal conductivity (not used by skeleton; documented for PR-B+1) | 0.2 W/(m·K) | Rubitherm datasheet, both phases |

The single-value specific heat and density are deliberate: the skeleton's apparent-heat-capacity method needs ONE representative c_sensible (used outside the melting band) and ONE representative density (used for energy conversion); PR-B+1 may add temperature-dependent c_p(T) and ρ(T) once the real reference data lands.

## BLOCKER on experimental curve (Refs #3986 spike)

`rt27_solid_fraction.csv` is **NOT present** in this directory. The Mazzeo et al.
RT27 PCM test-box reference data is not freely obtainable within the spike window
(time-boxed ≤2h). See `docs/research/mazzeo-rt27-pcm-spike.md` for the full
survey.

Per issue #4118 BLOCKER branch:
- `PhaseChangeMaterial::rt27()` constructs with the nominal values above (real
  manufacturer data, not a placeholder).
- `PCMTestBox::solid_fraction_at_time(t)` returns `None` when no reference data
  is loaded (sentinel per the BLOCKER branch in #4118).
- The skeleton behavior test asserts this sentinel behavior, not a numerical
  match against a curve.

## Unlock criteria (PR-B+1)

PR-B+1 unlocks when any of:
1. The Mazzeo group releases the RT27 reference data as a public supplement.
2. A free mirror of the Kraiem 2019 thesis surfaces (Tebessa repository,
   ResearchGate, academic torrents).
3. The user acquires a non-redistribution license and provides the raw data file.
4. The user substitutes a different open-access PCM test-box curve (Assis et al.
   2009 spherical shell, Bareiss & Lau 2014, Shmueli et al. 2010, etc.).

When the data lands, the skeleton `solid_fraction_at_time` sentinel test is
replaced by `solid_fraction_at_time_returns_some_after_reference_data_loaded`
and the match tolerance is set to ±2% absolute (per the issue #4118 success
criteria).
