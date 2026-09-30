# Synthetic series - NOT ASHRAE 140 reference data

synthetic: true

The two CSVs here are **generated fixtures, not program results**, and must never be
used as a validation reference or as the basis of a scorecard band.

They are physically impossible as simulation output:

- `series_195.csv`: `zone1_heating > 0` and `zone1_cooling > 0` in all 8760 hours of
  every case, with a constant `peak_load` of 3506.9. A zone cannot heat and cool
  simultaneously on one side of a deadband.
- `series_800.csv`: hour 1 of case 800 is `800,1,16.1,82.8,83.2,16.1,82.4,82.8,331.2`,
  i.e. 82.8 W of heating and 83.2 W of cooling in the same zone in the same hour.

They are kept only so the defect they caused stays auditable. `data/reference/ashrae140/`
is for transcribed standard values with a per-metric citation; nothing in this directory
qualifies.

## What replaces them

`data/reference/ashrae140/annex_b8_section7_statistics.json` carries the real
ANSI/ASHRAE Standard 140-2023 Informative Annex B8 Section B8.1 example-result
statistics for Section 7 Cases 195-995 and 600FF-980FF, transcribed from
Std140_TF_Results.pdf (TESS, 19-Aug-2024), Tables B8-1 through B8-5.

Read the `NOT_ACCEPTANCE_CRITERIA` field in that file before using any number in it
as a gate. The standard explicitly disclaims the example-result spread as acceptance
criteria.

## Why there is no hourly replacement

Annex B8 publishes annual loads, annual peaks, monthly loads, monthly peaks and
free-float temperature extrema. It does **not** publish an hourly 8760-value reference
series for these cases; the only hourly content is a handful of single-day figures
(B8-H1 through B8-H38) presented as charts. Any code path that compares fluxion's
hourly output against a published hourly reference for Cases 195-470 or 800-810 has
no source to compare against, and that is a scope question, not a missing file.
