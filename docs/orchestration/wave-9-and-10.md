# Wave 9 + Wave 10 Re-Run Plan

> Tracking issue: [#1508](https://github.com/anchapin/fluxion/issues/1508)

## Background

The wave orchestration run on 2026-07-10/11 completed Waves 1–8 of the prior
10-wave plan. Waves 9 and 10 were scoped but never implemented because the run
hit its time budget. Four issues were left without work-in-progress:

| Issue | Title | Current wave (this orchestration) | Status |
|-------|-------|-----------------------------------|--------|
| [#1457](https://github.com/anchapin/fluxion/issues/1457) | ASHRAE 140 Case 600 series — 16/27 cooling tests still failing | **Wave 2** | scheduled |
| [#1435](https://github.com/anchapin/fluxion/issues/1435) | `TryFrom<IdfFile>` for `SimulationSchemaV1` (design §4.3) | **Wave 3** | scheduled |
| [#1434](https://github.com/anchapin/fluxion/issues/1434) | NAPI exporters for OSM/gbXML/FMI | **Wave 3** | scheduled |
| [#1432](https://github.com/anchapin/fluxion/issues/1432) | OSM writer: emit `OS:Thermostat` for setpoint round-trip | **Wave 4** | scheduled |

Wave assignments are sourced from the live plan at
`/home/alex/Projects/worktrees/wave-state.json` (4-wave re-run, started
2026-07-11).

## Ordering notes

Issue #1508 proposed that **#1435** (IDF `TryFrom`) land *before* **#1457**,
treating the schema-parsing work as a potential prerequisite for re-parsing
the ASHRAE 140 reference IDF files. In the current orchestration **#1457**
was assigned to **Wave 2** and **#1435** to **Wave 3**, so #1435 is scheduled
*after* #1457 rather than ahead of it. This deviates from the issue's preferred
ordering; the rationale is to give the highest-complexity item early, focused
attention. **#1457 alone is the most complex item** in this group — it is the
original wide-gap physics issue (16/27 Case 600 cooling tests failing) that
AGENTS.md §"Validation Strategy / Phase 1: Module Isolation" flags as blocked
on individual module E+ reference tests. #1435 and #1434 are grouped together
in Wave 3, and #1432 closes out the run in Wave 4.

## Acceptance criteria (from #1508)

- [x] All four issues have a wave assignment in the current orchestration (see table above and `wave-state.json`)
- [ ] #1457 reduces the failure count from 16/27 toward ASHRAE 140-band compliance
- [ ] #1435 lands first if schema-doc linking turns out to be a hard prerequisite for #1457

## Live state

The authoritative wave plan lives in `wave-state.json`. This document is a
snapshot as of 2026-07-11; consult the JSON file for real-time status updates.
