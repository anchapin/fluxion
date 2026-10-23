# LIMIT-44 — Case 800 H/C and Case 640 cooling equal Case 600 in the strict gate (by-construction harness equality)
1. Observation: strict-gate rows `case_800_heating` == `case_600_heating` (6.915 MWh) and `case_800_cooling` == `case_600_cooling` == `case_640_cooling` (2.444 MWh) print bit-identical values in every battery; the `FLX_R91_SOL2ST` flip moved all three together.
2. Mechanism: ASHRAE 140 defines Case 800 as Case 600's geometry, construction, loads and setpoints with high-mass walls and added HVAC (air + radiative panels + reheat); the strict ideal-loads solver drives both to the same setpoints, so the metered annual loads coincide by construction.
3. Case 640 is heating-setback-only (night setback on heating; cooling identical to 600), so its cooling leg equals 600's cooling by the same argument.
4. Classification: by-construction harness equality — the same documentation class as §LIMIT-36 (Case 810 heating == Case 900 heating via the 9R4C auto-promotion). A checksum property, not an engine bug.
5. Evidence: 42/42 strict table + output-diff on 9e61b840 (flag-off bit-identical to the round-91a record); 800 H 6.915 == 600 H 6.915; 800 C 2.444 == 640 C 2.444 == 600 C 2.444.
6. Consequence: under the strict harness the 800 and 640 rows carry no information independent of 600; band attributions for 800/640 follow 600's.
7. History: the equality is stable across the round-71→91 re-record chain in the strict-gate JSON `_doc` history; first formalized 2026-10-23 (loop round 91 flip 2, branch physics/r91-sol2st).

## Detail

The strict ±15 % gate meters annual heating/cooling per ASHRAE 140 case with
an ideal-loads solver: each case's HVAC is sized to hold the published
setpoints. Where two cases differ only in equipment modeled *around* those
setpoints (Case 800's panels/reheat vs Case 600's base system), the solver's
answer is the same energy by definition of the case set, exactly as §LIMIT-36
records for the 810/900 heating pair under the 9R4C auto-promotion. The
values diverge only when a modeling change breaks the shared path (e.g. a
high-mass-only defect would move 800 off 600) — which is precisely what makes
the equality worth watching as a checksum: if `800 != 600` ever prints, the
engine has grown a mass-path asymmetry that the standard's case definitions
do not license.
