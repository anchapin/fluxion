# §7 conformance table (#4276)

Coverage of the Standard 140 §6.2.1 output requirements (in-repo `docs/140_2017_a_20200901.pdf`) for the pinned 18-case §7 set, evaluated from the committed workflow CSV (`data/reference/section7/workflow_results.csv`). Case states per #4276: `conformant`, `incomplete` (ran, missing required outputs), `not implemented`. Per-requirement cells: `report` / `partial` / `missing` / `na`. No score, no percentage (#4277).

Honest gaps this table records (scope decisions, not engine defects):

- 6.2.1.3/.4/.5: the emitted transmitted-solar quantity is the 6.2.1.2.2 unshaded reference window; the shaded-south (610), west (620) and shaded-west (630) annual transmissions are not emitted.
- 6.2.1.6 for Case 960 requires the sun zone (5.2.2.2.7.7.b); the emitter records zone 1.
- 6.2.1.8: emitted days are the GSR contract days (Feb 1, May 4, Jul 14); the addendum Table 6-1 day cells carry unresolved tracked changes (candidate days Jan 4 / Mar 5 / Jul 27), so affected cases hold at `partial` pending a clean copy of the standard.
- Case 970 has no 140-2017 definition; evaluated against 6.2.1.1 only.

| Case | Status | 6.2.1.1.1 | 6.2.1.1.2 | 6.2.1.1.3 | 6.2.1.1.4 | 6.2.1.1.5 | 6.2.1.2.1 | 6.2.1.2.2 | 6.2.1.2.3 | 6.2.1.2.4 | 6.2.1.3 | 6.2.1.4 | 6.2.1.5 | 6.2.1.6.min | 6.2.1.6.max | 6.2.1.6.mean | 6.2.1.7 | 6.2.1.8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 600 | incomplete | report | report | report | report | report | report | report | report | report | na | na | na | na | na | na | na | partial |
| 610 | incomplete | report | report | report | report | report | na | na | na | na | missing | na | na | na | na | na | na | na |
| 620 | incomplete | report | report | report | report | report | na | na | na | na | na | missing | na | na | na | na | na | na |
| 630 | incomplete | report | report | report | report | report | na | na | na | na | na | na | missing | na | na | na | na | na |
| 640 | incomplete | report | report | report | report | report | na | na | na | na | na | na | na | na | na | na | na | partial |
| 650 | conformant | report | report | report | report | report | na | na | na | na | na | na | na | na | na | na | na | na |
| 900 | incomplete | report | report | report | report | report | na | na | na | report | na | na | na | na | na | na | na | partial |
| 910 | conformant | report | report | report | report | report | na | na | na | na | na | na | na | na | na | na | na | na |
| 920 | conformant | report | report | report | report | report | na | na | na | na | na | na | na | na | na | na | na | na |
| 930 | conformant | report | report | report | report | report | na | na | na | na | na | na | na | na | na | na | na | na |
| 940 | incomplete | report | report | report | report | report | na | na | na | na | na | na | na | na | na | na | na | partial |
| 950 | conformant | report | report | report | report | report | na | na | na | na | na | na | na | na | na | na | na | na |
| 960 | incomplete | report | report | report | report | report | na | na | na | na | na | na | na | missing | missing | missing | na | na |
| 970 | conformant | report | report | report | report | report | na | na | na | na | na | na | na | na | na | na | na | na |
| 600FF | incomplete | na | na | na | na | na | na | na | na | na | na | na | na | report | report | report | na | partial |
| 650FF | incomplete | na | na | na | na | na | na | na | na | na | na | na | na | report | report | report | na | partial |
| 900FF | incomplete | na | na | na | na | na | na | na | na | na | na | na | na | report | report | report | report | partial |
| 950FF | incomplete | na | na | na | na | na | na | na | na | na | na | na | na | report | report | report | na | partial |
