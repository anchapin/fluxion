# LIMIT-36 — investigation history

Narrative history for **LIMIT-36** (Case 810 annual heating below its
published band after the #4336 9R4C mass coupling). The current state of
this limitation is the `LIMIT-36` row in `docs/KNOWN_ISSUES.md`; this file
is the provenance behind it.

---

#### LIMIT-36: Case 810 heating moved honestly out of band — in band before the coupling, below after

- **Movement (2026-10-08, PR #4336 merge run):** strict-gate Case 810
  annual heating 3.823 → 2.455 MWh against band [3.357, 4.543] (issue
  #1147 band table) — from mid-band to 26.9 % under the lower bound. The
  movement is the #4336 mass-coupling fix doing what §LIMIT-35 §9 measured:
  the 9R4C mass network now charges from the conditioned air, which reduces
  the over-requested heating everywhere on the 9R4C path. Case 900's
  residual and Case 810's under-band are two views of the same correction;
  neither number was tuned.
- **Why 810 tracks 900 exactly:** the Case 810 `CaseSpec` is built from
  `CaseBuilder::case_900_baseline()` with only `case_id`, `description` and
  `hvac_equipment` (a heat pump) changed
  (`src/validation/ashrae_140_cases.rs`, `case_810_comprehensive_hvac`).
  The strict gate meters the ideal-load heating demand, so 810's metric has
  equaled 900's before (both 3.823 MWh pre-coupling; earlier still, both
  1.633 MWh per the #3572 gate notes) and after the fix (both 2.455 MWh).
  The identical value is by construction of the spec, not a metering error —
  but it also means this row carries no information independent of
  §LIMIT-35 until the equipment path (COP/EER delivery) enters the metered
  quantity.
- **Closes when:** a physics decision on the 9R4C coupling (see §LIMIT-35
  §10 — the metering-basis refinement is falsified as the residual driver,
  so 810's drop is the coupling's genuine magnitude effect) restores 810
  inside [3.357, 4.543] without re-inflating Case 900 out of
  [1,170, 2,040]. Options under that decision include revisiting the
  coupling variant, the window-solar suspect (§LIMIT-35 §10 lever C), or
  accepting the 800/810-family band as stale relative to the corrected
  engine — that acceptance is Alex's call, never a bound change made
  silently.
