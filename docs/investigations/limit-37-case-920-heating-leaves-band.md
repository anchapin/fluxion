# LIMIT-37 — investigation history

Narrative history for **LIMIT-37** (Case 920 annual heating left its
published band after the #4347 envelope-reroute phi_m delivery). The
current state of this limitation is the `LIMIT-37` row in
`docs/KNOWN_ISSUES.md`; this file is the provenance behind it.

---

#### LIMIT-37: Case 920 heating moved honestly out of band — in band only while phi_m was dead

- **Movement (2026-10-08, PR #4347 envelope-reroute, rebased onto develop
  95f09286):** strict-gate Case 920 annual heating 3.302 → 2.615 MWh
  against band [3.213, 4.347] (issue #1147 band table) — gap 0 → 15.82 %
  of band midpoint, just under the lower bound. Validator path
  3,613.71 → 2,829.88 kWh heating.
- **Why this is honest movement, not a regression:** Case 920 was in band
  only because the pre-#4347 engine computed the envelope `phi_m` gains
  and then dropped them (the defect documented in §LIMIT-35 §12), which
  under-delivered solar to the high mass and inflated the heating
  request. Delivering the gains through the envelope mass nodes (the fix
  Alex approved 2026-10-08, "go with merge option") brings Case 900
  heating into its band (2.455 → 1.649 MWh) and correspondingly lowers
  920's request below its lower bound. No threshold was widened and no
  output was tuned; both numbers are recorded in
  `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`
  with dated provenance.
- **Cooling moves the same direction but stays under band:** 920 cooling
  0.633 → 1.395 MWh vs [2.189, 2.961] — the below-band gap narrows.
- **Closes when:** a physics decision restores Case 920 heating inside
  [3.213, 4.347] without pushing Case 900 heating back out of
  [1.364, 1.846].
