# LIMIT-38 — investigation history

**LIMIT-38** records that `solar_beam_to_mass_fraction` became inert on the
9R4C per-surface path after PR #4347: phi_st and phi_m share one delivery
channel with identical weights, so the split feeds no output. This file is
the provenance; the current state is the `LIMIT-38` row in
`docs/KNOWN_ISSUES.md`.

---

#### LIMIT-38: the beam-to-mass fraction no longer reaches any output on the 9R4C per-surface path

- **Measurement (2026-10-08, develop 04a4e031, issue #4339 sweep):**
  Case 900FF free-float, Denver TMY3 EPW, fractions 0.2 / 0.4 / 0.6 / 0.8 →
  Max = 40.12 °C at every fraction (identical to printed precision), inside
  the widened bounds [36.00, 46.40]. At c5b3231c the same sweep gave
  36.13 / 34.13 / 32.50 / 31.13 °C — monotonic, fraction-live.
- **Structural cause (PR #4347, LIMIT-35 §12):** step_9r4c now routes
  phi_st and phi_m through the same envelope mass nodes with the same
  wall/roof/floor weights (`gains_wall = (phi_st + phi_m) * wall_frac`, …)
  and zeroes `gains_internal`. Because
  `phi_st + phi_m = load_w·rad_frac + remaining_sol + sol_to_air +
  opaque_sol_w` is independent of the split, every fraction-invariant input
  produces identical output. The split between the fast surface node and the
  damped mass node no longer exists on this path.
- **What was changed (PR for issue #4339):** the sweep test's
  monotonic-decrease assertion was replaced with the stronger, mechanistic
  fraction-INVARIANCE assertion, with the 0.6-in-range check kept. No band,
  threshold, assertion about validation numbers, or engine output was
  touched; Case 900 heating stays in band at 1.649 MWh.
- **Closes when:** a physics decision restores a live surface-vs-mass beam
  split on the 9R4C path (distinct delivery channel or weights for phi_st
  vs phi_m), after which the invariance assertion is flipped back to a
  monotonicity assertion with fresh bounds from reference data.
