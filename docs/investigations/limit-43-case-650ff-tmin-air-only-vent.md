# LIMIT-43 — Case 650FF minimum temperature leaves band under the air-only night-vent default
Summary: the mass-side night-vent coupling on the 5R1C sub-step path was removed from the engine default (loop round 87, 2026-10-22, Alex-approved "I approve the vent arm", branch `physics/r87-vent-air-default` on a77be343).
Summary: night ventilation is an air-side exchange — ASHRAE 140 §5.4 / Table 5-160 define the Case 650/950 night-vent fan as outdoor air supplied to the zone, and ISO 13790 §C.4 Eq. C.10 routes ventilation only through the air node; the air-side `h_ve_night` conductance in h_ext (Issues #824/#839) implements exactly that.
Summary: the removed term double-counted the same exchange mass-side: it folded `h_ve_night` into the h3 air↔mass coupling (per-class site `h3_c = h_am_c + h_vent_i*w`, lumped site `h_tr3_arg_i = h_air_mass_i + h_vent_i`) and, pre-round-87, into the legacy hourly `h_vent_mass_zone` site.
Summary: the round-86 energy-balance audit (docs/investigations/2026-10-22-free-float-energy-balance-audit-r86.md) measured the term at 3.1× the ISO air-side exchange on 650FF (−20,469 vs −6,509 MJ/yr) while chilling the mass below freezing overnight; the 9R4C path has run the same air-only routing since §LIMIT-17.
Summary: measured old → new on the strict battery: 650 C 1.226 → 2.270 MWh (gap 58.15 → 35.29 pp, improves); 650FF Tmax 41.89 → 46.11 °C (gap 34.50 → 27.90 pp, improves); 650FF Tmin −17.41 → −16.20 °C — leaves its band [−17.8, −16.7] upward by 0.5 K, honestly re-recorded pass → known_fail (gap 6.09 pp).
Summary: every other strict row is byte-identical, including Case 950 (high-mass 9R4C auto-promote, inert control) and the other free-float cases; `FLX_R86_VENT_AIR_ONLY=0` restores the pre-87 coupling bit-for-bit (Run B verified against the old strict + output-diff baselines).
Summary: checker partition after the in-PR re-record: 9 PASS / 33 KNOWN-FAIL / 0 REGRESSION; closure when the 650FF Tmin row re-enters [−17.8, −16.7] via the R67 surface-pool release-timing / afternoon-stack work (audit doc candidate 2), or is accepted as the honest cost of spec-fidelity per the LIMIT-12 precedent.

## Detail

- Engine change: `src/sim/thermal_model_physics/physics_impl/step_5r1c.rs` —
  `FLX_R86_VENT_AIR_ONLY` (env_bool, default air-only) gates both live sites.
  The first arm at the `h_vent_mass_zone` site alone was byte-neutral on the
  strict battery (structurally unreachable on the coupled-N default); the
  sub-step `h_vent_i` sites are the live coupling the round-86 audit measured.
- The invariant checker needed no change: its receipt field
  `ConductionState::h_vent_mass_zone` is never written by the engine, so its
  mass identity already assumed the air-only algebra; the flip makes the
  receipt truthful rather than widening any ceiling.
- Re-records in the same PR: strict_energy_gate_baseline.json (3 rows, dated
  _doc), data/reference/ashrae140/output_diff_baseline.csv (650/C, 650FF/Tmin,
  650FF/Tmax via --update-baseline), §7 headline pins + re-emitted
  workflow_results.csv. Verified unchanged: §6 (WD100–600), fabric
  (600/900/950), hotloop goldens (Case 600), grid-adapter baseline (Case 600),
  #4536 mass-advance vent contracts (state-level; green under the default).
- History: audit doc §Candidate-experiments item 1; #3983 §66.
