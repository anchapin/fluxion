# Fins sky-diffuse isotropic route — scoping for the r120-pattern extension (loop r128, 2026-11-11)

**Status:** scoping record, landed in loop round 128. This doc changes no
engine code; the fins slice is sized as a multi-day full-battery round on
the r120 precedent and keeps its queue position for its own round.

## The named slice (from LOOP127)

Round 127 named the next physics-correctness slice after the fins OCR lead
resolved negative: extend the r120 sky-diffuse reduction to shading fins.
On the current tip (`72e0994c`), an overhang already reduces the window's
sky-diffuse gain by its visible-sky share (r120, PR #4686, `f66743c1`),
while a fin pair reduces only the beam component
(`calculate_shaded_fraction`) and leaves its sky-diffuse share at 1.0.
The physics defect: a vertical fin pair geometrically blocks part of the
isotropic sky dome for window points near the edges, so the diffuse
component is over-counted for finned windows. This is a known, verifiable,
testable physics gap — judged per the standing fidelity-first direction,
not by band movement.

## The r120 pattern being extended (verified on the tip)

- **Engine:** `shading::overhang_sky_diffuse_fraction(&WindowArea, &Overhang)`
  (`src/sim/shading.rs`, tip line 336) — isotropic-sky visible fraction by
  Simpson-4096 azimuth × 64×64 midpoint area-grid quadrature; numerics
  mirror the Python twin exactly (scheme-vs-continuum bound 8e-6 vs
  GL-2048×512×512, LOOP120.md); the factor is hoisted out of the hourly
  loops and computed once per orientation at spec-to-surface wiring (the
  #1212/#1385 pattern).
- **Threading:** `sky_diffuse_view_factor` rides the wiring through
  `construction.rs` into `thermal_model_core/mod.rs` +
  `thermal_model_iterative.rs` and is consumed at `solar.rs`'s diffuse
  sites (`diffuse_shgc = window.shgc * 0.9 * sky_diffuse_view_factor`,
  three call sites on the tip).
- **Fins geometry already exists:** `ShadeFin { depth, distance_from_edge,
  side, height }` (`src/sim/shading.rs`, tip line 23) — beam-side only
  today.

## The fins extension (sized)

1. `shading::fins_sky_diffuse_fraction(&WindowArea, fins) -> f64`: the same
   quadrature scheme; per window-area point the fin pair at
   `x = ±(width/2 + distance_from_edge)` blocks an azimuth interval whose
   half-angle follows from the horizontal offset over the fin depth, bounded
   by the fin height via the elevation cut; the share is the window-area
   average of the unblocked azimuth measure. `1.0` when `depth <= 0.0`
   (exact no-device identity, same as the overhang).
2. **Composition semantics when an overhang and fins shade the same
   window** (product vs minimum vs true union of blocked solid angle) must
   be decided from the reference semantics before the engine change — this
   is exactly where the blocked CSE constant enters (below). The decision
   is an engine-semantics call and stays with Alex per standing practice.
3. **TDD route, red-first:** Python twin first mirroring the quadrature
   bit-exactly; Monte-Carlo direction check (uniform sky-dome sampling vs
   the quadrature, agreement inside MC noise); three teeth with predicted
   failing-pin sets (branch erasure → factor forced 1.0, coefficient
   drift, fail-open default); contract pins in
   `tests/shading_geometry_contract.rs` next to the r120 family.
4. **Full battery:** output-diff + section-7 + strict-gate re-record with
   old→new provenance, dated KNOWN_ISSUES row recording any CSE-constant
   discrepancy honestly (exactly as §SOLAR-03 did for the overhang),
   `docs/ashrae_140/mathematical_model.md` section, AGENTS.md +
   test-inventory + ratchet in lockstep, doc gates.

## The blocked reference constant (unchanged from round 127)

The CSE reference constant for a fins sky view factor lives in
`Std140-5-2-CSE-04152019-InputFiles.zip` on the ASHRAE 140
accompanying-files server, still behind the Cloudflare bot challenge
(live probe 2026-11-11 ~11:54Z: `cf-mitigated: challenge`). The in-tree
OSTI/UM Part I carries the 610 overhang block only. Without it, the slice
lands the geometric route with the CSE discrepancy recorded for the
record (r120's precedent: engine geometric value vs
`wnVfSkyDf = 0.39363710370433924` recorded in the model doc), or waits
for the zip if Alex prefers the anchor first.

## Sizing verdict

Multi-day at r120's scale (engine + twin + teeth + full re-record battery
+ docs). Not landed this round; this scoping doc is the round's Item-2
deliverable.
