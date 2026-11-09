# Boundary-condition precompute audit (fluxion #4689, spike)

> Summary: audit instrument for the boundary-precompute spike —
> relaxed-atomic recomputation counters in `src/sim/boundary_counters.rs`
> (off by default, counting only, bit-identical outputs on/off),
> instrumented at the solar, shading, sky-temperature, longwave and
> exterior-convection computation sites. Measured on full-year ASHRAE
> 140 Case 600 and Case 900 validation runs: every exercised quantity
> is already computed once per recorded hour per surface/zone — small
> integer multiples per hour, with no sign of per-Newton-iteration
> recomputation. Verdict: KILL for the precompute prototype on this
> evidence (see §3 for the two unexercised-site caveats).

## 1. Instrument

- `src/sim/boundary_counters.rs`: six quantities (solar_incident,
  shading_factor, sky_temperature, longwave_coefficient,
  longwave_network, exterior_convection), relaxed-atomic counters,
  process-global master switch defaulting OFF, no env var. Unit tests
  7/7 (stateful tests serialise on a module lock — the counters are
  process-global and parallel tests otherwise corrupt each other).
- Instrument call sites (counting only, no value changes):
  src/sim/solar.rs, src/sim/shading.rs, src/sim/longwave_exchange.rs,
  src/sim/exterior_boundary.rs, src/physics/exterior_convection.rs,
  and the thermal-model step paths (step_dispatcher.rs, step_9r4c.rs,
  zone_balance.rs, invariant_checker.rs).
- Audit harness (temporary, not shipped): ran each case twice through
  `ASHRAE140Validator::validate_single_case_with_diagnostics`, counters
  OFF then ON, asserting every reported metric is bit-identical
  (to_bits) between the two runs. The assertion passed for both cases.

## 2. Measured counts (supervisor-run, base 711cbbf7 + this diff)

Per recorded hour (8,760 recorded hours per annual run):

| Quantity | Case 600 total | Case 600 /h | Case 900 total | Case 900 /h |
|---|---|---|---|---|
| solar_incident | 43,800 | 5.0 | 70,080 | 8.0 |
| shading_factor | 0 | 0.0 | 0 | 0.0 |
| sky_temperature | 8,760 | 1.0 | 8,760 | 1.0 |
| longwave_coefficient | 52,560 | 6.0 | 0 | 0.0 |
| longwave_network | 0 | 0.0 | 0 | 0.0 |
| exterior_convection | 61,320 | 7.0 | 17,520 | 2.0 |

## 3. Reading of the counts

- Sky temperature: exactly 1.0/h on both cases — computed once per
  recorded hour, full stop.
- Solar incident (5.0/h and 8.0/h) and exterior convection (7.0/h and
  2.0/h) scale with surface count: once per hour per surface, not per
  solver iteration. Longwave coefficients on Case 600: 6.0/h (per
  surface), likewise once per hour.
- If any of these were recomputed inside Newton/BDF iterations, the
  per-hour counts would be large and load-dependent; they are small
  exact integers instead. On the paths these two cases exercise, the
  premise of the spike — boundary quantities recomputed inside the
  timestep/Newton loop — does not hold.
- Caveats (unexercised sites, not evidence of absence): shading_factor
  and longwave_network counted 0 on both cases, and
  longwave_coefficient counted 0 on Case 900 — those instrumented
  sites are not on the validator path for these cases (no shading
  devices in 600/900; the network evaluation used here flows through a
  different site). A workload with window shading and a multi-zone
  longwave network would be needed before declaring those two
  quantities once-per-hour as well.

## 4. Verdict

Kill criteria from the issue: "audit shows boundary quantities are
already computed once per hour (say so and close)". For every quantity
the audit exercised, that is what the counts show. **Verdict: KILL**
the per-surface precompute prototype as motivated by recomputation
inside the solver loop; do not build it on this evidence. The counters
land as a permanent, off-by-default audit instrument so a future
workload (shaded windows, multi-zone) can re-run the same audit in
minutes. Closing the issue is Alex's call, with the §3 caveats noted.

## 5. Verification

- `cargo test -p fluxion --lib boundary_counters` — 7 passed
  (supervisor added the serialisation lock described in §1).
- Audit harness (Case 600 + 900, counters off vs on) — passed,
  including the bit-identity assertion on all reported metrics;
  harness deleted after measurement, counts recorded above.
- Test inventory regenerated; drift gate re-run (see PR).
