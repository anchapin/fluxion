# #4699 parallel-in-time spike — warm-up-as-state-bridge measurement, LOW-MASS class only
> Summary: measurement-only spike for issue #4699 (parallel-in-time
> annual simulation): whether a warm-up on real preceding weather
> bridges the month-boundary state for the low-mass class (Case 600).
> At 14 days the bridge passes both win bands at every measured
> boundary vs the periodic reference; the N-vs-2N self-check is
> validated as conservative. High-mass and ground-coupled classes
> are NOT MEASURED here. Harness: benches/spike_warmup_bridge_chunking.rs.

**Status: measured for the low-mass case class (ASHRAE 140 Case 600). High-mass
(Case 900) and ground-coupled classes: NOT MEASURED in this run — no claim is
made about them. All chunked-mode numbers below are labelled chunked-mode;
nothing here is a sequential-run result presented as one, and no production
code, default, baseline, or tolerance was changed.**

Harness: `benches/spike_warmup_bridge_chunking.rs` (measurement-only,
`harness = false`; registered in `Cargo.toml`). Reproduce with:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release --bench spike_warmup_bridge_chunking
cargo bench --bench spike_warmup_bridge_chunking   # prints the full table quoted below
```

Supervisor re-ran the harness binary at collection: every energy/seam/state
number reproduced identically (only wall-clock seconds jitter).

## Audit finding: no checkpoint/restore exists

The engine exposes **no serialization / checkpoint / restore of full solver
state**. State at a month boundary can therefore only be observed through the
engine's outputs (zone, mass-node, air, wall-surface temperatures). This
harness measures the bridge through those observables; an explicit state
hand-off (the parareal fallback's prerequisite) would first need a
checkpoint/restore API, which does not exist today.

## Method

- Reference A — the sequential annual run **as-is**: Case 600 stepped exactly
  as `Case600Model::simulate_year`. Harness validation: annual heating
  1.5607 MWh and cooling 9.7551 MWh, identical to `simulate_year`; max
  zone-temperature difference vs the production path 0.000 °C. Reference A
  starts the year from the 20 °C initial condition with no warm-up, so its
  own January is a cold-start transient.
- Reference B — a second consecutive annual pass (periodic steady state),
  included to separate "the warm-up bridge fails" from "the as-is reference
  is itself transient in January". B vs A annual: heating −0.0271 MWh,
  cooling +0.0110 MWh.
- Chunked mode: for each boundary (January — exercises the Dec→Jan wrap —
  April, July, October) and warm-up length N ∈ {7, 14, 30} days, simulate
  [N preceding days on the **real preceding weather**, wrapping Dec→Jan]
  [the month], discard the warm-up output, and compare the month portion
  against both references: hourly zone-temperature seam error over the first
  72 h, monthly energy drift, and the N-vs-2N end-of-warm-up state
  self-check (observable state after N days vs after 2N days of warm-up,
  both ending at the same boundary).
- Win band (from the issue): monthly energy drift < 0.5% and hourly
  zone-temperature seam error < 0.2 °C after the first 24 h of the month.

## Results (chunked-mode)

Worst case over the measured boundaries at each warm-up length:

| Warm-up | Worst seam 24–72 h, Apr/Jul/Oct (vs A = vs B) | Worst \|energy drift\|, Apr/Jul/Oct | January vs B: seam 24–72 h / drift | January vs A: seam 24–72 h / drift | Worst N-vs-2N self-check Δ |
|---|---|---|---|---|---|
| 7 d  | 0.1553 °C (PASS) | 0.262% (PASS) | 0.1246 °C / 0.101% (PASS) | 2.9916 °C / 2.910% (FAIL) | 2.2606 °C — NOT CONVERGED |
| 14 d | 0.0080 °C (PASS) | 0.011% (PASS) | 0.0051 °C / 0.004% (PASS) | 3.1111 °C / 3.004% (FAIL) | 0.0966 °C — NOT CONVERGED |
| 30 d | 0.0000 °C (PASS) | 0.000% (PASS) | 0.0000 °C / 0.000% (PASS) | 3.1162 °C / 3.008% (FAIL) | 0.0001 °C — CONVERGED |

Reading of the table:

1. **The warm-up bridge works for low-mass at 14 days.** At N = 14, every
   non-January boundary and January-vs-B sit at least an order of magnitude
   inside both win bands (seam ≤ 0.008 °C, drift ≤ 0.011%).
2. **January vs reference A "fails" at every warm-up length — and the failure
   is the reference's, not the bridge's.** Reference A's January is its own
   cold-start transient from the 20 °C initial condition; a chunk warmed up
   on December weather starts from the periodic state and cannot (and
   should not) reproduce that transient. Against the periodic reference B,
   the same January chunks pass with seam 0.0051 °C / drift 0.004% at 14 d
   and exactly 0.000 at 30 d. A production chunked mode must therefore
   define its January contract against a warmed-up sequential reference
   (the ASHRAE 140 validator path already runs a 14-day warm-up before the
   annual run), not against the cold-start `simulate_year`.
3. **The N-vs-2N self-check is conservative — it never cried wolf in
   reverse.** At 7 d it reports NOT CONVERGED at every boundary, and at
   14 d it still flags July (Δ = 0.0966 °C in mass/wall state) even though
   July's measured seam error is 0.0000 °C — because July zone temperature
   is clamped at the 27 °C cooling setpoint, the zone signal hides mass-state
   divergence that the self-check catches. In all 12 measured cells the
   self-check never declared CONVERGED while the seam failed against the
   periodic reference; its disagreements with the verdict are all in the
   safe direction (flagging warm-ups whose seam already passes). It is
   usable as a production sufficiency criterion, biased toward longer
   warm-ups.
4. **Honest speedup arithmetic.** A 14-day warm-up makes 12 chunks cost
   12 × (30 + 14)/365 = ×1.447 sequential-years of total work. Ideal
   parallel speedup: ×2.77 on 4 workers, **×5.53 on 8**, ×11.06 on 16 —
   above the issue's ≥4× net criterion at ≥8 workers for this case class.
   (Measured: reference year 0.087 s for this 5R1C case; chunk wall clocks
   0.009–0.015 s. Absolute times are case-specific; the ratios are the
   result.)

## Verdict by case class

- **Low-mass (Case 600): WIN, with the January-reference caveat.** Warm-up on
  real preceding weather bridges the boundary state at 14 days, inside both
  win bands at every boundary measured, provided the comparison reference
  is the warmed-up/periodic sequential run. 30 days is exact to printed
  precision.
- **High-mass (Case 900): NOT MEASURED.** Thermal time constants are longer;
  the 14-day result must not be extrapolated. The conservative self-check
  is the tool designed for exactly this class and is validated above as
  safe-direction.
- **Ground-coupled: NOT MEASURED.** Still the expected hardest class
  (ground state converges over weeks–months); the self-check at least
  gives a production-usable "warm-up too short" signal, and the missing
  checkpoint/restore API (audit finding above) is the blocker for any
  explicit state hand-off design.

## Parareal fallback

Not implemented and, for the low-mass class, not needed: the warm-up bridge
already meets the win band, so waveform-relaxation iterations would buy
nothing here. Parareal remains a design option only for classes where the
bridge fails (candidate: ground-coupled), and it is blocked on the missing
checkpoint/restore API in any case. No follow-up is filed by this spike;
whether to measure Case 900 / ground-coupled next, or to design a
production chunked mode, is a physics call for Alex from this report.

## Provenance

Worker run `fluxion-4699-1794263910` (balanced, MiniMax-M2.7, 60/60 steps)
built the harness and produced the measurements; it hit the step cap while
drafting this report, so the supervisor completed the report from the
worker's quoted harness output and verified it by independently re-running
the harness binary (all numbers identical), `cargo fmt --check` (clean),
and `python3 scripts/check_test_inventory_drift.py --no-verify` (exit 0;
no tests added, no ratchet change).
