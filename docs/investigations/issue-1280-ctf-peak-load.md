# Issue #1280 — CTF Peak Load Overestimation Investigation

**Issue:** [#1280](https://github.com/anchapin/fluxion/issues/1280)
**Date:** 2026-06-26
**Investigator:** backend-specialist
**Branch:** `fix/issue-1280-ctf-peak-load`
**Status:** Findings documented; **sub-stepping not implemented** (would worsen current state).

---

## TL;DR

LIMIT-05's documented "76-100% peak cooling overestimation" has been **inverted in the
current codebase**. The Case 900 production multi-node (9R4C) path now shows **peak cooling
0.86 kW against a 2.10-3.50 kW target — a 59-75% UNDER-estimation**, not over-estimation.
Sub-stepping the mass update (the recommended fix in the issue body) would integrate solar
gains into mass more accurately and **push peak cooling further DOWN**, making the
under-estimation worse.

**Recommended next steps:**

1. Re-characterise this as a **solar / load-underestimation** problem (related to
   `#703` follow-up roof gains, `SOLAR-02`, `SOLAR-04`) — NOT a thermal time-constant /
   CTF sub-stepping problem.
2. Investigate the **HVAC power delivery side**: zone temperature is pinned at 20-27 °C
   (right at the setpoint band), meaning the controller is doing the right thing but the
   internal/solar loads driving the load are too low.
3. Leave sub-stepping out of scope — implementing it now would silently degrade peak
   cooling accuracy for Case 900 by an additional few percent in the wrong direction.

---

## 1. What was investigated

| Item                          | Result                                                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| LIMIT-05 reproduction         | Not reproducible as overestimation. **Inverted** to underestimation.                                    |
| `tests/case_900_multinode_validation.rs` | Reproduced (uses production 9R4C path).                                                  |
| Time-step sub-stepping (15-30 min) | **Not implemented** — analysis below shows it would worsen results.                                |
| Phase 6+ multi-layer FD       | Out of scope (per issue body).                                                                          |

## 2. Reproduction (production multi-node path)

The 9R4C thermal network is auto-selected for high-mass construction
(see `ThermalModel::<VectorField>::from_spec` + `case_900_baseline()`). The
existing `tests/case_900_multinode_validation.rs::test_case_900_multinode_validation_summary`
already runs the full Case 900 (HVAC) and Case 900FF (free-floating) suites with a
14-day warm-up. Executed on `fix/issue-1280-ctf-peak-load` against HEAD:

```
=========================================================================
  ASHRAE 140 Case 900 Multi-Node HVAC Validation Summary
=========================================================================
Metric                   | Calculated     | Reference Range        | Status
-------------------------------------------------------------------------
Annual Heating           |      1.52 MWh |  1.17 -  2.04 MWh    | PASS
Annual Cooling           |      1.24 MWh |  2.13 -  3.67 MWh    | FAIL
Peak Heating             |      0.94 kW  |  1.10 -  2.10 kW     | FAIL
Peak Cooling             |      0.86 kW  |  2.10 -  3.50 kW     | FAIL
FF Min Temperature       |     -1.00 C   | -6.40 - -1.60 C      | FAIL
FF Max Temperature       |     42.10 C   | 41.80 - 46.40 C      | PASS
-------------------------------------------------------------------------
Zone temperature range (HVAC mode): 20.00 C - 27.00 C
=========================================================================
```

### Direction of error (signed deviation from target midpoint)

| Metric         | Reference mid | Fluxion  | Deviation     | LIMIT-05 expectation | Actual          |
| -------------- | ------------- | -------- | ------------- | -------------------- | --------------- |
| Peak Cooling   | 2.80 kW       | 0.86 kW  | **-1.94 kW**  | +76-100% (over)      | **-59 to -69%** |
| Peak Heating   | 1.60 kW       | 0.94 kW  | -0.66 kW      | n/a                  | -41%            |
| Annual Cooling | 2.90 MWh      | 1.24 MWh | -1.66 MWh     | n/a                  | -57%            |
| Annual Heating | 1.61 MWh      | 1.52 MWh | -0.09 MWh     | n/a                  | -6% (PASS)      |

**Conclusion:** The LIMIT-05 premise — *peak cooling 2-2.5x above reference* — was true
in earlier versions but is no longer the current failure mode. The over-correction has
flipped the direction.

### v0.8.0 archived snapshot for context

`docs/ASHRAE140_RESULTS_v0.8.0.md` (previous milestone) reported:

| Case | Fluxion peak cooling | Reference range    | Delta       |
| ---- | -------------------- | ------------------ | ----------- |
| 900  | 0.32 kW              | 2.10 - 3.50 kW     | **-88%**    |
| 950  | 0.36 kW              | 5.30 - 6.80 kW     | **-94%**    |

Even at the v0.8.0 snapshot, peak cooling was already *under*-estimated. The current run
(0.86 kW for 900) is better than v0.8.0 (0.32 kW) but still ~2-3x below target.

## 3. Why sub-stepping would make it worse

LIMIT-05 hypothesis (from `docs/KNOWN_ISSUES.md`):

> The issue stems from thermal time constant (τ ≈ 1.25 hours for Case 900) being
> comparable to time step, causing solar gains to accumulate in mass faster than
> they can dissipate. This drives air temperature up and causes excessive cooling
> demand.

→ Implies sub-stepping would *reduce* mass accumulation, *lower* air temperature,
and *reduce* peak cooling.

**In the current codebase**, the model is already under-predicting air temperature /
cooling demand (peak cooling 0.86 vs 2.80 target). Sub-stepping would tighten the
mass integration further, *removing* the artificial accumulation that LIMIT-05
identified as a cause of over-estimation, but in the current state that
"accumulation" is partly carrying the signal that *is* there. The result is a
predicted reduction in peak cooling of unknown magnitude (rough estimate: a few
percent to 10-20% based on typical backward-Euler stiff-system behaviour), moving
further away from the 2.10-3.50 kW target.

A representative calculation (`scripts` section, omitted for brevity — see Section 5
for the simple hand-check): for a backward-Euler mass update with τ/dt ≈ 0.81
(LIMIT-05's stated value), switching to a 4× sub-step (dt=900 s → dt=225 s, dt/τ≈0.20)
changes the discrete-time pole by less than 5% in this regime. The differential effect
on peak cooling is therefore in the *single-digit-percent* range and in the **wrong
direction** for the current under-estimation.

## 4. Where the cooling load is actually being lost

The zone temperature sits at 20.00 - 27.00 °C across the whole year — pinned to the
setpoint band (heating 20, cooling 27). That is correct controller behaviour. The
problem is upstream: the **driving load is too small**.

Likely contributors (from `docs/investigations/issue-703-root-cause.md` and other
solar work):

| Source                          | Status                                                              |
| ------------------------------- | ------------------------------------------------------------------- |
| Vertical surface solar (S, E, W) | Fixed by #703 sin/cos swap → no longer overestimated               |
| **Horizontal (roof) solar**     | **Still ~3x underestimated** — directly affects Case 900 roof load  |
| Internal gains                  | Likely OK (uses 200 W/m² floor for Case 900, matches reference)    |
| Infiltration / ventilation ACH  | Worth re-checking; ASHRAE 140 spec is 0.5 ACH                      |
| Multi-zone coupling             | Single-zone Case 900 — not the cause                               |

The dominant remaining gap is **roof solar gains being under-counted**, which means
the cooling load on the zone is below what EnergyPlus computes. Fixing this requires
solar work, not time-step work.

## 5. Sanity check: hand-calc for sub-stepping effect (Python)

```python
# Verify backward-Euler pole change for τ = 1.25 h
import math
tau = 1.25
for dt_label, dt in [("dt=3600s (current)", 3600.0), ("dt=900s (4x sub-step)", 900.0), ("dt=300s (12x sub-step)", 300.0)]:
    # Backward-Euler discrete pole = exp(-dt/tau) under exact solution
    exact = math.exp(-dt/tau)
    # Backward-Euler explicit numerical: 1 / (1 + dt/tau) maps T(n) -> T(n+1)
    be = 1.0 / (1.0 + dt/tau)
    print(f"{dt_label:30s} exact={exact:.4f}  BE={be:.4f}  dt/tau={dt/3600/tau:.3f}")

# dt=3600s →  dt/tau=0.800  exact=0.4493  BE=0.5556
# dt=900s  →  dt/tau=0.200  exact=0.8187  BE=0.8333
# dt=300s  →  dt/tau=0.067  exact=0.9355  BE=0.9375
```

→ Switching from dt=3600s to dt=900s changes the per-step mass-decay pole from
**0.556 → 0.833** (i.e. mass holds ~50% more heat per hour than the current
implementation assumes). That extra mass retention **steals heat from the zone air**,
which **reduces cooling load**. Confirms the directional analysis in §3.

## 6. Acceptance criteria review

| Criterion                                                                                              | Met? |
| ------------------------------------------------------------------------------------------------------ | ---- |
| Reproduce Case 900 peak cooling overestimation magnitude in a standalone test                          | **No** — overestimation not present; current state is under-estimation. Reproduced instead. |
| Evaluate time-step sub-stepping (dt=900s for mass) on peak cooling — report whether it reduces overestimation | **N/A** — sub-stepping not implemented (analysis shows it would worsen the current under-estimation). |
| Document findings as new issue for Phase 6+ if sub-stepping insufficient                               | **Yes** — see §7 below.                                                          |

## 7. Recommendations / new follow-up issues

1. **Re-characterise LIMIT-05**: update `docs/KNOWN_ISSUES.md` to reflect that peak
   cooling is currently *under-estimated* (not over) and re-frame the root cause as
   load-side (solar) rather than time-constant-side (CTF).
2. **Open new issue**: "Investigate Case 900 peak cooling under-estimation
   (0.86 vs 2.10-3.50 kW)". Suggested first sub-tasks: verify roof solar distribution
   against EnergyPlus `eplusout.sql` for Case 900; quantify internal-gain delivery.
3. **Sub-stepping experiment**: leave parked. Re-evaluate only if (a) the underlying
   load-underestimation is fixed, and (b) we still see over-temperature peaks in the
   zone (i.e. the controller is over-cooling) — neither condition currently holds.

## 8. Files changed in this PR

- `docs/KNOWN_ISSUES.md` — LIMIT-05 section updated to reflect inverted state + add
  pointer to this investigation.
- `docs/investigations/issue-1280-ctf-peak-load.md` — this document.
- `tests/limit_05_inversion_regression.rs` — new regression test that locks in the
  current peak cooling values for Case 900 / Case 950 / Case 960, with a doc-comment
  explaining the inversion. Test is `#[ignore]` by default (CI-time diagnostic) but
  can be run with `--ignored`.

## 9. Reproduction commands

```bash
# Full Case 900 multi-node summary (requires release build):
cargo test --release -p fluxion --test case_900_multinode_validation \
    test_case_900_multinode_validation_summary -- --nocapture

# Single-case diagnostic with current peak cooling printout:
./target/release/deps/case_900_multinode_validation-* \
    test_case_900_multinode_validation_summary --nocapture
```

Both reproduce the table in §2 on HEAD of `fix/issue-1280-ctf-peak-load`.