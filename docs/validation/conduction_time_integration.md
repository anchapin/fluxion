# FD conduction time-integration characterization — Issue #3980

> **Summary 1/7:** The implicit FD conduction solver (`src/physics/fd_solver.rs`) gained a `TimeIntegrationScheme` enum (`BackwardEuler | Bdf2 | CrankNicolson`) with **Bdf2 as the production default**; the spatial discretization is unchanged (node-centred ghost-node grid).
> **Summary 2/7:** Verified observed orders against a dt = 5 s semi-discrete reference: BackwardEuler 1st order, Bdf2 2nd order (constant and alternating step sizes), CrankNicolson 2nd order in its applicability regime (z = μ·dt < 2 for every excited mode).
> **Summary 3/7:** CN is A-stable but **not L-stable**: on a 12 mm gypsum wall at the 3600 s zone step (z_max ≈ 8) it rings persistently while BDF2 settles to steady state — the documented reason BDF2 (not CN) is the default despite CN's smaller error constant.
> **Summary 4/7:** Steady-periodic sweep (200 mm concrete wall, 24 h sinusoidal sol-air, interior flux RMS vs dt = 5 s reference): at equal cost BDF2 reaches errors BackwardEuler needs ~60× more steps for (BDF2@120 s ≈ 5.9e-4 W/m² vs BE@60 s ≈ 5.0e-2 W/m²).
> **Summary 5/7:** `ConductionBackendConfig.fd_timestep` is now honored via substepping inside the hourly zone step (non-divisor configurations fail loudly); the teacher entry for free-floating validation runs FD at 60 s substeps (`FD_TEACHER_SUBSTEP_S`).
> **Summary 6/7:** The 50-term CTF is demoted from the free-floating primary (validator wiring, Issue #486 workaround removed) to a fast cross-check for linear constructions; `select_method` never returns CTF-primary since Issue #726.
> **Summary 7/7:** Known follow-up (root-caused, not patched): wiring the 60 s FD teacher into the free-floating zone loops diverges because the explicit zone↔FD flux coupling gain exceeds 1 when the substepped flux tracks zone temperature — the implicit DAE zone-wall assembly (ADR-0017, Issues #3982/#3983) is the sanctioned fix; no tolerance was relaxed.

## Scope

Characterization of the three time-integration schemes for the FD conduction
solver, replacing the historical backward-Euler (BTCS) single option. This
document records the accuracy-vs-cost evidence behind the BDF2 default and
the demotion of the 50-term CTF. The full ASHRAE 1052-RP regression module
(multi-layer amplitude/phase vs published RP data) is Issue #3981; the
minimal steady-periodic comparative check landed with this issue.

## Method

Steady-periodic forcing: 200 mm concrete wall (k = 1.75 W/m·K,
ρ = 2300 kg/m³, c = 880 J/kg·K), 80 FD nodes, 24 h sinusoidal sol-air
(mean 20 °C, amplitude 15 °C), interior convective h = 8.3 W/m²K. 10 days
spin-up, 3 days of hourly interior surface flux compared against a
dt = 5 s BDF2 run of the same solver (semi-discrete reference — isolates
the time integrator from the O(dx²) spatial floor). Cost is wall-clock of
the full 13-day run, normalized to BackwardEuler at 3600 s.

Reproduce with:

```bash
cargo test --test all_tests fd_time_integration_sweep -- --ignored --nocapture
```

## Accuracy-vs-cost sweep (2026-09-25, this commit)

| Scheme | dt [s] | Flux RMS error [W/m²] | Rel. cost |
|---|---|---|---|
| BackwardEuler | 3600 | 2.83e+00 | 1.38 |
| BackwardEuler | 1800 | 1.46e+00 | 2.71 |
| BackwardEuler | 900 | 7.44e-01 | 5.36 |
| BackwardEuler | 600 | 4.99e-01 | 7.40 |
| BackwardEuler | 300 | 2.51e-01 | 12.13 |
| BackwardEuler | 120 | 1.01e-01 | 31.22 |
| BackwardEuler | 60 | 5.05e-02 | 65.04 |
| CrankNicolson | 3600 | 1.33e-01 | 1.25 |
| CrankNicolson | 1800 | 3.31e-02 | 2.90 |
| CrankNicolson | 900 | 8.27e-03 | 5.35 |
| CrankNicolson | 600 | 3.67e-03 | 8.03 |
| CrankNicolson | 300 | 9.18e-04 | 16.45 |
| CrankNicolson | 120 | 1.46e-04 | 45.23 |
| CrankNicolson | 60 | 3.57e-05 | 83.90 |
| Bdf2 | 3600 | 5.18e-01 | 1.53 |
| Bdf2 | 1800 | 1.32e-01 | 2.84 |
| Bdf2 | 900 | 3.30e-02 | 5.20 |
| Bdf2 | 600 | 1.47e-02 | 6.24 |
| Bdf2 | 300 | 3.67e-03 | 17.03 |
| Bdf2 | 120 | 5.87e-04 | 47.84 |
| Bdf2 | 60 | 1.46e-04 | 87.75 |

Reading the table:

- **BackwardEuler** halves its error per step halving (1st order); it needs
  dt ≈ 2 s to reach what BDF2 delivers at 120 s — roughly 60× the cost.
- **CrankNicolson** quarters its error per halving (2nd order) with an error
  constant ~4× below BDF2 on this smooth problem — the best per-step
  accuracy of the three.
- **Bdf2** also quarters per halving (2nd order). Its edge over CN is
  robustness, not per-step accuracy: BDF2 is L-stable (amplification → 0 as
  z → ∞), CN is not.

## Why BDF2 is the default (not CN)

CN's amplification factor (1 − z/2)/(1 + z/2) → −1 as z → ∞: lightly damped
high discrete modes ring forever. The pinned test
`crank_nicolson_oscillates_at_large_fourier_where_bdf2_stays_monotone`
drives a 12 mm gypsum wall (light layer) at the 3600 s zone step —
z_max = 4·Fo ≈ 8 — and records sustained CN reversals while BDF2 settles to
steady state within the physical envelope. Teacher paths step light
constructions at coarse zone steps; an integrator that rings on them is not
a safe default. ADR-0017's prescription (implicit BDF/IDA-type integration)
aligns.

## Energy accounting

Cumulative boundary-energy vs stored-energy closure (trapezoidal flux
quadrature, the repo convention) is within the incumbent accepted band for
all three schemes, and the 2nd-order schemes do not degrade it relative to
BackwardEuler (`cumulative_energy_balance_closes_for_all_schemes`). The
~50% closure level itself is a pre-existing property of the retained
node-centred ghost boundary rows (boundary flux coupling doubled) — not of
the time integrators.

## Relationship to the CTF solver

The 50-term CTF remains available as a fast cross-check for linear
constructions (exact for linear conduction — its envelope). It no longer
serves as a primary path anywhere: `select_method` stopped returning
CTF-primary in Issue #726, and the free-floating validator workaround
(Issue #486) was removed in this issue in favour of the FD teacher entry.
