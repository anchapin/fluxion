# FD conduction time-integration characterization — Issues #3980 / #3981

> **Summary 1/7:** The implicit FD conduction solver (`src/physics/fd_solver.rs`) gained a `TimeIntegrationScheme` enum (`BackwardEuler | Bdf2 | CrankNicolson`) with **Bdf2 as the production default**; since #3981 the spatial operator is the **conservative per-side conductance assembly** (Patankar CV form, films through the boundary half-cell — multi-layer walls realize their exact series resistance).
> **Summary 2/7:** Verified observed orders against a dt = 5 s semi-discrete reference: BackwardEuler 1st order, Bdf2 2nd order (constant and alternating step sizes), CrankNicolson 2nd order in its applicability regime (z = μ·dt < 2 for every excited mode).
> **Summary 3/7:** CN is A-stable but **not L-stable**: on a 12 mm gypsum wall at the 3600 s zone step (z_max ≈ 8) it rings persistently while BDF2 settles to steady state — the documented reason BDF2 (not CN) is the default despite CN's smaller error constant.
> **Summary 4/7:** Steady-periodic sweep (200 mm concrete wall, 24 h sinusoidal sol-air, interior flux RMS vs dt = 5 s reference): at equal cost BDF2 reaches errors BackwardEuler needs ~60× more steps for (BDF2@120 s ≈ 5.9e-4 W/m² vs BE@60 s ≈ 5.0e-2 W/m²).
> **Summary 5/7:** `ConductionBackendConfig.fd_timestep` is now honored via substepping inside the hourly zone step (non-divisor configurations fail loudly); the teacher entry for free-floating validation runs FD at 60 s substeps (`FD_TEACHER_SUBSTEP_S`).
> **Summary 6/7:** The 50-term CTF is demoted from the free-floating primary (validator wiring, Issue #486 workaround removed) to a fast cross-check for linear constructions; `select_method` never returns CTF-primary since Issue #726; the #3981 analytical module measured its pole-residue envelope and found the **light-wall DC gain has the wrong sign (Issue #4062)**.
> **Summary 7/7:** Known follow-up (root-caused, not patched): wiring the 60 s FD teacher into the free-floating zone loops diverges because the explicit zone↔FD flux coupling gain exceeds 1 when the substepped flux tracks zone temperature — the implicit DAE zone-wall assembly (ADR-0017, Issues #3982/#3983) is the sanctioned fix; no tolerance was relaxed.

## Scope

Characterization of the three time-integration schemes for the FD conduction
solver, replacing the historical backward-Euler (BTCS) single option. This
document records the accuracy-vs-cost evidence behind the BDF2 default and
the demotion of the 50-term CTF. The full ASHRAE 1052-RP regression module
(analytical steady-periodic ground truth, four walls, FD + CTF) landed with
Issue #3981 — see the dedicated section below.

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
constructions. It no longer serves as a primary path anywhere:
`select_method` stopped returning CTF-primary in Issue #726, and the
free-floating validator workaround (Issue #486) was removed in favour of the
FD teacher entry. The #3981 module measured its actual envelope (below) and
found the light-wall DC defect (Issue #4062).

## ASHRAE 1052-RP analytical steady-periodic regression (Issue #3981)

Module: `tests/all_tests/conduction_1052rp_analytical.rs`. Ground truth is
the closed-form complex transmission-matrix solution of the 1-D conduction
PDE (cross-validated against an independent fine-grid Crank-Nicolson solve,
amp err ≤ 0.5%, phase err ≤ 0.006 rad), evaluated for the **staircase
(ZOH) sol-air excitation** the solvers actually integrate — the solvers see
held samples, not a continuous sinusoid, and the reference must model that.

Setup: light (80 mm EPS) / medium (100 mm brick) / heavy (200 mm concrete) /
multi (brick + EPS + gypsum) walls; T_zone = 21.5 °C constant with
h_int = 8.3, sol-air 29.5 + 12.5·sin(2πt/24 h), h_ext = 18.3; FD films on
the BCs; CTF comparisons use its baked ASHRAE 140 films (R_SI = 0.125,
R_SE = 0.044). Metrics: demodulated fundamental amplitude / phase-lag / mean
of the into-zone flux after 14 d spin-up (periodicity asserted to 1e-6·amp,
except CN-light@3600).

### Defects found and fixed by this ground truth

1. **fd_solver multi-layer U was wrong** (fixed in this issue): the former
   uniform-grid stencil `Fo·(T[i-1] − 2T[i] + T[i+1])` collapses at steady
   state to a profile linear in node *index* — interior resistance split by
   node count, not layer R-value (brick+EPS realized U 6.3× too high, and
   steady face fluxes did not conserve energy). The conservative per-side
   conductance assembly now realizes exact series resistance and telescoping
   face fluxes on every layer stack; the mean column below is 0.000% for
   all cells.
2. **CTF light-wall DC gain has the wrong sign** (Issue #4062, open): q_ss =
   −5.53 vs +3.69 W/m² under constant forcing for 80 mm EPS — the
   y-coefficient `abs().max(0)` clamping breaks the DC identity when all
   poles have τ ≪ dt.
3. **FD-vs-E+ step-response tests were a circular identity** (Issue #4058,
   quarantined): the old back-calculation passed through the same formula
   the extraction used.

### FD budget table (30 nodes/layer, pinned from measured actuals)

| Wall | Scheme | dt = 3600 s: amp / phi [rad] | dt = 900 s: amp / phi [rad] |
|---|---|---|---|
| heavy | Bdf2 | −1.7% / +0.15 | −0.1% / +0.03 |
| heavy | BackwardEuler | −9.9% / +0.03 | −2.8% / +0.01 |
| heavy | CrankNicolson | −0.2% / +0.14 | −0.1% / +0.03 |
| medium | Bdf2 | −0.4% / +0.15 | −0.0% / +0.03 |
| medium | BackwardEuler | −6.8% / +0.10 | −1.9% / +0.02 |
| medium | CrankNicolson | +0.3% / +0.14 | −0.0% / +0.03 |
| light | Bdf2 | −0.2% / +0.07 | +0.0% / +0.03 |
| light | BackwardEuler | −1.2% / +0.07 | −0.2% / +0.03 |
| light | CrankNicolson | −0.1% / +0.07 (rings) | +0.0% / +0.03 |
| multi | Bdf2 | −1.4% / +0.15 | −0.1% / +0.03 |
| multi | BackwardEuler | −10.1% / +0.05 | −2.8% / +0.01 |
| multi | CrankNicolson | −0.1% / +0.14 | −0.0% / +0.03 |

Mean error is 0.000% in every cell (exact steady-state U realization).
Notes: BE's smaller phase error at dt = 3600 is **not** scheme superiority —
its amplitude damping cancels part of the first-order ZOH/sampling phase
residual; the amplitude metric is the honest discriminator and the pinned
gate asserts `err_amp(BDF2) < err_amp(BE)` at both dt for all four walls,
plus BDF2 refinement from 3600 s to 900 s.

### CTF envelope (measured, current implementation)

| Wall | amp err | phi err | mean err |
|---|---|---|---|
| heavy | +57% | −0.39 rad | 0.000% |
| medium | +10% | +0.02 rad | 0.000% |
| light | −4% | +0.43 rad | **−269% (defect #4062)** |
| multi | +29% | −0.21 rad | 0.000% |

The pole-residue coefficient approximation is normalized to the exact U at
DC (heavy/medium/multi) but has no tight dynamic envelope; reworking it is
tracked by #4062.
