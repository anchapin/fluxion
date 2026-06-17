# Result: Perez Formula Mathematical Analysis (Agent 3)

**Status**: BUG CONFIRMED
**Confidence**: HIGH (100%)
**Date**: 2026-05-18

## Executive Summary

**BUG CONFIRMED**: The `calculate_cos_incidence` function in `sky_radiation.rs` (lines 546-548) contains a **sin/cos swap of the zenith angle**. The formula uses `cos(θz)` where it should use `sin(θz)` and vice versa. This causes the diffuse irradiance calculation via the Perez model to be incorrect for all non-45° zenith angles.

A secondary bug exists on line 743 where the GHI fallback computation also swaps `sin(θz)` for `cos(θz)`.

## 1. Mathematical Derivation

### The Code Formula (lines 546-548)
```rust
let cos_incidence = tilt.sin() * surface_az.sin() * zenith.cos() * solar_az.sin()
    + tilt.sin() * surface_az.cos() * zenith.cos() * solar_az.cos()
    + tilt.cos() * zenith.sin();
```

Using symbols β=tilt, γs=surface_azimuth, θz=zenith, γ=solar_azimuth:

```
Code: cos(θ) = sin(β)·sin(γs)·cos(θz)·sin(γ) + sin(β)·cos(γs)·cos(θz)·cos(γ) + cos(β)·sin(θz)
```

Applying the trig identity `cos(A-B) = cos(A)cos(B) + sin(A)sin(B)` to simplify the first two terms:

```
Code: cos(θ) = sin(β)·cos(θz)·cos(γ - γs) + cos(β)·sin(θz)
```

### The Standard Formula (Duffie & Beckman, "Solar Engineering of Thermal Processes")

Using zenith angle and solar azimuth:
```
Standard: cos(θ) = cos(θz)·cos(β) + sin(θz)·sin(β)·cos(γs - γ)
```

### Comparison

| Term | Code | Standard |
|------|------|----------|
| Constant (no cos(Δγ)) | cos(β)·sin(θz) | cos(θz)·cos(β) |
| cos(Δγ) coefficient | sin(β)·cos(θz) | sin(θz)·sin(β) |

**The code has cos(θz) and sin(θz) swapped.** Where the standard has cos(θz), the code has sin(θz), and vice versa.

### Algebraic Proof of the Bug

The difference between code and standard:
```
Δ = Code - Standard
  = [cos(θz) - sin(θz)] · [sin(β)·cos(Δγ) - cos(β)]
```

This is zero only when:
1. θz = 45° (cos(θz) = sin(θz)), or
2. sin(β)·cos(Δγ) = cos(β) (specific geometric coincidence)

In general, Δ ≠ 0 → **the formula is WRONG**.

### Cross-Validation with Correct Implementation

`solar.rs` lines 57-72 contain `SolarPosition::incidence_cosine()` which uses the altitude-based form:
```rust
// cos(θ) = sin(α)·cos(β) + cos(α)·sin(β)·cos(φ - γ)
let cos_theta_i = alpha.sin() * beta.cos() + alpha.cos() * beta.sin() * (phi - gamma).cos();
```

Since altitude α = 90° - θz:
- sin(α) = cos(θz)
- cos(α) = sin(θz)

Substituting: cos(θ) = cos(θz)·cos(β) + sin(θz)·sin(β)·cos(Δγ) ✓ **This matches the standard.**

The codebase has **two different implementations** of the same formula — one correct (`incidence_cosine` in solar.rs) and one buggy (`calculate_cos_incidence` in sky_radiation.rs).

## 2. Numerical Test Cases

### Test 1: Horizontal Surface (tilt=0°, zenith=30°, solar_az=180°)

| Implementation | Result | Expected |
|---------------|--------|----------|
| **Buggy code** | sin(30°) = **0.500** | - |
| **Correct** | cos(30°) = **0.866** | 0.866 ✓ |
| Error | | **-42.3%** |

For horizontal surface, incidence angle = zenith angle, so cos(θ) = cos(θz) = cos(30°) ≈ 0.866. The code returns sin(30°) = 0.5 — off by 42%.

### Test 2: Vertical South-Facing, Solar Noon (tilt=90°, surface_az=180°, zenith=30°, solar_az=180°)

| Implementation | Result | Expected |
|---------------|--------|----------|
| **Buggy code** | cos(30°) = **0.866** | - |
| **Correct** | sin(30°) = **0.500** | 0.500 ✓ |
| Error | | **+73.2%** |

### Test 3: Error Varies with Zenith Angle (Vertical Surface)

| Zenith | Buggy cos(θ) | Correct cos(θ) | Error |
|--------|-------------|----------------|-------|
| 10° | 0.985 | 0.174 | +467% |
| 20° | 0.940 | 0.342 | +175% |
| 30° | 0.866 | 0.500 | +73% |
| 45° | 0.707 | 0.707 | 0% (accidental) |
| 60° | 0.500 | 0.866 | -42% |
| 70° | 0.342 | 0.940 | -64% |
| 80° | 0.174 | 0.985 | -82% |

## 3. Impact on ASHRAE 140 Results

### Affected Code Path

The bug affects the **Perez diffuse irradiance calculation** used in `calculate_surface_irradiance()` (solar.rs line 268):

```
calculate_surface_irradiance()
  → sun_pos.incidence_cosine()    ← CORRECT (beam irradiance)
  → PerezSkyModel::calculate_diffuse_tilted()
      → calculate_cos_incidence()  ← BUGGY (affects circumsolar term)
```

The buggy `cos_incidence` feeds into the Perez model's circumsolar brightness term:
- `a = cos_incidence.max(0.0)` (line 458)
- `term2 = f1 * a / b` (line 462)

For a south-facing wall at solar noon with low zenith (summer):
- Buggy `a` is too large → circumsolar diffuse is **overestimated**
- This causes **excessive diffuse solar gain** in summer

For a south-facing wall at solar noon with high zenith (winter):
- Buggy `a` is too small → circumsolar diffuse is **underestimated**
- This causes **insufficient diffuse solar gain** in winter

### Not Affected

- **Beam irradiance**: Uses `sun_pos.incidence_cosine()` which is correct
- **Ground-reflected irradiance**: Does not use cos_incidence
- **Perez isotropic/horizon terms** (term1, term3): Do not use cos_incidence

### Secondary Bug (line 743)

```rust
// total_irradiance_tilted, line 741-743:
let ghi = ghi.unwrap_or_else(|| {
    let zenith_rad = zenith_deg.to_radians();
    dni * zenith_rad.sin() + dhi  // ← BUG: should be .cos()
});
```

This function is NOT used in the main ASHRAE 140 simulation path (which uses `calculate_surface_irradiance` instead), so this bug is latent.

## 4. The Correct Formula

### Fix for calculate_cos_incidence (lines 546-548)

**Current (buggy):**
```rust
let cos_incidence = tilt.sin() * surface_az.sin() * zenith.cos() * solar_az.sin()
    + tilt.sin() * surface_az.cos() * zenith.cos() * solar_az.cos()
    + tilt.cos() * zenith.sin();
```

**Corrected:**
```rust
let cos_incidence = zenith.sin() * tilt.sin() * surface_az.cos() * solar_az.cos()
    + zenith.sin() * tilt.sin() * surface_az.sin() * solar_az.sin()
    + zenith.cos() * tilt.cos();
```

Or equivalently (simplified):
```rust
let cos_incidence = zenith.cos() * tilt.cos()
    + zenith.sin() * tilt.sin() * (solar_az - surface_az).cos();
```

### Fix for GHI fallback (line 743)

**Current (buggy):**
```rust
dni * zenith_rad.sin() + dhi
```

**Corrected:**
```rust
dni * zenith_rad.cos() + dhi
```

## 5. Confidence Assessment

**HIGH (100%)** — The bug is confirmed through:
1. Algebraic proof showing cos(θz)/sin(θz) swap
2. Multiple numerical test cases with known expected values
3. Cross-validation against the correct `incidence_cosine` in the same codebase
4. Cross-validation against Duffie & Beckman standard reference
5. The codebase itself proves the inconsistency: two implementations of the same formula give different results

## 6. Root Cause Hypothesis

The developer likely transcribed the formula from a reference that used **solar altitude** (α) instead of **zenith angle** (θz), then mechanically substituted zenith without accounting for the identity:

```
sin(altitude) = cos(zenith)
cos(altitude) = sin(zenith)
```

The `solar.rs` implementation correctly uses altitude. The `sky_radiation.rs` implementation attempted to use zenith directly but failed to apply the altitude↔zenith conversion, resulting in swapped sin/cos.
