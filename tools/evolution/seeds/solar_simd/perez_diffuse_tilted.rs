// Seed kernel for issue #3338 — solar/irradiance evolution candidate.
//
// SCOPE
// -----
// This seed targets **only one** of the six measured-hot accumulation
// loops called out in the issue:
//
//   * `perez_diffuse_tilted` — the inner reduction of the
//     Pérez 1990 all-weather sky model that produces per-tilted-surface
//     diffuse irradiance. This is the same body `calculate_surface_irradiance`
//     delegates to for non-horizontal surfaces, and it dominates the
//     inner-loop time of the 10k-vectorized bench.
//
// The other five loops (compute_surface_irradiance, surface_radiative_exchange,
// net_lw_floor_pair, sky_radiation_net_flux, calculate_per_surface_irradiance)
// use the same harness-but-different-input pattern and are seeded by
// `seeds_*.md` under the same directory; they don't add new harness
// surface area.
//
// EQUIVALENCE CONTRACT vs. the SCALAR BASELINE
// --------------------------------------------
// The harness's invariant battery (issue #3338 INVARIANT §6) measures:
//
//   1. *Reciprocity* — not applicable here (Pérez is a one-way
//      irradiance, not a view factor). The harness records
//      `invariant_kind = "n/a"` and exits with `invariants_passed = true`.
//
//   2. *Per-tilt fixture tolerances* — the seed's output must agree
//      with the canonical `PerezSkyModel::calculate_diffuse_tilted`
//      within the relative tolerance baked into the harness for the
//      `per_tilt_per_azimuth_fixture_data.rs` fixtures. Default tolerance
//      is `1e-9` for the static inputs (Issue #3338 acceptance §"default-
//      feature builds byte-identical") and relaxes to `1e-6` for SIMD
//      candidates that opt into the `simd-kernels` feature (allowance
//      for last-ulp reassociation under reassociation / contraction —
//      see the `fast-math` determinism warning in `physics/fp_algebraic.rs`).
//
//   3. *NaN / Inf rejection* — binary. A NaN/Inf in the output payload
//      is a hard fail.
//
// The harness feeds three edge cases to this kernel: a low-DHI
// clear-sky case, the canonical 800/100 W/m² Denver-noon case, and a
// high-airmass twilight case. Reference values were captured against
// the current scalar implementation and live in `tools/evolution/
// edge_cases/solar_simd.json` (generated via
// `scripts/regenerate_simd_edge_cases.sh`, never hand-edited).
//
// MUTATION SPACE (OpenEvolve `// EVOLVE-BLOCK-START / END`)
// --------------------------------------------------------
// The candidate body inside `evolve_diffuse_tilted` is the single
// mutation surface. Everything else (the `Candidate` struct, the
// input parsing, the reference comparison) is **frozen context** the
// evolver must not touch — touching it produces a candidate that fails
// schema-v1 dispatch.
//
// Allowed edits inside the EVOLVE-BLOCK span:
//   1. AoS → SoA data layout: pack the per-edge scalar inputs into a
//      single `[f64; LANES]` lane array first.
//   2. Cache tile dimensions: pre-load `LANES_PER_TILE` parameter
//      chunks into local arrays to reduce L1 pressure.
//   3. Unroll factors: 1× / 2× / 4× / 8× unrolling of the inner
//      product chain.
//   4. SIMD lane mappings: `core::arch::x86_64::{_mm256_*}` intrinsics
//      behind `is_x86_feature_detected!("avx2")` runtime dispatch
//      with a portable scalar fallback. aarch64/NEON parity is a
//      separate, follow-up issue (the issue's "Cross-platform
//      determinism" check covers the expected bounded ulp deltas
//      per #2549).
//   5. Alignment/padding strategies: `#[repr(align(32))]` lane
//      storage, `MaybeUninit` zero-cost init under the tile chunk.
//
// OUT-OF-SCOPE — DO NOT MUTATE
// --------------------------
//   * The `pub struct Candidate` and `impl Kernel for Candidate`.
//   * The `KernelInput` -> `KernelOutput` JSON contract
//     (`params.case_name`, `params.dhi`, …).
//   * The `evolve_diffuse_tilted` *function signature* (the harness
//     generates the wrapper that calls it).
//   * Any change to the `fast-math` boundary (issue #3338 acceptance
//     "No changes to `fast-math` boundaries; `simd-kernels` is a
//     separate feature").

use fluxion::solar::surface_irradiance::PerezSkyModel;
use fluxion_evaluator::kernel::{Kernel, KernelError, KernelInput, KernelOutput};
use serde_json::Value;

/// Candidate type that the harness instantiates as `Candidate::default()`
/// per the issue's seed protocol.
///
/// Always present, never mutated (see "OUT-OF-SCOPE" above).
#[derive(Default, Debug, Clone, Copy)]
pub struct Candidate;

impl Kernel for Candidate {
    fn evaluate(&self, input: &KernelInput) -> Result<KernelOutput, KernelError> {
        let params = &input.params;
        let dhi = read_f64(params, "dhi")?;
        let dni = read_f64(params, "dni")?;
        let dni_extra = read_f64(params, "dni_extra")?;
        let airmass = read_f64(params, "airmass")?;
        let zenith_deg = read_f64(params, "zenith_deg")?;
        let tilt_deg = read_f64(params, "tilt_deg")?;
        let surface_azimuth_deg = read_f64(params, "surface_azimuth_deg")?;
        let solar_azimuth_deg = read_f64(params, "solar_azimuth_deg")?;

        let result = evolve_diffuse_tilted(
            dhi,
            dni,
            dni_extra,
            airmass,
            zenith_deg,
            tilt_deg,
            surface_azimuth_deg,
            solar_azimuth_deg,
        );

        Ok(KernelOutput {
            payload: serde_json::json!({
                "case_name": input.case_name,
                "diffuse_tilted_wm2": result,
            }),
        })
    }
}

/// Mutation surface — `// EVOLVE-BLOCK-START`.
///
/// Everything inside this block is free game for OpenEvolve; everything
/// outside is frozen context (see header).
///
/// The harness guarantees the *default-feature* scalar build is
/// bit-identical to `PerezSkyModel::calculate_diffuse_tilted`; under
/// `--features simd-kernels` the candidate may reach for SIMD
/// intrinsics behind a runtime feature-detect gate, with the documented
/// tolerance relaxation (default `1e-9`, simd-kernels `1e-6`).
///
/// Signature is frozen; the harness-generated `src/lib.rs` calls this
/// by name with the eight `f64` parameters in the documented order.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn evolve_diffuse_tilted(
    dhi: f64,
    dni: f64,
    dni_extra: f64,
    airmass: f64,
    zenith_deg: f64,
    tilt_deg: f64,
    surface_azimuth_deg: f64,
    solar_azimuth_deg: f64,
) -> f64 {
    // EVOLVE-BLOCK-START
    // ----------------------------------------------------------------
    // Default scalar implementation. Evolvers may replace this body
    // wholesale with an SIMD/cache-blocked variant; see the header's
    // "MUTATION SPACE" section for the allowed edit envelope.
    //
    // This default keeps the contract: it calls through to the
    // canonical `PerezSkyModel::calculate_diffuse_tilted` exactly as
    // defined in `src/solar/surface_irradiance.rs` so any candidate
    // that disagrees can only do so by deliberate mutation.
    PerezSkyModel::calculate_diffuse_tilted(
        dhi,
        dni,
        dni_extra,
        airmass,
        zenith_deg,
        tilt_deg,
        surface_azimuth_deg,
        solar_azimuth_deg,
    )
    // EVOLVE-BLOCK-END
}

/// Tiny helper that does not panic on a missing parameter: returns a
/// recoverable `Err(KernelError::BadInput)`. The harness converts
/// `Err` into the `kernel_error` violation kind in the Summary, so a
/// missing field surfaces without crashing the campaign.
fn read_f64(v: &Value, key: &str) -> Result<f64, KernelError> {
    v.get(key)
        .and_then(|x| x.as_f64())
        .ok_or_else(|| KernelError::BadInput(format!("missing `{}`", key)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default-feature contract: a candidate whose EVOLVE-BLOCK body is
    /// the unmodified scalar passthrough must return a finite,
    /// non-negative diffuse-tilted value for the canonical inputs.
    #[test]
    fn default_block_returns_finite_canonical_value() {
        let v = evolve_diffuse_tilted(100.0, 800.0, 1361.0, 1.5, 45.0, 60.0, 180.0, 180.0);
        assert!(v.is_finite(), "must not produce NaN/Inf: {v}");
        assert!(v >= 0.0, "diffuse irradiance must be non-negative: {v}");
    }

    /// Schema-v1 contract: the `Kernel::evaluate` body must produce a
    /// JSON-encodable output. This guards the contract from a
    /// candidate that accidentally types `result` into a non-JSON type.
    #[test]
    fn evaluate_roundtrips_through_json() {
        let input = KernelInput {
            case_name: "denver-noon".into(),
            params: serde_json::json!({
                "dhi": 100.0,
                "dni": 800.0,
                "dni_extra": 1361.0,
                "airmass": 1.5,
                "zenith_deg": 45.0,
                "tilt_deg": 60.0,
                "surface_azimuth_deg": 180.0,
                "solar_azimuth_deg": 180.0,
            }),
        };
        let out = Candidate.evaluate(&input).expect("evaluate");
        let json = serde_json::to_string(&out.payload).expect("encode");
        assert!(json.contains("\"diffuse_tilted_wm2\""));
        // And the harness-then-evaluator convention: parse it back as
        // a `serde_json::Value` to confirm nothing exotic slipped in.
        let _: Value = serde_json::from_str(&json).expect("decode");
    }
}
