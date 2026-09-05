// Seed kernel for issue #3338 — inter-zone radiation evolution candidate.
//
// SCOPE
// -----
// This seed targets the **single-pair Stefan-Boltzmann kernel**
// `surface_radiative_exchange` from `src/sim/interzone_radiation.rs`,
// which is the inner call inside `net_lw_*` longwave-exchange kernels
// and the inter-zone radiation accumulation loops in
// `src/sim/solar_gain_distribution.rs`. The kernel itself is a
// 6-product-fma-friendly reduction that the issue calls out as the
// most amenable of the measured-hot accumulation loops to SIMD
// (issue #3338 §"tasks").
//
// EQUIVALENCE CONTRACT
// --------------------
// Output must equal the canonical `surface_radiative_exchange` to
// within the harness's per-edge tolerance (`1e-9` default,
// `1e-6` under `--features simd-kernels`). NaN/Inf is a hard fail.
// Reciprocity is mechanically satisfied by `T⁴ - t⁴` being
// skew-symmetric in `t_a, t_b` *and* by the radiative-exchanger's
// own internal tests; the harness's reciprocity invariant records
// `n/a` for this kernel.
//
// MUTATION SPACE
// --------------
// The body of `evolve_stefan_boltzmann_pair` below is the only
// mutation surface. The kernel is small enough that even a cache-
// friendly 4-lane explicit unroll with strict associativity is enough
// to demonstrate the SIMD path; further widening to AVX-512 / NEON
// is encouraged in `// EVOLVE-BLOCK`. Alignment/padding, FMA
// contraction, and reciprocal-product merge are all in scope; the
// `core::arch` runtime-detect gate should stay inside the block so
// the harness can verify the contract is enforced.
//
// OUT-OF-SCOPE: same as the perez seed (struct, contract, signature).

use fluxion::sim::interzone_radiation::surface_radiative_exchange;
use fluxion_evaluator::kernel::{Kernel, KernelError, KernelInput, KernelOutput};
use serde_json::Value;

#[derive(Default, Debug, Clone, Copy)]
pub struct Candidate;

impl Kernel for Candidate {
    fn evaluate(&self, input: &KernelInput) -> Result<KernelOutput, KernelError> {
        let p = &input.params;
        let t_a_c = read_f64(p, "t_a_c")?;
        let t_b_c = read_f64(p, "t_b_c")?;
        let eps_a = read_f64(p, "emissivity_a")?;
        let eps_b = read_f64(p, "emissivity_b")?;
        let f_ab = read_f64(p, "view_factor")?;
        let area = read_f64(p, "area")?;

        let q = evolve_stefan_boltzmann_pair(t_a_c, t_b_c, eps_a, eps_b, f_ab, area);
        Ok(KernelOutput {
            payload: serde_json::json!({
                "case_name": input.case_name,
                "q_w": q,
            }),
        })
    }
}

#[inline(never)]
pub fn evolve_stefan_boltzmann_pair(
    t_a_c: f64,
    t_b_c: f64,
    eps_a: f64,
    eps_b: f64,
    view_factor: f64,
    area: f64,
) -> f64 {
    // EVOLVE-BLOCK-START
    // -------------------------------------------------------------
    // Default scalar passthrough (issue #3338: default-feature builds
    // bit-identical to today). Evolvers may replace this with an
    // FMA-contracted, lane-packed, runtime-dispatched kernel under
    // `--features simd-kernels` only — see the header for the
    // allowed edit envelope.
    surface_radiative_exchange(t_a_c, t_b_c, eps_a, eps_b, view_factor, area)
    // EVOLVE-BLOCK-END
}

fn read_f64(v: &Value, key: &str) -> Result<f64, KernelError> {
    v.get(key)
        .and_then(|x| x.as_f64())
        .ok_or_else(|| KernelError::BadInput(format!("missing `{}`", key)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_block_matches_canonical_sunspace() {
        let q = evolve_stefan_boltzmann_pair(40.0, 20.0, 0.9, 0.9, 1.0, 21.6);
        // From the existing test in `src/sim/interzone_radiation.rs`
        // we know the canonical answer is ~2214 W for this input.
        assert!((q - 2214.0).abs() < 10.0, "expected ~2214 W, got {q}");
    }

    #[test]
    fn evaluate_roundtrips_through_json() {
        let input = KernelInput {
            case_name: "sunspace-40-20".into(),
            params: serde_json::json!({
                "t_a_c": 40.0,
                "t_b_c": 20.0,
                "emissivity_a": 0.9,
                "emissivity_b": 0.9,
                "view_factor": 1.0,
                "area": 21.6,
            }),
        };
        let out = Candidate.evaluate(&input).expect("evaluate");
        let json = serde_json::to_string(&out.payload).expect("encode");
        assert!(json.contains("\"q_w\""));
    }
}
