// =============================================================================
// Quantum annealing bridge (Phase 2b — issue #1464)
// =============================================================================
//
// Research scaffolding for offloading sub-problems of the gauge-theory thermal
// pipeline to a quantum annealer (D-Wave Advantage and successors). The bridge
// is intentionally narrow: it maps the continuous `ThermalManifold` tensors to
// a Quadratic Unconstrained Binary Optimization (QUBO) matrix `Q` so that
//
//     x^T Q x   =   T_recon^T · metric_tensor · T_recon
//
// (optionally plus a `-gauge_connection^T T_recon` linear bias), where
// `T_recon = decode(x)` is the temperature vector recovered from the bit
// string `x` via fixed-point decoding.
//
// **No production annealer integration is performed** — this module only
// constructs the matrix. Wiring the actual D-Wave Ocean SDK is the planned
// Phase 2c follow-up.
//
// **Why fixed-point?** Quantum annealers are natively binary; representing a
// real-valued vector field `T ∈ R^4` requires either unary (one bit per
// temperature × precision bit) or analog encoding. Unary fixed-point is the
// simplest and most hardware-friendly choice: K bits per temperature ⇒ total
// `N = 4*K` qubits, K=8 gives ~0.2 °C resolution which is well below the
// 0.5 °C precision of typical ASHRAE 140 reference data.
//
// **Why these particular QUBO coefficients?** Working back from
// `T[i] = (Σ_k 2^k x[(i,k)]) / scale` and `x^T Q x = T^T M T` gives
//
//     Q[(i,k), (j,l)] = metric[i,j] * 2^k * 2^l / scale^2
//
// The same pattern extends the gauge connection to the diagonal as a linear
// bias `-gauge[i] * 2^k / scale`. This was verified numerically in
// `.agents/results/issue-1464-qubo-verification.py` across the 5R1C, 9R4C,
// and flat manifold scenes — see `tests.rs` for the round-trip assertion.

pub mod qubo_mapping;

#[cfg(feature = "dwave")]
pub mod dwave_client;

pub mod qubo_scaling;
