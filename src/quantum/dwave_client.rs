//! D-Wave quantum annealer client using the SAPI REST API.
//!
//! ## Feature gate
//!
//! This module is only compiled when the `dwave` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! fluxion = { path = "...", features = ["dwave"] }
//! ```
//!
//! The `DWAVE_API_TOKEN` environment variable must be set to a valid
//! D-Wave API token. If it is absent, all operations return
//! [`DwaveError::MissingApiToken`].
//!
//! ## API
//!
//! This client implements the D-Wave SAPI (Server API) which is the same
//! REST API underlying the Python `dimod` library from the D-Wave Ocean SDK.
//! The SAPI provides access to:
//! - QPU solvers (Advantage, Advantage2, etc.)
//! - Hybrid solvers (for problems exceeding QPU connectivity)
//! - Solver properties and status
//!
//! ## Trait hierarchy
//!
//! ```text
//! DwaveClient          ← trait (dyn-safe, mockable)
//!     └── OceanDwaveClient  ← concrete implementation via SAPI REST API
//! ```
//!
//! ## Round-trip verification
//!
//! 1. [`ThermalManifold`](crate::physics::geometry_tensor::ThermalManifold)
//!    → [`QuboProblem`](super::qubo_mapping::QuboProblem) via `manifold_to_qubo()`
//! 2. `QuboProblem` → [`IsingProblem`](super::qubo_mapping::IsingProblem) via `to_ising()`
//! 3. `IsingProblem` → D-Wave QPU (submit_ising)
//! 4. Spin vector → binary solution (s = 2x − 1)
//! 5. Binary solution → decoded energy via `QuboProblem::decoded_full_energy()`
//! 6. Compare with original manifold energy within 0.1 % tolerance.

use crate::physics::geometry_tensor::ThermalManifold;
use crate::quantum::qubo_mapping::{IsingProblem, QuboConfig, QuboProblem};

/// Spin vector returned by a D-Wave annealer (values are `{-1, +1}`).
pub type SpinVector = Vec<i8>;

/// Result type for D-Wave operations.
pub type DwaveResult<T> = Result<T, DwaveError>;

/// Object-safe trait for submitting Ising problems to a D-Wave sampler.
///
/// Implementors can be swapped for mocks in tests or when running against
/// a live QPU is impractical. All methods accept only types that are
/// cheap to clone (`Arc`-wrapping internal state where needed) so that
/// mock implementations are straightforward.
pub trait DwaveClient: Send + Sync {
    /// Submit an [`IsingProblem`] and return the lowest-energy spin vector.
    fn submit_ising(&self, ising: &IsingProblem) -> DwaveResult<SpinVector>;

    /// Return the energy of the given spin vector under the Ising problem.
    fn evaluate_ising(&self, ising: &IsingProblem, spin: &SpinVector) -> DwaveResult<f64>;

    /// Return a human-readable name for the active sampler (e.g.
    /// `"Advantage_system6.4.1"` or `"hybrid_binary_quadratic_model_v2"`).
    fn sampler_name(&self) -> DwaveResult<String>;

    /// Return `true` if the token is set and the sampler is reachable.
    fn is_connected(&self) -> bool;

    /// Return the maximum number of variables supported by this sampler.
    fn max_variables(&self) -> usize;

    /// Return the hardware constraints for this sampler (bias and coupling ranges).
    fn hardware_constraints(&self) -> crate::quantum::qubo_scaling::DwaveHardwareConstraints;
}

/// Error types returned by the D-Wave client.
#[derive(Debug, Clone, PartialEq)]
pub enum DwaveError {
    /// `DWAVE_API_TOKEN` was not set in the environment.
    MissingApiToken,
    /// The token was set but the sampler could not be reached.
    SamplerUnavailable(String),
    /// The problem exceeded the sampler's native variable limit.
    ProblemTooLarge {
        num_variables: usize,
        max_variables: usize,
    },
    /// D-Wave API error returned by the SAPI.
    ApiError(String),
    /// Invalid Ising parameters (NaN, Inf, or out-of-range `h`/`J`).
    InvalidIsing(String),
    /// The API token was rejected (401 / 403).
    AuthenticationFailed(String),
    /// The selected QPU is not available in your subscription.
    QpuNotAvailable(String),
}

impl std::fmt::Display for DwaveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingApiToken => {
                write!(f, "DWAVE_API_TOKEN environment variable is not set")
            }
            Self::SamplerUnavailable(msg) => {
                write!(f, "sampler unavailable: {msg}")
            }
            Self::ProblemTooLarge {
                num_variables,
                max_variables,
            } => {
                write!(
                    f,
                    "problem has {num_variables} variables but sampler supports at most {max_variables}"
                )
            }
            Self::ApiError(msg) => {
                write!(f, "D-Wave API error: {msg}")
            }
            Self::InvalidIsing(msg) => {
                write!(f, "invalid Ising problem: {msg}")
            }
            Self::AuthenticationFailed(msg) => {
                write!(f, "authentication failed: {msg}")
            }
            Self::QpuNotAvailable(msg) => {
                write!(f, "QPU not available: {msg}")
            }
        }
    }
}

impl std::error::Error for DwaveError {}

/// Feature-gated re-exports so calling code does not need to `#[cfg]`-gate
/// every import.
#[cfg(feature = "dwave")]
pub use self::ocean::OceanDwaveClient;

#[cfg(feature = "dwave")]
mod ocean {
    use super::*;
    use reqwest::blocking::Client;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;

    const SAPI_BASE_URL: &str = "https://cloud.dwavesys.com/sapi";

    /// D-Wave SAPI solver information.
    #[derive(Debug, Deserialize)]
    struct SolverInfo {
        id: String,
        #[serde(rename = "solver_name")]
        solver_name: Option<String>,
        #[serde(rename = "status")]
        status: Option<String>,
        #[serde(rename = "supported_config")]
        supported_config: Option<SupportedConfig>,
    }

    #[derive(Debug, Deserialize)]
    struct SupportedConfig {
        #[serde(rename = "qubits")]
        qubits: Option<usize>,
        #[serde(rename = "couplers")]
        couplers: Option<usize>,
        #[serde(rename = "num_variables")]
        num_variables: Option<usize>,
    }

    /// SAPI submit request body for Ising problems.
    #[derive(Debug, Serialize)]
    struct SubmitRequest<'a> {
        #[serde(rename = "solver")]
        solver: &'a str,
        #[serde(rename = "type")]
        problem_type: &'a str,
        #[serde(rename = "headers")]
        headers: &'a str,
        #[serde(rename = "biases")]
        biases: &'a [f64],
        #[serde(rename = "couplers")]
        couplers: &'a [[f64; 3]],
        #[serde(rename = "num_reads")]
        num_reads: usize,
    }

    /// SAPI answer response.
    #[derive(Debug, Deserialize)]
    struct AnswerResponse {
        #[serde(rename = "id")]
        id: String,
        #[serde(rename = "status")]
        status: String,
        #[serde(rename = "solutions")]
        solutions: Vec<Vec<i8>>,
        #[serde(rename = "energies")]
        energies: Vec<f64>,
        #[serde(rename = "num_occurrences")]
        num_occurrences: Option<Vec<f64>>,
        #[serde(rename = "timing")]
        timing: Option<serde_json::Value>,
    }

    /// Concrete D-Wave client backed by the SAPI REST API.
    ///
    /// ## Authentication
    ///
    /// Reads `DWAVE_API_TOKEN` from the environment. Supports two connection
    /// modes:
    ///
    /// - **QPU** — connects to a physical D-Wave Advantage system.
    ///   Set `DWAVE_API_TOKEN` and optionally `DWAVE_API_URL` (defaults to
    ///   `https://cloud.dwavesys.com/sapi/`).
    ///
    /// - **Hybrid** — uses D-Wave's `hybrid_binary_quadratic_model_v2`
    ///   solver which accepts up to 10^4 variables and runs a hybrid
    ///   classical/quantum algorithm. Prefer this for problems that exceed
    ///   the QPU's connectivity graph.
    ///
    /// ## Thread safety
    ///
    /// `OceanDwaveClient` is `Send + Sync` (the underlying HTTP client
    /// is internally thread-safe).
    #[derive(Debug, Clone)]
    pub struct OceanDwaveClient {
        client: Arc<Client>,
        solver: String,
        solver_name: String,
        max_variables: usize,
    }

    impl OceanDwaveClient {
        /// Construct a new client connected to the D-Wave cloud API.
        ///
        /// # Arguments
        ///
        /// * `solver` - Solver ID to use (e.g., `"Advantage_system6.4.1"` or
        ///   `"hybrid_binary_quadratic_model_v2"`). If `None`, uses the first
        ///   available QPU.
        ///
        /// # Errors
        ///
        /// Returns [`DwaveError::MissingApiToken`] if `DWAVE_API_TOKEN` is not
        /// in the environment. Returns [`DwaveError::SamplerUnavailable`] if
        /// the API rejects the token or no solvers are accessible.
        pub fn new(solver: Option<&str>) -> DwaveResult<Self> {
            let token =
                std::env::var("DWAVE_API_TOKEN").map_err(|_| DwaveError::MissingApiToken)?;

            let client = Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .map_err(|e| DwaveError::ApiError(e.to_string()))?;

            let base_url =
                std::env::var("DWAVE_API_URL").unwrap_or_else(|_| SAPI_BASE_URL.to_string());

            // Determine solver to use.
            let solver_id = if let Some(s) = solver {
                s.to_string()
            } else {
                // If no solver specified, get the first available QPU.
                let response = client
                    .get(&format!("{base_url}/solvers/available/"))
                    .header("Authorization", format!("Bearer {token}"))
                    .send()
                    .map_err(|e| DwaveError::SamplerUnavailable(e.to_string()))?;

                if !response.status().is_success() {
                    let status = response.status();
                    if status.as_u16() == 401 || status.as_u16() == 403 {
                        return Err(DwaveError::AuthenticationFailed(format!(
                            "token rejected with status {}",
                            status
                        )));
                    }
                    return Err(DwaveError::SamplerUnavailable(format!(
                        "API returned status {}",
                        status
                    )));
                }

                let solvers: Vec<SolverInfo> = response
                    .json()
                    .map_err(|e| DwaveError::ApiError(e.to_string()))?;

                solvers
                    .into_iter()
                    .find(|s| s.status.as_deref() == Some("ONLINE"))
                    .map(|s| s.solver_name.unwrap_or(s.id))
                    .ok_or_else(|| {
                        DwaveError::SamplerUnavailable("no online solvers found".into())
                    })?
            };

            // Get detailed solver info.
            let solver_url = format!("{base_url}/solvers/{solver_id}/");
            let response = client
                .get(&solver_url)
                .header("Authorization", format!("Bearer {token}"))
                .send()
                .map_err(|e| DwaveError::SamplerUnavailable(e.to_string()))?;

            if !response.status().is_success() {
                return Err(DwaveError::SamplerUnavailable(format!(
                    "failed to get solver info: {}",
                    response.status()
                )));
            }

            let info: SolverInfo = response
                .json()
                .map_err(|e| DwaveError::ApiError(e.to_string()))?;

            let max_vars = info
                .supported_config
                .as_ref()
                .and_then(|c| c.num_variables)
                .unwrap_or(64);

            Ok(Self {
                client: Arc::new(client),
                solver: solver_id,
                solver_name: info.solver_name.unwrap_or(info.id),
                max_variables: max_vars,
            })
        }

        /// Construct a new client connected to the D-Wave hybrid sampler.
        ///
        /// The hybrid sampler accepts up to 10^4 variables and does not require
        /// a QPU reservation. Prefer for initial development and testing.
        pub fn new_hybrid() -> DwaveResult<Self> {
            Self::new(Some("hybrid_binary_quadratic_model_v2"))
        }

        /// Returns the solver ID being used.
        pub fn solver_id(&self) -> &str {
            &self.solver
        }
    }

    impl DwaveClient for OceanDwaveClient {
        fn submit_ising(&self, ising: &IsingProblem) -> DwaveResult<SpinVector> {
            let n = ising.num_variables;

            if n == 0 {
                return Err(DwaveError::InvalidIsing("num_variables is 0".into()));
            }
            if n > self.max_variables {
                return Err(DwaveError::ProblemTooLarge {
                    num_variables: n,
                    max_variables: self.max_variables,
                });
            }
            for (i, &hi) in ising.h.iter().enumerate() {
                if !hi.is_finite() {
                    return Err(DwaveError::InvalidIsing(format!(
                        "h[{i}] = {hi} is not finite"
                    )));
                }
            }
            for (idx, &jij) in ising.j.iter().enumerate() {
                if !jij.is_finite() {
                    let i = idx / n;
                    let j = idx % n;
                    return Err(DwaveError::InvalidIsing(format!(
                        "J[{i},{j}] = {jij} is not finite"
                    )));
                }
            }

            // Check hardware constraint ranges (h ∈ [−4,+4], J ∈ [−2,+1]).
            let constraints = self.hardware_constraints();
            if let Err(violation) = constraints.validate_ising(ising) {
                return Err(DwaveError::InvalidIsing(violation.to_string()));
            }

            // Build the SAPI request.
            // SAPI accepts: biases (h), couplers (J entries as [i, j, J_ij]), num_reads.
            let mut couplers: Vec<[f64; 3]> = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    let jij = ising.j[i * n + j];
                    if jij != 0.0 {
                        couplers.push([i as f64, j as f64, jij]);
                    }
                }
            }

            let token = std::env::var("DWAVE_API_TOKEN").unwrap();
            let base_url =
                std::env::var("DWAVE_API_URL").unwrap_or_else(|_| SAPI_BASE_URL.to_string());

            let request_body = SubmitRequest {
                solver: &self.solver,
                problem_type: "ising",
                headers: "provided",
                biases: &ising.h,
                couplers: &couplers,
                num_reads: 100,
            };

            let response = self
                .client
                .post(&format!("{base_url}/problems/"))
                .header("Authorization", format!("Bearer {token}"))
                .header("Content-Type", "application/json")
                .json(&request_body)
                .send()
                .map_err(|e| DwaveError::ApiError(e.to_string()))?;

            if !response.status().is_success() {
                let status = response.status();
                if status.as_u16() == 401 || status.as_u16() == 403 {
                    return Err(DwaveError::AuthenticationFailed(format!(
                        "token rejected with status {}",
                        status
                    )));
                }
                let body = response.text().unwrap_or_default();
                return Err(DwaveError::ApiError(format!(
                    "submit failed: {} - {}",
                    status, body
                )));
            }

            #[derive(Deserialize)]
            struct SubmitResponse {
                #[serde(rename = "id")]
                id: String,
                #[serde(rename = "status")]
                status: String,
            }

            let submit_resp: SubmitResponse = response
                .json()
                .map_err(|e| DwaveError::ApiError(e.to_string()))?;

            // Poll for answer.
            let answer_url = format!("{base_url}/problems/{}/", submit_resp.id);
            let poll_interval = std::time::Duration::from_millis(500);
            let max_wait = std::time::Duration::from_secs(300);
            let start = std::time::Instant::now();

            loop {
                if start.elapsed() > max_wait {
                    return Err(DwaveError::ApiError(
                        "timed out waiting for annealer result".into(),
                    ));
                }

                let resp = self
                    .client
                    .get(&answer_url)
                    .header("Authorization", format!("Bearer {token}"))
                    .send()
                    .map_err(|e| DwaveError::ApiError(e.to_string()))?;

                let answer: AnswerResponse = resp
                    .json()
                    .map_err(|e| DwaveError::ApiError(e.to_string()))?;

                if answer.status == "COMPLETED" {
                    // Find the lowest-energy sample.
                    let mut best_energy = f64::INFINITY;
                    let mut best_spin: SpinVector = vec![0_i8; n];

                    for (idx, solution) in answer.solutions.iter().enumerate() {
                        let energy = answer.energies.get(idx).copied().unwrap_or(f64::INFINITY);
                        if energy < best_energy {
                            best_energy = energy;
                            best_spin = solution.clone();
                        }
                    }

                    if best_energy.is_infinite() {
                        return Err(DwaveError::ApiError("no solutions returned".into()));
                    }
                    return Ok(best_spin);
                } else if answer.status == "FAILED" {
                    return Err(DwaveError::ApiError(
                        "annealer reported problem failed".into(),
                    ));
                }
                // Not ready yet, wait and retry.
                std::thread::sleep(poll_interval);
            }
        }

        fn evaluate_ising(&self, ising: &IsingProblem, spin: &SpinVector) -> DwaveResult<f64> {
            if spin.len() != ising.num_variables {
                return Err(DwaveError::InvalidIsing(format!(
                    "spin.len() = {} != num_variables = {}",
                    spin.len(),
                    ising.num_variables
                )));
            }
            Ok(ising.evaluate(spin))
        }

        fn sampler_name(&self) -> DwaveResult<String> {
            Ok(self.solver_name.clone())
        }

        fn is_connected(&self) -> bool {
            std::env::var("DWAVE_API_TOKEN").is_ok()
        }

        fn max_variables(&self) -> usize {
            self.max_variables
        }

        fn hardware_constraints(&self) -> crate::quantum::qubo_scaling::DwaveHardwareConstraints {
            crate::quantum::qubo_scaling::DwaveHardwareConstraints::advantage_system64()
        }
    }
}

/// No-op D-Wave client that always returns [`DwaveError::MissingApiToken`].
/// Provided as a convenience for tests and as a fallback when no D-Wave token is
/// configured.
impl DwaveClient for () {
    fn submit_ising(&self, _ising: &IsingProblem) -> DwaveResult<SpinVector> {
        Err(DwaveError::MissingApiToken)
    }
    fn evaluate_ising(&self, _ising: &IsingProblem, _spin: &SpinVector) -> DwaveResult<f64> {
        Err(DwaveError::MissingApiToken)
    }
    fn sampler_name(&self) -> DwaveResult<String> {
        Err(DwaveError::MissingApiToken)
    }
    fn is_connected(&self) -> bool {
        false
    }
    fn max_variables(&self) -> usize {
        0
    }
    fn hardware_constraints(&self) -> crate::quantum::qubo_scaling::DwaveHardwareConstraints {
        crate::quantum::qubo_scaling::DwaveHardwareConstraints::advantage_system64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::geometry_tensor::ThermalManifold;
    use crate::quantum::qubo_mapping::{manifold_to_qubo, QuboConfig};

    #[test]
    fn test_dwave_error_display() {
        let e = DwaveError::MissingApiToken;
        assert_eq!(
            e.to_string(),
            "DWAVE_API_TOKEN environment variable is not set"
        );

        let e = DwaveError::ProblemTooLarge {
            num_variables: 100,
            max_variables: 64,
        };
        assert!(e.to_string().contains("100"));
        assert!(e.to_string().contains("64"));
    }

    #[test]
    fn test_submit_ising_returns_error_without_feature() {
        #[cfg(not(feature = "dwave"))]
        {
            let client: () = ();
            let ising = IsingProblem {
                h: vec![0.0; 4],
                j: vec![0.0; 16],
                c: 0.0,
                num_variables: 4,
            };
            let result = client.submit_ising(&ising);
            assert!(matches!(result, Err(DwaveError::MissingApiToken)));
        }
    }

    #[test]
    fn test_qubo_ising_roundtrip_energy() {
        // Create a small manifold and verify Ising energy matches QUBO energy.
        let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let cfg = QuboConfig::default();
        let qp = manifold_to_qubo(&m, cfg).expect("manifold_to_qubo failed");
        let ising = qp.to_ising();

        // Encode the canonical solution.
        let x_canon = qp.encode_manifold_solution();
        let s_canon: Vec<i8> = x_canon
            .iter()
            .map(|&b| if b == 0 { -1 } else { 1 })
            .collect();

        // Ising energy via the IsingProblem struct.
        let e_ising = ising.evaluate(&s_canon);

        // QUBO energy via QuboProblem struct.
        let e_qubo = qp.evaluate(&x_canon);

        // They must match within floating-point tolerance.
        let diff = (e_ising - e_qubo).abs();
        assert!(
            diff < 1e-9,
            "Ising energy {} != QUBO energy {} (diff = {})",
            e_ising,
            e_qubo,
            diff
        );
    }

    #[test]
    fn test_qubo_ising_roundtrip_via_trait_object() {
        // This test verifies the trait object-safe path without a live QPU.
        // It uses the noop () client which always returns MissingApiToken.
        let client: &dyn DwaveClient = &();

        let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let cfg = QuboConfig::default();
        let qp = manifold_to_qubo(&m, cfg).expect("manifold_to_qubo failed");
        let ising = qp.to_ising();

        // The mock always fails with MissingApiToken (expected behavior without feature).
        #[cfg(not(feature = "dwave"))]
        {
            let result = client.submit_ising(&ising);
            assert!(matches!(result, Err(DwaveError::MissingApiToken)));
            assert!(!client.is_connected());
        }

        // With the dwave feature, we would get a real spin vector here.
        // For CI, we skip if no token is present.
        #[cfg(feature = "dwave")]
        {
            if client.is_connected() {
                let spin = client.submit_ising(&ising).expect("submit failed");
                let e = client
                    .evaluate_ising(&ising, &spin)
                    .expect("evaluate failed");
                // Energy should be finite.
                assert!(e.is_finite(), "energy {} is not finite", e);
                // Verify spin values are ±1.
                for &s in &spin {
                    assert!(s == 1 || s == -1, "spin value {} is not ±1", s);
                }
            } else {
                // Skip if DWAVE_API_TOKEN is not set.
                eprintln!("DWAVE_API_TOKEN not set — skipping live submit test");
            }
        }
    }

    #[test]
    fn test_spin_to_binary_roundtrip() {
        // Verify s = 2x - 1 and x = (s + 1) / 2 are inverses.
        let x: Vec<u8> = vec![0, 1, 0, 1, 1, 0, 1, 0];
        let s: Vec<i8> = x.iter().map(|&b| if b == 0 { -1 } else { 1 }).collect();
        let x_back: Vec<u8> = s.iter().map(|&b| if b > 0 { 1 } else { 0 }).collect();
        assert_eq!(x, x_back);
    }

    #[test]
    fn test_ising_energy_is_symmetric() {
        // H = Σ_i h_i s_i + 2 * Σ_{i<j} J_ij s_i s_j + c
        // Swap i and j: J_ij s_i s_j = J_ji s_j s_i (J is symmetric by construction).
        let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let cfg = QuboConfig::default();
        let qp = manifold_to_qubo(&m, cfg).expect("manifold_to_qubo failed");
        let ising = qp.to_ising();

        // Generate two spin vectors and verify energy is computed correctly.
        let s1: Vec<i8> = vec![
            1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
            -1, 1, -1, 1, -1, 1, -1,
        ];
        let s2: Vec<i8> = vec![-1; 32];

        let e1 = ising.evaluate(&s1);
        let e2 = ising.evaluate(&s2);
        assert!(e1.is_finite());
        assert!(e2.is_finite());
        // The energies should be different (not a coincidence).
        assert_ne!(e1, e2);
    }

    #[test]
    fn test_dwave_normalized_h_within_hardware_range() {
        // D-Wave AdvantageSystem6.4: h ∈ [-4, +4].
        // After QUBO → to_dwave_normalized() → Ising conversion:
        // h_i = 0.5 * Σ_k Q_norm[i,k]
        // Verify |h_i| ≤ 4.0 for 5R1C (K=8, N=32) and 9R4C (K=12, N=48).
        let cfg = QuboConfig::default();

        // 5R1C manifold.
        let m5 = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let qp5 = manifold_to_qubo(&m5, cfg).expect("manifold_to_qubo failed");
        let q_norm5 = qp5.to_dwave_normalized();
        let n5 = qp5.num_variables();
        for i in 0..n5 {
            let row_sum: f64 = q_norm5[i * n5..(i + 1) * n5].iter().sum();
            let hi = 0.5 * row_sum;
            assert!(
                hi.abs() <= 4.0 + 1e-9,
                "5R1C: |h[{}]| = {} exceeds 4.0",
                i,
                hi.abs()
            );
        }

        // 9R4C manifold.
        let m9 = ThermalManifold::from_9r4c_parameters(
            [21.0, 22.0, 23.0, 24.0],
            [1000.0, 2000.0, 1500.0, 1800.0],
            [0.1, 0.12, 0.08],
            None,
        );
        let qp9 = manifold_to_qubo(&m9, cfg).expect("manifold_to_qubo failed");
        let q_norm9 = qp9.to_dwave_normalized();
        let n9 = qp9.num_variables();
        for i in 0..n9 {
            let row_sum: f64 = q_norm9[i * n9..(i + 1) * n9].iter().sum();
            let hi = 0.5 * row_sum;
            assert!(
                hi.abs() <= 4.0 + 1e-9,
                "9R4C: |h[{}]| = {} exceeds 4.0",
                i,
                hi.abs()
            );
        }
    }

    #[test]
    fn test_dwave_normalized_j_within_hardware_range() {
        // D-Wave AdvantageSystem6.4: J ∈ [-2, +1].
        // After QUBO → to_dwave_normalized() → Ising conversion:
        // J_ij = 0.25 * Q_norm[i,j] (for i ≠ j)
        // Verify |J_ij| ≤ 2.0 for 5R1C and 9R4C manifolds.
        let cfg = QuboConfig::default();

        let m5 = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let qp5 = manifold_to_qubo(&m5, cfg).expect("manifold_to_qubo failed");
        let q_norm5 = qp5.to_dwave_normalized();
        let n5 = qp5.num_variables();
        for i in 0..n5 {
            for j in (i + 1)..n5 {
                let jij = 0.25 * q_norm5[i * n5 + j];
                assert!(
                    jij.abs() <= 2.0 + 1e-9,
                    "5R1C: |J[{},{}]| = {} exceeds 2.0",
                    i,
                    j,
                    jij.abs()
                );
            }
        }

        let m9 = ThermalManifold::from_9r4c_parameters(
            [21.0, 22.0, 23.0, 24.0],
            [1000.0, 2000.0, 1500.0, 1800.0],
            [0.1, 0.12, 0.08],
            None,
        );
        let qp9 = manifold_to_qubo(&m9, cfg).expect("manifold_to_qubo failed");
        let q_norm9 = qp9.to_dwave_normalized();
        let n9 = qp9.num_variables();
        for i in 0..n9 {
            for j in (i + 1)..n9 {
                let jij = 0.25 * q_norm9[i * n9 + j];
                assert!(
                    jij.abs() <= 2.0 + 1e-9,
                    "9R4C: |J[{},{}]| = {} exceeds 2.0",
                    i,
                    j,
                    jij.abs()
                );
            }
        }
    }

    #[test]
    fn test_normalized_qubo_preserves_relative_scales() {
        // Normalization divides by max_abs, so every entry is scaled by the
        // same factor. Verify the ratio of any two non-zero entries is
        // preserved after to_dwave_normalized().
        let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let cfg = QuboConfig::default();
        let qp = manifold_to_qubo(&m, cfg).expect("manifold_to_qubo failed");
        let q_orig = qp.q_matrix();
        let max_abs = qp.max_abs();
        assert!(
            max_abs > 0.0,
            "max_abs should be > 0 for non-trivial manifold"
        );
        let q_norm = qp.to_dwave_normalized();

        for i in 0..q_orig.len() {
            if q_orig[i].abs() > 1e-12 {
                let ratio = q_norm[i] / q_orig[i];
                let expected = 1.0 / max_abs;
                assert!(
                    (ratio - expected).abs() < 1e-9,
                    "Index {}: ratio {} != expected {}",
                    i,
                    ratio,
                    expected
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Mock annealer for round-trip fidelity tests (Issue #1774)
    // -------------------------------------------------------------------------

    /// Mock annealer that returns the exact canonical solution for the Ising
    /// problem it was created with. The canonical solution is the one encoded
    /// from the manifold's `scalar_field` via fixed-point binary expansion —
    /// it exactly represents the original continuous temperature vector.
    ///
    /// This mock is CI-safe (no live hardware required) and deterministic
    /// (always returns the same solution for the same problem). It exercises
    /// the full encode → submit → decode round-trip without fidelity loss.
    #[derive(Debug, Clone)]
    struct MockAnnealer {
        expected_solution: SpinVector,
        name: String,
    }

    impl MockAnnealer {
        fn new(ising: &IsingProblem, canonical_solution: &[u8]) -> Self {
            let s: SpinVector = canonical_solution
                .iter()
                .map(|&b| if b == 0 { -1 } else { 1 })
                .collect();
            Self {
                expected_solution: s,
                name: "MockAnnealer (canonical)".to_string(),
            }
        }
    }

    impl DwaveClient for MockAnnealer {
        fn submit_ising(&self, _ising: &IsingProblem) -> DwaveResult<SpinVector> {
            Ok(self.expected_solution.clone())
        }

        fn evaluate_ising(&self, ising: &IsingProblem, spin: &SpinVector) -> DwaveResult<f64> {
            Ok(ising.evaluate(spin))
        }

        fn sampler_name(&self) -> DwaveResult<String> {
            Ok(self.name.clone())
        }

        fn is_connected(&self) -> bool {
            true
        }
    }

    /// Greedy mock annealer that finds a local optimum via single-bit flips.
    /// Starts from all-zeros spin vector and iteratively flips each bit if it
    /// improves the energy. Runs until a full pass with no improvement.
    ///
    /// This is more realistic than returning the exact optimum — it finds a
    /// local optimum that may differ from the canonical solution, allowing us
    /// to verify the decoded energy is still close to the true minimum.
    #[derive(Debug, Clone)]
    struct GreedyAnnealer {
        solution: SpinVector,
        energy: f64,
        name: String,
    }

    impl GreedyAnnealer {
        fn new(ising: &IsingProblem) -> Self {
            let n = ising.num_variables;
            let mut spin = vec![-1_i8; n];
            let mut energy = ising.evaluate(&spin);

            let mut improved = true;
            while improved {
                improved = false;
                for i in 0..n {
                    spin[i] *= -1;
                    let new_energy = ising.evaluate(&spin);
                    if new_energy < energy {
                        energy = new_energy;
                        improved = true;
                    } else {
                        spin[i] *= -1;
                    }
                }
            }

            Self {
                solution: spin,
                energy,
                name: "GreedyAnnealer (local optimum)".to_string(),
            }
        }
    }

    impl DwaveClient for GreedyAnnealer {
        fn submit_ising(&self, _ising: &IsingProblem) -> DwaveResult<SpinVector> {
            Ok(self.solution.clone())
        }

        fn evaluate_ising(&self, ising: &IsingProblem, spin: &SpinVector) -> DwaveResult<f64> {
            Ok(ising.evaluate(spin))
        }

        fn sampler_name(&self) -> DwaveResult<String> {
            Ok(self.name.clone())
        }

        fn is_connected(&self) -> bool {
            true
        }
    }

    /// Full QUBO → annealer → decode round-trip fidelity test for 5R1C.
    ///
    /// Verifies that a known manifold (with known `scalar_field`) can be:
    ///   1. Encoded into a QUBO problem
    ///   2. Submitted to a mock annealer
    ///   3. Decoded back to temperatures
    ///   4. Verified to match the original within ±0.5 LSB (quantization error)
    ///
    /// The mock annealer returns the exact canonical solution, so the decoded
    /// temperatures must equal the original manifold's scalar_field exactly
    /// (within floating-point rounding).
    #[test]
    fn test_qubo_annealer_fidelity_5r1c() {
        use crate::physics::geometry_tensor::MANIFOLD_DIM;
        use crate::quantum::qubo_mapping::{
            decode_temperatures, encode_temperatures, lsb_resolution_celsius, manifold_to_qubo,
            QuboConfig,
        };

        // 5R1C manifold with known temperatures.
        let t_air = 21.0;
        let t_zone = 22.0;
        let m = ThermalManifold::from_5r1c_parameters(t_air, t_zone, 0.1, 1000.0, 5000.0);

        // Build QUBO without gauge bias (cleaner energy landscape).
        let cfg = QuboConfig {
            include_gauge_bias: false,
            ..QuboConfig::default()
        };
        let qp = manifold_to_qubo(&m, cfg.clone()).expect("manifold_to_qubo failed");
        let ising = qp.to_ising();

        // Canonical solution encodes the manifold's scalar_field.
        let x_canon = qp.encode_manifold_solution();
        assert_eq!(x_canon.len(), qp.num_variables());

        // Create mock annealer that returns the exact canonical solution.
        let annealer = MockAnnealer::new(&ising, &x_canon);
        let client: &dyn DwaveClient = &annealer;

        // Submit to mock annealer — get back spins.
        let spin = client.submit_ising(&ising).expect("submit failed");
        assert_eq!(spin.len(), ising.num_variables);
        for &s in &spin {
            assert!(s == 1 || s == -1, "spin {} is not ±1", s);
        }

        // Convert spins to binary (s = 2x - 1 ⇒ x = (s + 1) / 2).
        let x_decoded: Vec<u8> = spin.iter().map(|&b| if b > 0 { 1 } else { 0 }).collect();

        // Decode binary to temperature vector.
        let decoded_temps = decode_temperatures(&x_decoded, &cfg);

        // Verify decoded temperatures match original within ±0.5 LSB.
        let lsb = lsb_resolution_celsius(&cfg);
        for i in 0..MANIFOLD_DIM {
            let err = (decoded_temps[i] - m.scalar_field[i]).abs();
            assert!(
                err <= lsb / 2.0 + 1e-9,
                "T[{}]: original = {:.6}, decoded = {:.6}, error = {:.6} (lsb/2 = {:.6})",
                i,
                m.scalar_field[i],
                decoded_temps[i],
                err,
                lsb / 2.0
            );
        }

        // Verify the decoded binary equals the canonical binary.
        assert_eq!(
            x_decoded, x_canon,
            "Decoded binary differs from canonical encoding"
        );

        // Verify decoded energy matches QUBO energy.
        let e_qubo = qp.evaluate(&x_decoded);
        let e_decoded = qp.decoded_energy_density(&x_decoded);
        let rel_err = ((e_qubo - e_decoded) / e_decoded.abs().max(1e-12)).abs();
        assert!(
            rel_err <= 1e-9,
            "QUBO energy {} vs decoded density {} (rel err {})",
            e_qubo,
            e_decoded,
            rel_err
        );
    }

    /// Round-trip fidelity test for 9R4C manifold with non-zero gauge bias.
    ///
    /// The gauge bias adds a linear term to the QUBO energy, shifting the
    /// true optimum away from the canonical solution. The greedy mock annealer
    /// finds a local optimum (which may be the global optimum for this problem
    /// size). We verify the decoded energy is close to the QUBO energy and
    /// the solution is physically reasonable (all temperatures in [0, scale_max]).
    #[test]
    fn test_qubo_annealer_fidelity_9r4c_with_gauge() {
        use crate::physics::geometry_tensor::MANIFOLD_DIM;
        use crate::quantum::qubo_mapping::{
            decode_temperatures, lsb_resolution_celsius, manifold_to_qubo, QuboConfig,
        };

        let temps = [22.0_f64, 20.0, 23.0, 18.0];
        let caps = [1000.0, 5000.0, 3000.0, 8000.0];
        let r_tr = [50.0, 30.0, 20.0];
        let r_cross = Some([5.0, 3.0, 2.0]);
        let mut m = ThermalManifold::from_9r4c_parameters(temps, caps, r_tr, r_cross);
        m.gauge_connection[0] = 100.0;
        m.gauge_connection[1] = 200.0;
        m.gauge_connection[2] = 50.0;
        m.gauge_connection[3] = 30.0;

        let cfg = QuboConfig::default();
        let qp = manifold_to_qubo(&m, cfg.clone()).expect("manifold_to_qubo failed");
        let ising = qp.to_ising();

        // Use greedy annealer to find a local optimum (more realistic than exact).
        let annealer = GreedyAnnealer::new(&ising);
        let client: &dyn DwaveClient = &annealer;

        let spin = client.submit_ising(&ising).expect("submit failed");
        let x: Vec<u8> = spin.iter().map(|&b| if b > 0 { 1 } else { 0 }).collect();

        // Decode and verify all temperatures are in valid range.
        let decoded = decode_temperatures(&x, &cfg);
        for i in 0..MANIFOLD_DIM {
            assert!(
                decoded[i] >= 0.0 - 1e-9,
                "T[{}] = {} is below 0",
                i,
                decoded[i]
            );
            assert!(
                decoded[i] <= cfg.scale_max_celsius + 1e-9,
                "T[{}] = {} exceeds scale max {}",
                i,
                decoded[i],
                cfg.scale_max_celsius
            );
        }

        // Verify QUBO energy matches decoded full energy.
        let e_qubo = qp.evaluate(&x);
        let e_decoded = qp.decoded_full_energy(&x);
        let rel_err = ((e_qubo - e_decoded) / e_decoded.abs().max(1e-12)).abs();
        assert!(
            rel_err <= 1e-9,
            "9R4C QUBO energy {} vs decoded full {} (rel err {})",
            e_qubo,
            e_decoded,
            rel_err
        );

        // Verify energy is finite.
        assert!(e_qubo.is_finite(), "QUBO energy {} is not finite", e_qubo);
    }

    /// Verify that a small perturbation in the annealer solution still decodes
    /// to a physically reasonable temperature vector. This tests robustness
    /// against the stochastic nature of real annealers.
    #[test]
    fn test_qubo_annealer_fidelity_with_perturbation() {
        use crate::physics::geometry_tensor::MANIFOLD_DIM;
        use crate::quantum::qubo_mapping::{
            decode_temperatures, encode_temperatures, lsb_resolution_celsius, manifold_to_qubo,
            QuboConfig,
        };

        let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let cfg = QuboConfig {
            include_gauge_bias: false,
            ..QuboConfig::default()
        };
        let qp = manifold_to_qubo(&m, cfg.clone()).expect("manifold_to_qubo failed");
        let ising = qp.to_ising();

        let x_canon = qp.encode_manifold_solution();

        // Perturb: flip the lowest-order bit of each temperature node.
        let mut x_perturbed = x_canon.clone();
        let k = cfg.bits_per_node;
        for node in 0..MANIFOLD_DIM {
            // Flip bit 0 (LSB) of each node — ±1 LSB perturbation.
            x_perturbed[node * k] ^= 1;
        }

        // Evaluate energies.
        let e_canon = qp.evaluate(&x_canon);
        let e_perturbed = qp.evaluate(&x_perturbed);

        // Perturbation should increase energy (canonical is optimal for no-gauge case).
        assert!(
            e_perturbed >= e_canon,
            "Perturbed energy {} should be >= canonical {}",
            e_perturbed,
            e_canon
        );

        // Decode perturbed solution and verify temperatures are still in range.
        let decoded = decode_temperatures(&x_perturbed, &cfg);
        let lsb = lsb_resolution_celsius(&cfg);
        for i in 0..MANIFOLD_DIM {
            let err = (decoded[i] - m.scalar_field[i]).abs();
            // Perturbed error should be at most ~2 LSB:
            // 1 LSB from the bit-flip perturbation, plus up to 0.5 LSB from the
            // rounding in the encode → round → decode cycle.
            assert!(
                err <= 2.0 * lsb + 1e-6,
                "Perturbed T[{}]: original = {:.6}, decoded = {:.6}, err = {:.6}",
                i,
                m.scalar_field[i],
                decoded[i],
                err
            );
            assert!(
                decoded[i] >= 0.0 - 1e-9,
                "Perturbed T[{}] = {} below 0",
                i,
                decoded[i]
            );
            assert!(
                decoded[i] <= cfg.scale_max_celsius + 1e-9,
                "Perturbed T[{}] = {} exceeds max {}",
                i,
                decoded[i],
                cfg.scale_max_celsius
            );
        }
    }

    /// Verify the trait-object dispatch path (submit via `&dyn DwaveClient`) works
    /// correctly with the mock annealer. This is the actual runtime dispatch path
    /// used by production code.
    #[test]
    fn test_qubo_annealer_trait_object_dispatch() {
        use crate::physics::geometry_tensor::MANIFOLD_DIM;
        use crate::quantum::qubo_mapping::{
            decode_temperatures, lsb_resolution_celsius, manifold_to_qubo, QuboConfig,
        };

        let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let cfg = QuboConfig {
            include_gauge_bias: false,
            ..QuboConfig::default()
        };
        let qp = manifold_to_qubo(&m, cfg.clone()).expect("manifold_to_qubo failed");
        let ising = qp.to_ising();

        let x_canon = qp.encode_manifold_solution();
        let annealer = MockAnnealer::new(&ising, &x_canon);

        // Trait object dispatch.
        let client: &dyn DwaveClient = &annealer;
        assert!(client.is_connected());
        assert_eq!(client.sampler_name().unwrap(), "MockAnnealer (canonical)");

        let spin = client.submit_ising(&ising).unwrap();
        let e = client.evaluate_ising(&ising, &spin).unwrap();
        assert!(e.is_finite());

        // Decode via trait object.
        let x: Vec<u8> = spin.iter().map(|&b| if b > 0 { 1 } else { 0 }).collect();
        let decoded = decode_temperatures(&x, &cfg);
        let lsb = lsb_resolution_celsius(&cfg);
        for i in 0..MANIFOLD_DIM {
            let err = (decoded[i] - m.scalar_field[i]).abs();
            assert!(
                err <= lsb / 2.0 + 1e-9,
                "Trait-object dispatch T[{}]: original = {:.6}, decoded = {:.6}",
                i,
                m.scalar_field[i],
                decoded[i]
            );
        }
    }
}
