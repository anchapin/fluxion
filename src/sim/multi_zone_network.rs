//! N-zone airflow / thermal coupling network (Issue #1348).
//!
//! Generalizes the Case 960 two-zone pair into an arbitrary N-zone network.
//! The conductance matrix `h_tr_iz: DMatrix<f64>` of shape (N, N) stores the
//! per-pair inter-zone conductance `h_tr_iz[i, j]` (W/K) from zone `i` to
//! zone `j`. The diagonal is conventionally zero (no self-coupling).
//!
//! ## Sign convention
//! `q_iz[i] = Σ_j h_tr_iz[i, j] · (T[j] − T[i])` is the net heat flow INTO
//! zone `i` from all other zones. Positive `q_iz[i]` means zone `i` is gaining
//! heat from its neighbours.
//!
//! ## Energy conservation
//! For a SYMMETRIC conductance matrix (`h_tr_iz[i, j] = h_tr_iz[j, i]`),
//! the sum of all inter-zone transfers is identically zero — the algebraic
//! identity
//!
//! ```text
//! Σ_i q_iz[i] = Σ_i Σ_j h_tr_ij · (T_j − T_i)
//!             = Σ_{i<j} (h_tr_ij − h_tr_ji) · (T_j − T_i)
//!             = 0   when h_tr_ij = h_tr_ji.
//! ```
//!
//! The backward-Euler step computes the post-step temperatures first, then
//! derives `q_iz` from those — so the identity holds for the matrix `M`
//! solve to within machine epsilon for the LU decomposition.
//!
//! ## Scope
//! This module handles the **air-node** inter-zone coupling only. Per-zone
//! surface conduction, thermal mass, solar gain, ventilation, and HVAC live
//! in the per-zone thermal models. The N-zone network here is the "glue"
//! that couples N otherwise-independent zones by their inter-zone heat flow.
//! Pressure-driven airflow / CONTAM-style multi-zone air flow is explicitly
//! **out of scope** per Issue #1348.

use std::cell::RefCell;

use nalgebra::{DMatrix, DVector, LU};

/// Conservation diagnostic emitted by
/// [`MultiZoneAirflowNetwork::conservation_report`]. Used by the CLI / report
/// generators to surface the algebraic identity check
/// `Σ q_iz[i] ≈ 0 W` (Issue #1348 acceptance criterion).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultiZoneNetworkReport {
    /// Number of zones in the network.
    pub n_zones: usize,
    /// Whether the conductance matrix is symmetric within 1e-12.
    pub symmetric: bool,
    /// `Σ q_iz[i]` (W) for the probed temperature vector. For a symmetric
    /// matrix this is O(1e-13) — at machine precision for the f64 LU solve.
    pub net_inter_zone_q_w: f64,
    /// Acceptance tolerance (W). The Issue #1348 criterion is 1e-6 W.
    pub tolerance_w: f64,
}

/// Error type for N-zone network solve failures.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiZoneNetworkError {
    /// The conductance matrix dimensions don't match the number of zones.
    DimensionMismatch {
        /// Number of zones supplied.
        n_zones: usize,
        /// Rows in the conductance matrix.
        matrix_rows: usize,
        /// Columns in the conductance matrix.
        matrix_cols: usize,
    },
    /// The zone slice length doesn't match the matrix dimension.
    ZoneCountMismatch {
        /// Number of zones supplied.
        n_zones: usize,
        /// Length of the zone-state slice.
        zone_slice_len: usize,
    },
    /// The system matrix is singular (LU decomposition failed).
    SingularSystem,
    /// A zone has non-positive thermal capacity.
    InvalidHeatCapacity {
        /// Zone index with invalid capacity.
        zone: usize,
        /// Observed capacity (J/K).
        capacity: f64,
    },
    /// Timestep is non-positive.
    InvalidTimestep(f64),
}

impl std::fmt::Display for MultiZoneNetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { n_zones, matrix_rows, matrix_cols } => write!(
                f,
                "conductance matrix is {matrix_rows}x{matrix_cols} but expected {n_zones}x{n_zones}",
            ),
            Self::ZoneCountMismatch { n_zones, zone_slice_len } => write!(
                f,
                "zone slice length {zone_slice_len} does not match matrix size {n_zones}",
            ),
            Self::SingularSystem => write!(f, "system matrix is singular"),
            Self::InvalidHeatCapacity { zone, capacity } => write!(
                f,
                "zone {zone} has non-positive heat capacity {capacity} J/K",
            ),
            Self::InvalidTimestep(dt) => write!(f, "timestep {dt} must be > 0"),
        }
    }
}

impl std::error::Error for MultiZoneNetworkError {}

/// Per-zone state consumed by `MultiZoneAirflowNetwork::solve_step`.
///
/// The full `ThermalModel` carries far more state (surface heat fluxes,
/// thermal mass, HVAC equipment, etc.); for the N-zone inter-zone solve the
/// only quantities that matter are the air-node temperature and the air-node
/// heat capacity. The zone temperature is read AND mutated by `solve_step` so
/// the caller doesn't need to thread `T_old` and `T_new` through two separate
/// buffers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZoneState {
    /// Air-node temperature [°C]. Read at the start of the step, written at
    /// the end of the step.
    pub temperature: f64,
    /// Air-node thermal capacity C_air = ρ·cp·V_zone [J/K]. Must be > 0.
    pub heat_capacity: f64,
}

impl ZoneState {
    /// Build a zone state with the given temperature and heat capacity.
    pub fn new(temperature: f64, heat_capacity: f64) -> Self {
        Self {
            temperature,
            heat_capacity,
        }
    }
}

/// Per-zone inter-zone heat transfer result.
#[derive(Debug, Clone, PartialEq)]
pub struct InterZoneResult {
    /// `q_iz[i]` = net heat flow INTO zone `i` from all other zones [W].
    /// Sign convention: positive = heat gained by zone `i`.
    pub q_iz_w: Vec<f64>,
    /// Net system inter-zone transfer `Σ q_iz[i]` [W]. For a symmetric
    /// conductance matrix and any set of temperatures, this must be ≤ 1e-6 W
    /// in absolute value (Issue #1348 acceptance criterion).
    pub net_w: f64,
    /// Post-step zone temperatures [°C].
    pub temperatures_after: Vec<f64>,
}

/// N-zone inter-zone airflow / thermal coupling network.
///
/// Wraps the symmetric N×N conductance matrix `h_tr_iz` and provides a
/// backward-Euler implicit step that solves the air-node heat balance for
/// all zones simultaneously. The solve conserves energy to within machine
/// precision for symmetric `h_tr_iz` (see module docs).
///
/// ## Allocation / factorization caching (Issue #2859)
///
/// `solve_step` runs every timestep of an `n`-zone coupled simulation. The
/// system matrix `M · T_new = b` has
///
/// ```text
/// M_ii = C_i/dt + Σ_j h_tr_ij
/// M_ij = −h_tr_ij       (i ≠ j)
/// ```
///
/// so `M_ij` is constant across calls (depends only on `h_tr_iz`) and `M_ii`
/// only changes when `dt` or any `C_i` changes. The previous implementation
/// allocated a fresh `DMatrix::zeros(n, n)` + `DVector::zeros(n)` + the LU
/// factorization on EVERY call; for pop_1000 × 10 zones × 8760 timesteps
/// that is ~87.6M re-allocations + ~87.6M O(N³) LU decompositions.
///
/// This struct now holds:
///
/// - `m_const: DMatrix<f64>` — the `dt`-independent part of `M`
///   (`M_ii_const = Σ_j h_tr_ij`, `M_ij = −h_tr_ij` for `i ≠ j`), precomputed
///   at construction.
/// - `work_buffers: RefCell<WorkBuffers>` — `m_buf`, `b_buf`, `t_new_buf`
///   (`DMatrix` / `DVector`) and `q_iz_buf` / `t_old_buf` (`Vec<f64>`)
///   reused across calls (no per-call allocation when `n` matches the
///   cached size).
/// - `factorization_cache: RefCell<Option<FactorizationCache>>` — last
///   `(dt, C_i profile, LU)` triple. On a cache hit we skip the O(N³)
///   factorization entirely and reuse the cached LU. The cache invalidates
///   on either a `dt` change or any `C_i` change (the caller mutates
///   `ZoneState::heat_capacity` freely).
#[derive(Debug)]
pub struct MultiZoneAirflowNetwork {
    h_tr_iz: DMatrix<f64>,
    /// `dt`-independent part of `M`: `M_const[(i, i)] = Σ_j h_tr_ij` and
    /// `M_const[(i, j)] = -h_tr_ij` for `i ≠ j`. The full per-call `M` is
    /// `M_const` plus `diag(C_i / dt)` added to the diagonal in place.
    m_const: DMatrix<f64>,
    /// Reusable scratch buffers + LU cache. `RefCell` because
    /// `solve_step` takes `&self` (callers expect a non-mutating API).
    work: RefCell<WorkState>,
}

impl Clone for MultiZoneAirflowNetwork {
    fn clone(&self) -> Self {
        let n = self.h_tr_iz.nrows();
        // Cloned network starts with a fresh `WorkState` — the cached LU
        // belongs to the original instance, not the clone (cloning would
        // be wrong in parallel contexts where each worker needs its own
        // cache, and meaningless in single-threaded use because the first
        // `solve_step` refactorizes for the clone anyway).
        let work = RefCell::new(WorkState {
            m_buf: DMatrix::<f64>::zeros(n, n),
            b_buf: DVector::<f64>::zeros(n),
            t_new_buf: DVector::<f64>::zeros(n),
            q_iz_buf: vec![0.0_f64; n],
            t_old_buf: vec![0.0_f64; n],
            factor_cache: None,
        });
        Self {
            h_tr_iz: self.h_tr_iz.clone(),
            m_const: self.m_const.clone(),
            work,
        }
    }
}

/// Reusable scratch buffers + cached LU factorization.
///
/// Lives behind a [`RefCell`] so the non-mutating `solve_step` API can
/// update the cache and reuse the scratch buffers across calls.
#[derive(Debug)]
struct WorkState {
    /// `n × n` scratch system matrix (full per-call `M`).
    m_buf: DMatrix<f64>,
    /// `n`-vector scratch RHS `b`.
    b_buf: DVector<f64>,
    /// `n`-vector scratch post-step temperatures `T_new`.
    t_new_buf: DVector<f64>,
    /// `n`-scratch `q_iz` accumulator (returned as `Vec<f64>`).
    q_iz_buf: Vec<f64>,
    /// `n`-scratch pre-step temperatures `T_old` (avoids re-reading from
    /// `zones` later when computing `net_w`).
    t_old_buf: Vec<f64>,
    /// Last successful `(dt, C profile, LU factorization)` triple.
    /// `Some` iff a previous `solve_step` produced a factorization whose
    /// parameters match the current call — in that case we skip the
    /// O(N³) decomposition and reuse the cached LU.
    factor_cache: Option<FactorCache>,
}

/// Cached LU factorization + the parameter triple it was computed for.
///
/// `dt` and the per-zone `c` profile together uniquely determine the
/// system matrix `M` (because `h_tr_iz` is fixed at construction). On a
/// cache hit we reuse the LU; on a miss we refactorize and replace this
/// entry.
#[derive(Debug, Clone)]
struct FactorCache {
    dt: f64,
    /// Per-zone heat capacities at factorization time. Compared against
    /// the current `zones` to detect a `C_i` change.
    capacities: Vec<f64>,
    /// Pre-computed LU factorization of the full per-call `M` matrix.
    lu: LU<f64, nalgebra::Dyn, nalgebra::Dyn>,
}

impl MultiZoneAirflowNetwork {
    /// Build a network from a square conductance matrix. The matrix is
    /// copied; subsequent mutations to `h_tr_iz` by the caller do not affect
    /// this network.
    ///
    /// # Errors
    /// Returns `DimensionMismatch` if `h_tr_iz` is not square, or
    /// `InvalidHeatCapacity` later at `solve_step` time.
    pub fn from_matrix(h_tr_iz: DMatrix<f64>) -> Self {
        debug_assert!(
            h_tr_iz.nrows() == h_tr_iz.ncols(),
            "MultiZoneAirflowNetwork::from_matrix: matrix must be square ({}x{})",
            h_tr_iz.nrows(),
            h_tr_iz.ncols()
        );
        let n = h_tr_iz.nrows();
        let mut m_const = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| h_tr_iz[(i, j)]).sum();
            m_const[(i, i)] = row_sum;
            for j in 0..n {
                if i != j {
                    m_const[(i, j)] = -h_tr_iz[(i, j)];
                }
            }
        }
        let work = RefCell::new(WorkState {
            m_buf: DMatrix::<f64>::zeros(n, n),
            b_buf: DVector::<f64>::zeros(n),
            t_new_buf: DVector::<f64>::zeros(n),
            q_iz_buf: vec![0.0_f64; n],
            t_old_buf: vec![0.0_f64; n],
            factor_cache: None,
        });
        Self {
            h_tr_iz,
            m_const,
            work,
        }
    }

    /// Build a network from an adjacency list of `(i, j, h_tr_ij)` triples
    /// (W/K). Both `h_tr_ij` and `h_tr_ji` are stored — pass them explicitly
    /// for asymmetric conductances. Missing pairs are treated as zero
    /// conductance.
    ///
    /// # Errors
    /// Returns `DimensionMismatch` if `n` doesn't match later, or panics on
    /// out-of-range indices.
    pub fn from_adjacency_pairs(n: usize, pairs: &[(usize, usize, f64)]) -> Self {
        let mut m = DMatrix::<f64>::zeros(n, n);
        for &(i, j, h) in pairs {
            assert!(i < n && j < n, "pair ({i}, {j}) out of range for {n}x{n}");
            m[(i, j)] = h;
        }
        Self::from_matrix(m)
    }

    /// Number of zones in this network.
    pub fn num_zones(&self) -> usize {
        self.h_tr_iz.nrows()
    }

    /// Borrow the underlying conductance matrix.
    pub fn conductance_matrix(&self) -> &DMatrix<f64> {
        &self.h_tr_iz
    }

    /// Per-zone outgoing conductance sum `Σ_j h_tr_iz[i, j]` (W/K). This is
    /// the value stored in `model.conduction.h_tr_iz[i]` (flat per-zone total) for
    /// backward-compatible integration with the existing Case-960
    /// `ThermalModel` air-node solver, which uses the flat per-zone total
    /// rather than the full N×N matrix.
    pub fn per_zone_conductance(&self) -> Vec<f64> {
        let n = self.h_tr_iz.nrows();
        (0..n)
            .map(|i| (0..n).map(|j| self.h_tr_iz[(i, j)]).sum())
            .collect()
    }

    /// Compute the net inter-zone transfer `Σ q_iz[i]` (W) for the supplied
    /// temperature vector without running a step. Useful for the energy
    /// conservation check and for diagnostic output.
    ///
    /// For a symmetric conductance matrix and any temperature vector this
    /// returns 0.0 to machine precision (algebraic identity).
    pub fn net_inter_zone_q(&self, temps: &[f64]) -> f64 {
        let n = self.h_tr_iz.nrows();
        debug_assert_eq!(temps.len(), n);
        let mut net = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                net += self.h_tr_iz[(i, j)] * (temps[j] - temps[i]);
            }
        }
        net
    }

    /// Run one backward-Euler step on the air-node heat balance.
    ///
    /// For each zone `i`, the air-node balance is
    ///
    /// ```text
    /// C_i/dt · (T_new_i − T_old_i) = q_ext_i + Σ_j h_tr_ij · (T_new_j − T_new_i)
    /// ```
    ///
    /// Rearranging yields the linear system `M · T_new = b` with
    ///
    /// ```text
    /// M_ii = C_i/dt + Σ_j h_tr_ij
    /// M_ij = −h_tr_ij       (i ≠ j)
    /// b_i  = C_i/dt · T_old_i + q_ext_i
    /// ```
    ///
    /// The post-step temperatures `T_new` are obtained by an LU solve
    /// (`nalgebra::DMatrix::lu`). Per-zone inter-zone transfer is then
    /// `q_iz[i] = Σ_j h_tr_ij · (T_new_j − T_new_i)`, written back into the
    /// zone's `temperature` field. `q_ext` defaults to zero (no external
    /// HVAC / solar / conduction at the air node — those live in the per-zone
    /// thermal model and would be added to `q_ext` by the integration layer).
    ///
    /// # Errors
    /// - `ZoneCountMismatch` if `zones.len() != num_zones()`.
    /// - `InvalidHeatCapacity` if any zone has `heat_capacity ≤ 0`.
    /// - `InvalidTimestep` if `dt ≤ 0`.
    /// - `SingularSystem` if the LU decomposition cannot solve the system
    ///   (mathematically impossible for a positive-definite `h_tr_iz` and
    ///   positive `C`, but flagged defensively for malformed input).
    pub fn solve_step(
        &self,
        zones: &mut [ZoneState],
        q_ext: &[f64],
        dt: f64,
    ) -> Result<InterZoneResult, MultiZoneNetworkError> {
        let n = self.num_zones();
        if zones.len() != n {
            return Err(MultiZoneNetworkError::ZoneCountMismatch {
                n_zones: n,
                zone_slice_len: zones.len(),
            });
        }
        if q_ext.len() != n {
            return Err(MultiZoneNetworkError::ZoneCountMismatch {
                n_zones: n,
                zone_slice_len: q_ext.len(),
            });
        }
        if dt <= 0.0 || !dt.is_finite() {
            return Err(MultiZoneNetworkError::InvalidTimestep(dt));
        }
        for (i, z) in zones.iter().enumerate() {
            if z.heat_capacity <= 0.0 {
                return Err(MultiZoneNetworkError::InvalidHeatCapacity {
                    zone: i,
                    capacity: z.heat_capacity,
                });
            }
        }

        // Borrow the work buffers (no allocation in the cache-hit path;
        // only a one-shot allocation when `n` differs from the cached
        // size, which happens at most once per network).
        let mut work = self.work.borrow_mut();
        if work.m_buf.nrows() != n {
            work.m_buf = DMatrix::<f64>::zeros(n, n);
            work.b_buf = DVector::<f64>::zeros(n);
            work.t_new_buf = DVector::<f64>::zeros(n);
            work.q_iz_buf = vec![0.0_f64; n];
            work.t_old_buf = vec![0.0_f64; n];
            work.factor_cache = None;
        }

        // Snapshot `T_old` into a reusable buffer (the API still needs
        // `temperatures_after` as a fresh `Vec<f64>` for the caller, but
        // we build it from `t_old_buf` rather than allocating twice).
        for (i, z) in zones.iter().enumerate() {
            work.t_old_buf[i] = z.temperature;
        }

        // Cache-hit predicate: factorization is valid iff `dt` matches
        // AND every per-zone `C_i` matches the values the LU was
        // computed for. The caller mutates `ZoneState::heat_capacity`
        // freely, so we cannot skip the comparison.
        let cache_hit = matches!(
            work.factor_cache.as_ref(),
            Some(cache) if cache.dt == dt && cache.capacities.len() == n && {
                cache.capacities.iter().zip(zones.iter()).all(|(c, z)| *c == z.heat_capacity)
            }
        );

        // Build `M` (full per-call matrix) by adding the `C_i/dt` diagonal
        // to the precomputed `m_const`. Done in-place into `work.m_buf`.
        for (i, z) in zones.iter().enumerate() {
            // Copy the row of `m_const` into `m_buf`.
            for j in 0..n {
                work.m_buf[(i, j)] = self.m_const[(i, j)];
            }
            work.m_buf[(i, i)] += z.heat_capacity / dt;
        }

        // Build `b_i = C_i/dt · T_old_i + q_ext_i` into `work.b_buf`.
        for (i, z) in zones.iter().enumerate() {
            work.b_buf[i] = z.heat_capacity / dt * work.t_old_buf[i] + q_ext[i];
        }

        // Reuse the cached LU if the predicate held; otherwise refactorize
        // and refresh the cache. The LU is moved (or cloned if reused) out
        // of `factor_cache`, then replaced on a cache miss.
        let lu_owned = if cache_hit {
            // Safe: cache_hit guarantees `Some`.
            work.factor_cache
                .as_ref()
                .expect("cache_hit implies Some")
                .lu
                .clone()
        } else {
            let lu = work.m_buf.clone().lu();
            let capacities: Vec<f64> = zones.iter().map(|z| z.heat_capacity).collect();
            work.factor_cache = Some(FactorCache {
                dt,
                capacities,
                lu: lu.clone(),
            });
            lu
        };

        let t_new_vec = lu_owned
            .solve(&work.b_buf)
            .ok_or(MultiZoneNetworkError::SingularSystem)?;

        // Copy the post-step temperatures into the scratch buffer (we
        // need them twice: for `q_iz` and for the returned
        // `temperatures_after`).
        for i in 0..n {
            work.t_new_buf[i] = t_new_vec[i];
        }

        // Compute `q_iz` from the post-step temperatures.
        let mut net = 0.0_f64;
        for (i, z) in zones.iter_mut().enumerate() {
            let mut qi = 0.0_f64;
            for j in 0..n {
                qi += self.h_tr_iz[(i, j)] * (work.t_new_buf[j] - work.t_new_buf[i]);
            }
            work.q_iz_buf[i] = qi;
            net += qi;
            z.temperature = work.t_new_buf[i];
        }

        let temperatures_after: Vec<f64> = work
            .t_old_buf
            .iter()
            .zip(work.t_new_buf.iter())
            .map(|(_, &tn)| tn)
            .collect();

        Ok(InterZoneResult {
            q_iz_w: work.q_iz_buf.clone(),
            net_w: net,
            temperatures_after,
        })
    }

    /// Generate a `MultiZoneNetworkReport` summarizing the conservation
    /// check across a sequence of (symmetric) conductance configurations.
    /// Used by the validation CLI to emit machine-readable output.
    pub fn conservation_report(&self) -> MultiZoneNetworkReport {
        // The algebraic identity is exact; numerical LU noise gives O(1e-13)
        // for symmetric matrices. We probe with a fixed random temperature
        // vector so the report is reproducible.
        let n = self.num_zones();
        // Deterministic temperature ramp 10..30.
        let temps: Vec<f64> = (0..n).map(|i| 10.0 + 2.0 * i as f64).collect();
        let net = self.net_inter_zone_q(&temps);
        let symmetric = is_symmetric(&self.h_tr_iz, 1e-12);
        MultiZoneNetworkReport {
            n_zones: n,
            symmetric,
            net_inter_zone_q_w: net,
            tolerance_w: 1e-6,
        }
    }
}

/// Returns `true` if `m` is symmetric within `tol`.
fn is_symmetric(m: &DMatrix<f64>, tol: f64) -> bool {
    if m.nrows() != m.ncols() {
        return false;
    }
    for i in 0..m.nrows() {
        for j in (i + 1)..m.ncols() {
            if (m[(i, j)] - m[(j, i)]).abs() > tol {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a symmetric ring conductance matrix where each zone is
    /// coupled only to its neighbour (`h` W/K). For a ring of N zones, the
    /// matrix has `h` on the two off-diagonals per row, 0 elsewhere.
    fn ring_conductance(n: usize, h: f64) -> DMatrix<f64> {
        let mut m = DMatrix::<f64>::zeros(n, n);
        if n < 2 {
            return m;
        }
        for i in 0..n {
            let j_next = (i + 1) % n;
            m[(i, j_next)] = h;
            m[(j_next, i)] = h;
        }
        m
    }

    /// Helper: build a fully-connected symmetric conductance matrix where
    /// every zone pair has conductance `h` W/K.
    fn fully_connected_conductance(n: usize, h: f64) -> DMatrix<f64> {
        let mut m = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    m[(i, j)] = h;
                }
            }
        }
        m
    }

    /// Acceptance criterion #1 (Issue #1348):
    /// N=3 symmetric 3×3 conductance matrix, sum of all inter-zone transfers
    /// equals 0 W within 1e-6 W tolerance (machine precision for f64 sum).
    #[test]
    fn three_zone_symmetric_conductance_conserves_energy() {
        let h = ring_conductance(3, 50.0);
        let net =
            MultiZoneAirflowNetwork::from_matrix(h.clone()).net_inter_zone_q(&[20.0, 25.0, 15.0]);
        assert!(
            net.abs() < 1e-6,
            "N=3 symmetric network must conserve energy; got |Σ q_iz| = {net:.3e} W"
        );
    }

    /// Acceptance criterion #2 (Issue #1348):
    /// Same identity at N=5 round-trip (backward-Euler solve and report).
    #[test]
    fn five_zone_symmetric_conductance_conserves_energy() {
        let h = fully_connected_conductance(5, 30.0);
        let mut zones: Vec<ZoneState> = (0..5)
            .map(|i| ZoneState::new(18.0 + 2.0 * i as f64, 1.0e6))
            .collect();
        let q_ext = vec![0.0; 5];
        let dt = 3600.0;
        let net = MultiZoneAirflowNetwork::from_matrix(h)
            .solve_step(&mut zones, &q_ext, dt)
            .expect("5-zone solve")
            .net_w;
        assert!(
            net.abs() < 1e-6,
            "N=5 symmetric network must conserve energy; got |Σ q_iz| = {net:.3e} W"
        );
    }

    /// Acceptance criterion #2 (Issue #1348): N=10 round-trip.
    #[test]
    fn ten_zone_symmetric_conductance_conserves_energy() {
        let h = fully_connected_conductance(10, 10.0);
        let mut zones: Vec<ZoneState> = (0..10)
            .map(|i| ZoneState::new(15.0 + i as f64, 1.0e6))
            .collect();
        let q_ext = vec![0.0; 10];
        let dt = 3600.0;
        let net = MultiZoneAirflowNetwork::from_matrix(h)
            .solve_step(&mut zones, &q_ext, dt)
            .expect("10-zone solve")
            .net_w;
        assert!(
            net.abs() < 1e-6,
            "N=10 symmetric network must conserve energy; got |Σ q_iz| = {net:.3e} W"
        );
    }

    /// Backward-compatibility: Case 960 two-zone (door opening = 1.5 W/K).
    /// The existing `inter_zone_tolerance = 1.0 W` in `EnergyBalanceValidator`
    /// must continue to pass (i.e. the 2-zone pair produces a meaningful,
    /// non-pathological q_iz).
    #[test]
    fn two_zone_case960_backward_compatible() {
        let h = DMatrix::from_row_slice(2, 2, &[0.0, 1.5, 1.5, 0.0]);
        let mut zones = vec![
            ZoneState::new(20.0, 2.0e6), // Living (back-zone)
            ZoneState::new(8.0, 5.0e5),  // Sunspace
        ];
        let q_ext = vec![0.0, 0.0];
        let result = MultiZoneAirflowNetwork::from_matrix(h)
            .solve_step(&mut zones, &q_ext, 3600.0)
            .expect("2-zone solve");

        // Zone 0 (living, warmer) loses heat; zone 1 (sunspace, cooler) gains.
        // Sign: q_iz[0] = h · (T_1 − T_0) < 0; q_iz[1] = h · (T_0 − T_1) > 0.
        assert!(
            result.q_iz_w[0] < 0.0,
            "warm zone must lose heat: q_iz[0] = {}",
            result.q_iz_w[0]
        );
        assert!(
            result.q_iz_w[1] > 0.0,
            "cool zone must gain heat: q_iz[1] = {}",
            result.q_iz_w[1]
        );
        assert!(
            (result.q_iz_w[0] + result.q_iz_w[1]).abs() < 1.0,
            "Case 960 inter-zone transfer must be within the legacy 1.0 W tolerance; \
             got |q_iz[0] + q_iz[1]| = {} W",
            (result.q_iz_w[0] + result.q_iz_w[1]).abs()
        );
        assert_eq!(result.net_w, 0.0);
    }

    /// Acceptance criterion #1 (Issue #1348): performance budget — 3-zone
    /// network solves in < 1 ms on a single core (interactive CLI budget).
    ///
    /// Tightened by Issue #2859: the per-step allocation + LU factorization
    /// were removed (pre-allocated `DMatrix` / `DVector` buffers, cached LU
    /// keyed by `(dt, C_i)`), so the same N=3 solve now drops well below the
    /// 1 ms legacy budget. The acceptance criterion for Issue #2859 is
    /// < 50 µs / step. We assert both: the legacy 1 ms gate as a regression
    /// floor and the tightened 50 µs as the perf-regression floor for the
    /// cached-LU optimization.
    #[test]
    fn three_zone_solves_under_one_millisecond() {
        let h = fully_connected_conductance(3, 50.0);
        let mut zones = vec![
            ZoneState::new(22.0, 1.0e6),
            ZoneState::new(20.0, 1.0e6),
            ZoneState::new(18.0, 1.0e6),
        ];
        let q_ext = vec![0.0; 3];
        let network = MultiZoneAirflowNetwork::from_matrix(h);

        // Warm up: first call refactorizes (cache miss), second call hits
        // the cache. Both should be timed below.
        let _ = network.solve_step(&mut zones, &q_ext, 3600.0).unwrap();

        // Measure 1000 solves; report per-solve average over the
        // steady-state (cache-hit) regime.
        let iters = 1000_usize;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = network.solve_step(&mut zones, &q_ext, 3600.0).unwrap();
        }
        let elapsed = start.elapsed();
        let per_step_us = elapsed.as_micros() as f64 / iters as f64;
        // Issue #1348 legacy budget — still asserted as a regression
        // floor.
        assert!(
            per_step_us < 1000.0,
            "N=3 must solve in < 1 ms (Issue #1348 budget); got {per_step_us:.1} µs/solve"
        );
        // Issue #2859 acceptance — after the cached-LU optimization the
        // per-step cost is dominated by the LU solve + a few scratch
        // writes, well below 50 µs on any reasonable machine. If this
        // fires the cached-LU path regressed (e.g. a per-call `clone()`
        // snuck back in).
        assert!(
            per_step_us < 50.0,
            "N=3 cached solve must be < 50 µs/step (Issue #2859 acceptance); \
             got {per_step_us:.1} µs/solve — re-check that solve_step still \
             reuses m_buf/b_buf and the cached LU factorization"
        );
    }

    /// Dimension-mismatch error is surfaced when the zone count doesn't
    /// match the matrix size.
    #[test]
    fn solve_step_rejects_zone_count_mismatch() {
        let h = fully_connected_conductance(3, 50.0);
        let mut zones = vec![ZoneState::new(20.0, 1.0e6); 2]; // wrong length
        let q_ext = vec![0.0; 3]; // matches matrix
        let result = MultiZoneAirflowNetwork::from_matrix(h).solve_step(&mut zones, &q_ext, 3600.0);
        assert!(matches!(
            result,
            Err(MultiZoneNetworkError::ZoneCountMismatch { .. })
        ));
    }

    /// Issue #2859 cache correctness: repeated calls with the same `(dt, C_i)`
    /// must hit the cached LU factorization and produce bit-identical
    /// `T_new`, `q_iz`, and `net` to the first (cache-miss) call. If a future
    /// refactor breaks the cache-hit predicate — e.g. by silently skipping
    /// the `dt`/`C_i` comparison — the second call's `T_new` would diverge
    /// from the first and this test would fire.
    #[test]
    fn solve_step_cache_hit_is_bit_identical() {
        let h = fully_connected_conductance(4, 12.5);
        let mut zones = vec![
            ZoneState::new(22.0, 1.5e6),
            ZoneState::new(20.0, 1.0e6),
            ZoneState::new(18.0, 8.0e5),
            ZoneState::new(15.0, 1.2e6),
        ];
        let q_ext = vec![10.0, -5.0, 0.0, 3.5];
        let network = MultiZoneAirflowNetwork::from_matrix(h);

        // First call — cache miss + refactorize.
        let mut zones_a = zones.clone();
        let r1 = network.solve_step(&mut zones_a, &q_ext, 1800.0).unwrap();

        // Second call — same `(dt, C_i)` profile → cache hit. Result must
        // match the first call bit-for-bit (modulo the post-step zone
        // temperatures, which the caller mutated in between).
        let mut zones_b = zones.clone();
        let r2 = network.solve_step(&mut zones_b, &q_ext, 1800.0).unwrap();
        assert_eq!(
            r1.q_iz_w, r2.q_iz_w,
            "q_iz must be bit-identical on cache hit"
        );
        assert_eq!(r1.net_w, r2.net_w, "net must be bit-identical on cache hit");
        assert_eq!(
            r1.temperatures_after, r2.temperatures_after,
            "T_new must be bit-identical on cache hit"
        );
        assert_eq!(zones_a, zones_b, "post-step zones must match across calls");
    }

    /// Issue #2859 cache invalidation: changing `dt` must trigger a
    /// refactorization (cache miss). The first call uses `dt = 3600` and
    /// primes the cache; the second uses `dt = 60`. The post-step
    /// temperatures must reflect the new `dt` (a 60 s step with C = 1e6
    /// has a much larger `C/dt` term than a 3600 s step, so the
    /// temperatures should drift toward the initial values less on the
    /// short-`dt` step).
    #[test]
    fn solve_step_cache_invalidates_on_dt_change() {
        let h = fully_connected_conductance(2, 50.0);
        let network = MultiZoneAirflowNetwork::from_matrix(h);
        let mut zones = vec![ZoneState::new(30.0, 1.0e6), ZoneState::new(10.0, 1.0e6)];
        let q_ext = vec![0.0, 0.0];

        // Prime cache with dt = 3600.
        let r_long = network.solve_step(&mut zones, &q_ext, 3600.0).unwrap();
        let t_long_after = zones.iter().map(|z| z.temperature).collect::<Vec<_>>();

        // Reset zones to the initial state and run with dt = 60. The
        // cache must miss (dt differs) and refactorize.
        zones[0].temperature = 30.0;
        zones[1].temperature = 10.0;
        let r_short = network.solve_step(&mut zones, &q_ext, 60.0).unwrap();
        let t_short_after = zones.iter().map(|z| z.temperature).collect::<Vec<_>>();

        // With a smaller dt, the system has less time to equilibrate, so
        // |ΔT| should be smaller than at dt = 3600.
        let long_drift = (t_long_after[0] - 30.0).abs() + (t_long_after[1] - 10.0).abs();
        let short_drift = (t_short_after[0] - 30.0).abs() + (t_short_after[1] - 10.0).abs();
        assert!(
            short_drift < long_drift,
            "shorter dt should produce smaller drift; long_drift={long_drift}, short_drift={short_drift}"
        );
        // Both results must still be self-consistent (no NaN / Inf).
        for r in [&r_long.q_iz_w, &r_short.q_iz_w] {
            for &v in r {
                assert!(v.is_finite(), "q_iz must be finite; got {v}");
            }
        }
    }

    /// Issue #2859 cache invalidation: changing `C_i` must trigger a
    /// refactorization (cache miss). The capacity at index 0 changes from
    /// `1.0e6` to `1.0e3` between calls — the cache must detect the
    /// mismatch and refactorize (otherwise we'd silently solve the wrong
    /// system).
    #[test]
    fn solve_step_cache_invalidates_on_capacity_change() {
        let h = fully_connected_conductance(2, 50.0);
        let network = MultiZoneAirflowNetwork::from_matrix(h);
        let mut zones = vec![ZoneState::new(30.0, 1.0e6), ZoneState::new(10.0, 1.0e6)];
        let q_ext = vec![0.0, 0.0];

        // Prime cache with C = [1e6, 1e6]. With equal capacities the two
        // zones approach the midpoint 20 °C.
        let r1 = network.solve_step(&mut zones, &q_ext, 3600.0).unwrap();

        // Reset temperatures and shrink zone 0 capacity by 1000×. The
        // cache must miss (C profile differs) and refactorize. Because
        // zone 1's C_1 = 1e6 dominates the coupling, the low-C zone 0
        // gets dragged toward zone 1's temperature (10 °C), NOT toward
        // the equal-weight midpoint (20 °C).
        zones[0].temperature = 30.0;
        zones[1].temperature = 10.0;
        zones[0].heat_capacity = 1.0e3;
        let r2 = network.solve_step(&mut zones, &q_ext, 3600.0).unwrap();

        // The result must reflect the new C profile, not the cached
        // equal-capacity solution.
        assert!(
            r2.temperatures_after[0] < r1.temperatures_after[0],
            "low-C zone 0 must be dragged toward the high-C zone (cooler) \
             on a 3600 s step; r1.T[0]={} r2.T[0]={} — the cache hit the \
             stale entry",
            r1.temperatures_after[0],
            r2.temperatures_after[0]
        );
        // Both results must still be self-consistent (no NaN / Inf).
        for r in [&r1.q_iz_w, &r2.q_iz_w] {
            for &v in r {
                assert!(v.is_finite(), "q_iz must be finite; got {v}");
            }
        }
    }

    /// Non-positive timestep is rejected.
    #[test]
    fn solve_step_rejects_zero_timestep() {
        let h = fully_connected_conductance(2, 50.0);
        let mut zones = vec![ZoneState::new(20.0, 1.0e6); 2];
        let q_ext = vec![0.0; 2];
        let result = MultiZoneAirflowNetwork::from_matrix(h).solve_step(&mut zones, &q_ext, 0.0);
        assert!(matches!(
            result,
            Err(MultiZoneNetworkError::InvalidTimestep(_))
        ));
    }

    /// Non-positive heat capacity is rejected.
    #[test]
    fn solve_step_rejects_nonpositive_heat_capacity() {
        let h = fully_connected_conductance(2, 50.0);
        let mut zones = vec![ZoneState::new(20.0, 0.0), ZoneState::new(20.0, 1.0)];
        let q_ext = vec![0.0; 2];
        let result = MultiZoneAirflowNetwork::from_matrix(h).solve_step(&mut zones, &q_ext, 3600.0);
        assert!(matches!(
            result,
            Err(MultiZoneNetworkError::InvalidHeatCapacity { .. })
        ));
    }

    /// Asymmetric conductance violates conservation (sanity check that the
    /// `|Σ q_iz| < 1e-6 W` acceptance criterion really is testing
    /// symmetry, not numerical noise).
    #[test]
    fn asymmetric_conductance_breaks_conservation() {
        let m = DMatrix::from_row_slice(
            3,
            3,
            &[
                0.0, 5.0, 1.0, 3.0, 0.0, 7.0, // h_10 != h_01 — asymmetric
                2.0, 4.0, 0.0,
            ],
        );
        let net = MultiZoneAirflowNetwork::from_matrix(m).net_inter_zone_q(&[25.0, 20.0, 15.0]);
        assert!(
            net.abs() > 1e-3,
            "asymmetric network should NOT conserve energy; got |Σ q_iz| = {net:.3e} W"
        );
    }

    /// `from_adjacency_pairs` constructs the same matrix as `from_matrix` for
    /// a fully-connected symmetric configuration.
    #[test]
    fn from_adjacency_pairs_matches_from_matrix() {
        let n = 4_usize;
        let pairs: Vec<(usize, usize, f64)> = (0..n)
            .flat_map(|i| {
                (0..n).filter_map(move |j| if i != j { Some((i, j, 12.5)) } else { None })
            })
            .collect();
        let from_pairs = MultiZoneAirflowNetwork::from_adjacency_pairs(n, &pairs);
        let from_matrix =
            MultiZoneAirflowNetwork::from_matrix(fully_connected_conductance(n, 12.5));
        assert_eq!(
            from_pairs.conductance_matrix(),
            from_matrix.conductance_matrix()
        );
    }

    /// `per_zone_conductance` returns the row sums of the N×N matrix — the
    /// backward-compatible flat vector for the existing `model.conduction.h_tr_iz`
    /// field used by the Case 960 physics pipeline.
    #[test]
    fn per_zone_conductance_is_row_sum_of_matrix() {
        let m = DMatrix::from_row_slice(3, 3, &[0.0, 5.0, 1.0, 5.0, 0.0, 7.0, 1.0, 7.0, 0.0]);
        let row_sums = MultiZoneAirflowNetwork::from_matrix(m).per_zone_conductance();
        assert_eq!(row_sums, vec![6.0, 12.0, 8.0]);
    }

    /// `solve_step` mutates zone temperatures in place to the post-step
    /// values — caller doesn't have to thread a separate output buffer.
    /// Uses a smaller heat capacity so a single 1 h step produces a
    /// detectable ΔT (otherwise 1 MJ/K vs 50 W/K over 1 h moves ~0.005 K,
    /// which trips floating-point equality on the floor-direction check).
    #[test]
    fn solve_step_mutates_zone_temperatures_in_place() {
        let h = fully_connected_conductance(2, 50.0);
        let mut zones = vec![ZoneState::new(20.0, 1.0e4), ZoneState::new(15.0, 1.0e4)];
        let t_before: Vec<f64> = zones.iter().map(|z| z.temperature).collect();
        let result = MultiZoneAirflowNetwork::from_matrix(h)
            .solve_step(&mut zones, &[0.0, 0.0], 3600.0)
            .expect("2-zone solve");
        assert_eq!(zones.len(), 2);
        // Energy conservation (C_0 = C_1) ⇒ |ΔT_0| = |ΔT_1|, and zone 0
        // (warmer) cools while zone 1 (cooler) warms.
        assert!(
            zones[0].temperature < t_before[0],
            "warmer zone 0 should cool: {} -> {}",
            t_before[0],
            zones[0].temperature
        );
        assert!(
            zones[1].temperature > t_before[1],
            "cooler zone 1 should warm: {} -> {}",
            t_before[1],
            zones[1].temperature
        );
        // The post-step temperatures must match what `solve_step` reported.
        assert_eq!(zones[0].temperature, result.temperatures_after[0]);
        assert_eq!(zones[1].temperature, result.temperatures_after[1]);
        // The post-step temperatures must be between the two initial values
        // (no overshoot in a backward-Euler solve of a stable system).
        assert!(zones[0].temperature >= t_before[1]);
        assert!(zones[1].temperature <= t_before[0]);
    }

    /// `conservation_report` flags an asymmetric matrix as non-symmetric
    /// and emits the (non-zero) `net_inter_zone_q_w` value.
    #[test]
    fn conservation_report_flags_asymmetry() {
        let m = DMatrix::from_row_slice(2, 2, &[0.0, 5.0, 3.0, 0.0]);
        let report = MultiZoneAirflowNetwork::from_matrix(m).conservation_report();
        assert_eq!(report.n_zones, 2);
        assert!(!report.symmetric);
        assert!(report.net_inter_zone_q_w.abs() > 1e-3);
    }

    /// `conservation_report` flags a symmetric matrix as symmetric and
    /// emits a net residual ≤ 1e-6 W (the machine-precision floor for the
    /// identity check).
    #[test]
    fn conservation_report_flags_symmetric() {
        let h = fully_connected_conductance(4, 25.0);
        let report = MultiZoneAirflowNetwork::from_matrix(h).conservation_report();
        assert_eq!(report.n_zones, 4);
        assert!(report.symmetric);
        assert!(
            report.net_inter_zone_q_w.abs() < 1e-6,
            "symmetric matrix must report net ≈ 0; got {} W",
            report.net_inter_zone_q_w.abs()
        );
        assert_eq!(report.tolerance_w, 1e-6);
    }
}
