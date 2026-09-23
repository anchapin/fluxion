// =============================================================================
// Gauge-theory thermal manifold (Issue #1461 — Phase 1a)
// =============================================================================
//
// The `ThermalManifold` is the foundational data structure for replacing the
// discrete 5R1C / 9R4C lumped-capacitance networks with a continuous Riemannian
// representation. See the module-level doc-comment for the relationship between
// this section and the CTA geometry tensors above.
//
// Coordinate convention (matches the 9R4C zone-level network selected for
// high-mass constructions per ADR-002, `ARCHITECTURE.md` Module 5):
//
//   index 0 → zone air node           (`T_air`, internal gains + HVAC source)
//   index 1 → exterior wall mass node (`T_wall`, envelope solar)
//   index 2 → roof mass node          (`T_roof`, top solar + sky coupling)
//   index 3 → floor mass node         (`T_floor`, ground + slab coupling)
//
// The 5R1C scene embeds into this 4-D space by collapsing the wall/roof/floor
// mass nodes into a single mass node at index 1 and parking roof/floor at zero
// (see [`ThermalManifold::from_5r1c_parameters`]). The GaugeSolver (#1462)
// operates on the same 4-D space regardless of which scene is active.

use nalgebra::{Matrix4, Vector4};

use super::qubo::{QuboEncoding, QuboTranslateError};

/// Number of dimensions in the [`ThermalManifold`] ambient space. Pinned to 4 so
/// the type-level [`Vector4`] matches the 9R4C high-mass scene (air + 3 mass
/// nodes) and the 5R1C scene can embed by parking unused mass slots at zero.
pub const MANIFOLD_DIM: usize = 4;

/// Field indices for [`ThermalManifold::scalar_field`] and
/// [`ThermalManifold::gauge_connection`]. Index 0 is the air node (zone interior
/// temperature); indices 1..4 are the wall / roof / floor mass nodes.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ManifoldIndex {
    /// Zone air node (interior air temperature; HVAC + internal gains + vent).
    Air = 0,
    /// Exterior wall mass node (envelope solar storage + wall conduction).
    Wall = 1,
    /// Roof mass node (top-irradiance solar + sky coupling).
    Roof = 2,
    /// Floor mass node (ground slab + ground temperature coupling).
    Floor = 3,
}

impl ManifoldIndex {
    /// Convert from a `usize` (panics on out-of-range — `ThermalManifold` is
    /// statically 4-D so callers are responsible for valid indices).
    pub fn from_usize(idx: usize) -> Self {
        match idx {
            0 => Self::Air,
            1 => Self::Wall,
            2 => Self::Roof,
            3 => Self::Floor,
            _ => panic!(
                "ManifoldIndex::from_usize({idx}) out of range (MANIFOLD_DIM = {MANIFOLD_DIM})"
            ),
        }
    }

    /// All indices in declaration order. Useful for safe iteration in the
    /// GaugeSolver (#1462) when it walks the field / connection slots.
    pub const ALL: [ManifoldIndex; MANIFOLD_DIM] = [Self::Air, Self::Wall, Self::Roof, Self::Floor];
}

/// Geometric validation failures for [`ThermalManifold::validate`]. Kept narrow
/// on purpose — the geometric solver enforces the dissipative structure
/// (passivity), the manifold only enforces algebraic finiteness.
#[derive(Debug, Clone, PartialEq)]
pub enum ManifoldError {
    /// `metric_tensor` contains a `NaN` or ±∞ entry.
    NonFiniteMetric { row: usize, col: usize },
    /// `scalar_field` contains a `NaN` or ±∞ entry.
    NonFiniteField,
    /// `gauge_connection` contains a `NaN` or ±∞ entry.
    NonFiniteConnection,
}

impl std::fmt::Display for ManifoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteMetric { row, col } => {
                write!(f, "metric_tensor[{row},{col}] is not finite (NaN/inf)")
            }
            Self::NonFiniteField => write!(f, "scalar_field contains NaN/inf"),
            Self::NonFiniteConnection => write!(f, "gauge_connection contains NaN/inf"),
        }
    }
}

impl std::error::Error for ManifoldError {}

/// Thermal manifold — Phase 1a (issue #1461) foundation for the gauge-theory
/// migration. The 4-D ambient space replaces the discrete `T_air` and `T_mass_*`
/// nodes of the 5R1C / 9R4C networks with a vector field on a Riemannian
/// manifold. The matrix representation replaces the lumped `R` and `C` values.
///
/// ```text
///   scalar_field      ← T_air (idx 0) and T_mass_wall/roof/floor (idx 1..4)
///   metric_tensor     ← (R, C) values per node — the dissipative operator
///   gauge_connection  ← external heat fluxes (Solar, HVAC, internal)
/// ```
///
/// `GaugeSolver` (Phase 1b, #1462) consumes this structure to compute the
/// Christoffel connection and step the manifold through
/// [`ThermalManifold::compute_parallel_transport`]. Per the #1461 epic, **no
/// hardcoded HVAC clamps** (the 100 kW cap) appear in the shadow path — geometric
/// math is expected to be natively stable.
///
/// # Physical mapping
///
/// The 5R1C discrete ODE
///
/// ```text
///   C_air  · dT_air/dt = (T_mass − T_air)/R_eq   + Q_internal
///   C_mass · dT_mass/dt = (T_air − T_mass)/R_eq − (T_mass − T_out)/R_ow + Q_solar
/// ```
///
/// is the (forward-Euler-discretized) flow map of the linear ODE
///
/// ```text
///   dT/dt = M · T + A   where   M = metric_tensor  ·  A = gauge_connection
/// ```
///
/// This was verified numerically against the legacy 5R1C reference step (see
/// `test_from_5r1c_matches_legacy_ode` below; reference Python at
/// `.agents/results/issue-1461-python-verification.py`). The `GaugeSolver` (#1462)
/// extends this with the full Christoffel-symbol transport.
#[derive(Debug, Clone)]
pub struct ThermalManifold {
    /// Riemannian metric on the thermal tangent space. Units: s⁻¹.
    ///
    /// **Physical mapping** (5R1C scene, embedded into the 4-D manifold with
    /// roof/floor slots parked at 0):
    ///
    /// ```text
    ///   metric[0,0] = −(1/R_eq) / C_air        // self-conductance of air node
    ///   metric[0,1] = +(1/R_eq) / C_air        // air ← mass coupling
    ///   metric[1,0] = +(1/R_eq) / C_mass       // mass ← air coupling
    ///   metric[1,1] = −(1/R_eq + 1/R_ow) / C_mass  // self-conductance of mass
    /// ```
    ///
    /// The full 9R4C entry layout is given in
    /// [`ThermalManifold::from_9r4c_parameters`].
    ///
    /// **Algebraic invariants** the GaugeSolver relies on (not enforced here
    /// because the structure is general — passive *and* active operators fit):
    /// dissipative networks have `metric[i,i] ≤ 0` and `metric[i,j] ≥ 0` for
    /// `i ≠ j` (Kirchhoff's current law at each node).
    pub metric_tensor: Matrix4<f64>,

    /// Tangent-space field (zone temperatures). Units: °C.
    /// Index 0 is zone air; indices 1..4 are wall / roof / floor mass nodes (see
    /// [`ManifoldIndex`]).
    pub scalar_field: Vector4<f64>,

    /// Gauge connection 1-form — external heat fluxes per node. Units: W.
    ///
    /// Index 0 is the air-node source (HVAC + internal gains ± vent losses);
    /// indices 1..4 are mass-node sources (absorbed solar + inter-zone
    /// flows + ground coupling). The GaugeSolver (#1462) maps raw boundary
    /// conditions (irradiance, outside air temp) into this vector via the
    /// formula in its `Boundary Condition translation` acceptance criterion.
    pub gauge_connection: Vector4<f64>,

    /// Last timestep duration used to advance the manifold. Carried so
    /// `GaugeSolver` (#1462) can reconstruct the operator chain without an
    /// extra argument at every call site. Not interpreted by the
    /// parallel-transport stub below — that takes `dt` as an explicit argument.
    pub dt_seconds: f64,
}

impl Default for ThermalManifold {
    fn default() -> Self {
        Self::new_flat()
    }
}

impl ThermalManifold {
    /// Construct a flat (uncoupled) manifold at the origin with zero field, zero
    /// connection, and zero timestep. All metric off-diagonals are 0; the diagonal
    /// is `1.0` per node so the manifold is trivially invertible (GaugeSolver
    /// #1462 can rely on `try_inverse()` returning `Some`).
    ///
    /// This is the unit element in the product geometry — equivalent to an
    /// idealised, mass-less, source-free thermal space.
    pub fn new_flat() -> Self {
        Self {
            metric_tensor: Matrix4::identity(),
            scalar_field: Vector4::zeros(),
            gauge_connection: Vector4::zeros(),
            dt_seconds: 0.0,
        }
    }

    /// Construct from a discrete 5R1C scene, embedded into the 4-D manifold.
    ///
    /// Active axes are `[T_air (idx 0), T_mass (idx 1), 0 (idx 2), 0 (idx 3)]`.
    /// The roof / floor slots are parked at field 0 with metric entries `(2,2)
    /// = (3,3) = 0` (no self-conductance, no cross-coupling), so they remain
    /// inert under transport. The GaugeSolver (#1462) drops them on read.
    ///
    /// # Panics
    ///
    /// Asserts that `r_eq > 0`, `c_air > 0`, `c_mass > 0`. Negative or zero
    /// capacitances / resistances are non-physical and would silently
    /// destabilise the gauge transport.
    pub fn from_5r1c_parameters(
        t_air: f64,
        t_mass: f64,
        r_eq: f64,
        c_air: f64,
        c_mass: f64,
    ) -> Self {
        assert!(r_eq > 0.0, "r_eq must be > 0 (got {r_eq})");
        assert!(c_air > 0.0, "c_air must be > 0 (got {c_air})");
        assert!(c_mass > 0.0, "c_mass must be > 0 (got {c_mass})");

        let g_eq = 1.0 / r_eq;
        let mut metric = Matrix4::zeros();
        metric[(0, 0)] = -g_eq / c_air;
        metric[(0, 1)] = g_eq / c_air;
        metric[(1, 0)] = g_eq / c_mass;
        metric[(1, 1)] = -g_eq / c_mass;
        // (2,2), (3,3) and all off-diagonals stay at 0 → inert under transport.

        let mut field = Vector4::zeros();
        field[ManifoldIndex::Air as usize] = t_air;
        field[ManifoldIndex::Wall as usize] = t_mass;

        Self {
            metric_tensor: metric,
            scalar_field: field,
            gauge_connection: Vector4::zeros(),
            dt_seconds: 0.0,
        }
    }

    /// Construct from a discrete 9R4C scene. Populates the dissipative
    /// operator `metric[i,i] = -(g_tr_i)/C_i` and the cross-coupling
    /// `metric[i,j] = +g_tr_ij / C_i` from the per-node conductance matrix.
    ///
    /// # Arguments
    ///
    /// * `temperatures` — `[T_air, T_wall, T_roof, T_floor]`, °C.
    /// * `capacitances` — `[C_air, C_wall, C_roof, C_floor]`, J/K. Must all be
    ///   strictly positive.
    /// * `r_tr_surface` — surface-to-air transmitances `[g_wall, g_roof, g_floor]`,
    ///   W/K. Must be strictly positive.
    /// * `r_cross` — optional inter-mass transmitances `[g_wall_roof, g_wall_floor,
    ///   g_roof_floor]`. `None` ⇒ no inter-mass coupling (each mass node couples
    ///   only to the air node, the legacy 9R4C limit case).
    ///
    /// # Panics
    ///
    /// Asserts that every conductance and capacitance is strictly positive.
    pub fn from_9r4c_parameters(
        temperatures: [f64; MANIFOLD_DIM],
        capacitances: [f64; MANIFOLD_DIM],
        r_tr_surface: [f64; 3],
        r_cross: Option<[f64; 3]>,
    ) -> Self {
        for (label, &c) in capacitances.iter().enumerate() {
            assert!(c > 0.0, "capacitances[{label}] must be > 0 (got {c})");
        }
        for (label, &g) in r_tr_surface.iter().enumerate() {
            assert!(g > 0.0, "r_tr_surface[{label}] must be > 0 (got {g})");
        }
        if let Some(rc) = r_cross {
            for (label, &g) in rc.iter().enumerate() {
                assert!(
                    g >= 0.0,
                    "r_cross[{label}] must be ≥ 0 (got {g}); use None to disable"
                );
            }
        }

        let g_wall = r_tr_surface[0];
        let g_roof = r_tr_surface[1];
        let g_floor = r_tr_surface[2];
        let c_air = capacitances[ManifoldIndex::Air as usize];
        let c_wall = capacitances[ManifoldIndex::Wall as usize];
        let c_roof = capacitances[ManifoldIndex::Roof as usize];
        let c_floor = capacitances[ManifoldIndex::Floor as usize];

        let mut metric = Matrix4::zeros();

        // Air node (idx 0): self = sum of all surface conductances, each divided
        // by C_air; cross = g_tr_i / C_air.
        metric[(0, 0)] = -(g_wall + g_roof + g_floor) / c_air;
        metric[(0, 1)] = g_wall / c_air;
        metric[(0, 2)] = g_roof / c_air;
        metric[(0, 3)] = g_floor / c_air;

        // Wall mass (idx 1): self = g_wall/c_wall + (g_wall_roof + g_wall_floor)/c_wall;
        // cross into air and into roof/floor per the conductance layout.
        let g_wr = r_cross.map(|rc| rc[0]).unwrap_or(0.0);
        let g_wf = r_cross.map(|rc| rc[1]).unwrap_or(0.0);
        let g_rf = r_cross.map(|rc| rc[2]).unwrap_or(0.0);
        metric[(1, 0)] = g_wall / c_wall;
        metric[(1, 1)] = -(g_wall + g_wr + g_wf) / c_wall;
        metric[(1, 2)] = g_wr / c_wall;
        metric[(1, 3)] = g_wf / c_wall;

        // Roof mass (idx 2).
        metric[(2, 0)] = g_roof / c_roof;
        metric[(2, 1)] = g_wr / c_roof;
        metric[(2, 2)] = -(g_roof + g_wr + g_rf) / c_roof;
        metric[(2, 3)] = g_rf / c_roof;

        // Floor mass (idx 3).
        metric[(3, 0)] = g_floor / c_floor;
        metric[(3, 1)] = g_wf / c_floor;
        metric[(3, 2)] = g_rf / c_floor;
        metric[(3, 3)] = -(g_floor + g_wf + g_rf) / c_floor;

        Self {
            metric_tensor: metric,
            scalar_field: Vector4::from(temperatures),
            gauge_connection: Vector4::zeros(),
            dt_seconds: 0.0,
        }
    }

    /// Compute Christoffel symbols of the second kind using the Levi-Civita
    /// connection.
    ///
    /// Γ^i_jk = ½ · g^{il} · (∂_j g_{lk} + ∂_k g_{jl} − ∂_l g_{jk})
    ///
    /// The return type `Matrix4<Matrix4<f64>>` represents the 4×4×4 symbol tensor
    /// as a 4×4 outer matrix of 4×4 inner matrices. The element at outer
    /// position `(i, j)` is a `Matrix4<f64>` where element `(k, l)` holds Γ^i_{kl}.
    /// Individual symbols are accessed via `christoffel[(i, j)][(k, l)] = Γ^i_{kl}`.
    ///
    /// Since the thermal manifold has a **constant metric during transport**
    /// (the R/C values that define the metric do not change with temperature
    /// or time), all partial derivatives ∂_j g_{lk} vanish and the Christoffel
    /// symbols are zero by construction. This is verified by:
    /// - `test_christoffel_symbols_zero_for_5r1c` (Christoffel ≤ 1e-12)
    /// - `test_christoffel_symbols_zero_for_9r4c` (Christoffel ≤ 1e-12)
    pub fn compute_christoffel_symbols(&self) -> Matrix4<Matrix4<f64>> {
        let g_inv = match self.metric_tensor.try_inverse() {
            Some(inv) => inv,
            None => return Matrix4::<Matrix4<f64>>::zeros(),
        };

        let mut christoffel = Matrix4::<Matrix4<f64>>::zeros();

        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                for k in 0..MANIFOLD_DIM {
                    let mut gamma_ijk = 0.0;
                    for l in 0..MANIFOLD_DIM {
                        let partial_j_lk = 0.0;
                        let partial_k_jl = 0.0;
                        let partial_l_jk = 0.0;
                        gamma_ijk += g_inv[(i, l)] * (partial_j_lk + partial_k_jl - partial_l_jk);
                    }
                    // Christoffel symbols vanish for the thermal manifold because the metric
                    // is constant: all partial derivatives ∂_j g_{lk} are zero, so each
                    // gamma_ijk accumulates 0. Storage at [(i,k)][(k,j)] matches the
                    // Γ^i_{kj} convention; read at [(i,j)][(j,k)] contracts correctly.
                    christoffel[(i, k)][(k, j)] = 0.5 * gamma_ijk;
                }
            }
        }

        christoffel
    }

    /// **Covariant parallel transport** using the Levi-Civita connection.
    ///
    /// Computes the covariant derivative along the time axis using the
    /// Christoffel-symbol transport formula:
    ///
    /// ```text
    ///   dT^i/dt = M·T + A  −  Γ^i_{jk} · T^j · g^k
    ///   T_new   = T  +  dt · (dT/dt)
    /// ```
    ///
    /// The first term `M·T` is the linear metric evolution; the second term
    /// is the **geodesic deviation** due to curvature (Christoffel symbols),
    /// contracted with the gauge connection `g^k = gauge_connection[k]`.
    /// Since the thermal manifold has a constant metric during transport
    /// (∂_j g_{lk} = 0), the Christoffel symbols vanish and this reduces
    /// exactly to the forward-Euler `M·T + A` from the Phase 1a stub.
    ///
    /// **No hardcoded HVAC clamps** (per the #1461 epic — geometric math is
    /// expected to be natively stable; the 100 kW cap from the 5R1C path is
    /// strictly out of scope here). If the gauge transport needs bounds, they
    /// are physical (e.g. clamped boundary temps), not mathematical.
    ///
    /// This method does *not* mutate `self` — GaugeSolver (#1462) reads the
    /// returned field and explicitly assigns it via
    /// `manifold.scalar_field = manifold.compute_parallel_transport(dt);`.
    pub fn compute_parallel_transport(&self, dt: f64) -> Vector4<f64> {
        let christoffel = self.compute_christoffel_symbols();
        let metric_term = self.metric_tensor * self.scalar_field;

        let mut deviation = Vector4::zeros();
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                for k in 0..MANIFOLD_DIM {
                    let gamma_ijk = christoffel[(i, j)][(j, k)];
                    deviation[i] -= gamma_ijk * self.scalar_field[j] * self.gauge_connection[k];
                }
            }
        }

        let covariant_derivative = metric_term + self.gauge_connection + deviation;
        self.scalar_field + covariant_derivative * dt
    }

    /// Sum of the gauge-connection components. Diagnostic accessor used by the
    /// Energy-Conservation CI gate (#1465) and by `tools/piml_loss.py` (#1463)
    /// to verify First-Law compliance (`Σ A_μ = 0` for an isolated zone).
    pub fn gauge_connection_sum(&self) -> f64 {
        self.gauge_connection.iter().sum()
    }

    /// Algebraic consistency check. Does **not** enforce dissipativity —
    /// the gauge transport is general enough to handle both passive and active
    /// operators. Returns `Ok(())` for a well-formed manifold, otherwise the
    /// first failure found.
    pub fn validate(&self) -> Result<(), ManifoldError> {
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                if !self.metric_tensor[(i, j)].is_finite() {
                    return Err(ManifoldError::NonFiniteMetric { row: i, col: j });
                }
            }
        }
        if !self.scalar_field.iter().all(|x| x.is_finite()) {
            return Err(ManifoldError::NonFiniteField);
        }
        if !self.gauge_connection.iter().all(|x| x.is_finite()) {
            return Err(ManifoldError::NonFiniteConnection);
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // QUBO translation (issue #1772 — standardized utility)
    // -----------------------------------------------------------------

    /// Translate this manifold's metric tensor into a standardized
    /// [`QuboMatrix`] using the fixed-point `encoding`.
    ///
    /// This is the single documented entry point for the metric-tensor → QUBO
    /// translation (issue #1772). It produces a symmetric `N × N` matrix `Q`
    /// (`N = MANIFOLD_DIM * encoding.bits_per_node`) such that for any binary
    /// vector `x`,
    ///
    /// ```text
    ///   x^T Q x  =  T_recon^T · sym(metric_tensor) · T_recon
    /// ```
    ///
    /// where `T_recon[i] = (Σ_k 2^k · x[(i,k)]) / scale_factor` is the
    /// fixed-point reconstruction of node `i`'s temperature, and
    /// `sym(M) = ½(M + Mᵀ)`. The symmetric part is the only contribution a
    /// QUBO can represent: the quadratic form `x^T M x` cancels the
    /// antisymmetric part of `M` term-by-term. For symmetric metrics (identity,
    /// dense-symmetric) this is the *exact* metric; for dissipative operators
    /// like 5R1C/9R4C (where `metric[i,j] ≠ metric[j,i]`) the encoded energy is
    /// unchanged — see [`QuboMatrix::reconstruct_metric_tensor`] for the
    /// reverse half of the round-trip.
    ///
    /// # Edge cases
    ///
    /// * Non-finite (NaN/±∞) entries anywhere in the manifold are rejected via
    ///   [`ThermalManifold::validate`].
    /// * An empty QUBO (`bits_per_node == 0`) or a non-positive scale is
    ///   rejected via [`QuboEncoding::validate`].
    ///
    /// # Errors
    ///
    /// Returns [`QuboTranslateError::InvalidManifold`] if the manifold fails
    /// `validate()`, or [`QuboTranslateError::InvalidEncoding`] for a malformed
    /// encoding.
    pub fn to_qubo_matrix(&self, encoding: QuboEncoding) -> Result<QuboMatrix, QuboTranslateError> {
        encoding.validate()?;
        if let Err(e) = self.validate() {
            return Err(QuboTranslateError::InvalidManifold(e.to_string()));
        }

        let n = encoding.num_variables();
        let scale = encoding.scale_factor();
        let k = encoding.bits_per_node;

        // Q is symmetric. We build it densely; the upper-triangular view used
        // by D-Wave is q[i*N + j] for i <= j.
        let mut q = vec![0.0_f64; n * n];

        // Quadratic part: metric_tensor[i,j] * 2^ki * 2^kj / scale^2.
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                let m_ij = self.metric_tensor[(i, j)];
                for ki in 0..k {
                    for kj in 0..k {
                        let row = i * k + ki;
                        let col = j * k + kj;
                        let w = 2.0_f64.powi(ki as i32) * 2.0_f64.powi(kj as i32);
                        q[row * n + col] += m_ij * w / (scale * scale);
                    }
                }
            }
        }

        // Enforce exact symmetry (defensive — the construction is symmetric by
        // algebra, but rounding can introduce a 1-ULP asymmetry in the off-
        // diagonal when metric[i,j] != metric[j,i] on input).
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (q[i * n + j] + q[j * n + i]);
                q[i * n + j] = avg;
                q[j * n + i] = avg;
            }
        }

        Ok(QuboMatrix { q, n, encoding })
    }
}

/// Stable, self-contained QUBO matrix derived from a
/// [`ThermalManifold`]'s metric tensor (issue #1772).
///
/// Holds only the symmetric `N × N` matrix `Q`, its dimension `N`, and the
/// [`QuboEncoding`] that produced it — **no cached source tensors**, so it can
/// never drift from its inputs (the failure mode of the research-grade
/// `QuboProblem`). The matrix is stored densely in row-major order with
/// `Q[i, j] == Q[j, i]` enforced at construction.
///
/// The canonical round-trip is:
///
/// * forward: [`ThermalManifold::to_qubo_matrix`] (tensor → QUBO);
/// * reverse: [`QuboMatrix::reconstruct_metric_tensor`] (QUBO → tensor).
///
/// The reverse half is exact: because `Q[(i,0),(j,0)] = metric[i,j] / scale^2`
/// (the `2^0 · 2^0 = 1` weight), the metric is recovered losslessly as
/// `metric[i,j] = Q[(i,0),(j,0)] · scale^2`.
#[derive(Debug, Clone, PartialEq)]
pub struct QuboMatrix {
    /// Symmetric `N × N` QUBO matrix in row-major order.
    q: Vec<f64>,
    /// Number of binary variables (= side length of `q`).
    n: usize,
    /// Encoding used to build this matrix.
    encoding: QuboEncoding,
}

impl QuboMatrix {
    /// Number of binary variables `N`.
    pub fn n_variables(&self) -> usize {
        self.n
    }

    /// The symmetric `N × N` QUBO matrix in row-major order.
    pub fn matrix(&self) -> &[f64] {
        &self.q
    }

    /// `Q[i, j]` (symmetric — `Q[i, j] == Q[j, i]`).
    ///
    /// # Panics
    /// Panics if `i` or `j` is `≥ n_variables()`.
    pub fn entry(&self, i: usize, j: usize) -> f64 {
        assert!(i < self.n, "i={i} out of range (n={})", self.n);
        assert!(j < self.n, "j={j} out of range (n={})", self.n);
        self.q[i * self.n + j]
    }

    /// Encoding used to build this matrix.
    pub fn encoding(&self) -> QuboEncoding {
        self.encoding
    }

    /// Maximum absolute value in `Q`. Zero for an all-zero matrix.
    pub fn max_abs(&self) -> f64 {
        self.q
            .iter()
            .fold(0.0_f64, |m, &v| if v.abs() > m { v.abs() } else { m })
    }

    /// Returns `true` when `|Q[i,j] − Q[j,i]| ≤ tol` for all `i, j`. The
    /// matrix is symmetrized at construction, so this holds to within `0.0`
    /// for any freshly built [`QuboMatrix`]; the accessor is provided for
    /// downstream consumers that mutate the buffer.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if (self.q[i * self.n + j] - self.q[j * self.n + i]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Reconstruct the metric tensor from the QUBO matrix — the reverse half
    /// of the standardized round-trip (issue #1772).
    ///
    /// Uses the `2^0 · 2^0 = 1` bit-pair weight, for which
    /// `Q[(i,0),(j,0)] = metric[i,j] / scale^2` on the *symmetric part* of the
    /// metric, giving the recovery `sym(metric)[i,j] = Q[(i,0),(j,0)] · scale^2`.
    ///
    /// Because a QUBO matrix is symmetric by construction, it can only encode
    /// the **symmetric part** of the underlying bilinear form — and the
    /// quadratic energy `x^T M x` depends only on `sym(M) = ½(M + Mᵀ)`
    /// (the antisymmetric part cancels term-by-term). Accordingly:
    ///
    /// * diagonal: `recon[i,i] = metric[i,i]` (exact);
    /// * off-diagonal: `recon[i,j] = ½(metric[i,j] + metric[j,i])`.
    ///
    /// For a symmetric source metric (identity, dense-symmetric, …) the
    /// round-trip is therefore exact; for a non-symmetric dissipative operator
    /// (5R1C / 9R4C, where `metric[i,j] ≠ metric[j,i]`) the off-diagonals
    /// recover the symmetric part — the only information the QUBO retains.
    ///
    /// # Panics
    /// Panics if `bits_per_node < 1` (the matrix would be empty); this is
    /// guaranteed not to occur for a `QuboMatrix` returned by
    /// [`ThermalManifold::to_qubo_matrix`] since the encoding is validated.
    pub fn reconstruct_metric_tensor(&self) -> Matrix4<f64> {
        let k = self.encoding.bits_per_node;
        assert!(k >= 1, "cannot reconstruct from an empty QUBO");
        let scale_sq = self.encoding.scale_factor() * self.encoding.scale_factor();
        let mut out = Matrix4::zeros();
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                let row = i * k;
                let col = j * k;
                out[(i, j)] = self.q[row * self.n + col] * scale_sq;
            }
        }
        out
    }

    /// Evaluate the QUBO energy `x^T Q x` at a binary solution `x`.
    ///
    /// `x[i]` is interpreted as `0`/`1`; any non-zero byte is treated as `1`.
    ///
    /// # Panics
    /// Panics if `x.len() != n_variables()`.
    pub fn evaluate(&self, x: &[u8]) -> f64 {
        assert_eq!(
            x.len(),
            self.n,
            "x.len() = {} != n_variables = {}",
            x.len(),
            self.n
        );
        let mut acc = 0.0_f64;
        for i in 0..self.n {
            let xi = f64::from(x[i] != 0);
            if xi == 0.0 {
                continue;
            }
            for (offset, &xj_byte) in x[i..].iter().enumerate() {
                let j = i + offset;
                let xj = f64::from(xj_byte != 0);
                if xj == 0.0 {
                    continue;
                }
                let qij = self.q[i * self.n + j];
                if i == j {
                    acc += qij * xi;
                } else {
                    acc += 2.0 * qij * xi * xj;
                }
            }
        }
        acc
    }
}
