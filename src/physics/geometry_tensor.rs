//! Geometry Tensor Module
//!
//! Two parallel concerns share this file:
//!
//! 1. **CTA geometry tensors** ([`GeometryTensor`], [`WallData`]) — flat,
//!    copy-through tensors on the Python↔Rust boundary that carry zone coordinates,
//!    wall geometry, and inter-zone adjacency for the PDF/CAD ingestion pipeline
//!    (issues #448 / #453). These are intentionally kept as `Vec<f64>` so the
//!    Python bindings expose them with zero-copy semantics.
//!
//! 2. **Gauge-theory thermal manifolds** ([`ThermalManifold`], [`ManifoldIndex`]) —
//!    the **foundational data structure** for the gauge-theory migration
//!    (issue #1461 — Phase 1a). Replaces the discrete 5R1C / 9R4C lumped-capacitance
//!    networks with a continuous Riemannian representation:
//!    `metric_tensor` + `scalar_field` + `gauge_connection` on a fixed 4-D ambient
//!    space. `GaugeSolver` (Phase 1b, #1462) consumes this structure to compute the
//!    Christoffel connection and step the manifold through parallel transport.
//!
//! The two domains are deliberately kept on different storage representations
//! (`Vec<f64>` for the CTA tensors, [`nalgebra::Matrix4`] / [`nalgebra::Vector4`]
//! for the manifold) because their consumers diverge: the Python bridge needs
//! contiguous slices, while the geometric solver needs typed linear-algebra
//! operators with `try_inverse`.

use crate::physics::cta::VectorField;

/// Maximum number of thermal zones supported
pub const MAX_ZONES: usize = 100;
/// Maximum number of walls supported
pub const MAX_WALLS: usize = 500;

/// Zone coordinates tensor shape: (MAX_ZONES, 20)
/// Format: [x1, y1, x2, y2, ..., x8, y8, floor_height, ceiling_height, area, volume, perimeter, zone_id]
pub const ZONE_COORDS_DIMS: (usize, usize) = (MAX_ZONES, 20);

/// Wall matrix shape: (MAX_WALLS, 6)
/// Format: [x1, y1, x2, y2, height, thickness]
pub const WALL_MATRIX_DIMS: (usize, usize) = (MAX_WALLS, 6);

/// Window matrix shape: (MAX_WALLS, 6)
/// Format: [x1, y1, x2, y2, height, sill_height]
pub const WINDOW_MATRIX_DIMS: (usize, usize) = (MAX_WALLS, 6);

/// Adjacency matrix shape: (MAX_ZONES, MAX_ZONES)
pub const ADJACENCY_MATRIX_DIMS: (usize, usize) = (MAX_ZONES, MAX_ZONES);

/// Zone properties shape: (MAX_ZONES, 5)
/// Format: [floor_area, volume, perimeter, num_windows, num_doors]
pub const ZONE_PROPERTIES_DIMS: (usize, usize) = (MAX_ZONES, 5);

/// GeometryTensor - A container for CTA geometry tensors.
///
/// This struct holds all the geometry information extracted from PDF/CAD files
/// in a format ready for use in the Fluxion thermal simulation.
#[derive(Debug, Clone)]
pub struct GeometryTensor {
    /// Zone coordinates: (MAX_ZONES, 20)
    pub zone_coords: Vec<f64>,
    /// Wall matrix: (MAX_WALLS, 6)
    pub wall_matrix: Vec<f64>,
    /// Window matrix: (MAX_WALLS, 6)
    pub window_matrix: Vec<f64>,
    /// Adjacency matrix: (MAX_ZONES, MAX_ZONES)
    pub adjacency_matrix: Vec<f64>,
    /// Zone properties: (MAX_ZONES, 5)
    pub zone_properties: Vec<f64>,
    /// Summary: [num_zones, num_walls, num_windows, num_doors, total_area, total_volume]
    pub summary: Vec<f64>,
}

impl GeometryTensor {
    /// Create a new empty GeometryTensor.
    pub fn new() -> Self {
        let zone_coords = vec![0.0; MAX_ZONES * 20];
        let wall_matrix = vec![0.0; MAX_WALLS * 6];
        let window_matrix = vec![0.0; MAX_WALLS * 6];
        let adjacency_matrix = vec![0.0; MAX_ZONES * MAX_ZONES];
        let zone_properties = vec![0.0; MAX_ZONES * 5];
        let summary = vec![0.0; 6];

        GeometryTensor {
            zone_coords,
            wall_matrix,
            window_matrix,
            adjacency_matrix,
            zone_properties,
            summary,
        }
    }

    /// Create a GeometryTensor from numpy arrays (zero-copy when possible).
    #[cfg(feature = "python-bindings")]
    pub fn from_numpy_arrays(
        zone_coords: &[f64],
        wall_matrix: &[f64],
        window_matrix: &[f64],
        adjacency_matrix: &[f64],
        zone_properties: &[f64],
        summary: &[f64],
    ) -> Result<Self, String> {
        // Validate sizes
        if zone_coords.len() != MAX_ZONES * 20 {
            return Err(format!(
                "zone_coords has {} elements, expected {}",
                zone_coords.len(),
                MAX_ZONES * 20
            ));
        }
        if wall_matrix.len() != MAX_WALLS * 6 {
            return Err(format!(
                "wall_matrix has {} elements, expected {}",
                wall_matrix.len(),
                MAX_WALLS * 6
            ));
        }
        if window_matrix.len() != MAX_WALLS * 6 {
            return Err(format!(
                "window_matrix has {} elements, expected {}",
                window_matrix.len(),
                MAX_WALLS * 6
            ));
        }
        if adjacency_matrix.len() != MAX_ZONES * MAX_ZONES {
            return Err(format!(
                "adjacency_matrix has {} elements, expected {}",
                adjacency_matrix.len(),
                MAX_ZONES * MAX_ZONES
            ));
        }
        if zone_properties.len() != MAX_ZONES * 5 {
            return Err(format!(
                "zone_properties has {} elements, expected {}",
                zone_properties.len(),
                MAX_ZONES * 5
            ));
        }

        Ok(GeometryTensor {
            zone_coords: zone_coords.to_vec(),
            wall_matrix: wall_matrix.to_vec(),
            window_matrix: window_matrix.to_vec(),
            adjacency_matrix: adjacency_matrix.to_vec(),
            zone_properties: zone_properties.to_vec(),
            summary: summary.to_vec(),
        })
    }

    /// Get the number of zones in the geometry.
    pub fn num_zones(&self) -> usize {
        self.summary[0] as usize
    }

    /// Get the number of walls in the geometry.
    pub fn num_walls(&self) -> usize {
        self.summary[1] as usize
    }

    /// Get the total floor area.
    pub fn total_area(&self) -> f64 {
        self.summary[4]
    }

    /// Get the total volume.
    pub fn total_volume(&self) -> f64 {
        self.summary[5]
    }

    /// Get zone coordinates at index.
    pub fn get_zone_coords(&self, index: usize) -> Option<&[f64; 20]> {
        if index < MAX_ZONES {
            let start = index * 20;
            let slice = &self.zone_coords[start..start + 20];
            // Convert slice to array
            Some(unsafe { &*(slice.as_ptr() as *const [f64; 20]) })
        } else {
            None
        }
    }

    /// Get wall data at index.
    pub fn get_wall(&self, index: usize) -> Option<WallData> {
        if index < MAX_WALLS {
            let start = index * 6;
            let data = &self.wall_matrix[start..start + 6];
            Some(WallData {
                x1: data[0],
                y1: data[1],
                x2: data[2],
                y2: data[3],
                height: data[4],
                thickness: data[5],
            })
        } else {
            None
        }
    }

    /// Check if zones i and j are adjacent.
    pub fn zones_adjacent(&self, i: usize, j: usize) -> bool {
        if i < MAX_ZONES && j < MAX_ZONES {
            let idx = i * MAX_ZONES + j;
            self.adjacency_matrix[idx] > 0.5
        } else {
            false
        }
    }

    /// Validate the geometry tensor.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for NaN
        if self.zone_coords.iter().any(|x| x.is_nan()) {
            issues.push("zone_coords contains NaN".to_string());
        }
        if self.wall_matrix.iter().any(|x| x.is_nan()) {
            issues.push("wall_matrix contains NaN".to_string());
        }

        // Check for negative areas
        if self.zone_properties.iter().any(|&x| x.is_nan()) {
            // Check floor_area (index 0 in each zone's properties)
            for i in 0..MAX_ZONES {
                let area = self.zone_properties[i * 5];
                if area < 0.0 {
                    issues.push(format!("Zone {} has negative area: {}", i, area));
                }
            }
        }

        // Check adjacency symmetry
        for i in 0..MAX_ZONES {
            for j in 0..MAX_ZONES {
                let a = self.adjacency_matrix[i * MAX_ZONES + j];
                let b = self.adjacency_matrix[j * MAX_ZONES + i];
                if (a > 0.5) != (b > 0.5) {
                    issues.push(format!(
                        "Adjacency matrix asymmetry at ({}, {}): {} vs {}",
                        i, j, a, b
                    ));
                    break;
                }
            }
        }

        issues
    }
}

/// Wall data structure.
#[derive(Debug, Clone, Copy)]
pub struct WallData {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub height: f64,
    pub thickness: f64,
}

impl WallData {
    /// Calculate wall length.
    pub fn length(&self) -> f64 {
        let dx = self.x2 - self.x1;
        let dy = self.y2 - self.y1;
        (dx * dx + dy * dy).sqrt()
    }

    /// Calculate wall area.
    pub fn area(&self) -> f64 {
        self.length() * self.height
    }
}

impl Default for GeometryTensor {
    fn default() -> Self {
        Self::new()
    }
}

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
    pub const ALL: [ManifoldIndex; MANIFOLD_DIM] =
        [Self::Air, Self::Wall, Self::Roof, Self::Floor];
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
            assert!(
                c > 0.0,
                "capacitances[{label}] must be > 0 (got {c})"
            );
        }
        for (label, &g) in r_tr_surface.iter().enumerate() {
            assert!(
                g > 0.0,
                "r_tr_surface[{label}] must be > 0 (got {g})"
            );
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

    /// **Stub for the gauge-theoretic parallel transport** that `GaugeSolver`
    /// (Phase 1b, #1462) will implement in full.
    ///
    /// Computes the covariant derivative along the time axis by `dt` seconds:
    ///
    /// ```text
    ///   ∇_A T  =  metric_tensor · scalar_field  +  gauge_connection
    ///   T_new  =  scalar_field  +  dt · ∇_A T
    /// ```
    ///
    /// Equivalently: forward Euler along the time-axis curve at rate `∇_A T`,
    /// returning the post-transport state as a fresh [`Vector4`].
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
        let covariant_derivative =
            self.metric_tensor * self.scalar_field + self.gauge_connection;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometry_tensor_creation() {
        let tensor = GeometryTensor::new();
        assert_eq!(tensor.zone_coords.len(), MAX_ZONES * 20);
        assert_eq!(tensor.wall_matrix.len(), MAX_WALLS * 6);
        assert_eq!(tensor.window_matrix.len(), MAX_WALLS * 6);
        assert_eq!(tensor.adjacency_matrix.len(), MAX_ZONES * MAX_ZONES);
        assert_eq!(tensor.zone_properties.len(), MAX_ZONES * 5);
        assert_eq!(tensor.summary.len(), 6);
    }

    #[test]
    fn test_geometry_tensor_default() {
        let tensor = GeometryTensor::default();
        assert_eq!(tensor.zone_coords.len(), MAX_ZONES * 20);
        assert_eq!(tensor.wall_matrix.len(), MAX_WALLS * 6);
    }

    #[test]
    fn test_wall_data_length_horizontal() {
        let wall = WallData {
            x1: 0.0,
            y1: 0.0,
            x2: 3.0,
            y2: 0.0,
            height: 2.4,
            thickness: 0.2,
        };
        assert!((wall.length() - 3.0).abs() < 1e-10);
        assert!((wall.area() - 7.2).abs() < 1e-10);
    }

    #[test]
    fn test_wall_data_length_vertical() {
        let wall = WallData {
            x1: 0.0,
            y1: 0.0,
            x2: 0.0,
            y2: 4.0,
            height: 2.4,
            thickness: 0.2,
        };
        assert!((wall.length() - 4.0).abs() < 1e-10);
        assert!((wall.area() - 9.6).abs() < 1e-10);
    }

    #[test]
    fn test_wall_data_diagonal() {
        let wall = WallData {
            x1: 0.0,
            y1: 0.0,
            x2: 3.0,
            y2: 4.0,
            height: 2.4,
            thickness: 0.2,
        };
        assert!((wall.length() - 5.0).abs() < 1e-10);
        assert!((wall.area() - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_wall_data_zero_height_area() {
        let wall = WallData {
            x1: 0.0,
            y1: 0.0,
            x2: 3.0,
            y2: 4.0,
            height: 0.0,
            thickness: 0.2,
        };
        assert_eq!(wall.area(), 0.0);
    }

    #[test]
    fn test_num_zones() {
        let mut tensor = GeometryTensor::new();
        tensor.summary[0] = 5.0;
        assert_eq!(tensor.num_zones(), 5);
    }

    #[test]
    fn test_num_walls() {
        let mut tensor = GeometryTensor::new();
        tensor.summary[1] = 20.0;
        assert_eq!(tensor.num_walls(), 20);
    }

    #[test]
    fn test_total_area() {
        let mut tensor = GeometryTensor::new();
        tensor.summary[4] = 500.0;
        assert!((tensor.total_area() - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_volume() {
        let mut tensor = GeometryTensor::new();
        tensor.summary[5] = 1500.0;
        assert!((tensor.total_volume() - 1500.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_zone_coords_valid() {
        let mut tensor = GeometryTensor::new();
        // Set some test values in zone 0
        tensor.zone_coords[0] = 1.0;
        tensor.zone_coords[1] = 2.0;
        tensor.zone_coords[19] = 3.0;

        let coords = tensor.get_zone_coords(0).unwrap();
        assert_eq!(coords[0], 1.0);
        assert_eq!(coords[1], 2.0);
        assert_eq!(coords[19], 3.0);
    }

    #[test]
    fn test_get_zone_coords_invalid() {
        let tensor = GeometryTensor::new();
        assert!(tensor.get_zone_coords(MAX_ZONES).is_none());
        assert!(tensor.get_zone_coords(MAX_ZONES + 10).is_none());
    }

    #[test]
    fn test_get_wall_valid() {
        let mut tensor = GeometryTensor::new();
        // Set some test values for wall 0
        tensor.wall_matrix[0] = 1.0;
        tensor.wall_matrix[1] = 2.0;
        tensor.wall_matrix[2] = 3.0;
        tensor.wall_matrix[3] = 4.0;
        tensor.wall_matrix[4] = 2.5;
        tensor.wall_matrix[5] = 0.2;

        let wall = tensor.get_wall(0).unwrap();
        assert_eq!(wall.x1, 1.0);
        assert_eq!(wall.y1, 2.0);
        assert_eq!(wall.x2, 3.0);
        assert_eq!(wall.y2, 4.0);
        assert_eq!(wall.height, 2.5);
        assert_eq!(wall.thickness, 0.2);
    }

    #[test]
    fn test_get_wall_invalid() {
        let tensor = GeometryTensor::new();
        assert!(tensor.get_wall(MAX_WALLS).is_none());
        assert!(tensor.get_wall(MAX_WALLS + 10).is_none());
    }

    #[test]
    fn test_zones_adjacent_true() {
        let mut tensor = GeometryTensor::new();
        let idx = 2 * MAX_ZONES + 3;
        tensor.adjacency_matrix[idx] = 1.0;
        assert!(tensor.zones_adjacent(2, 3));
    }

    #[test]
    fn test_zones_adjacent_false() {
        let tensor = GeometryTensor::new();
        // Default values are 0.0
        assert!(!tensor.zones_adjacent(0, 1));
    }

    #[test]
    fn test_zones_adjacent_symmetry() {
        let mut tensor = GeometryTensor::new();
        // Set adjacency for zone 0 -> 1
        let idx_01 = 1;
        tensor.adjacency_matrix[idx_01] = 1.0;
        // Also set reverse
        let idx_10 = MAX_ZONES;
        tensor.adjacency_matrix[idx_10] = 1.0;

        assert!(tensor.zones_adjacent(0, 1));
        assert!(tensor.zones_adjacent(1, 0));
    }

    #[test]
    fn test_zones_adjacent_out_of_bounds() {
        let tensor = GeometryTensor::new();
        assert!(!tensor.zones_adjacent(MAX_ZONES, 0));
        assert!(!tensor.zones_adjacent(0, MAX_ZONES));
    }

    #[test]
    fn test_validate_clean() {
        let tensor = GeometryTensor::new();
        let issues = tensor.validate();
        // Empty tensor should have no issues (zeros are valid)
        assert!(issues.is_empty());
    }

    #[test]
    fn test_validate_zone_coords_nan() {
        let mut tensor = GeometryTensor::new();
        tensor.zone_coords[0] = f64::NAN;
        let issues = tensor.validate();
        assert!(issues.iter().any(|s| s.contains("NaN")));
    }

    #[test]
    fn test_validate_wall_matrix_nan() {
        let mut tensor = GeometryTensor::new();
        tensor.wall_matrix[0] = f64::NAN;
        let issues = tensor.validate();
        assert!(issues
            .iter()
            .any(|s| s.contains("NaN") && s.contains("wall")));
    }

    #[test]
    fn test_validate_negative_zone_area() {
        let mut tensor = GeometryTensor::new();
        tensor.zone_properties[0] = -50.0; // Negative area for zone 0
        tensor.zone_properties[1] = f64::NAN; // Add NaN to trigger nested check
        let issues = tensor.validate();
        assert!(issues.iter().any(|s| s.contains("negative area")));
    }

    #[test]
    fn test_validate_adjacency_asymmetry() {
        let mut tensor = GeometryTensor::new();
        // Set zone 0 -> 1 adjacency but not 1 -> 0
        let idx_01 = 1;
        tensor.adjacency_matrix[idx_01] = 1.0;
        let issues = tensor.validate();
        assert!(issues.iter().any(|s| s.contains("asymmetry")));
    }

    #[test]
    fn test_validate_multiple_issues() {
        let mut tensor = GeometryTensor::new();
        tensor.zone_coords[0] = f64::NAN;
        tensor.zone_properties[0] = -50.0;
        tensor.zone_properties[1] = f64::NAN; // Add NaN to trigger area check
        let issues = tensor.validate();
        assert!(issues.len() >= 2);
    }

    #[test]
    fn test_geometry_tensor_clone() {
        let mut tensor = GeometryTensor::new();
        tensor.summary[0] = 3.0;
        tensor.summary[4] = 100.0;
        let cloned = tensor.clone();
        assert_eq!(cloned.summary[0], 3.0);
        assert_eq!(cloned.summary[4], 100.0);
    }

    #[test]
    fn test_constants_values() {
        assert_eq!(MAX_ZONES, 100);
        assert_eq!(MAX_WALLS, 500);
        assert_eq!(ZONE_COORDS_DIMS, (100, 20));
        assert_eq!(WALL_MATRIX_DIMS, (500, 6));
        assert_eq!(WINDOW_MATRIX_DIMS, (500, 6));
        assert_eq!(ADJACENCY_MATRIX_DIMS, (100, 100));
        assert_eq!(ZONE_PROPERTIES_DIMS, (100, 5));
    }

    #[test]
    fn test_wall_data_debug() {
        let wall = WallData {
            x1: 1.0,
            y1: 2.0,
            x2: 3.0,
            y2: 4.0,
            height: 2.5,
            thickness: 0.2,
        };
        let debug_str = format!("{:?}", wall);
        assert!(debug_str.contains("WallData"));
    }

    #[test]
    fn test_wall_data_copy() {
        let wall = WallData {
            x1: 1.0,
            y1: 2.0,
            x2: 3.0,
            y2: 4.0,
            height: 2.5,
            thickness: 0.2,
        };
        let copied = wall;
        assert_eq!(copied.x1, 1.0);
        assert_eq!(copied.y2, 4.0);
    }

    #[test]
    fn test_geometry_tensor_debug() {
        let tensor = GeometryTensor::new();
        let debug_str = format!("{:?}", tensor);
        assert!(debug_str.contains("GeometryTensor"));
    }

    #[test]
    fn test_get_multiple_zones() {
        let mut tensor = GeometryTensor::new();
        // Set markers for first 3 zones
        for i in 0..3 {
            let idx = i * 20;
            tensor.zone_coords[idx] = i as f64;
        }

        for i in 0..3 {
            let coords = tensor.get_zone_coords(i).unwrap();
            assert_eq!(coords[0], i as f64);
        }
    }

    #[test]
    fn test_get_multiple_walls() {
        let mut tensor = GeometryTensor::new();
        // Set markers for first 5 walls
        for i in 0..5 {
            let idx = i * 6;
            tensor.wall_matrix[idx] = i as f64;
        }

        for i in 0..5 {
            let wall = tensor.get_wall(i).unwrap();
            assert_eq!(wall.x1, i as f64);
        }
    }

    #[test]
    fn test_wall_data_zero_length() {
        let wall = WallData {
            x1: 0.0,
            y1: 0.0,
            x2: 0.0,
            y2: 0.0,
            height: 2.4,
            thickness: 0.2,
        };
        assert_eq!(wall.length(), 0.0);
        assert_eq!(wall.area(), 0.0);
    }

    // -------------------------------------------------------------------------
    // ThermalManifold tests (Issue #1461 — Phase 1a)
    // -------------------------------------------------------------------------
    //
    // The following tests cover the gauge-theory data structure introduced in
    // #1461 and exercise the boundary surface that downstream PRs (#1462
    // GaugeSolver, #1463 surrogate training, #1464 QUBO mapping, #1465 Case
    // 900 validation) will depend on.

    /// 4-D matrix dimensions and 4-vector field/connection dimensions — the
    /// pinned shape that the GaugeSolver (#1462) and surrogate (#1463) treat
    /// as invariants.
    #[test]
    fn test_manifold_has_four_dimensions() {
        let m = ThermalManifold::new_flat();
        assert_eq!(m.metric_tensor.nrows(), MANIFOLD_DIM);
        assert_eq!(m.metric_tensor.ncols(), MANIFOLD_DIM);
        assert_eq!(m.scalar_field.len(), MANIFOLD_DIM);
        assert_eq!(m.gauge_connection.len(), MANIFOLD_DIM);
        assert_eq!(MANIFOLD_DIM, 4);
    }

    /// `new_flat()` should hand back a manifold with the identity metric, zero
    /// field, zero connection, zero timestep — the unit element of the product
    /// geometry used by GaugeSolver (#1462) for transport.
    #[test]
    fn test_manifold_new_flat_is_identity() {
        let m = ThermalManifold::new_flat();
        assert_eq!(m.metric_tensor, Matrix4::identity());
        assert_eq!(m.scalar_field, Vector4::zeros());
        assert_eq!(m.gauge_connection, Vector4::zeros());
        assert_eq!(m.dt_seconds, 0.0);
    }

    /// `Default::default()` must be the flat manifold so callers can build a
    /// placeholder and fill in geometry later.
    #[test]
    fn test_manifold_default_matches_new_flat() {
        let m = ThermalManifold::default();
        assert_eq!(m.metric_tensor, ThermalManifold::new_flat().metric_tensor);
        assert_eq!(m.scalar_field, ThermalManifold::new_flat().scalar_field);
    }

    /// `ManifoldIndex` enum maps to its `#[repr(usize)]` discriminant so the
    /// down-stream CRs can do `manifold.scalar_field[idx as usize] = ...`
    /// without a runtime match.
    #[test]
    fn test_manifold_index_layout_matches_repr_usize() {
        for (i, idx) in ManifoldIndex::ALL.iter().enumerate() {
            assert_eq!(*idx as usize, i);
        }
        assert_eq!(ManifoldIndex::from_usize(0), ManifoldIndex::Air);
        assert_eq!(ManifoldIndex::from_usize(1), ManifoldIndex::Wall);
        assert_eq!(ManifoldIndex::from_usize(2), ManifoldIndex::Roof);
        assert_eq!(ManifoldIndex::from_usize(3), ManifoldIndex::Floor);
        assert_eq!(ManifoldIndex::ALL.len(), MANIFOLD_DIM);
    }

    /// 5R1C embedding: only `T_air` and `T_mass` are present; the wall/roof/floor
    /// mass slots are parked at field = 0 with metric `(2,2) = (3,3) = 0`. The
    /// active 2×2 sub-block reproduces the canonical 5R1C dissipative operator.
    #[test]
    fn test_from_5r1c_layout() {
        let r_eq = 0.10; // K/W
        let c_air = 10_000.0; // J/K
        let c_mass = 50_000.0; // J/K
        let m = ThermalManifold::from_5r1c_parameters(20.0, 21.0, r_eq, c_air, c_mass);

        // Field layout — only air and mass active.
        assert_eq!(m.scalar_field[ManifoldIndex::Air as usize], 20.0);
        assert_eq!(m.scalar_field[ManifoldIndex::Wall as usize], 21.0);
        assert_eq!(m.scalar_field[ManifoldIndex::Roof as usize], 0.0);
        assert_eq!(m.scalar_field[ManifoldIndex::Floor as usize], 0.0);

        // Metric — the 2×2 active block matches the discrete 5R1C operator.
        let g_eq = 1.0 / r_eq;
        let expected_self_air = -g_eq / c_air;
        let expected_cross_air = g_eq / c_air;
        let expected_self_mass = -g_eq / c_mass;
        let expected_cross_mass = g_eq / c_mass;

        let diff = |a: f64, b: f64| (a - b).abs() < 1e-12;
        assert!(diff(m.metric_tensor[(0, 0)], expected_self_air));
        assert!(diff(m.metric_tensor[(0, 1)], expected_cross_air));
        assert!(diff(m.metric_tensor[(1, 0)], expected_cross_mass));
        assert!(diff(m.metric_tensor[(1, 1)], expected_self_mass));

        // Inert slots — roof/floor metric entries are 0; off-diagonals tying
        // air/mass to roof/floor are also 0.
        assert_eq!(m.metric_tensor[(2, 2)], 0.0);
        assert_eq!(m.metric_tensor[(3, 3)], 0.0);
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                if !(i <= 1 && j <= 1) {
                    assert_eq!(
                        m.metric_tensor[(i, j)],
                        0.0,
                        "metric[{i},{j}] should be 0 in the 5R1C embedding"
                    );
                }
            }
        }
    }

    /// 5R1C embedding reproduces the **simplified 5R1C ODE** that the
    /// `from_5r1c_parameters` constructor encodes EXACTLY, to within
    /// floating-point rounding across many timesteps (verified by Python in
    /// `.agents/results/issue-1461-python-verification.py`). Phase 1a's whole
    /// point is that the matrix form is bit-identical to the discrete flow
    /// map that [`ThermalManifold::from_5r1c_parameters`] represents, so
    /// `GaugeSolver` (#1462) can shadow the legacy path without numeric drift
    /// on day 1.
    ///
    /// **What "simplified" means**: the constructor embeds the 5R1C scene
    /// where the mass node couples to the **air node only** through `R_eq`
    /// (no separate `R_ow` envelope resistance). The corresponding ODE is:
    ///
    /// ```text
    ///   C_air  · dT_air/dt = (T_mass - T_air)/R_eq + Q_internal
    ///   C_mass · dT_mass/dt = (T_air - T_mass)/R_eq + Q_solar
    /// ```
    ///
    /// Outdoor coupling is intentionally absent — it's the GaugeSolver's job
    /// (#1462) to translate raw BCs (irradiance, outdoor temp) into the
    /// gauge_connection vector under a richer scene. The legacy reference
    /// below uses exactly this ODE form.
    #[test]
    fn test_from_5r1c_matches_legacy_ode() {
        let r_eq = 0.10;
        let c_air = 10_000.0;
        let c_mass = 50_000.0;
        let t_air_0 = 20.0;
        let t_mass_0 = 20.0;
        let q_int = 200.0;
        let q_solar = 800.0;
        let dt = 60.0;

        // Build the geometric manifold (active 2×2 + inert roof/floor).
        let mut manifold = ThermalManifold::from_5r1c_parameters(
            t_air_0,
            t_mass_0,
            r_eq,
            c_air,
            c_mass,
        );
        // Inject the BC terms (internal gains → air; solar → mass) per the
        // simplified 5R1C ODE above.
        manifold.gauge_connection[ManifoldIndex::Air as usize] = q_int / c_air;
        manifold.gauge_connection[ManifoldIndex::Wall as usize] = q_solar / c_mass;

        // Step in lock-step: simultaneous forward Euler (matches the
        // pattern in `physics/five_r1c_solver.rs::FiveR1CSolver::step`) on
        // the legacy side, and `compute_parallel_transport` on the matrix
        // side. The two are the same linear map, so they must agree.
        let mut legacy_air = t_air_0;
        let mut legacy_mass = t_mass_0;
        for _step in 0..50 {
            // Pre-step rates (simultaneous Euler).
            let air_rate = ((legacy_mass - legacy_air) / r_eq + q_int) / c_air;
            let mass_rate = ((legacy_air - legacy_mass) / r_eq + q_solar) / c_mass;
            legacy_air += dt * air_rate;
            legacy_mass += dt * mass_rate;

            // Geometric transport.
            let t_next = manifold.compute_parallel_transport(dt);
            manifold.scalar_field[ManifoldIndex::Air as usize] = t_next[0];
            manifold.scalar_field[ManifoldIndex::Wall as usize] = t_next[1];

            let matrix_air = manifold.scalar_field[ManifoldIndex::Air as usize];
            let matrix_mass = manifold.scalar_field[ManifoldIndex::Wall as usize];
            let tol = 1e-9 * (1.0 + legacy_air.abs());
            assert!(
                (matrix_air - legacy_air).abs() < tol,
                "5R1C ↔ matrix form drift on T_air (step={_step}): \
                 matrix={matrix_air:.6e}, legacy={legacy_air:.6e}, |Δ|={:.3e}",
                (matrix_air - legacy_air).abs()
            );
            assert!(
                (matrix_mass - legacy_mass).abs() < tol,
                "5R1C ↔ matrix form drift on T_mass (step={_step}): \
                 matrix={matrix_mass:.6e}, legacy={legacy_mass:.6e}, |Δ|={:.3e}",
                (matrix_mass - legacy_mass).abs()
            );
            // Roof/floor slots must remain at 0 (the 5R1C scene is 2-D).
            assert_eq!(
                manifold.scalar_field[ManifoldIndex::Roof as usize],
                0.0
            );
            assert_eq!(
                manifold.scalar_field[ManifoldIndex::Floor as usize],
                0.0
            );
        }
    }

    /// 9R4C embedding populates the full 4-D dissipative operator and writes
    /// the temperatures into the matching slots.
    #[test]
    fn test_from_9r4c_layout() {
        let temperatures = [21.0, 19.0, 22.0, 18.0];
        let capacitances = [10_000.0, 50_000.0, 30_000.0, 80_000.0];
        let r_tr = [120.0, 80.0, 200.0]; // g_tr per surface

        let m = ThermalManifold::from_9r4c_parameters(
            temperatures,
            capacitances,
            r_tr,
            None,
        );

        // Field slots.
        assert_eq!(m.scalar_field[ManifoldIndex::Air as usize], 21.0);
        assert_eq!(m.scalar_field[ManifoldIndex::Wall as usize], 19.0);
        assert_eq!(m.scalar_field[ManifoldIndex::Roof as usize], 22.0);
        assert_eq!(m.scalar_field[ManifoldIndex::Floor as usize], 18.0);

        // Air row: self = -(g_wall + g_roof + g_floor)/C_air.
        let g_total = 120.0 + 80.0 + 200.0;
        let expected_self_air = -g_total / 10_000.0;
        let diff = |a: f64, b: f64| (a - b).abs() < 1e-12;
        assert!(diff(m.metric_tensor[(0, 0)], expected_self_air));
        assert!(diff(m.metric_tensor[(0, 1)], 120.0 / 10_000.0));
        assert!(diff(m.metric_tensor[(0, 2)], 80.0 / 10_000.0));
        assert!(diff(m.metric_tensor[(0, 3)], 200.0 / 10_000.0));

        // No inter-mass coupling when `r_cross = None`.
        assert_eq!(m.metric_tensor[(1, 2)], 0.0);
        assert_eq!(m.metric_tensor[(1, 3)], 0.0);
        assert_eq!(m.metric_tensor[(2, 3)], 0.0);
    }

    /// `compute_parallel_transport` returns a fresh `Vector4<f64>` and does not
    /// mutate `self` — the GaugeSolver (#1462) explicitly assigns the result
    /// back into the manifold, so the non-mutating signature is part of the
    /// public contract.
    #[test]
    fn test_parallel_transport_signature_does_not_mutate() {
        let mut m = ThermalManifold::new_flat();
        m.scalar_field = Vector4::new(20.0, 19.0, 21.0, 18.0);
        m.gauge_connection = Vector4::new(10.0, 0.0, 0.0, 0.0);
        let snapshot_field = m.scalar_field;
        let snapshot_conn = m.gauge_connection;

        let _transported: Vector4<f64> = m.compute_parallel_transport(60.0);

        assert_eq!(m.scalar_field, snapshot_field, "parallel_transport must not mutate scalar_field");
        assert_eq!(m.gauge_connection, snapshot_conn, "parallel_transport must not mutate gauge_connection");
    }

    /// The zero source / zero field manifold is a fixed point of parallel
    /// transport — useful as a trivial sanity check.
    #[test]
    fn test_parallel_transport_zero_field_zero_source_is_fixed_point() {
        let m = ThermalManifold::new_flat();
        let transported = m.compute_parallel_transport(123.456);
        for v in transported.iter() {
            assert_eq!(*v, 0.0, "zero manifold transported must remain at 0");
        }
    }

    /// Nonzero source produces a non-trivial field in the air slot after one
    /// transport step — confirms the matrix-vector product + add layout.
    #[test]
    fn test_parallel_transport_unit_source_advances_air_slot() {
        let mut m = ThermalManifold::new_flat();
        // Identity metric, zero field, unit source on air slot.
        m.gauge_connection[ManifoldIndex::Air as usize] = 1.0;

        let t_new = m.compute_parallel_transport(2.0);
        // dT/dt = 1·0 + 1 = 1 (air slot); T_new = 0 + 2·1 = 2.
        assert_eq!(t_new[ManifoldIndex::Air as usize], 2.0);
        // Other slots: no source, no field, no transport.
        for axis in [ManifoldIndex::Wall, ManifoldIndex::Roof, ManifoldIndex::Floor] {
            assert_eq!(
                t_new[axis as usize],
                0.0,
                "{:?} should remain 0 under isolated air-slot source",
                axis
            );
        }
    }

    /// Per the #1461 epic, no hardcoded HVAC clamps live in the manifold path —
    /// arbitrary large connection values must transport without intervention.
    /// (The 100 kW cap from the 5R1C production path is intentionally absent.)
    #[test]
    fn test_parallel_transport_does_not_clamp_arbitrary_sources() {
        let mut m = ThermalManifold::new_flat();
        // 1 MW into the air slot — many orders of magnitude above the legacy
        // 100 kW cap. The manifold must not clamp or bounds-check.
        m.gauge_connection[ManifoldIndex::Air as usize] = 1_000_000.0;
        m.gauge_connection[ManifoldIndex::Wall as usize] = -500_000.0;

        let t_new = m.compute_parallel_transport(1.0);
        // dT/dt at air slot = 0 (I·0) + 1e6 = 1e6; transport = 0 + 1·1e6 = 1e6.
        assert_eq!(t_new[ManifoldIndex::Air as usize], 1_000_000.0);
        assert_eq!(t_new[ManifoldIndex::Wall as usize], -500_000.0);
    }

    /// Negative timestep must be a legal input — it just produces a backward
    /// transport (geometric interpretability). The manifold imposes no sign
    /// restriction on `dt`.
    #[test]
    fn test_parallel_transport_negative_dt_is_backward_euler() {
        let mut m = ThermalManifold::new_flat();
        m.gauge_connection[ManifoldIndex::Air as usize] = 10.0;

        let forward = m.compute_parallel_transport(1.0);
        let backward = m.compute_parallel_transport(-1.0);
        for i in 0..MANIFOLD_DIM {
            assert!(
                (forward[i] + backward[i]).abs() < 1e-12,
                "forward and backward transports must cancel at slot {i}"
            );
        }
    }

    /// `validate` accepts a well-formed manifold and rejects NaN/Infinity
    /// across all three storage buffers.
    #[test]
    fn test_validate_accepts_well_formed_manifold() {
        let m = ThermalManifold::new_flat();
        assert!(m.validate().is_ok());

        let mut well_formed = ThermalManifold::new_flat();
        well_formed.scalar_field = Vector4::new(20.0, 19.5, 21.0, 18.0);
        well_formed.gauge_connection = Vector4::new(100.0, 200.0, 300.0, 400.0);
        assert!(well_formed.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_nan_in_metric() {
        let mut m = ThermalManifold::new_flat();
        m.metric_tensor[(1, 2)] = f64::NAN;
        let err = m.validate();
        assert_eq!(err, Err(ManifoldError::NonFiniteMetric { row: 1, col: 2 }));
    }

    #[test]
    fn test_validate_rejects_inf_in_metric() {
        let mut m = ThermalManifold::new_flat();
        m.metric_tensor[(0, 0)] = f64::INFINITY;
        let err = m.validate();
        assert_eq!(err, Err(ManifoldError::NonFiniteMetric { row: 0, col: 0 }));
    }

    #[test]
    fn test_validate_rejects_nan_in_scalar_field() {
        let mut m = ThermalManifold::new_flat();
        m.scalar_field[ManifoldIndex::Roof as usize] = f64::NAN;
        assert_eq!(m.validate(), Err(ManifoldError::NonFiniteField));
    }

    #[test]
    fn test_validate_rejects_nan_in_gauge_connection() {
        let mut m = ThermalManifold::new_flat();
        m.gauge_connection[ManifoldIndex::Floor as usize] = f64::NAN;
        assert_eq!(m.validate(), Err(ManifoldError::NonFiniteConnection));
    }

    /// `gauge_connection_sum` is the First-Law diagnostic that
    /// `tools/piml_loss.py` (#1463) and the ASHRAE 140 Case 900 CI gate (#1465)
    /// will use to penalize / verify energy conservation across the gauge
    /// transport. Zero for an isolated zone; nonzero for a powered zone.
    #[test]
    fn test_gauge_connection_sum_is_sum_of_components() {
        let mut m = ThermalManifold::new_flat();
        m.gauge_connection = Vector4::new(10.0, -5.0, 2.5, -7.5);
        assert_eq!(m.gauge_connection_sum(), 0.0);

        m.gauge_connection = Vector4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(m.gauge_connection_sum(), 10.0);
    }

    /// `Clone` produces an independent deep copy — GaugeSolver (#1462) might
    /// shadow-step without disturbing the in-flight manifold.
    #[test]
    fn test_manifold_clone_is_independent() {
        let mut m = ThermalManifold::new_flat();
        m.scalar_field = Vector4::new(1.0, 2.0, 3.0, 4.0);
        m.gauge_connection = Vector4::new(-1.0, -2.0, -3.0, -4.0);
        m.metric_tensor[(0, 0)] = -0.5;

        let cloned = m.clone();
        assert_eq!(cloned.scalar_field, m.scalar_field);
        assert_eq!(cloned.gauge_connection, m.gauge_connection);
        assert_eq!(cloned.metric_tensor, m.metric_tensor);

        // Mutate the original — the clone must be untouched.
        m.scalar_field[0] = 99.0;
        m.metric_tensor[(0, 0)] = -9.0;
        assert_eq!(cloned.scalar_field[0], 1.0);
        assert_eq!(cloned.metric_tensor[(0, 0)], -0.5);
    }

    /// `Debug` rendering includes the type name — useful for the tracetape
    /// `tracing::debug!` instrumentation GaugeSolver (#1462) will emit.
    #[test]
    fn test_manifold_debug_includes_type_name() {
        let m = ThermalManifold::new_flat();
        let dbg = format!("{:?}", m);
        assert!(dbg.contains("ThermalManifold"), "{dbg}");
    }

    /// `from_5r1c_parameters` must reject non-physical inputs (negative or
    /// zero R/C). Negative capacitances would silently flip the gauge-stability
    /// guarantee; we want them caught at construction time.
    #[test]
    #[should_panic(expected = "r_eq must be > 0")]
    fn test_from_5r1c_rejects_zero_resistance() {
        let _ = ThermalManifold::from_5r1c_parameters(20.0, 20.0, 0.0, 10_000.0, 50_000.0);
    }

    #[test]
    #[should_panic(expected = "c_air must be > 0")]
    fn test_from_5r1c_rejects_zero_air_capacitance() {
        let _ = ThermalManifold::from_5r1c_parameters(20.0, 20.0, 0.1, 0.0, 50_000.0);
    }

    #[test]
    #[should_panic(expected = "c_mass must be > 0")]
    fn test_from_5r1c_rejects_zero_mass_capacitance() {
        let _ = ThermalManifold::from_5r1c_parameters(20.0, 20.0, 0.1, 10_000.0, -1.0);
    }

    /// `from_9r4c_parameters` rejects non-physical (negative) capacitances
    /// the same way.
    #[test]
    #[should_panic(expected = "capacitances[0] must be > 0")]
    fn test_from_9r4c_rejects_zero_capacitance() {
        let _ = ThermalManifold::from_9r4c_parameters(
            [20.0, 20.0, 20.0, 20.0],
            [0.0, 50_000.0, 30_000.0, 80_000.0],
            [120.0, 80.0, 200.0],
            None,
        );
    }

    /// `compute_parallel_transport` returns a `Vector4<f64>` with the
    /// gauge-invariant shape — after a non-trivial transport, the result has
    /// 4 finite components.
    #[test]
    fn test_parallel_transport_returns_well_typed_vector4() {
        let mut m = ThermalManifold::new_flat();
        m.scalar_field = Vector4::new(20.0, 19.0, 21.0, 18.0);
        m.gauge_connection = Vector4::new(0.0, -100.0, 0.0, 0.0);

        let transported: Vector4<f64> = m.compute_parallel_transport(60.0);
        assert_eq!(transported.len(), MANIFOLD_DIM);
        for v in transported.iter() {
            assert!(v.is_finite(), "transport entries must stay finite");
        }
    }

    /// Gauge connection symmetric: transporting with `+A` and then with `-A`
    /// on the same field must cancel the source contribution.
    #[test]
    fn test_parallel_transport_connection_reversibility() {
        let mut m = ThermalManifold::new_flat();
        m.scalar_field = Vector4::new(20.0, 19.0, 21.0, 18.0);
        m.gauge_connection = Vector4::new(10.0, -5.0, 2.5, -7.5);

        let plus = m.compute_parallel_transport(0.5);
        let transport_no_conn = {
            let mut no_conn = m.clone();
            no_conn.gauge_connection = Vector4::zeros();
            no_conn.compute_parallel_transport(0.5)
        };
        let expected_diff = m.gauge_connection * 0.5;
        for i in 0..MANIFOLD_DIM {
            assert!(
                ((plus[i] - transport_no_conn[i]) - expected_diff[i]).abs() < 1e-12,
                "field + transport_with_A − transport_without_A should equal A · dt at slot {i}"
            );
        }
    }

    /// The `Display` impl on `ManifoldError` is what the `GaugeSolver` (#1462)
    /// will surface through `anyhow!` — the messages must be human-readable.
    #[test]
    fn test_manifold_error_display_messages_are_descriptive() {
        let err = ManifoldError::NonFiniteMetric { row: 2, col: 3 };
        assert!(format!("{err}").contains("metric_tensor[2,3]"));
        assert!(format!("{err}").contains("NaN/inf"));
        assert!(format!("{}", ManifoldError::NonFiniteField).contains("scalar_field"));
        assert!(format!("{}", ManifoldError::NonFiniteConnection).contains("gauge_connection"));
    }
}
