//! Two-dimensional ground-conduction domain model (ASHRAE 140 §8 foundation slice).
//!
//! Issue #4270 lays the groundwork for the §8 "Ground-Coupled Slab-on-Grade
//! Analytical Verification" cases. This module is a **standalone** 2-D
//! (x–z) ground-conduction domain:
//!
//! * a slab-on-grade floor of finite width sitting on a soil domain,
//! * a two-dimensional material field (the slab layer exists only under the
//!   floor footprint; everything else is soil),
//! * prescribed surface conditions on the top surface (Dirichlet, Robin film,
//!   or a linear temperature ramp), on the deep-ground boundary and on the
//!   lateral far-field boundaries,
//! * a cell-centred finite-volume discretisation on a piecewise-uniform /
//!   geometrically graded grid whose segment boundaries are aligned with the
//!   floor footprint edges, the slab/soil interface and the top-surface
//!   segment boundaries,
//! * a steady-state solve by Jacobi-preconditioned conjugate gradients.
//!
//! # Scope and hard boundaries
//!
//! This module is deliberately **not** wired into the Section 7 floor
//! boundary, the 5R1C zone model, the HVAC loop or the main simulation
//! loop: §8 verifies the ground domain standalone before it is wired back.
//! Nothing here changes an existing physics result.
//!
//! The steady-state analytical verification cases live in
//! `tests/ashrae_140/ground_slab_steady_state.rs`; the transient / periodic
//! §8 cases remain open on #4270.
//!
//! # Discretisation
//!
//! Cell-centred finite volumes. For two adjacent cells `i` and `i+1`
//! sharing a face of area `A` (the cell height times the unit out-of-plane
//! length), the face conductance uses the series of the two half-cell
//! resistances
//!
//! ```text
//! G_face = A / ( (dx_i / 2) / k_i + (dx_{i+1} / 2) / k_{i+1} )
//! ```
//!
//! which reduces to the harmonic mean for equal conductivities and is exact
//! at a material interface when the interface sits on a cell face (this grid
//! aligns the slab/soil interface with a face by construction).
//!
//! Boundary faces use the same half-cell treatment:
//!
//! * Dirichlet `T_b`: `G = A / ((d / 2) / k)`, boundary temperature `T_b`.
//! * Robin film `h` with ambient `T_inf`: the film resistance and the
//!   half-cell conduction resistance are in series,
//!   `G = 1 / ( 1/(h A) + (d / 2) / (k A) )`, with `T_b = T_inf`.
//! * A linear ramp over a cell is represented by the cell-average of the
//!   ramp (exact for a linear profile) as a Dirichlet face temperature.
//! * Adiabatic: no face term.
//!
//! The steady system is symmetric positive definite (as soon as one
//! Dirichlet anchor exists) and is solved with Jacobi-preconditioned CG.
//!
//! # Sign convention
//!
//! All reported heat flows are **positive into the ground domain** and are
//! expressed per metre of out-of-plane length (the cross-section is one
//! metre deep out of plane). At steady state their sum is zero to
//! round-off; [`BoundaryHeatFlows::net`] exposes the closure residual so a
//! caller can assert the energy balance.

use std::fmt;

/// Out-of-plane depth represented by the 2-D cross-section [m].
pub const UNIT_OUT_OF_PLANE_LENGTH_M: f64 = 1.0;

/// Errors raised by the ground-conduction solver.
#[derive(Debug, Clone, PartialEq)]
pub enum GroundError {
    /// A geometry, material or boundary parameter is invalid.
    InvalidConfiguration(String),
    /// The system has no Dirichlet anchor, so it is singular.
    SingularSystem(String),
    /// The conjugate-gradient iteration did not reach the requested residual.
    NotConverged {
        /// Iterations performed.
        iterations: usize,
        /// Final relative residual `||A x - b|| / ||b||`.
        relative_residual: f64,
    },
}

impl fmt::Display for GroundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GroundError::InvalidConfiguration(msg) => {
                write!(f, "invalid ground configuration: {msg}")
            }
            GroundError::SingularSystem(msg) => write!(f, "singular ground system: {msg}"),
            GroundError::NotConverged {
                iterations,
                relative_residual,
            } => write!(
                f,
                "ground solve did not converge in {iterations} iterations \
                 (relative residual {relative_residual:.3e})"
            ),
        }
    }
}

impl std::error::Error for GroundError {}

/// A prescribed boundary condition on one surface of the domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SurfaceCondition {
    /// Fixed surface temperature [°C].
    Dirichlet(f64),
    /// Convective surface film towards an ambient temperature.
    Robin {
        /// Surface film coefficient [W/(m²·K)].
        film_coefficient_w_m2k: f64,
        /// Ambient (air-side) temperature the film sees [°C].
        ambient_temperature_c: f64,
    },
    /// No heat flux through the surface.
    Adiabatic,
}

/// The condition prescribed on one segment of the top (surface) boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TopCondition {
    /// Convective surface film towards an ambient temperature.
    Robin {
        /// Surface film coefficient [W/(m²·K)].
        film_coefficient_w_m2k: f64,
        /// Ambient (air-side) temperature the film sees [°C].
        ambient_temperature_c: f64,
    },
    /// Fixed surface temperature [°C].
    Dirichlet(f64),
    /// Linear temperature ramp from `from_c` at the segment start to `to_c`
    /// at the segment end (the ASHRAE 140 §8 GC10a perimeter boundary).
    Ramp {
        /// Temperature at the segment start [°C].
        from_c: f64,
        /// Temperature at the segment end [°C].
        to_c: f64,
    },
}

impl TopCondition {
    /// Temperature of a linear ramp at position `x` [°C].
    ///
    /// `x_start_m` / `x_end_m` bound the ramp segment. Non-ramp conditions
    /// return `None`.
    pub fn ramp_temperature(&self, x_start_m: f64, x_end_m: f64, x: f64) -> Option<f64> {
        match self {
            TopCondition::Ramp { from_c, to_c } => {
                let span = x_end_m - x_start_m;
                if span <= 0.0 {
                    return Some(*from_c);
                }
                let f = ((x - x_start_m) / span).clamp(0.0, 1.0);
                Some(from_c + (to_c - from_c) * f)
            }
            _ => None,
        }
    }
}

/// One segment of the top (surface) boundary of the ground domain.
#[derive(Debug, Clone, PartialEq)]
pub struct TopSegment {
    /// Start of the segment along the lateral axis [m].
    pub x_start_m: f64,
    /// End of the segment along the lateral axis [m].
    pub x_end_m: f64,
    /// Prescribed condition over the segment.
    pub condition: TopCondition,
}

/// One segment of a grid axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GridSegment {
    /// Uniform subdivision of `[start_m, end_m]` into `cells` cells.
    Uniform {
        /// Segment start [m].
        start_m: f64,
        /// Segment end [m].
        end_m: f64,
        /// Number of cells.
        cells: usize,
    },
    /// Geometrically graded subdivision of `[start_m, end_m]`: the first
    /// cell is `first_cell_m` long and each subsequent cell is `growth`
    /// times the previous one, capped at `max_cell_m`.
    Geometric {
        /// Segment start [m].
        start_m: f64,
        /// Segment end [m].
        end_m: f64,
        /// Length of the first cell [m].
        first_cell_m: f64,
        /// Growth factor between consecutive cells.
        growth: f64,
        /// Upper bound on the cell length [m].
        max_cell_m: f64,
    },
}

impl GridSegment {
    /// Start of the segment [m].
    pub fn start_m(&self) -> f64 {
        match self {
            GridSegment::Uniform { start_m, .. } | GridSegment::Geometric { start_m, .. } => {
                *start_m
            }
        }
    }

    /// End of the segment [m].
    pub fn end_m(&self) -> f64 {
        match self {
            GridSegment::Uniform { end_m, .. } | GridSegment::Geometric { end_m, .. } => *end_m,
        }
    }
}

/// A piecewise-uniform grid axis: cell edges, centres and widths.
#[derive(Debug, Clone, PartialEq)]
pub struct GridAxis {
    /// Cell edge coordinates, `edges.len() == widths.len() + 1`.
    pub edges: Vec<f64>,
    /// Cell centre coordinates.
    pub centers: Vec<f64>,
    /// Cell widths.
    pub widths: Vec<f64>,
}

impl GridAxis {
    /// Build an axis from a list of segments.
    pub fn from_segments(segments: &[GridSegment]) -> Result<GridAxis, GroundError> {
        let mut edges: Vec<f64> = Vec::new();
        let mut widths: Vec<f64> = Vec::new();
        let mut position: f64 = 0.0;
        for segment in segments {
            let start = segment.start_m();
            let end = segment.end_m();
            if end <= start {
                return Err(GroundError::InvalidConfiguration(format!(
                    "grid segment [{start}, {end}] is not increasing"
                )));
            }
            if (start - position).abs() > 1.0e-9 {
                return Err(GroundError::InvalidConfiguration(format!(
                    "grid segments must be contiguous: expected start {position}, got {start}"
                )));
            }
            match *segment {
                GridSegment::Uniform { cells, .. } => {
                    if cells == 0 {
                        return Err(GroundError::InvalidConfiguration(
                            "grid segment must have at least one cell".to_string(),
                        ));
                    }
                    let step = (end - start) / (cells as f64);
                    for k in 0..cells {
                        edges.push(start + step * (k as f64));
                        widths.push(if k + 1 == cells {
                            end - (start + step * (k as f64))
                        } else {
                            step
                        });
                    }
                }
                GridSegment::Geometric {
                    first_cell_m,
                    growth,
                    max_cell_m,
                    ..
                } => {
                    if first_cell_m <= 0.0 || growth < 1.0 || max_cell_m <= 0.0 {
                        return Err(GroundError::InvalidConfiguration(
                            "geometric grid segment needs a positive first cell, growth >= 1 \
                             and a positive maximum cell"
                                .to_string(),
                        ));
                    }
                    let mut z = start;
                    let mut h = first_cell_m;
                    while z < end - 1.0e-12 {
                        let h_used = h.min(max_cell_m).min(end - z);
                        edges.push(z);
                        widths.push(h_used);
                        z += h_used;
                        h = (h * growth).min(max_cell_m);
                    }
                }
            }
            position = end;
        }
        edges.push(position);
        let centers = (0..widths.len())
            .map(|i| (edges[i] + edges[i + 1]) / 2.0)
            .collect::<Vec<f64>>();
        Ok(GridAxis {
            edges,
            centers,
            widths,
        })
    }

    /// Number of cells on the axis.
    pub fn len(&self) -> usize {
        self.widths.len()
    }

    /// Whether the axis has no cells.
    pub fn is_empty(&self) -> bool {
        self.widths.is_empty()
    }
}

/// Geometry, materials, boundary conditions and discretisation of the 2-D
/// ground domain.
///
/// The cross-section is `x ∈ [0, domain_length_m]` and
/// `z ∈ [0, domain_depth_m]`, with `z = 0` at the top surface. The slab
/// occupies `x ∈ [slab_x0_m, slab_x1_m]` and `z ∈ [0, slab_thickness_m]`.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundDomainConfig {
    /// Lateral extent of the ground domain [m].
    pub domain_length_m: f64,
    /// Depth of the ground domain, measured from the top surface [m].
    pub domain_depth_m: f64,
    /// Left edge of the slab footprint [m].
    pub slab_x0_m: f64,
    /// Right edge of the slab footprint [m].
    pub slab_x1_m: f64,
    /// Thickness of the slab layer [m].
    pub slab_thickness_m: f64,
    /// Slab conductivity [W/(m·K)].
    pub slab_conductivity_w_mk: f64,
    /// Soil conductivity [W/(m·K)].
    pub ground_conductivity_w_mk: f64,
    /// Lateral grid segmentation (must tile `[0, domain_length_m]`).
    pub lateral_segments: Vec<GridSegment>,
    /// Vertical grid segmentation (must tile `[0, domain_depth_m]`).
    pub vertical_segments: Vec<GridSegment>,
    /// Top-surface boundary segmentation (must tile `[0, domain_length_m]`).
    pub top_segments: Vec<TopSegment>,
    /// Condition on the deep-ground boundary (`z = domain_depth_m`).
    pub deep_ground: SurfaceCondition,
    /// Condition on the lateral far-field boundaries (`x = 0` and `x = L`).
    pub lateral_boundary: SurfaceCondition,
    /// Maximum conjugate-gradient iterations.
    pub max_iterations: usize,
    /// Target relative residual for the steady solve.
    pub tolerance: f64,
}

impl GroundDomainConfig {
    /// Validate the configuration; returns the first problem found.
    pub fn validate(&self) -> Result<(), GroundError> {
        let positive = [
            ("domain_length_m", self.domain_length_m),
            ("domain_depth_m", self.domain_depth_m),
            ("slab_conductivity_w_mk", self.slab_conductivity_w_mk),
            ("ground_conductivity_w_mk", self.ground_conductivity_w_mk),
        ];
        for (name, value) in positive {
            if !(value.is_finite() && value > 0.0) {
                return Err(GroundError::InvalidConfiguration(format!(
                    "{name} must be finite and positive, got {value}"
                )));
            }
        }
        if !(self.slab_x0_m.is_finite() && self.slab_x1_m.is_finite()) {
            return Err(GroundError::InvalidConfiguration(
                "slab footprint edges must be finite".to_string(),
            ));
        }
        if self.slab_x0_m < 0.0 || self.slab_x1_m > self.domain_length_m {
            return Err(GroundError::InvalidConfiguration(format!(
                "slab footprint [{}, {}] must lie inside [0, {}]",
                self.slab_x0_m, self.slab_x1_m, self.domain_length_m
            )));
        }
        if self.slab_x1_m <= self.slab_x0_m {
            return Err(GroundError::InvalidConfiguration(format!(
                "slab footprint must have positive width, got [{}, {}]",
                self.slab_x0_m, self.slab_x1_m
            )));
        }
        if !(self.slab_thickness_m > 0.0 && self.slab_thickness_m < self.domain_depth_m) {
            return Err(GroundError::InvalidConfiguration(format!(
                "slab thickness {} must be positive and below the domain depth {}",
                self.slab_thickness_m, self.domain_depth_m
            )));
        }
        if self.max_iterations == 0 {
            return Err(GroundError::InvalidConfiguration(
                "max_iterations must be at least 1".to_string(),
            ));
        }
        if !(self.tolerance.is_finite() && self.tolerance > 0.0) {
            return Err(GroundError::InvalidConfiguration(
                "tolerance must be finite and positive".to_string(),
            ));
        }

        let lateral = GridAxis::from_segments(&self.lateral_segments)?;
        let vertical = GridAxis::from_segments(&self.vertical_segments)?;
        if (lateral.edges[0]).abs() > 1.0e-9
            || (lateral.edges[lateral.len()] - self.domain_length_m).abs() > 1.0e-9
        {
            return Err(GroundError::InvalidConfiguration(
                "lateral segments must tile [0, domain_length_m]".to_string(),
            ));
        }
        if (vertical.edges[0]).abs() > 1.0e-9
            || (vertical.edges[vertical.len()] - self.domain_depth_m).abs() > 1.0e-9
        {
            return Err(GroundError::InvalidConfiguration(
                "vertical segments must tile [0, domain_depth_m]".to_string(),
            ));
        }

        // The footprint edges, the slab/soil interface and the top-segment
        // boundaries must sit on cell faces so that no cell straddles a
        // material or boundary discontinuity.
        for (name, position) in [("slab_x0_m", self.slab_x0_m), ("slab_x1_m", self.slab_x1_m)] {
            if !is_cell_edge(&lateral.edges, position) {
                return Err(GroundError::InvalidConfiguration(format!(
                    "{name} = {position} must be a lateral cell edge"
                )));
            }
        }
        if !is_cell_edge(&vertical.edges, self.slab_thickness_m) {
            return Err(GroundError::InvalidConfiguration(format!(
                "slab_thickness_m = {} must be a vertical cell edge",
                self.slab_thickness_m
            )));
        }
        if self.top_segments.is_empty() {
            return Err(GroundError::InvalidConfiguration(
                "at least one top segment is required".to_string(),
            ));
        }
        for (index, segment) in self.top_segments.iter().enumerate() {
            if segment.x_end_m <= segment.x_start_m {
                return Err(GroundError::InvalidConfiguration(format!(
                    "top segment {index} must have positive width"
                )));
            }
            if !is_cell_edge(&lateral.edges, segment.x_start_m)
                || !is_cell_edge(&lateral.edges, segment.x_end_m)
            {
                return Err(GroundError::InvalidConfiguration(format!(
                    "top segment {index} boundaries must be lateral cell edges"
                )));
            }
            if let TopCondition::Robin {
                film_coefficient_w_m2k: h,
                ..
            } = segment.condition
            {
                if !(h.is_finite() && h > 0.0) {
                    return Err(GroundError::InvalidConfiguration(format!(
                        "top segment {index} needs a positive film coefficient"
                    )));
                }
            }
        }
        for (index, segment) in self.top_segments.iter().enumerate() {
            let expected_start = if index == 0 {
                0.0
            } else {
                self.top_segments[index - 1].x_end_m
            };
            if (segment.x_start_m - expected_start).abs() > 1.0e-9 {
                return Err(GroundError::InvalidConfiguration(format!(
                    "top segments must be contiguous: segment {index} starts at {} but \
                     the previous one ends at {expected_start}",
                    segment.x_start_m
                )));
            }
        }
        let last = self.top_segments[self.top_segments.len() - 1].x_end_m;
        if (last - self.domain_length_m).abs() > 1.0e-9 {
            return Err(GroundError::InvalidConfiguration(
                "top segments must tile [0, domain_length_m]".to_string(),
            ));
        }
        Ok(())
    }
}

/// Whether `position` coincides with one of the cell edges.
fn is_cell_edge(edges: &[f64], position: f64) -> bool {
    edges.iter().any(|edge| (edge - position).abs() < 1.0e-9)
}

/// The discretised ground grid.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundGrid {
    /// Lateral axis.
    pub x: GridAxis,
    /// Vertical axis (depth, `z = 0` at the top surface).
    pub z: GridAxis,
    /// Left edge of the slab footprint [m].
    pub slab_x0_m: f64,
    /// Right edge of the slab footprint [m].
    pub slab_x1_m: f64,
    /// Thickness of the slab layer [m].
    pub slab_thickness_m: f64,
}

impl GroundGrid {
    /// Build the grid for a configuration.
    pub fn build(config: &GroundDomainConfig) -> Result<GroundGrid, GroundError> {
        Ok(GroundGrid {
            x: GridAxis::from_segments(&config.lateral_segments)?,
            z: GridAxis::from_segments(&config.vertical_segments)?,
            slab_x0_m: config.slab_x0_m,
            slab_x1_m: config.slab_x1_m,
            slab_thickness_m: config.slab_thickness_m,
        })
    }

    /// Conductivity of the cell at column `i`, row `j` [W/(m·K)].
    ///
    /// The slab material exists only under the footprint and above the
    /// slab/soil interface; both are cell faces by construction.
    pub fn cell_conductivity(&self, config: &GroundDomainConfig, i: usize, j: usize) -> f64 {
        let x = self.x.centers[i];
        let z = self.z.centers[j];
        let under_slab = x >= self.slab_x0_m && x <= self.slab_x1_m;
        if under_slab && z < self.slab_thickness_m {
            config.slab_conductivity_w_mk
        } else {
            config.ground_conductivity_w_mk
        }
    }
}

/// The steady-state solution of a ground domain.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundSolution {
    /// Cell-centre temperatures [°C], indexed as `j * nx + i`.
    pub temperature_c: Vec<f64>,
    /// The grid the solution was computed on.
    pub grid: GroundGrid,
    /// Conjugate-gradient iterations performed.
    pub iterations: usize,
    /// Final relative residual `||A x - b|| / ||b||`.
    pub relative_residual: f64,
}

impl GroundSolution {
    /// Cell-centre temperature at column `i`, row `j` [°C].
    pub fn cell_temperature(&self, i: usize, j: usize) -> f64 {
        self.temperature_c[j * self.grid.x.len() + i]
    }

    /// Bilinearly interpolated temperature at `(x, z)` [°C].
    ///
    /// Coordinates outside the domain are clamped to the domain bounds.
    pub fn temperature_at(&self, x: f64, z: f64) -> f64 {
        let xc = &self.grid.x.centers;
        let zc = &self.grid.z.centers;
        let (i0, i1, fx) = bracket(xc, x);
        let (j0, j1, fz) = bracket(zc, z);
        let t00 = self.cell_temperature(i0, j0);
        let t10 = self.cell_temperature(i1, j0);
        let t01 = self.cell_temperature(i0, j1);
        let t11 = self.cell_temperature(i1, j1);
        let top = t00 + (t10 - t00) * fx;
        let bottom = t01 + (t11 - t01) * fx;
        top + (bottom - top) * fz
    }

    /// Heat flows through the domain boundaries [W per m of out-of-plane length].
    pub fn heat_flows(&self, config: &GroundDomainConfig) -> BoundaryHeatFlows {
        compute_heat_flows(config, &self.grid, &self.temperature_c)
    }
}

/// Index of the cell interval containing `value` in a sorted coordinate list.
fn bracket(coords: &[f64], value: f64) -> (usize, usize, f64) {
    let n = coords.len();
    if n == 1 {
        return (0, 0, 0.0);
    }
    let clamped = value.clamp(coords[0], coords[n - 1]);
    let mut hi = 1;
    while hi < n - 1 && coords[hi] < clamped {
        hi += 1;
    }
    let lo = hi - 1;
    let span = coords[hi] - coords[lo];
    let f = if span > 0.0 {
        (clamped - coords[lo]) / span
    } else {
        0.0
    };
    (lo, hi, f)
}

/// Heat flows through the boundaries of the ground domain.
///
/// All values are [W per m of out-of-plane length] and are positive when heat
/// flows **into** the domain.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BoundaryHeatFlows {
    /// Heat entering through the slab top surface (the floor footprint).
    pub slab_surface_w: f64,
    /// Heat entering through the exterior ground surface (outside the footprint).
    pub ground_surface_w: f64,
    /// Heat entering through the deep-ground boundary.
    pub deep_ground_w: f64,
    /// Heat entering through the lateral far-field boundaries (both sides).
    pub lateral_w: f64,
    /// Heat crossing the slab/soil interface downwards (positive downwards).
    pub slab_underside_w: f64,
}

impl BoundaryHeatFlows {
    /// Net heat into the domain [W per m]. Zero at steady state.
    pub fn net(&self) -> f64 {
        self.slab_surface_w + self.ground_surface_w + self.deep_ground_w + self.lateral_w
    }
}

/// Face term of a boundary: series conductance and the temperature it sees.
struct BoundaryFace {
    /// Series conductance of the face [W/K per m].
    conductance: f64,
    /// Temperature at the outer end of the face [°C].
    temperature: f64,
}

/// Compute the series conductance of a boundary face.
///
/// `half_resistance` is the half-cell conduction resistance
/// `(d / 2) / (k A)`; for a Dirichlet boundary the face conductance is the
/// reciprocal of that resistance, for a Robin film the film resistance
/// `1 / (h A)` is added in series.
fn boundary_face(
    condition: SurfaceCondition,
    area: f64,
    half_resistance: f64,
) -> Option<BoundaryFace> {
    match condition {
        SurfaceCondition::Adiabatic => None,
        SurfaceCondition::Dirichlet(t) => {
            if half_resistance <= 0.0 {
                return None;
            }
            Some(BoundaryFace {
                conductance: 1.0 / half_resistance,
                temperature: t,
            })
        }
        SurfaceCondition::Robin {
            film_coefficient_w_m2k: h,
            ambient_temperature_c: t_inf,
        } => {
            if h <= 0.0 || area <= 0.0 {
                return None;
            }
            let film_resistance = 1.0 / (h * area);
            let total = film_resistance + half_resistance;
            if total <= 0.0 {
                return None;
            }
            Some(BoundaryFace {
                conductance: 1.0 / total,
                temperature: t_inf,
            })
        }
    }
}

/// The 2-D ground-conduction solver.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundSolver {
    /// Domain configuration.
    pub config: GroundDomainConfig,
}

impl GroundSolver {
    /// Create a solver for a configuration.
    pub fn new(config: GroundDomainConfig) -> Result<GroundSolver, GroundError> {
        config.validate()?;
        Ok(GroundSolver { config })
    }

    /// Solve the steady-state conduction problem.
    pub fn solve(&self) -> Result<GroundSolution, GroundError> {
        let config = &self.config;
        let grid = GroundGrid::build(config)?;
        let nx = grid.x.len();
        let nz = grid.z.len();
        let n = nx * nz;

        // Per-cell assembly: diagonal, off-diagonal couplings and the RHS.
        let mut diagonal = vec![0.0_f64; n];
        let mut rhs = vec![0.0_f64; n];
        let mut couplings: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

        let k_of = |i: usize, j: usize| grid.cell_conductivity(config, i, j);

        // Interior faces (x direction).
        for j in 0..nz {
            for i in 0..nx.saturating_sub(1) {
                let half_i = (grid.x.widths[i] / 2.0) / k_of(i, j);
                let half_ip = (grid.x.widths[i + 1] / 2.0) / k_of(i + 1, j);
                let area = grid.z.widths[j] * UNIT_OUT_OF_PLANE_LENGTH_M;
                let g = area / (half_i + half_ip);
                let a = j * nx + i;
                let b = j * nx + i + 1;
                diagonal[a] += g;
                diagonal[b] += g;
                couplings[a].push((b, g));
                couplings[b].push((a, g));
            }
        }

        // Interior faces (z direction).
        for j in 0..nz.saturating_sub(1) {
            for i in 0..nx {
                let half_j = (grid.z.widths[j] / 2.0) / k_of(i, j);
                let half_jp = (grid.z.widths[j + 1] / 2.0) / k_of(i, j + 1);
                let area = grid.x.widths[i] * UNIT_OUT_OF_PLANE_LENGTH_M;
                let g = area / (half_j + half_jp);
                let a = j * nx + i;
                let b = (j + 1) * nx + i;
                diagonal[a] += g;
                diagonal[b] += g;
                couplings[a].push((b, g));
                couplings[b].push((a, g));
            }
        }

        // Top boundary: the prescribed condition of the top segment that
        // contains the cell.
        for i in 0..nx {
            let k = k_of(i, 0);
            let half = (grid.z.widths[0] / 2.0) / (k * grid.x.widths[i]);
            let area = grid.x.widths[i] * UNIT_OUT_OF_PLANE_LENGTH_M;
            if let Some(face) = top_boundary_face(config, &grid, i, area, half) {
                let idx = i;
                diagonal[idx] += face.conductance;
                rhs[idx] += face.conductance * face.temperature;
            }
        }

        // Deep-ground boundary.
        {
            let j = nz - 1;
            for i in 0..nx {
                let k = k_of(i, j);
                let half = (grid.z.widths[j] / 2.0) / (k * grid.x.widths[i]);
                let area = grid.x.widths[i] * UNIT_OUT_OF_PLANE_LENGTH_M;
                if let Some(face) = boundary_face(config.deep_ground, area, half) {
                    let idx = j * nx + i;
                    diagonal[idx] += face.conductance;
                    rhs[idx] += face.conductance * face.temperature;
                }
            }
        }

        // Lateral far-field boundaries.
        for j in 0..nz {
            for i in [0usize, nx - 1] {
                let k = k_of(i, j);
                let half = (grid.x.widths[i] / 2.0) / (k * grid.z.widths[j]);
                let area = grid.z.widths[j] * UNIT_OUT_OF_PLANE_LENGTH_M;
                if let Some(face) = boundary_face(config.lateral_boundary, area, half) {
                    let idx = j * nx + i;
                    diagonal[idx] += face.conductance;
                    rhs[idx] += face.conductance * face.temperature;
                }
            }
        }

        if diagonal.iter().all(|d| *d == 0.0) {
            return Err(GroundError::SingularSystem(
                "no boundary face carries heat: the steady system is singular".to_string(),
            ));
        }

        let (temperature_c, iterations, relative_residual) = conjugate_gradient(
            &diagonal,
            &couplings,
            &rhs,
            config.tolerance,
            config.max_iterations,
        )?;

        Ok(GroundSolution {
            temperature_c,
            grid,
            iterations,
            relative_residual,
        })
    }
}

/// Boundary face of the top surface for the cell at column `i`.
fn top_boundary_face(
    config: &GroundDomainConfig,
    grid: &GroundGrid,
    i: usize,
    area: f64,
    half_resistance: f64,
) -> Option<BoundaryFace> {
    let x_center = grid.x.centers[i];
    let segment = config.top_segments.iter().find(|segment| {
        x_center >= segment.x_start_m - 1.0e-9 && x_center <= segment.x_end_m + 1.0e-9
    })?;
    match segment.condition {
        TopCondition::Dirichlet(t) => {
            boundary_face(SurfaceCondition::Dirichlet(t), area, half_resistance)
        }
        TopCondition::Robin {
            film_coefficient_w_m2k,
            ambient_temperature_c,
        } => boundary_face(
            SurfaceCondition::Robin {
                film_coefficient_w_m2k,
                ambient_temperature_c,
            },
            area,
            half_resistance,
        ),
        TopCondition::Ramp { .. } => {
            // Cell-average of a linear ramp: exact for a linear profile.
            let t_left = segment
                .condition
                .ramp_temperature(segment.x_start_m, segment.x_end_m, grid.x.edges[i])
                .unwrap_or(0.0);
            let t_right = segment
                .condition
                .ramp_temperature(segment.x_start_m, segment.x_end_m, grid.x.edges[i + 1])
                .unwrap_or(0.0);
            boundary_face(
                SurfaceCondition::Dirichlet(0.5 * (t_left + t_right)),
                area,
                half_resistance,
            )
        }
    }
}

/// Jacobi-preconditioned conjugate gradients for a symmetric positive
/// definite system assembled as `diagonal` + off-diagonal `couplings`.
fn conjugate_gradient(
    diagonal: &[f64],
    couplings: &[Vec<(usize, f64)>],
    rhs: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<(Vec<f64>, usize, f64), GroundError> {
    let n = rhs.len();
    let mut x = vec![0.0_f64; n];
    let mut r = rhs.to_vec();
    let mut z = vec![0.0_f64; n];
    let mut p = vec![0.0_f64; n];
    let mut ap = vec![0.0_f64; n];
    let mut rho_old = 0.0_f64;

    let rhs_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    if rhs_norm == 0.0 {
        return Ok((x, 0, 0.0));
    }

    for iteration in 0..max_iterations {
        for i in 0..n {
            z[i] = r[i] / diagonal[i];
        }
        let rho: f64 = r.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
        if iteration == 0 {
            p.copy_from_slice(&z);
        } else {
            let beta = rho / rho_old;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
        }
        rho_old = rho;

        for i in 0..n {
            let mut acc = diagonal[i] * p[i];
            for (j, g) in &couplings[i] {
                acc -= g * p[*j];
            }
            ap[i] = acc;
        }
        let denom: f64 = p.iter().zip(ap.iter()).map(|(a, b)| a * b).sum();
        if denom == 0.0 {
            return Err(GroundError::SingularSystem(
                "conjugate-gradient breakdown (zero curvature)".to_string(),
            ));
        }
        let alpha = rho / denom;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let residual = r.iter().map(|v| v * v).sum::<f64>().sqrt() / rhs_norm;
        if residual <= tolerance {
            return Ok((x, iteration + 1, residual));
        }
    }
    let residual = r.iter().map(|v| v * v).sum::<f64>().sqrt() / rhs_norm;
    Err(GroundError::NotConverged {
        iterations: max_iterations,
        relative_residual: residual,
    })
}

/// Compute the boundary heat flows of a converged solution.
fn compute_heat_flows(
    config: &GroundDomainConfig,
    grid: &GroundGrid,
    temperature_c: &[f64],
) -> BoundaryHeatFlows {
    let nx = grid.x.len();
    let nz = grid.z.len();
    let k_of = |i: usize, j: usize| grid.cell_conductivity(config, i, j);
    let mut flows = BoundaryHeatFlows::default();

    // Top surface: slab footprint and exterior ground surface.
    for i in 0..nx {
        let k = k_of(i, 0);
        let half = (grid.z.widths[0] / 2.0) / (k * grid.x.widths[i]);
        let area = grid.x.widths[i] * UNIT_OUT_OF_PLANE_LENGTH_M;
        if let Some(face) = top_boundary_face(config, grid, i, area, half) {
            let q = face.conductance * (face.temperature - temperature_c[i]);
            let under_slab =
                grid.x.centers[i] >= grid.slab_x0_m && grid.x.centers[i] <= grid.slab_x1_m;
            if under_slab {
                flows.slab_surface_w += q;
            } else {
                flows.ground_surface_w += q;
            }
        }
    }

    // Deep-ground boundary.
    {
        let j = nz - 1;
        for i in 0..nx {
            let k = k_of(i, j);
            let half = (grid.z.widths[j] / 2.0) / (k * grid.x.widths[i]);
            let area = grid.x.widths[i] * UNIT_OUT_OF_PLANE_LENGTH_M;
            if let Some(face) = boundary_face(config.deep_ground, area, half) {
                flows.deep_ground_w +=
                    face.conductance * (face.temperature - temperature_c[j * nx + i]);
            }
        }
    }

    // Lateral boundaries.
    for j in 0..nz {
        for i in [0usize, nx - 1] {
            let k = k_of(i, j);
            let half = (grid.x.widths[i] / 2.0) / (k * grid.z.widths[j]);
            let area = grid.z.widths[j] * UNIT_OUT_OF_PLANE_LENGTH_M;
            if let Some(face) = boundary_face(config.lateral_boundary, area, half) {
                flows.lateral_w +=
                    face.conductance * (face.temperature - temperature_c[j * nx + i]);
            }
        }
    }

    // Slab/soil interface: downward heat crossing the last slab row into the
    // first soil row, summed over the slab footprint columns.
    if nz >= 2 {
        // The interface row is the last row whose centre is above the
        // slab thickness.
        let j_slab = (0..nz)
            .rev()
            .find(|j| grid.z.centers[*j] < grid.slab_thickness_m)
            .unwrap_or(0);
        let j_soil = j_slab + 1;
        if j_soil < nz {
            for i in 0..nx {
                let under_slab =
                    grid.x.centers[i] >= grid.slab_x0_m && grid.x.centers[i] <= grid.slab_x1_m;
                if !under_slab {
                    continue;
                }
                let half_a = (grid.z.widths[j_slab] / 2.0) / k_of(i, j_slab);
                let half_b = (grid.z.widths[j_soil] / 2.0) / k_of(i, j_soil);
                let area = grid.x.widths[i] * UNIT_OUT_OF_PLANE_LENGTH_M;
                let g = area / (half_a + half_b);
                let q = g * (temperature_c[j_slab * nx + i] - temperature_c[j_soil * nx + i]);
                flows.slab_underside_w += q;
            }
        }
    }

    flows
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One-dimensional configuration: the slab spans the whole domain width
    /// and the lateral boundaries are adiabatic, so the solution is exactly
    /// one-dimensional.
    fn one_dimensional_config() -> GroundDomainConfig {
        GroundDomainConfig {
            domain_length_m: 4.0,
            domain_depth_m: 4.0,
            slab_x0_m: 0.0,
            slab_x1_m: 4.0,
            slab_thickness_m: 0.2,
            slab_conductivity_w_mk: 1.4,
            ground_conductivity_w_mk: 1.9,
            lateral_segments: vec![GridSegment::Uniform {
                start_m: 0.0,
                end_m: 4.0,
                cells: 8,
            }],
            vertical_segments: vec![
                GridSegment::Uniform {
                    start_m: 0.0,
                    end_m: 0.2,
                    cells: 2,
                },
                GridSegment::Geometric {
                    start_m: 0.2,
                    end_m: 4.0,
                    first_cell_m: 0.05,
                    growth: 1.15,
                    max_cell_m: 0.4,
                },
            ],
            top_segments: vec![TopSegment {
                x_start_m: 0.0,
                x_end_m: 4.0,
                condition: TopCondition::Robin {
                    film_coefficient_w_m2k: 8.29,
                    ambient_temperature_c: 30.0,
                },
            }],
            deep_ground: SurfaceCondition::Dirichlet(10.0),
            lateral_boundary: SurfaceCondition::Adiabatic,
            max_iterations: 20_000,
            tolerance: 1.0e-12,
        }
    }

    #[test]
    fn one_dimensional_limit_matches_series_resistance() {
        let config = one_dimensional_config();
        let solver = GroundSolver::new(config.clone()).expect("valid config");
        let solution = solver.solve().expect("steady solve");

        // Closed form: q'' = (Ti - Tg) / (1/h + ds/ks + Dg/kg).
        let h = 8.29;
        let r_total = 1.0 / h + 0.2 / 1.4 + (4.0 - 0.2) / 1.9;
        let q_flux = (30.0 - 10.0) / r_total;
        let expected_total = q_flux * config.domain_length_m;

        let flows = solution.heat_flows(&config);
        assert!(
            (flows.slab_surface_w - expected_total).abs() / expected_total < 1.0e-9,
            "slab heat {} vs analytical {}",
            flows.slab_surface_w,
            expected_total
        );
        assert!(
            flows.net().abs() / expected_total < 1.0e-9,
            "energy balance residual {}",
            flows.net()
        );
    }

    #[test]
    fn one_dimensional_temperature_profile_matches_closed_form() {
        let config = one_dimensional_config();
        let solver = GroundSolver::new(config.clone()).expect("valid config");
        let solution = solver.solve().expect("steady solve");

        let h = 8.29;
        let r_total = 1.0 / h + 0.2 / 1.4 + (4.0 - 0.2) / 1.9;
        let q_flux = (30.0 - 10.0) / r_total;
        for (j, z_center) in solution.grid.z.centers.iter().enumerate() {
            let resistance = if *z_center < 0.2 {
                1.0 / h + z_center / 1.4
            } else {
                1.0 / h + 0.2 / 1.4 + (z_center - 0.2) / 1.9
            };
            let expected = 30.0 - q_flux * resistance;
            let actual = solution.cell_temperature(4, j);
            assert!(
                (actual - expected).abs() < 1.0e-8,
                "z = {z_center}: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn configuration_validation_rejects_bad_input() {
        let mut config = one_dimensional_config();
        config.slab_x1_m = config.slab_x0_m;
        assert!(matches!(
            GroundSolver::new(config),
            Err(GroundError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn configuration_validation_rejects_unaligned_footprint() {
        let mut config = one_dimensional_config();
        config.slab_x1_m = 3.7; // not a lateral cell edge
        assert!(matches!(
            GroundSolver::new(config),
            Err(GroundError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn energy_balance_closes_on_a_two_dimensional_domain() {
        let mut config = one_dimensional_config();
        config.lateral_segments = vec![
            GridSegment::Uniform {
                start_m: 0.0,
                end_m: 1.0,
                cells: 4,
            },
            GridSegment::Uniform {
                start_m: 1.0,
                end_m: 3.0,
                cells: 8,
            },
            GridSegment::Uniform {
                start_m: 3.0,
                end_m: 4.0,
                cells: 4,
            },
        ];
        config.slab_x0_m = 1.0;
        config.slab_x1_m = 3.0;
        config.lateral_boundary = SurfaceCondition::Dirichlet(10.0);
        config.top_segments = vec![
            TopSegment {
                x_start_m: 0.0,
                x_end_m: 1.0,
                condition: TopCondition::Robin {
                    film_coefficient_w_m2k: 18.3,
                    ambient_temperature_c: 10.0,
                },
            },
            TopSegment {
                x_start_m: 1.0,
                x_end_m: 3.0,
                condition: TopCondition::Robin {
                    film_coefficient_w_m2k: 8.29,
                    ambient_temperature_c: 30.0,
                },
            },
            TopSegment {
                x_start_m: 3.0,
                x_end_m: 4.0,
                condition: TopCondition::Robin {
                    film_coefficient_w_m2k: 18.3,
                    ambient_temperature_c: 10.0,
                },
            },
        ];
        let solver = GroundSolver::new(config.clone()).expect("valid config");
        let solution = solver.solve().expect("steady solve");
        let flows = solution.heat_flows(&config);
        let scale = flows.slab_surface_w.abs() + flows.ground_surface_w.abs();
        assert!(
            flows.net().abs() / scale < 1.0e-9,
            "energy balance residual {} (scale {scale})",
            flows.net()
        );
    }
}
