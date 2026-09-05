// =============================================================================
// SOLVER CONSOLIDATION (Issue #624)
// =============================================================================
//
// The physics module implements a unified solver architecture to prevent
// solver proliferation. Rather than creating separate solver types for each
// use case (multi-node CTF, per-surface, per-zone, etc.), we use a single
// HeatConductionSolver trait with runtime dispatch via SolverManager.
//
// Unified Architecture:
//   HeatConductionSolver trait (solver_trait.rs)
//       |
//   +----+----+----------+
//   |         |          |
//  5R1C      CTF        FD
// Solver  Solver    Solver
// Wrapper Wrapper  Wrapper
//   |         |          |
//   +----+----+----------+
//            |
//     SolverManager
//   (automatic selection + lifecycle)
//
// Benefits:
// - Single interface for all heat conduction calculations
// - Automatic method selection based on thermal mass
// - CTF-FD fallback for numerical robustness
// - No duplication of solver logic across use cases
//
// Additional solver types (multi_node_ctf, per_surface_ctf, etc.) were
// considered but NOT implemented because:
// 1. The existing 5R1C/CTF/FD triad covers all ASHRAE 140 cases
// 2. Multi-node adds complexity without validation benefit
// 3. Per-surface models add overhead without accuracy gain
//
// Future solvers should be added ONLY if a validation case demonstrates
// that the existing solvers cannot meet accuracy requirements.

pub mod bdf_engine;
pub mod constants;
pub mod continuous;
pub mod exterior_convection;

pub mod cta;
pub mod ctf_coefficients;
pub mod ctf_solver;
pub mod ctf_solver_wrapper;
pub mod ctf_zone_coupling;
pub mod fd_discretization;
pub mod fd_solver;
pub mod fd_solver_wrapper;
pub mod fd_surface_balance;
pub mod ffd_solver;
pub mod five_r1c_solver;
// Algebraic-FP helper layer (issue #3322): cfg-routed float ops that are
// plain IEEE 754 under default features and std `algebraic_*` methods under
// `--features fast-math`. No kernel consumes it yet (#3324/#3325 do that).
pub mod fp_algebraic;
pub mod gauge_solver;
pub mod gauge_zone_solver;
pub mod geometry_tensor;
pub mod method_selector;
pub mod nd_array;
pub mod state_space_ctf;
pub mod thermal_mass;
pub mod zero_copy_matrix;

pub mod multi_node_solver;
pub mod nine_r4c_nodal_trace;
// Issue #3338 — opt-in SIMD/cache-blocked dispatch layer for the
// solar/radiation accumulation loops (`perez_diffuse_tilted`,
// `surface_radiative_exchange`, …). Off by default; reaches runtime
// dispatch only under `--features simd-kernels`. Bit-identical under
// the default feature (every helper here resolves to a passthrough).
pub mod simd_kernels;
pub mod solver_manager;
pub mod solver_registry;
pub mod solver_trait;
pub mod units;
pub mod wall_properties;
pub mod wall_spec;
