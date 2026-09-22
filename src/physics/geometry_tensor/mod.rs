//! Geometry Tensor Module — public re-export shim
//!
//! Decomposed from the original `geometry_tensor.rs` into three sub-modules:
//! - [`geometry`] — CTA geometry tensors (`GeometryTensor`, `WallData`)
//! - [`manifold`] — gauge-theory thermal manifold (`ThermalManifold`, `ManifoldIndex`)
//! - [`qubo`] — QUBO encoding types (`QuboEncoding`, `QuboMatrix`)
//!
//! The original file-level doc-comment is preserved in [`manifold`].

pub use fluxion_core::zone_count_policy::{ZoneCountPolicy, ZoneCountTier, MAX_ZONES};
pub use nalgebra::{Matrix4, Vector4};

pub mod geometry;
pub mod manifold;
pub mod qubo;

// Re-export all public types at the module level so `super::*` in tests
// resolves all types from a single `use super::*` in mod.rs.
pub use geometry::{
    GeometryTensor, WallData, ADJACENCY_MATRIX_DIMS, MAX_WALLS, WALL_MATRIX_DIMS,
    WINDOW_MATRIX_DIMS, ZONE_COORDS_DIMS, ZONE_PROPERTIES_DIMS,
};
pub use manifold::QuboMatrix;
pub use manifold::{ManifoldError, ManifoldIndex, ThermalManifold, MANIFOLD_DIM};
pub use qubo::{QuboEncoding, QuboTranslateError};

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
        let mut manifold =
            ThermalManifold::from_5r1c_parameters(t_air_0, t_mass_0, r_eq, c_air, c_mass);
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
            assert_eq!(manifold.scalar_field[ManifoldIndex::Roof as usize], 0.0);
            assert_eq!(manifold.scalar_field[ManifoldIndex::Floor as usize], 0.0);
        }
    }

    /// 9R4C embedding populates the full 4-D dissipative operator and writes
    /// the temperatures into the matching slots.
    #[test]
    fn test_from_9r4c_layout() {
        let temperatures = [21.0, 19.0, 22.0, 18.0];
        let capacitances = [10_000.0, 50_000.0, 30_000.0, 80_000.0];
        let r_tr = [120.0, 80.0, 200.0]; // g_tr per surface

        let m = ThermalManifold::from_9r4c_parameters(temperatures, capacitances, r_tr, None);

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

        assert_eq!(
            m.scalar_field, snapshot_field,
            "parallel_transport must not mutate scalar_field"
        );
        assert_eq!(
            m.gauge_connection, snapshot_conn,
            "parallel_transport must not mutate gauge_connection"
        );
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
        for axis in [
            ManifoldIndex::Wall,
            ManifoldIndex::Roof,
            ManifoldIndex::Floor,
        ] {
            assert_eq!(
                t_new[axis as usize], 0.0,
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

    // -------------------------------------------------------------------------
    // Christoffel-symbol tests (Issue #1600 — Phase 1b)
    // -------------------------------------------------------------------------

    /// Christoffel symbols vanish for the 5R1C metric — the embedded metric is
    /// constant (R and C values do not vary), so ∂_j g_{lk} = 0 and the
    /// Levi-Civita connection coefficients are identically zero.
    #[test]
    fn test_christoffel_symbols_zero_for_5r1c() {
        let m = ThermalManifold::from_5r1c_parameters(20.0, 21.0, 0.10, 10_000.0, 50_000.0);
        let gamma = m.compute_christoffel_symbols();

        let tol = 1e-12;
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                for k in 0..MANIFOLD_DIM {
                    let gamma_ijk = gamma[(i, j)][(j, k)];
                    assert!(
                        gamma_ijk.abs() <= tol,
                        "Christoffel symbol Γ[{},{},{}] = {:.3e} exceeds tolerance {} for 5R1C",
                        i,
                        j,
                        k,
                        gamma_ijk,
                        tol
                    );
                }
            }
        }
    }

    /// Christoffel symbols vanish for the 9R4C metric — same reasoning as the
    /// 5R1C case: the conductance matrix is constant during transport.
    #[test]
    fn test_christoffel_symbols_zero_for_9r4c() {
        let temperatures = [21.0, 19.0, 22.0, 18.0];
        let capacitances = [10_000.0, 50_000.0, 30_000.0, 80_000.0];
        let r_tr = [120.0, 80.0, 200.0];

        let m = ThermalManifold::from_9r4c_parameters(temperatures, capacitances, r_tr, None);
        let gamma = m.compute_christoffel_symbols();

        let tol = 1e-12;
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                for k in 0..MANIFOLD_DIM {
                    let gamma_ijk = gamma[(i, j)][(j, k)];
                    assert!(
                        gamma_ijk.abs() <= tol,
                        "Christoffel symbol Γ[{},{},{}] = {:.3e} exceeds tolerance {} for 9R4C",
                        i,
                        j,
                        k,
                        gamma_ijk,
                        tol
                    );
                }
            }
        }
    }

    /// Verifies that all Christoffel symbols are ≤ 1e-30 for the constant-metric
    /// thermal manifold (both 5R1C and 9R4C scenes). This tight tolerance confirms
    /// that the metric is truly constant — no residual curvature from numerical noise.
    #[test]
    fn test_christoffel_symbols_constant_metric_zero() {
        let tol = 1e-30;

        let m_5r1c = ThermalManifold::from_5r1c_parameters(20.0, 21.0, 0.10, 10_000.0, 50_000.0);
        let gamma_5r1c = m_5r1c.compute_christoffel_symbols();
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                for k in 0..MANIFOLD_DIM {
                    let gamma_ijk = gamma_5r1c[(i, j)][(j, k)];
                    assert!(
                        gamma_ijk.abs() <= tol,
                        "5R1C Christoffel symbol Γ[{},{},{}] = {:.3e} exceeds {}",
                        i,
                        j,
                        k,
                        gamma_ijk,
                        tol
                    );
                }
            }
        }

        let temperatures = [21.0, 19.0, 22.0, 18.0];
        let capacitances = [10_000.0, 50_000.0, 30_000.0, 80_000.0];
        let r_tr = [120.0, 80.0, 200.0];
        let m_9r4c = ThermalManifold::from_9r4c_parameters(temperatures, capacitances, r_tr, None);
        let gamma_9r4c = m_9r4c.compute_christoffel_symbols();
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                for k in 0..MANIFOLD_DIM {
                    let gamma_ijk = gamma_9r4c[(i, j)][(j, k)];
                    assert!(
                        gamma_ijk.abs() <= tol,
                        "9R4C Christoffel symbol Γ[{},{},{}] = {:.3e} exceeds {}",
                        i,
                        j,
                        k,
                        gamma_ijk,
                        tol
                    );
                }
            }
        }
    }

    /// When all Christoffel symbols are zero (as they are for both 5R1C and
    /// 9R4C by construction), covariant transport reduces exactly to the
    /// forward-Euler `M·T + A`. This verifies the invariant that the
    /// Phase 1b covariant formula is a **strict generalisation** of the
    /// Phase 1a stub.
    #[test]
    fn test_christoffel_transport_zero_connection_gives_forward_euler() {
        let r_eq = 0.10;
        let c_air = 10_000.0;
        let c_mass = 50_000.0;
        let t_air_0 = 20.0;
        let t_mass_0 = 20.0;
        let q_int = 200.0;
        let q_solar = 800.0;
        let dt = 60.0;

        let mut manifold =
            ThermalManifold::from_5r1c_parameters(t_air_0, t_mass_0, r_eq, c_air, c_mass);
        manifold.gauge_connection[ManifoldIndex::Air as usize] = q_int / c_air;
        manifold.gauge_connection[ManifoldIndex::Wall as usize] = q_solar / c_mass;

        let gamma = manifold.compute_christoffel_symbols();
        let mut all_zero = true;
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                for k in 0..MANIFOLD_DIM {
                    if gamma[(i, j)][(j, k)].abs() >= 1e-12 {
                        all_zero = false;
                    }
                }
            }
        }
        assert!(
            all_zero,
            "Christoffel symbols must all be zero for this test (5R1C metric is constant)"
        );

        let covariant = manifold.compute_parallel_transport(dt);

        let metric_term = manifold.metric_tensor * manifold.scalar_field;
        let forward_euler = manifold.scalar_field + (metric_term + manifold.gauge_connection) * dt;

        for i in 0..MANIFOLD_DIM {
            assert!(
                (covariant[i] - forward_euler[i]).abs() < 1e-10,
                "covariant transport must reduce to forward-Euler when Γ=0 at index {}: \
                 covariant={:.6e}, forward_euler={:.6e}",
                i,
                covariant[i],
                forward_euler[i]
            );
        }
    }

    /// Identity metric gives zero Christoffel symbols (trivially flat space).
    /// This is a basic sanity check that the Levi-Civita computation is
    /// correct for the simplest possible case.
    #[test]
    fn test_christoffel_symbols_identity_metric_is_zero() {
        let m = ThermalManifold::new_flat();
        let gamma = m.compute_christoffel_symbols();

        let tol = 1e-15;
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                for k in 0..MANIFOLD_DIM {
                    let gamma_ijk = gamma[(i, j)][(j, k)];
                    assert!(
                        gamma_ijk.abs() <= tol,
                        "Christoffel symbol Γ[{},{},{}] = {:.3e} should be ~0 for identity metric",
                        i,
                        j,
                        k,
                        gamma_ijk
                    );
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // QUBO translation utility tests (issue #1772)
    // -------------------------------------------------------------------------
    //
    // The acceptance criterion for #1772 is a round-trip
    // (tensor → QUBO → tensor) plus edge-case coverage. These tests cover the
    // standardized `QuboMatrix` / `QuboEncoding` surface and the
    // `ThermalManifold::to_qubo_matrix` entry point.

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn test_qubo_encoding_default() {
        let e = QuboEncoding::default();
        assert_eq!(e.bits_per_node, 8);
        assert_eq!(e.scale_max_celsius, 50.0);
        assert_eq!(e.num_variables(), MANIFOLD_DIM * 8);
    }

    #[test]
    fn test_qubo_encoding_scale_and_lsb() {
        let e = QuboEncoding::default();
        // K=8, scale_max=50: scale = 255/50 = 5.1
        assert!(approx_eq(e.scale_factor(), 5.1, 1e-12));
        // LSB = 50/255 ≈ 0.19608
        assert!(approx_eq(e.lsb_resolution_celsius(), 50.0 / 255.0, 1e-12));
    }

    #[test]
    fn test_qubo_encoding_validate_rejects_invalid() {
        let zero_bits = QuboEncoding {
            bits_per_node: 0,
            ..Default::default()
        };
        assert!(matches!(
            zero_bits.validate(),
            Err(QuboTranslateError::InvalidEncoding(_))
        ));

        let too_many = QuboEncoding {
            bits_per_node: 17,
            ..Default::default()
        };
        assert!(matches!(
            too_many.validate(),
            Err(QuboTranslateError::InvalidEncoding(_))
        ));

        let bad_scale = QuboEncoding {
            scale_max_celsius: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            bad_scale.validate(),
            Err(QuboTranslateError::InvalidEncoding(_))
        ));

        let neg_scale = QuboEncoding {
            scale_max_celsius: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            neg_scale.validate(),
            Err(QuboTranslateError::InvalidEncoding(_))
        ));
    }

    #[test]
    fn test_qubo_encoding_validate_accepts_valid() {
        assert!(QuboEncoding::default().validate().is_ok());
        for k in [1_usize, 4, 8, 12, 16] {
            assert!(QuboEncoding {
                bits_per_node: k,
                ..Default::default()
            }
            .validate()
            .is_ok());
        }
    }

    #[test]
    fn test_qubo_matrix_size_scales_with_bits() {
        let m = ThermalManifold::new_flat();
        for k in [1_usize, 4, 8, 12, 16] {
            let e = QuboEncoding {
                bits_per_node: k,
                ..Default::default()
            };
            let qm = m.to_qubo_matrix(e).expect("ok");
            assert_eq!(qm.n_variables(), MANIFOLD_DIM * k);
            assert_eq!(qm.matrix().len(), (MANIFOLD_DIM * k) * (MANIFOLD_DIM * k));
        }
    }

    #[test]
    fn test_qubo_matrix_rejects_invalid_encoding() {
        let m = ThermalManifold::new_flat();
        let bad = QuboEncoding {
            bits_per_node: 0,
            ..Default::default()
        };
        assert!(matches!(
            m.to_qubo_matrix(bad),
            Err(QuboTranslateError::InvalidEncoding(_))
        ));
    }

    #[test]
    fn test_qubo_matrix_rejects_nan_manifold() {
        let mut m = ThermalManifold::new_flat();
        m.metric_tensor[(0, 0)] = f64::NAN;
        assert!(matches!(
            m.to_qubo_matrix(QuboEncoding::default()),
            Err(QuboTranslateError::InvalidManifold(_))
        ));
    }

    #[test]
    fn test_qubo_matrix_is_symmetric_flat() {
        let m = ThermalManifold::new_flat();
        let qm = m.to_qubo_matrix(QuboEncoding::default()).expect("ok");
        assert!(qm.is_symmetric(1e-12));
        for i in 0..qm.n_variables() {
            for j in 0..qm.n_variables() {
                assert!(
                    approx_eq(qm.entry(i, j), qm.entry(j, i), 1e-12),
                    "Q[{i},{j}] != Q[{j},{i}]"
                );
            }
        }
    }

    #[test]
    fn test_qubo_matrix_is_symmetric_5r1c() {
        let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let qm = m.to_qubo_matrix(QuboEncoding::default()).expect("ok");
        assert!(qm.is_symmetric(1e-12));
    }

    #[test]
    fn test_qubo_matrix_is_symmetric_random_metric() {
        let mut m = ThermalManifold::new_flat();
        m.metric_tensor = Matrix4::from_row_slice(&[
            0.1, 0.02, 0.0, 0.0, //
            0.02, -0.05, 0.01, 0.0, //
            0.0, 0.01, -0.03, 0.005, //
            0.0, 0.0, 0.005, -0.04, //
        ]);
        let qm = m.to_qubo_matrix(QuboEncoding::default()).expect("ok");
        assert!(qm.is_symmetric(1e-12));
    }

    #[test]
    fn test_qubo_matrix_entry_panics_on_out_of_bounds() {
        let m = ThermalManifold::new_flat();
        let qm = m.to_qubo_matrix(QuboEncoding::default()).expect("ok");
        let n = qm.n_variables();
        assert_eq!(qm.entry(0, 0), qm.matrix()[0]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| qm.entry(n, 0)));
        assert!(result.is_err(), "entry(n, 0) must panic");
    }

    #[test]
    fn test_qubo_round_trip_tensor_to_qubo_and_back_flat() {
        // Identity metric — the simplest round-trip.
        let m = ThermalManifold::new_flat();
        let e = QuboEncoding::default();
        let qm = m.to_qubo_matrix(e).expect("ok");
        let recon = qm.reconstruct_metric_tensor();
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                assert!(
                    approx_eq(m.metric_tensor[(i, j)], recon[(i, j)], 1e-12),
                    "metric[{i},{j}] = {} != recon {}",
                    m.metric_tensor[(i, j)],
                    recon[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_qubo_round_trip_tensor_to_qubo_and_back_5r1c() {
        // The 5R1C metric is non-symmetric (metric[0,1] = g/C_air ≠ g/C_mass =
        // metric[1,0]). A symmetric QUBO encodes only sym(M); the round-trip
        // therefore recovers sym(metric), which is what x^T Q x evaluates.
        let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let e = QuboEncoding::default();
        let qm = m.to_qubo_matrix(e).expect("ok");
        let recon = qm.reconstruct_metric_tensor();
        let sym_metric = 0.5 * (m.metric_tensor + m.metric_tensor.transpose());
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                assert!(
                    approx_eq(sym_metric[(i, j)], recon[(i, j)], 1e-12),
                    "5R1C sym(metric)[{i},{j}] = {} != recon {}",
                    sym_metric[(i, j)],
                    recon[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_qubo_round_trip_tensor_to_qubo_and_back_9r4c() {
        let temps = [22.0, 20.0, 23.0, 18.0];
        let caps = [1000.0, 5000.0, 3000.0, 8000.0];
        let r_tr = [50.0, 30.0, 20.0];
        let r_cross = Some([5.0, 3.0, 2.0]);
        let m = ThermalManifold::from_9r4c_parameters(temps, caps, r_tr, r_cross);
        let e = QuboEncoding::default();
        let qm = m.to_qubo_matrix(e).expect("ok");
        let recon = qm.reconstruct_metric_tensor();
        let sym_metric = 0.5 * (m.metric_tensor + m.metric_tensor.transpose());
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                assert!(
                    approx_eq(sym_metric[(i, j)], recon[(i, j)], 1e-12),
                    "9R4C sym(metric)[{i},{j}] = {} != recon {}",
                    sym_metric[(i, j)],
                    recon[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_qubo_round_trip_tensor_to_qubo_and_back_dense_metric() {
        // A fully dense, non-symmetric-on-input metric: the symmetrization
        // step averages off-diagonals, so the round-trip recovers the *averaged*
        // metric (which is what the QUBO encodes). Use a symmetric input so
        // the round-trip is exact.
        let mut m = ThermalManifold::new_flat();
        m.metric_tensor = Matrix4::from_row_slice(&[
            0.10, 0.02, 0.01, 0.005, //
            0.02, -0.05, 0.03, 0.0, //
            0.01, 0.03, -0.04, 0.015, //
            0.005, 0.0, 0.015, -0.07, //
        ]);
        let e = QuboEncoding::default();
        let qm = m.to_qubo_matrix(e).expect("ok");
        let recon = qm.reconstruct_metric_tensor();
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                assert!(
                    approx_eq(m.metric_tensor[(i, j)], recon[(i, j)], 1e-12),
                    "dense metric[{i},{j}] = {} != recon {}",
                    m.metric_tensor[(i, j)],
                    recon[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_qubo_round_trip_holds_across_bit_widths() {
        // 5R1C metric is non-symmetric → reconstruct recovers sym(metric).
        let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let sym_metric = 0.5 * (m.metric_tensor + m.metric_tensor.transpose());
        for k in [1_usize, 2, 4, 8, 12, 16] {
            let e = QuboEncoding {
                bits_per_node: k,
                ..Default::default()
            };
            let qm = m.to_qubo_matrix(e).expect("ok");
            let recon = qm.reconstruct_metric_tensor();
            for i in 0..MANIFOLD_DIM {
                for j in 0..MANIFOLD_DIM {
                    assert!(
                        approx_eq(sym_metric[(i, j)], recon[(i, j)], 1e-12),
                        "k={k} sym(metric)[{i},{j}] round-trip failed"
                    );
                }
            }
        }
    }

    #[test]
    fn test_qubo_energy_matches_metric_density() {
        // x^T Q x == T_recon^T M T_recon for any binary x (within fp tolerance).
        let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let e = QuboEncoding::default();
        let qm = m.to_qubo_matrix(e).expect("ok");
        let k = e.bits_per_node;
        let scale = e.scale_factor();

        // Deterministic pseudo-random binary vectors (LCG).
        for seed in 0..32_u64 {
            let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut x = Vec::with_capacity(qm.n_variables());
            for _ in 0..qm.n_variables() {
                x.push((rng & 1) as u8);
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            }

            // Reconstruct T from x using the same fixed-point encoding.
            let mut t_recon = Vector4::zeros();
            for i in 0..MANIFOLD_DIM {
                let mut v = 0u64;
                for bit in 0..k {
                    if x[i * k + bit] != 0 {
                        v |= 1u64 << bit;
                    }
                }
                t_recon[i] = v as f64 / scale;
            }

            let mut e_density = 0.0_f64;
            for i in 0..MANIFOLD_DIM {
                for j in 0..MANIFOLD_DIM {
                    e_density += m.metric_tensor[(i, j)] * t_recon[i] * t_recon[j];
                }
            }
            let e_qubo = qm.evaluate(&x);
            assert!(
                approx_eq(e_qubo, e_density, 1e-9),
                "E_QUBO {e_qubo} != E_density {e_density} for seed {seed}"
            );
        }
    }

    #[test]
    fn test_qubo_matrix_max_abs() {
        let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let qm = m.to_qubo_matrix(QuboEncoding::default()).expect("ok");
        let max_abs = qm.max_abs();
        assert!(max_abs > 0.0);
        // max_abs must be ≥ every |entry|.
        for &v in qm.matrix() {
            assert!(v.abs() <= max_abs + 1e-15);
        }
    }

    #[test]
    fn test_qubo_matrix_max_abs_zero_for_zero_metric() {
        // An all-zero metric produces an all-zero QUBO → max_abs == 0.
        let mut m = ThermalManifold::new_flat();
        m.metric_tensor = Matrix4::zeros();
        let qm = m.to_qubo_matrix(QuboEncoding::default()).expect("ok");
        assert!(qm.matrix().iter().all(|&v| v == 0.0));
        assert_eq!(qm.max_abs(), 0.0);
    }

    #[test]
    fn test_qubo_matrix_evaluate_panics_on_wrong_length() {
        let m = ThermalManifold::new_flat();
        let qm = m.to_qubo_matrix(QuboEncoding::default()).expect("ok");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| qm.evaluate(&[0u8])));
        assert!(result.is_err(), "evaluate must panic on wrong length");
    }

    #[test]
    fn test_qubo_error_display() {
        let e = QuboTranslateError::InvalidEncoding("bad".to_string());
        assert!(format!("{e}").contains("invalid QUBO encoding"));
        let e = QuboTranslateError::InvalidManifold("bad".to_string());
        assert!(format!("{e}").contains("invalid manifold"));
    }

    #[test]
    fn test_qubo_minimal_bit_width_round_trip() {
        // K=1 is the minimal non-empty encoding: one bit per node, 4 qubits.
        // 5R1C metric is non-symmetric → reconstruct recovers sym(metric).
        let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let sym_metric = 0.5 * (m.metric_tensor + m.metric_tensor.transpose());
        let e = QuboEncoding {
            bits_per_node: 1,
            scale_max_celsius: 50.0,
        };
        let qm = m.to_qubo_matrix(e).expect("ok");
        assert_eq!(qm.n_variables(), MANIFOLD_DIM);
        let recon = qm.reconstruct_metric_tensor();
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                assert!(
                    approx_eq(sym_metric[(i, j)], recon[(i, j)], 1e-12),
                    "K=1 sym(metric)[{i},{j}] round-trip failed"
                );
            }
        }
    }

    #[test]
    fn test_qubo_matrix_rejects_inf_manifold() {
        let mut m = ThermalManifold::new_flat();
        m.scalar_field[1] = f64::INFINITY;
        assert!(matches!(
            m.to_qubo_matrix(QuboEncoding::default()),
            Err(QuboTranslateError::InvalidManifold(_))
        ));
    }

    // ---- ZoneCountPolicy tests moved to fluxion-core/src/zone_count_policy.rs
    // (Issue #3871 — hoist to leaf crate). The 11 unit tests now live with
    // the type they exercise; geometry_tensor.rs only re-exports the type.
}
