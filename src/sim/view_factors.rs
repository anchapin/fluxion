//! Radiative view factor calculations for inter-zone heat transfer.
//!
//! This module provides functions for computing geometric view factors between
//! parallel rectangular surfaces, particularly for radiative exchange between
//! zones in building energy simulations.
//!
//! # Reciprocity
//!
//! All view factors obey the geometric identity
//! `F_AB * A_A = F_BA * A_B` (reciprocity). Functions in this module are
//! **directional**: [`hottels_rectangular_view_factor`] returns `F_AB` (the
//! fraction of A's emission reaching B). To obtain `F_BA`, use
//! [`reciprocal_view_factor`], [`hottels_rectangular_view_factor_pair`], or
//! [`build_zone_view_factors`].
//!
//! # Hottel's method — common-wall limit
//!
//! For two parallel rectangles of dimensions `a × b` (surface A) and `c × d`
//! (surface B), the view factor reduces in the common-wall limit (separation
//! ≈ 0) to
//!
//! ```text
//! F_AB = A_overlap / A_A   clamped to [0, 1]
//! ```
//!
//! where `A_overlap = min(a, c) * min(b, d)` for parallel aligned rectangles.
//! This is the directional — **not symmetric** — view factor. For nonzero
//! separation the same expression is used as a conservative approximation;
//! the full Hottel crossed-string formula is future work tracked outside
//! issue #1444.
//!
//! # Issue #1444 — reciprocity violation
//!
//! The previous implementation returned `(common / A_a) * min(common / A_b, 1)`,
//! which is symmetric in A and B and therefore violates reciprocity for
//! asymmetric rectangles (e.g. 8 m × 3 m vs 8 m × 2 m). Python verification
//! gave `F_AB * A_A = 16` while `F_BA * A_B = 10.67`, residual 5.33 — radiative
//! energy was not conserved across the partition. The new formula is
//! directional; reciprocity is recovered by deriving `F_BA` via
//! `F_BA = F_AB * A_A / A_B`.

use nalgebra::DMatrix;

/// Tolerance below which two lengths are considered equal (m).
const LENGTH_TOL: f64 = 1e-6;

/// Threshold below which `separation` is treated as a common wall (m).
const COMMON_WALL_SEP: f64 = 0.01;

/// Calculates the directional view factor `F_AB` from surface A to surface B
/// for two parallel rectangles using Hottel's common-wall method.
///
/// Returns `F_AB ∈ [0, 1]`, the fraction of A's radiative emission that reaches
/// B. **This is not symmetric**: `hottels_rectangular_view_factor(a, b, c, d, s)`
/// is not equal to `hottels_rectangular_view_factor(c, d, a, b, s)` in general.
/// To obtain the reverse direction, use [`reciprocal_view_factor`] or
/// [`hottels_rectangular_view_factor_pair`].
///
/// # Arguments
/// * `a_length` - Length of surface A (m)
/// * `a_width`  - Width of surface A (m)
/// * `b_length` - Length of surface B (m)
/// * `b_width`  - Width of surface B (m)
/// * `separation` - Distance between the two surfaces along the normal (m).
///   Values `< 0.01` are treated as a common wall so the analytical limit
///   `A_overlap / A_A` applies. For larger separations the same expression is
///   used as a conservative approximation.
///
/// # Returns
/// View factor `F_AB` ∈ `[0, 1]`.
pub fn hottels_rectangular_view_factor(
    a_length: f64,
    a_width: f64,
    b_length: f64,
    b_width: f64,
    separation: f64,
) -> f64 {
    let area_a = a_length * a_width;
    if area_a <= 0.0 {
        return 0.0;
    }

    // Special case: equal-sized aligned rectangles on a common (or near-common)
    // wall.  F_AB = 1.0 because every ray from A reaches B and vice versa.
    if separation < COMMON_WALL_SEP && approx_eq(a_length, b_length) && approx_eq(a_width, b_width)
    {
        return 1.0;
    }

    // Common-wall (or near-zero-separation) limit:
    //   F_AB = A_overlap / A_A
    // For aligned rectangles this is the fraction of A's footprint covered
    // by B's footprint (or vice versa when A is contained in B's projection,
    // in which case the formula clamps to 1.0). For nonzero separation the
    // same expression is used as a conservative approximation.
    let overlap = a_length.min(b_length) * a_width.min(b_width);
    (overlap / area_a).clamp(0.0, 1.0)
}

/// Derives the reciprocal view factor `F_BA` from `F_AB` using the
/// reciprocity identity `F_BA * A_B = F_AB * A_A`.
///
/// Use this whenever you have computed `F_AB` and need the reverse direction.
///
/// # Arguments
/// * `f_ab`  - View factor from A to B (dimensionless)
/// * `area_a` - Area of surface A (m²)
/// * `area_b` - Area of surface B (m²)
///
/// # Returns
/// View factor `F_BA` ∈ `[0, 1]` that satisfies `F_BA * A_B = F_AB * A_A`.
pub fn reciprocal_view_factor(f_ab: f64, area_a: f64, area_b: f64) -> f64 {
    if area_b <= 0.0 {
        return 0.0;
    }
    (f_ab * area_a / area_b).clamp(0.0, 1.0)
}

/// Returns both directional view factors `(F_AB, F_BA)` for two parallel
/// rectangles in a single call, enforcing reciprocity by construction.
///
/// # Arguments
/// Same as [`hottels_rectangular_view_factor`].
///
/// # Returns
/// Tuple `(F_AB, F_BA)` satisfying `F_AB * A_A == F_BA * A_B`.
pub fn hottels_rectangular_view_factor_pair(
    a_length: f64,
    a_width: f64,
    b_length: f64,
    b_width: f64,
    separation: f64,
) -> (f64, f64) {
    let f_ab = hottels_rectangular_view_factor(a_length, a_width, b_length, b_width, separation);
    let area_a = a_length * a_width;
    let area_b = b_length * b_width;
    let f_ba = reciprocal_view_factor(f_ab, area_a, area_b);
    (f_ab, f_ba)
}

/// Returns view factor between two windows on a common wall.
///
/// For windows directly opposite each other with negligible wall thickness, the
/// view factor is effectively 1.0 (every ray leaving one window reaches the
/// other). This is the common-wall limit of Hottel's method for aligned
/// equal-area rectangles at zero separation.
///
/// # Arguments
/// * `_window_area` - Area of window through which radiation passes (m²)
///   (unused — kept for API compatibility)
///
/// # Returns
/// View factor (dimensionless, equal to 1.0 for aligned equal-area windows).
pub fn window_to_window_view_factor(_window_area: f64) -> f64 {
    // For Case 960, windows are aligned with negligible wall thickness.
    // F = 1.0 is the common-wall limit of Hottel's method.
    1.0
}

// ---------------------------------------------------------------------------
// Matrix builder — Issue #1444
// ---------------------------------------------------------------------------

/// Geometry for one common wall (or window pair) shared between two zones.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommonWallGeometry {
    /// Index of zone A in the resulting view-factor matrix.
    pub zone_a: usize,
    /// Index of zone B in the resulting view-factor matrix.
    pub zone_b: usize,
    /// Length of the window on the A-side of the wall (m).
    pub a_length: f64,
    /// Width of the window on the A-side of the wall (m).
    pub a_width: f64,
    /// Length of the window on the B-side of the wall (m).
    pub b_length: f64,
    /// Width of the window on the B-side of the wall (m).
    pub b_width: f64,
    /// Distance between the two windows along the wall normal (m).
    pub separation: f64,
}

impl CommonWallGeometry {
    /// Area of the window on the A-side (m²).
    pub fn area_a(&self) -> f64 {
        self.a_length * self.a_width
    }

    /// Area of the window on the B-side (m²).
    pub fn area_b(&self) -> f64 {
        self.b_length * self.b_width
    }
}

/// Builds the inter-zone view-factor matrix `F` for `n_zones` zones given a
/// list of shared walls.
///
/// The returned `DMatrix<f64>` of shape `(n_zones, n_zones)` follows the
/// convention from issue #1444: `F[i, j]` is the view factor **from** zone
/// `j` **to** zone `i`. The diagonal is zero (no self-view).
///
/// Reciprocity is enforced by construction per wall:
/// `F[i, j] * A_j == F[j, i] * A_i` for every common wall `(i, j)`, where
/// `A_k` is the area of the window surface on zone `k`'s side of that wall
/// (taken from `CommonWallGeometry::area_a` / `area_b`). Walls whose
/// `zone_a` or `zone_b` fall outside `[0, n_zones)`, or whose endpoints
/// coincide, are silently ignored.
///
/// If multiple walls connect the same `(i, j)` pair, the later entry in
/// `common_walls` overwrites the earlier one.
///
/// # Arguments
/// * `n_zones` - Number of zones (matrix dimension)
/// * `common_walls` - List of shared wall geometries (one per window pair)
///
/// # Returns
/// `DMatrix<f64>` of shape `(n_zones, n_zones)` with `F[i, j]` = view factor
/// from zone `j` to zone `i`.
pub fn build_zone_view_factors(
    n_zones: usize,
    common_walls: &[CommonWallGeometry],
) -> DMatrix<f64> {
    let mut m = DMatrix::<f64>::zeros(n_zones, n_zones);
    for wall in common_walls {
        if wall.zone_a >= n_zones || wall.zone_b >= n_zones || wall.zone_a == wall.zone_b {
            continue;
        }
        let (f_ab, f_ba) = hottels_rectangular_view_factor_pair(
            wall.a_length,
            wall.a_width,
            wall.b_length,
            wall.b_width,
            wall.separation,
        );
        // Per-wall reciprocity sanity check.
        debug_assert!(
            (f_ab * wall.area_a() - f_ba * wall.area_b()).abs() < 1e-9,
            "build_zone_view_factors: reciprocity violated for wall {:?}: \
             F_AB*A_A={:.6e} F_BA*A_B={:.6e}",
            wall,
            f_ab * wall.area_a(),
            f_ba * wall.area_b(),
        );
        // Convention: F[i, j] = view factor FROM zone j TO zone i.
        // So F[zone_b, zone_a] = F_AB (from zone_a to zone_b).
        m[(wall.zone_b, wall.zone_a)] = f_ab;
        m[(wall.zone_a, wall.zone_b)] = f_ba;
    }
    m
}

/// Returns the largest reciprocity residual over all `(i, j)` pairs in the
/// view-factor matrix, given a function that supplies the relevant area
/// pair `(A_i, A_j)` for each `(i, j)`. Pairs where both `F[i, j]` and
/// `F[j, i]` are zero are skipped.
///
/// This helper is most useful for validating matrix builders that aggregate
/// multiple walls per zone pair — the per-wall reciprocity check inside
/// [`build_zone_view_factors`] only catches single-wall bugs.
pub fn max_reciprocity_residual<F>(matrix: &DMatrix<f64>, areas_for_pair: F) -> f64
where
    F: Fn(usize, usize) -> (f64, f64),
{
    let n = matrix.nrows();
    debug_assert_eq!(matrix.ncols(), n);
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let fij = matrix[(i, j)];
            let fji = matrix[(j, i)];
            if fij.abs() < 1e-15 && fji.abs() < 1e-15 {
                continue;
            }
            let (a_i, a_j) = areas_for_pair(i, j);
            let residual = (fij * a_j - fji * a_i).abs();
            if residual > worst {
                worst = residual;
            }
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= LENGTH_TOL
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for reciprocity in the unit tests.
    const RECIPROCITY_TOL: f64 = 1e-9;

    // -----------------------------------------------------------------------
    // Reciprocity — issue #1444 acceptance criteria
    // -----------------------------------------------------------------------

    /// The bug example from issue #1444: 8m × 3m vs 8m × 2m, separation 0.1m.
    /// The old formula returned the same value regardless of which surface was
    /// "from", violating `F_AB * A_A == F_BA * A_B`.
    #[test]
    fn test_reciprocity_asymmetric_8x3_vs_8x2() {
        let (f_ab, f_ba) = hottels_rectangular_view_factor_pair(8.0, 3.0, 8.0, 2.0, 0.1);
        let a_a = 8.0 * 3.0;
        let a_b = 8.0 * 2.0;
        let residual = (f_ab * a_a - f_ba * a_b).abs();
        assert!(
            residual < RECIPROCITY_TOL,
            "reciprocity violated: F_AB*A_A={:.6e} F_BA*A_B={:.6e} residual={:.3e}",
            f_ab * a_a,
            f_ba * a_b,
            residual
        );
        // For aligned 8x3 vs 8x2 with 8x2 fully contained, common-wall limit
        // says F_AB → 16/24 ≈ 0.667 and F_BA → 1.0.
        assert!((f_ab - 16.0 / 24.0).abs() < 1e-9);
        assert!((f_ba - 1.0).abs() < 1e-6);
    }

    /// Reciprocity must hold for the reverse call as well — F_AB when A is
    /// the smaller surface must differ from F_AB when A is the larger.
    #[test]
    fn test_reciprocity_swapped_arguments() {
        let (f_ab, f_ba) = hottels_rectangular_view_factor_pair(8.0, 3.0, 8.0, 2.0, 0.1);
        let (f_ab_swapped, f_ba_swapped) =
            hottels_rectangular_view_factor_pair(8.0, 2.0, 8.0, 3.0, 0.1);
        // F_AB in one call equals F_BA in the swapped call (same geometry).
        assert!((f_ab - f_ba_swapped).abs() < 1e-12);
        assert!((f_ba - f_ab_swapped).abs() < 1e-12);
    }

    /// Reciprocity must hold for many rectangular configurations spanning
    /// the common cases (aligned equal, aligned asymmetric, slight offset,
    /// large separation).
    #[test]
    fn test_reciprocity_random_configurations() {
        let configs: &[(f64, f64, f64, f64, f64)] = &[
            // (aL, aW, bL, bW, sep)
            (8.0, 3.0, 8.0, 2.0, 0.1), // issue #1444 example
            (8.0, 3.0, 8.0, 2.9, 0.1),
            (8.0, 3.0, 8.0, 3.0, 0.0),
            (8.0, 3.0, 8.0, 3.0, 0.001),
            (10.0, 4.0, 6.0, 2.0, 0.2),
            (5.0, 5.0, 5.0, 5.0, 0.1),
            (2.0, 1.5, 4.0, 1.0, 0.5),
            (1.0, 1.0, 3.0, 3.0, 0.0),
            (12.0, 2.0, 4.0, 2.0, 0.05),
            (8.0, 3.0, 8.0, 2.0, 2.0), // large separation
            (1.5, 1.0, 4.0, 2.5, 0.3),
            (6.0, 4.0, 2.0, 1.0, 0.0),
            (8.0, 3.0, 8.0, 2.0, 0.0), // perfect common wall
            (8.0, 3.0, 8.0, 2.0, 0.005),
            (20.0, 5.0, 4.0, 1.0, 0.1),
            (3.0, 2.0, 6.0, 4.0, 0.1),
            (8.0, 3.0, 8.0, 1.0, 0.05),
        ];
        for &(aL, aW, bL, bW, sep) in configs {
            let (f_ab, f_ba) = hottels_rectangular_view_factor_pair(aL, aW, bL, bW, sep);
            let a_a = aL * aW;
            let a_b = bL * bW;
            let residual = (f_ab * a_a - f_ba * a_b).abs();
            assert!(
                residual < RECIPROCITY_TOL,
                "reciprocity violated for ({aL}x{aW}, {bL}x{bW}, sep={sep}): \
                 F_AB={f_ab:.9e} F_BA={f_ba:.9e} residual={residual:.3e}"
            );
        }
    }

    /// Reciprocity must hold for the degenerate (b contained in a, common wall)
    /// case where F_AB = b_area / a_area and F_BA = 1.0.
    #[test]
    fn test_reciprocity_contained_geometry() {
        let (f_ab, f_ba) = hottels_rectangular_view_factor_pair(8.0, 3.0, 4.0, 1.0, 0.0);
        let a_a = 24.0;
        let a_b = 4.0;
        assert!((f_ab - 4.0 / 24.0).abs() < 1e-9);
        assert!((f_ba - 1.0).abs() < 1e-6);
        let residual = (f_ab * a_a - f_ba * a_b).abs();
        assert!(residual < RECIPROCITY_TOL);
    }

    /// Reciprocity must hold when one surface area is zero (degenerate).
    #[test]
    fn test_reciprocity_zero_area() {
        let (f_ab, f_ba) = hottels_rectangular_view_factor_pair(8.0, 3.0, 0.0, 2.0, 0.1);
        assert_eq!(f_ab, 0.0);
        assert_eq!(f_ba, 0.0);
    }

    // -----------------------------------------------------------------------
    // Directionality — the old formula was symmetric; the new one is not.
    // -----------------------------------------------------------------------

    #[test]
    fn test_directional_not_symmetric() {
        // The whole point of issue #1444: F_AB ≠ F_BA in general.
        let f_ab = hottels_rectangular_view_factor(8.0, 3.0, 8.0, 2.0, 0.1);
        let f_ba_direct = hottels_rectangular_view_factor(8.0, 2.0, 8.0, 3.0, 0.1);
        assert!((f_ab - 16.0 / 24.0).abs() < 1e-9);
        assert!((f_ba_direct - 1.0).abs() < 1e-6);
        // They must NOT be equal — the old buggy formula made them equal.
        assert!((f_ab - f_ba_direct).abs() > 0.1);
    }

    /// Specifically — reproduce the residual 5.33 from the issue text.
    /// Old code returned 0.667 for both directions ⇒ F_AB*A_A=16, F_BA*A_B=10.67.
    /// New code returns 0.667 / 1.0 ⇒ both products equal 16.
    #[test]
    fn test_reciprocity_residual_issue_example() {
        let (f_ab, f_ba) = hottels_rectangular_view_factor_pair(8.0, 3.0, 8.0, 2.0, 0.1);
        let f_ab_old: f64 = 16.0 / 24.0; // old formula's value in both directions
        let f_ba_old: f64 = 16.0 / 24.0; // (bug: was symmetric)
        let old_residual = (f_ab_old * 24.0 - f_ba_old * 16.0).abs();
        let new_residual = (f_ab * 24.0 - f_ba * 16.0).abs();
        // Old code violated reciprocity by ~5.33 m².
        assert!(old_residual > 5.0);
        // New code satisfies reciprocity to machine precision.
        assert!(new_residual < RECIPROCITY_TOL);
    }

    // -----------------------------------------------------------------------
    // Matrix builder — Issue #1444
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_zone_view_factors_two_zones_asymmetric() {
        let walls = vec![CommonWallGeometry {
            zone_a: 0,
            zone_b: 1,
            a_length: 8.0,
            a_width: 3.0,
            b_length: 8.0,
            b_width: 2.0,
            separation: 0.1,
        }];
        let m = build_zone_view_factors(2, &walls);

        // F[1, 0] = F_AB (from zone 0 to zone 1) = 16/24
        assert!((m[(1, 0)] - 16.0 / 24.0).abs() < 1e-9);
        // F[0, 1] = F_BA (from zone 1 to zone 0) = 1.0
        assert!((m[(0, 1)] - 1.0).abs() < 1e-6);
        // Diagonal zero
        assert_eq!(m[(0, 0)], 0.0);
        assert_eq!(m[(1, 1)], 0.0);

        // Reciprocity using the wall's own areas (A_0 = 24, A_1 = 16).
        let residual = (m[(1, 0)] * 24.0 - m[(0, 1)] * 16.0).abs();
        assert!(residual < RECIPROCITY_TOL);
    }

    #[test]
    fn test_build_zone_view_factors_three_zones() {
        let walls = vec![
            CommonWallGeometry {
                zone_a: 0,
                zone_b: 1,
                a_length: 8.0,
                a_width: 3.0,
                b_length: 8.0,
                b_width: 2.0,
                separation: 0.1,
            },
            // 8x2 (a) vs 4x4 (b), common wall — they partially overlap
            // (4x2 = 8 of 16 each side).  F_AB = F_BA = 0.5.
            CommonWallGeometry {
                zone_a: 1,
                zone_b: 2,
                a_length: 8.0,
                a_width: 2.0,
                b_length: 4.0,
                b_width: 4.0,
                separation: 0.0,
            },
        ];
        let m = build_zone_view_factors(3, &walls);

        // Pair 0-1: 8x3 vs 8x2 (issue #1444 geometry).
        assert!((m[(1, 0)] - 16.0 / 24.0).abs() < 1e-9);
        assert!((m[(0, 1)] - 1.0).abs() < 1e-6);
        // Pair 1-2: 8x2 vs 4x4, partial overlap ⇒ F_AB = F_BA = 0.5.
        assert!((m[(2, 1)] - 0.5).abs() < 1e-9);
        assert!((m[(1, 2)] - 0.5).abs() < 1e-9);
        // Reciprocity for the 1-2 pair: F[2,1] * A_1 = F[1,2] * A_2.
        assert!((m[(2, 1)] * 16.0 - m[(1, 2)] * 16.0).abs() < RECIPROCITY_TOL);
        // No direct connection between 0 and 2.
        assert_eq!(m[(2, 0)], 0.0);
        assert_eq!(m[(0, 2)], 0.0);
    }

    /// Reciprocity must hold across many rectangular configurations for the
    /// matrix builder, not just the pair helper.
    #[test]
    fn test_matrix_reciprocity_random_walls() {
        let walls = vec![
            CommonWallGeometry {
                zone_a: 0,
                zone_b: 1,
                a_length: 8.0,
                a_width: 3.0,
                b_length: 8.0,
                b_width: 2.0,
                separation: 0.1,
            },
            CommonWallGeometry {
                zone_a: 0,
                zone_b: 2,
                a_length: 8.0,
                a_width: 3.0,
                b_length: 4.0,
                b_width: 1.0,
                separation: 0.2,
            },
            CommonWallGeometry {
                zone_a: 1,
                zone_b: 2,
                a_length: 8.0,
                a_width: 2.0,
                b_length: 8.0,
                b_width: 2.5,
                separation: 0.05,
            },
        ];
        // The debug_assert inside build_zone_view_factors checks per-wall
        // reciprocity.  Here we also cross-check via the public helper.
        let m = build_zone_view_factors(3, &walls);
        let walls_by_pair = |i: usize, j: usize| -> (f64, f64) {
            for w in &walls {
                if (w.zone_a == i && w.zone_b == j) || (w.zone_a == j && w.zone_b == i) {
                    return (w.area_a(), w.area_b());
                }
            }
            (0.0, 0.0)
        };
        let worst = max_reciprocity_residual(&m, walls_by_pair);
        assert!(
            worst < RECIPROCITY_TOL,
            "matrix builder violates reciprocity, worst residual = {worst:.3e}"
        );
    }

    // -----------------------------------------------------------------------
    // Backward-compatible behavior for the existing call sites
    // -----------------------------------------------------------------------

    #[test]
    fn test_window_to_window_view_factor() {
        let f = window_to_window_view_factor(10.8);
        assert_eq!(f, 1.0);
    }

    #[test]
    fn test_hottels_aligned_windows() {
        let f = hottels_rectangular_view_factor(8.0, 3.0, 8.0, 3.0, 0.0);
        assert_eq!(f, 1.0);
    }

    #[test]
    fn test_hottels_area_ratio_offset() {
        // F_AB (8x3 → 8x2) = common/area_a = 16/24 ≈ 0.667.
        // This was the existing test value (the old formula happened to give
        // the same number in this specific direction because the second factor
        // min(common/area_b, 1) = 1).
        let f = hottels_rectangular_view_factor(8.0, 3.0, 8.0, 2.0, 0.1);
        let expected = 16.0 / 24.0;
        assert!(
            (f - expected).abs() < 1e-9,
            "Expected {:.9}, got {:.9}",
            expected,
            f
        );
    }

    #[test]
    fn test_hottels_separation_effect() {
        // Aligned with small separation still 1.0 (special case).
        let f_small = hottels_rectangular_view_factor(8.0, 3.0, 8.0, 3.0, 0.001);
        assert_eq!(f_small, 1.0);
        // Slight offset, larger separation: F_AB (8x3 → 8x2.9) < 1.0.
        let f_large = hottels_rectangular_view_factor(8.0, 3.0, 8.0, 2.9, 0.1);
        assert!(f_large < 1.0);
        assert!(f_large > 0.0);
    }

    #[test]
    fn test_hottels_case_960_scenario() {
        let f = hottels_rectangular_view_factor(8.0, 3.0, 8.0, 3.0, 0.0);
        assert_eq!(f, 1.0);
    }
}
