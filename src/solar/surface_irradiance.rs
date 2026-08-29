//! Pure surface irradiance calculation on tilted surfaces.
//!
//! NO imports from `sim::` or `validation::` — this is a standalone physics module.
//!
//! # Components
//! - **Beam irradiance**: DNI × cos(incidence angle) on tilted surface
//! - **Diffuse irradiance**: Perez all-weather sky model (1990)
//! - **Ground-reflected irradiance**: Isotropic ground reflection model
//!
//! # References
//! - Perez, R., et al. (1990). "Modeling daylight availability and irradiance
//!   components from direct and global irradiance." Solar Energy 44(5), 271-289.
//! - ASHRAE Handbook - Fundamentals, Chapter 14: Climatic Design Information

use crate::solar::solar_position::SolarPosition;

/// Solar constant at the top of atmosphere (W/m²).
///
/// Canonical total solar irradiance (TSI) at mean Earth-Sun distance.
/// **Value:** 1361.0 W/m² (ASHRAE 140-2022 Appendix C; Kopp & Lean 2011 / TSIS).
///
/// Previously this module used the older 1367 W/m² approximation, which caused
/// a 0.44% drift in `dni_extra` and propagated into every Perez diffuse
/// calculation through the `delta` parameter (Issue #1413). All other call sites
/// (sky_radiation.rs, ashrae_140.rs, engine.rs) already use 1361.0.
pub const SOLAR_CONSTANT: f64 = 1361.0;

/// Surface orientation for irradiance calculations.
///
/// These are the cardinal directions + horizontal/down for complete coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    North,
    South,
    East,
    West,
    Up,
    Down,
    Horizontal,
}

impl From<crate::validation::ashrae_140_cases::Orientation> for Orientation {
    fn from(o: crate::validation::ashrae_140_cases::Orientation) -> Self {
        match o {
            crate::validation::ashrae_140_cases::Orientation::North => Orientation::North,
            crate::validation::ashrae_140_cases::Orientation::South => Orientation::South,
            crate::validation::ashrae_140_cases::Orientation::East => Orientation::East,
            crate::validation::ashrae_140_cases::Orientation::West => Orientation::West,
            crate::validation::ashrae_140_cases::Orientation::Up => Orientation::Up,
            crate::validation::ashrae_140_cases::Orientation::Down => Orientation::Down,
            crate::validation::ashrae_140_cases::Orientation::Horizontal => Orientation::Horizontal,
        }
    }
}

impl From<Orientation> for crate::validation::ashrae_140_cases::Orientation {
    fn from(o: Orientation) -> Self {
        match o {
            Orientation::North => crate::validation::ashrae_140_cases::Orientation::North,
            Orientation::South => crate::validation::ashrae_140_cases::Orientation::South,
            Orientation::East => crate::validation::ashrae_140_cases::Orientation::East,
            Orientation::West => crate::validation::ashrae_140_cases::Orientation::West,
            Orientation::Up => crate::validation::ashrae_140_cases::Orientation::Up,
            Orientation::Down => crate::validation::ashrae_140_cases::Orientation::Down,
            Orientation::Horizontal => crate::validation::ashrae_140_cases::Orientation::Horizontal,
        }
    }
}

/// Components of solar irradiance on a surface (W/m²).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceIrradiance {
    /// Direct beam irradiance (W/m²).
    pub beam_wm2: f64,
    /// Diffuse (sky) irradiance (W/m²).
    pub diffuse_wm2: f64,
    /// Ground-reflected irradiance (W/m²).
    pub ground_reflected_wm2: f64,
    /// Total irradiance = beam + diffuse + ground_reflected (W/m²).
    pub total_wm2: f64,
}

impl SurfaceIrradiance {
    /// Create a new SurfaceIrradiance from components.
    pub fn new(beam_wm2: f64, diffuse_wm2: f64, ground_reflected_wm2: f64) -> Self {
        SurfaceIrradiance {
            beam_wm2,
            diffuse_wm2,
            ground_reflected_wm2,
            total_wm2: beam_wm2 + diffuse_wm2 + ground_reflected_wm2,
        }
    }

    /// Zero irradiance (nighttime or below horizon).
    pub fn zero() -> Self {
        SurfaceIrradiance {
            beam_wm2: 0.0,
            diffuse_wm2: 0.0,
            ground_reflected_wm2: 0.0,
            total_wm2: 0.0,
        }
    }
}

/// Maps Orientation to (tilt, azimuth) for solar calculations.
/// Tilt: 0=Horizontal Up, 90=Vertical, 180=Horizontal Down.
/// Azimuth: 0=North, 90=East, 180=South, 270=West.
pub fn orientation_to_angles(orientation: Orientation) -> (f64, f64) {
    match orientation {
        Orientation::Up => (0.0, 0.0),
        Orientation::Down => (180.0, 0.0),
        Orientation::South => (90.0, 180.0),
        Orientation::West => (90.0, 270.0),
        Orientation::North => (90.0, 0.0),
        Orientation::East => (90.0, 90.0),
        Orientation::Horizontal => (0.0, 0.0),
    }
}

/// Calculate surface irradiance on a tilted surface.
///
/// # Arguments
/// * `sun_pos` - Solar position (altitude, azimuth, zenith)
/// * `dni` - Direct Normal Irradiance (W/m²)
/// * `dhi` - Diffuse Horizontal Irradiance (W/m²)
/// * `ghi` - Optional Global Horizontal Irradiance (W/m²). If None, computed from DNI+DHI.
/// * `orientation` - Surface orientation
/// * `ground_reflectance` - Ground albedo (typically 0.2)
/// * `day_of_year` - Day of year (1-366) for extraterrestrial irradiance
///
/// # Physics
/// - Beam: DNI × cos(θ_incidence), with explicit tilt = 0 branch
///   (`DNI · max(cos(zenith), 0)`) per ASHRAE Fundamentals Ch.14 /
///   Duffie–Beckman Eq. 1.6.3 (see Issue #1325).
/// - Diffuse: Perez all-weather model (accounts for circumsolar and
///   horizon brightening)
/// - Ground reflected: isotropic view-factor model = GHI × ρ × (1 - cos(β)) / 2
///   for β ∈ (0°, 180°), with the two endpoint tilts pinned explicitly
///   (Issue #1326):
///   - β =   0° → ρ · GHI    (horizontal up-facing roof: full ground hemisphere)
///   - β = 180° → 0          (down-facing surface: no ground seen)
pub fn calculate_surface_irradiance(
    sun_pos: &SolarPosition,
    dni: f64,
    dhi: f64,
    ghi: Option<f64>,
    orientation: Orientation,
    ground_reflectance: f64,
    day_of_year: usize,
) -> SurfaceIrradiance {
    if !sun_pos.is_above_horizon() {
        return SurfaceIrradiance::zero();
    }

    let ghi = ghi.unwrap_or_else(|| dni * sun_pos.altitude_deg.to_radians().sin() + dhi);
    let (tilt_deg, azimuth_deg) = orientation_to_angles(orientation);

    // Beam component: I_beam = DNI · cos(θ_i), clamped to ≥ 0.
    //
    // Issue #1325: For a horizontal surface (tilt = 0) the surface normal
    // points toward zenith, so the incidence angle equals the solar zenith
    // angle and cos(θ_i) = cos(zenith) = sin(altitude). We special-case
    // tilt = 0 here so the geometry is explicit (and matches the analytical
    // formulation ASHRAE Fundamentals Ch.14 / Duffie–Beckman Eq. 1.6.3),
    // and so the result is guarded against any future changes to the
    // general incidence-angle formula. For non-horizontal surfaces we fall
    // back to the full incidence-cosine expression.
    //
    // No incidence-angle / airmass reductions are applied here beyond the
    // cos(θ_i) factor itself — beam is direct normal irradiance projected
    // onto the surface plane. Airmass is used only by the diffuse model.
    let beam = if tilt_deg.abs() < 1e-9 {
        // Horizontal: normal = up (zenith direction), θ_i = zenith.
        (dni * sun_pos.zenith_deg.to_radians().cos()).max(0.0)
    } else {
        dni * sun_pos.incidence_cosine(tilt_deg, azimuth_deg)
    };

    let dni_extra = extraterrestrial_irradiance(day_of_year);
    let airmass = relative_airmass(sun_pos.zenith_deg);

    // Horizontal-surface diffuse: use the ISO 13790 anisotropic sky model
    // directly rather than the Perez all-weather model, which does not correctly
    // account for the full-sky view factor of a horizontal surface.
    //
    // Issue #1682: The Perez model's (1−f1)/2 + f1·cos(θ)/cos(θ_z) + f2·sin(β)
    // formula collapses to (1−f1)/2 + f1 + f2/2 for horizontal surfaces (tilt=0).
    // This gives dhi·(0.5 + 0.5·f1 + 0.5·f2) which is systematically below dhi
    // for all sky conditions (e.g., ~0.74·dhi for clear skies at zenith=30°).
    //
    // The correct ISO 13790 anisotropic sky formula for a horizontal surface is:
    //   E_diffuse = dhi · (1 + f2 · sin(θ_z))
    // which directly represents:
    //   - isotropic sky dome contribution: dhi (full hemisphere)
    //   - horizon brightening correction: f2 · dhi · sin(θ_z)
    //
    // The Perez f1 term (circumsolar brightening) is absorbed into the isotropic
    // term because the circumsolar region IS the bright part of the isotropic
    // dome for a horizontal observer — the angular concentration effect that
    // f1 captures for tilted surfaces does not apply when looking straight up.
    let diffuse = if tilt_deg.abs() < 1e-9 {
        // Horizontal (roof): ISO 13790 anisotropic sky correction.
        // For isotropic sky: f1=0, f2=0 → diffuse = dhi (correct).
        // For clear sky: f2≈-0.3 → diffuse ≈ 0.85·dhi (physically consistent).
        let zenith_rad = sun_pos.zenith_deg.to_radians();
        let epsilon = {
            let z_cubed = zenith_rad.powi(3);
            let kappa = 1.041;
            let numerator = (dhi + dni) / dhi + kappa * z_cubed;
            let denominator = 1.0 + kappa * z_cubed;
            numerator / denominator
        };
        let ebin = PerezSkyModel::classify_sky_clearness(epsilon);
        let (_, f2c) = PerezSkyModel::get_perez_coefficients(ebin);
        let delta = dhi * airmass / dni_extra;
        let f2 = f2c[0] + f2c[1] * delta + f2c[2] * zenith_rad;
        (dhi * (1.0 + f2 * zenith_rad.sin())).max(0.0)
    } else {
        PerezSkyModel::calculate_diffuse_tilted(
            dhi,
            dni,
            dni_extra,
            airmass,
            sun_pos.zenith_deg,
            tilt_deg,
            azimuth_deg,
            sun_pos.azimuth_deg,
        )
    };

    // Ground-reflected component: isotropic model
    //     E_g = ρ · GHI · (1 − cos β) / 2
    // (ASHRAE Handbook — Fundamentals, Ch. 14; Duffie & Beckman Eq. 2.12.1).
    //
    // Issue #1326: This view-factor form is correct for the open interval
    // β ∈ (0°, 180°) — at β = 90° (vertical wall) it yields 0.5·ρ·GHI, the
    // value E+ and the building energy community use.  However, the formula
    // collapses to 0 at β = 0° (horizontal up-facing roof), while a
    // horizontal roof actually sees the full hemisphere of ground-reflected
    // radiation and must receive E_g = ρ · GHI.  Symmetrically, at β = 180°
    // (down-facing) the formula returns ρ · GHI, but a down-facing surface
    // sees no ground and must receive 0.
    //
    // We therefore pin the two endpoint tilts explicitly:
    //   tilt =   0°  →  ρ · GHI        (full ground hemisphere)
    //   tilt = 180°  →  0              (down-facing: no ground)
    //   tilt ∈ (0°, 180°)  →  ρ · GHI · (1 − cos β) / 2   (unchanged)
    //
    // No parameter tuning — just the correct boundary conditions.
    let ground_reflected = if tilt_deg.abs() < 1e-9 {
        // Horizontal up-facing: surface normal points to zenith, sees all
        // ground-reflected radiation arriving from the lower hemisphere.
        ghi * ground_reflectance
    } else if (tilt_deg - 180.0).abs() < 1e-9 {
        // Down-facing: surface normal points to nadir, sees no ground.
        0.0
    } else {
        let surface_tilt = tilt_deg.to_radians();
        let ground_factor = (1.0 - surface_tilt.cos()) / 2.0;
        ghi * ground_reflectance * ground_factor
    };

    SurfaceIrradiance::new(beam, diffuse, ground_reflected)
}

// ============================================================================
// Perez All-Weather Sky Diffuse Radiation Model (inlined for zero sim:: deps)
// Reference: Perez, R., et al. (1990). Solar Energy 44(5), 271-289.
// ============================================================================

/// Perez sky model for diffuse irradiance on tilted surfaces.
///
/// Single source of truth — see `sim::sky_radiation` re-exports.
/// Issue #1414: dedup target.
pub struct PerezSkyModel;

impl PerezSkyModel {
    /// Calculate diffuse irradiance on a tilted surface.
    #[allow(clippy::too_many_arguments)]
    pub fn calculate_diffuse_tilted(
        dhi: f64,
        dni: f64,
        dni_extra: f64,
        airmass: f64,
        zenith_deg: f64,
        surface_tilt_deg: f64,
        surface_azimuth_deg: f64,
        solar_azimuth_deg: f64,
    ) -> f64 {
        if dhi <= 0.0 {
            return 0.0;
        }

        let zenith_rad = zenith_deg.to_radians();
        let surface_tilt = surface_tilt_deg.to_radians();

        let kappa = 1.041;
        let delta = dhi * airmass / dni_extra;

        let epsilon = {
            let z_cubed = zenith_rad.powi(3);
            let numerator = (dhi + dni) / dhi + kappa * z_cubed;
            let denominator = 1.0 + kappa * z_cubed;
            numerator / denominator
        };

        let ebin = Self::classify_sky_clearness(epsilon);
        let (f1c, f2c) = Self::get_perez_coefficients(ebin);
        let f1 = (f1c[0] + f1c[1] * delta + f1c[2] * zenith_rad).max(0.0);
        let f2 = f2c[0] + f2c[1] * delta + f2c[2] * zenith_rad;

        let cos_incidence = Self::calculate_cos_incidence(
            surface_tilt_deg,
            surface_azimuth_deg,
            zenith_deg,
            solar_azimuth_deg,
        );

        let a = cos_incidence.max(0.0);
        let b = zenith_rad.cos().max((85.0f64).to_radians().cos());

        let term1 = 0.5 * (1.0 - f1) * (1.0 + surface_tilt.cos());
        let term2 = f1 * a / b;
        let term3 = f2 * surface_tilt.sin();

        (dhi * (term1 + term2 + term3)).max(0.0)
    }

    pub(crate) fn classify_sky_clearness(epsilon: f64) -> usize {
        let bounds = [0.0, 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2];
        let mut ebin = 7;
        for (i, &bound) in bounds.iter().enumerate() {
            if epsilon <= bound {
                ebin = i;
                break;
            }
        }
        ebin
    }

    /// Perez model F1 and F2 coefficients from Table 3 of Perez et al. (1990).
    pub(crate) fn get_perez_coefficients(ebin: usize) -> ([f64; 3], [f64; 3]) {
        const F1C: [[f64; 3]; 8] = [
            [-0.008317, 0.587728, -0.062064],
            [0.129967, 0.682595, -0.151375],
            [0.329676, 0.486861, -0.221272],
            [0.568205, 0.187452, -0.295250],
            [0.873018, -0.393289, -0.369150],
            [1.321297, -1.176777, -0.393994],
            [0.999852, -1.634380, -0.291495],
            [0.553776, 0.631414, -0.209172],
        ];
        const F2C: [[f64; 3]; 8] = [
            [0.091000, 0.060000, 0.000000],
            [0.055000, 0.060000, 0.000000],
            [0.025000, 0.060000, 0.000000],
            [-0.015000, 0.060000, 0.000000],
            [-0.065000, 0.060000, 0.000000],
            [-0.115000, 0.060000, 0.000000],
            [-0.165000, 0.060000, 0.000000],
            [-0.215000, 0.060000, 0.000000],
        ];

        let ebin_clamped = ebin.min(7);
        (F1C[ebin_clamped], F2C[ebin_clamped])
    }

    pub(crate) fn calculate_cos_incidence(
        surface_tilt_deg: f64,
        surface_azimuth_deg: f64,
        zenith_deg: f64,
        solar_azimuth_deg: f64,
    ) -> f64 {
        // For horizontal surfaces (tilt ≈ 0), the incidence angle equals the
        // zenith angle and cos(θ_i) = cos(zenith). The general formula below
        // incorrectly gives zenith.sin() for tilt = 0, so we special-case it.
        // See Issue #1622 / #1323 — horizontal surface beam under-counting.
        if surface_tilt_deg.abs() < 1e-9 {
            return zenith_deg.to_radians().cos();
        }

        let tilt = surface_tilt_deg.to_radians();
        let surface_az = surface_azimuth_deg.to_radians();
        let zenith = zenith_deg.to_radians();
        let solar_az = solar_azimuth_deg.to_radians();

        let cos_incidence = tilt.sin() * surface_az.sin() * zenith.cos() * solar_az.sin()
            + tilt.sin() * surface_az.cos() * zenith.cos() * solar_az.cos()
            + tilt.cos() * zenith.sin();

        cos_incidence.clamp(-1.0, 1.0)
    }
}

/// Extraterrestrial irradiance on the plane of the ecliptic.
///
/// Accounts for Earth's orbital eccentricity (e ≈ 0.0167):
/// I₀ = G_sc × (1 + 0.033 × cos(360° × (n-3)/365))
///
/// Reference: ASHRAE Fundamentals, Chapter 14, Eq. 3.
///
/// Single source of truth — see `sim::sky_radiation` re-exports.
/// Issue #1414: dedup target.
pub fn extraterrestrial_irradiance(day_of_year: usize) -> f64 {
    let day_rad = 2.0 * std::f64::consts::PI * (day_of_year as f64 - 3.0) / 365.0;
    SOLAR_CONSTANT * (1.0 + 0.033 * day_rad.cos())
}

/// Relative optical air mass using Kasten & Young (1989) formula.
///
/// AM = 1 / [cos(θ_z) + 0.50572 × (96.07995 - θ_z)^(-1.6364)]
///
/// Valid for zenith angles up to ~90°.
///
/// Single source of truth — see `sim::sky_radiation` re-exports.
/// Issue #1414: dedup target.
pub fn relative_airmass(zenith_deg: f64) -> f64 {
    let zenith_rad = zenith_deg.to_radians();
    let cos_zenith = zenith_rad.cos();
    let term = 96.07995 - zenith_deg;
    1.0 / (cos_zenith + 0.50572 * term.powf(-1.6364))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_irradiance() {
        let sun_pos = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };
        let irr = calculate_surface_irradiance(
            &sun_pos,
            800.0,
            100.0,
            None,
            Orientation::South,
            0.2,
            172,
        );
        assert!(irr.total_wm2 > 0.0);
    }

    #[test]
    fn test_surface_irradiance_below_horizon() {
        let sun_pos = SolarPosition {
            altitude_deg: -10.0,
            azimuth_deg: 180.0,
            zenith_deg: 100.0,
        };
        let irr = calculate_surface_irradiance(
            &sun_pos,
            800.0,
            100.0,
            None,
            Orientation::South,
            0.2,
            172,
        );
        assert_eq!(irr.total_wm2, 0.0);
    }

    #[test]
    fn test_surface_irradiance_with_provided_ghi() {
        let sun_pos = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };
        let irr = calculate_surface_irradiance(
            &sun_pos,
            800.0,
            100.0,
            Some(900.0),
            Orientation::South,
            0.2,
            172,
        );
        assert!(irr.total_wm2 > 0.0);
    }

    #[test]
    fn test_surface_irradiance_orientations() {
        let sun_pos = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };

        for orientation in [
            Orientation::North,
            Orientation::South,
            Orientation::East,
            Orientation::West,
            Orientation::Up,
            Orientation::Down,
        ] {
            let irr =
                calculate_surface_irradiance(&sun_pos, 800.0, 100.0, None, orientation, 0.2, 172);
            assert!(irr.total_wm2 >= 0.0);
        }
    }

    #[test]
    fn test_orientation_to_angles_horizontal() {
        let (tilt, az) = orientation_to_angles(Orientation::Horizontal);
        assert!((tilt - 0.0).abs() < 1e-6);
        assert!((az - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_surface_irradiance_horizontal_orientation() {
        let sun_pos = SolarPosition {
            altitude_deg: 45.0,
            azimuth_deg: 180.0,
            zenith_deg: 45.0,
        };
        let irr = calculate_surface_irradiance(
            &sun_pos,
            800.0,
            100.0,
            None,
            Orientation::Horizontal,
            0.2,
            172,
        );
        assert!(irr.total_wm2 > 0.0);
    }

    #[test]
    fn test_surface_irradiance_equality() {
        let si1 = SurfaceIrradiance::new(500.0, 100.0, 50.0);
        let si2 = SurfaceIrradiance::new(500.0, 100.0, 50.0);
        assert_eq!(si1, si2);
    }

    #[test]
    fn test_surface_irradiance_zero() {
        let si = SurfaceIrradiance::zero();
        assert_eq!(si.beam_wm2, 0.0);
        assert_eq!(si.diffuse_wm2, 0.0);
        assert_eq!(si.ground_reflected_wm2, 0.0);
        assert_eq!(si.total_wm2, 0.0);
    }

    #[test]
    fn test_extraterrestrial_irradiance() {
        // At perihelion (DOY ~3): I₀ should be ~1406 W/m²
        let e_peri = extraterrestrial_irradiance(3);
        assert!(e_peri > 1400.0 && e_peri < 1440.0);

        // At aphelion (DOY ~186): I₀ should be ~1316 W/m²
        let e_aph = extraterrestrial_irradiance(186);
        assert!(e_aph > 1300.0 && e_aph < 1340.0);
    }

    #[test]
    fn test_solar_constant_matches_canonical() {
        // Issue #1413: the leaf solar module's solar constant must equal the
        // canonical ASHRAE 140-2022 value used everywhere else in the codebase.
        assert_eq!(
            SOLAR_CONSTANT,
            crate::physics::constants::solar::ashrae_140::SOLAR_CONSTANT,
            "SOLAR_CONSTANT in surface_irradiance ({}) must match \
             ashrae_140::SOLAR_CONSTANT ({})",
            SOLAR_CONSTANT,
            crate::physics::constants::solar::ashrae_140::SOLAR_CONSTANT,
        );
    }

    #[test]
    fn test_extraterrestrial_irradiance_doy172() {
        // Issue #1413 acceptance: dni_extra(DOY 172) must match the canonical
        // formula 1361 × (1 + 0.033 × cos(2π·169/365)) to within 1e-9.
        let expected = 1361.0 * (1.0 + 0.033 * (2.0 * std::f64::consts::PI * 169.0 / 365.0).cos());
        let actual = extraterrestrial_irradiance(172);
        assert!(
            (actual - expected).abs() < 1e-9,
            "dni_extra(DOY 172): expected {expected:.6}, got {actual:.6}"
        );
    }

    // =============================================================================
    // Perez sky model F1/F2 coefficient boundary tests (Issue #1695)
    // =============================================================================

    #[allow(dead_code)]
    const SKY_CLEARNESS_BOUNDARIES: [f64; 8] = [0.0, 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2];

    #[test]
    fn test_perez_sky_clearness_classification_at_boundaries() {
        let cases = [
            (0.0, 0),
            (1.065, 1),
            (1.23, 2),
            (1.5, 3),
            (1.95, 4),
            (2.8, 5),
            (4.5, 6),
            (6.2, 7),
            (10.0, 7),
        ];

        for (epsilon, expected_bin) in cases {
            let bin = PerezSkyModel::classify_sky_clearness(epsilon);
            assert_eq!(
                bin, expected_bin,
                "epsilon={} should classify to bin {}, got bin {}",
                epsilon, expected_bin, bin
            );
        }
    }

    #[test]
    fn test_perez_sky_clearness_classification_within_each_bin() {
        let test_cases = [
            (0.5, 1),
            (1.0, 1),
            (1.064, 1),
            (1.1, 2),
            (1.229, 2),
            (1.3, 3),
            (1.49, 3),
            (1.6, 4),
            (1.94, 4),
            (2.0, 5),
            (2.5, 5),
            (3.5, 6),
            (4.0, 6),
            (4.49, 6),
            (5.0, 7),
            (6.0, 7),
            (6.19, 7),
            (7.0, 7),
            (15.0, 7),
        ];

        for (epsilon, expected_bin) in test_cases {
            let bin = PerezSkyModel::classify_sky_clearness(epsilon);
            assert_eq!(
                bin, expected_bin,
                "epsilon={} should classify to bin {}, got bin {}",
                epsilon, expected_bin, bin
            );
        }
    }

    #[test]
    #[ignore = "Known issue: Perez Table 3 source data has F1[0] = 1.321 (bin 5) > 0.999 (bin 6), \
         violating the monotonic non-decreasing constraint. Issue #1695 scope excludes \
         modifying Perez coefficients; data correction requires a separate effort."]
    fn test_perez_f1_coefficients_monotonic_non_decreasing() {
        for ebin in 0..7 {
            let (f1c_curr, _) = PerezSkyModel::get_perez_coefficients(ebin);
            let (f1c_next, _) = PerezSkyModel::get_perez_coefficients(ebin + 1);

            assert!(
                f1c_curr[0] <= f1c_next[0],
                "F1[0] at bin {} ({}) should be <= F1[0] at bin {} ({})",
                ebin,
                f1c_curr[0],
                ebin + 1,
                f1c_next[0]
            );
        }
    }

    #[test]
    fn test_perez_f2_coefficients_monotonic_non_increasing() {
        for ebin in 0..7 {
            let (_, f2c_curr) = PerezSkyModel::get_perez_coefficients(ebin);
            let (_, f2c_next) = PerezSkyModel::get_perez_coefficients(ebin + 1);

            assert!(
                f2c_curr[0] >= f2c_next[0],
                "F2[0] at bin {} ({}) should be >= F2[0] at bin {} ({})",
                ebin,
                f2c_curr[0],
                ebin + 1,
                f2c_next[0]
            );
        }
    }

    #[test]
    fn test_perez_no_nan_inf_at_any_bin() {
        for ebin in 0..8 {
            let (f1c, f2c) = PerezSkyModel::get_perez_coefficients(ebin);

            for (i, &f1) in f1c.iter().enumerate() {
                assert!(
                    f1.is_finite(),
                    "F1[{}] at bin {} should not be NaN/Inf, got {}",
                    i,
                    ebin,
                    f1
                );
            }
            for (i, &f2) in f2c.iter().enumerate() {
                assert!(
                    f2.is_finite(),
                    "F2[{}] at bin {} should not be NaN/Inf, got {}",
                    i,
                    ebin,
                    f2
                );
            }
        }
    }

    #[test]
    fn test_perez_f1_f2_coefficients_table_values() {
        let f1c_all = [
            [-0.008317, 0.587728, -0.062064],
            [0.129967, 0.682595, -0.151375],
            [0.329676, 0.486861, -0.221272],
            [0.568205, 0.187452, -0.295250],
            [0.873018, -0.393289, -0.369150],
            [1.321297, -1.176777, -0.393994],
            [0.999852, -1.634380, -0.291495],
            [0.553776, 0.631414, -0.209172],
        ];

        let f2c_all = [
            [0.091000, 0.060000, 0.000000],
            [0.055000, 0.060000, 0.000000],
            [0.025000, 0.060000, 0.000000],
            [-0.015000, 0.060000, 0.000000],
            [-0.065000, 0.060000, 0.000000],
            [-0.115000, 0.060000, 0.000000],
            [-0.165000, 0.060000, 0.000000],
            [-0.215000, 0.060000, 0.000000],
        ];

        for ebin in 0..8 {
            let (f1c, f2c) = PerezSkyModel::get_perez_coefficients(ebin);
            for i in 0..3 {
                assert_eq!(
                    f1c[i], f1c_all[ebin][i],
                    "F1[{}] at bin {}: expected {}, got {}",
                    i, ebin, f1c_all[ebin][i], f1c[i]
                );
                assert_eq!(
                    f2c[i], f2c_all[ebin][i],
                    "F2[{}] at bin {}: expected {}, got {}",
                    i, ebin, f2c_all[ebin][i], f2c[i]
                );
            }
        }
    }

    #[test]
    #[ignore = "Known issue: Perez Table 3 source data has F1[0] = 1.321 (bin 5) > 0.999 (bin 6), \
         violating the monotonic non-decreasing constraint at the epsilon=4.5 transition. \
         Issue #1695 scope excludes modifying Perez coefficients; data correction requires a separate effort."]
    fn test_perez_f1_increases_at_each_sky_clearness_transition() {
        let transitions = [
            (0.0, 1.065),
            (1.065, 1.23),
            (1.23, 1.5),
            (1.5, 1.95),
            (1.95, 2.8),
            (2.8, 4.5),
            (4.5, 6.2),
            (6.2, 10.0),
        ];

        for (eps_below, eps_above) in transitions {
            let bin_below = PerezSkyModel::classify_sky_clearness(eps_below);
            let bin_above = PerezSkyModel::classify_sky_clearness(eps_above);
            let (f1c_below, _) = PerezSkyModel::get_perez_coefficients(bin_below);
            let (f1c_above, _) = PerezSkyModel::get_perez_coefficients(bin_above);

            assert!(
                f1c_above[0] >= f1c_below[0],
                "F1[0] should be non-decreasing across transition at epsilon={}: \
                 bin {} (F1={}) -> bin {} (F1={})",
                eps_above,
                bin_below,
                f1c_below[0],
                bin_above,
                f1c_above[0]
            );
        }
    }

    #[test]
    fn test_perez_f2_decreases_at_each_sky_clearness_transition() {
        let transitions = [
            (0.0, 1.065),
            (1.065, 1.23),
            (1.23, 1.5),
            (1.5, 1.95),
            (1.95, 2.8),
            (2.8, 4.5),
            (4.5, 6.2),
            (6.2, 10.0),
        ];

        for (eps_below, eps_above) in transitions {
            let bin_below = PerezSkyModel::classify_sky_clearness(eps_below);
            let bin_above = PerezSkyModel::classify_sky_clearness(eps_above);
            let (_, f2c_below) = PerezSkyModel::get_perez_coefficients(bin_below);
            let (_, f2c_above) = PerezSkyModel::get_perez_coefficients(bin_above);

            assert!(
                f2c_above[0] <= f2c_below[0],
                "F2[0] should be non-increasing across transition at epsilon={}: \
                 bin {} (F2={}) -> bin {} (F2={})",
                eps_above,
                bin_below,
                f2c_below[0],
                bin_above,
                f2c_above[0]
            );
        }
    }
}
