//! Foundational physics constants for Fluxion.
//!
//! This module holds leaf physics constants that both `fluxion_core::construction`
//! and `fluxion::sim::sky_radiation` need. Prior to #2462 these were scattered
//! across `fluxion::physics::constants::*` (a chain the `physics ↔ sim` cycle
//! could not import from) and `fluxion::sim::sky_radiation` (a sim module the
//! physics module could not import from). Hoisting the leaf constants here
//! lets `fluxion_core` keep the cycle broken.
//!
//! # Crate split (Issue #2462 — Phase 2 of the crate split)
//!
//! As of #2462, `STEFAN_BOLTZMANN` moved out of `fluxion::sim::sky_radiation`
//! into this leaf module so that `physics::multi_node_solver` (a leaf
//! physics implementation that uses this constant directly) no longer needs
//! to import from `sim::sky_radiation` to break the cycle.
//!
//! `src/sim/sky_radiation.rs` keeps a `pub use fluxion_core::physics_constants::STEFAN_BOLTZMANN`
//! re-export so existing call sites (`crate::sim::sky_radiation::STEFAN_BOLTZMANN`,
//! `fluxion::sim::sky_radiation::STEFAN_BOLTZMANN`) continue to work.

/// Stefan–Boltzmann constant (W/m²·K⁴).
///
/// **Value:** 5.67 × 10⁻⁸ W/m²·K⁴
///
/// Used in longwave radiation exchange between surfaces and sky (and any other
/// T⁴ radiative coupling). The canonical ASHRAE value is 5.670374419e-8;
/// the 5.67e-8 approximation is universally used in building energy modeling
/// (e.g., `fluxion::sim::sky_radiation::SkyRadiationExchange::radiative_coefficient`)
/// and matches within 0.06 % — well below any meaningful error contribution.
pub const STEFAN_BOLTZMANN: f64 = 5.67e-8;

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip sanity check: 5.67e-8 should be within 1 % of the
    /// canonical ASHRAE / NIST 2018 value of 5.670374419e-8.
    #[test]
    fn test_stefan_boltzmann_within_one_percent_of_nist() {
        const NIST_2018: f64 = 5.670_374_419e-8;
        let rel_err = ((STEFAN_BOLTZMANN - NIST_2018) / NIST_2018).abs();
        assert!(
            rel_err < 0.01,
            "STEFAN_BOLTZMANN {} differs from NIST 2018 {} by {:.4}%",
            STEFAN_BOLTZMANN,
            NIST_2018,
            rel_err * 100.0,
        );
    }
}
