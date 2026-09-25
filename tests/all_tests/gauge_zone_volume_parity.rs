//! Zone-volume parity probe (Issue #3962 diagnostics; LIMIT-21 Item 2).
//!
//! The dispatcher derives mechanical-ventilation ACH from
//! `setpoints.zone_volume` (populated from `spec.geometry[zone].volume()`,
//! i.e. width × depth × height), while the gauge zone solver computes
//! `h_vent = ρ·cp·(ACH/3600)·zone_volume` against its own derived volume
//! (`floor_area × ceiling_height`, where `floor_area` is the largest zone
//! surface area and `ceiling_height` is a hardcoded 2.7 m proxy at assembly
//! time). Any mismatch between the two volumes scales `h_vent` by exactly
//! their ratio, over- or under-damping the zone thermal response.
//!
//! This gate pins parity for every registered ASHRAE 140 case.

#![cfg(feature = "gauge-solver")]

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

const CASES: [ASHRAE140Case; 10] = [
    ASHRAE140Case::Case600,
    ASHRAE140Case::Case620,
    ASHRAE140Case::Case630,
    ASHRAE140Case::Case640,
    ASHRAE140Case::Case650,
    ASHRAE140Case::Case900,
    ASHRAE140Case::Case920,
    ASHRAE140Case::Case930,
    ASHRAE140Case::Case940,
    ASHRAE140Case::Case950,
];

#[test]
fn gauge_zone_volume_parity_all_ashrae_cases() {
    for case in CASES {
        let spec = case.spec();
        let model =
            ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
                .unwrap_or_else(|e| panic!("{case:?}: default (Gauge) selector must initialize: {e:?}"));

        let setpoint_volumes = model.0.setpoints.zone_volume.as_ref().to_vec();
        assert!(
            !setpoint_volumes.is_empty(),
            "{case:?}: setpoints.zone_volume must be populated"
        );

        let gauge = model
            .0
            .conduction
            .backend
            .gauge_zone_solver
            .as_ref()
            .unwrap_or_else(|| panic!("{case:?}: Gauge selector must build a gauge zone solver"));
        let gauge_volume = gauge.zone_volume_m3();

        // Diagnostic trail for Issue #3962: the ratio is the exact factor by
        // which h_vent is mis-scaled if the two volumes disagree.
        println!(
            "{case:?}: setpoints.zone_volume[0] = {:.6} m³, gauge zone_volume = {:.6} m³, ratio = {:.9}",
            setpoint_volumes[0],
            gauge_volume,
            gauge_volume / setpoint_volumes[0]
        );

        assert!(
            (gauge_volume - setpoint_volumes[0]).abs() < 1e-9,
            "{case:?}: gauge zone volume {gauge_volume} m³ != setpoints zone volume {} m³ \
             (ratio {:.6}); h_vent is scaled by this mismatch",
            setpoint_volumes[0],
            gauge_volume / setpoint_volumes[0]
        );
    }
}
