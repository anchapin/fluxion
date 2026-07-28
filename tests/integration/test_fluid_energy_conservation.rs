//! Integration tests for fluxion-fluid energy conservation verification.
//!
//! These tests verify that the EnergyConservationVerifier correctly identifies
//! energy conservation violations in fluid network systems.
//!
//! # CI Gate
//!
//! These tests implement the energy conservation CI gate for fluxion-fluid
//! (Issue #2005). The `grep "violated energy conservation" test_output.log`
//! command must return zero matches before merging.

use fluxion_fluid::energy::{
    ConservationNode, EnergyConservationError, EnergyConservationVerifier, EnthalpyFlow,
    FluidNetworkGraph, SimulationResults,
};

#[derive(Debug, Clone)]
struct SimpleNode {
    id: usize,
    inlet_flow: EnthalpyFlow,
    outlet_flow: EnthalpyFlow,
    energy_added: f64,
}

impl SimpleNode {
    fn new(
        id: usize,
        inlet_flow: EnthalpyFlow,
        outlet_flow: EnthalpyFlow,
        energy_added: f64,
    ) -> Self {
        Self {
            id,
            inlet_flow,
            outlet_flow,
            energy_added,
        }
    }
}

impl ConservationNode for SimpleNode {
    fn id(&self) -> usize {
        self.id
    }

    fn mass_balance_residual(&self) -> f64 {
        self.inlet_flow.mass_flow_rate - self.outlet_flow.mass_flow_rate
    }

    fn energy_balance_residual(&self) -> f64 {
        let enthalpy_in = self.inlet_flow.enthalpy_rate();
        let enthalpy_out = self.outlet_flow.enthalpy_rate();
        enthalpy_in + self.energy_added - enthalpy_out
    }
}

fn make_balanced_network() -> (FluidNetworkGraph<SimpleNode>, SimulationResults) {
    let nodes = vec![
        SimpleNode::new(
            0,
            EnthalpyFlow::new(0.5, 4184.0),
            EnthalpyFlow::new(0.5, 4184.0),
            0.0,
        ),
        SimpleNode::new(
            1,
            EnthalpyFlow::new(0.5, 4184.0),
            EnthalpyFlow::new(0.5, 4184.0),
            0.0,
        ),
    ];
    let graph = FluidNetworkGraph::new(nodes);
    let results = SimulationResults::new(
        0,
        vec![2092.0, 2092.0],
        vec![2092.0, 2092.0],
        vec![0.0, 0.0],
    );
    (graph, results)
}

fn make_unbalanced_network() -> (FluidNetworkGraph<SimpleNode>, SimulationResults) {
    let nodes = vec![SimpleNode::new(
        0,
        EnthalpyFlow::new(0.5, 4184.0),
        EnthalpyFlow::new(0.5, 4184.0),
        5.0,
    )];
    let graph = FluidNetworkGraph::new(nodes);
    let results = SimulationResults::new(0, vec![2092.0], vec![2092.0], vec![0.0]);
    (graph, results)
}

#[test]
fn test_fluid_network_energy_conservation_balanced() {
    let (graph, results) = make_balanced_network();
    let verifier = EnergyConservationVerifier::new(1e-3);

    let result = verifier.verify(&graph, &results);
    assert!(
        result.is_ok(),
        "Energy conservation should hold for balanced network, got {:?}",
        result
    );
}

#[test]
fn test_fluid_network_energy_conservation_unbalanced() {
    let (graph, results) = make_unbalanced_network();
    let verifier = EnergyConservationVerifier::new(1e-3);

    let result = verifier.verify(&graph, &results);
    assert!(
        result.is_err(),
        "Energy conservation should be violated for unbalanced network"
    );

    if let Err(EnergyConservationError::Violation {
        node_id,
        residual,
        timestep,
    }) = result
    {
        println!(
            "Case 600: timestep {} violated energy conservation (> {}). Residual: {:.6e} W",
            timestep, 1e-3, residual
        );
        assert_eq!(node_id, 0);
        assert!((residual - 5.0).abs() < 1e-6);
    }
}

#[test]
fn test_fluid_network_multiple_timesteps() {
    let verifier = EnergyConservationVerifier::new(1e-3);

    for hour in 0..24 {
        let (graph, mut results) = make_balanced_network();
        results = SimulationResults::new(
            hour,
            vec![2092.0, 2092.0],
            vec![2092.0, 2092.0],
            vec![0.0, 0.0],
        );

        let result = verifier.verify(&graph, &results);
        assert!(
            result.is_ok(),
            "Energy conservation violated at hour {}: {:?}",
            hour,
            result
        );
    }
}

#[test]
fn test_chilled_water_loop_energy_conservation() {
    let verifier = EnergyConservationVerifier::new(1e-3);

    let nodes = vec![
        SimpleNode::new(
            0,
            EnthalpyFlow::new(0.5, 4184.0),
            EnthalpyFlow::new(0.5, 8368.0),
            2092.0,
        ),
        SimpleNode::new(
            1,
            EnthalpyFlow::new(0.5, 8368.0),
            EnthalpyFlow::new(0.5, 4184.0),
            -2092.0,
        ),
    ];
    let graph = FluidNetworkGraph::new(nodes);

    let results = SimulationResults::new(
        0,
        vec![2092.0, 4184.0],
        vec![4184.0, 2092.0],
        vec![0.0, 0.0],
    );

    let result = verifier.verify(&graph, &results);
    assert!(
        result.is_ok(),
        "Chilled water loop should conserve energy: {:?}",
        result
    );
}

#[test]
fn test_hot_water_loop_energy_conservation() {
    let verifier = EnergyConservationVerifier::new(1e-3);

    let nodes = vec![
        SimpleNode::new(
            0,
            EnthalpyFlow::new(0.3, 4184.0),
            EnthalpyFlow::new(0.3, 83680.0),
            23848.8,
        ),
        SimpleNode::new(
            1,
            EnthalpyFlow::new(0.3, 83680.0),
            EnthalpyFlow::new(0.3, 4184.0),
            -23848.8,
        ),
    ];
    let graph = FluidNetworkGraph::new(nodes);

    let results = SimulationResults::new(
        0,
        vec![1255.2, 25104.0],
        vec![25104.0, 1255.2],
        vec![0.0, 0.0],
    );

    let result = verifier.verify(&graph, &results);
    assert!(
        result.is_ok(),
        "Hot water loop should conserve energy: {:?}",
        result
    );
}

#[test]
fn test_five_zone_office_chw_energy_conservation() {
    let verifier = EnergyConservationVerifier::new(1e-3);

    let nodes = vec![
        SimpleNode::new(
            0,
            EnthalpyFlow::new(0.1, 4184.0),
            EnthalpyFlow::new(0.1, 8368.0),
            418.4,
        ),
        SimpleNode::new(
            1,
            EnthalpyFlow::new(0.1, 4184.0),
            EnthalpyFlow::new(0.1, 8368.0),
            418.4,
        ),
        SimpleNode::new(
            2,
            EnthalpyFlow::new(0.1, 4184.0),
            EnthalpyFlow::new(0.1, 8368.0),
            418.4,
        ),
        SimpleNode::new(
            3,
            EnthalpyFlow::new(0.1, 4184.0),
            EnthalpyFlow::new(0.1, 8368.0),
            418.4,
        ),
        SimpleNode::new(
            4,
            EnthalpyFlow::new(0.1, 4184.0),
            EnthalpyFlow::new(0.1, 8368.0),
            418.4,
        ),
    ];
    let graph = FluidNetworkGraph::new(nodes);

    for hour in 0..8760 {
        let results = SimulationResults::new(
            hour,
            vec![418.4, 418.4, 418.4, 418.4, 418.4],
            vec![836.8, 836.8, 836.8, 836.8, 836.8],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
        );

        let result = verifier.verify(&graph, &results);
        assert!(
            result.is_ok(),
            "Energy conservation violated at hour {}: {:?}",
            hour,
            result
        );
    }
}

#[test]
fn test_enthalpy_balance_direct() {
    let verifier = EnergyConservationVerifier::new(1e-3);

    assert!(verifier
        .verify_enthalpy_balance(1000.0, 900.0, 100.0)
        .is_ok());

    let result = verifier.verify_enthalpy_balance(1000.0, 800.0, 100.0);
    assert!(result.is_err());
}
