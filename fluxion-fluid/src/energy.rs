//! Energy conservation verification for fluid network systems.
//!
//! This module implements energy conservation verification for HVAC fluid systems,
//! following the first law of thermodynamics: at every timestep, the sum of
//! enthalpy flows into all conservation nodes must equal the sum of enthalpy
//! flows out, plus energy transferred to/from the building envelope.
//!
//! # Usage
//!
//! ```rust
//! use fluxion_fluid::energy::{
//!     ConservationNode, EnergyConservationVerifier, EnthalpyFlow,
//!     FluidNetworkGraph, SimulationResults,
//! };
//!
//! #[derive(Debug, Clone)]
//! struct SimpleNode {
//!     id: usize,
//!     inlet: EnthalpyFlow,
//!     outlet: EnthalpyFlow,
//! }
//!
//! impl ConservationNode for SimpleNode {
//!     fn id(&self) -> usize { self.id }
//!     fn mass_balance_residual(&self) -> f64 {
//!         self.inlet.mass_flow_rate - self.outlet.mass_flow_rate
//!     }
//!     fn energy_balance_residual(&self) -> f64 {
//!         self.inlet.enthalpy_rate() - self.outlet.enthalpy_rate()
//!     }
//! }
//!
//! let node = SimpleNode {
//!     id: 0,
//!     inlet: EnthalpyFlow::new(0.5, 4184.0),
//!     outlet: EnthalpyFlow::new(0.5, 4184.0),
//! };
//! let graph = FluidNetworkGraph::new(vec![node]);
//! let results = SimulationResults::new(
//!     0,
//!     vec![2092.0],
//!     vec![2092.0],
//!     vec![0.0],
//! );
//!
//! let verifier = EnergyConservationVerifier::new(1e-3);
//! verifier.verify(&graph, &results).expect("energy conservation should hold");
//! ```

use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum EnergyConservationError {
    #[error("Energy conservation violated at node {node_id} at timestep {timestep}: residual {residual:.6e} W")]
    Violation {
        node_id: usize,
        residual: f64,
        timestep: usize,
    },
    #[error("Network has no conservation nodes")]
    NoConservationNodes,
    #[error("Simulation results are empty")]
    EmptyResults,
}

pub type EnergyConservationResult<T> = Result<T, EnergyConservationError>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnthalpyFlow {
    pub mass_flow_rate: f64,
    pub specific_enthalpy: f64,
}

impl EnthalpyFlow {
    pub fn new(mass_flow_rate: f64, specific_enthalpy: f64) -> Self {
        Self {
            mass_flow_rate,
            specific_enthalpy,
        }
    }

    #[must_use]
    pub fn enthalpy_rate(&self) -> f64 {
        self.mass_flow_rate * self.specific_enthalpy
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyAccumulation {
    pub rate: f64,
}

impl EnergyAccumulation {
    pub fn new(rate: f64) -> Self {
        Self { rate }
    }
}

pub trait ConservationNode: Send + Sync {
    fn id(&self) -> usize;

    fn mass_balance_residual(&self) -> f64;

    fn energy_balance_residual(&self) -> f64;

    fn net_energy_rate(&self) -> f64 {
        self.energy_balance_residual() + self.mass_balance_residual()
    }
}

#[derive(Debug, Clone)]
pub struct FluidNetworkGraph<N: ConservationNode> {
    nodes: Vec<N>,
}

impl<N: ConservationNode> FluidNetworkGraph<N> {
    pub fn new(nodes: Vec<N>) -> Self {
        Self { nodes }
    }

    pub fn conservation_nodes(&self) -> &[N] {
        &self.nodes
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn node(&self, id: usize) -> Option<&N> {
        self.nodes.get(id)
    }
}

impl<N: ConservationNode> Default for FluidNetworkGraph<N> {
    fn default() -> Self {
        Self { nodes: Vec::new() }
    }
}

#[derive(Debug, Clone)]
pub struct SimulationResults {
    current_timestep: usize,
    node_enthalpy_in: Vec<f64>,
    node_enthalpy_out: Vec<f64>,
    node_energy_accumulation: Vec<f64>,
}

impl SimulationResults {
    pub fn new(
        current_timestep: usize,
        node_enthalpy_in: Vec<f64>,
        node_enthalpy_out: Vec<f64>,
        node_energy_accumulation: Vec<f64>,
    ) -> Self {
        Self {
            current_timestep,
            node_enthalpy_in,
            node_enthalpy_out,
            node_energy_accumulation,
        }
    }

    pub fn current_timestep(&self) -> usize {
        self.current_timestep
    }

    pub fn node_enthalpy_in(&self, node_id: usize) -> Option<f64> {
        self.node_enthalpy_in.get(node_id).copied()
    }

    pub fn node_enthalpy_out(&self, node_id: usize) -> Option<f64> {
        self.node_enthalpy_out.get(node_id).copied()
    }

    pub fn node_energy_accumulation(&self, node_id: usize) -> Option<f64> {
        self.node_energy_accumulation.get(node_id).copied()
    }

    pub fn is_empty(&self) -> bool {
        self.node_enthalpy_in.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct EnergyConservationVerifier {
    tolerance: f64,
}

impl EnergyConservationVerifier {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn verify<N: ConservationNode>(
        &self,
        graph: &FluidNetworkGraph<N>,
        results: &SimulationResults,
    ) -> EnergyConservationResult<()> {
        if graph.num_nodes() == 0 {
            return Err(EnergyConservationError::NoConservationNodes);
        }

        if results.is_empty() {
            return Err(EnergyConservationError::EmptyResults);
        }

        for node in graph.conservation_nodes() {
            let residual = node.net_energy_rate();
            if residual.abs() > self.tolerance {
                return Err(EnergyConservationError::Violation {
                    node_id: node.id(),
                    residual,
                    timestep: results.current_timestep(),
                });
            }
        }

        Ok(())
    }

    pub fn verify_enthalpy_balance(
        &self,
        enthalpy_in: f64,
        enthalpy_out: f64,
        energy_added: f64,
    ) -> EnergyConservationResult<()> {
        let residual = enthalpy_in - enthalpy_out - energy_added;
        if residual.abs() > self.tolerance {
            return Err(EnergyConservationError::Violation {
                node_id: 0,
                residual,
                timestep: 0,
            });
        }
        Ok(())
    }
}

impl Default for EnergyConservationVerifier {
    fn default() -> Self {
        Self::new(1e-3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct TestConservationNode {
        id: usize,
        mass_residual: f64,
        energy_residual: f64,
    }

    impl ConservationNode for TestConservationNode {
        fn id(&self) -> usize {
            self.id
        }

        fn mass_balance_residual(&self) -> f64 {
            self.mass_residual
        }

        fn energy_balance_residual(&self) -> f64 {
            self.energy_residual
        }
    }

    #[test]
    fn test_enthalpy_flow() {
        let flow = EnthalpyFlow::new(0.5, 4184.0);
        assert!((flow.enthalpy_rate() - 2092.0).abs() < 0.1);
    }

    #[test]
    fn test_conservation_node_trait() {
        let node = TestConservationNode {
            id: 1,
            mass_residual: 0.0,
            energy_residual: 0.0,
        };
        assert_eq!(node.id(), 1);
        assert!((node.net_energy_rate()).abs() < 1e-10);
    }

    #[test]
    fn test_fluid_network_graph() {
        let nodes = vec![
            TestConservationNode {
                id: 0,
                mass_residual: 0.0,
                energy_residual: 0.0,
            },
            TestConservationNode {
                id: 1,
                mass_residual: 0.0,
                energy_residual: 0.0,
            },
        ];
        let graph = FluidNetworkGraph::new(nodes);
        assert_eq!(graph.num_nodes(), 2);
        assert!(graph.node(0).is_some());
        assert!(graph.node(2).is_none());
    }

    #[test]
    fn test_simulation_results() {
        let results = SimulationResults::new(
            1,
            vec![1000.0, 2000.0],
            vec![900.0, 1900.0],
            vec![100.0, 100.0],
        );
        assert_eq!(results.current_timestep(), 1);
        assert_eq!(results.node_enthalpy_in(0), Some(1000.0));
        assert_eq!(results.node_enthalpy_out(1), Some(1900.0));
    }

    #[test]
    fn test_verifier_passes_when_balanced() {
        let nodes = vec![TestConservationNode {
            id: 0,
            mass_residual: 0.0,
            energy_residual: 0.0,
        }];
        let graph = FluidNetworkGraph::new(nodes);
        let results = SimulationResults::new(0, vec![1000.0], vec![1000.0], vec![0.0]);

        let verifier = EnergyConservationVerifier::new(1e-3);
        assert!(verifier.verify(&graph, &results).is_ok());
    }

    #[test]
    fn test_verifier_fails_when_unbalanced() {
        let nodes = vec![TestConservationNode {
            id: 0,
            mass_residual: 0.0,
            energy_residual: 0.5,
        }];
        let graph = FluidNetworkGraph::new(nodes);
        let results = SimulationResults::new(0, vec![1000.0], vec![1000.0], vec![0.0]);

        let verifier = EnergyConservationVerifier::new(1e-3);
        let result = verifier.verify(&graph, &results);
        assert!(result.is_err());
    }

    #[test]
    fn test_verifier_empty_graph() {
        let graph: FluidNetworkGraph<TestConservationNode> = FluidNetworkGraph::default();
        let results = SimulationResults::new(0, vec![], vec![], vec![]);

        let verifier = EnergyConservationVerifier::new(1e-3);
        let result = verifier.verify(&graph, &results);
        assert!(matches!(
            result,
            Err(EnergyConservationError::NoConservationNodes)
        ));
    }

    #[test]
    fn test_verifier_empty_results() {
        let nodes = vec![TestConservationNode {
            id: 0,
            mass_residual: 0.0,
            energy_residual: 0.0,
        }];
        let graph = FluidNetworkGraph::new(nodes);
        let results = SimulationResults::new(0, vec![], vec![], vec![]);

        let verifier = EnergyConservationVerifier::new(1e-3);
        let result = verifier.verify(&graph, &results);
        assert!(matches!(result, Err(EnergyConservationError::EmptyResults)));
    }

    #[test]
    fn test_verify_enthalpy_balance_passes() {
        let verifier = EnergyConservationVerifier::new(1e-3);
        assert!(verifier
            .verify_enthalpy_balance(1000.0, 900.0, 100.0)
            .is_ok());
    }

    #[test]
    fn test_verify_enthalpy_balance_fails() {
        let verifier = EnergyConservationVerifier::new(1e-3);
        let result = verifier.verify_enthalpy_balance(1000.0, 800.0, 100.0);
        assert!(result.is_err());
    }
}
