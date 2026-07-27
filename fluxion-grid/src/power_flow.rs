//! Basic power flow interface types.
//!
//! This module defines the types for power flow analysis state.
//! Actual solver implementations are not included in this foundation crate.

use serde::{Deserialize, Serialize};

/// Power flow solution state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PowerFlowState {
    /// Converged flag
    pub converged: bool,
    /// Maximum power mismatch (pu)
    pub max_mismatch: f64,
    /// Number of iterations to converge
    pub iterations: u32,
    /// System frequency (Hz)
    pub frequency: f64,
}

impl Default for PowerFlowState {
    fn default() -> Self {
        Self {
            converged: false,
            max_mismatch: f64::MAX,
            iterations: 0,
            frequency: 60.0,
        }
    }
}

impl PowerFlowState {
    /// Create a new converged state.
    pub fn converged(iterations: u32, max_mismatch: f64) -> Self {
        Self {
            converged: true,
            max_mismatch,
            iterations,
            frequency: 60.0,
        }
    }

    /// Create a new failed-to-converge state.
    pub fn not_converged(iterations: u32, max_mismatch: f64) -> Self {
        Self {
            converged: false,
            max_mismatch,
            iterations,
            frequency: 60.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_state() {
        let state = PowerFlowState::default();
        assert!(!state.converged);
        assert_eq!(state.iterations, 0);
    }

    #[test]
    fn test_converged_state() {
        let state = PowerFlowState::converged(5, 1e-6);
        assert!(state.converged);
        assert_eq!(state.iterations, 5);
        assert!(state.max_mismatch < 1e-4);
    }
}
