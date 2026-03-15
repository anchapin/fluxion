//! Runtime tracing for wiring validation
//!
//! Provides instrumentation to verify that modules are correctly wired together.
//! Catches issues like solve_timesteps() never calling predict_loads() when use_ai=true.

use std::sync::{Arc, Mutex};

/// Runtime tracer for detecting wiring issues
pub struct WiringTracer {
    calls: Arc<Mutex<Vec<String>>>,
}

impl WiringTracer {
    /// Create a new wiring tracer
    pub fn new() -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Record a function call
    pub fn record_call(&self, name: &str) {
        self.calls.lock().unwrap().push(name.to_string());
    }

    /// Verify that expected functions were called
    pub fn verify_called(&self, expected: &[&str]) -> bool {
        let calls = self.calls.lock().unwrap();
        expected.iter().all(|exp| calls.contains(&exp.to_string()))
    }

    /// Get all recorded calls
    pub fn get_calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }

    /// Clear recorded calls
    pub fn clear(&self) {
        self.calls.lock().unwrap().clear();
    }
}

impl Clone for WiringTracer {
    fn clone(&self) -> Self {
        Self {
            calls: Arc::clone(&self.calls),
        }
    }
}
