//! Runtime tracing for wiring validation
//!
//! Provides instrumentation to verify that modules are correctly wired together.
//! Catches issues like solve_timesteps() never calling predict_loads() when use_ai=true.

use std::sync::{Arc, Mutex};

/// Runtime tracer for detecting wiring issues
#[derive(Debug)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wiring_tracer_new_is_empty() {
        let tracer = WiringTracer::new();
        assert!(tracer.get_calls().is_empty());
    }

    #[test]
    fn test_wiring_tracer_record_call() {
        let tracer = WiringTracer::new();
        tracer.record_call("function_a");
        let calls = tracer.get_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0], "function_a");
    }

    #[test]
    fn test_wiring_tracer_record_multiple_calls() {
        let tracer = WiringTracer::new();
        tracer.record_call("func1");
        tracer.record_call("func2");
        tracer.record_call("func3");

        let calls = tracer.get_calls();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0], "func1");
        assert_eq!(calls[1], "func2");
        assert_eq!(calls[2], "func3");
    }

    #[test]
    fn test_wiring_tracer_verify_called_all_present() {
        let tracer = WiringTracer::new();
        tracer.record_call("init");
        tracer.record_call("run");
        tracer.record_call("cleanup");

        assert!(tracer.verify_called(&["init", "run", "cleanup"]));
    }

    #[test]
    fn test_wiring_tracer_verify_called_missing() {
        let tracer = WiringTracer::new();
        tracer.record_call("init");
        tracer.record_call("run");

        assert!(!tracer.verify_called(&["init", "missing"]));
    }

    #[test]
    fn test_wiring_tracer_verify_called_empty_expected() {
        let tracer = WiringTracer::new();
        assert!(tracer.verify_called(&[]));
    }

    #[test]
    fn test_wiring_tracer_clear() {
        let tracer = WiringTracer::new();
        tracer.record_call("func1");
        tracer.record_call("func2");
        assert_eq!(tracer.get_calls().len(), 2);

        tracer.clear();
        assert!(tracer.get_calls().is_empty());
    }

    #[test]
    fn test_wiring_tracer_clone_shares_state() {
        let tracer = WiringTracer::new();
        tracer.record_call("original");

        let cloned = tracer.clone();
        assert!(tracer.verify_called(&["original"]));
        assert!(cloned.verify_called(&["original"]));

        cloned.record_call("from_clone");
        assert!(tracer.verify_called(&["original", "from_clone"]));
    }

    #[test]
    fn test_wiring_tracer_duplicate_calls() {
        let tracer = WiringTracer::new();
        tracer.record_call("func1");
        tracer.record_call("func1");
        tracer.record_call("func1");

        let calls = tracer.get_calls();
        assert_eq!(calls.len(), 3);
        assert!(tracer.verify_called(&["func1"]));
    }

    #[test]
    fn test_wiring_tracer_thread_safety() {
        use std::thread;

        let tracer = Arc::new(WiringTracer::new());
        let mut handles = vec![];

        for i in 0..10 {
            let tracer_clone = tracer.clone();
            let handle = thread::spawn(move || {
                tracer_clone.record_call(&format!("thread_{}", i));
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let calls = tracer.get_calls();
        assert_eq!(calls.len(), 10);
    }
}
