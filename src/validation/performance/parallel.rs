// Parallel validation executor for performance testing
use crate::validation::ASHRAE140Case;
use std::sync::{Arc, Mutex};
use std::thread;

pub struct ParallelValidationExecutor {
    results: Arc<Mutex<Vec<(ASHRAE140Case, crate::validation::ValidationResult)>>>,
}

impl ParallelValidationExecutor {
    pub fn new() -> Self {
        Self {
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn execute(&mut self, cases: Vec<ASHRAE140Case>) {
        let results = self.results.clone();

        let handles: Vec<_> = cases
            .into_iter()
            .map(|case| {
                let results = results.clone();
                thread::spawn(move || {
                    let validator = crate::validation::ASHRAE140Validator::new();
                    let validation_result = validator.validate_case(&case);
                    results.lock().unwrap().push((case, validation_result));
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    pub fn get_results(&self) -> Vec<(ASHRAE140Case, crate::validation::ValidationResult)> {
        Arc::clone(&self.results).lock().unwrap().clone()
    }
}
