//! Tests for modular surrogate composition and integration.

use fluxion::ai::modular_surrogate::{ComponentSurrogate, CompositeSurrogate};
use fluxion::ai::surrogate::SurrogateManager;
use rand::Rng;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_surrogate_creation() {
        let manager = SurrogateManager::new().unwrap();
        let comp = ComponentSurrogate::new("test", manager);
        assert_eq!(comp.name, "test");
    }

    #[test]
    fn test_composite_surrogate_single_component() {
        let manager = SurrogateManager::new().unwrap();
        let comp = ComponentSurrogate::new("solar", manager);
        let composite = CompositeSurrogate::new(vec![comp]);

        let temps = vec![20.0, 21.0, 22.0];
        let loads = composite.predict_loads(&temps);

        // Mock manager returns 1.2 for each temperature
        assert_eq!(loads, vec![1.2, 1.2, 1.2]);
    }

    #[test]
    fn test_composite_surrogate_two_components_sum() {
        let manager1 = SurrogateManager::new().unwrap();
        let manager2 = SurrogateManager::new().unwrap();
        let comp1 = ComponentSurrogate::new("solar", manager1);
        let comp2 = ComponentSurrogate::new("hvac", manager2);
        let composite = CompositeSurrogate::new(vec![comp1, comp2]);

        let temps = vec![20.0, 21.0, 22.0];
        let loads = composite.predict_loads(&temps);

        // Both mock managers return 1.2, so sum = 2.4
        assert_eq!(loads, vec![2.4, 2.4, 2.4]);
    }

    #[test]
    fn test_composite_surrogate_three_components() {
        let managers: Vec<_> = (0..3).map(|_| SurrogateManager::new().unwrap()).collect();
        let components = managers
            .into_iter()
            .enumerate()
            .map(|(i, m)| ComponentSurrogate::new(&format!("comp{}", i), m))
            .collect();
        let composite = CompositeSurrogate::new(components);

        let temps = vec![20.0, 25.0, 30.0];
        let loads = composite.predict_loads(&temps);

        // 3 * 1.2 = 3.6
        assert_eq!(loads, vec![3.6, 3.6, 3.6]);
    }

    #[test]
    #[should_panic(expected = "assertion failed: !components.is_empty()")]
    fn test_composite_surrogate_empty_panics() {
        let empty_components: Vec<ComponentSurrogate> = Vec::new();
        CompositeSurrogate::new(empty_components);
    }

    #[test]
    fn test_composite_surrogate_component_names() {
        let manager = SurrogateManager::new().unwrap();
        let comp1 = ComponentSurrogate::new("solar", manager.clone());
        let comp2 = ComponentSurrogate::new("hvac", manager);
        let composite = CompositeSurrogate::new(vec![comp1, comp2]);

        let names = composite.component_names();
        assert_eq!(names, vec!["solar", "hvac"]);
    }

    #[test]
    fn test_composite_surrogate_num_components() {
        let manager = SurrogateManager::new().unwrap();
        let comps = vec![
            ComponentSurrogate::new("a", manager.clone()),
            ComponentSurrogate::new("b", manager.clone()),
            ComponentSurrogate::new("c", manager),
        ];
        let composite = CompositeSurrogate::new(comps);
        assert_eq!(composite.num_components(), 3);
    }

    #[test]
    fn test_surrogate_manager_modular_loading() {
        // This test requires actual ONNX model files. Skip if not present.
        let model_paths: Vec<&str> = vec!["models/solar.onnx", "models/hvac.onnx"];
        let models_exist = model_paths.iter().all(|p| std::path::Path::new(p).exists());

        if !models_exist {
            eprintln!("Skipping modular loading test: ONNX model files not found");
            return;
        }

        let component_configs = vec![
            (
                "models/solar.onnx",
                fluxion::ai::surrogate::InferenceBackend::CPU,
            ),
            (
                "models/hvac.onnx",
                fluxion::ai::surrogate::InferenceBackend::CPU,
            ),
        ];

        let manager = SurrogateManager::load_modular(&component_configs).unwrap();
        assert!(manager.composite.is_some());
        let composite = manager.composite.as_ref().unwrap();
        assert_eq!(composite.num_components(), 2);
        let names = composite.component_names();
        assert!(names.contains(&"solar".to_string()));
        assert!(names.contains(&"hvac".to_string()));
    }

    #[test]
    fn test_surrogate_manager_predict_delegates_to_composite() {
        // Create a composite with two mock managers
        let manager1 = SurrogateManager::new().unwrap();
        let manager2 = SurrogateManager::new().unwrap();
        let comp1 = ComponentSurrogate::new("comp1", manager1);
        let comp2 = ComponentSurrogate::new("comp2", manager2);
        let composite = CompositeSurrogate::new(vec![comp1, comp2]);

        // Create a SurrogateManager with the composite
        let mut manager = SurrogateManager::new().unwrap();
        manager.composite = Some(composite);

        let temps = vec![20.0, 22.0];
        let loads = manager.predict_loads(&temps);

        // Should sum two 1.2's = 2.4
        assert_eq!(loads, vec![2.4, 2.4]);
    }

    #[test]
    fn test_surrogate_manager_predict_uses_single_model_when_no_composite() {
        // Manager without composite should use mock loads (model_loaded = false)
        let manager = SurrogateManager::new().unwrap();
        assert!(manager.composite.is_none());

        let temps = vec![20.0, 21.0];
        let loads = manager.predict_loads(&temps);

        assert_eq!(loads, vec![1.2, 1.2]);
    }

    #[test]
    fn test_surrogate_manager_predict_batched_delegates_to_composite() {
        let manager1 = SurrogateManager::new().unwrap();
        let manager2 = SurrogateManager::new().unwrap();
        let comp1 = ComponentSurrogate::new("a", manager1);
        let comp2 = ComponentSurrogate::new("b", manager2);
        let composite = CompositeSurrogate::new(vec![comp1, comp2]);

        let mut manager = SurrogateManager::new().unwrap();
        manager.composite = Some(composite);

        let batch = vec![vec![20.0, 21.0], vec![22.0, 23.0]];
        let results = manager.predict_loads_batched(&batch);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], vec![2.4, 2.4]);
        assert_eq!(results[1], vec![2.4, 2.4]);
    }

    #[test]
    fn test_surrogate_manager_batched_single_model_when_no_composite() {
        let manager = SurrogateManager::new().unwrap();
        let batch = vec![vec![20.0], vec![21.0]];
        let results = manager.predict_loads_batched(&batch);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], vec![1.2]);
        assert_eq!(results[1], vec![1.2]);
    }

    // Helper function to compute mean relative error
    fn mean_relative_error(predictions: &[f64], expected: &[f64]) -> f64 {
        assert_eq!(predictions.len(), expected.len());
        let mut total_rel_error = 0.0;
        for (p, e) in predictions.iter().zip(expected.iter()) {
            let denominator = e.abs().max(1e-6);
            total_rel_error += (p - e).abs() / denominator;
        }
        total_rel_error / predictions.len() as f64
    }

    #[test]
    #[ignore]
    fn test_holdout_accuracy_solar_component() {
        // This test validates that a trained solar component surrogate achieves <5% mean relative error
        // on a holdout dataset. It requires a trained ONNX model at models/solar.onnx.
        // The test is ignored by default; enable when model is available.

        let model_path = "models/solar.onnx";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping holdout accuracy test: {} not found", model_path);
            return;
        }

        // Load the component surrogate via modular loading
        let component_configs = &[(model_path, fluxion::ai::surrogate::InferenceBackend::CPU)];
        let manager = SurrogateManager::load_modular(component_configs).unwrap();

        // Generate synthetic holdout dataset using known physics: load = U * (T_setpoint - T_outdoor) + small noise
        // For simplicity, we generate a small dataset with varying temperatures
        let num_samples = 100;
        let mut rng = rand::thread_rng();
        let mut temps = Vec::new();
        let mut expected_loads = Vec::new();
        for _ in 0..num_samples {
            // Zone temperature uniformly distributed in [15, 30]
            let t_zone: f64 = rng.gen_range(15.0..30.0);
            // Simulate solar component: a simple function, e.g., proportional to (t_zone - 20)
            let load = 0.8 * (t_zone - 20.0) + rng.gen_range(-0.2..0.2);
            temps.push(vec![t_zone]);
            expected_loads.push(load);
        }

        // Run predictions
        let mut predictions = Vec::new();
        for t in &temps {
            let pred = manager.predict_loads(t);
            predictions.push(pred[0]);
        }

        // Compute mean relative error
        let mre = mean_relative_error(&predictions, &expected_loads);
        println!("Solar component holdout MRE: {:.4}", mre);

        assert!(
            mre < 0.05,
            "Mean relative error {:.4} exceeds 5% threshold",
            mre
        );
    }

    #[test]
    #[ignore]
    fn test_holdout_accuracy_hvac_component() {
        // Similar test for HVAC component
        let model_path = "models/hvac.onnx";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping holdout accuracy test: {} not found", model_path);
            return;
        }

        let component_configs = &[(model_path, fluxion::ai::surrogate::InferenceBackend::CPU)];
        let manager = SurrogateManager::load_modular(component_configs).unwrap();

        // HVAC component: heating/cooling load based on difference from setpoint
        let num_samples = 100;
        let mut rng = rand::thread_rng();
        let mut temps = Vec::new();
        let mut expected_loads = Vec::new();
        for _ in 0..num_samples {
            let t_zone: f64 = rng.gen_range(15.0..30.0);
            // Simple HVAC response: negative when above setpoint (cooling), positive when below (heating)
            let load = 1.5 * (21.0 - t_zone) + rng.gen_range(-0.1..0.1);
            temps.push(vec![t_zone]);
            expected_loads.push(load);
        }

        let mut predictions = Vec::new();
        for t in &temps {
            let pred = manager.predict_loads(t);
            predictions.push(pred[0]);
        }

        let mre = mean_relative_error(&predictions, &expected_loads);
        println!("HVAC component holdout MRE: {:.4}", mre);

        assert!(mre < 0.05, "MRE {:.4} exceeds 5%", mre);
    }
}
