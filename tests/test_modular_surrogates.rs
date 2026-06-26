//! Tests for modular surrogate composition and integration.

use fluxion::ai::modular_surrogate::{ComponentSurrogate, CompositeSurrogate};
use fluxion::ai::surrogate::SurrogateManager;
use rand::{Rng, SeedableRng};

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
        let comp = ComponentSurrogate::new("solar", manager.clone());
        let composite = CompositeSurrogate::new(vec![comp]);

        let temps = vec![20.0, 21.0, 22.0];
        let loads = composite.predict_loads(&temps);

        // Issue #1285: when no model is loaded, the fallback is
        // `analytical_loads` (NOT the historical 1.2 mock constant).
        let expected = manager.analytical_loads(&temps).unwrap();
        assert_eq!(loads, expected);
        // Sanity: must NOT be the deprecated mock constant.
        assert!(loads.iter().any(|&v| (v - 1.2).abs() > 1e-9));
    }

    #[test]
    fn test_composite_surrogate_weighted_sum() {
        let manager1 = SurrogateManager::new().unwrap();
        let manager2 = SurrogateManager::new().unwrap();
        let comp1 = ComponentSurrogate::new("solar", manager1.clone());
        let comp2 = ComponentSurrogate::new("hvac", manager2);

        let weights = vec![0.7, 0.3];
        let composite = CompositeSurrogate::with_weights(vec![comp1, comp2], weights).unwrap();

        let temps = vec![20.0, 21.0, 22.0];
        let loads = composite.predict_loads(&temps);

        // Issue #1285: fallback is `analytical_loads` per component.
        let expected = manager1.analytical_loads(&temps).unwrap();
        assert_eq!(loads, expected);
    }

    #[test]
    fn test_composite_surrogate_default_equal_weights() {
        let manager = SurrogateManager::new().unwrap();
        let comp = ComponentSurrogate::new("test", manager.clone());
        let composite = CompositeSurrogate::new(vec![comp.clone(), comp]);

        let temps = vec![20.0, 21.0, 22.0];
        let loads = composite.predict_loads(&temps);

        // Issue #1285: fallback is `analytical_loads`, not 1.2 mock.
        let expected = manager.analytical_loads(&temps).unwrap();
        assert_eq!(loads, expected);
    }

    #[test]
    fn test_composite_surrogate_weights_must_sum_to_one() {
        let manager = SurrogateManager::new().unwrap();
        let comp1 = ComponentSurrogate::new("solar", manager.clone());
        let comp2 = ComponentSurrogate::new("hvac", manager);

        let result = CompositeSurrogate::with_weights(vec![comp1, comp2], vec![0.5, 0.3]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must sum to 1.0"));
    }

    #[test]
    fn test_composite_surrogate_weights_length_mismatch() {
        let manager = SurrogateManager::new().unwrap();
        let comp1 = ComponentSurrogate::new("solar", manager.clone());
        let comp2 = ComponentSurrogate::new("hvac", manager);

        let result = CompositeSurrogate::with_weights(vec![comp1, comp2], vec![0.5, 0.3, 0.2]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("length mismatch"));
    }

    #[test]
    fn test_composite_surrogate_three_components_equal_weights() {
        let managers: Vec<_> = (0..3).map(|_| SurrogateManager::new().unwrap()).collect();
        let components = managers
            .iter()
            .enumerate()
            .map(|(i, m)| ComponentSurrogate::new(&format!("comp{}", i), m.clone()))
            .collect();
        let composite = CompositeSurrogate::new(components);

        let temps = vec![20.0, 25.0, 30.0];
        let loads = composite.predict_loads(&temps);

        // Issue #1285: fallback is `analytical_loads`, not 1.2 mock.
        let expected = managers[0].analytical_loads(&temps).unwrap();
        assert_eq!(loads, expected);
        for &val in &loads {
            assert!(
                (val - 1.2).abs() > 1e-9,
                "must not return 1.2 mock, got {}",
                val
            );
        }
    }

    #[test]
    fn test_composite_surrogate_three_components_custom_weights() {
        let managers: Vec<_> = (0..3).map(|_| SurrogateManager::new().unwrap()).collect();
        let components = managers
            .iter()
            .enumerate()
            .map(|(i, m)| ComponentSurrogate::new(&format!("comp{}", i), m.clone()))
            .collect();
        let weights = vec![0.5, 0.3, 0.2];
        let composite = CompositeSurrogate::with_weights(components, weights).unwrap();

        let temps = vec![20.0, 25.0, 30.0];
        let loads = composite.predict_loads(&temps);

        // Issue #1285: fallback is `analytical_loads`, not 1.2 mock.
        let expected = managers[0].analytical_loads(&temps).unwrap();
        assert_eq!(loads, expected);
    }

    #[test]
    #[should_panic(expected = "CompositeSurrogate requires at least one component")]
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
    fn test_predict_with_uncertainty_single_component() {
        let manager = SurrogateManager::new().unwrap();
        let comp = ComponentSurrogate::new("solar", manager);
        let composite = CompositeSurrogate::new(vec![comp]);

        let temps = vec![20.0, 21.0, 22.0];
        let (mean, std) = composite.predict_with_uncertainty(&temps);

        // `predict_with_uncertainty` routes through `predict_loads`
        // (NOT `predict_loads_with_fallback`), so mock managers return
        // the 1.2 constant — Issue #1285 did not change that path.
        assert_eq!(mean, vec![1.2, 1.2, 1.2]);
        assert_eq!(std, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_predict_with_uncertainty_identical_components() {
        let manager1 = SurrogateManager::new().unwrap();
        let manager2 = SurrogateManager::new().unwrap();
        let comp1 = ComponentSurrogate::new("solar", manager1);
        let comp2 = ComponentSurrogate::new("hvac", manager2);
        let composite = CompositeSurrogate::new(vec![comp1, comp2]);

        let temps = vec![20.0, 21.0];
        let (mean, std) = composite.predict_with_uncertainty(&temps);

        // Mock managers → 1.2 weighted average; same as legacy behaviour.
        assert_eq!(mean, vec![1.2, 1.2]);
        assert_eq!(std, vec![0.0, 0.0]);
    }

    #[test]
    fn test_predict_with_confidence_intervals() {
        let manager1 = SurrogateManager::new().unwrap();
        let manager2 = SurrogateManager::new().unwrap();
        let comp1 = ComponentSurrogate::new("solar", manager1);
        let comp2 = ComponentSurrogate::new("hvac", manager2);
        let composite = CompositeSurrogate::new(vec![comp1, comp2]);

        let temps = vec![20.0, 21.0];
        let (mean, lower, upper) = composite.predict_with_confidence(&temps);

        // Mock managers → mean is the 1.2 constant.
        assert_eq!(mean, vec![1.2, 1.2]);
        for i in 0..mean.len() {
            assert!(lower[i] <= mean[i], "lower bound should be <= mean");
            assert!(upper[i] >= mean[i], "upper bound should be >= mean");
        }
    }

    #[test]
    fn test_component_confidence_scores() {
        let manager1 = SurrogateManager::new().unwrap();
        let manager2 = SurrogateManager::new().unwrap();
        let comp1 = ComponentSurrogate::new("solar", manager1);
        let comp2 = ComponentSurrogate::new("hvac", manager2);
        let composite = CompositeSurrogate::new(vec![comp1, comp2]);

        let temps = vec![20.0, 21.0];
        let scores = composite.component_confidence_scores(&temps);

        assert_eq!(scores.len(), 2);
        let sum: f64 = scores.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "scores should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_component_confidence_scores_single_component() {
        let manager = SurrogateManager::new().unwrap();
        let comp = ComponentSurrogate::new("solar", manager);
        let composite = CompositeSurrogate::new(vec![comp]);

        let temps = vec![20.0, 21.0];
        let scores = composite.component_confidence_scores(&temps);

        assert_eq!(scores.len(), 1);
        assert!((scores[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_predict_loads_with_fallback() {
        let manager1 = SurrogateManager::new().unwrap();
        let manager2 = SurrogateManager::new().unwrap();
        let comp1 = ComponentSurrogate::new("solar", manager1.clone());
        let comp2 = ComponentSurrogate::new("hvac", manager2);
        let composite = CompositeSurrogate::new(vec![comp1, comp2]);

        let temps = vec![20.0, 21.0];
        let result = composite.predict_loads_with_fallback(&temps);

        assert!(result.is_ok());
        // Issue #1285: fallback is `analytical_loads`, not 1.2 mock.
        let expected = manager1.analytical_loads(&temps).unwrap();
        assert_eq!(result.unwrap(), expected);
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
        let manager1 = SurrogateManager::new().unwrap();
        let manager2 = SurrogateManager::new().unwrap();
        let comp1 = ComponentSurrogate::new("comp1", manager1.clone());
        let comp2 = ComponentSurrogate::new("comp2", manager2);
        let composite = CompositeSurrogate::new(vec![comp1, comp2]);

        let mut manager = SurrogateManager::new().unwrap();
        manager.composite = Some(composite);

        let temps = vec![20.0, 22.0];
        let loads = manager.predict_loads(&temps);

        // Issue #1285: composite predict_loads routes through
        // predict_loads_with_fallback, which now uses analytical_loads.
        let expected = manager1.analytical_loads(&temps).unwrap();
        assert_eq!(loads, expected);
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
        let comp1 = ComponentSurrogate::new("a", manager1.clone());
        let comp2 = ComponentSurrogate::new("b", manager2);
        let composite = CompositeSurrogate::new(vec![comp1, comp2]);

        let mut manager = SurrogateManager::new().unwrap();
        manager.composite = Some(composite);

        let batch = vec![vec![20.0, 21.0], vec![22.0, 23.0]];
        let results = manager.predict_loads_batched(&batch);

        assert_eq!(results.len(), 2);
        // Issue #1285: composite → predict_loads_with_fallback →
        // analytical_loads, not 1.2 mock.
        let exp0 = manager1.analytical_loads(&batch[0]).unwrap();
        let exp1 = manager1.analytical_loads(&batch[1]).unwrap();
        assert_eq!(results[0], exp0);
        assert_eq!(results[1], exp1);
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
}
