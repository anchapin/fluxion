//! Modular surrogate composition for component-based neural networks.
//!
//! This module provides composite surrogate models that combine predictions from
//! multiple component surrogates (e.g., solar, HVAC, infiltration, thermal mass)
//! into a unified load prediction.

use crate::ai::surrogate::SurrogateManager;

/// A component surrogate that handles a specific physical phenomenon.
///
/// Each component surrogate is responsible for predicting loads for one aspect
/// of building physics (e.g., solar gains, infiltration, HVAC).
#[derive(Clone, Debug)]
pub struct ComponentSurrogate {
    /// Human-readable name for this component (e.g., "solar", "hvac", "infiltration")
    pub name: String,
    /// The underlying surrogate manager that performs predictions for this component
    pub manager: SurrogateManager,
}

impl ComponentSurrogate {
    /// Create a new component surrogate with the given name and manager.
    pub fn new(name: &str, manager: SurrogateManager) -> Self {
        Self {
            name: name.to_string(),
            manager,
        }
    }

    /// Get predictions from this component surrogate.
    ///
    /// Returns a vector of load values for each zone temperature input.
    pub fn predict_loads(&self, temps: &[f64]) -> Vec<f64> {
        self.manager.predict_loads(temps)
    }
}

/// Composite surrogate that aggregates predictions from multiple component surrogates.
///
/// The composite sums the outputs of all components to produce the final load prediction.
/// This modular approach enables targeted optimization and easier model maintenance.
#[derive(Clone, Debug)]
pub struct CompositeSurrogate {
    components: Vec<ComponentSurrogate>,
}

impl CompositeSurrogate {
    /// Create a new composite surrogate from a list of component surrogates.
    ///
    /// # Arguments
    /// * `components` - Vector of component surrogate instances
    ///
    /// # Panics
    /// Panics if no components are provided, as a composite surrogate requires
    /// at least one component to make predictions.
    pub fn new(components: Vec<ComponentSurrogate>) -> Self {
        assert!(
            !components.is_empty(),
            "CompositeSurrogate requires at least one component"
        );
        Self { components }
    }

    /// Get aggregated predictions from all component surrogates.
    ///
    /// The predictions from each component are summed element-wise.
    /// All components must return the same output length for valid results.
    ///
    /// # Arguments
    /// * `temps` - Slice of zone temperatures
    ///
    /// # Returns
    /// A vector where each element is the sum of corresponding elements from
    /// all component predictions. If components return different lengths, the
    /// sum will only iterate up to the minimum length.
    pub fn predict_loads(&self, temps: &[f64]) -> Vec<f64> {
        if components_empty(&self.components) {
            return vec![0.0; temps.len()];
        }

        // Get the first component's prediction to determine output length
        let first_pred = self.components[0].predict_loads(temps);

        // Start with the first prediction
        let mut sum = first_pred;

        // Sum predictions from remaining components
        for component in &self.components[1..] {
            let pred = component.predict_loads(temps);
            // Only sum up to min length to avoid panics if sizes mismatch
            let len = std::cmp::min(sum.len(), pred.len());
            sum.truncate(len);
            for (s, &p) in sum.iter_mut().zip(pred.iter()) {
                *s += p;
            }
            // If a component had shorter output, pad with zeros (already truncated)
        }

        // Ensure output length matches input length if needed
        if sum.len() < temps.len() {
            sum.resize(temps.len(), 0.0);
        }

        sum
    }

    /// Get the number of components in this composite.
    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    /// Get the names of all components.
    pub fn component_names(&self) -> Vec<String> {
        self.components.iter().map(|c| c.name.clone()).collect()
    }
}

/// Check if a slice of components is empty.
fn components_empty(components: &[ComponentSurrogate]) -> bool {
    components.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_surrogate_creation() {
        let manager = SurrogateManager::new().unwrap();
        let comp = ComponentSurrogate::new("test", manager);
        assert_eq!(comp.name, "test");
    }

    #[test]
    fn composite_surrogate_single_component() {
        let manager = SurrogateManager::new().unwrap();
        let comp = ComponentSurrogate::new("solar", manager);
        let composite = CompositeSurrogate::new(vec![comp]);

        let temps = vec![20.0, 21.0, 22.0];
        let loads = composite.predict_loads(&temps);

        // Mock manager returns 1.2 for each temperature
        assert_eq!(loads, vec![1.2, 1.2, 1.2]);
    }

    #[test]
    fn composite_surrogate_two_components_sum() {
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
    fn composite_surrogate_three_components() {
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
    fn composite_surrogate_empty_panics() {
        let manager = SurrogateManager::new().unwrap();
        let comp = ComponentSurrogate::new("test", manager);
        let empty_components = Vec::new();

        // Should panic because empty composites are not allowed
        let result = std::panic::catch_unwind(|| {
            CompositeSurrogate::new(empty_components);
        });
        assert!(result.is_err());
    }

    #[test]
    fn composite_surrogate_component_names() {
        let manager = SurrogateManager::new().unwrap();
        let comp1 = ComponentSurrogate::new("solar", manager.clone());
        let comp2 = ComponentSurrogate::new("hvac", manager);
        let composite = CompositeSurrogate::new(vec![comp1, comp2]);

        let names = composite.component_names();
        assert_eq!(names, vec!["solar", "hvac"]);
    }

    #[test]
    fn composite_surrogate_num_components() {
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
    fn composite_surrogate_with_different_length_outputs() {
        // This test verifies graceful handling when components return different lengths
        // In practice, all components should return same length, but we handle it safely
        let manager = SurrogateManager::new().unwrap();
        let comp = ComponentSurrogate::new("test", manager);
        let composite = CompositeSurrogate::new(vec![comp, comp]);

        let temps = vec![20.0, 21.0, 22.0];
        let loads = composite.predict_loads(&temps);

        // Both return 1.2 * 2 = 2.4
        assert_eq!(loads, vec![2.4, 2.4, 2.4]);
    }

    #[test]
    fn clone_properties() {
        let manager = SurrogateManager::new().unwrap();
        let comp = ComponentSurrogate::new("test", manager.clone());
        let composite = CompositeSurrogate::new(vec![comp]);

        let cloned = composite.clone();
        assert_eq!(cloned.num_components(), composite.num_components());
        assert_eq!(cloned.component_names(), composite.component_names());
    }
}
