//! Modular surrogate composition for component-based neural networks.
//!
//! This module provides composite surrogate models that combine predictions from
//! multiple component surrogates (e.g., solar, HVAC, infiltration, thermal mass)
//! into a unified load prediction.

use crate::ai::surrogate::{SurrogateDomain, SurrogateInputs, SurrogateManager};

#[derive(Clone, Debug)]
pub struct ComponentSurrogate {
    pub name: String,
    pub manager: SurrogateManager,
    pub domain: SurrogateDomain,
}

impl ComponentSurrogate {
    pub fn new(name: &str, manager: SurrogateManager) -> Self {
        Self {
            name: name.to_string(),
            manager,
            domain: SurrogateDomain::default_residential(),
        }
    }

    pub fn with_domain(name: &str, manager: SurrogateManager, domain: SurrogateDomain) -> Self {
        Self {
            name: name.to_string(),
            manager,
            domain,
        }
    }

    pub fn predict_loads(&self, temps: &[f64]) -> Vec<f64> {
        self.manager.predict_loads(temps)
    }

    pub fn predict_loads_governed(
        &self,
        temps: &[f64],
        mode: crate::ai::surrogate::SurrogateMode,
    ) -> Result<Vec<f64>, String> {
        self.manager
            .predict_loads_governed(temps, &self.domain, mode)
    }

    pub fn get_domain(&self) -> &SurrogateDomain {
        &self.domain
    }
}

#[derive(Clone, Debug)]
pub struct CompositeSurrogate {
    components: Vec<ComponentSurrogate>,
    weights: Vec<f64>,
    domain: SurrogateDomain,
}

impl CompositeSurrogate {
    pub fn new(components: Vec<ComponentSurrogate>) -> Self {
        assert!(
            !components.is_empty(),
            "CompositeSurrogate requires at least one component"
        );
        let domain = Self::compute_intersection_domain(&components);
        let weights = vec![1.0 / components.len() as f64; components.len()];
        Self {
            components,
            weights,
            domain,
        }
    }

    pub fn with_weights(
        components: Vec<ComponentSurrogate>,
        weights: Vec<f64>,
    ) -> Result<Self, String> {
        if components.is_empty() {
            return Err("CompositeSurrogate requires at least one component".to_string());
        }
        if components.len() != weights.len() {
            return Err(format!(
                "Components ({}) and weights ({}) length mismatch",
                components.len(),
                weights.len()
            ));
        }
        let weight_sum: f64 = weights.iter().sum();
        if (weight_sum - 1.0).abs() > 1e-9 {
            return Err(format!("Weights must sum to 1.0, got {}", weight_sum));
        }
        let domain = Self::compute_intersection_domain(&components);
        Ok(Self {
            components,
            weights,
            domain,
        })
    }

    fn compute_intersection_domain(components: &[ComponentSurrogate]) -> SurrogateDomain {
        if components.is_empty() {
            return SurrogateDomain::default_residential();
        }

        let first_domain = &components[0].domain;
        let mut combined = first_domain.clone();

        for comp in &components[1..] {
            let dom = &comp.domain;
            combined.temp_bounds.0 = combined.temp_bounds.0.max(dom.temp_bounds.0);
            combined.temp_bounds.1 = combined.temp_bounds.1.min(dom.temp_bounds.1);
            combined.zone_temp_bounds.0 = combined.zone_temp_bounds.0.max(dom.zone_temp_bounds.0);
            combined.zone_temp_bounds.1 = combined.zone_temp_bounds.1.min(dom.zone_temp_bounds.1);
            combined.solar_bounds.0 = combined.solar_bounds.0.max(dom.solar_bounds.0);
            combined.solar_bounds.1 = combined.solar_bounds.1.min(dom.solar_bounds.1);
            combined.humidity_bounds.0 = combined.humidity_bounds.0.max(dom.humidity_bounds.0);
            combined.humidity_bounds.1 = combined.humidity_bounds.1.min(dom.humidity_bounds.1);
            combined.occupancy_bounds.0 = combined.occupancy_bounds.0.max(dom.occupancy_bounds.0);
            combined.occupancy_bounds.1 = combined.occupancy_bounds.1.min(dom.occupancy_bounds.1);

            combined
                .climate_zones
                .retain(|z| dom.climate_zones.contains(z));
            combined
                .building_types
                .retain(|b| dom.building_types.contains(b));
        }

        combined
    }

    pub fn predict_loads(&self, temps: &[f64]) -> Vec<f64> {
        self.predict_loads_with_fallback(temps)
            .unwrap_or_else(|_| vec![0.0; temps.len()])
    }

    pub fn predict_loads_with_fallback(&self, temps: &[f64]) -> Result<Vec<f64>, String> {
        if components_empty(&self.components) {
            return Ok(vec![0.0; temps.len()]);
        }

        let predictions: Result<Vec<Vec<f64>>, String> = self
            .components
            .iter()
            .map(|c| {
                c.predict_loads_governed(
                    temps,
                    crate::ai::surrogate::SurrogateMode::NeuralWithFallback,
                )
            })
            .collect();

        let predictions = match predictions {
            Ok(p) => p,
            Err(e) => {
                log::warn!("Component failed, using analytical fallback: {}", e);
                return self.components[0].manager.analytical_loads(temps);
            }
        };

        let num_outputs = predictions.first().map(|p| p.len()).unwrap_or(temps.len());

        let mut weighted_sum = vec![0.0; num_outputs];
        for (pred, &weight) in predictions.iter().zip(self.weights.iter()) {
            for (i, &val) in pred.iter().enumerate().take(num_outputs) {
                weighted_sum[i] += val * weight;
            }
        }

        Ok(weighted_sum)
    }

    pub fn component_confidence_scores(&self, temps: &[f64]) -> Vec<f64> {
        let predictions: Vec<Vec<f64>> = self
            .components
            .iter()
            .map(|c| c.predict_loads(temps))
            .collect();

        if predictions.is_empty() {
            return vec![];
        }

        let num_outputs = predictions[0].len();
        let n = predictions.len();

        let mut means = vec![0.0; num_outputs];
        for pred in &predictions {
            for (i, &val) in pred.iter().enumerate().take(num_outputs) {
                means[i] += val / n as f64;
            }
        }

        let mut scores = vec![0.0; n];
        for (j, pred) in predictions.iter().enumerate() {
            let mut total_deviation = 0.0;
            for (i, &val) in pred.iter().enumerate().take(num_outputs) {
                let deviation = (val - means[i]).powi(2);
                total_deviation += deviation;
            }
            scores[j] = (-total_deviation.sqrt()).exp();
        }

        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            for score in &mut scores {
                *score /= sum;
            }
        }

        scores
    }

    pub fn predict_with_uncertainty(&self, temps: &[f64]) -> (Vec<f64>, Vec<f64>) {
        if components_empty(&self.components) {
            return (vec![0.0; temps.len()], vec![0.0; temps.len()]);
        }

        let predictions: Vec<Vec<f64>> = self
            .components
            .iter()
            .map(|c| c.predict_loads(temps))
            .collect();

        let num_outputs = predictions.first().map(|p| p.len()).unwrap_or(temps.len());

        let mut weighted_sum = vec![0.0; num_outputs];
        for (pred, &weight) in predictions.iter().zip(self.weights.iter()) {
            for (i, &val) in pred.iter().enumerate().take(num_outputs) {
                weighted_sum[i] += val * weight;
            }
        }

        let mut variances = vec![0.0; num_outputs];
        if self.components.len() > 1 {
            for pred in &predictions {
                for (i, &val) in pred.iter().enumerate().take(num_outputs) {
                    let diff = val - weighted_sum[i];
                    variances[i] += diff * diff;
                }
            }
            for var in &mut variances {
                *var /= (self.components.len() - 1).max(1) as f64;
            }
        }

        let std: Vec<f64> = variances.iter().map(|v| v.sqrt()).collect();
        (weighted_sum, std)
    }

    pub fn predict_with_confidence(&self, temps: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mean, std) = self.predict_with_uncertainty(temps);
        let lower: Vec<f64> = mean
            .iter()
            .zip(std.iter())
            .map(|(&m, &s)| m - 2.0 * s)
            .collect();
        let upper: Vec<f64> = mean
            .iter()
            .zip(std.iter())
            .map(|(&m, &s)| m + 2.0 * s)
            .collect();
        (mean, lower, upper)
    }

    pub fn predict_loads_governed(
        &self,
        temps: &[f64],
        mode: crate::ai::surrogate::SurrogateMode,
    ) -> Result<Vec<f64>, String> {
        let inputs = SurrogateInputs::from_temps(temps);
        if !self.domain.is_valid(&inputs) {
            log::warn!(
                "CompositeSurrogate: inputs out of domain bounds. \
                 Temp: {:.1}, Zone: {:.1}, Solar: {:.1}",
                inputs.exterior_temp,
                inputs.zone_temp,
                inputs.solar_rad
            );
            return self.components[0].manager.analytical_loads(temps);
        }

        if mode == crate::ai::surrogate::SurrogateMode::AnalyticalOnly {
            return self.components[0].manager.analytical_loads(temps);
        }

        Ok(self.predict_loads(temps))
    }

    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    pub fn component_names(&self) -> Vec<String> {
        self.components.iter().map(|c| c.name.clone()).collect()
    }

    pub fn get_domain(&self) -> &SurrogateDomain {
        &self.domain
    }

    pub fn is_valid(&self, inputs: &SurrogateInputs) -> bool {
        self.domain.is_valid(inputs)
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
        let path = "tests_tmp_dummy.onnx";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let manager = SurrogateManager::load_onnx(path).unwrap();
        let comp = ComponentSurrogate::new("solar", manager);
        let composite = CompositeSurrogate::new(vec![comp]);

        let temps = vec![20.0, 21.0];
        let loads = composite.predict_loads(&temps);
        assert_eq!(loads.len(), 2);
        assert!(loads[0] > 0.0);
    }

    #[test]
    fn composite_surrogate_two_components_sum() {
        let path = "tests_tmp_dummy.onnx";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let manager1 = SurrogateManager::load_onnx(path).unwrap();
        let manager2 = SurrogateManager::load_onnx(path).unwrap();
        let comp1 = ComponentSurrogate::new("solar", manager1);
        let comp2 = ComponentSurrogate::new("hvac", manager2);
        let composite = CompositeSurrogate::new(vec![comp1, comp2]);

        let temps = vec![20.0, 21.0];
        let loads = composite.predict_loads(&temps);
        assert_eq!(loads.len(), 2);
        assert!(loads[0] > 0.0);
    }

    #[test]
    fn composite_surrogate_three_components() {
        let path = "tests_tmp_dummy.onnx";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let managers: Vec<_> = (0..3)
            .map(|_| SurrogateManager::load_onnx(path).unwrap())
            .collect();
        let components = managers
            .into_iter()
            .enumerate()
            .map(|(i, m)| ComponentSurrogate::new(&format!("comp{}", i), m))
            .collect();
        let composite = CompositeSurrogate::new(components);

        let temps = vec![20.0, 25.0];
        let loads = composite.predict_loads(&temps);
        assert_eq!(loads.len(), 2);
    }

    #[test]
    fn composite_surrogate_empty_panics() {
        // Should panic because empty composites are not allowed
        let result = std::panic::catch_unwind(|| {
            CompositeSurrogate::new(Vec::new());
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
    fn composite_surrogate_with_valid_outputs() {
        let path = "tests_tmp_dummy.onnx";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let manager = SurrogateManager::load_onnx(path).unwrap();
        let comp = ComponentSurrogate::new("test", manager);
        let composite = CompositeSurrogate::new(vec![comp.clone(), comp]);

        let temps = vec![20.0, 21.0];
        let loads = composite.predict_loads(&temps);
        assert_eq!(loads.len(), 2);
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

    #[test]
    fn component_surrogate_with_domain() {
        let manager = SurrogateManager::new().unwrap();
        let domain = SurrogateDomain::default_residential();
        let comp = ComponentSurrogate::with_domain("solar", manager, domain);
        assert_eq!(comp.name, "solar");
        assert!(comp.get_domain().climate_zones.contains(&"4A".to_string()));
    }

    #[test]
    fn composite_surrogate_domain_intersection() {
        let manager = SurrogateManager::new().unwrap();
        let domain1 = SurrogateDomain::default_residential();
        let mut domain2 = SurrogateDomain::default_residential();
        domain2.temp_bounds = (-40.0, 50.0);
        domain2.climate_zones = vec!["4A".to_string(), "7A".to_string()];

        let comp1 = ComponentSurrogate::with_domain("solar", manager.clone(), domain1);
        let comp2 = ComponentSurrogate::with_domain("hvac", manager, domain2);
        let composite = CompositeSurrogate::new(vec![comp1, comp2]);

        let domain = composite.get_domain();
        assert_eq!(domain.temp_bounds.0, -40.0);
        assert_eq!(domain.temp_bounds.1, 50.0);
        assert!(domain.climate_zones.contains(&"4A".to_string()));
        assert!(!domain.climate_zones.contains(&"7A".to_string()));
    }

    #[test]
    fn composite_surrogate_is_valid() {
        let manager = SurrogateManager::new().unwrap();
        let comp = ComponentSurrogate::new("test", manager);
        let composite = CompositeSurrogate::new(vec![comp]);

        let valid_inputs = SurrogateInputs {
            exterior_temp: 20.0,
            zone_temp: 22.0,
            solar_rad: 500.0,
            humidity: 50.0,
            occupancy: 0.1,
            climate_zone: "4A".to_string(),
        };
        assert!(composite.is_valid(&valid_inputs));

        let invalid_inputs = SurrogateInputs {
            exterior_temp: -60.0,
            zone_temp: 22.0,
            solar_rad: 500.0,
            humidity: 50.0,
            occupancy: 0.1,
            climate_zone: "4A".to_string(),
        };
        assert!(!composite.is_valid(&invalid_inputs));
    }

    #[test]
    fn component_surrogate_predict_loads_governed() {
        let manager = SurrogateManager::new().unwrap();
        let comp = ComponentSurrogate::new("test", manager);
        let temps = vec![20.0, 22.0];

        let result = comp.predict_loads_governed(
            &temps,
            crate::ai::surrogate::SurrogateMode::NeuralWithFallback,
        );
        assert!(result.is_ok());
    }
}
