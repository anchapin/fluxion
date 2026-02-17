//! Risk-Aware Optimization Example
//!
//! This example demonstrates how to use uncertainty quantification for risk-aware
//! decision making in building energy optimization using the Rust API.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example risk_aware_example
//! ```
//!
//! Related Issue: #176 - Phase 7: Create risk-aware optimization examples

use fluxion::ai::surrogate::{PredictionWithUncertainty, SurrogateManager};

/// Risk tolerance levels for optimization
#[derive(Debug, Clone, Copy)]
pub enum RiskPreference {
    /// High safety margin, lower energy efficiency
    Conservative,
    /// Balanced between safety and efficiency
    Balanced,
    /// Higher risk, maximum energy efficiency
    Aggressive,
}

impl RiskPreference {
    /// Get heating setpoint based on preference
    pub fn heating_setpoint(&self) -> f64 {
        match self {
            RiskPreference::Conservative => 21.0,
            RiskPreference::Balanced => 20.0,
            RiskPreference::Aggressive => 18.0,
        }
    }

    /// Get cooling setpoint based on preference
    pub fn cooling_setpoint(&self) -> f64 {
        match self {
            RiskPreference::Conservative => 25.0,
            RiskPreference::Balanced => 26.0,
            RiskPreference::Aggressive => 28.0,
        }
    }

    /// Get acceptable probability of exceeding target
    pub fn max_exceed_probability(&self) -> f64 {
        match self {
            RiskPreference::Conservative => 0.05, // 5%
            RiskPreference::Balanced => 0.15,     // 15%
            RiskPreference::Aggressive => 0.25,   // 25%
        }
    }
}

/// Risk assessment result
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    /// Probability of exceeding target load
    pub probability_exceed: f64,
    /// Expected shortfall if target is exceeded
    pub expected_shortfall: f64,
    /// Risk level classification
    pub risk_level: RiskLevel,
    /// Whether risk is within tolerance
    pub within_tolerance: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

impl RiskAssessment {
    /// Create a new risk assessment
    pub fn new(probability_exceed: f64, expected_shortfall: f64, tolerance: f64) -> Self {
        let risk_level = if probability_exceed < 0.05 {
            RiskLevel::Low
        } else if probability_exceed < 0.25 {
            RiskLevel::Medium
        } else {
            RiskLevel::High
        };

        let within_tolerance = probability_exceed <= tolerance;

        RiskAssessment {
            probability_exceed,
            expected_shortfall,
            risk_level,
            within_tolerance,
        }
    }

    /// Get risk level as string
    pub fn risk_level_str(&self) -> &str {
        match self.risk_level {
            RiskLevel::Low => "LOW",
            RiskLevel::Medium => "MEDIUM",
            RiskLevel::High => "HIGH",
        }
    }
}

/// Risk-aware optimizer for building energy decisions
pub struct RiskAwareOptimizer {
    /// Optional surrogate model
    model: Option<SurrogateManager>,
    /// Risk preference
    preference: RiskPreference,
}

impl RiskAwareOptimizer {
    /// Create a new risk-aware optimizer
    pub fn new(model: Option<SurrogateManager>, preference: RiskPreference) -> Self {
        RiskAwareOptimizer { model, preference }
    }

    /// Predict with confidence intervals
    pub fn predict_with_confidence(&self, temperatures: &[f64]) -> PredictionWithUncertainty {
        if let Some(ref model) = self.model {
            // Use actual model for predictions
            model.predict_with_uncertainty(temperatures, 100, 0.5)
        } else {
            // Mock predictions for demonstration
            Self::mock_predictions(temperatures)
        }
    }

    /// Mock predictions for demonstration
    fn mock_predictions(temperatures: &[f64]) -> PredictionWithUncertainty {
        let mean: Vec<f64> = temperatures.iter().map(|t| t * 0.5 + 10.0).collect();
        let std: Vec<f64> = temperatures.iter().map(|t| t.abs() * 0.1).collect();
        PredictionWithUncertainty::new(mean, std)
    }

    /// Evaluate risk of not meeting target
    pub fn evaluate_risk(
        &self,
        predictions: &PredictionWithUncertainty,
        target_load: f64,
    ) -> RiskAssessment {
        let prob_exceed = self.calculate_exceed_probability(predictions, target_load);
        let expected_shortfall = self.calculate_expected_shortfall(predictions, target_load);

        RiskAssessment::new(
            prob_exceed,
            expected_shortfall,
            self.preference.max_exceed_probability(),
        )
    }

    /// Calculate probability of exceeding target using normal CDF approximation
    fn calculate_exceed_probability(
        &self,
        predictions: &PredictionWithUncertainty,
        target: f64,
    ) -> f64 {
        // Calculate z-score for each output
        let mut total_prob = 0.0;
        let n = predictions.mean.len();

        for i in 0..n {
            let mean = predictions.mean[i];
            let std = predictions.std[i];

            if std > 0.0 {
                let z = (target - mean) / std;
                // Approximate normal CDF using error function
                let p = 0.5 * (1.0 + Self::erf(z / std::f64::consts::SQRT_2));
                total_prob += p;
            }
        }

        if n > 0 {
            total_prob / n as f64
        } else {
            0.5
        }
    }

    /// Approximate error function
    fn erf(x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Calculate expected shortfall
    fn calculate_expected_shortfall(
        &self,
        predictions: &PredictionWithUncertainty,
        target: f64,
    ) -> f64 {
        let mut total_shortfall = 0.0;
        let n = predictions.mean.len();

        for i in 0..n {
            let mean = predictions.mean[i];
            if mean > target {
                total_shortfall += mean - target;
            }
        }

        if n > 0 {
            total_shortfall / n as f64
        } else {
            0.0
        }
    }

    /// Optimize HVAC setpoints with risk awareness
    pub fn optimize_setpoints(&self, current_temps: &[f64]) -> SetpointRecommendation {
        let predictions = self.predict_with_confidence(current_temps);

        // Evaluate different setpoint options
        let scenarios = vec![
            SetpointScenario {
                name: "Conservative".to_string(),
                heating: RiskPreference::Conservative.heating_setpoint(),
                cooling: RiskPreference::Conservative.cooling_setpoint(),
                risk: self.evaluate_risk(
                    &predictions,
                    RiskPreference::Conservative.heating_setpoint() * 10.0,
                ),
            },
            SetpointScenario {
                name: "Balanced".to_string(),
                heating: RiskPreference::Balanced.heating_setpoint(),
                cooling: RiskPreference::Balanced.cooling_setpoint(),
                risk: self.evaluate_risk(
                    &predictions,
                    RiskPreference::Balanced.heating_setpoint() * 10.0,
                ),
            },
            SetpointScenario {
                name: "Aggressive".to_string(),
                heating: RiskPreference::Aggressive.heating_setpoint(),
                cooling: RiskPreference::Aggressive.cooling_setpoint(),
                risk: self.evaluate_risk(
                    &predictions,
                    RiskPreference::Aggressive.heating_setpoint() * 10.0,
                ),
            },
        ];

        // Select best scenario based on preference
        let selected = match self.preference {
            RiskPreference::Conservative => &scenarios[0],
            RiskPreference::Balanced => &scenarios[1],
            RiskPreference::Aggressive => &scenarios[2],
        };

        SetpointRecommendation {
            outdoor_temp: 5.0, // Would be actual outdoor temp
            current_temps: current_temps.to_vec(),
            selected: selected.clone(),
            all_scenarios: scenarios,
        }
    }
}

/// Setpoint optimization scenario
#[derive(Debug, Clone)]
pub struct SetpointScenario {
    pub name: String,
    pub heating: f64,
    pub cooling: f64,
    pub risk: RiskAssessment,
}

/// Setpoint recommendation result
#[derive(Debug, Clone)]
pub struct SetpointRecommendation {
    pub outdoor_temp: f64,
    pub current_temps: Vec<f64>,
    pub selected: SetpointScenario,
    pub all_scenarios: Vec<SetpointScenario>,
}

impl SetpointRecommendation {
    /// Print recommendation to console
    pub fn print(&self) {
        println!("\n=== Risk-Aware Setpoint Recommendation ===");
        println!("Outdoor Temperature: {:.1}°C", self.outdoor_temp);
        println!("Current Temperatures: {:?}", self.current_temps);
        println!("\nRecommended: {}", self.selected.name);
        println!("  Heating Setpoint: {:.1}°C", self.selected.heating);
        println!("  Cooling Setpoint: {:.1}°C", self.selected.cooling);
        println!("  Risk Level: {}", self.selected.risk.risk_level_str());
        println!(
            "  Probability of Exceed: {:.1}%",
            self.selected.risk.probability_exceed * 100.0
        );
        println!(
            "  Within Tolerance: {}",
            self.selected.risk.within_tolerance
        );

        println!("\nAll Scenarios:");
        println!(
            "{:<15} {:>12} {:>12} {:>10} {:>15}",
            "Name", "Heat SP", "Cool SP", "Risk", "Prob Exceed"
        );
        println!("{}", "-".repeat(70));
        for scenario in &self.all_scenarios {
            println!(
                "{:<15} {:>12.1} {:>12.1} {:>10} {:>15.1}%",
                scenario.name,
                scenario.heating,
                scenario.cooling,
                scenario.risk.risk_level_str(),
                scenario.risk.probability_exceed * 100.0
            );
        }
    }
}

/// Demonstrate risk-aware optimization
pub fn demonstrate_risk_aware_optimization() {
    println!("\n{}", "=".repeat(70));
    println!("RISK-AWARE OPTIMIZATION DEMONSTRATION");
    println!("Issue #176: Phase 7 - Risk-aware optimization examples");
    println!("{}", "=".repeat(70));

    // Create optimizer with mock model
    let optimizer = RiskAwareOptimizer::new(None, RiskPreference::Balanced);

    // Example zone temperatures
    let zone_temps = vec![20.0, 21.0, 19.5, 22.0, 20.5];

    println!("\nScenario:");
    println!("  Zone temperatures: {:?}", zone_temps);

    // Step 1: Get predictions with uncertainty
    println!("\n--- Step 1: Get Predictions with Uncertainty ---");
    let predictions = optimizer.predict_with_confidence(&zone_temps);

    println!("\nPrediction Results:");
    println!("  Mean loads: {:?}", predictions.mean);
    println!("  Std dev: {:?}", predictions.std);
    println!(
        "  95% CI: [{}, {}]",
        predictions
            .lower_bound
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
            .join(", "),
        predictions
            .upper_bound
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Step 2: Evaluate risk
    println!("\n--- Step 2: Risk Evaluation ---");
    let target_load = 50.0;
    let risk = optimizer.evaluate_risk(&predictions, target_load);

    println!("\nTarget Load: {:.1} W/m²", target_load);
    println!(
        "Probability of Exceed: {:.1}%",
        risk.probability_exceed * 100.0
    );
    println!("Expected Shortfall: {:.2} W/m²", risk.expected_shortfall);
    println!("Risk Level: {}", risk.risk_level_str());
    println!("Within Tolerance: {}", risk.within_tolerance);

    // Step 3: Optimize setpoints
    println!("\n--- Step 3: Risk-Aware Setpoint Optimization ---");

    for preference in [
        RiskPreference::Conservative,
        RiskPreference::Balanced,
        RiskPreference::Aggressive,
    ] {
        println!("\n--- {:?} Preference ---", preference);
        let opt = RiskAwareOptimizer::new(None, preference);
        let recommendation = opt.optimize_setpoints(&zone_temps);
        recommendation.print();
    }

    println!("\n{}", "=".repeat(70));
    println!("DEMONSTRATION COMPLETE");
    println!("{}", "=".repeat(70));
}

fn main() {
    println!("Risk-Aware Optimization Example");
    println!("Issue #176: Phase 7 - Create risk-aware optimization examples");
    println!();
    println!("This example demonstrates:");
    println!("  1. Uncertainty-aware predictions with confidence intervals");
    println!("  2. Risk evaluation based on prediction uncertainty");
    println!("  3. Risk-aware HVAC setpoint optimization");

    demonstrate_risk_aware_optimization();

    println!("\n{}", "-".repeat(70));
    println!("USAGE INSTRUCTIONS:");
    println!("{}", "-".repeat(70));
    println!(
        "
To use in your own code:

    use fluxion::ai::surrogate::SurrogateManager;
    use examples::risk_aware_example::RiskAwareOptimizer;

    // Load a model
    let model = SurrogateManager::load_onnx(\"model.onnx\").unwrap();

    // Create optimizer with your preferred risk level
    let optimizer = RiskAwareOptimizer::new(Some(model), RiskPreference::Balanced);

    // Get predictions with uncertainty
    let predictions = optimizer.predict_with_confidence(&[20.0, 21.0, 22.0]);

    // Evaluate risk
    let risk = optimizer.evaluate_risk(&predictions, 50.0);

    // Optimize setpoints
    let recommendation = optimizer.optimize_setpoints(&[20.0, 21.0, 22.0]);
    recommendation.print();

Best Practices:
    - Use Conservative preference for critical applications
    - Increase num_samples for more accurate uncertainty estimates
    - Consider both mean prediction and uncertainty in decisions
    - Monitor risk levels over time for operational insights
"
    );
}
