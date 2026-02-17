#!/usr/bin/env python3
"""
Risk-Aware Optimization Example

This example demonstrates how to use uncertainty quantification for risk-aware
decision making in building energy optimization.

It shows:
1. How to obtain predictions with uncertainty bounds
2. How to make risk-aware decisions based on confidence intervals
3. How to visualize uncertainty for better understanding

Requirements:
- A trained surrogate model with uncertainty support
- NumPy and Matplotlib for visualization

Usage:
    python examples/risk_aware_optimization.py

Related Issue: #176 - Phase 7: Create risk-aware optimization examples
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import fluxion
    import numpy as np
except ImportError as e:
    print("Note: This example requires the fluxion Python package to be built.")
    print("Run 'maturin develop' first to build the Python bindings.")
    print()
    print("This example will demonstrate the API concepts:")
    
    # Demo mode with mock data when fluxion is not available
    fluxion = None
    np = None


class RiskAwareOptimizer:
    """
    Risk-aware optimizer for building energy decisions.
    
    Uses uncertainty quantification to make robust decisions that
    account for prediction confidence.
    """
    
    def __init__(self, model=None):
        self.model = model
        self.decision_history = []
    
    def predict_with_confidence(
        self, 
        temperatures: list, 
        num_samples: int = 100
    ) -> dict:
        """
        Get predictions with confidence intervals.
        
        Args:
            temperatures: List of zone temperatures
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with mean, std, lower_bound, upper_bound
        """
        if fluxion is None:
            # Mock predictions for demonstration
            return self._mock_predictions(temperatures)
        
        # Real implementation would use the model's uncertainty API
        # This is a placeholder for the actual implementation
        mean = np.array(temperatures) * 0.5 + 10  # Mock prediction
        std = np.abs(np.array(temperatures)) * 0.1  # 10% uncertainty
        lower = mean - 2 * std
        upper = mean + 2 * std
        
        return {
            'mean': mean.tolist(),
            'std': std.tolist(),
            'lower_bound': lower.tolist(),
            'upper_bound': upper.tolist()
        }
    
    def _mock_predictions(self, temperatures):
        """Generate mock predictions for demonstration."""
        if np is None:
            # Simple mock without numpy
            mean = [t * 0.5 + 10 for t in temperatures]
            std = [abs(t) * 0.1 for t in temperatures]
            lower = [m - 2 * s for m, s in zip(mean, std)]
            upper = [m + 2 * s for m, s in zip(mean, std)]
        else:
            temps = np.array(temperatures)
            mean = temps * 0.5 + 10
            std = np.abs(temps) * 0.1
            lower = mean - 2 * std
            upper = mean + 2 * std
            
            mean = mean.tolist()
            std = std.tolist()
            lower = lower.tolist()
            upper = upper.tolist()
        
        return {
            'mean': mean,
            'std': std,
            'lower_bound': lower,
            'upper_bound': upper
        }
    
    def evaluate_risk(
        self,
        predictions: dict,
        target_load: float,
        risk_tolerance: float = 0.1
    ) -> dict:
        """
        Evaluate risk of not meeting target.
        
        Args:
            predictions: Prediction dict from predict_with_confidence
            target_load: Target thermal load (W/m²)
            risk_tolerance: Acceptable probability of exceeding target
            
        Returns:
            Risk assessment dictionary
        """
        mean = np.array(predictions['mean']) if np else predictions['mean']
        std = np.array(predictions['std']) if np else predictions['std']
        
        # Calculate z-score for target
        if np:
            z_scores = (target_load - mean) / (std + 1e-10)
            
            # Probability of exceeding target (using normal CDF approximation)
            # Using np.erf for the error function
            prob_exceed = 1 - 0.5 * (1 + np.erf(z_scores / np.sqrt(2)))
            
            # Expected shortfall (average amount over target if over)
            expected_shortfall = np.where(
                mean > target_load,
                mean - target_load,
                0
            )
            
            risk_level = []
            for p in prob_exceed:
                if p < 0.05:
                    risk_level.append("LOW")
                elif p < 0.25:
                    risk_level.append("MEDIUM")
                else:
                    risk_level.append("HIGH")
        else:
            # Simple non-numpy version
            prob_exceed = []
            expected_shortfall = []
            risk_level = []
            
            for m, s in zip(mean, std):
                if s > 0:
                    z = (target_load - m) / s
                    # Approximate probability
                    p = max(0, min(1, 0.5 - z * 0.2))
                else:
                    p = 0.5 if m > target_load else 0
                
                prob_exceed.append(p)
                expected_shortfall.append(max(0, m - target_load))
                
                if p < 0.05:
                    risk_level.append("LOW")
                elif p < 0.25:
                    risk_level.append("MEDIUM")
                else:
                    risk_level.append("HIGH")
        
        return {
            'target_load': target_load,
            'probability_exceed': prob_exceed if np is None else prob_exceed.tolist(),
            'expected_shortfall': expected_shortfall if np is None else expected_shortfall.tolist(),
            'risk_level': risk_level,
            'within_tolerance': [p <= risk_tolerance for p in (prob_exceed if np is None else prob_exceed.tolist())]
        }
    
    def optimize_setpoint(
        self,
        outdoor_temp: float,
        current_temps: list,
        risk_preference: str = "balanced"
    ) -> dict:
        """
        Optimize HVAC setpoint with risk awareness.
        
        Args:
            outdoor_temp: Current outdoor temperature
            current_temps: Current zone temperatures
            risk_preference: 'conservative', 'balanced', or 'aggressive'
            
        Returns:
            Optimal setpoint recommendation with risk assessment
        """
        # Setpoint ranges based on preference
        if risk_preference == "conservative":
            heating_setpoint = 21.0
            cooling_setpoint = 25.0
        elif risk_preference == "aggressive":
            heating_setpoint = 18.0
            cooling_setpoint = 28.0
        else:  # balanced
            heating_setpoint = 20.0
            cooling_setpoint = 26.0
        
        # Get predictions with uncertainty
        predictions = self.predict_with_confidence(current_temps)
        
        # Evaluate different setpoint scenarios
        scenarios = []
        
        # Conservative scenario
        conservative_load = self.estimate_load(
            current_temps, heating_setpoint, cooling_setpoint
        )
        conservative_risk = self.evaluate_risk(
            predictions, conservative_load, risk_tolerance=0.1
        )
        scenarios.append({
            'name': 'Conservative',
            'heating_setpoint': heating_setpoint,
            'cooling_setpoint': cooling_setpoint,
            'estimated_load': conservative_load,
            'risk': conservative_risk
        })
        
        # Balanced scenario
        balanced_load = self.estimate_load(
            current_temps, 20.0, 26.0
        )
        balanced_risk = self.evaluate_risk(
            predictions, balanced_load, risk_tolerance=0.15
        )
        scenarios.append({
            'name': 'Balanced',
            'heating_setpoint': 20.0,
            'cooling_setpoint': 26.0,
            'estimated_load': balanced_load,
            'risk': balanced_risk
        })
        
        # Aggressive scenario
        aggressive_load = self.estimate_load(
            current_temps, 18.0, 28.0
        )
        aggressive_risk = self.evaluate_risk(
            predictions, aggressive_load, risk_tolerance=0.25
        )
        scenarios.append({
            'name': 'Aggressive',
            'heating_setpoint': 18.0,
            'cooling_setpoint': 28.0,
            'estimated_load': aggressive_load,
            'risk': aggressive_risk
        })
        
        # Select best scenario based on preference
        if risk_preference == "conservative":
            selected = scenarios[0]
        elif risk_preference == "aggressive":
            selected = scenarios[2]
        else:
            # Balanced: select medium risk
            selected = scenarios[1]
        
        return {
            'outdoor_temp': outdoor_temp,
            'current_temps': current_temps,
            'selected_scenario': selected,
            'all_scenarios': scenarios,
            'recommendation': self._generate_recommendation(selected, risk_preference)
        }
    
    def estimate_load(self, temps, heating_sp, cooling_sp):
        """Estimate thermal load based on setpoints."""
        if np:
            temps = np.array(temps)
            avg_temp = np.mean(temps)
        else:
            avg_temp = sum(temps) / len(temps)
        
        # Simple load estimate
        if avg_temp < heating_sp:
            return (heating_sp - avg_temp) * 10  # Heating load
        elif avg_temp > cooling_sp:
            return (avg_temp - cooling_sp) * 10  # Cooling load
        else:
            return 0  # No load
    
    def _generate_recommendation(self, scenario, preference):
        """Generate human-readable recommendation."""
        name = scenario['name']
        h_sp = scenario['heating_setpoint']
        c_sp = scenario['cooling_setpoint']
        
        rec = f"Recommend {name} setpoints: "
        rec += f"heating={h_sp}C, cooling={c_sp}C. "
        
        risk = scenario['risk']
        if all(risk['within_tolerance']):
            rec += "Risk level is acceptable."
        else:
            rec += "Warning: Some zones have elevated risk."
        
        return rec


def visualize_confidence_intervals(predictions: dict, zone_names: list = None):
    """
    Visualize predictions with confidence intervals.
    
    Args:
        predictions: Dictionary with mean, std, bounds
        zone_names: Optional names for zones
    """
    print("\n" + "="*60)
    print("CONFIDENCE INTERVAL VISUALIZATION")
    print("="*60)
    
    mean = predictions['mean']
    lower = predictions['lower_bound']
    upper = predictions['upper_bound']
    
    n_zones = len(mean)
    if zone_names is None:
        zone_names = [f"Zone {i+1}" for i in range(n_zones)]
    
    print(f"\n{'Zone':<12} {'Mean':>10} {'Lower':>10} {'Upper':>10} {'Uncertainty':>12}")
    print("-" * 60)
    
    for i, name in enumerate(zone_names):
        print(f"{name:<12} {mean[i]:>10.2f} {lower[i]:>10.2f} {upper[i]:>10.2f} "
              f"{upper[i]-lower[i]:>10.2f}")


def demonstrate_risk_analysis():
    """Demonstrate risk-aware optimization workflow."""
    print("\n" + "="*70)
    print("RISK-AWARE OPTIMIZATION DEMONSTRATION")
    print("="*70)
    
    optimizer = RiskAwareOptimizer()
    
    # Example zone temperatures
    zone_temps = [20.0, 21.0, 19.5, 22.0, 20.5]
    outdoor_temp = 5.0
    
    print(f"\nScenario:")
    print(f"  Zone temperatures: {zone_temps}")
    print(f"  Outdoor temperature: {outdoor_temp}C")
    
    # Get predictions with uncertainty
    print("\n--- Step 1: Get Predictions with Uncertainty ---")
    predictions = optimizer.predict_with_confidence(zone_temps, num_samples=100)
    
    print("\nPrediction Results:")
    print(f"  Mean loads: {[f'{m:.2f}' for m in predictions['mean']]}")
    print(f"  Std dev: {[f'{s:.2f}' for s in predictions['std']]}")
    print(f"  95% CI: [{[f'{l:.2f}' for l in predictions['lower_bound']]}, "
          f"{[f'{u:.2f}' for u in predictions['upper_bound']]}]")
    
    # Visualize confidence intervals
    visualize_confidence_intervals(predictions, [f"Zone {i+1}" for i in range(len(zone_temps))])
    
    # Evaluate risk
    print("\n--- Step 2: Risk Evaluation ---")
    target_load = 50.0  # W/m²
    risk_assessment = optimizer.evaluate_risk(predictions, target_load)
    
    print(f"\nTarget load: {target_load} W/m²")
    print(f"\n{'Zone':<10} {'Prob > Target':>15} {'Expected Shortfall':>20} {'Risk':>10}")
    print("-" * 60)
    
    for i in range(len(zone_temps)):
        print(f"Zone {i+1:<4} {risk_assessment['probability_exceed'][i]:>15.1%} "
              f"{risk_assessment['expected_shortfall'][i]:>20.2f} "
              f"{risk_assessment['risk_level'][i]:>10}")
    
    # Optimize setpoints
    print("\n--- Step 3: Risk-Aware Setpoint Optimization ---")
    
    for preference in ["conservative", "balanced", "aggressive"]:
        print(f"\n--- {preference.capitalize()} Preference ---")
        
        result = optimizer.optimize_setpoint(
            outdoor_temp, 
            zone_temps, 
            risk_preference=preference
        )
        
        scenario = result['selected_scenario']
        print(f"  Heating setpoint: {scenario['heating_setpoint']}C")
        print(f"  Cooling setpoint: {scenario['cooling_setpoint']}C")
        print(f"  Estimated load: {scenario['estimated_load']:.2f} W/m²")
        print(f"  {result['recommendation']}")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)


def main():
    """Main entry point."""
    print("="*70)
    print("Risk-Aware Optimization Example")
    print("Issue #176: Phase 7 - Create risk-aware optimization examples")
    print("="*70)
    
    print("\nThis example demonstrates:")
    print("  1. Uncertainty-aware predictions with confidence intervals")
    print("  2. Risk evaluation based on prediction uncertainty")
    print("  3. Risk-aware HVAC setpoint optimization")
    print("  4. Visualization of confidence intervals")
    
    demonstrate_risk_analysis()
    
    print("\n" + "-"*70)
    print("USAGE INSTRUCTIONS:")
    print("-"*70)
    print("""
To use in your own code:

    from examples.risk_aware_optimization import RiskAwareOptimizer
    
    # Create optimizer
    optimizer = RiskAwareOptimizer(model)
    
    # Get predictions with uncertainty
    predictions = optimizer.predict_with_confidence(
        temperatures=[20.0, 21.0, 22.0],
        num_samples=100
    )
    
    # Evaluate risk
    risk = optimizer.evaluate_risk(predictions, target_load=50.0)
    
    # Optimize setpoints
    result = optimizer.optimize_setpoint(
        outdoor_temp=5.0,
        current_temps=[20.0, 21.0, 22.0],
        risk_preference="balanced"
    )
    
    # Visualize
    visualize_confidence_intervals(predictions)
    
Best Practices:
    - Use conservative preference for critical applications
    - Increase num_samples for more accurate uncertainty estimates
    - Consider both mean prediction and uncertainty in decisions
    - Monitor risk levels over time for operational insights
""")


if __name__ == "__main__":
    main()
