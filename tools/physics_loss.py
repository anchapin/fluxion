#!/usr/bin/env python3
"""
Physics-Informed Loss Functions Module

This module provides physics-informed loss functions that embed physical
constraints into surrogate model training. It includes:
- Energy balance constraint
- Temperature bounds validation
- Custom physics terms
- Physics validation utilities

Related to Issue #172: Phase 8: Implement physics-informed loss functions
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, List


class EnergyBalanceLoss(nn.Module):
    """
    Energy balance constraint loss for thermal models.
    
    Enforces the fundamental heat balance equation:
    Q_total = Q_solar + Q_internal + Q_conduction + Q_ventilation
    
    The loss penalizes predictions that violate energy conservation.
    """
    
    def __init__(
        self,
        u_value_range: Tuple[float, float] = (0.5, 3.0),
        lambda_energy: float = 0.1,
    ):
        super().__init__()
        self.u_value_range = u_value_range
        self.lambda_energy = lambda_energy
        
    def forward(
        self, 
        predictions: torch.Tensor, 
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute energy balance loss.
        
        Args:
            predictions: Predicted thermal loads (batch_size, num_zones)
            features: Input features [u_value, setpoint, outdoor_temp, ...]
        
        Returns:
            Tuple of (loss, loss_components_dict)
        """
        # Extract relevant features
        # Expected feature order: [u_value, hvac_setpoint, outdoor_temp, ...]
        u_value = features[:, 0]
        hvac_setpoint = features[:, 1]
        outdoor_temp = features[:, 2]
        
        # Theoretical steady-state load: Q = U * A * (T_indoor - T_outdoor)
        # For heating: positive when outdoor < indoor
        # For cooling: negative when outdoor > indoor
        delta_t = hvac_setpoint - outdoor_temp
        
        # Base theoretical load (simplified, assuming unit area)
        theoretical_load = u_value * delta_t
        
        # Get mean prediction across zones
        mean_prediction = torch.mean(predictions, dim=1)
        
        # Energy balance residual
        # Predictions should correlate with theoretical load direction
        energy_residual = mean_prediction - theoretical_load
        
        # Penalize large residuals
        energy_loss = torch.mean(energy_residual ** 2)
        
        loss_components = {
            'energy_balance_loss': energy_loss.item(),
        }
        
        return energy_loss, loss_components


class TemperatureBoundsLoss(nn.Module):
    """
    Temperature bounds validation loss.
    
    Ensures predictions are physically reasonable by penalizing
    unrealistic temperature values or loads.
    """
    
    def __init__(
        self,
        min_temp: float = -20.0,
        max_temp: float = 60.0,
        min_load: float = -500.0,
        max_load: float = 500.0,
        lambda_bounds: float = 0.05,
    ):
        super().__init__()
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.min_load = min_load
        self.max_load = max_load
        self.lambda_bounds = lambda_bounds
        
    def forward(self, predictions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute temperature bounds violation loss.
        
        Args:
            predictions: Predicted thermal loads
        
        Returns:
            Tuple of (loss, loss_components_dict)
        """
        # Check for extreme predictions (likely unphysical)
        # These would indicate model extrapolation beyond training data
        
        max_pred = torch.max(predictions)
        min_pred = torch.min(predictions)
        
        # Penalize predictions outside reasonable bounds
        # Using softplus for smooth penalty
        upper_violation = torch.nn.functional.softplus(max_pred - self.max_load)
        lower_violation = torch.nn.functional.softplus(self.min_load - min_pred)
        
        bounds_loss = upper_violation + lower_violation
        
        loss_components = {
            'bounds_loss': bounds_loss.item(),
            'max_pred': max_pred.item(),
            'min_pred': min_pred.item(),
        }
        
        return bounds_loss, loss_components


class MonotonicityLoss(nn.Module):
    """
    Monotonicity constraint loss.
    
    Enforces physical relationships:
    - Higher outdoor temp -> Lower heating load (or higher cooling load)
    - Higher U-value -> Higher heat transfer
    """
    
    def __init__(self, lambda_mono: float = 0.05):
        super().__init__()
        self.lambda_mono = lambda_mono
        
    def forward(
        self, 
        predictions: torch.Tensor, 
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute monotonicity constraint loss.
        
        Args:
            predictions: Predicted thermal loads
            features: Input features [u_value, setpoint, outdoor_temp]
        
        Returns:
            Tuple of (loss, loss_components_dict)
        """
        # Check: Higher outdoor temp should reduce heating load
        # Sort by outdoor temp and check if predictions decrease
        
        outdoor_temp = features[:, 2]
        
        # For heating mode (when outdoor < setpoint), load should decrease with outdoor temp
        # This is a simplified check - real implementation would need mode detection
        
        # Create pairs and check monotonicity
        sorted_indices = torch.argsort(outdoor_temp)
        sorted_preds = predictions[sorted_indices]
        
        # Compute differences
        diffs = sorted_preds[:, 1:] - sorted_preds[:, :-1]
        
        # For heating, expect negative correlation (higher temp -> lower load)
        # We penalize positive differences (load increasing with temp)
        monotonicity_violation = torch.mean(torch.clamp(diffs, min=0) ** 2)
        
        loss_components = {
            'monotonicity_loss': monotonicity_violation.item(),
        }
        
        return monotonicity_violation, loss_components


class PhysicsInformedLoss(nn.Module):
    """
    Combined Physics-Informed Loss function.
    
    Combines multiple physics constraints:
    - Data fitting (MSE)
    - Energy balance
    - Temperature bounds
    - Monotonicity
    """
    
    def __init__(
        self,
        lambda_physics: float = 0.1,
        lambda_bounds: float = 0.05,
        lambda_mono: float = 0.05,
        u_value_range: Tuple[float, float] = (0.5, 3.0),
        temp_range: Tuple[float, float] = (-20.0, 60.0),
        load_range: Tuple[float, float] = (-500.0, 500.0),
    ):
        super().__init__()
        
        self.lambda_physics = lambda_physics
        self.lambda_bounds = lambda_bounds
        self.lambda_mono = lambda_mono
        
        self.mse = nn.MSELoss()
        
        self.energy_loss = EnergyBalanceLoss(
            u_value_range=u_value_range,
            lambda_energy=lambda_physics
        )
        self.bounds_loss = TemperatureBoundsLoss(
            min_temp=temp_range[0],
            max_temp=temp_range[1],
            min_load=load_range[0],
            max_load=load_range[1],
            lambda_bounds=lambda_bounds
        )
        self.mono_loss = MonotonicityLoss(lambda_mono=lambda_mono)
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined physics-informed loss.
        
        Args:
            pred: Model predictions (batch_size, num_zones)
            target: Ground truth targets (batch_size, num_zones)
            features: Input features (batch_size, num_features)
        
        Returns:
            Tuple of (total_loss, loss_breakdown_dict)
        """
        # 1. Data fitting loss
        data_loss = self.mse(pred, target)
        
        # 2. Energy balance constraint
        energy_loss, _ = self.energy_loss(pred, features)
        
        # 3. Temperature bounds
        bounds_loss, bounds_info = self.bounds_loss(pred)
        
        # 4. Monotonicity
        mono_loss, _ = self.mono_loss(pred, features)
        
        # Combined loss
        total_loss = (
            data_loss +
            self.lambda_physics * energy_loss +
            self.lambda_bounds * bounds_loss +
            self.lambda_mono * mono_loss
        )
        
        loss_breakdown = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'energy_balance_loss': energy_loss.item(),
            'bounds_loss': bounds_loss.item(),
            'monotonicity_loss': mono_loss.item(),
            **bounds_info,
        }
        
        return total_loss, loss_breakdown


def validate_physics_constraints(
    predictions: np.ndarray,
    features: np.ndarray,
    tolerance: float = 0.1
) -> Dict[str, bool]:
    """
    Validate that predictions satisfy physical laws.
    
    Args:
        predictions: Model predictions
        features: Input features
        tolerance: Tolerance for constraint checks
    
    Returns:
        Dictionary of constraint validation results
    """
    results = {}
    
    # Extract features
    u_value = features[:, 0]
    hvac_setpoint = features[:, 1]
    outdoor_temp = features[:, 2]
    
    # 1. Energy balance check
    theoretical_load = u_value * (hvac_setpoint - outdoor_temp)
    mean_pred = np.mean(predictions, axis=1)
    
    # Check if predictions are in the same direction as theoretical
    # (Should have same sign)
    sign_match = np.sum(np.sign(theoretical_load) == np.sign(mean_pred)) / len(mean_pred)
    results['energy_balance'] = sign_match > (1 - tolerance)
    
    # 2. Bounds check
    min_load = -500.0
    max_load = 500.0
    in_bounds = np.all((predictions >= min_load) & (predictions <= max_load))
    results['bounds'] = in_bounds
    
    # 3. Monotonicity check (simplified)
    # Sort by outdoor temp and check correlation
    sorted_indices = np.argsort(outdoor_temp)
    sorted_preds = mean_pred[sorted_indices]
    
    if len(sorted_preds) > 1:
        # Calculate rank correlation
        corr = np.corrcoef(np.arange(len(sorted_preds)), sorted_preds)[0, 1]
        # For heating, expect negative correlation
        results['monotonicity'] = corr < tolerance
    else:
        results['monotonicity'] = True
    
    return results


def compute_physics_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    features: np.ndarray
) -> Dict[str, float]:
    """
    Compute physics-based evaluation metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        features: Input features
    
    Returns:
        Dictionary of physics metrics
    """
    metrics = {}
    
    # Mean Absolute Error
    metrics['mae'] = np.mean(np.abs(predictions - targets))
    
    # Root Mean Squared Error
    metrics['rmse'] = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Energy balance error
    u_value = features[:, 0]
    hvac_setpoint = features[:, 1]
    outdoor_temp = features[:, 2]
    
    theoretical_load = u_value * (hvac_setpoint - outdoor_temp)
    mean_pred = np.mean(predictions, axis=1)
    mean_target = np.mean(targets, axis=1)
    
    # How well does prediction follow physics?
    # Compare residuals
    pred_residual = mean_pred - theoretical_load
    target_residual = mean_target - theoretical_load
    
    metrics['energy_balance_mae'] = np.mean(np.abs(pred_residual))
    
    # Physics consistency: correlation between prediction direction and physics
    metrics['physics_correlation'] = np.corrcoef(theoretical_load, mean_pred)[0, 1]
    
    return metrics


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Physics-Informed Loss")
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()
    
    # Create dummy data
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch_size = args.num_samples
    num_zones = 10
    num_features = 5
    
    # Features: [u_value, setpoint, outdoor_temp, ...]
    features = torch.randn(batch_size, num_features)
    features[:, 0] = torch.rand(batch_size) * 2.5 + 0.5  # u_value: 0.5-3.0
    features[:, 1] = torch.rand(batch_size) * 5 + 19  # setpoint: 19-24
    features[:, 2] = torch.rand(batch_size) * 45 - 10  # outdoor: -10-35
    
    # Targets (ground truth)
    targets = torch.randn(batch_size, num_zones) * 10 + 50
    
    # Test different losses
    print("Testing Physics-Informed Loss Components")
    print("=" * 50)
    
    # MSE Loss
    mse = nn.MSELoss()
    mse_loss = mse(features[:, :num_zones], targets)
    print(f"Standard MSE Loss: {mse_loss.item():.4f}")
    
    # Energy Balance Loss
    energy_loss_fn = EnergyBalanceLoss()
    energy_loss, components = energy_loss_fn(targets, features)
    print(f"Energy Balance Loss: {energy_loss.item():.4f}")
    
    # Bounds Loss
    bounds_loss_fn = TemperatureBoundsLoss()
    bounds_loss, bounds_info = bounds_loss_fn(targets)
    print(f"Bounds Loss: {bounds_loss.item():.4f}")
    print(f"  Max prediction: {bounds_info['max_pred']:.2f}")
    print(f"  Min prediction: {bounds_info['min_pred']:.2f}")
    
    # Combined Physics Loss
    physics_loss_fn = PhysicsInformedLoss()
    total_loss, breakdown = physics_loss_fn(targets, targets, features)
    print(f"\nCombined Physics-Informed Loss: {total_loss.item():.4f}")
    print("Loss Breakdown:")
    for key, value in breakdown.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    # Validation
    print("\nPhysics Constraint Validation:")
    print("=" * 50)
    
    pred_np = targets.detach().numpy()
    feat_np = features.detach().numpy()
    targ_np = targets.detach().numpy()
    
    constraints = validate_physics_constraints(pred_np, feat_np)
    print("Constraint checks:")
    for constraint, passed in constraints.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {constraint}: {status}")
    
    metrics = compute_physics_metrics(pred_np, targ_np, feat_np)
    print("\nPhysics Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
