#!/usr/bin/env python3
"""
Physics-Informed Loss Functions

This module provides physics-informed loss functions for surrogate training:
- Energy balance constraints
- Temperature bounds validation
- Custom physics-informed loss
- Physics validation tests

Related to Issue #172: Phase 8: Implement physics-informed loss functions
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemperatureBounds:
    """Temperature bounds for validation."""
    
    # Common temperature limits in Celsius
    MIN_INDOOR_TEMP = -30.0  # Very cold
    MAX_INDOOR_TEMP = 50.0   # Very hot
    MIN_OUTDOOR_TEMP = -50.0
    MAX_OUTDOOR_TEMP = 55.0
    
    # Reasonable comfort range
    COMFORT_MIN = 18.0
    COMFORT_MAX = 28.0
    
    @classmethod
    def validate(
        cls,
        temperatures: torch.Tensor,
        min_temp: Optional[float] = None,
        max_temp: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Validate temperature bounds.
        
        Returns:
            - Mask of valid temperatures (True = valid)
            - Dictionary with validation metrics
        """
        min_temp = min_temp or cls.MIN_INDOOR_TEMP
        max_temp = max_temp or cls.MAX_INDOOR_TEMP
        
        valid = (temperatures >= min_temp) & (temperatures <= max_temp)
        
        # Calculate metrics
        below_min = (temperatures < min_temp).float().mean().item()
        above_max = (temperatures > max_temp).float().mean().item()
        
        metrics = {
            "valid_fraction": valid.float().mean().item(),
            "below_min_fraction": below_min,
            "above_max_fraction": above_max,
            "min_temp": temperatures.min().item(),
            "max_temp": temperatures.max().item(),
        }
        
        return valid, metrics


class EnergyBalanceValidator:
    """Validates energy balance constraints."""
    
    def __init__(
        self,
        thermal_mass: float = 1.0,  # kWh/K
        heat_transfer_coeff: float = 10.0,  # W/K
        time_step: float = 1.0,  # hours
    ):
        self.thermal_mass = thermal_mass
        self.heat_transfer_coeff = heat_transfer_coeff
        self.time_step = time_step
    
    def compute_energy_balance_residual(
        self,
        current_temps: torch.Tensor,
        next_temps: torch.Tensor,
        heating_load: torch.Tensor,
        cooling_load: torch.Tensor,
        outdoor_temps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute energy balance residual.
        
        Energy balance: C * (T_next - T) = Q_heat - Q_cool - H * (T - T_outdoor)
        
        Where:
        - C = thermal mass
        - T_next = next temperature
        - T = current temperature
        - Q_heat = heating load
        - Q_cool = cooling load
        - H = heat transfer coefficient
        - T_outdoor = outdoor temperature
        """
        if outdoor_temps is None:
            outdoor_temps = torch.zeros_like(current_temps)
        
        # Expected temperature change due to heat transfer
        # Q = H * A * delta_T, convert to kWh
        heat_transfer = self.heat_transfer_coeff * (current_temps - outdoor_temps) * self.time_step / 1000.0
        
        # Expected temperature change
        # C * delta_T = (Q_heat - Q_cool - Q_transfer) * dt
        energy_input = (heating_load - cooling_load) * self.time_step / 1000.0  # kWh
        expected_delta = (energy_input - heat_transfer) / self.thermal_mass
        
        # Actual delta
        actual_delta = next_temps - current_temps
        
        # Residual
        residual = actual_delta - expected_delta
        
        return residual
    
    def validate_energy_balance(
        self,
        current_temps: torch.Tensor,
        next_temps: torch.Tensor,
        heating_load: torch.Tensor,
        cooling_load: torch.Tensor,
        outdoor_temps: Optional[torch.Tensor] = None,
        tolerance: float = 2.0,  # Kelvin tolerance
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Validate energy balance is satisfied.
        
        Returns:
            - Whether energy balance is satisfied
            - Dictionary with validation metrics
        """
        residual = self.compute_energy_balance_residual(
            current_temps, next_temps, heating_load, cooling_load, outdoor_temps
        )
        
        # Check if residual is within tolerance
        max_residual = torch.max(torch.abs(residual)).item()
        mean_residual = torch.mean(torch.abs(residual)).item()
        satisfied = max_residual <= tolerance
        
        metrics = {
            "max_residual": max_residual,
            "mean_residual": mean_residual,
            "tolerance": tolerance,
            "satisfied": satisfied,
        }
        
        return satisfied, metrics


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function that combines MSE with physics constraints.
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        energy_balance_weight: float = 0.1,
        temperature_bounds_weight: float = 0.1,
        energy_tolerance: float = 2.0,
        min_temp: float = TemperatureBounds.COMFORT_MIN,
        max_temp: float = TemperatureBounds.COMFORT_MAX,
        thermal_mass: float = 1.0,
        heat_transfer_coeff: float = 10.0,
    ):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.energy_balance_weight = energy_balance_weight
        self.temperature_bounds_weight = temperature_bounds_weight
        self.energy_tolerance = energy_tolerance
        
        # Physics validators
        self.temp_bounds = TemperatureBounds()
        self.energy_validator = EnergyBalanceValidator(
            thermal_mass=thermal_mass,
            heat_transfer_coeff=heat_transfer_coeff,
        )
        
        # Register bounds as buffers
        self.register_buffer('min_temp', torch.tensor(min_temp))
        self.register_buffer('max_temp', torch.tensor(max_temp))
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        current_temps: Optional[torch.Tensor] = None,
        heating_load: Optional[torch.Tensor] = None,
        cooling_load: Optional[torch.Tensor] = None,
        outdoor_temps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics-informed loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            current_temps: Current zone temperatures (for energy balance)
            heating_load: Heating energy input
            cooling_load: Cooling energy input
            outdoor_temps: Outdoor temperatures
        
        Returns:
            - Combined loss
            - Dictionary with loss components
        """
        loss_components = {}
        
        # 1. Standard MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        loss_components["mse"] = mse_loss.item()
        
        total_loss = self.mse_weight * mse_loss
        
        # 2. Energy balance constraint (if physics data provided)
        if (current_temps is not None and 
            next_temps := predictions[:, 0] if predictions.ndim > 1 else predictions is not None and
            heating_load is not None and cooling_load is not None):
            
            # Use predictions as next temps if not provided separately
            if predictions.ndim > 1:
                next_temps = predictions[:, 0]
            else:
                next_temps = predictions
            
            residual = self.energy_validator.compute_energy_balance_residual(
                current_temps, next_temps, heating_load, cooling_load, outdoor_temps
            )
            
            # Soft penalty for energy balance violation
            energy_loss = torch.mean(residual ** 2)
            loss_components["energy_balance"] = energy_loss.item()
            
            total_loss = total_loss + self.energy_balance_weight * energy_loss
        
        # 3. Temperature bounds penalty
        if predictions.ndim > 1:
            temps = predictions[:, 0]  # First output is typically temperature
        else:
            temps = predictions
        
        below_min_penalty = F.relu(self.min_temp - temps).mean()
        above_max_penalty = F.relu(temps - self.max_temp).mean()
        temp_penalty = below_min_penalty + above_max_penalty
        loss_components["temperature_bounds"] = temp_penalty.item()
        
        total_loss = total_loss + self.temperature_bounds_weight * temp_penalty
        
        loss_components["total"] = total_loss.item()
        
        return total_loss, loss_components
    
    def validate_predictions(
        self,
        predictions: torch.Tensor,
        current_temps: Optional[torch.Tensor] = None,
        heating_load: Optional[torch.Tensor] = None,
        cooling_load: Optional[torch.Tensor] = None,
        outdoor_temps: Optional[torch.Tensor] = None,
    ) -> Dict[str, any]:
        """
        Validate that predictions satisfy physics constraints.
        
        Returns:
            Dictionary with validation results
        """
        results = {}
        
        # Temperature bounds validation
        if predictions.ndim > 1:
            temps = predictions[:, 0]
        else:
            temps = predictions
        
        valid_temp, temp_metrics = self.temp_bounds.validate(
            temps, self.min_temp.item(), self.max_temp.item()
        )
        results["temperature"] = temp_metrics
        
        # Energy balance validation
        if (current_temps is not None and heating_load is not None and cooling_load is not None):
            if predictions.ndim > 1:
                next_temps = predictions[:, 0]
            else:
                next_temps = predictions
            
            energy_satisfied, energy_metrics = self.energy_validator.validate_energy_balance(
                current_temps, next_temps, heating_load, cooling_load, outdoor_temps,
                tolerance=self.energy_tolerance
            )
            results["energy_balance"] = energy_metrics
        
        return results


class CustomPhysicsLoss(nn.Module):
    """
    Flexible physics-informed loss with customizable constraints.
    """
    
    def __init__(
        self,
        constraints: Optional[List[Callable]] = None,
        constraint_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        
        self.constraints = constraints or []
        self.constraint_weights = constraint_weights or [1.0] * len(self.constraints)
    
    def add_constraint(
        self,
        constraint_fn: Callable,
        weight: float = 1.0,
    ) -> None:
        """Add a physics constraint."""
        self.constraints.append(constraint_fn)
        self.constraint_weights.append(weight)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss with custom constraints."""
        # Base MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        loss_components = {"mse": mse_loss.item()}
        
        total_loss = mse_loss
        
        # Apply custom constraints
        for constraint_fn, weight in zip(self.constraints, self.constraint_weights):
            constraint_loss = constraint_fn(predictions, targets)
            loss_components[f"constraint_{len(loss_components)}"] = constraint_loss.item()
            total_loss = total_loss + weight * constraint_loss
        
        loss_components["total"] = total_loss.item()
        
        return total_loss, loss_components


# Example constraint functions
def monotonicity_constraint(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Enforce monotonic relationship between input and output.
    
    For heating/cooling: higher outdoor temp should lead to lower heating load.
    """
    if predictions.ndim < 2:
        return torch.tensor(0.0, device=predictions.device)
    
    # Assume first input is outdoor temperature
    # Check if prediction decreases with increasing outdoor temp
    diff = predictions[:, 0].unsqueeze(1) - predictions[:, 0].unsqueeze(0)
    monotonic_penalty = F.relu(diff).mean()
    
    return monotonic_penalty


def smoothness_constraint(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Enforce smoothness in predictions (temporal or spatial)."""
    if predictions.ndim < 2:
        return torch.tensor(0.0, device=predictions.device)
    
    # Finite difference for smoothness
    diff = predictions[:, 1:] - predictions[:, :-1]
    smoothness = torch.mean(diff ** 2)
    
    return smoothness


def physical_bounds_constraint(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    min_val: float = -1e6,
    max_val: float = 1e6,
) -> torch.Tensor:
    """Enforce physical bounds on predictions."""
    below_min = F.relu(torch.tensor(min_val, device=predictions.device) - predictions)
    above_max = F.relu(predictions - torch.tensor(max_val, device=predictions.device))
    
    return below_max.mean() + above_max.mean()


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Physics-Informed Loss")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-zones", type=int, default=5)
    args = parser.parse_args()
    
    print("Testing Physics-Informed Loss Functions")
    print("=" * 50)
    
    # Create dummy data
    batch_size = args.batch_size
    num_zones = args.num_zones
    
    # Predictions (temperatures)
    predictions = torch.randn(batch_size, num_zones)
    targets = predictions + torch.randn(batch_size, num_zones) * 0.1
    
    # Physics data
    current_temps = torch.randn(batch_size) * 10 + 20  # Around 20C
    heating_load = torch.abs(torch.randn(batch_size)) * 10  # kWh
    cooling_load = torch.abs(torch.randn(batch_size)) * 5  # kWh
    outdoor_temps = torch.randn(batch_size) * 15 + 10  # Around 10C
    
    # Test PhysicsInformedLoss
    print("\n1. Testing PhysicsInformedLoss...")
    loss_fn = PhysicsInformedLoss(
        mse_weight=1.0,
        energy_balance_weight=0.1,
        temperature_bounds_weight=0.1,
    )
    
    loss, components = loss_fn(
        predictions,
        targets,
        current_temps=current_temps,
        heating_load=heating_load,
        cooling_load=cooling_load,
        outdoor_temps=outdoor_temps,
    )
    
    print(f"  Total loss: {components['total']:.4f}")
    print(f"  MSE: {components['mse']:.4f}")
    print(f"  Energy balance: {components.get('energy_balance', 0):.4f}")
    print(f"  Temperature bounds: {components['temperature_bounds']:.4f}")
    
    # Validate predictions
    print("\n2. Validating predictions...")
    validation_results = loss_fn.validate_predictions(
        predictions,
        current_temps=current_temps,
        heating_load=heating_load,
        cooling_load=cooling_load,
        outdoor_temps=outdoor_temps,
    )
    
    print("  Temperature validation:")
    for key, value in validation_results["temperature"].items():
        print(f"    {key}: {value:.4f}")
    
    print("  Energy balance validation:")
    for key, value in validation_results["energy_balance"].items():
        print(f"    {key}: {value:.4f}")
    
    # Test custom constraints
    print("\n3. Testing CustomPhysicsLoss with constraints...")
    custom_loss = CustomPhysicsLoss()
    custom_loss.add_constraint(monotonicity_constraint, weight=0.1)
    custom_loss.add_constraint(smoothness_constraint, weight=0.1)
    
    loss, components = custom_loss(predictions, targets)
    print(f"  Total loss: {components['total']:.4f}")
    
    print("\nAll tests completed!")
