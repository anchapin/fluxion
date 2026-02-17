#!/usr/bin/env python3
"""
Tests for Physics-Informed Loss Functions

Tests for Issue #172: Phase 8: Implement physics-informed loss functions
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from physics_informed_loss import (
    TemperatureBounds,
    EnergyBalanceValidator,
    PhysicsInformedLoss,
    CustomPhysicsLoss,
    monotonicity_constraint,
    smoothness_constraint,
    physical_bounds_constraint,
)


class TestTemperatureBounds:
    """Tests for TemperatureBounds class."""
    
    def test_validate_within_bounds(self):
        """Test validation with temperatures within bounds."""
        temps = torch.tensor([20.0, 22.0, 25.0])
        
        valid, metrics = TemperatureBounds.validate(temps, min_temp=18.0, max_temp=28.0)
        
        assert valid.all()
        assert metrics["valid_fraction"] == 1.0
    
    def test_validate_outside_bounds(self):
        """Test validation with temperatures outside bounds."""
        temps = torch.tensor([15.0, 22.0, 35.0])
        
        valid, metrics = TemperatureBounds.validate(temps, min_temp=18.0, max_temp=28.0)
        
        assert not valid.all()
        assert metrics["valid_fraction"] == pytest.approx(1/3, rel=0.1)
        assert metrics["below_min_fraction"] == pytest.approx(1/3, rel=0.1)
        assert metrics["above_max_fraction"] == pytest.approx(1/3, rel=0.1)
    
    def test_validate_default_bounds(self):
        """Test validation with default bounds."""
        temps = torch.tensor([20.0, 22.0, 25.0])
        
        valid, metrics = TemperatureBounds.validate(temps)
        
        assert valid.all()
    
    def test_comfort_range(self):
        """Test comfort range constants."""
        assert TemperatureBounds.COMFORT_MIN == 18.0
        assert TemperatureBounds.COMFORT_MAX == 28.0


class TestEnergyBalanceValidator:
    """Tests for EnergyBalanceValidator class."""
    
    def test_initialization(self):
        """Test validator initialization."""
        validator = EnergyBalanceValidator(
            thermal_mass=2.0,
            heat_transfer_coeff=15.0,
            time_step=0.5,
        )
        
        assert validator.thermal_mass == 2.0
        assert validator.heat_transfer_coeff == 15.0
        assert validator.time_step == 0.5
    
    def test_energy_balance_residual_zero(self):
        """Test energy balance residual when physics is satisfied."""
        validator = EnergyBalanceValidator(thermal_mass=1.0, heat_transfer_coeff=0.0)
        
        current_temps = torch.tensor([20.0])
        next_temps = torch.tensor([21.0])  # Temperature increased by 1K
        heating_load = torch.tensor([1.0])  # 1 kWh input
        cooling_load = torch.tensor([0.0])
        
        # With no heat transfer and 1 kWh input to 1 kWh/K thermal mass = 1K increase
        residual = validator.compute_energy_balance_residual(
            current_temps, next_temps, heating_load, cooling_load
        )
        
        assert torch.allclose(residual, torch.tensor([0.0]), atol=1e-5)
    
    def test_energy_balance_residual_nonzero(self):
        """Test energy balance residual with imbalance."""
        validator = EnergyBalanceValidator(thermal_mass=1.0, heat_transfer_coeff=10.0)
        
        current_temps = torch.tensor([20.0])
        next_temps = torch.tensor([21.0])
        heating_load = torch.tensor([0.0])  # No heating
        cooling_load = torch.tensor([0.0])
        outdoor_temps = torch.tensor([10.0])  # Colder outside
        
        residual = validator.compute_energy_balance_residual(
            current_temps, next_temps, heating_load, cooling_load, outdoor_temps
        )
        
        # Residual should be non-zero due to heat loss
        assert torch.any(residual != 0)
    
    def test_validate_energy_balance_satisfied(self):
        """Test energy balance validation when satisfied."""
        validator = EnergyBalanceValidator(thermal_mass=1.0, heat_transfer_coeff=0.0)
        
        current_temps = torch.tensor([20.0])
        next_temps = torch.tensor([21.0])
        heating_load = torch.tensor([1.0])
        cooling_load = torch.tensor([0.0])
        
        satisfied, metrics = validator.validate_energy_balance(
            current_temps, next_temps, heating_load, cooling_load, tolerance=0.1
        )
        
        assert satisfied
        assert metrics["max_residual"] < 0.1
    
    def test_validate_energy_balance_not_satisfied(self):
        """Test energy balance validation when not satisfied."""
        validator = EnergyBalanceValidator(thermal_mass=1.0, heat_transfer_coeff=0.0)
        
        current_temps = torch.tensor([20.0])
        next_temps = torch.tensor([30.0])  # Big jump
        heating_load = torch.tensor([1.0])  # Not enough to explain the jump
        cooling_load = torch.tensor([0.0])
        
        satisfied, metrics = validator.validate_energy_balance(
            current_temps, next_temps, heating_load, cooling_load, tolerance=0.1
        )
        
        assert not satisfied
        assert metrics["max_residual"] > 0.1


class TestPhysicsInformedLoss:
    """Tests for PhysicsInformedLoss class."""
    
    def test_initialization(self):
        """Test loss function initialization."""
        loss_fn = PhysicsInformedLoss(
            mse_weight=1.0,
            energy_balance_weight=0.2,
            temperature_bounds_weight=0.1,
        )
        
        assert loss_fn.mse_weight == 1.0
        assert loss_fn.energy_balance_weight == 0.2
        assert loss_fn.temperature_bounds_weight == 0.1
    
    def test_forward_mse_only(self):
        """Test forward pass with MSE only."""
        loss_fn = PhysicsInformedLoss(
            mse_weight=1.0,
            energy_balance_weight=0.0,
            temperature_bounds_weight=0.0,
        )
        
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)
        
        loss, components = loss_fn(predictions, targets)
        
        assert "mse" in components
        assert "total" in components
        assert components["mse"] >= 0
    
    def test_forward_with_physics(self):
        """Test forward pass with physics constraints."""
        loss_fn = PhysicsInformedLoss(
            mse_weight=1.0,
            energy_balance_weight=0.1,
            temperature_bounds_weight=0.1,
        )
        
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)
        current_temps = torch.randn(10) * 10 + 20
        heating_load = torch.abs(torch.randn(10)) * 10
        cooling_load = torch.abs(torch.randn(10)) * 5
        outdoor_temps = torch.randn(10) * 15 + 10
        
        loss, components = loss_fn(
            predictions, targets,
            current_temps=current_temps,
            heating_load=heating_load,
            cooling_load=cooling_load,
            outdoor_temps=outdoor_temps,
        )
        
        assert "mse" in components
        assert "energy_balance" in components
        assert "temperature_bounds" in components
    
    def test_temperature_bounds_penalty(self):
        """Test temperature bounds penalty."""
        loss_fn = PhysicsInformedLoss(
            mse_weight=0.0,
            energy_balance_weight=0.0,
            temperature_bounds_weight=1.0,
            min_temp=20.0,
            max_temp=25.0,
        )
        
        # Predictions outside bounds
        predictions = torch.tensor([[30.0], [15.0], [22.0]])  # One out of bounds each side
        targets = torch.zeros(3, 1)
        
        loss, components = loss_fn(predictions, targets)
        
        assert components["temperature_bounds"] > 0
    
    def test_validate_predictions(self):
        """Test prediction validation."""
        loss_fn = PhysicsInformedLoss(
            min_temp=20.0,
            max_temp=25.0,
        )
        
        predictions = torch.tensor([[22.0], [30.0], [18.0]])
        
        results = loss_fn.validate_predictions(predictions)
        
        assert "temperature" in results
        assert results["temperature"]["valid_fraction"] == pytest.approx(1/3, rel=0.1)
    
    def test_validate_with_physics(self):
        """Test validation with physics data."""
        loss_fn = PhysicsInformedLoss(energy_tolerance=0.1)
        
        predictions = torch.randn(10, 5)
        current_temps = torch.randn(10) * 10 + 20
        heating_load = torch.abs(torch.randn(10)) * 10
        cooling_load = torch.abs(torch.randn(10)) * 5
        
        results = loss_fn.validate_predictions(
            predictions,
            current_temps=current_temps,
            heating_load=heating_load,
            cooling_load=cooling_load,
        )
        
        assert "temperature" in results
        assert "energy_balance" in results


class TestCustomPhysicsLoss:
    """Tests for CustomPhysicsLoss class."""
    
    def test_initialization(self):
        """Test custom loss initialization."""
        loss_fn = CustomPhysicsLoss()
        
        assert len(loss_fn.constraints) == 0
    
    def test_add_constraint(self):
        """Test adding constraints."""
        loss_fn = CustomPhysicsLoss()
        loss_fn.add_constraint(monotonicity_constraint, weight=0.5)
        
        assert len(loss_fn.constraints) == 1
        assert loss_fn.constraint_weights[0] == 0.5
    
    def test_forward_no_constraints(self):
        """Test forward with no constraints."""
        loss_fn = CustomPhysicsLoss()
        
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)
        
        loss, components = loss_fn(predictions, targets)
        
        assert "mse" in components
        assert components["mse"] >= 0
    
    def test_forward_with_constraints(self):
        """Test forward with constraints."""
        loss_fn = CustomPhysicsLoss()
        loss_fn.add_constraint(smoothness_constraint, weight=0.1)
        
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)
        
        loss, components = loss_fn(predictions, targets)
        
        assert "mse" in components
        assert "constraint_1" in components


class TestConstraintFunctions:
    """Tests for constraint functions."""
    
    def test_monotonicity_constraint(self):
        """Test monotonicity constraint."""
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)
        
        loss = monotonicity_constraint(predictions, targets)
        
        assert loss >= 0
    
    def test_monotonicity_constraint_1d(self):
        """Test monotonicity constraint with 1D predictions."""
        predictions = torch.randn(10)
        targets = torch.randn(10)
        
        loss = monotonicity_constraint(predictions, targets)
        
        assert loss == 0.0  # Should return 0 for 1D
    
    def test_smoothness_constraint(self):
        """Test smoothness constraint."""
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)
        
        loss = smoothness_constraint(predictions, targets)
        
        assert loss >= 0
    
    def test_physical_bounds_constraint(self):
        """Test physical bounds constraint."""
        predictions = torch.tensor([[-100.0], [0.0], [100.0]])
        targets = torch.zeros(3, 1)
        
        loss = physical_bounds_constraint(predictions, targets, min_val=-50.0, max_val=50.0)
        
        assert loss > 0
    
    def test_physical_bounds_constraint_within(self):
        """Test physical bounds when within bounds."""
        predictions = torch.tensor([[10.0], [20.0], [30.0]])
        targets = torch.zeros(3, 1)
        
        loss = physical_bounds_constraint(predictions, targets, min_val=-50.0, max_val=50.0)
        
        assert loss == 0.0


class TestIntegration:
    """Integration tests for physics-informed loss."""
    
    def test_training_loop_with_physics_loss(self):
        """Test a full training loop with physics-informed loss."""
        # Create simple model
        model = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Create loss function
        loss_fn = PhysicsInformedLoss(
            mse_weight=1.0,
            temperature_bounds_weight=0.1,
            min_temp=15.0,
            max_temp=30.0,
        )
        
        # Simple optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Training data
        X = torch.randn(100, 5)
        y = X[:, 0:1] * 2 + torch.randn(100, 1) * 0.1
        
        # Training loop
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            
            predictions = model(X)
            loss, components = loss_fn(predictions, y)
            
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_pred = model(X[:10])
            results = loss_fn.validate_predictions(test_pred)
        
        assert "temperature" in results
        assert results["temperature"]["min_temp"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
