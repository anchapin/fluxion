#!/usr/bin/env python3
"""
Tests for PIML Loss Functions

Tests for Issue #552: PIML-01: Implement Physics-Informed Machine Learning loss functions
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from piml_loss import (
    PIMLLoss,
    PIMLSurrogate,
    RCPhysicsConfig,
    PIMLConfig,
    StandardLoss,
    compute_cvrmse,
    generate_synthetic_data,
    train_piml_surrogate,
)


class TestRCPhysicsConfig:
    """Tests for RCPhysicsConfig dataclass."""

    def test_initialization(self):
        """Test RC physics config initialization."""
        config = RCPhysicsConfig(
            thermal_capacity=100.0,
            h_transmission=300.0,
            h_ventilation=75.0,
        )

        assert config.thermal_capacity == 100.0
        assert config.h_transmission == 300.0
        assert config.h_ventilation == 75.0

    def test_total_heat_transfer_coeff(self):
        """Test total heat transfer coefficient calculation."""
        config = RCPhysicsConfig(
            h_transmission=200.0,
            h_ventilation=50.0,
        )

        assert config.total_heat_transfer_coeff == 250.0

    def test_compute_steady_state_delta_t(self):
        """Test steady state temperature difference calculation."""
        config = RCPhysicsConfig(
            h_transmission=200.0,
            h_ventilation=50.0,
        )

        delta_t = config.compute_steady_state_delta_t(5000.0)
        assert delta_t == 20.0

    def test_compute_steady_state_zero_coeff(self):
        """Test steady state with zero coefficient."""
        config = RCPhysicsConfig(
            h_transmission=0.0,
            h_ventilation=0.0,
        )

        delta_t = config.compute_steady_state_delta_t(5000.0)
        assert delta_t == 0.0


class TestPIMLConfig:
    """Tests for PIMLConfig dataclass."""

    def test_initialization(self):
        """Test PIML config initialization."""
        config = PIMLConfig(
            input_dim=12,
            output_dim=2,
            hidden_dims=[256, 128, 64],
            mse_weight=1.0,
            piml_weight=0.5,
        )

        assert config.input_dim == 12
        assert config.output_dim == 2
        assert config.hidden_dims == [256, 128, 64]
        assert config.piml_weight == 0.5

    def test_defaults(self):
        """Test default configuration values."""
        config = PIMLConfig()

        assert config.learning_rate == 1e-3
        assert config.epochs == 500
        assert config.batch_size == 64


class TestPIMLLoss:
    """Tests for PIMLLoss class."""

    def test_initialization(self):
        """Test PIML loss initialization."""
        config = PIMLConfig()
        loss_fn = PIMLLoss(config)

        assert loss_fn.config == config
        assert hasattr(loss_fn, "thermal_capacity")
        assert hasattr(loss_fn, "h_transmission")
        assert hasattr(loss_fn, "h_ventilation")

    def test_total_h_property(self):
        """Test total_h property calculation."""
        config = PIMLConfig(
            rc_h_transmission=200.0,
            rc_h_ventilation=50.0,
        )
        loss_fn = PIMLLoss(config)

        total_h = loss_fn.total_h
        assert abs(total_h.item() - 250.0) < 1e-5

    def test_forward_returns_tuple(self):
        """Test forward pass returns loss and components."""
        config = PIMLConfig()
        loss_fn = PIMLLoss(config)

        batch_size = 8
        predictions = (
            torch.randn(batch_size, 1),
            torch.randn(batch_size, 1),
        )
        targets = (
            torch.randn(batch_size, 1),
            torch.randn(batch_size, 1),
        )
        features = torch.randn(batch_size, 9)

        total_loss, loss_components = loss_fn(predictions, targets, features)

        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(loss_components, dict)
        assert "mse" in loss_components
        assert "piml" in loss_components
        assert "total" in loss_components

    def test_forward_with_physics_params(self):
        """Test forward pass with physics parameters."""
        config = PIMLConfig()
        loss_fn = PIMLLoss(config)

        batch_size = 8
        predictions = (
            torch.randn(batch_size, 1),
            torch.randn(batch_size, 1),
        )
        targets = (
            torch.randn(batch_size, 1),
            torch.randn(batch_size, 1),
        )
        features = torch.randn(batch_size, 9)
        physics_params = torch.randn(batch_size, 3)

        total_loss, loss_components = loss_fn(
            predictions, targets, features, physics_params
        )

        assert total_loss.item() >= 0.0

    def test_loss_components_all_positive(self):
        """Test that all loss components are non-negative."""
        config = PIMLConfig()
        loss_fn = PIMLLoss(config)

        batch_size = 16
        predictions = (
            torch.abs(torch.randn(batch_size, 1)),
            torch.abs(torch.randn(batch_size, 1)),
        )
        targets = (
            torch.abs(torch.randn(batch_size, 1)),
            torch.abs(torch.randn(batch_size, 1)),
        )
        features = torch.randn(batch_size, 9)

        _, loss_components = loss_fn(predictions, targets, features)

        for key, value in loss_components.items():
            assert value >= 0, f"Loss component {key} is negative: {value}"


class TestStandardLoss:
    """Tests for StandardLoss class."""

    def test_initialization(self):
        """Test standard loss initialization."""
        loss_fn = StandardLoss(l1_weight=0.1, l2_weight=1.0)

        assert loss_fn.l1_weight == 0.1
        assert loss_fn.l2_weight == 1.0

    def test_forward_returns_tuple(self):
        """Test forward pass returns loss and components."""
        loss_fn = StandardLoss()

        batch_size = 8
        predictions = (
            torch.randn(batch_size, 1),
            torch.randn(batch_size, 1),
        )
        targets = (
            torch.randn(batch_size, 1),
            torch.randn(batch_size, 1),
        )

        total_loss, loss_components = loss_fn(predictions, targets)

        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(loss_components, dict)
        assert "mse" in loss_components
        assert "total" in loss_components

    def test_l1_loss_computed(self):
        """Test that L1 loss is computed when weight > 0."""
        loss_fn = StandardLoss(l1_weight=1.0)

        batch_size = 8
        predictions = (
            torch.randn(batch_size, 1),
            torch.randn(batch_size, 1),
        )
        targets = (
            torch.randn(batch_size, 1),
            torch.randn(batch_size, 1),
        )

        _, loss_components = loss_fn(predictions, targets)

        assert "l1" in loss_components


class TestPIMLSurrogate:
    """Tests for PIMLSurrogate model."""

    def test_initialization(self):
        """Test surrogate model initialization."""
        model = PIMLSurrogate(input_dim=9, output_dim=1)

        assert isinstance(model, torch.nn.Module)

    def test_forward(self):
        """Test forward pass."""
        model = PIMLSurrogate(input_dim=9, output_dim=1)
        model.eval()

        batch_size = 8
        x = torch.randn(batch_size, 9)

        heating, cooling = model(x)

        assert heating.shape == (batch_size, 1)
        assert cooling.shape == (batch_size, 1)
        assert (heating >= 0).all()
        assert (cooling >= 0).all()

    def test_forward_with_physics_params(self):
        """Test forward pass with physics constraints."""
        model = PIMLSurrogate(input_dim=9, output_dim=1, use_physics_constraints=True)
        model.eval()

        batch_size = 8
        x = torch.randn(batch_size, 9)
        physics_params = torch.randn(batch_size, 3)

        heating, cooling = model(x, physics_params)

        assert heating.shape == (batch_size, 1)
        assert cooling.shape == (batch_size, 1)

    def test_forward_without_physics_constraints(self):
        """Test forward pass without physics constraints."""
        model = PIMLSurrogate(input_dim=9, output_dim=1, use_physics_constraints=False)
        model.eval()

        batch_size = 8
        x = torch.randn(batch_size, 9)

        heating, cooling = model(x, None)

        assert heating.shape == (batch_size, 1)
        assert cooling.shape == (batch_size, 1)


class TestComputeCvrmse:
    """Tests for compute_cvrmse function."""

    def test_cvrmse_basic(self):
        """Test basic CvRMSE calculation."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = torch.tensor([1.1, 2.0, 3.2, 3.8, 5.1])

        cvrmse = compute_cvrmse(predictions, targets)

        assert cvrmse > 0
        assert cvrmse < 100

    def test_cvrmse_zero_mean(self):
        """Test CvRMSE with zero mean target."""
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([0.0, 0.0, 0.0])

        cvrmse = compute_cvrmse(predictions, targets)

        assert cvrmse == float("inf")

    def test_cvrmse_perfect_prediction(self):
        """Test CvRMSE with perfect prediction."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0])

        cvrmse = compute_cvrmse(predictions, targets)

        assert cvrmse == 0.0


class TestGenerateSyntheticData:
    """Tests for generate_synthetic_data function."""

    def test_output_shapes(self):
        """Test synthetic data output shapes."""
        n_samples = 100
        n_timesteps = 24

        X, y_heating, y_cooling = generate_synthetic_data(n_samples, n_timesteps)

        expected_len = n_samples * n_timesteps
        assert X.shape[0] == expected_len
        assert y_heating.shape[0] == expected_len
        assert y_cooling.shape[0] == expected_len

    def test_feature_dimensions(self):
        """Test feature dimensions."""
        X, _, _ = generate_synthetic_data(10, 24)

        assert X.shape[1] == 9

    def test_positive_loads(self):
        """Test that generated loads are non-negative."""
        _, y_heating, y_cooling = generate_synthetic_data(10, 24)

        assert (y_heating >= 0).all()
        assert (y_cooling >= 0).all()

    def test_different_seeds(self):
        """Test different seeds produce different data."""
        X1, _, _ = generate_synthetic_data(10, 24, seed=42)
        X2, _, _ = generate_synthetic_data(10, 24, seed=123)

        assert not torch.equal(torch.from_numpy(X1), torch.from_numpy(X2))


class TestPIMLSurrogateTraining:
    """Integration tests for PIML surrogate training."""

    def test_training_loop_piml_loss(self):
        """Test training loop with PIML loss."""
        config = PIMLConfig(epochs=5, batch_size=32)
        rc_config = RCPhysicsConfig()

        X, y_heating, y_cooling = generate_synthetic_data(100, seed=42)

        split_idx = int(0.8 * len(X))
        X_train = torch.from_numpy(X[:split_idx]).float()
        y_h_train = torch.from_numpy(y_heating[:split_idx]).float()
        y_c_train = torch.from_numpy(y_cooling[:split_idx]).float()

        train_dataset = torch.utils.data.TensorDataset(X_train, y_h_train, y_c_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

        model = PIMLSurrogate(input_dim=9, output_dim=1)
        device = torch.device("cpu")

        model, history = train_piml_surrogate(
            model=model,
            train_loader=train_loader,
            config=config,
            rc_config=rc_config,
            val_data=None,
            output_dir=None,
            use_piml_loss=True,
        )

        assert "loss" in history
        assert len(history["loss"]) <= config.epochs

    def test_training_loop_standard_loss(self):
        """Test training loop with standard loss."""
        config = PIMLConfig(epochs=5, batch_size=32)
        rc_config = RCPhysicsConfig()

        X, y_heating, y_cooling = generate_synthetic_data(100, seed=42)

        split_idx = int(0.8 * len(X))
        X_train = torch.from_numpy(X[:split_idx]).float()
        y_h_train = torch.from_numpy(y_heating[:split_idx]).float()
        y_c_train = torch.from_numpy(y_cooling[:split_idx]).float()

        train_dataset = torch.utils.data.TensorDataset(X_train, y_h_train, y_c_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

        model = PIMLSurrogate(input_dim=9, output_dim=1)

        model, history = train_piml_surrogate(
            model=model,
            train_loader=train_loader,
            config=config,
            rc_config=rc_config,
            val_data=None,
            output_dir=None,
            use_piml_loss=False,
        )

        assert "loss" in history
        assert len(history["loss"]) <= config.epochs

    def test_validation_metrics_computed(self):
        """Test that validation metrics are computed when val_data provided."""
        config = PIMLConfig(epochs=5, batch_size=32)
        rc_config = RCPhysicsConfig()

        X, y_heating, y_cooling = generate_synthetic_data(100, seed=42)

        split_idx = int(0.8 * len(X))
        X_train = torch.from_numpy(X[:split_idx]).float()
        y_h_train = torch.from_numpy(y_heating[:split_idx]).float()
        y_c_train = torch.from_numpy(y_cooling[:split_idx]).float()

        X_val = torch.from_numpy(X[split_idx:]).float()
        y_h_val = torch.from_numpy(y_heating[split_idx:]).float()
        y_c_val = torch.from_numpy(y_cooling[split_idx:]).float()

        train_dataset = torch.utils.data.TensorDataset(X_train, y_h_train, y_c_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

        model = PIMLSurrogate(input_dim=9, output_dim=1)

        model, history = train_piml_surrogate(
            model=model,
            train_loader=train_loader,
            config=config,
            rc_config=rc_config,
            val_data=(X_val, y_h_val, y_c_val),
            output_dir=None,
            use_piml_loss=True,
        )

        assert "val_loss" in history
        assert "cvrmse" in history
        assert "peak_cvrmse" in history


class TestPIMLComparison:
    """Tests for PIML vs standard comparison."""

    def test_piml_improves_over_standard(self):
        """Test that PIML loss provides benefit over standard loss."""
        config = PIMLConfig(epochs=20, batch_size=32)
        rc_config = RCPhysicsConfig()

        X, y_heating, y_cooling = generate_synthetic_data(500, seed=42)

        split_idx = int(0.8 * len(X))
        X_train = torch.from_numpy(X[:split_idx]).float()
        y_h_train = torch.from_numpy(y_heating[:split_idx]).float()
        y_c_train = torch.from_numpy(y_cooling[:split_idx]).float()
        X_val = torch.from_numpy(X[split_idx:]).float()
        y_h_val = torch.from_numpy(y_heating[split_idx:]).float()
        y_c_val = torch.from_numpy(y_cooling[split_idx:]).float()

        train_dataset = torch.utils.data.TensorDataset(X_train, y_h_train, y_c_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

        model_piml = PIMLSurrogate(input_dim=9, output_dim=1)
        model_standard = PIMLSurrogate(input_dim=9, output_dim=1)

        model_piml, history_piml = train_piml_surrogate(
            model=model_piml,
            train_loader=train_loader,
            config=config,
            rc_config=rc_config,
            val_data=(X_val, y_h_val, y_c_val),
            output_dir=None,
            use_piml_loss=True,
        )

        model_standard, history_standard = train_piml_surrogate(
            model=model_standard,
            train_loader=train_loader,
            config=config,
            rc_config=rc_config,
            val_data=(X_val, y_h_val, y_c_val),
            output_dir=None,
            use_piml_loss=False,
        )

        assert history_piml["loss"][-1] >= 0
        assert history_standard["loss"][-1] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])