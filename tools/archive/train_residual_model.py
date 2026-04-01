#!/usr/bin/env python3
"""
ML Residual Correction Training Pipeline for Fluxion

This script trains a neural network to predict the residual error between
5R1C thermal network predictions and EnergyPlus ground truth.

The trained model can then be used to correct 5R1C predictions, achieving
high accuracy while maintaining fast inference speed.

Usage:
    python tools/train_residual_model.py --data-dir data/ml_training --output models/residual_model.onnx
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ResidualDataset(Dataset):
    """Dataset for ML residual correction training."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize dataset.

        Args:
            features: Feature matrix (N_samples, N_features)
            targets: Target residuals (N_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class ResidualMLP(nn.Module):
    """
    Multi-layer perceptron for residual error prediction.

    Architecture:
    - Input layer: N_features
    - Hidden layers: [128, 64, 32] with ReLU, LayerNorm, Dropout
    - Output layer: 1 (residual prediction in W)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input layer
        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LayerNorm(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def create_features_from_simulation(
    weather_data: pd.DataFrame,
    simulation_results: pd.DataFrame,
    building_params: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create feature matrix and targets from simulation data.

    Args:
        weather_data: Weather data (T_outdoor, solar, etc.)
        simulation_results: 5R1C and EnergyPlus results
        building_params: Building parameters (mass, U-value, etc.)

    Returns:
        features: Feature matrix (N_timesteps, N_features)
        targets: Residual targets (N_timesteps,)
    """
    n_timesteps = len(weather_data)

    # Feature dimensions (20 total)
    features = np.zeros((n_timesteps, 20))

    # 1-4: Building parameters (constant for all timesteps)
    features[:, 0] = (
        building_params.get("thermal_capacitance", 1.2e7) / 1.0e7
    )  # Normalized C
    features[:, 1] = building_params.get("u_value", 0.5) / 1.0  # Normalized U-value
    features[:, 2] = building_params.get("glazing_ratio", 0.2)  # Glazing ratio
    features[:, 3] = building_params.get("time_constant", 5.0) / 10.0  # Normalized τ

    # 5-7: Weather features (current)
    features[:, 4] = weather_data["dry_bulb_temp"].values / 30.0  # Normalized T_outdoor
    features[:, 5] = weather_data["direct_normal_rad"].values / 1000.0  # Normalized DNI
    features[:, 6] = (
        weather_data["diffuse_horizontal_rad"].values / 500.0
    )  # Normalized DHI

    # 8-9: Temporal features (cyclical encoding)
    hour = weather_data.index % 24
    day = weather_data.index // 24
    features[:, 7] = np.sin(2 * np.pi * hour / 24)  # Hour (sin)
    features[:, 8] = np.cos(2 * np.pi * hour / 24)  # Hour (cos)
    features[:, 9] = np.sin(2 * np.pi * day / 365)  # Day (sin)
    features[:, 10] = np.cos(2 * np.pi * day / 365)  # Day (cos)

    # 11-12: Simulation state
    features[:, 11] = simulation_results["zone_temp"].values / 25.0  # Normalized T_zone
    features[:, 12] = simulation_results["mass_temp"].values / 25.0  # Normalized T_mass

    # 13-14: 5R1C predictions
    features[:, 13] = (
        simulation_results["hvac_5r1c"].values / 5000.0
    )  # Normalized HVAC_5R1C
    features[:, 14] = (
        simulation_results["solar_gain_5r1c"].values / 1000.0
    )  # Normalized solar

    # 15-18: Lagged weather features (t-1, t-2, t-3, t-4)
    for i, lag in enumerate([1, 2, 3, 4]):
        if lag < n_timesteps:
            features[lag:, 15 + i] = weather_data["dry_bulb_temp"].values[:-lag] / 30.0
            features[:lag, 15 + i] = features[0, 4]  # Fill initial with first value

    # 19: HVAC state (one-hot: -1=cooling, 0=off, 1=heating)
    hvac_state = np.sign(simulation_results["hvac_5r1c"].values)
    features[:, 19] = hvac_state / 1.0

    # Target: residual (EnergyPlus - 5R1C)
    targets = (
        simulation_results["hvac_energyplus"].values
        - simulation_results["hvac_5r1c"].values
    )

    return features, targets


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    device: str = "cpu",
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
) -> Tuple[nn.Module, Dict]:
    """
    Train the residual prediction model.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        input_dim: Number of input features
        device: Training device ('cpu' or 'cuda')
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization

    Returns:
        model: Trained model
        history: Training history (loss curves)
    """
    # Initialize model
    model = ResidualMLP(input_dim=input_dim).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # Training history
    history = {"train_loss": [], "val_loss": [], "lr": []}

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    max_patience = 20

    logger.info(f"Starting training on {device}...")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                predictions = model(features)
                loss = criterion(predictions, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log progress
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"LR: {current_lr:.6f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model (val_loss={best_val_loss:.6f})")

    return model, history


def export_to_onnx(model: nn.Module, input_dim: int, output_path: Path) -> None:
    """
    Export trained model to ONNX format.

    Args:
        model: Trained PyTorch model
        input_dim: Number of input features
        output_path: Output file path
    """
    model.eval()
    dummy_input = torch.randn(1, input_dim)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["residual_prediction"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "residual_prediction": {0: "batch_size"},
        },
    )

    logger.info(f"Model exported to {output_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train ML residual correction model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ml_training",
        help="Directory containing training data CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/residual_model.onnx",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device",
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load training data
    data_dir = Path(args.data_dir)
    logger.info(f"Loading training data from {data_dir}...")

    # For demonstration, generate synthetic data
    # In production, load from CSV files generated by simulation comparison
    n_samples = 100000
    n_features = 20

    # Generate synthetic features (normalized)
    np.random.seed(42)
    features = np.random.randn(n_samples, n_features).astype(np.float32)

    # Generate synthetic targets (residual error with some structure)
    # Target = f(features) + noise
    targets = (
        -500 * features[:, 4]  # Correlation with T_outdoor
        - 300 * features[:, 5]  # Correlation with solar
        - 200 * features[:, 3]  # Correlation with time constant
        + 100 * np.random.randn(n_samples)  # Noise
    ).astype(np.float32)

    logger.info(f"Generated {n_samples} synthetic samples with {n_features} features")

    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    test_size = n_samples - train_size - val_size

    train_dataset = ResidualDataset(features[:train_size], targets[:train_size])
    val_dataset = ResidualDataset(
        features[train_size : train_size + val_size],
        targets[train_size : train_size + val_size],
    )
    test_dataset = ResidualDataset(
        features[train_size + val_size :], targets[train_size + val_size :]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    logger.info(
        f"Data split: {train_size} train, {val_size} validation, {test_size} test"
    )

    # Train model
    model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=n_features,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
    )

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    predictions_all = []
    targets_all = []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(args.device)
            targets = targets.to(args.device)
            predictions = model(features)
            test_loss += nn.MSELoss()(predictions, targets).item()
            predictions_all.extend(predictions.cpu().numpy())
            targets_all.extend(targets.cpu().numpy())

    test_loss /= len(test_loader)
    predictions_np = np.array(predictions_all)
    targets_np = np.array(targets_all)

    # Calculate metrics
    rmse = np.sqrt(np.mean((predictions_np - targets_np) ** 2))
    mae = np.mean(np.abs(predictions_np - targets_np))
    r2 = 1 - np.sum((predictions_np - targets_np) ** 2) / np.sum(
        (targets_np - np.mean(targets_np)) ** 2
    )

    logger.info("Test Results:")
    logger.info(f"  RMSE: {rmse:.2f} W")
    logger.info(f"  MAE: {mae:.2f} W")
    logger.info(f"  R²: {r2:.4f}")

    # Export to ONNX
    export_to_onnx(model, n_features, output_path)

    # Save training history
    history_path = output_path.with_suffix(".json")
    with open(history_path, "w") as f:
        json.dump(
            {
                "train_loss": history["train_loss"],
                "val_loss": history["val_loss"],
                "test_rmse": rmse,
                "test_mae": mae,
                "test_r2": r2,
            },
            f,
            indent=2,
        )

    logger.info(f"Training history saved to {history_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
