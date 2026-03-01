#!/usr/bin/env python3
"""
ML Surrogate Training Pipeline for Fluxion

This script processes training data collected from ASHRAE 140 validation runs
and trains ONNX surrogate models with physics-informed constraints.

The pipeline ensures:
- Surrogate models are validated against deterministic physics (R² > 0.98)
- Thermodynamic fidelity is maintained
- Models don't compromise physics accuracy

Usage:
    python tools/train_ml_surrogate.py --data-dir data/training --output models

Phase: Issue #383 - Integrate ML Surrogate FDD Pipeline
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class PhysicsInformedSurrogate(nn.Module):
    """
    Neural network surrogate model with physics-informed constraints.

    The model architecture ensures:
    1. Thermal load follows physical laws (U-value, temperature differences)
    2. Energy balance is preserved
    3. Gradients respect thermodynamic constraints
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
    ):
        super().__init__()

        # Feature encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ]
            )
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Physics-aware output layers
        # Separate heads for heating and cooling loads
        self.heating_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Softplus(),  # Ensure positive loads
        )

        self.cooling_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Softplus(),  # Ensure positive loads
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        physics_params: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the surrogate model.

        Args:
            x: Input features [batch_size, input_dim]
            physics_params: Optional physics parameters (U-value, setpoints, etc.)

        Returns:
            Tuple of (heating_loads, cooling_loads) tensors
        """
        # Encode features
        encoded = self.encoder(x)

        # Predict loads
        heating_pred = self.heating_head(encoded)
        cooling_pred = self.cooling_head(encoded)

        # Apply physics-informed constraints
        if physics_params is not None:
            # Extract U-value and temperatures for physics constraints
            u_value = physics_params[:, 0:1]
            t_outdoor = x[:, 0:1]
            t_zone = x[:, 1:2]
            heating_setpoint = physics_params[:, 1:2]
            cooling_setpoint = physics_params[:, 2:3]

            # Physical lower bound: Q >= U * A * (T_setpoint - T_outdoor)
            # This ensures the model doesn't predict loads below thermodynamic minimum
            physics_heating_min = u_value * (heating_setpoint - t_outdoor).clamp(
                min=0.0
            )
            physics_cooling_min = u_value * (t_outdoor - cooling_setpoint).clamp(
                min=0.0
            )

            # Enforce constraints
            heating_pred = heating_pred.clamp(min=physics_heating_min)
            cooling_pred = cooling_pred.clamp(min=physics_cooling_min)

        return heating_pred, cooling_pred


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss function combining data fitting with thermodynamic constraints.

    Loss = λ_data * MSE_loss + λ_physics * Physics_loss + λ_balance * Energy_balance_loss
    """

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.1,
        lambda_balance: float = 0.05,
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_balance = lambda_balance
        self.mse = nn.MSELoss()

    def forward(
        self,
        heating_pred: torch.Tensor,
        cooling_pred: torch.Tensor,
        heating_true: torch.Tensor,
        cooling_true: torch.Tensor,
        features: torch.Tensor,
        physics_params: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate total loss with physics-informed terms.

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # 1. Data fitting loss (MSE)
        heating_loss = self.mse(heating_pred, heating_true)
        cooling_loss = self.mse(cooling_pred, cooling_true)
        data_loss = (heating_loss + cooling_loss) / 2.0

        # 2. Physics regularization
        # Steady-state heat balance: Q = U * A * ΔT
        u_value = physics_params[:, 0:1]
        t_outdoor = features[:, 0:1]
        t_zone = features[:, 1:2]
        heating_setpoint = physics_params[:, 1:2]
        cooling_setpoint = physics_params[:, 2:3]

        # Theoretical minimum loads
        theory_heating_min = u_value * (heating_setpoint - t_outdoor).clamp(min=0.0)
        theory_cooling_min = u_value * (t_outdoor - cooling_setpoint).clamp(min=0.0)

        # Penalize predictions below thermodynamic minimum
        physics_loss = torch.mean(
            torch.relu(theory_heating_min - heating_pred) ** 2
            + torch.relu(theory_cooling_min - cooling_pred) ** 2
        )

        # 3. Energy balance regularization
        # Net energy should be balanced when HVAC is off (deadband)
        # This is a simplified check - in production, use full thermal balance
        balance_loss = torch.mean(
            (heating_pred - cooling_pred).abs()
            * (features[:, 1:2] > heating_setpoint)
            * (features[:, 1:2] < cooling_setpoint)
        )

        # Total loss
        total_loss = (
            self.lambda_data * data_loss
            + self.lambda_physics * physics_loss
            + self.lambda_balance * balance_loss
        )

        loss_components = {
            "data_loss": data_loss.item(),
            "physics_loss": physics_loss.item(),
            "balance_loss": balance_loss.item(),
        }

        return total_loss, loss_components


def load_training_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training data collected from ASHRAE 140 validation runs.

    Args:
        data_dir: Directory containing training data CSV files

    Returns:
        Tuple of (features, heating_targets, cooling_targets) numpy arrays
    """
    logger.info(f"Loading training data from {data_dir}...")

    # Find the latest samples file
    data_path = Path(data_dir)
    sample_files = list(data_path.glob("samples_*.csv"))

    if not sample_files:
        raise FileNotFoundError(f"No training data found in {data_dir}")

    # Use the most recent file
    latest_file = max(sample_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading data from {latest_file}")

    df = pd.read_csv(latest_file)
    logger.info(f"Loaded {len(df)} training samples")

    # Extract features
    feature_cols = [
        "outdoor_temp",
        "heating_setpoint",
        "cooling_setpoint",
        "hour_of_day",
        "day_of_year",
        "month",
        "u_value",
        "wwr",
    ]

    X = df[feature_cols].values.astype(np.float32)
    y_heating = df["heating_load"].values.astype(np.float32).reshape(-1, 1)
    y_cooling = df["cooling_load"].values.astype(np.float32).reshape(-1, 1)

    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_normalized = (X - X_mean) / X_std

    # Normalize targets
    y_heating_mean = y_heating.mean()
    y_heating_std = y_heating.std() + 1e-8
    y_cooling_mean = y_cooling.mean()
    y_cooling_std = y_cooling.std() + 1e-8

    y_heating_normalized = (y_heating - y_heating_mean) / y_heating_std
    y_cooling_normalized = (y_cooling - y_cooling_mean) / y_cooling_std

    normalization = {
        "X_mean": X_mean.tolist(),
        "X_std": X_std.tolist(),
        "y_heating_mean": float(y_heating_mean),
        "y_heating_std": float(y_heating_std),
        "y_cooling_mean": float(y_cooling_mean),
        "y_cooling_std": float(y_cooling_std),
    }

    logger.info(f"Feature shape: {X_normalized.shape}")
    logger.info(f"Heating target shape: {y_heating_normalized.shape}")
    logger.info(f"Cooling target shape: {y_cooling_normalized.shape}")

    return X_normalized, y_heating_normalized, y_cooling_normalized, normalization


def train_surrogate(
    X: np.ndarray,
    y_heating: np.ndarray,
    y_cooling: np.ndarray,
    hidden_dims: List[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> Tuple[PhysicsInformedSurrogate, Dict]:
    """
    Train the physics-informed surrogate model.

    Args:
        X: Training features
        y_heating: Heating load targets
        y_cooling: Cooling load targets
        hidden_dims: Hidden layer dimensions
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: PyTorch device

    Returns:
        Tuple of (trained_model, training_metrics_dict)
    """
    logger.info(f"Training surrogate model for {epochs} epochs...")

    input_dim = X.shape[1]
    output_dim = 1  # Per-zone load prediction

    # Split into train/validation
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_h_train, y_h_val = y_heating[:split_idx], y_heating[split_idx:]
    y_c_train, y_c_val = y_cooling[:split_idx], y_cooling[split_idx:]

    # Create tensors
    X_train_t = torch.from_numpy(X_train).to(device)
    y_h_train_t = torch.from_numpy(y_h_train).to(device)
    y_c_train_t = torch.from_numpy(y_c_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_h_val_t = torch.from_numpy(y_h_val).to(device)
    y_c_val_t = torch.from_numpy(y_c_val).to(device)

    # Create model
    model = PhysicsInformedSurrogate(input_dim, output_dim, hidden_dims).to(device)

    # Physics parameters (U-value, heating setpoint, cooling setpoint)
    # These are from the normalized features at indices 6, 1, 2
    physics_params_train = torch.stack(
        [
            X_train_t[:, 6:7],  # u_value
            X_train_t[:, 1:2],  # heating_setpoint
            X_train_t[:, 2:3],  # cooling_setpoint
        ],
        dim=1,
    ).squeeze(-1)

    physics_params_val = torch.stack(
        [
            X_val_t[:, 6:7],
            X_val_t[:, 1:2],
            X_val_t[:, 2:3],
        ],
        dim=1,
    ).squeeze(-1)

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=10, factor=0.5, verbose=True
    )
    criterion = PhysicsLoss()

    # Training loop
    best_val_loss = float("inf")
    history = {"loss": [], "val_loss": [], "r_squared": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # Batching
        dataset = TensorDataset(X_train_t, y_h_train_t, y_c_train_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch_X, batch_y_h, batch_y_c in loader:
            optimizer.zero_grad()

            heating_pred, cooling_pred = model(batch_X, physics_params_train)

            loss, _ = criterion(
                heating_pred,
                cooling_pred,
                batch_y_h,
                batch_y_c,
                batch_X,
                physics_params_train,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        # Validation
        model.eval()
        with torch.no_grad():
            heating_pred_val, cooling_pred_val = model(X_val_t, physics_params_val)
            val_loss, _ = criterion(
                heating_pred_val,
                cooling_pred_val,
                y_h_val_t,
                y_c_val_t,
                X_val_t,
                physics_params_val,
            )

            # Calculate R²
            r_squared = calculate_r_squared(
                (heating_pred_val.cpu().numpy() + cooling_pred_val.cpu().numpy()),
                (y_h_val_t.cpu().numpy() + y_c_val_t.cpu().numpy()),
            )

        history["loss"].append(avg_loss)
        history["val_loss"].append(val_loss.item())
        history["r_squared"].append(r_squared)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Loss: {avg_loss:.6f} - "
                f"Val Loss: {val_loss.item():.6f} - "
                f"R²: {r_squared:.4f}"
            )

    # Load best model
    model.load_state_dict(best_model_state)
    logger.info(f"Training complete. Best Val Loss: {best_val_loss:.6f}")

    # Final validation
    model.eval()
    with torch.no_grad():
        heating_pred_final, cooling_pred_final = model(X_val_t, physics_params_val)
        final_mae = np.mean(
            np.abs(
                (heating_pred_final.cpu().numpy() + cooling_pred_final.cpu().numpy())
                - (y_h_val_t.cpu().numpy() + y_c_val_t.cpu().numpy())
            )
        )
        final_r_squared = calculate_r_squared(
            (heating_pred_final.cpu().numpy() + cooling_pred_final.cpu().numpy()),
            (y_h_val_t.cpu().numpy() + y_c_val_t.cpu().numpy()),
        )

    metrics = {
        "best_val_loss": float(best_val_loss),
        "final_mae": float(final_mae),
        "final_r_squared": float(final_r_squared),
        "history": history,
    }

    return model, metrics


def calculate_r_squared(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Calculate R² score."""
    mean_actual = np.mean(actual)
    ss_tot = np.sum((actual - mean_actual) ** 2)
    ss_res = np.sum((actual - predicted) ** 2)

    if ss_tot < 1e-10:
        return 1.0 if ss_res < 1e-10 else -np.inf

    return 1.0 - (ss_res / ss_tot)


def export_onnx(
    model: PhysicsInformedSurrogate,
    sample_input: np.ndarray,
    output_path: str,
    normalization: Dict,
) -> None:
    """
    Export trained model to ONNX format.

    Args:
        model: Trained PyTorch model
        sample_input: Sample input for tracing
        output_path: Output file path
        normalization: Normalization parameters (saved as metadata)
    """
    logger.info(f"Exporting model to {output_path}...")
    model.eval()

    dummy_input = torch.from_numpy(sample_input[:1])
    physics_params = torch.zeros(1, 3)

    torch.onnx.export(
        model,
        (dummy_input, physics_params),
        output_path,
        input_names=["input", "physics_params"],
        output_names=["heating_load", "cooling_load"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "heating_load": {0: "batch_size"},
            "cooling_load": {0: "batch_size"},
        },
        opset_version=17,
    )

    # Save normalization metadata
    metadata_path = output_path.replace(".onnx", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(normalization, f, indent=2)

    logger.info(f"Export complete. Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train ML surrogate model for Fluxion with physics-informed constraints"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/training",
        help="Directory containing training data from ASHRAE 140 validation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/surrogate.onnx",
        help="Output path for trained ONNX model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 64, 32],
        help="Hidden layer dimensions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device (cuda or cpu)",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load training data
    try:
        X, y_heating, y_cooling, normalization = load_training_data(args.data_dir)
    except FileNotFoundError as e:
        logger.error(f"Failed to load training data: {e}")
        logger.error(
            "Please run ASHRAE 140 validation first to generate training data."
        )
        sys.exit(1)

    # Train model
    model, metrics = train_surrogate(
        X,
        y_heating,
        y_cooling,
        args.hidden_dims,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        device,
    )

    # Export to ONNX
    export_onnx(model, X, str(output_path), normalization)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Final R²: {metrics['final_r_squared']:.4f}")
    print(f"Final MAE: {metrics['final_mae']:.6f}")
    print(f"Best Validation Loss: {metrics['best_val_loss']:.6f}")

    # Check success metric
    if metrics["final_r_squared"] > 0.98:
        print(f"\n✓ SUCCESS: R² ({metrics['final_r_squared']:.4f}) > 0.98 threshold")
        print("Model is ready for production use as surrogate.")
    else:
        print(f"\n✗ WARNING: R² ({metrics['final_r_squared']:.4f}) < 0.98 threshold")
        print("Model may not be sufficiently accurate for production use.")

    print(f"\nModel exported to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
