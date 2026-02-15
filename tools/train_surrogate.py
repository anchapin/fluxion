#!/usr/bin/env python3
"""
Phase 4: Surrogate Model Training

This script trains a neural network surrogate model that predicts thermal loads.
It can generate synthetic training data using a simplified analytical model
or load existing data from a file.

The trained model is exported to ONNX format for integration with Fluxion.
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_samples: int = 5000, n_zones: int = 10, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data using a simplified thermal physics model.

    Args:
        n_samples: Number of samples to generate
        n_zones: Number of thermal zones
        seed: Random seed

    Returns:
        Tuple of (X, y) numpy arrays
    """
    logger.info(f"Generating {n_samples} synthetic samples for {n_zones} zones...")
    np.random.seed(seed)

    X_data = []
    y_data = []

    # Simplified physics simulation
    # In a real scenario, this would call the Rust engine
    for _ in tqdm(range(n_samples), desc="Generating Data"):
        # Inputs: U-value, HVAC Setpoint, Outdoor Temp
        u_value = np.random.uniform(0.5, 3.0)
        hvac_setpoint = np.random.uniform(19.0, 24.0)
        outdoor_temp = np.random.uniform(-10.0, 35.0)

        # Features: [u_value, hvac_setpoint, outdoor_temp] + current_temps
        # (simplified to mean)
        # For simplicity in this mock generator, we just use global params
        features = [u_value, hvac_setpoint, outdoor_temp]

        # Outputs: Thermal load for each zone
        # Physics approximation: Load ~ U * Area * (T_out - T_in)
        # We add some random variation per zone to simulate different geometries
        loads = []
        for z in range(n_zones):
            zone_factor = 1.0 + np.sin(z) * 0.2  # Variation
            delta_t = outdoor_temp - hvac_setpoint
            load = u_value * zone_factor * delta_t * -1.0  # Heating/Cooling load
            # Add noise
            load += np.random.normal(0, 0.1)
            loads.append(load)

        X_data.append(features)
        y_data.append(loads)

    return np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.float32)


def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load data from .npz or .csv file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    logger.info(f"Loading data from {path}...")
    if path.suffix == ".npz":
        data = np.load(path)
        return data["X"].astype(np.float32), data["y"].astype(np.float32)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        # Assume last N columns are targets, rest are features
        # This is brittle, expecting specific format, but sufficient for prototype
        target_cols = [c for c in df.columns if "load" in c or "target" in c]
        feature_cols = [c for c in df.columns if c not in target_cols]

        X = df[feature_cols].values.astype(np.float32)
        y = df[target_cols].values.astype(np.float32)
        return X, y
    else:
        raise ValueError("Unsupported file format. Use .npz or .csv")


class SurrogateModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PhysicsLoss(nn.Module):
    """
    Physics-Informed Loss function for thermal surrogates.
    
    Combines standard MSE (data loss) with a physics regularization term
    that penalizes energy balance violations.
    """
    def __init__(self, lambda_physics: float = 0.1):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Data Fitting Loss (MSE)
        data_loss = self.mse(pred, target)

        # 2. Physics Regularization Loss
        # Features: [u_value, hvac_setpoint, outdoor_temp]
        u_value = features[:, 0:1]
        hvac_setpoint = features[:, 1:2]
        outdoor_temp = features[:, 2:3]

        # Theoretical load based on steady-state heat balance:
        # Q_theory = U * (T_setpoint - T_outdoor)
        # Note: Synthetic data includes per-zone variation (zone_factor),
        # but the general physics law still holds for the aggregate or mean.
        theoretical_load_base = u_value * (hvac_setpoint - outdoor_temp)
        
        # Penalize deviation of average predicted load from theoretical base
        mean_pred_load = torch.mean(pred, dim=1, keepdim=True)
        physics_residual = mean_pred_load - theoretical_load_base
        physics_loss = torch.mean(physics_residual**2)

        # Combined loss
        total_loss = data_loss + self.lambda_physics * physics_loss
        
        return total_loss, data_loss, physics_loss


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dims: List[int],
    output_dir: Path,
    seed: int,
    use_physics_loss: bool = True,
    lambda_physics: float = 0.1,
):
    """Train the PyTorch model."""
    torch.manual_seed(seed)

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    # Dataloader
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    model = SurrogateModel(input_dim, output_dim, hidden_dims)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )
    
    # Loss criterion
    if use_physics_loss:
        criterion = PhysicsLoss(lambda_physics=lambda_physics)
        logger.info(f"Using Physics-Informed Loss (lambda={lambda_physics})")
    else:
        criterion = nn.MSELoss()
        logger.info("Using standard MSE Loss")

    logger.info(
        f"Model Architecture: Input={input_dim} -> {hidden_dims} -> Output={output_dim}"
    )
    logger.info("Starting training...")

    best_val_loss = float("inf")
    history: Dict[str, List[float]] = {"loss": [], "val_loss": [], "physics_loss": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_physics_loss = 0.0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            
            if use_physics_loss:
                loss, data_loss, phys_loss = criterion(pred, batch_y, batch_X)
                epoch_physics_loss += phys_loss.item()
            else:
                loss = criterion(pred, batch_y)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        avg_phys_loss = epoch_physics_loss / len(loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            if use_physics_loss:
                val_loss, _, _ = criterion(val_pred, y_val_t, X_val_t)
                val_loss = val_loss.item()
            else:
                val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step(val_loss)
        history["loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["physics_loss"].append(avg_phys_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        if (epoch + 1) % 10 == 0:
            msg = f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f} - Val Loss: {val_loss:.6f}"
            if use_physics_loss:
                msg += f" - Phys Loss: {avg_phys_loss:.6f}"
            logger.info(msg)

    # Load best model
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    logger.info(f"Training complete. Best Val Loss: {best_val_loss:.6f}")

    # Evaluation Metrics
    model.eval()
    with torch.no_grad():
        test_pred = model(X_val_t).numpy()

    mse = np.mean((y_val - test_pred) ** 2)
    mae = np.mean(np.abs(y_val - test_pred))
    r2 = 1 - (np.sum((y_val - test_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

    metrics = {"mse": float(mse), "mae": float(mae), "r2": float(r2)}
    logger.info(f"Validation Metrics: MAE={mae:.4f}, R2={r2:.4f}")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics


def export_onnx(model: nn.Module, sample_input: np.ndarray, output_path: Path):
    """Export to ONNX."""
    logger.info(f"Exporting to {output_path}...")
    model.eval()
    dummy_input = torch.from_numpy(sample_input[:1])

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
    )
    logger.info("Export successful.")


def main():
    parser = argparse.ArgumentParser(description="Train AI Surrogate Model")

    # Data Args
    parser.add_argument(
        "--input-file", type=str, help="Path to training data (.npz or .csv)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Samples to generate if no input file",
    )
    parser.add_argument(
        "--num-zones", type=int, default=10, help="Number of zones (output dim)"
    )

    # Training Args
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Hidden layer dimensions",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Physics Loss Args
    parser.add_argument(
        "--use-physics-loss",
        action="store_true",
        default=True,
        help="Use Physics-Informed Loss",
    )
    parser.add_argument(
        "--no-physics-loss",
        action="store_false",
        dest="use_physics_loss",
        help="Disable Physics-Informed Loss",
    )
    parser.add_argument(
        "--lambda-physics",
        type=float,
        default=0.1,
        help="Weight for physics loss term",
    )

    # Output Args
    parser.add_argument(
        "--output-dir", type=str, default="models", help="Output directory"
    )

    args = parser.parse_args()

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get Data
    if args.input_file:
        X, y = load_data(args.input_file)
    else:
        X, y = generate_synthetic_data(args.num_samples, args.num_zones, args.seed)
        # Save generated data
        np.savez(output_dir / "generated_data.npz", X=X, y=y)

    # Train
    model, metrics = train_model(
        X,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dims=args.hidden_dims,
        output_dir=output_dir,
        seed=args.seed,
        use_physics_loss=args.use_physics_loss,
        lambda_physics=args.lambda_physics,
    )

    # Export
    export_onnx(model, X, output_dir / "surrogate.onnx")

    # Plotting (Simple)
    try:
        import matplotlib.pyplot as plt

        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(X[:100])).numpy()

        plt.figure(figsize=(10, 5))
        plt.plot(y[:100, 0], label="Actual")
        plt.plot(pred[:, 0], label="Predicted")
        plt.title("Actual vs Predicted (Zone 0)")
        plt.legend()
        plt.savefig(output_dir / "prediction_plot.png")
        logger.info("Saved plot to prediction_plot.png")
    except ImportError:
        logger.warning("Matplotlib not installed, skipping plots")


if __name__ == "__main__":
    main()
