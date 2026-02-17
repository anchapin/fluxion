#!/usr/bin/env python3
"""
Phase 7: MC Dropout Training for Uncertainty Estimation

This script trains a neural network surrogate model with dropout layers
enabled during inference for Monte Carlo Dropout (MCD) uncertainty estimation.

MC Dropout provides a practical way to estimate prediction uncertainty
without requiring ensemble training.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
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


class MCDropoutModel(nn.Module):
    """
    Neural network with dropout layers for Monte Carlo Dropout.
    
    Dropout is applied during training to prevent overfitting,
    and also during inference (without updating weights) to generate
    multiple stochastic forward passes for uncertainty estimation.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int],
        dropout_rate: float = 0.1
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            # MC Dropout: keep dropout enabled during inference
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        # Store dropout rate for inference
        self.dropout_rate = dropout_rate

    def forward(self, x):
        return self.net(x)
    
    def enable_dropout(self):
        """Enable dropout layers for MC Dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def disable_dropout(self):
        """Disable dropout for standard inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()


class MCDropoutLoss(nn.Module):
    """
    Combined loss for MC Dropout training.
    
    Includes:
    - MSE loss for prediction accuracy
    - Variance regularization to encourage confident predictions
    """
    def __init__(self, lambda_variance: float = 0.01):
        super().__init__()
        self.lambda_variance = lambda_variance
        self.mse = nn.MSELoss()

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Main prediction loss
        data_loss = self.mse(pred, target)
        
        # Variance regularization: encourage low variance in predictions
        # This is applied per batch to push the model toward confident predictions
        variance = torch.var(pred, dim=0).mean()
        variance_loss = variance
        
        total_loss = data_loss + self.lambda_variance * variance_loss
        
        return total_loss, data_loss, variance_loss


def train_mc_dropout_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dims: List[int],
    dropout_rate: float,
    output_dir: Path,
    seed: int,
):
    """Train the MC Dropout model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    model = MCDropoutModel(input_dim, output_dim, hidden_dims, dropout_rate)
    
    # Set model to training mode (dropout enabled)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )
    
    criterion = MCDropoutLoss(lambda_variance=0.01)

    logger.info(
        f"MC Dropout Model: Input={input_dim} -> {hidden_dims} -> Output={output_dim}"
    )
    logger.info(f"Dropout rate: {dropout_rate}")
    logger.info("Starting training...")

    best_val_loss = float("inf")
    history: Dict[str, List[float]] = {
        "loss": [], 
        "val_loss": [], 
        "variance_loss": []
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_variance_loss = 0.0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            
            loss, data_loss, var_loss = criterion(pred, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_variance_loss += var_loss.item()

        avg_loss = epoch_loss / len(loader)
        avg_var_loss = epoch_variance_loss / len(loader)

        # Validation (with dropout enabled for MCD training)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss, _, _ = criterion(val_pred, y_val_t)
            val_loss = val_loss.item()

        scheduler.step(val_loss)
        history["loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["variance_loss"].append(avg_var_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_mc_dropout_model.pt")

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - Var Loss: {avg_var_loss:.6f}"
            )

    # Load best model
    model.load_state_dict(torch.load(output_dir / "best_mc_dropout_model.pt"))
    logger.info(f"Training complete. Best Val Loss: {best_val_loss:.6f}")

    # Evaluation Metrics
    model.eval()
    with torch.no_grad():
        # Standard prediction (dropout disabled)
        model.disable_dropout()
        test_pred = model(X_val_t).numpy()
        
        # MC Dropout prediction (multiple passes with dropout)
        model.enable_dropout()
        mc_samples = []
        for _ in range(20):
            with torch.no_grad():
                mc_pred = model(X_val_t).numpy()
                mc_samples.append(mc_pred)
        
        mc_samples = np.array(mc_samples)  # Shape: (20, batch, output)
        mc_mean = np.mean(mc_samples, axis=0)
        mc_std = np.std(mc_samples, axis=0)

    mse = np.mean((y_val - test_pred) ** 2)
    mae = np.mean(np.abs(y_val - test_pred))
    r2 = 1 - (np.sum((y_val - test_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

    # Uncertainty metrics
    avg_uncertainty = np.mean(mc_std)
    
    metrics = {
        "mse": float(mse), 
        "mae": float(mae), 
        "r2": float(r2),
        "avg_uncertainty": float(avg_uncertainty)
    }
    
    logger.info(f"Validation Metrics: MAE={mae:.4f}, R2={r2:.4f}")
    logger.info(f"Average Uncertainty (MC Dropout): {avg_uncertainty:.4f}")

    with open(output_dir / "metrics_mc_dropout.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics


def export_mc_dropout_onnx(model: nn.Module, sample_input: np.ndarray, output_path: Path):
    """Export MC Dropout model to ONNX."""
    logger.info(f"Exporting MC Dropout model to {output_path}...")
    model.eval()
    
    # For ONNX export, we need to handle dropout specially
    # ONNX Runtime doesn't support torch dropout during inference natively
    # So we export as a standard model and handle MCD in the Rust runtime
    dummy_input = torch.from_numpy(sample_input[:1])

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        # Dropout is exported with training_mode=False by default
        # We handle MCD in the runtime by running multiple inferences
        training=torch.onnx.TrainingMode.PRESERVE,
    )
    logger.info("Export successful. MC Dropout will be handled at runtime.")


def mc_dropout_inference(
    model: nn.Module,
    X: np.ndarray,
    num_samples: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Monte Carlo Dropout inference.
    
    Returns mean predictions and standard deviations across multiple
    stochastic forward passes.
    
    Args:
        model: Trained MC Dropout model
        X: Input features
        num_samples: Number of stochastic forward passes
    
    Returns:
        Tuple of (mean predictions, standard deviations)
    """
    model.eval()
    model.enable_dropout()  # Keep dropout layers in training mode
    
    X_t = torch.from_numpy(X)
    samples = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(X_t).numpy()
            samples.append(pred)
    
    samples = np.array(samples)
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    
    model.disable_dropout()  # Return to standard inference mode
    
    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Train MC Dropout Surrogate Model")

    # Data Args
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to training data (.npz, .csv, or directory)",
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
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.1,
        help="Dropout rate for MC Dropout",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output Args
    parser.add_argument(
        "--output-dir", type=str, default="models", help="Output directory"
    )

    args = parser.parse_args()

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get Data (use simplified generation for demo)
    if args.input_file:
        import pandas as pd
        path = Path(args.input_file)
        if path.suffix == ".npz":
            data = np.load(path)
            X, y = data["X"].astype(np.float32), data["y"].astype(np.float32)
        else:
            logger.warning("Using synthetic data generation for demo")
            X = np.random.randn(args.num_samples, 3).astype(np.float32)
            y = (X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(args.num_samples) * 0.1).reshape(-1, 1).astype(np.float32)
    else:
        # Generate synthetic data
        np.random.seed(args.seed)
        X = np.random.randn(args.num_samples, 3).astype(np.float32)
        # Simple linear relationship with noise
        y = (X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(args.num_samples) * 0.1).reshape(-1, 1).astype(np.float32)

    # Train
    model, metrics = train_mc_dropout_model(
        X,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate,
        output_dir=output_dir,
        seed=args.seed,
    )

    # Export
    export_mc_dropout_onnx(model, X, output_dir / "surrogate_mc_dropout.onnx")

    # Save training config for later use
    config = {
        "input_dim": int(X.shape[1]),
        "output_dim": int(y.shape[1]),
        "hidden_dims": args.hidden_dims,
        "dropout_rate": args.dropout_rate,
        "num_mc_samples": 20,  # Default for inference
    }
    with open(output_dir / "mc_dropout_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("MC Dropout training complete!")
    logger.info(f"Model saved to {output_dir}")
    logger.info(f"Use {config['num_mc_samples']} forward passes at inference for uncertainty")


if __name__ == "__main__":
    main()
