#!/usr/bin/env python3
"""
Phase 7: Ensemble Training for Multiple Surrogate Models

This script trains multiple neural network surrogate models that can be used
together in an ensemble for improved predictions with disagreement-based
uncertainty estimation.
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


class SurrogateModel(nn.Module):
    """Standard surrogate model for ensemble training."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
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


def train_single_model(
    X: np.ndarray,
    y: np.ndarray,
    model: nn.Module,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_dir: Path,
    model_name: str,
    seed: int,
) -> Tuple[nn.Module, Dict]:
    """Train a single model in the ensemble."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split data with different split for diversity
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

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / f"best_{model_name}.pt")

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f} - Val: {val_loss:.6f}")

    # Load best model
    model.load_state_dict(torch.load(output_dir / f"best_{model_name}.pt"))
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_val_t).numpy()

    mse = np.mean((y_val - test_pred) ** 2)
    mae = np.mean(np.abs(y_val - test_pred))
    r2 = 1 - (np.sum((y_val - test_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

    metrics = {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
    }
    
    return model, metrics


def export_onnx(model: nn.Module, sample_input: np.ndarray, output_path: Path):
    """Export model to ONNX."""
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


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    num_models: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dims: List[int],
    output_dir: Path,
    seed: int = 42,
):
    """Train an ensemble of models."""
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    
    all_metrics = []
    
    logger.info(f"Training ensemble of {num_models} models...")
    
    for i in range(num_models):
        model_seed = seed + i * 100  # Different seed for each model
        model_name = f"model_{i}"
        
        logger.info(f"Training {model_name} (seed={model_seed})...")
        
        # Create model with different initialization
        model = SurrogateModel(input_dim, output_dim, hidden_dims, seed=model_seed)
        
        # Train
        trained_model, metrics = train_single_model(
            X, y, model, epochs, batch_size, learning_rate,
            output_dir, model_name, model_seed
        )
        
        # Export to ONNX
        onnx_path = output_dir / f"ensemble_{model_name}.onnx"
        export_onnx(trained_model, X, onnx_path)
        
        metrics["model_path"] = str(onnx_path)
        all_metrics.append(metrics)
        
        logger.info(f"  {model_name} - MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
    
    # Save ensemble config
    ensemble_config = {
        "num_models": num_models,
        "model_paths": [m["model_path"] for m in all_metrics],
        "aggregation_method": "mean",
    }
    
    with open(output_dir / "ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2)
    
    # Save detailed metrics
    with open(output_dir / "ensemble_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f"Ensemble training complete! {num_models} models saved to {output_dir}")
    
    return ensemble_config, all_metrics


def evaluate_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    model_paths: List[str],
) -> Dict:
    """Evaluate ensemble predictions and disagreement."""
    import torch
    
    models = []
    for path in model_paths:
        model = SurrogateModel(
            X.shape[1], y.shape[1], [64, 64]
        )
        model.load_state_dict(torch.load(path.replace('.onnx', '.pt')))
        model.eval()
        models.append(model)
    
    X_t = torch.from_numpy(X)
    
    # Get predictions from all models
    all_predictions = []
    with torch.no_grad():
        for model in models:
            pred = model(X_t).numpy()
            all_predictions.append(pred)
    
    all_predictions = np.array(all_predictions)  # Shape: (num_models, num_samples, output_dim)
    
    # Calculate ensemble statistics
    mean_pred = np.mean(all_predictions, axis=0)
    std_pred = np.std(all_predictions, axis=0)
    
    # Calculate metrics
    mse = np.mean((y - mean_pred) ** 2)
    mae = np.mean(np.abs(y - mean_pred))
    r2 = 1 - (np.sum((y - mean_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
    
    # Disagreement metrics
    mean_disagreement = np.mean(std_pred)
    max_disagreement = np.max(std_pred)
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
        "mean_disagreement": float(mean_disagreement),
        "max_disagreement": float(max_disagreement),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Ensemble Surrogate Models")

    # Data Args
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to training data (.npz, .csv)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Samples to generate if no input file",
    )

    # Ensemble Args
    parser.add_argument(
        "--num-models",
        type=int,
        default=5,
        help="Number of models in ensemble",
    )

    # Training Args
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per model")
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
        path = Path(args.input_file)
        if path.suffix == ".npz":
            data = np.load(path)
            X, y = data["X"].astype(np.float32), data["y"].astype(np.float32)
        else:
            logger.warning("Using synthetic data generation for demo")
            np.random.seed(args.seed)
            X = np.random.randn(args.num_samples, 3).astype(np.float32)
            y = (X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(args.num_samples) * 0.1).reshape(-1, 1).astype(np.float32)
    else:
        # Generate synthetic data
        np.random.seed(args.seed)
        X = np.random.randn(args.num_samples, 3).astype(np.float32)
        y = (X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(args.num_samples) * 0.1).reshape(-1, 1).astype(np.float32)

    # Train ensemble
    ensemble_config, metrics = train_ensemble(
        X, y,
        num_models=args.num_models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dims=args.hidden_dims,
        output_dir=output_dir,
        seed=args.seed,
    )

    # Evaluate ensemble
    split_idx = int(0.8 * len(X))
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    
    eval_metrics = evaluate_ensemble(
        X_val, y_val,
        [m["model_path"] for m in metrics]
    )
    
    logger.info(f"Ensemble Evaluation: MAE={eval_metrics['mae']:.4f}, R2={eval_metrics['r2']:.4f}")
    logger.info(f"Disagreement: mean={eval_metrics['mean_disagreement']:.4f}, max={eval_metrics['max_disagreement']:.4f}")
    
    with open(output_dir / "ensemble_evaluation.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    logger.info("Ensemble training complete!")
    logger.info(f"Use the model paths in {output_dir / 'ensemble_config.json'}")


if __name__ == "__main__":
    main()
