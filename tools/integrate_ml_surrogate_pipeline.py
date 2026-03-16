#!/usr/bin/env python3
"""
Bridge script for ASHRAE 140 validation training data collection and ONNX model training.

This script:
1. Runs ASHRAE 140 validation tests
2. Collects training data from successful runs
3. Trains ONNX surrogate models with the collected data
4. Validates model performance against deterministic physics (R² > 0.98)

Implements Issue #383: Integrate ML Surrogate FDD Pipeline
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_ashrae_140_validation() -> bool:
    """
    Run ASHRAE 140 validation tests and collect training data.

    Returns:
        True if validation passed, False otherwise
    """
    logger.info("Running ASHRAE 140 validation tests...")

    # Run validation tests
    result = subprocess.run(
        [
            "cargo",
            "test",
            "--test",
            "ashrae_140_validation",
            "--release",
            "--",
            "--nocapture",
        ],
        capture_output=True,
        text=True,
    )

    # Print output for debugging
    logger.info("Validation output:")
    logger.info(result.stdout)
    if result.stderr:
        logger.warning("Validation errors:")
        logger.warning(result.stderr)

    # Check if validation passed
    success = result.returncode == 0

    if success:
        logger.info("✓ ASHRAE 140 validation passed!")
    else:
        logger.warning("⚠ ASHRAE 140 validation failed, but may still have useful data")

    return success


def collect_training_data(output_dir: Path = Path("data/training")) -> List[Path]:
    """
    Collect training data files from ASHRAE 140 validation runs.

    Args:
        output_dir: Directory containing training data files

    Returns:
        List of training data file paths
    """
    logger.info(f"Scanning for training data in {output_dir}...")

    if not output_dir.exists():
        logger.warning(f"Training data directory does not exist: {output_dir}")
        return []

    # Find all training data CSV files
    csv_files = list(output_dir.glob("*_training_data.csv"))

    if not csv_files:
        logger.warning("No training data files found")
        return []

    logger.info(f"Found {len(csv_files)} training data file(s)")
    for csv_file in csv_files:
        logger.info(f"  - {csv_file.name}")

    return csv_files


def load_training_data(
    csv_files: List[Path],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and combine training data from multiple CSV files.

    Args:
        csv_files: List of training data CSV file paths

    Returns:
        Tuple of (X, y, case_ids)
    """
    all_data = []

    for csv_file in csv_files:
        logger.info(f"Loading training data from {csv_file}...")
        df = pd.read_csv(csv_file)
        all_data.append(df)

    if not all_data:
        raise ValueError("No training data to load")

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Extract features and targets
    X = combined_df[["u_value", "hvac_setpoint", "outdoor_temp"]].values.astype(
        np.float32
    )
    y = combined_df[["target"]].values.astype(np.float32)

    logger.info(f"Loaded {len(X)} total training samples")

    return X, y, list(combined_df["case_id"].unique())


def train_surrogate_model(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path = Path("models/surrogate"),
    use_physics_loss: bool = True,
    lambda_physics: float = 0.1,
    epochs: int = 100,
) -> Tuple[float, float]:
    """
    Train a surrogate model using the collected training data.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples, 1)
        output_dir: Directory to save the trained model
        use_physics_loss: Whether to use physics-informed loss
        lambda_physics: Weight for physics loss term
        epochs: Number of training epochs

    Returns:
        Tuple of (R² score, MAE)
    """
    logger.info("Training surrogate model...")

    # Import PyTorch components
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.error("PyTorch not installed. Please install: pip install torch")
        return 0.0, float("inf")

    # Prepare data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    # Create dataloader
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define model
    class SurrogateModel(nn.Module):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.1),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.1),
                nn.Linear(64, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    input_dim = X.shape[1]
    output_dim = y.shape[1]
    model = SurrogateModel(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    # Physics-informed loss function
    class PhysicsLoss(nn.Module):
        def __init__(self, lambda_physics: float = 0.1):
            super().__init__()
            self.lambda_physics = lambda_physics
            self.mse = nn.MSELoss()

        def forward(self, pred, target, features):
            # Data fitting loss
            data_loss = self.mse(pred, target)

            # Physics regularization
            # Q_theory = U * (T_setpoint - T_outdoor)
            u_value = features[:, 0:1]
            hvac_setpoint = features[:, 1:2]
            outdoor_temp = features[:, 2:3]
            theoretical_load = u_value * (hvac_setpoint - outdoor_temp)

            mean_pred_load = torch.mean(pred, dim=1, keepdim=True)
            physics_residual = mean_pred_load - theoretical_load
            physics_loss = torch.mean(physics_residual**2)

            # Combined loss
            total_loss = data_loss + self.lambda_physics * physics_loss
            return total_loss, data_loss, physics_loss

    criterion = (
        PhysicsLoss(lambda_physics=lambda_physics) if use_physics_loss else nn.MSELoss()
    )

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_phys_loss = 0.0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)

            if use_physics_loss:
                loss, data_loss, phys_loss = criterion(pred, batch_y, batch_X)
                epoch_phys_loss += phys_loss.item()
            else:
                loss = criterion(pred, batch_y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            if use_physics_loss:
                val_loss, _, _ = criterion(val_pred, y_val_t, X_val_t)
            else:
                val_loss = criterion(val_pred, y_val_t)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / "best_surrogate.pt")

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / len(loader):.6f} - Val Loss: {val_loss:.6f}"
            )

    # Load best model and compute metrics
    model.load_state_dict(torch.load(output_dir / "best_surrogate.pt"))
    model.eval()

    with torch.no_grad():
        val_pred = model(X_val_t).numpy()

    # Compute R² and MAE
    mse = np.mean((y_val - val_pred) ** 2)
    mae = np.mean(np.abs(y_val - val_pred))
    r2 = 1 - (np.sum((y_val - val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

    logger.info(f"Training complete!")
    logger.info(f"  R² Score: {r2:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  MSE: {mse:.4f}")

    return r2, mae


def export_onnx_model(
    model_path: Path,
    output_path: Path,
    input_shape: Tuple[int, int],
) -> bool:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model_path: Path to trained PyTorch model
        output_path: Path to save ONNX model
        input_shape: Input tensor shape (batch_size, n_features)

    Returns:
        True if export succeeded, False otherwise
    """
    logger.info(f"Exporting model to ONNX format: {output_path}...")

    try:
        import torch
        import torch.nn as nn

        # Define model architecture (must match training)
        class SurrogateModel(nn.Module):
            def __init__(self, input_dim: int, output_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.1),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.1),
                    nn.Linear(64, output_dim),
                )

            def forward(self, x):
                return self.net(x)

        # Load trained model
        model = SurrogateModel(input_shape[1], 1)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(input_shape)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
        )

        logger.info(f"✓ Model exported successfully to {output_path}")
        return True

    except ImportError:
        logger.error("PyTorch not installed. Cannot export ONNX model.")
        return False
    except Exception as e:
        logger.error(f"Failed to export ONNX model: {e}")
        return False


def benchmark_model(
    model_path: Path,
    test_data_path: Path = None,
) -> Tuple[float, float]:
    """
    Benchmark a trained ONNX model against test data.

    Args:
        model_path: Path to ONNX model
        test_data_path: Path to test data CSV (optional)

    Returns:
        Tuple of (R² score, MAE)
    """
    logger.info(f"Benchmarking model: {model_path}...")

    try:
        import onnxruntime as ort
    except ImportError:
        logger.error(
            "onnxruntime not installed. Please install: pip install onnxruntime"
        )
        return 0.0, float("inf")

    # Load model
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Load test data
    if test_data_path and test_data_path.exists():
        df = pd.read_csv(test_data_path)
        X = df[["u_value", "hvac_setpoint", "outdoor_temp"]].values.astype(np.float32)
        y_true = df["target"].values.astype(np.float32)
    else:
        # Generate synthetic test data
        logger.warning("No test data provided, generating synthetic test data")
        n_test = 1000
        X = np.random.randn(n_test, 3).astype(np.float32)
        X[:, 0] = 0.5 + 2.5 * (X[:, 0] - X[:, 0].min()) / (
            X[:, 0].max() - X[:, 0].min() + 1e-6
        )  # u_value
        X[:, 1] = 19.0 + 5.0 * (X[:, 1] - X[:, 1].min()) / (
            X[:, 1].max() - X[:, 1].min() + 1e-6
        )  # hvac_setpoint
        X[:, 2] = -10.0 + 45.0 * (X[:, 2] - X[:, 2].min()) / (
            X[:, 2].max() - X[:, 2].min() + 1e-6
        )  # outdoor_temp
        y_true = X[:, 0] * (X[:, 1] - X[:, 2])  # Physics-based ground truth

    # Run inference
    y_pred = session.run(None, {input_name: X})[0].flatten()

    # Compute metrics
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

    logger.info(f"Benchmarking complete!")
    logger.info(f"  R² Score: {r2:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  MSE: {mse:.4f}")

    return r2, mae


def main():
    parser = argparse.ArgumentParser(
        description="AShRAE 140 ML Surrogate Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run ASHRAE 140 validation tests",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect training data from validation runs",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train surrogate model from collected data",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        metavar="MODEL_PATH",
        help="Benchmark an ONNX model",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full pipeline: validate, collect, train, benchmark",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/training",
        help="Directory for training data (default: data/training)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/surrogate",
        help="Directory for model outputs (default: models/surrogate)",
    )
    parser.add_argument(
        "--use-physics-loss",
        action="store_true",
        default=True,
        help="Use physics-informed loss during training",
    )
    parser.add_argument(
        "--lambda-physics",
        type=float,
        default=0.1,
        help="Weight for physics loss term (default: 0.1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )

    args = parser.parse_args()

    # Create directories
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Run full pipeline
    if args.all or (args.validate and args.collect and args.train):
        logger.info("=" * 70)
        logger.info("Running Full ML Surrogate FDD Pipeline")
        logger.info("=" * 70)

        # Step 1: Run validation
        logger.info("\n[Step 1/4] Running ASHRAE 140 validation...")
        validation_passed = run_ashrae_140_validation()

        # Step 2: Collect training data
        logger.info("\n[Step 2/4] Collecting training data...")
        training_files = collect_training_data(data_dir)

        if not training_files:
            logger.error("No training data collected. Exiting.")
            return 1

        # Step 3: Train model
        logger.info("\n[Step 3/4] Training surrogate model...")
        X, y, case_ids = load_training_data(training_files)
        logger.info(f"Training with data from cases: {', '.join(case_ids)}")

        r2, mae = train_surrogate_model(
            X,
            y,
            output_dir=model_dir,
            use_physics_loss=args.use_physics_loss,
            lambda_physics=args.lambda_physics,
            epochs=args.epochs,
        )

        # Check if R² meets threshold (Issue #383 success metric)
        if r2 < 0.98:
            logger.warning(
                f"⚠ R² ({r2:.4f}) below threshold 0.98. "
                "Model may not maintain sufficient physics accuracy."
            )
        else:
            logger.info(f"✓ R² ({r2:.4f}) meets threshold 0.98")

        # Step 4: Export and benchmark
        logger.info("\n[Step 4/4] Exporting and benchmarking ONNX model...")
        onnx_path = model_dir / "surrogate.onnx"
        export_onnx_model(
            model_dir / "best_surrogate.pt",
            onnx_path,
            input_shape=(1, X.shape[1]),
        )

        benchmark_r2, benchmark_mae = benchmark_model(onnx_path)

        logger.info("\n" + "=" * 70)
        logger.info("Pipeline Complete!")
        logger.info("=" * 70)
        logger.info(f"Final R²: {benchmark_r2:.4f} (target: >0.98)")
        logger.info(f"Final MAE: {benchmark_mae:.4f}")

        return 0 if benchmark_r2 >= 0.98 else 1

    # Individual steps
    if args.validate:
        run_ashrae_140_validation()

    if args.collect:
        training_files = collect_training_data(data_dir)
        if training_files:
            X, y, case_ids = load_training_data(training_files)
            logger.info(f"Collected {len(X)} samples from {len(case_ids)} cases")
        else:
            logger.warning("No training data collected")

    if args.train:
        training_files = collect_training_data(data_dir)
        if training_files:
            X, y, case_ids = load_training_data(training_files)
            train_surrogate_model(
                X,
                y,
                output_dir=model_dir,
                use_physics_loss=args.use_physics_loss,
                lambda_physics=args.lambda_physics,
                epochs=args.epochs,
            )
        else:
            logger.error("No training data available for training")

    if args.benchmark:
        onnx_path = Path(args.benchmark)
        if onnx_path.exists():
            r2, mae = benchmark_model(onnx_path)
            return 0 if r2 >= 0.98 else 1
        else:
            logger.error(f"Model file not found: {onnx_path}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
