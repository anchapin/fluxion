#!/usr/bin/env python3
"""
Auto-Train Surrogate Tool

This CLI tool accepts a dataset (X, y) and automatically trains multiple
surrogate models, selects the best one, and exports it to ONNX.

Supported Algorithms:
- Multiple Linear Regression
- Ridge Regression
- Polynomial Ridge Regression
- Interaction Linear Regression
- Random Forest
- MLP (PyTorch)
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

# Scikit-learn
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

# ONNX export
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchMLPWrapper(BaseEstimator):
    """
    Scikit-learn wrapper for a simple PyTorch MLP to allow uniform evaluation.
    """
    def __init__(self, input_dim=10, output_dim=1, hidden_dim=64, epochs=100, lr=0.001, seed=42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X, y):
        torch.manual_seed(self.seed)

        # Scale data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1] if len(y.shape) > 1 else 1

        # Define model
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_t = torch.FloatTensor(X_scaled)
        y_t = torch.FloatTensor(y_scaled)

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X):
        self.model.eval()
        X_scaled = self.scaler_X.transform(X)
        with torch.no_grad():
            pred_scaled = self.model(torch.FloatTensor(X_scaled)).numpy()
        return self.scaler_y.inverse_transform(pred_scaled)

    def export_onnx(self, path, sample_input):
        self.model.eval()

        # Create a wrapping model that includes scaling if possible,
        # but for simplicity we often export just the model and handle scaling outside
        # or bake it into the network.
        # To match the sklearn pipelines, we ideally want an ONNX that takes raw input.
        # For now, we will export the raw core model.
        # Note: Integrating scaler into ONNX for PyTorch is doable but requires custom modules.
        # To keep parity with the sklearn pipeline approach which exports the FULL pipeline,
        # we really should bake scaling in.

        class ScaledModel(nn.Module):
            def __init__(self, core_model, mean_x, scale_x, mean_y, scale_y):
                super().__init__()
                self.core = core_model
                self.register_buffer('mean_x', torch.FloatTensor(mean_x))
                self.register_buffer('scale_x', torch.FloatTensor(scale_x))
                self.register_buffer('mean_y', torch.FloatTensor(mean_y))
                self.register_buffer('scale_y', torch.FloatTensor(scale_y))

            def forward(self, x):
                # Scale input: (x - mean) / scale
                x_scaled = (x - self.mean_x) / self.scale_x
                y_scaled = self.core(x_scaled)
                # Unscale output: y * scale + mean
                y_out = y_scaled * self.scale_y + self.mean_y
                return y_out

        full_model = ScaledModel(
            self.model,
            self.scaler_X.mean_, self.scaler_X.scale_,
            self.scaler_y.mean_, self.scaler_y.scale_
        )

        dummy_input = torch.FloatTensor(sample_input[:1])
        torch.onnx.export(
            full_model,
            dummy_input,
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=12
        )


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load X, y from .npz file."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found.")
    data = np.load(path)
    if 'X' not in data or 'y' not in data:
        raise ValueError("File must contain 'X' and 'y' arrays.")
    return data['X'].astype(np.float32), data['y'].astype(np.float32)

def create_pipelines() -> Dict[str, Any]:
    """Create dictionary of candidate pipelines."""

    pipelines = {}

    # 1. Multiple Linear Regression
    pipelines['LinearRegression'] = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # 2. Ridge Regression
    pipelines['Ridge'] = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0))
    ])

    # 3. Polynomial Ridge
    pipelines['PolynomialRidge'] = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('regressor', Ridge(alpha=1.0))
    ])

    # 4. Interaction Linear
    pipelines['InteractionLinear'] = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
        ('regressor', LinearRegression())
    ])

    # 5. Random Forest
    pipelines['RandomForest'] = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
    ])

    return pipelines

def evaluate_models(X, y, pipelines, cv=5):
    results = {}

    print(f"Evaluating models with {cv}-fold CV...")

    for name, pipeline in pipelines.items():
        print(f"  Testing {name}...", end="", flush=True)
        cv_scores = cross_validate(pipeline, X, y, cv=cv, scoring=['neg_mean_squared_error', 'r2'])
        mse = -cv_scores['test_neg_mean_squared_error'].mean()
        r2 = cv_scores['test_r2'].mean()
        results[name] = {'mse': mse, 'r2': r2}
        print(f" MSE: {mse:.4f}, R2: {r2:.4f}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Auto-Train Surrogate Model")
    parser.add_argument("--input", required=True, help="Input .npz file with X and y")
    parser.add_argument("--output", required=True, help="Output .onnx file")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    # 1. Load Data
    print("Loading data...")
    X, y = load_data(args.input)
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    # Ensure y is 2D
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # 2. define candidates
    candidates = create_pipelines()

    # Add PyTorch MLP manually since it's not a standard sklearn pipeline
    # We evaluate it separately or wrap it.
    # Since we built a wrapper, we can treat it similarly, but cross_validate might clone it.
    # Our wrapper logic handles fit/predict, so it should work.
    candidates['MLP_PyTorch'] = PyTorchMLPWrapper(epochs=50) # Lightweight for demo

    # 3. Evaluate
    results = evaluate_models(X, y, candidates, cv=args.cv)

    # 4. Select Best
    best_name = max(results, key=lambda k: results[k]['r2'])
    best_score = results[best_name]
    print(f"\nBest Model: {best_name} (R2: {best_score['r2']:.4f})")

    # 5. Train Final Model on Full Data
    print(f"Training final {best_name} model on full dataset...")
    final_model = candidates[best_name]
    final_model.fit(X, y)

    # 6. Export to ONNX
    print(f"Exporting to {args.output}...")

    if best_name == 'MLP_PyTorch':
        # Custom export for our wrapper
        final_model.export_onnx(args.output, X)
    else:
        # Standard sklearn export
        # We need to define input type
        initial_type = [('input', FloatTensorType([None, X.shape[1]]))]

        # Check if output is multi-target (RandomForest and Linear handle it, but skl2onnx needs care)
        # skl2onnx usually handles multi-output automatically for regressors.

        onnx_model = to_onnx(final_model, X[:1], initial_types=initial_type)
        with open(args.output, "wb") as f:
            f.write(onnx_model.SerializeToString())

    print("Done.")

if __name__ == "__main__":
    main()
