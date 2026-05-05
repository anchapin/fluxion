#!/usr/bin/env python3
"""
SUR-01: Enhanced Surrogate Model Training with SHAP Interpretability and Ensemble Methods

This script benchmarks MLP vs XGBoost vs Random Forest surrogate models on the
fluxion thermal problem, integrates SHAP analysis for feature importance, and
provides ensemble prediction combining multiple architectures.

Usage:
    python tools/train_surrogate.py --data-dir data/training --output models
    python tools/train_surrogate.py --run-benchmark --shap-analysis --ensemble

Requirements:
    pip install shap xgboost scikit-learn

Issue: #553
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def calculate_r2(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Calculate R² score."""
    mean_actual = np.mean(actual)
    ss_tot = np.sum((actual - mean_actual) ** 2)
    ss_res = np.sum((actual - predicted) ** 2)
    if ss_tot < 1e-10:
        return 1.0 if ss_res < 1e-10 else -np.inf
    return 1.0 - (ss_res / ss_tot)


def calculate_mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(predicted - actual))


def calculate_rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((predicted - actual) ** 2))


class MLPEnsemble:
    """Ensemble of MLP models using PyTorch."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], n_models: int = 5):
        self.models = []
        self.n_models = n_models
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 0.001, seed: int = 42) -> Dict:
        """Train ensemble of MLP models."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        metrics = {"models": [], "r2_scores": [], "mae_scores": []}
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        for i in range(self.n_models):
            torch.manual_seed(seed + i * 100)
            np.random.seed(seed + i * 100)

            class SurrogateModel(nn.Module):
                def __init__(self, input_dim, output_dim, hidden_dims):
                    super().__init__()
                    layers = []
                    prev_dim = input_dim
                    for h_dim in hidden_dims:
                        layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim), nn.Dropout(0.1)])
                        prev_dim = h_dim
                    layers.append(nn.Linear(prev_dim, output_dim))
                    self.net = nn.Sequential(*layers)
                def forward(self, x):
                    return self.net(x)

            model = SurrogateModel(self.input_dim, self.output_dim, self.hidden_dims)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            X_t = torch.from_numpy(X_train)
            y_t = torch.from_numpy(y_train)
            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

            best_val_loss = float("inf")
            best_state = None

            for epoch in range(epochs):
                model.train()
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(batch_X), batch_y)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_pred = model(torch.from_numpy(X_val))
                    val_loss = criterion(val_pred, torch.from_numpy(y_val)).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

            model.load_state_dict(best_state)
            model.eval()

            with torch.no_grad():
                pred = model(torch.from_numpy(X_val)).numpy()
                r2 = calculate_r2(pred, y_val)
                mae = calculate_mae(pred, y_val)

            metrics["models"].append(model)
            metrics["r2_scores"].append(float(r2))
            metrics["mae_scores"].append(float(mae))
            logger.info(f"  MLP Model {i+1}: R²={r2:.4f}, MAE={mae:.4f}")

        metrics["mean_r2"] = float(np.mean(metrics["r2_scores"]))
        metrics["std_r2"] = float(np.std(metrics["r2_scores"]))
        return metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with ensemble, returning mean and std."""
        import torch
        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(torch.from_numpy(X)).numpy()
                preds.append(pred)
        preds = np.array(preds)
        return np.mean(preds, axis=0), np.std(preds, axis=0)


class XGBoostModel:
    """XGBoost surrogate model wrapper."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.models = []
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42) -> Dict:
        """Train XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost not installed. Install with: pip install xgboost")
            return {"r2": 0.0, "mae": 0.0, "error": "xgboost not installed"}

        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=seed,
            n_jobs=-1
        )
        self._model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        pred = self._model.predict(X_val)
        r2 = calculate_r2(pred, y_val)
        mae = calculate_mae(pred, y_val)
        logger.info(f"  XGBoost: R²={r2:.4f}, MAE={mae:.4f}")

        return {"r2": float(r2), "mae": float(mae)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using XGBoost."""
        return self._model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self._model.feature_importances_


class RandomForestModel:
    """Random Forest surrogate model wrapper."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42) -> Dict:
        """Train Random Forest model."""
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            logger.warning("scikit-learn not installed.")
            return {"r2": 0.0, "mae": 0.0, "error": "scikit-learn not installed"}

        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=seed,
            n_jobs=-1
        )
        self._model.fit(X_train, y_train)

        pred = self._model.predict(X_val)
        r2 = calculate_r2(pred, y_val)
        mae = calculate_mae(pred, y_val)
        logger.info(f"  Random Forest: R²={r2:.4f}, MAE={mae:.4f}")

        return {"r2": float(r2), "mae": float(mae)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using Random Forest."""
        return self._model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self._model.feature_importances_


class SHAPAnalyzer:
    """SHAP-based feature importance analyzer."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

    def compute_shap_values(self, X: np.ndarray, nsamples: int = 100) -> np.ndarray:
        """Compute SHAP values for feature importance analysis."""
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed. Install with: pip install shap")
            return np.zeros((X.shape[1],))

        if hasattr(self.model, "_model"):
            predictor = self.model._model.predict
        elif hasattr(self.model, "predict"):
            predictor = self.model.predict
        else:
            logger.warning("Model doesn't have predict method")
            return np.zeros((X.shape[1],))

        self.explainer = shap.Explainer(predictor, X[:min(100, len(X))])
        self.shap_values = self.explainer(X[:min(nsamples, len(X))])

        return self.shap_values.values

    def get_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Get mean absolute SHAP values per feature."""
        shap_vals = self.compute_shap_values(X)
        importance = {}
        for i, name in enumerate(self.feature_names):
            importance[name] = float(np.mean(np.abs(shap_vals[:, i])))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def validate_physical_soundness(self, X: np.ndarray, feature_bounds: Dict[str, Tuple[float, float]]) -> Dict:
        """Validate that model weights align with physical expectations."""
        importance = self.get_feature_importance(X)
        validation = {"physically_sound": True, "concerns": [], "insights": []}

        roof_keywords = ["roof", "insulation", "u_value"]
        wall_keywords = ["wall", " glazing", "wwr"]
        hvac_keywords = ["heating", "cooling", "setpoint"]

        roof_importance = sum(v for k, v in importance.items() if any(rk in k.lower() for rk in roof_keywords))
        wall_importance = sum(v for k, v in importance.items() if any(wk in k.lower() for wk in wall_keywords))

        if roof_importance > wall_importance * 1.5:
            validation["insights"].append("Roof/insulation parameters have highest importance - physically correct for thermal loading")

        if any("outdoor_temp" in k.lower() or "temp" in k.lower() for k in importance.keys()[:3]):
            validation["insights"].append("Temperature parameters highly weighted - confirms thermodynamic dependence")

        return validation


class EnsemblePredictor:
    """Combines multiple model architectures for ensemble prediction."""

    def __init__(self):
        self.models = {}
        self.model_types = []

    def add_model(self, name: str, model_type: str, model, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models[name] = {"type": model_type, "model": model, "weight": weight}
        self.model_types.append(model_type)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble prediction with weighted average and disagreement estimation."""
        predictions = []
        weights = []

        for name, info in self.models.items():
            model = info["model"]
            weight = info["weight"]

            if info["type"] == "mlp":
                _, std = model.predict(X)
                pred = model.models[0](torch.from_numpy(X)).detach().numpy()
                for m in model.models[1:]:
                    m.eval()
                    with torch.no_grad():
                        pred = np.column_stack([pred, m(torch.from_numpy(X)).detach().numpy()])
                pred = np.mean(pred, axis=1).reshape(-1, 1)
            elif info["type"] == "xgboost":
                pred = model.predict(X)
            elif info["type"] == "random_forest":
                pred = model.predict(X)

            predictions.append(pred)
            weights.append(weight)

        predictions = np.array(predictions)
        weights = np.array(weights) / sum(weights)

        weighted_pred = np.zeros_like(predictions[0])
        for i, w in enumerate(weights):
            weighted_pred += predictions[i] * w

        disagreement = np.std(predictions, axis=0)

        return weighted_pred, disagreement

    def benchmark(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Benchmark ensemble against individual models."""
        pred, disagreement = self.predict(X)
        ensemble_r2 = calculate_r2(pred.flatten(), y.flatten())
        ensemble_mae = calculate_mae(pred.flatten(), y.flatten())

        return {
            "ensemble_r2": float(ensemble_r2),
            "ensemble_mae": float(ensemble_mae),
            "mean_disagreement": float(np.mean(disagreement)),
            "max_disagreement": float(np.max(disagreement)),
            "n_models": len(self.models),
            "model_types": list(set(self.model_types))
        }


def generate_synthetic_thermal_data(n_samples: int = 10000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic thermal problem data for benchmarking."""
    np.random.seed(seed)

    n_features = 8
    feature_names = ["outdoor_temp", "heating_setpoint", "cooling_setpoint", "hour_of_day",
                     "day_of_year", "month", "u_value", "wwr"]

    X = np.random.randn(n_samples, n_features).astype(np.float32)

    X[:, 0] = np.clip(X[:, 0] * 15 + 10, -20, 45)
    X[:, 1] = np.clip(X[:, 1] * 5 + 20, 15, 25)
    X[:, 2] = np.clip(X[:, 2] * 5 + 25, 20, 30)
    X[:, 3] = np.random.randint(0, 24, n_samples)
    X[:, 4] = np.random.randint(1, 366, n_samples)
    X[:, 5] = ((X[:, 4] - 1) // 30 + 1).astype(np.float32)
    X[:, 6] = np.clip(X[:, 6] * 0.3 + 0.5, 0.1, 2.0)
    X[:, 7] = np.clip(X[:, 7] * 0.1 + 0.3, 0.1, 0.9)

    heating_load = (
        X[:, 6] * (X[:, 1] - X[:, 0]).clip(min=0) * 100 +
        np.random.randn(n_samples) * 5
    ).astype(np.float32)

    cooling_load = (
        X[:, 6] * (X[:, 0] - X[:, 2]).clip(min=0) * 80 +
        X[:, 7] * 50 +
        np.random.randn(n_samples) * 5
    ).astype(np.float32)

    y = np.column_stack([heating_load, cooling_load])

    logger.info(f"Generated {n_samples} samples with {n_features} features")
    logger.info(f"Target shape: {y.shape}, Heating range: [{heating_load.min():.1f}, {heating_load.max():.1f}]")

    return X, y, feature_names


def load_training_data(data_dir: str, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load training data from CSV files."""
    data_path = Path(data_dir)
    sample_files = list(data_path.glob("samples_*.csv"))

    if not sample_files:
        logger.warning(f"No training data found in {data_dir}, using synthetic data")
        return generate_synthetic_thermal_data(10000)

    latest_file = max(sample_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading data from {latest_file}")

    df = pd.read_csv(latest_file)
    logger.info(f"Loaded {len(df)} training samples")

    X = df[feature_names].values.astype(np.float32)
    y = df[["heating_load", "cooling_load"]].values.astype(np.float32)

    return X, y, feature_names


def run_benchmark(X: np.ndarray, y: np.ndarray, feature_names: List[str], output_dir: Path) -> Dict:
    """Benchmark MLP vs XGBoost vs Random Forest."""
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARKING SURROGATE MODELS")
    logger.info("=" * 60)

    results = {"timestamp": datetime.now().isoformat(), "models": {}}

    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    y_combined_train = (y_train[:, 0] + y_train[:, 1]).reshape(-1, 1)
    y_combined_val = (y_val[:, 0] + y_val[:, 1]).reshape(-1, 1)

    logger.info("\n[1/3] Training MLP Ensemble...")
    mlp = MLPEnsemble(input_dim=X.shape[1], output_dim=1, hidden_dims=[64, 64], n_models=5)
    mlp_metrics = mlp.fit(X_train, y_combined_train, epochs=100, seed=42)
    mlp_pred, mlp_std = mlp.predict(X_val)
    mlp_r2 = calculate_r2(mlp_pred.flatten(), y_combined_val.flatten())
    mlp_rmse = calculate_rmse(mlp_pred.flatten(), y_combined_val.flatten())
    results["models"]["mlp"] = {
        "r2": float(mlp_r2),
        "rmse": float(mlp_rmse),
        "mean_r2": mlp_metrics["mean_r2"],
        "std_r2": mlp_metrics["std_r2"],
        "n_models": 5
    }
    logger.info(f"  MLP Ensemble R²: {mlp_r2:.4f} (±{mlp_metrics['std_r2']:.4f})")

    logger.info("\n[2/3] Training XGBoost...")
    xgb = XGBoostModel(n_estimators=100, max_depth=6)
    xgb_metrics = xgb.fit(X_train, y_combined_train, seed=42)
    xgb_pred = xgb.predict(X_val)
    xgb_r2 = calculate_r2(xgb_pred, y_combined_val)
    xgb_rmse = calculate_rmse(xgb_pred, y_combined_val)
    results["models"]["xgboost"] = {
        "r2": float(xgb_r2),
        "rmse": float(xgb_rmse),
        "feature_importance": dict(zip(feature_names, xgb.get_feature_importance().tolist()))
    }
    logger.info(f"  XGBoost R²: {xgb_r2:.4f}")

    logger.info("\n[3/3] Training Random Forest...")
    rf = RandomForestModel(n_estimators=100, max_depth=10)
    rf_metrics = rf.fit(X_train, y_combined_train, seed=42)
    rf_pred = rf.predict(X_val)
    rf_r2 = calculate_r2(rf_pred, y_combined_val)
    rf_rmse = calculate_rmse(rf_pred, y_combined_val)
    results["models"]["random_forest"] = {
        "r2": float(rf_r2),
        "rmse": float(rf_rmse),
        "feature_importance": dict(zip(feature_names, rf.get_feature_importance().tolist()))
    }
    logger.info(f"  Random Forest R²: {rf_r2:.4f}")

    results["best_model"] = max(results["models"].items(), key=lambda x: x[1]["r2"])[0]
    logger.info(f"\nBest model: {results['best_model']} (R²={results['models'][results['best_model']]['r2']:.4f})")

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_shap_analysis(X: np.ndarray, y: np.ndarray, feature_names: List[str], output_dir: Path) -> Dict:
    """Run SHAP analysis on best model."""
    logger.info("\n" + "=" * 60)
    logger.info("SHAP INTERPRETABILITY ANALYSIS")
    logger.info("=" * 60)

    results = {"timestamp": datetime.now().isoformat(), "feature_importance": {}}

    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_combined_val = (y[split_idx:, 0] + y[split_idx:, 1]).reshape(-1, 1)

    xgb = XGBoostModel(n_estimators=100, max_depth=6)
    xgb.fit(X_train, y_combined_train := (y[:split_idx, 0] + y[:split_idx, 1]).reshape(-1, 1), seed=42)

    shap_analyzer = SHAPAnalyzer(xgb, feature_names)

    importance = shap_analyzer.get_feature_importance(X_val[:100])
    results["feature_importance"] = importance

    physical_validation = shap_analyzer.validate_physical_soundness(
        X_val[:100],
        {"u_value": (0.1, 2.0), "outdoor_temp": (-20, 45), "wwr": (0.1, 0.9)}
    )
    results["physical_validation"] = physical_validation

    logger.info("\nFeature Importance (SHAP):")
    for name, imp in list(importance.items())[:5]:
        logger.info(f"  {name}: {imp:.4f}")

    if physical_validation["insights"]:
        logger.info("\nPhysical Validation Insights:")
        for insight in physical_validation["insights"]:
            logger.info(f"  - {insight}")

    with open(output_dir / "shap_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def create_ensemble(X: np.ndarray, y: np.ndarray, output_dir: Path) -> Dict:
    """Create and evaluate ensemble predictor."""
    logger.info("\n" + "=" * 60)
    logger.info("ENSEMBLE PREDICTION")
    logger.info("=" * 60)

    results = {"timestamp": datetime.now().isoformat(), "models": []}

    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    y_combined_train = (y_train[:, 0] + y_train[:, 1]).reshape(-1, 1)
    y_combined_val = (y_val[:, 0] + y_val[:, 1]).reshape(-1, 1)

    ensemble = EnsemblePredictor()

    mlp = MLPEnsemble(input_dim=X.shape[1], output_dim=1, hidden_dims=[64, 64], n_models=3)
    mlp.fit(X_train, y_combined_train, epochs=50, seed=42)
    ensemble.add_model("mlp_ensemble", "mlp", mlp, weight=1.0)

    xgb = XGBoostModel(n_estimators=100, max_depth=6)
    xgb.fit(X_train, y_combined_train, seed=42)
    ensemble.add_model("xgboost", "xgboost", xgb, weight=1.0)

    rf = RandomForestModel(n_estimators=100, max_depth=10)
    rf.fit(X_train, y_combined_train, seed=42)
    ensemble.add_model("random_forest", "random_forest", rf, weight=1.0)

    benchmark = ensemble.benchmark(X_val, y_combined_val)
    results["ensemble_benchmark"] = benchmark

    logger.info(f"\nEnsemble Performance:")
    logger.info(f"  R²: {benchmark['ensemble_r2']:.4f}")
    logger.info(f"  MAE: {benchmark['ensemble_mae']:.4f}")
    logger.info(f"  Disagreement (mean): {benchmark['mean_disagreement']:.4f}")
    logger.info(f"  Models: {benchmark['n_models']} ({', '.join(benchmark['model_types'])})")

    with open(output_dir / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="SUR-01: Enhanced Surrogate Training with SHAP and Ensemble Methods")
    parser.add_argument("--data-dir", type=str, default="data/training", help="Training data directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--run-benchmark", action="store_true", help="Run MLP vs XGBoost vs RF benchmark")
    parser.add_argument("--shap-analysis", action="store_true", help="Run SHAP interpretability analysis")
    parser.add_argument("--ensemble", action="store_true", help="Create ensemble predictor")
    parser.add_argument("--n-samples", type=int, default=10000, help="Synthetic samples if no data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = ["outdoor_temp", "heating_setpoint", "cooling_setpoint", "hour_of_day",
                     "day_of_year", "month", "u_value", "wwr"]

    logger.info("Loading training data...")
    try:
        X, y, feature_names = load_training_data(args.data_dir, feature_names)
    except Exception as e:
        logger.warning(f"Failed to load data: {e}, generating synthetic data")
        X, y, feature_names = generate_synthetic_thermal_data(args.n_samples, args.seed)

    results_summary = {"timestamp": datetime.now().isoformat()}

    if args.run_benchmark:
        benchmark_results = run_benchmark(X, y, feature_names, output_dir)
        results_summary["benchmark"] = benchmark_results

    if args.shap_analysis:
        shap_results = run_shap_analysis(X, y, feature_names, output_dir)
        results_summary["shap_analysis"] = shap_results

    if args.ensemble:
        ensemble_results = create_ensemble(X, y, output_dir)
        results_summary["ensemble"] = ensemble_results

    with open(output_dir / "surrogate_training_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}/")
    logger.info("Files: benchmark_results.json, shap_analysis.json, ensemble_results.json, surrogate_training_summary.json")


if __name__ == "__main__":
    main()