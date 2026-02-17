#!/usr/bin/env python3
"""
Ensemble Training Framework for Improved Model Robustness

This module provides infrastructure for diverse ensemble training:
- Ensemble training infrastructure
- Model diversity metrics
- Ensemble prediction aggregation
- Tests for ensemble methods

Related to Issue #170: Phase 9: Implement diverse ensemble training
"""

import json
import logging
import random
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleConfig:
    """Configuration for ensemble training."""
    
    def __init__(
        self,
        num_models: int = 5,
        model_factory: Optional[Callable[[], nn.Module]] = None,
        aggregation_method: str = "mean",  # mean, weighted, stacking
        diversity_weight: float = 0.1,
        bootstrap_samples: bool = True,
        feature_dropout: Optional[float] = None,
    ):
        self.num_models = num_models
        self.model_factory = model_factory
        self.aggregation_method = aggregation_method
        self.diversity_weight = diversity_weight
        self.bootstrap_samples = bootstrap_samples
        self.feature_dropout = feature_dropout


class ModelDiversityMetrics:
    """Metrics for measuring model diversity in an ensemble."""
    
    @staticmethod
    def compute_diversity_score(
        predictions: List[np.ndarray],
    ) -> float:
        """
        Compute diversity score based on prediction variance.
        
        Higher variance = more diverse models.
        """
        if len(predictions) < 2:
            return 0.0
        
        # Stack predictions: (num_models, num_samples, ...)
        stacked = np.stack(predictions, axis=0)
        
        # Compute variance across models for each sample
        variance = np.var(stacked, axis=0)
        
        # Return mean variance
        return float(np.mean(variance))
    
    @staticmethod
    def compute_disagreement(
        predictions: List[np.ndarray],
        threshold: float = 0.1,
    ) -> float:
        """
        Compute disagreement rate between models.
        
        Returns fraction of samples where models disagree significantly.
        """
        if len(predictions) < 2:
            return 0.0
        
        stacked = np.stack(predictions, axis=0)
        num_models = stacked.shape[0]
        
        # Compute pairwise differences
        disagreement_count = 0
        total_samples = stacked.shape[1]
        
        for i in range(total_samples):
            sample_preds = stacked[:, i]
            max_diff = np.max(sample_preds) - np.min(sample_preds)
            if max_diff > threshold:
                disagreement_count += 1
        
        return disagreement_count / total_samples
    
    @staticmethod
    def compute_q_statistic(
        predictions_a: np.ndarray,
        predictions_b: np.ndarray,
    ) -> float:
        """
        Compute Q-statistic for pairwise model diversity.
        
        Measures correlation between model errors.
        """
        # Binary error indicators (1 if error, 0 otherwise)
        errors_a = np.abs(predictions_a) > 0.1
        errors_b = np.abs(predictions_b) > 0.1
        
        n11 = np.sum(errors_a & errors_b)  # Both wrong
        n00 = np.sum(~errors_a & ~errors_b)  # Both right
        n10 = np.sum(errors_a & ~errors_b)  # A wrong, B right
        n01 = np.sum(~errors_a & errors_b)  # A right, B wrong
        
        total = n11 + n00 + n10 + n01
        if total == 0:
            return 0.0
        
        q = (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01)
        return float(q)


class EnsembleModel:
    """
    Ensemble of multiple models with aggregation.
    """
    
    def __init__(
        self,
        config: EnsembleConfig,
    ):
        self.config = config
        self.models: List[nn.Module] = []
        self.optimizers: List[optim.Optimizer] = []
        self.model_weights: Optional[np.ndarray] = None
        
    def initialize_models(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 64,
    ) -> None:
        """Initialize ensemble models."""
        if self.config.model_factory is None:
            # Use default model factory
            def create_model():
                return nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                )
            self.config.model_factory = create_model
        
        for i in range(self.config.num_models):
            model = self.config.model_factory()
            self.models.append(model)
            self.optimizers.append(optim.Adam(model.parameters(), lr=0.001))
        
        # Initialize equal weights
        self.model_weights = np.ones(self.config.num_models) / self.config.num_models
        
        logger.info(f"Initialized ensemble with {self.config.num_models} models")
    
    def train_single_model(
        self,
        model_idx: int,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> float:
        """Train a single model in the ensemble."""
        if model_idx >= len(self.models):
            raise ValueError(f"Invalid model index: {model_idx}")
        
        model = self.models[model_idx]
        optimizer = self.optimizers[model_idx]
        criterion = nn.MSELoss()
        
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        total_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                
                # Apply feature dropout if configured
                if self.config.feature_dropout:
                    mask = torch.rand(batch_X.shape) > self.config.feature_dropout
                    batch_X = batch_X * mask.float()
                
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            total_loss = epoch_loss / len(loader)
        
        return total_loss
    
    def train_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        bootstrap: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train all models in the ensemble.
        
        Returns training history.
        """
        history = {"loss": [], "diversity": []}
        
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for model_idx in range(len(self.models)):
                # Create bootstrap sample if enabled
                if bootstrap and self.config.bootstrap_samples:
                    n_samples = len(X)
                    indices = np.random.choice(n_samples, size=n_samples, replace=True)
                    X_boot = X_tensor[indices]
                    y_boot = y_tensor[indices]
                else:
                    X_boot = X_tensor
                    y_boot = y_tensor
                
                loss = self.train_single_model(
                    model_idx, X_boot, y_boot, epochs=1, batch_size=batch_size
                )
                epoch_losses.append(loss)
            
            # Record metrics
            avg_loss = np.mean(epoch_losses)
            history["loss"].append(avg_loss)
            
            # Compute diversity
            preds = self.predict(X)
            diversity = ModelDiversityMetrics.compute_diversity_score(preds)
            history["diversity"].append(diversity)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, diversity={diversity:.4f}")
        
        return history
    
    def predict_single(
        self,
        model_idx: int,
        X: np.ndarray,
    ) -> np.ndarray:
        """Get prediction from a single model."""
        model = self.models[model_idx]
        model.eval()
        
        X_tensor = torch.from_numpy(X).float()
        
        with torch.no_grad():
            pred = model(X_tensor).numpy()
        
        return pred
    
    def predict(
        self,
        X: np.ndarray,
        method: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        Get predictions from all models.
        
        Returns list of predictions, one per model.
        """
        predictions = []
        
        for model_idx in range(len(self.models)):
            pred = self.predict_single(model_idx, X)
            predictions.append(pred)
        
        return predictions
    
    def predict_aggregate(
        self,
        X: np.ndarray,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """
        Get aggregated prediction from ensemble.
        """
        method = method or self.config.aggregation_method
        predictions = self.predict(X)
        
        if method == "mean":
            return np.mean(predictions, axis=0)
        
        elif method == "weighted":
            if self.model_weights is None:
                return np.mean(predictions, axis=0)
            weights = self.model_weights.reshape(-1, 1, 1)
            return np.sum(predictions * weights, axis=0)
        
        elif method == "median":
            return np.median(predictions, axis=0)
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def update_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Update model weights based on validation performance.
        """
        predictions = self.predict(X)
        losses = []
        
        for pred in predictions:
            loss = np.mean((pred - y) ** 2)
            losses.append(loss)
        
        # Convert losses to weights (inverse)
        losses = np.array(losses)
        weights = 1.0 / (losses + 1e-6)
        weights = weights / weights.sum()
        
        self.model_weights = weights
        logger.info(f"Updated weights: {weights}")
    
    def get_diversity_metrics(
        self,
        X: np.ndarray,
    ) -> Dict[str, float]:
        """Get diversity metrics for the ensemble."""
        predictions = self.predict(X)
        
        return {
            "diversity_score": ModelDiversityMetrics.compute_diversity_score(predictions),
            "disagreement": ModelDiversityMetrics.compute_disagreement(predictions),
        }
    
    def save(self, path: Path) -> None:
        """Save ensemble models."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), path / f"model_{i}.pt")
        
        # Save config
        config = {
            "num_models": self.config.num_models,
            "aggregation_method": self.config.aggregation_method,
            "model_weights": self.model_weights.tolist() if self.model_weights is not None else None,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)
        
        logger.info(f"Saved ensemble to {path}")
    
    def load(self, path: Path) -> None:
        """Load ensemble models."""
        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        
        self.config.num_models = config["num_models"]
        self.config.aggregation_method = config["aggregation_method"]
        
        if config.get("model_weights"):
            self.model_weights = np.array(config["model_weights"])
        
        # Load models
        for i in range(self.config.num_models):
            model = self.config.model_factory()
            model.load_state_dict(torch.load(path / f"model_{i}.pt"))
            self.models.append(model)
        
        logger.info(f"Loaded ensemble from {path}")


class DiverseEnsembleTrainer:
    """
    Trainer that emphasizes model diversity.
    """
    
    def __init__(
        self,
        ensemble: EnsembleModel,
        diversity_penalty: float = 0.1,
    ):
        self.ensemble = ensemble
        self.diversity_penalty = diversity_penalty
    
    def train_with_diversity(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """
        Train ensemble with diversity regularization.
        """
        history = {"loss": [], "diversity": [], "total": []}
        
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for model_idx in range(len(self.ensemble.models)):
                model = self.ensemble.models[model_idx]
                optimizer = self.ensemble.optimizers[model_idx]
                criterion = nn.MSELoss()
                
                dataset = TensorDataset(X_tensor, y_tensor)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                model.train()
                
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    pred = model(batch_X)
                    
                    # Main loss
                    main_loss = criterion(pred, batch_y)
                    
                    # Diversity loss (encourage different predictions)
                    other_preds = []
                    for other_idx, other_model in enumerate(self.ensemble.models):
                        if other_idx != model_idx:
                            other_pred = other_model(batch_X)
                            other_preds.append(other_pred)
                    
                    if other_preds:
                        # Penalize similarity (maximize diversity)
                        avg_other = torch.stack(other_preds).mean(dim=0)
                        diversity_loss = -torch.mean((pred - avg_other) ** 2)
                        total_loss = main_loss + self.diversity_penalty * diversity_loss
                    else:
                        total_loss = main_loss
                    
                    total_loss.backward()
                    optimizer.step()
                    epoch_losses.append(main_loss.item())
            
            # Record metrics
            avg_loss = np.mean(epoch_losses)
            history["loss"].append(avg_loss)
            
            # Compute diversity
            preds = self.ensemble.predict(X)
            diversity = ModelDiversityMetrics.compute_diversity_score(preds)
            history["diversity"].append(diversity)
            
            history["total"].append(avg_loss - self.diversity_penalty * diversity)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, diversity={diversity:.4f}")
        
        return history


# Example model for ensemble
class EnsembleMember(nn.Module):
    """Base model for ensemble members."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Ensemble Training")
    parser.add_argument("--num-models", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    print("Testing Ensemble Training")
    print("=" * 50)
    
    # Create config
    config = EnsembleConfig(
        num_models=args.num_models,
        aggregation_method="mean",
        diversity_weight=0.1,
    )
    
    # Create ensemble
    ensemble = EnsembleModel(config)
    ensemble.initialize_models(input_dim=5, hidden_dim=32)
    
    # Generate training data
    np.random.seed(42)
    X_train = np.random.randn(args.num_samples, 5).astype(np.float32)
    y_train = (X_train[:, 0] * 2 + X_train[:, 1] * 0.5 + 
               np.random.randn(args.num_samples) * 0.1).reshape(-1, 1).astype(np.float32)
    
    # Train ensemble
    print(f"\nTraining ensemble with {args.num_models} models...")
    history = ensemble.train_ensemble(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    
    # Get predictions
    X_test = np.random.randn(100, 5).astype(np.float32)
    predictions = ensemble.predict_aggregate(X_test)
    
    print(f"\nPrediction shape: {predictions.shape}")
    
    # Get diversity metrics
    diversity = ensemble.get_diversity_metrics(X_test)
    print(f"\nDiversity Metrics:")
    for key, value in diversity.items():
        print(f"  {key}: {value:.4f}")
    
    # Update weights based on performance
    y_test = (X_test[:, 0] * 2 + X_test[:, 1] * 0.5).reshape(-1, 1).astype(np.float32)
    ensemble.update_weights(X_test, y_test)
    
    print(f"\nFinal weights: {ensemble.model_weights}")
