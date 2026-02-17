#!/usr/bin/env python3
"""
Diverse Ensemble Training Module

This module provides infrastructure for training diverse ensembles:
- Ensemble training infrastructure
- Model diversity metrics
- Prediction aggregation
- Ensemble tests

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
)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDiversityMetrics:
    """
    Compute diversity metrics for ensemble models.
    
    Metrics include:
    - Disagreement rate
    - Q-statistic
    - Correlation coefficient
    - Entropy
    """
    
    @staticmethod
    def disagreement_rate(predictions: np.ndarray) -> float:
        """
        Calculate disagreement rate between models.
        
        Args:
            predictions: Array of shape (num_models, num_samples, num_outputs)
        
        Returns:
            Fraction of samples where models disagree
        """
        num_models = predictions.shape[0]
        num_samples = predictions.shape[1]
        
        if num_models < 2:
            return 0.0
        
        disagreement_count = 0
        
        for s in range(num_samples):
            # Get predictions from all models for this sample
            sample_preds = predictions[:, s, :]
            
            # Check if any model disagrees with the majority
            mean_pred = np.mean(sample_preds, axis=0)
            
            # Count models that differ from mean
            for m in range(num_models):
                if not np.allclose(sample_preds[m], mean_pred, atol=1e-4):
                    disagreement_count += 1
        
        total_predictions = num_models * num_samples
        return disagreement_count / total_predictions if total_predictions > 0 else 0.0
    
    @staticmethod
    def correlation_matrix(predictions: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise correlation between model predictions.
        
        Args:
            predictions: Array of shape (num_models, num_samples, num_outputs)
        
        Returns:
            Correlation matrix of shape (num_models, num_models)
        """
        num_models = predictions.shape[0]
        
        # Flatten predictions: (num_models, num_samples * num_outputs)
        flat_preds = predictions.reshape(num_models, -1)
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(flat_preds)
        
        # Handle NaN values (can occur if predictions are constant)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        return corr_matrix
    
    @staticmethod
    def average_correlation(predictions: np.ndarray) -> float:
        """Calculate average pairwise correlation."""
        corr_matrix = ModelDiversityMetrics.correlation_matrix(predictions)
        num_models = corr_matrix.shape[0]
        
        if num_models < 2:
            return 0.0
        
        # Get upper triangle (excluding diagonal)
        upper_tri = corr_matrix[np.triu_indices(num_models, k=1)]
        
        return np.mean(upper_tri) if len(upper_tri) > 0 else 0.0
    
    @staticmethod
    def q_statistic(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Calculate Q-statistic for each model pair.
        
        The Q-statistic measures how often two models make correct
        predictions on the same samples.
        
        Args:
            predictions: Array of shape (num_models, num_samples)
            target: Ground truth of shape (num_samples,)
        
        Returns:
            Q-statistic matrix
        """
        num_models = predictions.shape[0]
        num_samples = predictions.shape[1]
        
        # Convert predictions to binary correctness
        correct = np.zeros_like(predictions, dtype=bool)
        for m in range(num_models):
            correct[m] = np.abs(predictions[m] - target) < 0.5  # Threshold
        
        q_matrix = np.zeros((num_models, num_models))
        
        for i in range(num_models):
            for j in range(num_models):
                # Both correct
                both_correct = np.sum(correct[i] & correct[j])
                # Both incorrect
                both_incorrect = np.sum(~correct[i] & ~correct[j])
                
                q_matrix[i, j] = (both_correct + both_incorrect) / num_samples
        
        return q_matrix
    
    @staticmethod
    def entropy(predictions: np.ndarray) -> float:
        """
        Calculate average entropy of predictions.
        
        Args:
            predictions: Array of shape (num_models, num_samples, num_outputs)
        
        Returns:
            Average entropy across samples and outputs
        """
        num_models = predictions.shape[0]
        
        if num_models < 2:
            return 0.0
        
        # Convert to probabilities via softmax-like normalization
        mean_pred = np.mean(predictions, axis=0, keepdims=True)
        std_pred = np.std(predictions, axis=0, keepdims=True)
        
        # Avoid division by zero
        std_pred = np.where(std_pred < 1e-6, 1e-6, std_pred)
        
        # Normalize: how different is each model from the mean
        normalized = (predictions - mean_pred) / std_pred
        
        # Simple entropy approximation using variance
        entropy = np.mean(np.abs(normalized))
        
        return float(entropy)


class DiversityAwareLoss(nn.Module):
    """
    Loss function that encourages model diversity.
    
    Adds a diversity term to the standard MSE loss that penalizes
    models making similar predictions.
    """
    
    def __init__(self, lambda_diversity: float = 0.01):
        super().__init__()
        self.lambda_diversity = lambda_diversity
        self.mse = nn.MSELoss()
        
    def forward(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor,
        other_predictions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute diversity-aware loss.
        
        Args:
            predictions: This model's predictions
            target: Ground truth
            other_predictions: Predictions from other models in ensemble
        
        Returns:
            Tuple of (loss, loss_components_dict)
        """
        # Data fitting loss
        data_loss = self.mse(predictions, target)
        
        total_loss = data_loss
        loss_components = {
            'data_loss': data_loss.item(),
            'diversity_loss': 0.0,
        }
        
        # Add diversity loss if other predictions are provided
        if other_predictions is not None and self.lambda_diversity > 0:
            # Calculate variance across models (higher = more diverse)
            all_preds = torch.stack([predictions] + list(other_predictions), dim=0)
            diversity_loss = -torch.var(all_preds, dim=0).mean()  # Negative because we want to maximize
            
            total_loss = total_loss + self.lambda_diversity * diversity_loss
            loss_components['diversity_loss'] = diversity_loss.item()
        
        loss_components['total_loss'] = total_loss.item()
        
        return total_loss, loss_components


class EnsembleTrainer:
    """
    Train diverse ensemble models with diversity-aware training.
    """
    
    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        num_models: int = 5,
        hidden_dims_list: Optional[List[List[int]]] = None,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        diversity_weight: float = 0.01,
        seed: int = 42,
    ):
        self.model_fn = model_fn
        self.num_models = num_models
        self.hidden_dims_list = hidden_dims_list or [[64, 64]] * num_models
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.diversity_weight = diversity_weight
        self.seed = seed
        
        self.models: List[nn.Module] = []
        self.optimizers: List[optim.Optimizer] = []
        
    def _create_model(self, hidden_dims: List[int], seed: int) -> nn.Module:
        """Create a model with given hidden dimensions."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Use default model_fn but could be customized based on hidden_dims
        model = self.model_fn()
        
        # Modify architecture based on hidden_dims if needed
        # For now, just return the model
        return model
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Train diverse ensemble models.
        
        Args:
            X: Training features
            y: Training targets
            validation_split: Fraction for validation
        
        Returns:
            Training history and metrics
        """
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train_t = torch.from_numpy(X_train).float()
        y_train_t = torch.from_numpy(y_train).float()
        X_val_t = torch.from_numpy(X_val).float()
        y_val_t = torch.from_numpy(y_val).float()
        
        # Create dataloader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize models with different seeds for diversity
        self.models = []
        self.optimizers = []
        
        for i in range(self.num_models):
            # Different seed for each model
            model_seed = self.seed + i * 100
            torch.manual_seed(model_seed)
            np.random.seed(model_seed)
            random.seed(model_seed)
            
            model = self._create_model(self.hidden_dims_list[i], model_seed)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            self.models.append(model)
            self.optimizers.append(optimizer)
        
        # Train each model
        history = {
            'models': [],
            'val_losses': [],
            'diversity_scores': [],
        }
        
        for epoch in tqdm(range(self.epochs), desc="Training Ensemble"):
            # Train each model
            for model_idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                model.train()
                epoch_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    pred = model(batch_X)
                    
                    # Get predictions from other models for diversity loss
                    other_preds = None
                    if self.diversity_weight > 0 and model_idx > 0:
                        with torch.no_grad():
                            other_models = self.models[:model_idx]
                            other_preds = [m(batch_X) for m in other_models]
                    
                    # Use diversity-aware loss
                    criterion = DiversityAwareLoss(lambda_diversity=self.diversity_weight)
                    loss, _ = criterion(pred, batch_y, other_preds)
                    
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(train_loader)
                
            # Evaluate each model
            val_losses = []
            all_val_preds = []
            
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_t).numpy()
                    val_loss = np.mean((val_pred - y_val) ** 2)
                    val_losses.append(val_loss)
                    all_val_preds.append(val_pred)
            
            # Calculate diversity
            all_val_preds = np.array(all_val_preds)
            avg_correlation = ModelDiversityMetrics.average_correlation(
                all_val_preds.reshape(len(self.models), -1, 1)
            )
            
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Val Loss={np.mean(val_losses):.4f}, "
                    f"Diversity={1 - avg_correlation:.4f}"
                )
            
            history['val_losses'].append(val_losses)
            history['diversity_scores'].append(1 - avg_correlation)
        
        # Final evaluation
        final_preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_val_t).numpy()
                final_preds.append(pred)
        
        final_preds = np.array(final_preds)
        
        # Calculate final metrics
        ensemble_pred = np.mean(final_preds, axis=0)
        ensemble_loss = np.mean((ensemble_pred - y_val) ** 2)
        
        diversity = ModelDiversityMetrics.average_correlation(
            final_preds.reshape(len(self.models), -1, 1)
        )
        
        history['models'] = self.models
        history['final_ensemble_loss'] = float(ensemble_loss)
        history['final_diversity'] = float(1 - diversity)
        history['individual_losses'] = [float(l) for l in val_losses]
        
        return history


class EnsembleAggregator:
    """
    Aggregate predictions from ensemble models.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        method: str = "mean",
        weights: Optional[List[float]] = None,
    ):
        self.models = models
        self.method = method
        self.weights = weights
        
        if weights is not None and len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
        
        Returns:
            Tuple of (predictions, uncertainties)
        """
        X_t = torch.from_numpy(X).float()
        
        all_predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_t).numpy()
                all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        # Aggregate based on method
        if self.method == "mean":
            predictions = np.mean(all_predictions, axis=0)
            uncertainties = np.std(all_predictions, axis=0)
        elif self.method == "weighted" and self.weights is not None:
            weights = np.array(self.weights) / sum(self.weights)
            predictions = np.average(all_predictions, axis=0, weights=weights)
            uncertainties = np.std(all_predictions, axis=0)
        elif self.method == "median":
            predictions = np.median(all_predictions, axis=0)
            uncertainties = np.std(all_predictions, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")
        
        return predictions, uncertainties
    
    def predict_with_diversity(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict with diversity metrics.
        
        Returns:
            Dictionary with predictions, uncertainties, and diversity info
        """
        X_t = torch.from_numpy(X).float()
        
        all_predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_t).numpy()
                all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        # Calculate diversity metrics
        avg_correlation = ModelDiversityMetrics.average_correlation(all_predictions)
        
        # Aggregate
        predictions = np.mean(all_predictions, axis=0)
        uncertainties = np.std(all_predictions, axis=0)
        
        return {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'diversity_score': 1 - avg_correlation,
            'model_predictions': all_predictions,
        }


# Simple model factory for testing
def create_model(input_dim: int, output_dim: int, hidden_dims: List[int] = None):
    """Create a simple MLP model."""
    if hidden_dims is None:
        hidden_dims = [64, 64]
    
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    
    return nn.Sequential(*layers)


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Diverse Ensemble Training")
    parser.add_argument("--num-models", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    print("Testing Diverse Ensemble Training")
    print("=" * 50)
    
    # Generate synthetic data
    torch.manual_seed(42)
    np.random.seed(42)
    
    num_samples = args.num_samples
    input_dim = 5
    output_dim = 1
    
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(num_samples) * 0.1).reshape(-1, 1).astype(np.float32)
    
    # Model factory
    def model_factory():
        return create_model(input_dim, output_dim, [64, 64])
    
    # Train ensemble
    trainer = EnsembleTrainer(
        model_fn=model_factory,
        num_models=args.num_models,
        learning_rate=0.01,
        batch_size=32,
        epochs=args.epochs,
        diversity_weight=0.01,
    )
    
    history = trainer.train(X, y)
    
    print("\nTraining Complete!")
    print(f"Final Ensemble Loss: {history['final_ensemble_loss']:.4f}")
    print(f"Final Diversity Score: {history['final_diversity']:.4f}")
    print(f"Individual Model Losses: {history['individual_losses']}")
    
    # Test aggregation
    if trainer.models:
        aggregator = EnsembleAggregator(trainer.models, method="mean")
        test_X = np.random.randn(10, input_dim).astype(np.float32)
        predictions, uncertainties = aggregator.predict(test_X)
        
        print(f"\nPrediction shape: {predictions.shape}")
        print(f"Uncertainty shape: {uncertainties.shape}")
