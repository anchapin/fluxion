#!/usr/bin/env python3
"""
Tests for Ensemble Training

Tests for Issue #170: Phase 9: Implement diverse ensemble training
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble_training import (
    EnsembleConfig,
    ModelDiversityMetrics,
    EnsembleModel,
    DiverseEnsembleTrainer,
    EnsembleMember,
)


class TestModelDiversityMetrics:
    """Tests for ModelDiversityMetrics class."""
    
    def test_compute_diversity_score_single_model(self):
        """Test diversity score with single model."""
        predictions = [np.array([1.0, 2.0, 3.0])]
        
        score = ModelDiversityMetrics.compute_diversity_score(predictions)
        
        assert score == 0.0
    
    def test_compute_diversity_score_multiple_models(self):
        """Test diversity score with multiple models."""
        predictions = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.5, 2.5, 3.5]),
            np.array([0.5, 1.5, 2.5]),
        ]
        
        score = ModelDiversityMetrics.compute_diversity_score(predictions)
        
        assert score > 0.0
    
    def test_compute_diversity_score_identical(self):
        """Test diversity score with identical predictions."""
        predictions = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
        ]
        
        score = ModelDiversityMetrics.compute_diversity_score(predictions)
        
        assert score == 0.0
    
    def test_compute_disagreement(self):
        """Test disagreement computation."""
        predictions = [
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]),  # Very different
        ]
        
        disagreement = ModelDiversityMetrics.compute_disagreement(predictions, threshold=1.0)
        
        assert disagreement > 0.0
    
    def test_compute_disagreement_no_disagreement(self):
        """Test disagreement with similar predictions."""
        predictions = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 3.1]),
        ]
        
        disagreement = ModelDiversityMetrics.compute_disagreement(predictions, threshold=1.0)
        
        assert disagreement == 0.0
    
    def test_compute_q_statistic(self):
        """Test Q-statistic computation."""
        predictions_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions_b = np.array([1.2, 2.1, 3.3, 4.1, 5.2])
        
        q = ModelDiversityMetrics.compute_q_statistic(predictions_a, predictions_b)
        
        # Q should be between -1 and 1
        assert -1 <= q <= 1


class TestEnsembleConfig:
    """Tests for EnsembleConfig class."""
    
    def test_initialization(self):
        """Test config initialization."""
        config = EnsembleConfig(
            num_models=10,
            aggregation_method="weighted",
            diversity_weight=0.2,
        )
        
        assert config.num_models == 10
        assert config.aggregation_method == "weighted"
        assert config.diversity_weight == 0.2
    
    def test_default_values(self):
        """Test default values."""
        config = EnsembleConfig()
        
        assert config.num_models == 5
        assert config.aggregation_method == "mean"
        assert config.diversity_weight == 0.1


class TestEnsembleModel:
    """Tests for EnsembleModel class."""
    
    def test_initialization(self):
        """Test ensemble initialization."""
        config = EnsembleConfig(num_models=5)
        ensemble = EnsembleModel(config)
        
        assert ensemble.config.num_models == 5
        assert len(ensemble.models) == 0
    
    def test_initialize_models(self):
        """Test model initialization."""
        config = EnsembleConfig(num_models=3)
        ensemble = EnsembleModel(config)
        ensemble.initialize_models(input_dim=5, hidden_dim=32)
        
        assert len(ensemble.models) == 3
        assert len(ensemble.optimizers) == 3
        assert ensemble.model_weights is not None
        assert len(ensemble.model_weights) == 3
    
    def test_train_single_model(self):
        """Test training a single model."""
        config = EnsembleConfig(num_models=1)
        ensemble = EnsembleModel(config)
        ensemble.initialize_models(input_dim=3, hidden_dim=16)
        
        X = torch.randn(50, 3).float()
        y = torch.randn(50, 1).float()
        
        loss = ensemble.train_single_model(0, X, y, epochs=2, batch_size=10)
        
        assert isinstance(loss, float)
    
    def test_train_ensemble(self):
        """Test training the full ensemble."""
        config = EnsembleConfig(num_models=3)
        ensemble = EnsembleModel(config)
        ensemble.initialize_models(input_dim=3, hidden_dim=16)
        
        X = np.random.randn(100, 3).astype(np.float32)
        y = (X[:, 0] * 2 + np.random.randn(100) * 0.1).reshape(-1, 1).astype(np.float32)
        
        history = ensemble.train_ensemble(X, y, epochs=2, batch_size=16)
        
        assert "loss" in history
        assert "diversity" in history
        assert len(history["loss"]) == 2
    
    def test_predict_single(self):
        """Test single model prediction."""
        config = EnsembleConfig(num_models=1)
        ensemble = EnsembleModel(config)
        ensemble.initialize_models(input_dim=3, hidden_dim=16)
        
        X = np.random.randn(10, 3).astype(np.float32)
        pred = ensemble.predict_single(0, X)
        
        assert pred.shape == (10, 1)
    
    def test_predict(self):
        """Test ensemble prediction."""
        config = EnsembleConfig(num_models=3)
        ensemble = EnsembleModel(config)
        ensemble.initialize_models(input_dim=3, hidden_dim=16)
        
        X = np.random.randn(10, 3).astype(np.float32)
        preds = ensemble.predict(X)
        
        assert len(preds) == 3
        for pred in preds:
            assert pred.shape == (10, 1)
    
    def test_predict_aggregate_mean(self):
        """Test mean aggregation."""
        config = EnsembleConfig(num_models=3, aggregation_method="mean")
        ensemble = EnsembleModel(config)
        ensemble.initialize_models(input_dim=3, hidden_dim=16)
        
        X = np.random.randn(10, 3).astype(np.float32)
        pred = ensemble.predict_aggregate(X)
        
        assert pred.shape == (10, 1)
    
    def test_predict_aggregate_weighted(self):
        """Test weighted aggregation."""
        config = EnsembleConfig(num_models=3, aggregation_method="weighted")
        ensemble = EnsembleModel(config)
        ensemble.initialize_models(input_dim=3, hidden_dim=16)
        
        X = np.random.randn(10, 3).astype(np.float32)
        pred = ensemble.predict_aggregate(X)
        
        assert pred.shape == (10, 1)
    
    def test_update_weights(self):
        """Test weight update."""
        config = EnsembleConfig(num_models=3)
        ensemble = EnsembleModel(config)
        ensemble.initialize_models(input_dim=3, hidden_dim=16)
        
        X = np.random.randn(50, 3).astype(np.float32)
        y = (X[:, 0] * 2 + np.random.randn(50) * 0.1).reshape(-1, 1).astype(np.float32)
        
        old_weights = ensemble.model_weights.copy()
        ensemble.update_weights(X, y)
        
        assert not np.array_equal(old_weights, ensemble.model_weights)
        assert np.isclose(ensemble.model_weights.sum(), 1.0)
    
    def test_get_diversity_metrics(self):
        """Test diversity metrics computation."""
        config = EnsembleConfig(num_models=3)
        ensemble = EnsembleModel(config)
        ensemble.initialize_models(input_dim=3, hidden_dim=16)
        
        # Train a bit
        X = np.random.randn(50, 3).astype(np.float32)
        y = (X[:, 0] * 2).reshape(-1, 1).astype(np.float32)
        ensemble.train_ensemble(X, y, epochs=1)
        
        metrics = ensemble.get_diversity_metrics(X)
        
        assert "diversity_score" in metrics
        assert "disagreement" in metrics
    
    def test_save_load(self, tmp_path):
        """Test saving and loading ensemble."""
        config = EnsembleConfig(num_models=2)
        
        def model_factory():
            return nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
        
        config.model_factory = model_factory
        ensemble = EnsembleModel(config)
        ensemble.initialize_models(input_dim=3, hidden_dim=16)
        
        # Train a bit
        X = np.random.randn(50, 3).astype(np.float32)
        y = (X[:, 0] * 2).reshape(-1, 1).astype(np.float32)
        ensemble.train_ensemble(X, y, epochs=1)
        
        # Save
        ensemble.save(tmp_path / "ensemble")
        
        # Create new ensemble and load
        config2 = EnsembleConfig(num_models=2)
        config2.model_factory = model_factory
        ensemble2 = EnsembleModel(config2)
        ensemble2.load(tmp_path / "ensemble")
        
        assert len(ensemble2.models) == 2


class TestDiverseEnsembleTrainer:
    """Tests for DiverseEnsembleTrainer class."""
    
    def test_initialization(self):
        """Test trainer initialization."""
        config = EnsembleConfig(num_models=3)
        ensemble = EnsembleModel(config)
        ensemble.initialize_models(input_dim=3, hidden_dim=16)
        
        trainer = DiverseEnsembleTrainer(ensemble, diversity_penalty=0.1)
        
        assert trainer.ensemble is ensemble
        assert trainer.diversity_penalty == 0.1
    
    def test_train_with_diversity(self):
        """Test diversity-aware training."""
        config = EnsembleConfig(num_models=3)
        ensemble = EnsembleModel(config)
        ensemble.initialize_models(input_dim=3, hidden_dim=16)
        
        trainer = DiverseEnsembleTrainer(ensemble, diversity_penalty=0.1)
        
        X = np.random.randn(100, 3).astype(np.float32)
        y = (X[:, 0] * 2 + np.random.randn(100) * 0.1).reshape(-1, 1).astype(np.float32)
        
        history = trainer.train_with_diversity(X, y, epochs=2, batch_size=16)
        
        assert "loss" in history
        assert "diversity" in history
        assert "total" in history


class TestEnsembleMember:
    """Tests for EnsembleMember model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = EnsembleMember(input_dim=5, hidden_dim=32, output_dim=1)
        
        assert model is not None
    
    def test_forward(self):
        """Test forward pass."""
        model = EnsembleMember(input_dim=5, hidden_dim=32, output_dim=1)
        
        x = torch.randn(10, 5)
        y = model(x)
        
        assert y.shape == (10, 1)


class TestIntegration:
    """Integration tests for ensemble training."""
    
    def test_full_ensemble_workflow(self):
        """Test complete ensemble workflow."""
        # Create config
        config = EnsembleConfig(
            num_models=5,
            aggregation_method="mean",
            diversity_weight=0.1,
        )
        
        # Create ensemble
        ensemble = EnsembleModel(config)
        ensemble.initialize_models(input_dim=5, hidden_dim=32)
        
        # Generate data
        np.random.seed(42)
        X_train = np.random.randn(500, 5).astype(np.float32)
        y_train = (X_train[:, 0] * 2 + X_train[:, 1] * 0.5 + 
                   np.random.randn(500) * 0.1).reshape(-1, 1).astype(np.float32)
        
        X_test = np.random.randn(100, 5).astype(np.float32)
        y_test = (X_test[:, 0] * 2 + X_test[:, 1] * 0.5).reshape(-1, 1).astype(np.float32)
        
        # Train
        history = ensemble.train_ensemble(X_train, y_train, epochs=3, batch_size=32)
        
        # Predict
        predictions = ensemble.predict_aggregate(X_test)
        
        # Update weights
        ensemble.update_weights(X_test, y_test)
        
        # Check metrics
        metrics = ensemble.get_diversity_metrics(X_test)
        
        assert predictions.shape == y_test.shape
        assert "diversity_score" in metrics
        assert "disagreement" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
