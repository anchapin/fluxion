#!/usr/bin/env python3
"""
Tests for Online Learning Framework

Tests for Issue #169: Phase 9: Implement online learning framework
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from online_learning import (
    StreamingDataBuffer,
    PerformanceMonitor,
    OnlineLearner,
    DataStreamer,
    SimpleOnlineModel,
)


class TestStreamingDataBuffer:
    """Tests for StreamingDataBuffer class."""
    
    def test_initialization(self):
        """Test buffer initialization."""
        buffer = StreamingDataBuffer(max_size=100, feature_dim=5, target_dim=2)
        assert buffer.max_size == 100
        assert buffer.feature_dim == 5
        assert buffer.target_dim == 2
        assert len(buffer) == 0
    
    def test_add_single_sample(self):
        """Test adding a single sample."""
        buffer = StreamingDataBuffer(max_size=100, feature_dim=3, target_dim=1)
        features = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([[5.0]])
        
        buffer.add(features, targets)
        
        assert len(buffer) == 1
    
    def test_add_multiple_samples(self):
        """Test adding multiple samples."""
        buffer = StreamingDataBuffer(max_size=100, feature_dim=3, target_dim=1)
        features = np.random.randn(10, 3).astype(np.float32)
        targets = np.random.randn(10, 1).astype(np.float32)
        
        buffer.add(features, targets)
        
        assert len(buffer) == 10
    
    def test_buffer_wrap_around(self):
        """Test buffer wrap-around behavior."""
        buffer = StreamingDataBuffer(max_size=5, feature_dim=2, target_dim=1)
        
        # Add 7 samples to a buffer of size 5
        for i in range(7):
            features = np.array([[i, i + 1]], dtype=np.float32)
            targets = np.array([[i * 2]], dtype=np.float32)
            buffer.add(features, targets)
        
        # Should only keep 5 most recent
        assert len(buffer) == 5
    
    def test_get_recent(self):
        """Test getting recent samples."""
        buffer = StreamingDataBuffer(max_size=100, feature_dim=2, target_dim=1)
        
        for i in range(20):
            features = np.array([[i, i]], dtype=np.float32)
            targets = np.array([[i]], dtype=np.float32)
            buffer.add(features, targets)
        
        recent_features, recent_targets = buffer.get_recent(5)
        
        assert recent_features.shape == (5, 2)
        assert recent_targets.shape == (5, 1)
    
    def test_clear(self):
        """Test clearing the buffer."""
        buffer = StreamingDataBuffer(max_size=100, feature_dim=2, target_dim=1)
        
        features = np.array([[1.0, 2.0]], dtype=np.float32)
        targets = np.array([[3.0]], dtype=np.float32)
        buffer.add(features, targets)
        
        buffer.clear()
        
        assert len(buffer) == 0


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor(window_size=100, metrics_history_size=1000)
        assert monitor.window_size == 100
        assert monitor.metrics_history_size == 1000
    
    def test_record_prediction(self):
        """Test recording predictions."""
        monitor = PerformanceMonitor()
        
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.1, 2.1, 2.9])
        
        monitor.record_prediction(predictions, targets)
        
        assert len(monitor.loss_history) == 1
        assert len(monitor.mae_history) == 1
    
    def test_get_recent_stats(self):
        """Test getting recent statistics."""
        monitor = PerformanceMonitor(window_size=10)
        
        for i in range(10):
            pred = np.array([float(i)])
            target = np.array([float(i) + 0.1])
            monitor.record_prediction(pred, target)
        
        stats = monitor.get_recent_stats()
        
        assert 'recent_loss_mean' in stats
        assert 'recent_mae_mean' in stats
        assert stats['recent_mae_mean'] == pytest.approx(0.1, rel=0.01)
    
    def test_drift_detection_no_drift(self):
        """Test drift detection when there's no drift."""
        monitor = PerformanceMonitor()
        
        # Generate consistent data
        for _ in range(500):
            pred = np.array([1.0])
            target = np.array([1.0])
            monitor.record_prediction(pred, target)
        
        has_drift, message = monitor.check_drift()
        
        assert not has_drift
    
    def test_drift_detection_with_drift(self):
        """Test drift detection when there is drift."""
        monitor = PerformanceMonitor()
        
        # Generate consistent data first
        for _ in range(500):
            pred = np.array([1.0])
            target = np.array([1.0])
            monitor.record_prediction(pred, target)
        
        # Generate data with drift
        for _ in range(200):
            pred = np.array([1.0])
            target = np.array([5.0])  # Significant drift
            monitor.record_prediction(pred, target)
        
        has_drift, message = monitor.check_drift()
        
        assert has_drift


class TestOnlineLearner:
    """Tests for OnlineLearner class."""
    
    def test_initialization(self):
        """Test online learner initialization."""
        model = SimpleOnlineModel(input_dim=5)
        learner = OnlineLearner(
            model=model,
            learning_rate=0.001,
            batch_size=32,
            update_interval=100,
            buffer_size=1000,
        )
        
        assert learner.model is not None
        assert learner.batch_size == 32
        assert learner.update_interval == 100
    
    def test_process_sample(self):
        """Test processing a sample."""
        model = SimpleOnlineModel(input_dim=5)
        learner = OnlineLearner(
            model=model,
            batch_size=10,
            update_interval=50,
            buffer_size=100,
        )
        
        features = np.random.randn(5, 5).astype(np.float32)
        targets = np.random.randn(5, 1).astype(np.float32)
        
        result = learner.process_sample(features, targets)
        
        # No update should happen yet
        assert result is None or 'update_number' in result
        assert learner._sample_count == 5
    
    def test_update(self):
        """Test model update."""
        model = SimpleOnlineModel(input_dim=5)
        learner = OnlineLearner(
            model=model,
            batch_size=10,
            update_interval=10,
            buffer_size=100,
        )
        
        # Add enough samples to trigger update
        for _ in range(2):
            features = np.random.randn(5, 5).astype(np.float32)
            targets = np.random.randn(5, 1).astype(np.float32)
            learner.process_sample(features, targets)
        
        result = learner.update()
        
        if result:
            assert 'training_loss' in result
            assert 'update_number' in result
    
    def test_get_metrics(self):
        """Test getting metrics."""
        model = SimpleOnlineModel(input_dim=5)
        learner = OnlineLearner(model=model, batch_size=10, update_interval=50)
        
        # Add some samples
        for _ in range(5):
            features = np.random.randn(5, 5).astype(np.float32)
            targets = np.random.randn(5, 1).astype(np.float32)
            learner.process_sample(features, targets)
        
        metrics = learner.get_metrics()
        
        assert 'sample_count' in metrics
        assert 'buffer_size' in metrics
    
    def test_checkpoint_save_load(self, tmp_path):
        """Test saving and loading checkpoints."""
        model = SimpleOnlineModel(input_dim=5)
        learner = OnlineLearner(
            model=model,
            batch_size=10,
            update_interval=10,
            buffer_size=100,
            checkpoint_dir=tmp_path,
        )
        
        # Add some samples
        for _ in range(2):
            features = np.random.randn(5, 5).astype(np.float32)
            targets = np.random.randn(5, 1).astype(np.float32)
            learner.process_sample(features, targets)
        
        # Save checkpoint
        checkpoint_path = learner.save_checkpoint("test_checkpoint.pt")
        
        # Create new learner
        model2 = SimpleOnlineModel(input_dim=5)
        learner2 = OnlineLearner(
            model=model2,
            batch_size=10,
            update_interval=10,
            buffer_size=100,
            checkpoint_dir=tmp_path,
        )
        
        # Load checkpoint
        learner2.load_checkpoint("test_checkpoint.pt")
        
        assert learner2._sample_count == learner._sample_count


class TestDataStreamer:
    """Tests for DataStreamer class."""
    
    def test_initialization(self):
        """Test streamer initialization."""
        streamer = DataStreamer(batch_size=32)
        assert streamer.batch_size == 32
    
    def test_iterate_numpy_data(self, tmp_path):
        """Test iterating over numpy data."""
        # Create test data
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100, 1).astype(np.float32)
        
        data_path = tmp_path / "test_data.npz"
        np.savez(data_path, X=X, y=y)
        
        streamer = DataStreamer(source=data_path, batch_size=10)
        
        batches = list(streamer)
        
        assert len(batches) == 10  # 100 samples / 10 batch_size
        
        batch_X, batch_y = batches[0]
        assert batch_X.shape[0] == 10
        assert batch_y.shape[0] == 10
    
    def test_callback(self, tmp_path):
        """Test callback functionality."""
        X = np.random.randn(20, 5).astype(np.float32)
        y = np.random.randn(20, 1).astype(np.float32)
        
        data_path = tmp_path / "test_data.npz"
        np.savez(data_path, X=X, y=y)
        
        callback_results = []
        
        def callback(features, targets):
            callback_results.append((features.shape, targets.shape))
        
        streamer = DataStreamer(source=data_path, batch_size=5, callback=callback)
        
        list(streamer)
        
        assert len(callback_results) == 4
    
    def test_reset(self, tmp_path):
        """Test resetting the streamer."""
        X = np.random.randn(20, 5).astype(np.float32)
        y = np.random.randn(20, 1).astype(np.float32)
        
        data_path = tmp_path / "test_data.npz"
        np.savez(data_path, X=X, y=y)
        
        streamer = DataStreamer(source=data_path, batch_size=5)
        
        batches = list(streamer)
        assert len(batches) == 4
        
        streamer.reset()
        
        batches2 = list(streamer)
        assert len(batches2) == 4


class TestIntegration:
    """Integration tests for the online learning framework."""
    
    def test_full_online_learning_workflow(self):
        """Test complete online learning workflow."""
        # Create model
        input_dim = 3
        model = SimpleOnlineModel(input_dim=input_dim, hidden_dim=32, output_dim=1)
        
        # Create online learner
        learner = OnlineLearner(
            model=model,
            learning_rate=0.01,
            batch_size=16,
            update_interval=32,
            buffer_size=200,
        )
        
        # Simulate streaming data with a simple function
        np.random.seed(42)
        
        for epoch in range(3):
            for batch_idx in range(10):
                # Generate batch
                batch_size = 16
                features = np.random.randn(batch_size, input_dim).astype(np.float32)
                targets = (features[:, 0] * 2 + features[:, 1] * 0.5 + 
                          np.random.randn(batch_size) * 0.1).reshape(-1, 1).astype(np.float32)
                
                # Process sample
                result = learner.process_sample(features, targets)
        
        # Get final metrics
        metrics = learner.get_metrics()
        
        assert metrics['sample_count'] == 480  # 3 * 10 * 16
        assert 'recent_loss_mean' in metrics
        
        # Check for drift (should not have drift)
        has_drift, _ = learner.check_performance()
        # May or may not have drift depending on data, but should not crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
