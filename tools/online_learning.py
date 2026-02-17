#!/usr/bin/env python3
"""
Online Learning Framework for Continuous Model Improvement

This module provides infrastructure for online/incremental learning:
- Online learning architecture
- Incremental training capability
- Data streaming support
- Performance monitoring

Related to Issue #169: Phase 9: Implement online learning framework
"""

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingDataBuffer:
    """
    Ring buffer for streaming data in online learning.
    
    Maintains a fixed-size buffer of recent samples for incremental training.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        feature_dim: int = 1,
        target_dim: int = 1,
    ):
        self.max_size = max_size
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        
        # Pre-allocate arrays for efficiency
        self.features: np.ndarray = np.zeros((max_size, feature_dim), dtype=np.float32)
        self.targets: np.ndarray = np.zeros((max_size, target_dim), dtype=np.float32)
        
        self._head: int = 0
        self._size: int = 0
        self._lock = threading.Lock()
        
    def add(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Add a batch of samples to the buffer."""
        with self._lock:
            batch_size = len(features)
            
            if batch_size > self.max_size:
                # Handle case where batch is larger than buffer
                features = features[-self.max_size:]
                targets = targets[-self.max_size:]
                batch_size = self.max_size
                
            # Calculate wrap-around positions
            end_pos = (self._head + batch_size) % self.max_size
            
            if end_pos > self._head or batch_size == 0:
                # No wrap-around
                self.features[self._head:end_pos] = features
                self.targets[self._head:end_pos] = targets
            else:
                # Wrap-around case
                first_part = self.max_size - self._head
                self.features[self._head:] = features[:first_part]
                self.targets[self._head:] = targets[:first_part]
                self.features[:end_pos] = features[first_part:]
                self.targets[:end_pos] = targets[first_part:]
            
            self._head = end_pos
            self._size = min(self._size + batch_size, self.max_size)
            
    def get_recent(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get the n most recent samples."""
        with self._lock:
            if n > self._size:
                n = self._size
                
            if n == 0:
                return (
                    np.zeros((0, self.feature_dim), dtype=np.float32),
                    np.zeros((0, self.target_dim), dtype=np.float32),
                )
            
            # Calculate start position
            start_pos = (self._head - n) % self.max_size
            
            if start_pos <= self._head or n == 0:
                # No wrap-around
                return (
                    self.features[start_pos:self._head].copy(),
                    self.targets[start_pos:self._head].copy(),
                )
            else:
                # Wrap-around
                result_features = np.zeros((n, self.feature_dim), dtype=np.float32)
                result_targets = np.zeros((n, self.target_dim), dtype=np.float32)
                
                first_part = self.max_size - start_pos
                result_features[:first_part] = self.features[start_pos:]
                result_targets[:first_part] = self.targets[start_pos:]
                result_features[first_part:] = self.features[:self._head]
                result_targets[first_part:] = self.targets[:self._head]
                
                return result_features, result_targets
                
    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all samples in the buffer."""
        return self.get_recent(self._size)
    
    def __len__(self) -> int:
        return self._size
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._head = 0
            self._size = 0


class PerformanceMonitor:
    """
    Monitor performance metrics for online learning.
    
    Tracks prediction accuracy, loss, and other metrics over time.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        metrics_history_size: int = 10000,
    ):
        self.window_size = window_size
        self.metrics_history_size = metrics_history_size
        
        # Sliding windows for recent metrics
        self.loss_history: Deque[float] = deque(maxlen=window_size)
        self.mae_history: Deque[float] = deque(maxlen=window_size)
        
        # Full history for analysis
        self.full_loss_history: Deque[float] = deque(maxlen=metrics_history_size)
        self.full_mae_history: Deque[float] = deque(maxlen=metrics_history_size)
        
        # Timing metrics
        self.update_times: Deque[float] = deque(maxlen=100)
        self._last_update_time: float = 0
        
    def record_prediction(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """Record a prediction result for monitoring."""
        # Calculate metrics
        loss = np.mean((prediction - target) ** 2)
        mae = np.mean(np.abs(prediction - target))
        
        self.loss_history.append(loss)
        self.mae_history.append(mae)
        self.full_loss_history.append(loss)
        self.full_mae_history.append(mae)
        
    def record_update_time(self, duration: float) -> None:
        """Record the time taken for a model update."""
        self.update_times.append(duration)
        
    def get_recent_stats(self) -> Dict[str, float]:
        """Get statistics for recent predictions."""
        if not self.loss_history:
            return {}
        
        return {
            'recent_loss_mean': np.mean(self.loss_history),
            'recent_loss_std': np.std(self.loss_history),
            'recent_mae_mean': np.mean(self.mae_history),
            'recent_mae_std': np.std(self.mae_history),
        }
    
    def get_full_stats(self) -> Dict[str, float]:
        """Get statistics for all historical predictions."""
        if not self.full_loss_history:
            return {}
        
        return {
            'total_loss_mean': np.mean(self.full_loss_history),
            'total_loss_std': np.std(self.full_loss_history),
            'total_mae_mean': np.mean(self.full_mae_history),
            'total_mae_std': np.std(self.full_mae_history),
            'num_predictions': len(self.full_loss_history),
        }
    
    def get_update_stats(self) -> Dict[str, float]:
        """Get timing statistics for model updates."""
        if not self.update_times:
            return {}
        
        return {
            'avg_update_time_ms': np.mean(self.update_times) * 1000,
            'min_update_time_ms': np.min(self.update_times) * 1000,
            'max_update_time_ms': np.max(self.update_times) * 1000,
        }
    
    def check_drift(self, threshold: float = 1.5) -> Tuple[bool, str]:
        """
        Check if there's significant distribution drift.
        
        Compares recent performance to historical baseline.
        
        Returns:
            Tuple of (has_drift, message)
        """
        if len(self.loss_history) < 100 or len(self.full_loss_history) < 500:
            return False, "Insufficient data for drift detection"
        
        recent_mean = np.mean(self.loss_history)
        historical_mean = np.mean(list(self.full_loss_history)[:-100])
        
        if historical_mean > 0:
            ratio = recent_mean / historical_mean
            if ratio > threshold:
                return True, f"Performance degradation detected: {ratio:.2f}x baseline"
        
        return False, "No significant drift detected"


class OnlineLearner:
    """
    Online learning wrapper for PyTorch models.
    
    Supports:
    - Incremental training on streaming data
    - Adaptive learning rate
    - Performance monitoring
    - Checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        update_interval: int = 100,  # Update after this many samples
        buffer_size: int = 10000,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Data buffer
        self.buffer = StreamingDataBuffer(
            max_size=buffer_size,
            feature_dim=0,  # Will be set on first update
            target_dim=0,
        )
        
        # Performance monitoring
        self.monitor = PerformanceMonitor()
        
        # State tracking
        self._sample_count: int = 0
        self._update_count: int = 0
        self._is_updating: bool = False
        self._lock = threading.Lock()
        
        # Checkpointing
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
    def process_sample(
        self,
        features: np.ndarray,
        target: np.ndarray,
    ) -> Optional[Dict[str, float]]:
        """
        Process a single sample (or batch) for online learning.
        
        Args:
            features: Input features (batch_size, feature_dim)
            target: Target values (batch_size, target_dim)
        
        Returns:
            Metrics dict if update was performed, None otherwise
        """
        # Ensure 2D arrays
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if target.ndim == 1:
            target = target.reshape(1, -1)
        
        # Initialize buffer dimensions on first call
        if len(self.buffer) == 0:
            self.buffer = StreamingDataBuffer(
                max_size=10000,
                feature_dim=features.shape[1],
                target_dim=target.shape[1],
            )
        
        # Add to buffer
        self.buffer.add(features, target)
        
        # Make prediction for monitoring
        with torch.no_grad():
            features_tensor = torch.from_numpy(features).float()
            pred = self.model(features_tensor).numpy()
            self.monitor.record_prediction(pred, target)
        
        # Check if we should update
        self._sample_count += len(features)
        
        if self._sample_count >= self.update_interval * (self._update_count + 1):
            return self.update()
        
        return None
    
    def update(self) -> Optional[Dict[str, float]]:
        """Perform an incremental update on the model."""
        with self._lock:
            if self._is_updating:
                return None  # Prevent concurrent updates
            
            self._is_updating = True
            
        try:
            start_time = time.time()
            
            # Get training data from buffer
            train_features, train_targets = self.buffer.get_recent(
                min(self._sample_count, self.buffer.max_size)
            )
            
            if len(train_features) < self.batch_size:
                return None
            
            # Convert to tensors
            X = torch.from_numpy(train_features).float()
            y = torch.from_numpy(train_targets).float()
            
            # Create dataloader
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Training loop
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = self.criterion(pred, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            
            # Record timing
            duration = time.time() - start_time
            self.monitor.record_update_time(duration)
            
            self._update_count += 1
            
            # Create metrics dict
            metrics = {
                'update_number': self._update_count,
                'samples_seen': self._sample_count,
                'training_loss': avg_loss,
                'update_duration_ms': duration * 1000,
                **self.monitor.get_recent_stats(),
            }
            
            # Checkpoint if enabled
            if self.checkpoint_dir and self._update_count % 10 == 0:
                self.save_checkpoint(f"checkpoint_{self._update_count}.pt")
            
            return metrics
            
        finally:
            self._is_updating = False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'sample_count': self._sample_count,
            'update_count': self._update_count,
            'buffer_size': len(self.buffer),
            **self.monitor.get_recent_stats(),
            **self.monitor.get_full_stats(),
            **self.monitor.get_update_stats(),
        }
    
    def check_performance(self) -> Tuple[bool, str]:
        """Check if model performance is degrading."""
        return self.monitor.check_drift()
    
    def save_checkpoint(self, filename: str) -> Path:
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            raise ValueError("Checkpoint directory not set")
        
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'sample_count': self._sample_count,
            'update_count': self._update_count,
        }, path)
        
        logger.info(f"Saved checkpoint to {path}")
        return path
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        if not self.checkpoint_dir:
            raise ValueError("Checkpoint directory not set")
        
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._sample_count = checkpoint['sample_count']
        self._update_count = checkpoint['update_count']
        
        logger.info(f"Loaded checkpoint from {path}")


class DataStreamer:
    """
    Data streaming utility for online learning.
    
    Supports:
    - File-based streaming (CSV, NPZ)
    - Real-time data callbacks
    - Rate limiting
    """
    
    def __init__(
        self,
        source: Optional[Path] = None,
        batch_size: int = 32,
        callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
    ):
        self.source = source
        self.batch_size = batch_size
        self.callback = callback
        
        self._data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._index: int = 0
        
        if source:
            self._load_data(source)
            
    def _load_data(self, path: Path) -> None:
        """Load data from file."""
        if path.suffix == '.npz':
            data = np.load(path)
            self._data = (data['X'], data['y'])
        elif path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(path)
            # Assume last columns are targets
            target_cols = [c for c in df.columns if 'target' in c.lower()]
            if target_cols:
                X = df.drop(columns=target_cols).values
                y = df[target_cols].values
            else:
                X = df.values[:, :-1]
                y = df.values[:, -1:]
            self._data = (X, y)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over data in batches."""
        if self._data is None:
            return
            
        X, y = self._data
        
        for i in range(0, len(X), self.batch_size):
            batch_X = X[i:i + self.batch_size]
            batch_y = y[i:i + self.batch_size]
            
            if self.callback:
                self.callback(batch_X, batch_y)
                
            yield batch_X, batch_y
            
    def reset(self) -> None:
        """Reset the streamer to the beginning."""
        self._index = 0
        
    def set_callback(
        self,
        callback: Callable[[np.ndarray, np.ndarray], None]
    ) -> None:
        """Set callback function for each batch."""
        self.callback = callback


# Example model for testing
class SimpleOnlineModel(nn.Module):
    """Simple model for online learning demonstration."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x):
        return self.net(x)


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Online Learning Framework")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--update-interval", type=int, default=100)
    args = parser.parse_args()
    
    print("Testing Online Learning Framework")
    print("=" * 50)
    
    # Create model
    input_dim = 5
    output_dim = 1
    model = SimpleOnlineModel(input_dim)
    
    # Create online learner
    learner = OnlineLearner(
        model=model,
        batch_size=args.batch_size,
        update_interval=args.update_interval,
        buffer_size=1000,
    )
    
    # Simulate streaming data
    print(f"\nProcessing {args.num_samples} samples...")
    
    for i in range(args.num_samples // args.batch_size):
        # Generate random batch
        features = np.random.randn(args.batch_size, input_dim).astype(np.float32)
        targets = (features[:, 0] + features[:, 1] * 0.5 + np.random.randn(args.batch_size) * 0.1).reshape(-1, 1).astype(np.float32)
        
        # Process sample
        result = learner.process_sample(features, targets)
        
        if result:
            print(f"Update #{result['update_number']}: loss={result['training_loss']:.4f}")
    
    # Get final metrics
    print("\nFinal Metrics:")
    metrics = learner.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Check for drift
    has_drift, message = learner.check_performance()
    print(f"\nDrift Check: {message}")
