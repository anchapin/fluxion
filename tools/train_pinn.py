#!/usr/bin/env python3
"""
PINN (Physics-Informed Neural Network) Training Pipeline

This script implements a comprehensive PINN training pipeline for thermal modeling
that respects the underlying physics equations (5R1C Thermal Network model).

The PINN approach combines:
1. Data-driven learning from simulation data
2. Physics constraints via automatic differentiation
3. Energy balance constraints
4. Thermal mass dynamics

Physics Equations (5R1C Model):
- C * dT/dt = Q_solar + Q_internal + Q_heating + Q_cooling - Q_transmission - Q_ventilation
- Q_transmission = H_tr * (T_indoor - T_outdoor)
- Q_ventilation = H_ve * (T_indoor - T_outdoor)

Where:
- C = thermal capacity (kWh/K)
- H_tr = transmission heat transfer coefficient (W/K)
- H_ve = ventilation heat transfer coefficient (W/K)
- T = temperature (°C)
- Q = heat flux (W)

Related to Issue #326: Implement PINN (Physics-Informed Neural Network) Training
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Physics Constants and Configuration
# ============================================================================

@dataclass
class PhysicsConfig:
    """Configuration for thermal physics model."""
    # Thermal capacity (kWh/K) - typical residential: 50-100 kWh/K
    thermal_capacity: float = 50.0
    # Transmission heat transfer coefficient (W/K)
    h_transmission: float = 200.0
    # Ventilation heat transfer coefficient (W/K)
    h_ventilation: float = 50.0
    # Solar heat gain coefficient
    solar_gain: float = 50.0
    # Internal heat gains (W)
    internal_gains: float = 100.0
    # Time step in seconds (typically 3600 for hourly)
    time_step: float = 3600.0
    # Stefan-Boltzmann constant
    stefan_boltzmann: float = 5.67e-8


@dataclass
class PINNConfig:
    """Configuration for PINN training."""
    # Network architecture
    input_dim: int = 4  # [time, T_outdoor, Q_solar, Q_internal]
    output_dim: int = 1  # T_indoor
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64, 64])
    
    # Physics loss weights
    data_weight: float = 1.0
    physics_weight: float = 0.1
    initial_condition_weight: float = 1.0
    boundary_weight: float = 1.0
    energy_balance_weight: float = 0.1
    
    # Training
    learning_rate: float = 1e-3
    epochs: int = 5000
    batch_size: int = 128
    patience: int = 100
    
    # Collocation points for physics
    n_collocation: int = 10000
    
    # Seed
    seed: int = 42


# ============================================================================
# Neural Network Architecture
# ============================================================================

class ThermalPINN(nn.Module):
    """
    Physics-Informed Neural Network for thermal modeling.
    
    Takes (t, T_outdoor, Q_solar, Q_internal) as input and predicts T_indoor.
    The network is trained to satisfy the thermal physics equations.
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 1,
        hidden_dims: List[int] = [64, 64, 64],
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # First layer: input -> first hidden
        layers.append(nn.Linear(prev_dim, hidden_dims[0]))
        layers.append(nn.Tanh())
        prev_dim = hidden_dims[0]  # Update prev_dim after first layer
        
        # Hidden layers
        for hidden_dim in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        # Output layer - linear for temperature prediction
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Columns: [time, T_outdoor, Q_solar, Q_internal]
        
        Returns:
            Predicted T_indoor of shape (batch_size, 1)
        """
        return self.network(x)
    
    def predict_temperature(
        self,
        time: torch.Tensor,
        t_outdoor: torch.Tensor,
        q_solar: torch.Tensor,
        q_internal: torch.Tensor,
    ) -> torch.Tensor:
        """Helper to predict temperature with individual inputs."""
        x = torch.stack([time, t_outdoor, q_solar, q_internal], dim=1)
        return self.forward(x)


# ============================================================================
# Physics-Based Loss Functions
# ============================================================================

class PINNLoss(nn.Module):
    """
    Physics-Informed Neural Network Loss.
    
    Combines multiple loss terms:
    1. Data loss: MSE between predictions and observed data
    2. Physics loss: Residual of the PDE (thermal energy balance)
    3. Initial condition loss: Match known initial temperatures
    4. Boundary condition loss: Physical constraints (temperature bounds)
    5. Energy balance loss: Enforce energy conservation
    """
    
    def __init__(self, config: PINNConfig, physics_config: PhysicsConfig):
        super().__init__()
        self.config = config
        self.physics = physics_config
        
        # Register thermal capacity as learnable parameter
        self.thermal_capacity = nn.Parameter(
            torch.tensor([physics_config.thermal_capacity]), requires_grad=True
        )
        
        # Register heat transfer coefficients as learnable parameters
        self.h_transmission = nn.Parameter(
            torch.tensor([physics_config.h_transmission]), requires_grad=True
        )
        self.h_ventilation = nn.Parameter(
            torch.tensor([physics_config.h_ventilation]), requires_grad=True
        )
    
    def compute_physics_residual(
        self,
        model: nn.Module,
        t: torch.Tensor,
        t_outdoor: torch.Tensor,
        q_solar: torch.Tensor,
        q_internal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the physics residual using automatic differentiation.
        
        The thermal ODE is:
        C * dT/dt = Q_solar + Q_internal - H_tr * (T - T_outdoor) - H_ve * (T - T_outdoor)
        
        Rearranged:
        C * dT/dt + (H_tr + H_ve) * T = Q_solar + Q_internal + (H_tr + H_ve) * T_outdoor
        
        The residual should be zero when the physics is satisfied.
        """
        # Enable gradient computation for t
        t.requires_grad_(True)
        
        # Get prediction with gradient tracking
        t_pred = self._get_t_pred_with_grad(model, t, t_outdoor, q_solar, q_internal)
        
        # Compute dT/dt using automatic differentiation
        dt = grad(
            outputs=t_pred,
            inputs=t,
            grad_outputs=torch.ones_like(t_pred),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Thermal capacity (convert from kWh/K to J/K = 3.6e6 J/K)
        C = self.thermal_capacity * 3.6e6  # J/K
        
        # Total heat transfer coefficient
        H = self.h_transmission + self.h_ventilation  # W/K = J/(s·K)
        
        # Left side: C * dT/dt + H * (T - T_outdoor)
        # Right side: Q_solar + Q_internal (in Watts = J/s)
        
        # Heat gains in Watts
        q_total = q_solar + q_internal  # Assuming already in Watts
        
        # Physics residual: C * dT/dt - Q_solar - Q_internal + H * (T - T_outdoor)
        residual = C * dt - q_total + H * (t_pred - t_outdoor)
        
        return residual
    
    def _get_t_pred_with_grad(
        self,
        model: nn.Module,
        t: torch.Tensor,
        t_outdoor: torch.Tensor,
        q_solar: torch.Tensor,
        q_internal: torch.Tensor,
    ) -> torch.Tensor:
        """Get temperature prediction with gradient tracking."""
        # Stack inputs
        x = torch.stack([t.squeeze(), t_outdoor.squeeze(), q_solar.squeeze(), q_internal.squeeze()], dim=1)
        
        # Get prediction from the model
        t_pred = model(x)
        
        return t_pred
    
    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        t_collocation: torch.Tensor,
        t_outdoor_collocation: torch.Tensor,
        q_solar_collocation: torch.Tensor,
        q_internal_collocation: torch.Tensor,
        t_initial: Optional[torch.Tensor] = None,
        t_initial_pred: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined PINN loss.
        
        Args:
            predictions: Model predictions on data points
            targets: Ground truth targets
            t_collocation: Time values for physics collocation
            t_outdoor_collocation: Outdoor temperatures for collocation
            q_solar_collocation: Solar gains for collocation
            q_internal_collocation: Internal gains for collocation
            t_initial: Initial temperature observations (optional)
            t_initial_pred: Initial temperature predictions (optional)
        
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        loss_components = {}
        
        # 1. Data Fitting Loss (MSE)
        if targets is not None:
            data_loss = nn.functional.mse_loss(predictions, targets)
        else:
            data_loss = torch.tensor(0.0, device=predictions.device)
        loss_components["data"] = data_loss.item()
        
        # 2. Physics Loss (PDE residual)
        if t_collocation is not None and len(t_collocation) > 0:
            t_collocation.requires_grad_(True)
            physics_residual = self.compute_physics_residual(
                model,
                t_collocation,
                t_outdoor_collocation,
                q_solar_collocation,
                q_internal_collocation,
            )
            physics_loss = torch.mean(physics_residual ** 2)
        else:
            physics_loss = torch.tensor(0.0, device=predictions.device)
        loss_components["physics"] = physics_loss.item()
        
        # 3. Initial Condition Loss
        if t_initial is not None and t_initial_pred is not None:
            ic_loss = nn.functional.mse_loss(t_initial_pred, t_initial)
        else:
            ic_loss = torch.tensor(0.0, device=predictions.device)
        loss_components["initial_condition"] = ic_loss.item()
        
        # 4. Boundary Condition Loss (temperature bounds)
        if targets is not None:
            # Physical bounds: -30°C to 50°C for indoor temperature
            min_temp = -30.0
            max_temp = 50.0
            
            below_min = nn.functional.relu(torch.tensor(min_temp) - predictions)
            above_max = nn.functional.relu(predictions - torch.tensor(max_temp))
            boundary_loss = torch.mean(below_min + above_max)
        else:
            boundary_loss = torch.tensor(0.0, device=predictions.device)
        loss_components["boundary"] = boundary_loss.item()
        
        # 5. Energy Balance Loss
        # At steady state: Q_in = H * (T - T_outdoor)
        if targets is not None:
            # Expected steady-state temperature difference
            # Delta_T = Q_total / H
            # T = T_outdoor + Q_total / H
            total_heat = q_solar_collocation + q_internal_collocation
            H_total = self.h_transmission + self.h_ventilation
            expected_delta_t = total_heat / H_total
            energy_balance_loss = torch.mean(
                (predictions.mean() - t_outdoor_collocation.mean() - expected_delta_t.mean()) ** 2
            )
        else:
            energy_balance_loss = torch.tensor(0.0, device=predictions.device)
        loss_components["energy_balance"] = energy_balance_loss.item()
        
        # Combined loss with weights
        total_loss = (
            self.config.data_weight * data_loss +
            self.config.physics_weight * physics_loss +
            self.config.initial_condition_weight * ic_loss +
            self.config.boundary_weight * boundary_loss +
            self.config.energy_balance_weight * energy_balance_loss
        )
        
        loss_components["total"] = total_loss.item()
        
        return total_loss, loss_components


# ============================================================================
# Data Generation
# ============================================================================

class ThermalDataGenerator:
    """
    Generate training data using 5R1C thermal model.
    
    This simulates realistic thermal dynamics that the PINN will learn to predict.
    """
    
    def __init__(self, physics_config: PhysicsConfig, seed: int = 42):
        self.physics = physics_config
        self.rng = np.random.RandomState(seed)
    
    def simulate(
        self,
        t_outdoor: np.ndarray,
        q_solar: np.ndarray,
        q_internal: np.ndarray,
        t_initial: float = 20.0,
    ) -> np.ndarray:
        """
        Simulate indoor temperature over time.
        
        Args:
            t_outdoor: Outdoor temperature array (°C)
            q_solar: Solar heat gains array (W)
            q_internal: Internal heat gains array (W)
            t_initial: Initial indoor temperature (°C)
        
        Returns:
            Indoor temperature array (°C)
        """
        n_steps = len(t_outdoor)
        dt = self.physics.time_step  # seconds
        
        # Thermal capacity in J/K
        C = self.physics.thermal_capacity * 3.6e6
        
        # Total heat transfer coefficient
        H = self.physics.h_transmission + self.physics.h_ventilation
        
        # Initialize temperature array
        t_indoor = np.zeros(n_steps)
        t_indoor[0] = t_initial
        
        # Time stepping (explicit Euler)
        for i in range(1, n_steps):
            # Heat balance: C * dT/dt = Q_in - H * (T - T_outdoor)
            q_total = q_solar[i] + q_internal[i]
            dT = (q_total - H * (t_indoor[i-1] - t_outdoor[i])) / C * dt
            t_indoor[i] = t_indoor[i-1] + dT
        
        return t_indoor
    
    def generate_training_data(
        self,
        n_samples: int,
        n_timesteps: int = 24,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Generate training dataset.
        
        Returns:
            Tuple of (inputs_dict, targets_array)
        """
        X = {
            "time": [],
            "t_outdoor": [],
            "q_solar": [],
            "q_internal": [],
        }
        y = []
        
        for _ in range(n_samples):
            # Generate outdoor temperature profile (daily cycle with noise)
            t_base = self.rng.uniform(5, 30)  # Base outdoor temp
            t_amplitude = self.rng.uniform(5, 15)  # Daily variation
            t_outdoor = t_base + t_amplitude * np.sin(
                2 * np.pi * np.arange(n_timesteps) / n_timesteps
            ) + self.rng.normal(0, 1, n_timesteps)
            
            # Solar gains (peak at noon)
            solar_base = self.rng.uniform(0, 100)
            q_solar = solar_base * np.maximum(
                0, np.sin(2 * np.pi * np.arange(n_timesteps) / n_timesteps)
            ) + self.rng.normal(0, 5, n_timesteps)
            q_solar = np.maximum(0, q_solar)
            
            # Internal gains
            q_internal = self.rng.uniform(50, 200) + self.rng.normal(0, 10, n_timesteps)
            q_internal = np.maximum(0, q_internal)
            
            # Initial temperature
            t_initial = self.rng.uniform(18, 26)
            
            # Simulate
            t_indoor = self.simulate(t_outdoor, q_solar, q_internal, t_initial)
            
            # Add to dataset
            time_normalized = np.arange(n_timesteps) / n_timesteps
            X["time"].extend(time_normalized)
            X["t_outdoor"].extend(t_outdoor)
            X["q_solar"].extend(q_solar)
            X["q_internal"].extend(q_internal)
            y.extend(t_indoor)
        
        # Convert to arrays
        for key in X:
            X[key] = np.array(X[key], dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        
        return X, y
    
    def generate_collocation_points(
        self,
        n_points: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate collocation points for physics loss.
        
        These are random points in the input space where the PDE will be enforced.
        """
        # Time: uniform in [0, 1]
        t = torch.rand(n_points, 1)
        
        # Outdoor temperature: uniform in [-10, 40]
        t_outdoor = torch.rand(n_points, 1) * 50 - 10
        
        # Solar gains: uniform in [0, 200]
        q_solar = torch.rand(n_points, 1) * 200
        
        # Internal gains: uniform in [0, 300]
        q_internal = torch.rand(n_points, 1) * 300
        
        return t, t_outdoor, q_solar, q_internal


# ============================================================================
# Training Functions
# ============================================================================

def train_pinn(
    model: nn.Module,
    train_loader: DataLoader,
    collocation_data: Tuple[torch.Tensor, ...],
    config: PINNConfig,
    physics_config: PhysicsConfig,
    val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the PINN model.
    
    Args:
        model: PINN model
        train_loader: DataLoader for training data
        collocation_data: Tuple of (t, t_outdoor, q_solar, q_internal) for physics
        config: PINN configuration
        physics_config: Physics configuration
        val_data: Optional validation data (X, y)
        output_dir: Optional directory to save outputs
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    torch.manual_seed(config.seed)
    
    # Move collocation data to device
    device = next(model.parameters()).device
    t_coll, t_out_coll, q_sol_coll, q_int_coll = collocation_data
    t_coll = t_coll.to(device)
    t_out_coll = t_out_coll.to(device)
    q_sol_coll = q_sol_coll.to(device)
    q_int_coll = q_int_coll.to(device)
    
    # Loss function
    criterion = PINNLoss(config, physics_config).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50
    )
    
    # Training history
    history: Dict[str, List[float]] = {
        "loss": [],
        "val_loss": [],
        "data_loss": [],
        "physics_loss": [],
        "ic_loss": [],
        "boundary_loss": [],
        "energy_loss": [],
    }
    
    # Get validation data
    if val_data is not None:
        X_val, y_val = val_data
        X_val_t = torch.from_numpy(X_val).float().to(device)
        y_val_t = torch.from_numpy(y_val).float().to(device)
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    logger.info(f"Starting PINN training for {config.epochs} epochs...")
    logger.info(f"Physics config: C={physics_config.thermal_capacity} kWh/K, "
                f"H_tr={physics_config.h_transmission} W/K, "
                f"H_ve={physics_config.h_ventilation} W/K")
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        epoch_ic_loss = 0.0
        epoch_boundary_loss = 0.0
        epoch_energy_loss = 0.0
        
        n_batches = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_X)
            
            # Extract collocation data for this batch (sample from full set)
            batch_size = batch_X.shape[0]
            idx = torch.randperm(t_coll.shape[0])[:batch_size]
            t_col = t_coll[idx]
            t_out_col = t_out_coll[idx]
            q_sol_col = q_sol_coll[idx]
            q_int_col = q_int_coll[idx]
            
            # Compute loss
            loss, components = criterion(
                model=model,
                predictions=predictions,
                targets=batch_y,
                t_collocation=t_col,
                t_outdoor_collocation=t_out_col,
                q_solar_collocation=q_sol_col,
                q_internal_collocation=q_int_col,
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += components["total"]
            epoch_data_loss += components["data"]
            epoch_physics_loss += components["physics"]
            epoch_ic_loss += components["initial_condition"]
            epoch_boundary_loss += components["boundary"]
            epoch_energy_loss += components["energy_balance"]
            n_batches += 1
        
        # Average losses
        avg_loss = epoch_loss / n_batches
        avg_data_loss = epoch_data_loss / n_batches
        avg_physics_loss = epoch_physics_loss / n_batches
        avg_ic_loss = epoch_ic_loss / n_batches
        avg_boundary_loss = epoch_boundary_loss / n_batches
        avg_energy_loss = epoch_energy_loss / n_batches
        
        # Validation
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = nn.functional.mse_loss(val_pred, y_val_t).item()
            model.train()
        else:
            val_loss = avg_loss
        
        # Update history
        history["loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["data_loss"].append(avg_data_loss)
        history["physics_loss"].append(avg_physics_loss)
        history["ic_loss"].append(avg_ic_loss)
        history["boundary_loss"].append(avg_boundary_loss)
        history["energy_loss"].append(avg_energy_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if output_dir is not None:
                torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1
        
        # Logging
        if (epoch + 1) % 100 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{config.epochs} | "
                f"Loss: {avg_loss:.6f} | "
                f"Data: {avg_data_loss:.6f} | "
                f"Physics: {avg_physics_loss:.6f} | "
                f"Val: {val_loss:.6f}"
            )
        
        # Early stopping
        if patience_counter >= config.patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if output_dir is not None and (output_dir / "best_model.pt").exists():
        model.load_state_dict(torch.load(output_dir / "best_model.pt"))
        logger.info(f"Loaded best model with val_loss: {best_val_loss:.6f}")
    
    return model, history


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained PINN model
        X: Input features
        y: Ground truth targets
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)
    
    with torch.no_grad():
        predictions = model(X_t)
    
    # Compute metrics
    mse = nn.functional.mse_loss(predictions, y_t).item()
    mae = nn.functional.l1_loss(predictions, y_t).item()
    
    # R-squared
    ss_res = torch.sum((y_t - predictions) ** 2).item()
    ss_tot = torch.sum((y_t - torch.mean(y_t)) ** 2).item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # RMSE
    rmse = np.sqrt(mse)
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def export_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: Path,
):
    """Export PINN model to ONNX format."""
    logger.info(f"Exporting PINN to ONNX: {output_path}")
    
    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        input_names=["input"],
        output_names=["t_indoor"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "t_indoor": {0: "batch_size"},
        },
        opset_version=17,
    )
    logger.info("ONNX export successful")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PINN Training for Thermal Modeling"
    )
    
    # Data arguments
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--n-timesteps", type=int, default=24, help="Timesteps per sample")
    parser.add_argument("--n-collocation", type=int, default=10000, help="Collocation points")
    
    # Physics arguments
    parser.add_argument("--thermal-capacity", type=float, default=50.0, help="Thermal capacity (kWh/K)")
    parser.add_argument("--h-transmission", type=float, default=200.0, help="Transmission H (W/K)")
    parser.add_argument("--h-ventilation", type=float, default=50.0, help="Ventilation H (W/K)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Loss weights
    parser.add_argument("--data-weight", type=float, default=1.0, help="Data loss weight")
    parser.add_argument("--physics-weight", type=float, default=0.1, help="Physics loss weight")
    parser.add_argument("--boundary-weight", type=float, default=1.0, help="Boundary loss weight")
    parser.add_argument("--energy-weight", type=float, default=0.1, help="Energy balance weight")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="models/pinn", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Physics configuration
    physics_config = PhysicsConfig(
        thermal_capacity=args.thermal_capacity,
        h_transmission=args.h_transmission,
        h_ventilation=args.h_ventilation,
    )
    
    # PINN configuration
    config = PINNConfig(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        n_collocation=args.n_collocation,
        data_weight=args.data_weight,
        physics_weight=args.physics_weight,
        boundary_weight=args.boundary_weight,
        energy_balance_weight=args.energy_weight,
        seed=args.seed,
    )
    
    logger.info("=" * 60)
    logger.info("PINN Training Pipeline for Thermal Modeling")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config}")
    logger.info(f"Physics: {physics_config}")
    
    # Generate training data
    logger.info("\n[1/5] Generating training data...")
    generator = ThermalDataGenerator(physics_config, seed=args.seed)
    X, y = generator.generate_training_data(args.n_samples, args.n_timesteps)
    
    # Split data
    split_idx = int(0.8 * len(y))
    X_train = np.column_stack([X["time"][:split_idx], X["t_outdoor"][:split_idx], 
                               X["q_solar"][:split_idx], X["q_internal"][:split_idx]])
    y_train = y[:split_idx]
    X_val = np.column_stack([X["time"][split_idx:], X["t_outdoor"][split_idx:], 
                             X["q_solar"][split_idx:], X["q_internal"][split_idx:]])
    y_val = y[split_idx:]
    
    logger.info(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}")
    
    # Save data
    np.savez(output_dir / "training_data.npz", X=X_train, y=y_train)
    np.savez(output_dir / "validation_data.npz", X=X_val, y=y_val)
    
    # Create data loaders
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Generate collocation points
    logger.info("\n[2/5] Generating collocation points...")
    t_coll, t_out_coll, q_sol_coll, q_int_coll = generator.generate_collocation_points(
        args.n_collocation
    )
    
    # Create model
    logger.info("\n[3/5] Creating PINN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = ThermalPINN(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        hidden_dims=config.hidden_dims,
    ).to(device)
    
    logger.info(f"Model architecture: {model}")
    
    # Train
    logger.info("\n[4/5] Training PINN...")
    start_time = time.time()
    
    model, history = train_pinn(
        model=model,
        train_loader=train_loader,
        collocation_data=(t_coll, t_out_coll, q_sol_coll, q_int_coll),
        config=config,
        physics_config=physics_config,
        val_data=(X_val, y_val),
        output_dir=output_dir,
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate
    logger.info("\n[5/5] Evaluating model...")
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    
    logger.info(f"Training metrics: {train_metrics}")
    logger.info(f"Validation metrics: {val_metrics}")
    
    # Save metrics
    metrics = {
        "training": train_metrics,
        "validation": val_metrics,
        "training_time_seconds": training_time,
        "config": vars(args),
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Export to ONNX
    sample_input = torch.randn(1, config.input_dim).to(device)
    export_onnx(model, sample_input, output_dir / "pinn_thermal.onnx")
    
    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info("PINN training complete!")
    
    return model, metrics


if __name__ == "__main__":
    main()
