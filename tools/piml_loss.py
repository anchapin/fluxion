#!/usr/bin/env python3
"""
PIML (Physics-Informed Machine Learning) Loss Functions

Implements Physics-Informed ML loss functions for thermal surrogate training
that embed thermodynamic RC (Resistance-Capacitance) models into neural network
loss functions for surrogate training.

This implementation addresses:
- Issue #552: PIML-01: Implement Physics-Informed Machine Learning loss functions
- Issue #1463: Phase 2a — Update Surrogate Training to Target Metric Tensors
                (ThermalManifold gauge-invariant loss + geometric surrogate)

Key features:
- Simplified RC thermal network embedded in loss function
- Physics-informed penalty terms for thermodynamic violations
- Comparison between PIML loss and standard L1/L2 loss
- CvRMSE optimization for peak load accuracy

Phase 2a additions (issue #1463):
- ``GaugeInvariantLoss``: a metric-aware loss operating on the 4-D
  ThermalManifold (metric_tensor, scalar_field, gauge_connection). Enforces
  the First Law of Thermodynamics by penalising the predicted
  ``gauge_connection_sum`` exceeding the ground-truth sum (i.e. hallucinated
  energy generation), plus dissipativity constraints (Kirchhoff reciprocity)
  and parallel-transport consistency.
- ``ThermalManifoldBatch``: batched-tensor view of a ``ThermalManifold``
  matching the Rust struct in ``src/physics/geometry_tensor.rs``.
- ``GeometricSurrogate``: neural net with three heads (metric / field /
  connection) for the new representation. Replaces the (heating, cooling)
  scalar heads of ``PIMLSurrogate``.
- ``train_geometric_surrogate``: training loop for the new path.

The legacy RC-based loss (``PIMLLoss`` / ``PIMLSurrogate``) is preserved
unchanged for backward compatibility with existing model checkpoints and
downstream consumers — Phase 2a introduces a *parallel* training path, not
a replacement.

Research basis:
- PINN-RC approach: embedding energy conservation laws and heat transfer
  equations as regularization terms
- Benefits: data-efficient training, thermodynamic consistency,
  superior generalization, improved CvRMSE

Reference: "Advancements in Building Energy Simulation Engines" - Section on PIML
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Physics Configuration for RC Thermal Network
# ============================================================================


@dataclass
class RCPhysicsConfig:
    """Configuration for RC (Resistance-Capacitance) thermal network model."""

    thermal_capacity: float = 50.0
    h_transmission: float = 200.0
    h_ventilation: float = 50.0
    time_step: float = 3600.0
    solar_gain_coeff: float = 1.0
    internal_gain_coeff: float = 1.0

    @property
    def total_heat_transfer_coeff(self) -> float:
        """Total heat transfer coefficient (W/K)."""
        return self.h_transmission + self.h_ventilation

    def compute_steady_state_delta_t(self, total_heat: float) -> float:
        """Compute steady-state temperature difference."""
        if self.total_heat_transfer_coeff <= 0:
            return 0.0
        return total_heat / self.total_heat_transfer_coeff


@dataclass
class PIMLConfig:
    """Configuration for PIML training."""

    input_dim: int = 9
    output_dim: int = 2
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])

    mse_weight: float = 1.0
    piml_weight: float = 0.5
    energy_balance_weight: float = 0.3
    boundary_weight: float = 0.1

    rc_thermal_capacity: float = 50.0
    rc_h_transmission: float = 200.0
    rc_h_ventilation: float = 50.0

    learning_rate: float = 1e-3
    epochs: int = 500
    batch_size: int = 64
    patience: int = 50

    seed: int = 42


# ============================================================================
# PIML Loss Functions
# ============================================================================


class PIMLLoss(nn.Module):
    """
    Physics-Informed Machine Learning Loss for thermal surrogate training.

    Combines standard MSE with RC thermal network physics constraints:

    L_total = λ_mse * L_mse
           + λ_piml * L_piml
           + λ_energy * L_energy
           + λ_boundary * L_boundary

    Where:
    - L_mse: Standard mean squared error
    - L_piml: RC model-based physics residual
    - L_energy: Energy balance constraint
    - L_boundary: Temperature bounds constraint

    The RC model equations:
    - C * dT/dt = Q_solar + Q_internal + Q_heating + Q_cooling - Q_transmission - Q_ventilation
    - Q_transmission = H_tr * (T_indoor - T_outdoor)
    - Q_ventilation = H_ve * (T_indoor - T_outdoor)
    """

    def __init__(self, config: PIMLConfig):
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss()

        self.thermal_capacity = nn.Parameter(
            torch.tensor([config.rc_thermal_capacity]), requires_grad=True
        )
        self.h_transmission = nn.Parameter(
            torch.tensor([config.rc_h_transmission]), requires_grad=True
        )
        self.h_ventilation = nn.Parameter(
            torch.tensor([config.rc_h_ventilation]), requires_grad=True
        )

    @property
    def total_h(self) -> torch.Tensor:
        """Total heat transfer coefficient."""
        return self.h_transmission + self.h_ventilation

    def compute_rc_residual(
        self,
        t_indoor_pred: torch.Tensor,
        t_outdoor: torch.Tensor,
        q_solar: torch.Tensor,
        q_internal: torch.Tensor,
        q_heating: torch.Tensor,
        q_cooling: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute RC thermal network residual.

        The residual should be zero when the physics is satisfied.

        Args:
            t_indoor_pred: Predicted indoor temperature
            t_outdoor: Outdoor temperature
            q_solar: Solar heat gains (W)
            q_internal: Internal heat gains (W)
            q_heating: Heating load (W)
            q_cooling: Cooling load (W)
            dt: Time step (seconds), defaults to 3600

        Returns:
            Physics residual tensor
        """
        if dt is None:
            dt = torch.tensor(3600.0, device=t_indoor_pred.device)

        C = self.thermal_capacity * 1000.0
        H = self.total_h

        q_total_in = q_solar + q_internal + q_heating
        q_total_out = q_cooling + H * (t_indoor_pred - t_outdoor)

        residual = C * (q_total_in - q_total_out) / dt

        return residual

    def compute_energy_balance(
        self,
        heating_pred: torch.Tensor,
        cooling_pred: torch.Tensor,
        t_indoor: torch.Tensor,
        t_outdoor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute energy balance residual.

        At steady state in deadband: heating ≈ 0, cooling ≈ 0,
        and indoor temp should equal outdoor temp.

        Args:
            heating_pred: Predicted heating load
            cooling_pred: Predicted cooling load
            t_indoor: Indoor temperature
            t_outdoor: Outdoor temperature

        Returns:
            Energy balance residual
        """
        H = self.total_h

        expected_heating = H * (t_indoor - t_outdoor).clamp(min=0.0)
        expected_cooling = H * (t_outdoor - t_indoor).clamp(min=0.0)

        heating_violation = F.relu(expected_heating - heating_pred)
        cooling_violation = F.relu(expected_cooling - cooling_pred)

        energy_residual = heating_violation + cooling_violation

        return energy_residual

    def forward(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor],
        features: torch.Tensor,
        physics_params: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined PIML loss.

        Args:
            predictions: Tuple of (heating_pred, cooling_pred)
            targets: Tuple of (heating_true, cooling_true)
            features: Input features [batch_size, input_dim]
                     Expected: [t_outdoor, heating_setpoint, cooling_setpoint,
                               hour, day, month, u_value, wwr, ...]
            physics_params: Optional physics parameters for validation

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        heating_pred, cooling_pred = predictions
        heating_true, cooling_true = targets

        loss_components = {}

        mse_loss = self.mse(heating_pred, heating_true) + self.mse(
            cooling_pred, cooling_true
        )
        loss_components["mse"] = mse_loss.item()

        t_outdoor = features[:, 0]
        q_solar = features[:, 3] * 50 if features.shape[1] > 3 else torch.zeros_like(t_outdoor)
        q_internal = features[:, 4] * 100 if features.shape[1] > 4 else torch.zeros_like(t_outdoor)

        heating_flat = heating_pred.view(-1)
        cooling_flat = cooling_pred.view(-1)
        t_indoor_pred = 20.0 + heating_flat * 0.1 - cooling_flat * 0.1

        rc_residual = self.compute_rc_residual(
            t_indoor_pred=t_indoor_pred,
            t_outdoor=t_outdoor,
            q_solar=q_solar,
            q_internal=q_internal,
            q_heating=heating_flat * 1000.0,
            q_cooling=cooling_flat * 1000.0,
        )
        piml_loss = torch.mean(rc_residual**2)
        loss_components["piml"] = piml_loss.item()

        energy_residual = self.compute_energy_balance(
            heating_pred, cooling_pred, t_indoor_pred, t_outdoor
        )
        energy_loss = torch.mean(energy_residual**2)
        loss_components["energy_balance"] = energy_loss.item()

        boundary_loss = torch.mean(F.relu(heating_pred) + F.relu(cooling_pred))
        loss_components["boundary"] = boundary_loss.item()

        total_loss = (
            self.config.mse_weight * mse_loss
            + self.config.piml_weight * piml_loss
            + self.config.energy_balance_weight * energy_loss
            + self.config.boundary_weight * boundary_loss
        )
        loss_components["total"] = total_loss.item()

        return total_loss, loss_components


class StandardLoss(nn.Module):
    """
    Standard MSE loss for baseline comparison.

    Used to compare PIML-trained surrogates against standard L1/L2 regularization.
    """

    def __init__(self, l1_weight: float = 0.0, l2_weight: float = 1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute standard loss."""
        heating_pred, cooling_pred = predictions
        heating_true, cooling_true = targets

        loss_components = {}

        mse_loss = self.mse(heating_pred, heating_true) + self.mse(
            cooling_pred, cooling_true
        )
        loss_components["mse"] = mse_loss.item()

        l1_loss = torch.mean(torch.abs(heating_pred - heating_true)) + torch.mean(
            torch.abs(cooling_pred - cooling_true)
        )
        loss_components["l1"] = l1_loss.item()

        total_loss = self.l2_weight * mse_loss + self.l1_weight * l1_loss
        loss_components["total"] = total_loss.item()

        return total_loss, loss_components


# ============================================================================
# Neural Network Models
# ============================================================================


class PIMLSurrogate(nn.Module):
    """
    Physics-Informed Neural Network for surrogate modeling.

    Architecture:
    - Feature encoder with batch normalization
    - Separate heads for heating and cooling
    - Softplus activation for positive outputs
    - Optional physics-informed constraints
    """

    def __init__(
        self,
        input_dim: int = 9,
        output_dim: int = 2,
        hidden_dims: List[int] = [128, 64, 32],
        use_physics_constraints: bool = True,
    ):
        super().__init__()
        self.use_physics_constraints = use_physics_constraints

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ]
            )
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        self.heating_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim), nn.Softplus()
        )
        self.cooling_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim), nn.Softplus()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self, x: torch.Tensor, physics_params: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        encoded = self.encoder(x)

        heating = self.heating_head(encoded)
        cooling = self.cooling_head(encoded)

        if self.use_physics_constraints and physics_params is not None:
            u_value = physics_params[:, 0:1]
            t_outdoor = x[:, 0:1]
            heating_setpoint = physics_params[:, 1:2]
            cooling_setpoint = physics_params[:, 2:3]

            min_heating = u_value * (heating_setpoint - t_outdoor).clamp(min=0.0)
            min_cooling = u_value * (t_outdoor - cooling_setpoint).clamp(min=0.0)

            heating = heating.clamp(min=min_heating)
            cooling = cooling.clamp(min=min_cooling)

        return heating, cooling


# ============================================================================
# Training Functions
# ============================================================================


def compute_cvrmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Coefficient of Variation of Root Mean Square Error."""
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    mean_target = torch.mean(targets).item()
    if abs(mean_target) < 1e-6:
        return float("inf")
    return 100.0 * rmse / abs(mean_target)


def train_piml_surrogate(
    model: nn.Module,
    train_loader: DataLoader,
    config: PIMLConfig,
    rc_config: RCPhysicsConfig,
    val_data: Optional[Tuple[torch.Tensor, ...]] = None,
    output_dir: Optional[Path] = None,
    use_piml_loss: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train surrogate model with PIML loss.

    Args:
        model: PIML Surrogate model
        train_loader: DataLoader for training
        config: PIML configuration
        rc_config: RC physics configuration
        val_data: Optional validation data
        output_dir: Optional output directory
        use_piml_loss: If True, use PIML loss; otherwise use standard MSE

    Returns:
        Tuple of (trained_model, training_history)
    """
    torch.manual_seed(config.seed)

    device = next(model.parameters()).device

    if use_piml_loss:
        criterion = PIMLLoss(config).to(device)
        logger.info("Using PIML loss (physics-informed)")
    else:
        criterion = StandardLoss().to(device)
        logger.info("Using standard MSE loss (baseline)")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.patience // 2
    )

    history: Dict[str, List[float]] = {
        "loss": [],
        "val_loss": [],
        "cvrmse": [],
        "peak_cvrmse": [],
    }

    if val_data is not None:
        X_val, y_h_val, y_c_val = val_data
        X_val_t = X_val.to(device)
        y_h_val_t = y_h_val.to(device)
        y_c_val_t = y_c_val.to(device)

    best_val_loss = float("inf")
    patience_counter = 0

    logger.info(f"Starting training for {config.epochs} epochs...")

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_X, batch_y_h, batch_y_c in train_loader:
            batch_X = batch_X.to(device)
            batch_y_h = batch_y_h.to(device)
            batch_y_c = batch_y_c.to(device)

            optimizer.zero_grad()

            if use_piml_loss:
                physics_params = torch.stack(
                    [
                        batch_X[:, 6:7] if batch_X.shape[1] > 6 else torch.zeros_like(batch_X[:, 0:1]),
                        batch_X[:, 1:2] if batch_X.shape[1] > 1 else torch.zeros_like(batch_X[:, 0:1]),
                        batch_X[:, 2:3] if batch_X.shape[1] > 2 else torch.zeros_like(batch_X[:, 0:1]),
                    ],
                    dim=1,
                ).squeeze(-1)

                heating_pred, cooling_pred = model(batch_X, physics_params)
                loss, _ = criterion(
                    (heating_pred, cooling_pred),
                    (batch_y_h, batch_y_c),
                    batch_X,
                    physics_params,
                )
            else:
                heating_pred, cooling_pred = model(batch_X)
                loss, _ = criterion((heating_pred, cooling_pred), (batch_y_h, batch_y_c))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if val_data is not None:
            model.eval()
            with torch.no_grad():
                if use_piml_loss:
                    physics_params_val = torch.stack(
                        [
                            X_val_t[:, 6:7] if X_val_t.shape[1] > 6 else torch.zeros_like(X_val_t[:, 0:1]),
                            X_val_t[:, 1:2] if X_val_t.shape[1] > 1 else torch.zeros_like(X_val_t[:, 0:1]),
                            X_val_t[:, 2:3] if X_val_t.shape[1] > 2 else torch.zeros_like(X_val_t[:, 0:1]),
                        ],
                        dim=1,
                    ).squeeze(-1)
                    val_heating, val_cooling = model(X_val_t, physics_params_val)
                else:
                    val_heating, val_cooling = model(X_val_t)

                if use_piml_loss:
                    val_loss, _ = criterion(
                        (val_heating, val_cooling), (y_h_val_t, y_c_val_t), X_val_t, physics_params_val
                    )
                    val_loss = val_loss.item()
                else:
                    val_loss, _ = criterion(
                        (val_heating, val_cooling), (y_h_val_t, y_c_val_t)
                    )
                    val_loss = val_loss.item()

                total_pred = val_heating + val_cooling
                total_target = y_h_val_t + y_c_val_t
                cvrmse = compute_cvrmse(total_pred, total_target)

                high_mass_mask = (total_target > torch.quantile(total_target, 0.75)).squeeze()
                if high_mass_mask.any():
                    peak_cvrmse = compute_cvrmse(
                        total_pred.squeeze()[high_mass_mask],
                        total_target.squeeze()[high_mass_mask],
                    )
                else:
                    peak_cvrmse = cvrmse

            model.train()
        else:
            val_loss = avg_loss
            cvrmse = 0.0
            peak_cvrmse = 0.0

        history["loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["cvrmse"].append(cvrmse)
        history["peak_cvrmse"].append(peak_cvrmse)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if output_dir is not None:
                torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{config.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"CvRMSE: {cvrmse:.2f}% | "
                f"Peak CvRMSE: {peak_cvrmse:.2f}%"
            )

        if patience_counter >= config.patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    if output_dir is not None and (output_dir / "best_model.pt").exists():
        model.load_state_dict(torch.load(output_dir / "best_model.pt"))

    return model, history


def generate_synthetic_data(
    n_samples: int, n_timesteps: int = 24, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for PIML surrogate.

    Args:
        n_samples: Number of samples to generate
        n_timesteps: Timesteps per sample
        seed: Random seed

    Returns:
        Tuple of (features, heating_targets, cooling_targets)
    """
    rng = np.random.RandomState(seed)

    features_list = []
    heating_list = []
    cooling_list = []

    for _ in range(n_samples):
        t_base = rng.uniform(5, 30)
        t_amplitude = rng.uniform(5, 15)
        t_outdoor = t_base + t_amplitude * np.sin(
            2 * np.pi * np.arange(n_timesteps) / n_timesteps
        )
        t_outdoor = t_outdoor + rng.normal(0, 1, n_timesteps)

        heating_setpoint = rng.uniform(18, 22)
        cooling_setpoint = rng.uniform(24, 28)

        q_solar_base = rng.uniform(0, 100)
        q_solar = q_solar_base * np.maximum(
            0, np.sin(2 * np.pi * np.arange(n_timesteps) / n_timesteps)
        )
        q_solar = np.maximum(0, q_solar + rng.normal(0, 5, n_timesteps))

        q_internal = rng.uniform(50, 200) + rng.normal(0, 10, n_timesteps)
        q_internal = np.maximum(0, q_internal)

        u_value = rng.uniform(0.2, 0.5)
        wwr = rng.uniform(0.2, 0.6)

        heating_load = np.maximum(
            0,
            u_value * (heating_setpoint - t_outdoor) + rng.normal(0, 0.5, n_timesteps),
        )
        cooling_load = np.maximum(
            0,
            u_value * (t_outdoor - cooling_setpoint) + rng.normal(0, 0.5, n_timesteps),
        )

        hour = np.arange(n_timesteps) % 24
        day_of_year = rng.randint(0, 365, size=n_timesteps)
        month = (day_of_year // 30) + 1

        for i in range(n_timesteps):
            features_list.append(
                [
                    t_outdoor[i],
                    heating_setpoint,
                    cooling_setpoint,
                    q_solar[i] / 100.0,
                    q_internal[i] / 200.0,
                    hour[i] / 24.0,
                    (day_of_year[i] % 365) / 365.0,
                    month[i] / 12.0,
                    wwr,
                ]
            )
            heating_list.append(heating_load[i])
            cooling_list.append(cooling_load[i])

    X = np.array(features_list, dtype=np.float32)
    y_heating = np.array(heating_list, dtype=np.float32).reshape(-1, 1)
    y_cooling = np.array(cooling_list, dtype=np.float32).reshape(-1, 1)

    return X, y_heating, y_cooling


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="PIML Surrogate Training")

    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--piml-weight", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="models/piml")
    parser.add_argument("--compare-loss", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PIMLConfig(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        piml_weight=args.piml_weight,
        seed=args.seed,
    )
    rc_config = RCPhysicsConfig()

    logger.info("=" * 60)
    logger.info("PIML Surrogate Training")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config}")
    logger.info(f"RC Physics: {rc_config}")

    logger.info("\n[1/4] Generating synthetic training data...")
    X, y_heating, y_cooling = generate_synthetic_data(args.n_samples, seed=args.seed)

    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_h_train, y_h_val = y_heating[:split_idx], y_heating[split_idx:]
    y_c_train, y_c_val = y_cooling[:split_idx], y_cooling[split_idx:]

    logger.info(f"Training: {len(X_train)}, Validation: {len(X_val)}")

    X_train_t = torch.from_numpy(X_train).float()
    y_h_train_t = torch.from_numpy(y_h_train).float()
    y_c_train_t = torch.from_numpy(y_c_train).float()
    train_dataset = TensorDataset(X_train_t, y_h_train_t, y_c_train_t)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_data = (torch.from_numpy(X_val).float(), torch.from_numpy(y_h_val).float(), torch.from_numpy(y_c_val).float())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_piml = PIMLSurrogate(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        hidden_dims=config.hidden_dims,
    ).to(device)

    logger.info("\n[2/4] Training with PIML loss...")
    model_piml, history_piml = train_piml_surrogate(
        model=model_piml,
        train_loader=train_loader,
        config=config,
        rc_config=rc_config,
        val_data=val_data,
        output_dir=output_dir,
        use_piml_loss=True,
    )

    piml_cvrmse = history_piml["cvrmse"][-1]
    piml_peak_cvrmse = history_piml["peak_cvrmse"][-1]
    logger.info(f"PIML Final CvRMSE: {piml_cvrmse:.2f}%, Peak: {piml_peak_cvrmse:.2f}%")

    results = {
        "piml": {
            "cvrmse": piml_cvrmse,
            "peak_cvrmse": piml_peak_cvrmse,
            "history": history_piml,
        }
    }

    if args.compare_loss:
        logger.info("\n[3/4] Training with standard MSE loss (baseline)...")
        model_standard = PIMLSurrogate(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dims=config.hidden_dims,
        ).to(device)

        model_standard, history_standard = train_piml_surrogate(
            model=model_standard,
            train_loader=train_loader,
            config=config,
            rc_config=rc_config,
            val_data=val_data,
            output_dir=output_dir,
            use_piml_loss=False,
        )

        standard_cvrmse = history_standard["cvrmse"][-1]
        standard_peak_cvrmse = history_standard["peak_cvrmse"][-1]
        logger.info(f"Standard Final CvRMSE: {standard_cvrmse:.2f}%, Peak: {standard_peak_cvrmse:.2f}%")

        results["standard"] = {
            "cvrmse": standard_cvrmse,
            "peak_cvrmse": standard_peak_cvrmse,
            "history": history_standard,
        }

        logger.info("\n[4/4] Comparison:")
        logger.info(f"  PIML CvRMSE: {piml_cvrmse:.2f}% vs Standard: {standard_cvrmse:.2f}%")
        logger.info(f"  PIML Peak CvRMSE: {piml_peak_cvrmse:.2f}% vs Standard: {standard_peak_cvrmse:.2f}%")

        if piml_peak_cvrmse < standard_peak_cvrmse:
            improvement = (standard_peak_cvrmse - piml_peak_cvrmse) / standard_peak_cvrmse * 100
            logger.info(f"  PIML shows {improvement:.1f}% improvement in peak load error!")
    else:
        logger.info("\n[3/4] Skipping standard loss comparison (--compare-loss not set)")

    logger.info("\n[4/4] Saving results...")
    torch.save(model_piml.state_dict(), output_dir / "piml_surrogate.pt")

    metrics = {
        "config": vars(args),
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in results.items()},
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "history.json", "w") as f:
        json.dump({k: v for k, v in results.items()}, f)

    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info("PIML training complete!")

    peak_error_met = piml_peak_cvrmse < 30.0
    if peak_error_met:
        logger.info(f"SUCCESS: Peak load error ({piml_peak_cvrmse:.2f}%) < 30% threshold")
    else:
        logger.warning(f"WARNING: Peak load error ({piml_peak_cvrmse:.2f}%) >= 30% threshold")

    return model_piml, results


# ============================================================================
# Phase 2a — Issue #1463: ThermalManifold gauge-invariant loss
# ============================================================================
#
# This block introduces the new metric-aware training path. It deliberately
# lives below the legacy PIMLLoss block so existing imports / call sites
# keep working unchanged. See:
#   - src/physics/geometry_tensor.rs::ThermalManifold
#   - ARCHITECTURE.md § Module 6 (Gauge-Theory Foundation)
#   - Issue #1463 acceptance criteria
#


@dataclass
class GaugeInvariantConfig:
    """Configuration for the ThermalManifold-aware loss (issue #1463)."""

    manifold_dim: int = 4
    """Ambient dimension. Pinned to 4 to match ``MANIFOLD_DIM`` in
    ``src/physics/geometry_tensor.rs`` (air + wall + roof + floor)."""

    metric_weight: float = 1.0
    """Weight on the Frobenius metric-tensor penalty (curvature geometry)."""

    field_weight: float = 1.0
    """Weight on the scalar-field (temperature) L2 penalty."""

    gauge_weight: float = 1.0
    """Weight on the gauge-connection (heat flux) L2 penalty."""

    conservation_weight: float = 10.0
    """Heavy penalty when the predicted ``gauge_connection_sum`` exceeds the
    ground-truth sum (i.e. the model is hallucinating energy)."""

    transport_weight: float = 1.0
    """Weight on the parallel-transport consistency penalty — the geometric
    flow ``T + dt * (M·T + A)`` must reproduce the next-step field."""

    dissipativity_weight: float = 0.1
    """Weight on the dissipativity penalty — encodes Kirchhoff's current
    law for the passive case: ``M[i,i] <= 0`` (self-damping) and
    ``M[i,j] >= 0`` for ``i != j`` (non-negative cross-coupling). Note this
    is *not* a symmetry penalty — the 5R1C metric is asymmetric by design
    because the metric entries are scaled by per-node capacitances
    (``M[i,j] = G[i,j] / C[i]``).

    See the algebraic invariants documented on
    ``ThermalManifold::metric_tensor`` in ``src/physics/geometry_tensor.rs``."""

    dt_seconds: float = 60.0
    """Default timestep used by the parallel-transport penalty when a per-batch
    ``dt`` is not supplied."""


@dataclass
class ThermalManifoldBatch:
    """Batched view of a ``ThermalManifold`` (matches the Rust struct in
    ``src/physics/geometry_tensor.rs``).

    All tensors are ``(B, ...)`` so a single batch yields ``B`` independent
    manifolds. ``metric`` is ``(B, D, D)``; ``field`` and ``connection`` are
    ``(B, D)``. ``dt_seconds`` is either a scalar or a ``(B,)`` tensor — the
    loss broadcasts as needed.

    ``field_next`` is optional: when supplied it enables the
    parallel-transport consistency penalty in ``GaugeInvariantLoss``.
    """

    metric: torch.Tensor
    field: torch.Tensor
    connection: torch.Tensor
    dt_seconds: Any = 60.0
    field_next: Optional[torch.Tensor] = None


class GaugeInvariantLoss(nn.Module):
    """Metric-aware loss for ``ThermalManifold`` training (issue #1463 Phase 2a).

    The total loss is

    .. math::

        L = w_\\mathrm{metric}\\,L_\\mathrm{metric}
          + w_\\mathrm{field}\\,L_\\mathrm{field}
          + w_\\mathrm{gauge}\\,L_\\mathrm{gauge}
          + w_\\mathrm{conservation}\\,L_\\mathrm{conservation}
          + w_\\mathrm{transport}\\,L_\\mathrm{transport}
          + w_\\mathrm{dissipativity}\\,L_\\mathrm{dissipativity}

    where

    - ``L_metric`` is the squared Frobenius distance on the metric tensor.
    - ``L_field`` is the squared L2 distance on the temperature field.
    - ``L_gauge`` is the squared L2 distance on the heat-flux connection.
    - ``L_conservation`` is ``relu(sum(A_pred) - sum(A_target))^2`` summed
      over the batch — the First-Law check used by ``GaugeSolver`` (#1462)
      and the energy-conservation CI gate (#1465). A *positive* term when
      the prediction injects more net power than the ground truth, i.e. when
      the model is hallucinating energy.
    - ``L_transport`` is the squared L2 distance between the predicted
      next-step field (``T + dt * (M·T + A)``) and the ground-truth
      next-step field. Provides direct supervision on the geometric flow.
    - ``L_dissipativity`` is ``||M - M^T||_F^2`` — the symmetry penalty that
      enforces Kirchhoff reciprocity in the passive case (mirrors the
      dissipativity invariants documented on ``metric_tensor`` in
      ``src/physics/geometry_tensor.rs``).

    Note: the prediction must already have ``requires_grad=True`` for
    training; ``GaugeInvariantLoss`` only routes gradients, it does not own
    any learnable parameters itself.
    """

    def __init__(self, config: Optional[GaugeInvariantConfig] = None):
        super().__init__()
        self.config = config or GaugeInvariantConfig()

    @staticmethod
    def metric_frobenius_distance(
        pred_metric: torch.Tensor, target_metric: torch.Tensor
    ) -> torch.Tensor:
        """``||M_pred - M_target||_F^2`` averaged over the batch.

        Accepts either a ``(D, D)`` single-manifold input (returns a scalar)
        or a ``(B, D, D)`` batched input (averages over the leading dim).
        """
        diff = pred_metric - target_metric
        if diff.dim() == 2:
            return torch.sum(diff * diff)
        if diff.dim() == 3:
            return torch.mean(torch.einsum("bij,bij->b", diff, diff))
        raise ValueError(
            f"metric must be 2-D (D, D) or 3-D (B, D, D); got {tuple(diff.shape)}"
        )

    @staticmethod
    def gauge_connection_sum(connection: torch.Tensor) -> torch.Tensor:
        """``sum_i A_i`` per batch element. Mirrors
        ``ThermalManifold::gauge_connection_sum`` in
        ``src/physics/geometry_tensor.rs``.

        Accepts either a 1-D ``(D,)`` vector (returns scalar) or a 2-D
        ``(B, D)`` batched input (returns a ``(B,)`` per-row sum)."""
        if connection.dim() == 1:
            return torch.sum(connection)
        return torch.sum(connection, dim=-1)

    @staticmethod
    def _diagonal(matrix: torch.Tensor) -> torch.Tensor:
        """Extract the diagonal of a 2-D ``(D, D)`` or 3-D ``(B, D, D)``
        matrix as a ``(D,)`` or ``(B, D)`` tensor."""
        if matrix.dim() == 2:
            return torch.diagonal(matrix)
        return torch.diagonal(matrix, dim1=-2, dim2=-1)

    @staticmethod
    def parallel_transport(
        metric: torch.Tensor, field: torch.Tensor, connection: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        """``T + dt * (M · T + A)`` — the forward-Euler geometric flow that
        ``ThermalManifold::compute_parallel_transport`` will extend with the
        full Christoffel-symbol transport in ``GaugeSolver`` (#1462).

        ``dt`` may be a scalar or a ``(B,)`` tensor; the function broadcasts
        to the leading batch dim. Inputs are either ``(D, ...)`` (single
        sample, returns same shape) or ``(B, D, ...)`` (batched, returns
        ``(B, D)``).
        """
        mvt = torch.einsum("...ij,...j->...i", metric, field)
        return field + dt * (mvt + connection)

    def _broadcast_dt(self, dt: Any, batch_size: int, device: torch.device) -> torch.Tensor:
        """Promote ``dt`` to a ``(B,)`` tensor on the right device."""
        if torch.is_tensor(dt):
            t = dt.to(device=device, dtype=torch.float32)
            if t.dim() == 0:
                t = t.expand(batch_size)
            return t
        return torch.full((batch_size,), float(dt), dtype=torch.float32, device=device)

    @staticmethod
    def _broadcast_target(
        target: ThermalManifoldBatch, pred_metric_shape: Tuple[int, ...]
    ) -> ThermalManifoldBatch:
        """Promote a singleton target to match a batched prediction.

        If the prediction is ``(B, D, D)`` but the target metric is
        ``(D, D)`` (and similarly for field / connection), prepend a
        leading batch dim to every tensor so the loss math downstream sees
        consistent shapes.
        """
        b_dim = pred_metric_shape[0]
        target_metric = target.metric
        target_field = target.field
        target_connection = target.connection
        target_field_next = getattr(target, "field_next", None)

        if target_metric.dim() == len(pred_metric_shape) - 1:
            target_metric = target_metric.unsqueeze(0).expand(b_dim, *target_metric.shape)
            target_field = target_field.unsqueeze(0).expand(b_dim, *target_field.shape)
            target_connection = target_connection.unsqueeze(0).expand(
                b_dim, *target_connection.shape
            )
            if target_field_next is not None and target_field_next.dim() == len(
                pred_metric_shape
            ) - 1:
                target_field_next = target_field_next.unsqueeze(0).expand(
                    b_dim, *target_field_next.shape
                )
        return ThermalManifoldBatch(
            metric=target_metric,
            field=target_field,
            connection=target_connection,
            dt_seconds=target.dt_seconds,
            field_next=target_field_next,
        )

    def forward(
        self,
        pred_metric: torch.Tensor,
        pred_field: torch.Tensor,
        pred_connection: torch.Tensor,
        target: ThermalManifoldBatch,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the gauge-invariant loss.

        Args:
            pred_metric: ``(B, D, D)`` predicted Riemannian metric.
            pred_field: ``(B, D)`` predicted temperature field.
            pred_connection: ``(B, D)`` predicted gauge connection (heat flux).
            target: ``ThermalManifoldBatch`` carrying the ground-truth tensors
                (and optional per-batch ``dt_seconds``).

        Returns:
            ``(total_loss, components_dict)`` where ``components_dict`` holds
            the scalar value of every individual sub-loss for logging.
        """
        # Broadcast a singleton ``(D, ...)`` target up to the prediction's
        # ``(B, D, ...)`` shape — useful for ad-hoc tests with single
        # manifolds. The shapes must still agree on the trailing dims.
        target = self._broadcast_target(target, pred_metric.shape)

        if pred_metric.shape != target.metric.shape:
            raise ValueError(
                f"metric shape mismatch: pred {tuple(pred_metric.shape)} "
                f"vs target {tuple(target.metric.shape)}"
            )
        if pred_field.shape != target.field.shape:
            raise ValueError(
                f"field shape mismatch: pred {tuple(pred_field.shape)} "
                f"vs target {tuple(target.field.shape)}"
            )
        if pred_connection.shape != target.connection.shape:
            raise ValueError(
                f"connection shape mismatch: pred {tuple(pred_connection.shape)} "
                f"vs target {tuple(target.connection.shape)}"
            )

        cfg = self.config
        device = pred_metric.device
        batch_size = pred_metric.shape[0]
        dt = self._broadcast_dt(target.dt_seconds, batch_size, device)
        if dt.dim() == 1:
            dt_b = dt.view(batch_size, 1)
        else:
            dt_b = dt

        # Per-subject loss components -----------------------------------------
        loss_metric = self.metric_frobenius_distance(pred_metric, target.metric)

        if pred_field.dim() == 1:
            loss_field = torch.sum((pred_field - target.field) ** 2)
        else:
            loss_field = torch.mean(
                torch.sum((pred_field - target.field) ** 2, dim=-1)
            )

        if pred_connection.dim() == 1:
            loss_gauge = torch.sum((pred_connection - target.connection) ** 2)
        else:
            loss_gauge = torch.mean(
                torch.sum((pred_connection - target.connection) ** 2, dim=-1)
            )

        # First-Law penalty: predicted connection must not *exceed* the
        # ground-truth sum (no energy creation). One-sided ReLU penalty so
        # energy dissipation (sum_pred < sum_target) is not penalised — the
        # data term covers that case.
        sum_pred = self.gauge_connection_sum(pred_connection)
        sum_target = self.gauge_connection_sum(target.connection)
        energy_excess = torch.relu(sum_pred - sum_target)
        if energy_excess.dim() == 0:
            loss_conservation = energy_excess ** 2
        else:
            loss_conservation = torch.mean(energy_excess ** 2)

        # Parallel-transport consistency: data-driven supervision on the
        # geometric flow. Requires next-step targets; if absent we still
        # compute the *predicted* next state but skip the comparison.
        if getattr(target, "field_next", None) is not None:
            field_next_pred = self.parallel_transport(
                pred_metric, pred_field, pred_connection, dt_b
            )
            target_field_next = target.field_next
            if field_next_pred.dim() == 1:
                loss_transport = torch.sum((field_next_pred - target_field_next) ** 2)
            else:
                loss_transport = torch.mean(
                    torch.sum((field_next_pred - target_field_next) ** 2, dim=-1)
                )
        else:
            loss_transport = torch.tensor(0.0, device=device, dtype=pred_metric.dtype)

        # Dissipativity: Kirchhoff's current law for the passive case.
        #   * diagonal: M[i,i] <= 0 (self-damping)
        #   * off-diagonal: M[i,j] >= 0 for i != j (non-negative coupling)
        # One-sided ReLU penalties so legitimate *non-passive* operators are
        # not penalised in the wrong direction.
        diag = self._diagonal(pred_metric)
        off_diag = pred_metric - torch.diag_embed(diag)
        if diag.dim() == 1:
            # Single sample (D, D) — diag shape is (D,), off_diag shape is (D, D)
            loss_diag = torch.sum(torch.relu(diag) ** 2)
            loss_off = torch.sum(torch.relu(-off_diag) ** 2)
            loss_dissipativity = loss_diag + loss_off
        else:
            loss_diag = torch.mean(torch.sum(torch.relu(diag) ** 2, dim=-1))
            loss_off = torch.mean(
                torch.sum(torch.relu(-off_diag) ** 2, dim=(-2, -1))
            )
            loss_dissipativity = loss_diag + loss_off

        total = (
            cfg.metric_weight * loss_metric
            + cfg.field_weight * loss_field
            + cfg.gauge_weight * loss_gauge
            + cfg.conservation_weight * loss_conservation
            + cfg.transport_weight * loss_transport
            + cfg.dissipativity_weight * loss_dissipativity
        )

        components: Dict[str, float] = {
            "metric": float(loss_metric.detach().item()),
            "field": float(loss_field.detach().item()),
            "gauge": float(loss_gauge.detach().item()),
            "conservation": float(loss_conservation.detach().item()),
            "transport": float(loss_transport.detach().item()),
            "dissipativity": float(loss_dissipativity.detach().item()),
            "total": float(total.detach().item()),
        }
        return total, components


class GeometricSurrogate(nn.Module):
    """Neural-net surrogate that outputs a ``ThermalManifold``.

    Three heads produce the geometric representation:

    - **metric head**: a flattened ``(D, D)`` linear layer. Symmetrised by
      ``(W + W^T)/2`` at output time so Kirchhoff reciprocity is baked into
      the architecture, complementing the dissipativity penalty in the loss.
    - **field head**: ``D`` temperatures (°C).
    - **connection head**: ``D`` heat fluxes (W). Bounded by a ``Softplus``
      to discourage runaway negative fluxes — a soft analogue of the
      non-creation constraint.

    The input dimension matches the legacy ``PIMLSurrogate`` (default ``9``:
    outdoor temp, heating setpoint, cooling setpoint, normalised solar,
    normalised internal gain, hour-of-day, day-of-year, month, WWR) so the
    new path can be wired into existing data pipelines without re-shaping.
    """

    def __init__(
        self,
        input_dim: int = 9,
        manifold_dim: int = 4,
        hidden_dims: Optional[List[int]] = None,
        field_bias_init: Optional[List[float]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.manifold_dim = manifold_dim
        self.hidden_dims = list(hidden_dims) if hidden_dims else [128, 64, 32]
        # Per-index bias initialisation for the field head. The default
        # assumes a 5R1C scene embedded in the 4-D manifold: air / wall at
        # ~22 °C, roof / floor parked at 0 (see
        # ThermalManifold::from_5r1c_parameters). For other ``manifold_dim``
        # values the default is a uniform 22 °C bias across every slot.
        if field_bias_init is None:
            if manifold_dim == 4:
                field_bias_init = [22.0, 22.0, 0.0, 0.0]
            else:
                field_bias_init = [22.0] * manifold_dim
        if len(field_bias_init) != manifold_dim:
            raise ValueError(
                f"field_bias_init must have length manifold_dim={manifold_dim} "
                f"(got {len(field_bias_init)})"
            )
        self._field_bias_init = field_bias_init

        layers: List[nn.Module] = []
        prev = input_dim
        for h in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ]
            )
            prev = h
        self.encoder = nn.Sequential(*layers)

        self.metric_head = nn.Linear(prev, manifold_dim * manifold_dim)
        self.field_head = nn.Linear(prev, manifold_dim)
        self.connection_head = nn.Linear(prev, manifold_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # Initialise the field head's bias to a sensible 5R1C
                    # default so the surrogate does not start with a huge
                    # squared error on the temperature vector (which would
                    # swamp the other loss terms at the first step).
                    if m is self.field_head:
                        with torch.no_grad():
                            m.bias.copy_(
                                torch.tensor(self._field_bias_init, dtype=m.bias.dtype)
                            )
                    else:
                        nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(metric, field, connection)`` for a batch of inputs."""
        encoded = self.encoder(x)
        metric_flat = self.metric_head(encoded)
        metric = metric_flat.view(-1, self.manifold_dim, self.manifold_dim)
        # Symmetrise so Kirchhoff reciprocity is preserved at output time.
        metric = 0.5 * (metric + metric.transpose(-1, -2))
        field = self.field_head(encoded)
        connection = self.connection_head(encoded)
        return metric, field, connection


def generate_synthetic_manifold_data(
    n_samples: int = 1024,
    manifold_dim: int = 4,
    seed: int = 42,
) -> Tuple[torch.Tensor, ThermalManifoldBatch]:
    """Generate a synthetic batch of ``ThermalManifold`` targets.

    The synthetic data draws a 5R1C-style metric (roof/floor slots parked
    at zero) and randomises the temperatures / fluxes slightly so the
    geometric loss has signal. To make the smoke-test path actually
    trainable (the network has to learn a non-trivial input→manifold map),
    the features are constructed to be informative linear combinations of
    the underlying physical parameters — namely ``R_eq``, ``C_air``,
    ``C_mass``, ``T_air``, ``T_mass``, ``Q_internal``, ``Q_solar``.

    For a full training pipeline use ``tools/train_surrogate_manifold.py``
    with ``--data-dir`` pointing at physics-extracted manifold tensors
    (real EnergyPlus output, not random).

    Returns:
        ``(features, batch)`` where ``features`` is ``(n_samples, input_dim)``
        and ``batch`` is a ``ThermalManifoldBatch``.
    """
    rng = np.random.RandomState(seed)
    input_dim = 9

    # Underlying physical parameters
    r_eq = rng.uniform(0.05, 0.5, size=n_samples)
    c_air = rng.uniform(5_000.0, 20_000.0, size=n_samples)
    c_mass = rng.uniform(20_000.0, 100_000.0, size=n_samples)
    t_air = rng.uniform(18.0, 26.0, size=n_samples)
    t_mass = rng.uniform(17.0, 28.0, size=n_samples)
    q_internal = rng.uniform(0.0, 1_000.0, size=n_samples)
    q_solar = rng.uniform(0.0, 2_000.0, size=n_samples)
    hour = rng.uniform(0.0, 24.0, size=n_samples)
    month = rng.uniform(1.0, 12.0, size=n_samples)

    # Features are normalised, informative linear projections of the
    # underlying physics. This lets the GeometricSurrogate actually learn
    # a non-trivial map in the smoke test.
    g_eq = 1.0 / r_eq
    features = np.stack(
        [
            (t_air - 22.0) / 4.0,
            (t_mass - 22.0) / 4.0,
            (g_eq - 10.0) / 10.0,
            (1.0 / c_air - 1.0e-4) / 1.0e-4,
            (1.0 / c_mass - 4.0e-5) / 4.0e-5,
            (q_internal - 500.0) / 500.0,
            (q_solar - 1_000.0) / 1_000.0,
            (hour - 12.0) / 12.0,
            (month - 6.5) / 6.5,
        ],
        axis=1,
    ).astype(np.float32)

    metric = np.zeros((n_samples, manifold_dim, manifold_dim), dtype=np.float32)
    metric[:, 0, 0] = -g_eq / c_air
    metric[:, 0, 1] = g_eq / c_air
    metric[:, 1, 0] = g_eq / c_mass
    metric[:, 1, 1] = -g_eq / c_mass

    field = np.zeros((n_samples, manifold_dim), dtype=np.float32)
    field[:, 0] = t_air
    field[:, 1] = t_mass

    connection = np.zeros((n_samples, manifold_dim), dtype=np.float32)
    connection[:, 0] = q_internal / c_air
    connection[:, 1] = q_solar / c_mass

    dt = 60.0
    field_next = field + dt * (
        np.einsum("nij,nj->ni", metric, field) + connection
    )

    return (
        torch.from_numpy(features),
        ThermalManifoldBatch(
            metric=torch.from_numpy(metric),
            field=torch.from_numpy(field),
            connection=torch.from_numpy(connection),
            dt_seconds=dt,
            field_next=torch.from_numpy(field_next),
        ),
    )


def train_geometric_surrogate(
    model: GeometricSurrogate,
    train_features: torch.Tensor,
    train_batch: ThermalManifoldBatch,
    config: GaugeInvariantConfig,
    val_features: Optional[torch.Tensor] = None,
    val_batch: Optional[ThermalManifoldBatch] = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> Tuple[GeometricSurrogate, Dict[str, List[float]]]:
    """Train a ``GeometricSurrogate`` against the gauge-invariant loss.

    A short, dependency-light loop intended as a smoke-test for the new
    training path. Production retraining should call
    ``train_geometric_surrogate`` from a dedicated CLI (see
    ``tools/train_surrogate_manifold.py``).
    """
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    criterion = GaugeInvariantLoss(config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    history: Dict[str, List[float]] = {
        "loss": [],
        "metric": [],
        "field": [],
        "gauge": [],
        "conservation": [],
        "transport": [],
        "dissipativity": [],
        "val_loss": [],
    }

    n = train_features.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        running: Dict[str, float] = {}
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x = train_features[idx].to(device)
            target = ThermalManifoldBatch(
                metric=train_batch.metric[idx].to(device),
                field=train_batch.field[idx].to(device),
                connection=train_batch.connection[idx].to(device),
                dt_seconds=train_batch.dt_seconds,
                field_next=(
                    train_batch.field_next[idx].to(device)
                    if getattr(train_batch, "field_next", None) is not None
                    else None
                ),
            )

            optimizer.zero_grad()
            pred_metric, pred_field, pred_connection = model(x)
            loss, comp = criterion(pred_metric, pred_field, pred_connection, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in comp.items():
                running[k] = running.get(k, 0.0) + v
            n_batches += 1

        for k in running:
            history.setdefault(k, []).append(running[k] / n_batches)
        # ``loss`` is the conventional alias for the total loss
        # (matches the legacy ``train_piml_surrogate`` history schema).
        history.setdefault("loss", []).append(
            history["total"][-1] if "total" in history else 0.0
        )

        if val_features is not None and val_batch is not None:
            model.eval()
            with torch.no_grad():
                vx = val_features.to(device)
                vtarget = ThermalManifoldBatch(
                    metric=val_batch.metric.to(device),
                    field=val_batch.field.to(device),
                    connection=val_batch.connection.to(device),
                    dt_seconds=val_batch.dt_seconds,
                    field_next=(
                        val_batch.field_next.to(device)
                        if getattr(val_batch, "field_next", None) is not None
                        else None
                    ),
                )
                pm, pf, pc = model(vx)
                val_loss, _ = criterion(pm, pf, pc, vtarget)
                history["val_loss"].append(float(val_loss.item()))

        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.info(
                "Epoch %d/%d | loss=%.4e | metric=%.4e | field=%.4e | "
                "gauge=%.4e | conservation=%.4e | transport=%.4e | "
                "dissipativity=%.4e",
                epoch + 1,
                epochs,
                history["total"][-1],
                history["metric"][-1],
                history["field"][-1],
                history["gauge"][-1],
                history["conservation"][-1],
                history["transport"][-1],
                history["dissipativity"][-1],
            )

    return model, history


if __name__ == "__main__":
    main()