#!/usr/bin/env python3
"""
PIML (Physics-Informed Machine Learning) Loss Functions

Implements Physics-Informed ML loss functions for thermal surrogate training
that embed thermodynamic RC (Resistance-Capacitance) models into neural network
loss functions for surrogate training.

This implementation addresses Issue #552: PIML-01: Implement Physics-Informed Machine Learning loss functions

Key features:
- Simplified RC thermal network embedded in loss function
- Physics-informed penalty terms for thermodynamic violations
- Comparison between PIML loss and standard L1/L2 loss
- CvRMSE optimization for peak load accuracy

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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
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

    input_dim: int = 8
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
        q_solar = (
            features[:, 3] * 50
            if features.shape[1] > 3
            else torch.zeros_like(t_outdoor)
        )
        q_internal = (
            features[:, 4] * 100
            if features.shape[1] > 4
            else torch.zeros_like(t_outdoor)
        )

        t_indoor_pred = 20.0 + heating_pred * 0.1 - cooling_pred * 0.1

        rc_residual = self.compute_rc_residual(
            t_indoor_pred=t_indoor_pred,
            t_outdoor=t_outdoor,
            q_solar=q_solar,
            q_internal=q_internal,
            q_heating=heating_pred * 1000.0,
            q_cooling=cooling_pred * 1000.0,
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
        input_dim: int = 8,
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

    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-5
    )
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
                        (
                            batch_X[:, 6:7]
                            if batch_X.shape[1] > 6
                            else torch.zeros_like(batch_X[:, 0:1])
                        ),
                        (
                            batch_X[:, 1:2]
                            if batch_X.shape[1] > 1
                            else torch.zeros_like(batch_X[:, 0:1])
                        ),
                        (
                            batch_X[:, 2:3]
                            if batch_X.shape[1] > 2
                            else torch.zeros_like(batch_X[:, 0:1])
                        ),
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
                loss, _ = criterion(
                    (heating_pred, cooling_pred), (batch_y_h, batch_y_c)
                )

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
                            (
                                X_val_t[:, 6:7]
                                if X_val_t.shape[1] > 6
                                else torch.zeros_like(X_val_t[:, 0:1])
                            ),
                            (
                                X_val_t[:, 1:2]
                                if X_val_t.shape[1] > 1
                                else torch.zeros_like(X_val_t[:, 0:1])
                            ),
                            (
                                X_val_t[:, 2:3]
                                if X_val_t.shape[1] > 2
                                else torch.zeros_like(X_val_t[:, 0:1])
                            ),
                        ],
                        dim=1,
                    ).squeeze(-1)
                    val_heating, val_cooling = model(X_val_t, physics_params_val)
                else:
                    val_heating, val_cooling = model(X_val_t)

                val_loss = criterion(
                    (val_heating, val_cooling), (y_h_val_t, y_c_val_t)
                )[0].item()

                total_pred = val_heating + val_cooling
                total_target = y_h_val_t + y_c_val_t
                cvrmse = compute_cvrmse(total_pred, total_target)

                high_mass_mask = (
                    total_target > torch.quantile(total_target, 0.75)
                ).squeeze()
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

        q_solar = rng.uniform(0, 100) * np.maximum(
            0, np.sin(2 * np.pi * np.arange(n_timesteps) / n_timesteps)
        )
        q_solar = np.maximum(0, q_solar + rng.normal(0, 5, n_timesteps))

        q_internal = rng.uniform(50, 200) + rng.normal(0, 10, n_timesteps)
        q_internal = np.maximum(0, q_internal)

        u_value = rng.uniform(0.2, 0.5)

        heating_load = np.maximum(
            0,
            u_value * (heating_setpoint - t_outdoor) + rng.normal(0, 0.5, n_timesteps),
        )
        cooling_load = np.maximum(
            0,
            u_value * (t_outdoor - cooling_setpoint) + rng.normal(0, 0.5, n_timesteps),
        )
        cooling_load = np.maximum(0, cooling_load)

        hour = np.arange(n_timesteps) % 24
        day_of_year = rng.randint(0, 365, size=n_timesteps)
        month = (day_of_year // 30) + 1
        wwr = rng.uniform(0.2, 0.6)

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

    val_data = (
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_h_val).float(),
        torch.from_numpy(y_c_val).float(),
    )

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
        logger.info(
            f"Standard Final CvRMSE: {standard_cvrmse:.2f}%, Peak: {standard_peak_cvrmse:.2f}%"
        )

        results["standard"] = {
            "cvrmse": standard_cvrmse,
            "peak_cvrmse": standard_peak_cvrmse,
            "history": history_standard,
        }

        logger.info("\n[4/4] Comparison:")
        logger.info(
            f"  PIML CvRMSE: {piml_cvrmse:.2f}% vs Standard: {standard_cvrmse:.2f}%"
        )
        logger.info(
            f"  PIML Peak CvRMSE: {piml_peak_cvrmse:.2f}% vs Standard: {standard_peak_cvrmse:.2f}%"
        )

        if piml_peak_cvrmse < standard_peak_cvrmse:
            improvement = (
                (standard_peak_cvrmse - piml_peak_cvrmse) / standard_peak_cvrmse * 100
            )
            logger.info(
                f"  PIML shows {improvement:.1f}% improvement in peak load error!"
            )
    else:
        logger.info(
            "\n[3/4] Skipping standard loss comparison (--compare-loss not set)"
        )

    logger.info("\n[4/4] Saving results...")
    torch.save(model_piml.state_dict(), output_dir / "piml_surrogate.pt")

    metrics = {
        "config": vars(args),
        "results": {
            k: {kk: vv for kk, vv in v.items() if kk != "history"}
            for k, v in results.items()
        },
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "history.json", "w") as f:
        json.dump({k: v for k, v in results.items()}, f)

    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info("PIML training complete!")

    peak_error_met = piml_peak_cvrmse < 30.0
    if peak_error_met:
        logger.info(
            f"SUCCESS: Peak load error ({piml_peak_cvrmse:.2f}%) < 30% threshold"
        )
    else:
        logger.warning(
            f"WARNING: Peak load error ({piml_peak_cvrmse:.2f}%) >= 30% threshold"
        )

    return model_piml, results


if __name__ == "__main__":
    main()
