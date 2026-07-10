#!/usr/bin/env python3
"""
Tests for the Phase-2a ThermalManifold gauge-invariant loss (issue #1463).

These tests cover the new metric-aware loss path. The legacy RC-based
``PIMLLoss`` tests remain in ``test_piml_loss.py`` and are intentionally
untouched.

Numerical correctness is verified against the formulas encoded in
``src/physics/geometry_tensor.rs`` (issue #1461 — ``ThermalManifold``):

  - parallel_transport: T_new = T + dt * (M · T + A)
  - gauge_connection_sum: Σ A_i

Run with::

    pytest tools/tests/test_thermal_manifold_loss.py -v

or, from the project root::

    /home/alex/.venv/bin/python3 -m pytest tools/tests/test_thermal_manifold_loss.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piml_loss import (  # noqa: E402
    GaugeInvariantConfig,
    GaugeInvariantLoss,
    GeometricSurrogate,
    ThermalManifoldBatch,
    generate_synthetic_manifold_data,
    train_geometric_surrogate,
)


# ============================================================================
# Fixtures / helpers
# ============================================================================


@pytest.fixture
def reference_5r1c_scene() -> ThermalManifoldBatch:
    """5R1C scene embedded into the 4-D manifold (matches
    ``ThermalManifold::from_5r1c_parameters`` in
    ``src/physics/geometry_tensor.rs``).

    R_eq = 0.10 K/W, C_air = 10 000 J/K, C_mass = 50 000 J/K,
    T_air = 20 °C, T_mass = 21 °C, Q_internal = 200 W, Q_solar = 800 W.
    """
    r_eq = 0.10
    c_air = 10_000.0
    c_mass = 50_000.0
    g_eq = 1.0 / r_eq

    metric = np.zeros((4, 4), dtype=np.float32)
    metric[0, 0] = -g_eq / c_air
    metric[0, 1] = g_eq / c_air
    metric[1, 0] = g_eq / c_mass
    metric[1, 1] = -g_eq / c_mass

    field = np.array([20.0, 21.0, 0.0, 0.0], dtype=np.float32)
    connection = np.array(
        [200.0 / c_air, 800.0 / c_mass, 0.0, 0.0], dtype=np.float32
    )

    return ThermalManifoldBatch(
        metric=torch.from_numpy(metric),
        field=torch.from_numpy(field),
        connection=torch.from_numpy(connection),
        dt_seconds=60.0,
    )


# ============================================================================
# Tests for the loss math itself
# ============================================================================


class TestGaugeInvariantLossMath:
    """Verify the loss math matches the Rust ``ThermalManifold`` semantics."""

    def test_metric_frobenius_distance_zero_for_perfect_match(
        self, reference_5r1c_scene: ThermalManifoldBatch
    ):
        """``||M_pred - M_target||_F^2`` must be exactly 0 when pred == target."""
        loss_value = GaugeInvariantLoss.metric_frobenius_distance(
            reference_5r1c_scene.metric, reference_5r1c_scene.metric
        )
        assert loss_value.item() == 0.0

    def test_metric_frobenius_distance_symmetric(self):
        """``||A - B||_F^2 == ||B - A||_F^2``."""
        rng = np.random.RandomState(0)
        a = torch.from_numpy(rng.rand(4, 4).astype(np.float32))
        b = torch.from_numpy(rng.rand(4, 4).astype(np.float32))
        ab = GaugeInvariantLoss.metric_frobenius_distance(a, b)
        ba = GaugeInvariantLoss.metric_frobenius_distance(b, a)
        assert torch.allclose(ab, ba)

    def test_metric_frobenius_distance_matches_reference(self):
        """The Frobenius loss on a known perturbation matches a hand-computed
        reference value (computed via ctx_execute, see PR description)."""
        m = torch.zeros(4, 4)
        m[0, 0] = -1.0
        m[0, 1] = 2.0
        m[1, 0] = 3.0
        m[1, 1] = -4.0
        m_perturbed = m.clone()
        m_perturbed[0, 0] += 1e-3
        m_perturbed[2, 2] += 5e-4
        # Expected = sum of squared perturbations (allow 1e-7 for fp rounding)
        expected = (1e-3) ** 2 + (5e-4) ** 2
        got = GaugeInvariantLoss.metric_frobenius_distance(m_perturbed, m)
        assert abs(got.item() - expected) < 1e-7

    def test_gauge_connection_sum_matches_rust(
        self, reference_5r1c_scene: ThermalManifoldBatch
    ):
        """``gauge_connection_sum`` must equal the manual sum across the 4
        components — same formula as the Rust ``gauge_connection_sum``
        diagnostic in ``src/physics/geometry_tensor.rs``."""
        sums = GaugeInvariantLoss.gauge_connection_sum(
            reference_5r1c_scene.connection
        )
        expected = float(reference_5r1c_scene.connection.sum().item())
        assert sums.item() == pytest.approx(expected, rel=1e-12)

    def test_parallel_transport_matches_rust_stub(
        self, reference_5r1c_scene: ThermalManifoldBatch
    ):
        """``T + dt * (M·T + A)`` must match the
        ``ThermalManifold::compute_parallel_transport`` forward-Euler stub
        in Rust."""
        m = reference_5r1c_scene.metric
        t = reference_5r1c_scene.field
        a = reference_5r1c_scene.connection
        dt = reference_5r1c_scene.dt_seconds

        # Reference numpy calculation
        mvt = torch.einsum("ij,j->i", m, t)
        expected = t + float(dt) * (mvt + a)

        got = GaugeInvariantLoss.parallel_transport(
            m.unsqueeze(0), t.unsqueeze(0), a.unsqueeze(0), torch.tensor(dt)
        )[0]
        assert torch.allclose(got, expected, atol=1e-12)

    def test_parallel_transport_broadcasts_scalar_dt(self):
        """Scalar ``dt`` must broadcast to a (B,) tensor internally — the
        parallel-transport output must be identical to a per-sample ``dt``."""
        m = torch.eye(4).unsqueeze(0)
        t = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        a = torch.zeros(1, 4)
        dt = 60.0
        out_scalar = GaugeInvariantLoss.parallel_transport(m, t, a, dt)
        out_tensor = GaugeInvariantLoss.parallel_transport(
            m, t, a, torch.tensor(dt)
        )
        assert torch.allclose(out_scalar, out_tensor, atol=1e-12)


class TestGaugeInvariantLossForward:
    """End-to-end loss verification."""

    def test_perfect_prediction_yields_zero_loss(
        self, reference_5r1c_scene: ThermalManifoldBatch
    ):
        """When the prediction exactly matches the target, the total loss
        (and every sub-component except the dissipativity penalty) must be
        0. The dissipativity penalty is 0 because the reference 5R1C metric
        is symmetric."""
        cfg = GaugeInvariantConfig()
        loss_module = GaugeInvariantLoss(cfg)
        target = reference_5r1c_scene
        # Add a deterministic next field so transport loss is exercised.
        target.field_next = target.field + target.dt_seconds * (
            torch.einsum("ij,j->i", target.metric, target.field) + target.connection
        )

        loss, comp = loss_module(
            target.metric.unsqueeze(0),
            target.field.unsqueeze(0),
            target.connection.unsqueeze(0),
            target,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-10)
        for k, v in comp.items():
            if k == "total":
                continue
            assert v == pytest.approx(0.0, abs=1e-10), f"sub-loss {k} should be 0"

    def test_first_law_penalty_zero_for_perfect_match(
        self, reference_5r1c_scene: ThermalManifoldBatch
    ):
        """``L_conservation = 0`` when sum(A_pred) == sum(A_target)."""
        loss = GaugeInvariantLoss()
        # Sum_pred == Sum_target ⇒ no penalty (the relu(sum_pred - sum_target)
        # branch returns 0).
        sum_target = reference_5r1c_scene.connection.sum()
        same_connection = reference_5r1c_scene.connection.clone()
        # Negative perturbation so sum_pred <= sum_target ⇒ penalty is 0
        lower_connection = same_connection - sum_target
        # Actually we want sum_pred == sum_target — both connections identical.
        _, comp = loss(
            reference_5r1c_scene.metric.unsqueeze(0),
            reference_5r1c_scene.field.unsqueeze(0),
            same_connection.unsqueeze(0),
            reference_5r1c_scene,
        )
        assert comp["conservation"] == pytest.approx(0.0, abs=1e-12)

    def test_first_law_penalty_positive_for_hallucinated_energy(self):
        """``L_conservation > 0`` when sum(A_pred) > sum(A_target)."""
        loss = GaugeInvariantLoss()
        # Build a target with sum(A) = 0 (passive, isolated zone)
        metric = torch.eye(4).unsqueeze(0)
        field = torch.zeros(1, 4)
        connection_target = torch.zeros(1, 4)  # sum = 0
        target = ThermalManifoldBatch(
            metric=metric,
            field=field,
            connection=connection_target,
            dt_seconds=60.0,
        )
        # Build a prediction with sum(A) = 1.0 (hallucinated energy)
        connection_pred = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        _, comp = loss(metric, field, connection_pred, target)
        # relu(1 - 0)^2 = 1
        assert comp["conservation"] == pytest.approx(1.0, abs=1e-12)

    def test_first_law_penalty_ignores_dissipation(self):
        """``L_conservation = 0`` when sum(A_pred) < sum(A_target) — the
        penalty is one-sided so legitimate energy dissipation is not
        penalised."""
        loss = GaugeInvariantLoss()
        metric = torch.eye(4).unsqueeze(0)
        field = torch.zeros(1, 4)
        target = ThermalManifoldBatch(
            metric=metric,
            field=field,
            connection=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            dt_seconds=60.0,
        )
        # Prediction has sum=0, target has sum=1 ⇒ dissipated energy, OK.
        connection_pred = torch.zeros(1, 4)
        _, comp = loss(metric, field, connection_pred, target)
        assert comp["conservation"] == pytest.approx(0.0, abs=1e-12)

    def test_dissipativity_penalty_nonzero_for_non_passive_metric(self):
        """``L_dissipativity > 0`` when the metric violates Kirchhoff's
        current law (positive diagonal = active source, or negative
        off-diagonal = active coupling)."""
        loss = GaugeInvariantLoss()
        # Both diagonal entries are positive (should be <= 0 for passive),
        # both off-diagonal entries are negative (should be >= 0 for passive).
        metric_bad = torch.tensor(
            [
                [
                    [1.0, -2.0, 0.0, 0.0],
                    [-3.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ]
        )
        field = torch.zeros(1, 4)
        connection = torch.zeros(1, 4)
        target = ThermalManifoldBatch(
            metric=metric_bad,
            field=field,
            connection=connection,
            dt_seconds=60.0,
        )
        # Mirror into prediction so other losses are zero — isolate the
        # dissipativity penalty.
        _, comp = loss(metric_bad, field, connection, target)
        # diag = [1, 4, 0, 0] -> ReLU = [1, 4, 0, 0] -> sum sq = 1 + 16 = 17
        # off-diag (excluding diagonal): [−2, −3, …] -> ReLU(-off) = [2, 3, …] -> 4 + 9 = 13
        # Total = 17 + 13 = 30
        assert comp["dissipativity"] == pytest.approx(30.0, abs=1e-9)

    def test_dissipativity_penalty_zero_for_passive_metric(
        self, reference_5r1c_scene: ThermalManifoldBatch
    ):
        """``L_dissipativity = 0`` for the 5R1C reference metric — its
        diagonal is non-positive and its cross-coupling terms are
        non-negative, matching Kirchhoff's current law for a passive
        RC network."""
        loss = GaugeInvariantLoss()
        target = reference_5r1c_scene
        _, comp = loss(
            target.metric.unsqueeze(0),
            target.field.unsqueeze(0),
            target.connection.unsqueeze(0),
            target,
        )
        assert comp["dissipativity"] == pytest.approx(0.0, abs=1e-12)

    def test_transport_loss_matches_manual_euler(
        self, reference_5r1c_scene: ThermalManifoldBatch
    ):
        """``L_transport`` equals the manual forward-Euler comparison."""
        target = reference_5r1c_scene
        target.field_next = target.field + target.dt_seconds * (
            torch.einsum("ij,j->i", target.metric, target.field) + target.connection
        )
        loss = GaugeInvariantLoss()

        # Slightly perturbed prediction — compute expected transport loss
        # by hand.
        pm = target.metric.unsqueeze(0) + 1e-3 * torch.randn(1, 4, 4)
        pf = target.field.unsqueeze(0) + 0.1 * torch.randn(1, 4)
        pc = target.connection.unsqueeze(0) + 0.01 * torch.randn(1, 4)
        mvt = torch.einsum("bij,bj->bi", pm, pf)
        t_pred_next = pf + target.dt_seconds * (mvt + pc)
        expected_transport = torch.mean(
            torch.sum((t_pred_next - target.field_next.unsqueeze(0)) ** 2, dim=-1)
        )
        _, comp = loss(pm, pf, pc, target)
        assert comp["transport"] == pytest.approx(
            float(expected_transport.item()), rel=1e-6
        )

    def test_loss_is_differentiable_end_to_end(
        self, reference_5r1c_scene: ThermalManifoldBatch
    ):
        """Gradients must flow back to all three prediction tensors."""
        target = reference_5r1c_scene
        target.field_next = target.field + target.dt_seconds * (
            torch.einsum("ij,j->i", target.metric, target.field) + target.connection
        )
        loss = GaugeInvariantLoss()
        pm = (target.metric + 1e-3 * torch.randn(4, 4)).clone().requires_grad_(True)
        pf = (target.field + 0.1 * torch.randn(4)).clone().requires_grad_(True)
        pc = (target.connection + 0.01 * torch.randn(4)).clone().requires_grad_(True)
        total, _ = loss(
            pm.unsqueeze(0), pf.unsqueeze(0), pc.unsqueeze(0), target
        )
        total.backward()
        assert pm.grad is not None and pm.grad.abs().sum().item() > 0
        assert pf.grad is not None and pf.grad.abs().sum().item() > 0
        assert pc.grad is not None and pc.grad.abs().sum().item() > 0

    def test_shape_mismatch_raises(self):
        """A mis-shaped prediction must raise ``ValueError``, not silently
        pass through."""
        loss = GaugeInvariantLoss()
        target = ThermalManifoldBatch(
            metric=torch.zeros(2, 4, 4),
            field=torch.zeros(2, 4),
            connection=torch.zeros(2, 4),
            dt_seconds=60.0,
        )
        with pytest.raises(ValueError, match="metric shape mismatch"):
            loss(
                torch.zeros(2, 4, 3),  # wrong shape
                target.field,
                target.connection,
                target,
            )


# ============================================================================
# GeometricSurrogate (model) tests
# ============================================================================


class TestGeometricSurrogate:
    """Tests for the new ``GeometricSurrogate`` model."""

    def test_default_init_output_shapes(self):
        """Output shapes must match ``ThermalManifold``: (B, D, D), (B, D), (B, D)."""
        torch.manual_seed(0)
        model = GeometricSurrogate(input_dim=9, manifold_dim=4)
        x = torch.randn(8, 9)
        metric, field, connection = model(x)
        assert metric.shape == (8, 4, 4)
        assert field.shape == (8, 4)
        assert connection.shape == (8, 4)

    def test_metric_is_symmetric(self):
        """The metric head must be symmetrised at output time — Kirchhoff
        reciprocity is preserved by construction."""
        torch.manual_seed(0)
        model = GeometricSurrogate(input_dim=9, manifold_dim=4)
        x = torch.randn(5, 9)
        metric, _, _ = model(x)
        asym = metric - metric.transpose(-1, -2)
        assert torch.allclose(asym, torch.zeros_like(asym), atol=1e-12)

    def test_custom_manifold_dim(self):
        """``manifold_dim`` must propagate to all output heads."""
        model = GeometricSurrogate(input_dim=4, manifold_dim=6)
        x = torch.randn(3, 4)
        metric, field, connection = model(x)
        assert metric.shape == (3, 6, 6)
        assert field.shape == (3, 6)
        assert connection.shape == (3, 6)

    def test_field_bias_init_aligns_with_5r1c(self):
        """The default ``field_bias_init`` ([22, 22, 0, 0]) puts the initial
        prediction near a 5R1C scene — verifies the bias init override
        lands correctly on the right linear layer."""
        model = GeometricSurrogate(input_dim=9, manifold_dim=4)
        # Initial prediction (no info in input — ReLU(BN(0)) = 0 at init? No,
        # BN has running stats, but eval mode isn't toggled here). Check the
        # bias directly via the field_head parameter.
        bias = model.field_head.bias.detach()
        assert torch.allclose(
            bias, torch.tensor([22.0, 22.0, 0.0, 0.0]), atol=1e-6
        )

    def test_invalid_field_bias_init_length(self):
        """A ``field_bias_init`` of the wrong length must raise."""
        with pytest.raises(ValueError, match="must have length"):
            GeometricSurrogate(
                input_dim=9, manifold_dim=4, field_bias_init=[20.0, 21.0]
            )


# ============================================================================
# Synthetic data generator
# ============================================================================


class TestSyntheticManifoldData:
    """Sanity-check the synthetic data generator."""

    def test_synthetic_batch_shapes(self):
        features, batch = generate_synthetic_manifold_data(n_samples=32, seed=0)
        assert features.shape == (32, 9)
        assert batch.metric.shape == (32, 4, 4)
        assert batch.field.shape == (32, 4)
        assert batch.connection.shape == (32, 4)
        assert batch.field_next is not None
        assert batch.field_next.shape == (32, 4)

    def test_synthetic_metric_is_5r1c_compatible(self):
        """The synthetic metric must match the 5R1C embedding — roof/floor
        slots parked at zero."""
        _, batch = generate_synthetic_manifold_data(n_samples=16, seed=0)
        # All roof/floor rows and columns are zero
        assert torch.all(batch.metric[:, 2, :] == 0)
        assert torch.all(batch.metric[:, 3, :] == 0)
        assert torch.all(batch.metric[:, :, 2] == 0)
        assert torch.all(batch.metric[:, :, 3] == 0)
        # Air self-conductance is negative (dissipative)
        assert (batch.metric[:, 0, 0] <= 0).all()
        # Air-mass coupling is positive
        assert (batch.metric[:, 0, 1] > 0).all()
        assert (batch.metric[:, 1, 0] > 0).all()
        # Mass self-conductance is negative
        assert (batch.metric[:, 1, 1] <= 0).all()

    def test_synthetic_field_next_matches_parallel_transport(self):
        """``field_next = field + dt * (M·T + A)`` — verified for the synthetic
        generator's own data."""
        _, batch = generate_synthetic_manifold_data(n_samples=8, seed=0)
        mvt = torch.einsum("bij,bj->bi", batch.metric, batch.field)
        expected = batch.field + batch.dt_seconds * (mvt + batch.connection)
        assert torch.allclose(batch.field_next, expected, atol=1e-6)


# ============================================================================
# Training loop smoke test
# ============================================================================


class TestTrainGeometricSurrogate:
    """Smoke test for the training loop."""

    def test_loss_decreases_over_a_few_epochs(self):
        """A few training epochs must strictly decrease the loss on the
        synthetic data (which is informative by construction)."""
        torch.manual_seed(0)
        features, batch = generate_synthetic_manifold_data(n_samples=128, seed=0)
        model = GeometricSurrogate(input_dim=9, manifold_dim=4)
        cfg = GaugeInvariantConfig()
        _, history = train_geometric_surrogate(
            model=model,
            train_features=features,
            train_batch=batch,
            config=cfg,
            epochs=5,
            batch_size=32,
            learning_rate=1e-3,
        )
        # Loss is recorded under the key 'total' inside the loss components
        total_losses = history.get("total", [])
        assert len(total_losses) >= 2
        # First-epoch loss must exceed the final loss by a wide margin.
        assert total_losses[0] > total_losses[-1]
        assert total_losses[-1] < total_losses[0] * 0.95

    def test_training_history_keys(self):
        """The history dict must carry every documented sub-loss key."""
        features, batch = generate_synthetic_manifold_data(n_samples=64, seed=0)
        model = GeometricSurrogate(input_dim=9, manifold_dim=4)
        _, history = train_geometric_surrogate(
            model=model,
            train_features=features,
            train_batch=batch,
            config=GaugeInvariantConfig(),
            epochs=2,
            batch_size=32,
        )
        for key in ("loss", "metric", "field", "gauge", "conservation", "transport", "dissipativity"):
            assert key in history
            assert len(history[key]) == 2


# ============================================================================
# Backward-compatibility smoke test
# ============================================================================


class TestBackwardCompatibility:
    """The legacy ``PIMLLoss`` / ``PIMLSurrogate`` API must continue to work
    — Phase 2a introduces a *parallel* path, not a replacement."""

    def test_piml_loss_still_constructable(self):
        from piml_loss import PIMLConfig, PIMLLoss

        cfg = PIMLConfig()
        loss = PIMLLoss(cfg)
        # Predict and target shape: (B, output_dim)
        heating_pred = torch.tensor([[1.0], [2.0]])
        cooling_pred = torch.tensor([[3.0], [4.0]])
        heating_target = torch.tensor([[1.5], [2.5]])
        cooling_target = torch.tensor([[3.5], [4.5]])
        features = torch.randn(2, 9)
        physics_params = torch.randn(2, 3)
        loss_val, comp = loss(
            (heating_pred, cooling_pred),
            (heating_target, cooling_target),
            features,
            physics_params,
        )
        assert loss_val.item() > 0
        assert "mse" in comp and "piml" in comp

    def test_standard_loss_still_constructable(self):
        from piml_loss import StandardLoss

        loss = StandardLoss()
        heating_pred = torch.zeros(4, 1)
        cooling_pred = torch.zeros(4, 1)
        heating_target = torch.ones(4, 1)
        cooling_target = torch.ones(4, 1)
        loss_val, comp = loss(
            (heating_pred, cooling_pred),
            (heating_target, cooling_target),
        )
        assert loss_val.item() > 0
        assert "mse" in comp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])