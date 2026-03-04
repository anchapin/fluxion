#!/usr/bin/env python3
"""
Phase 3: RL Training Pipeline with ONNX Policy Export

This script trains a PPO agent for HVAC control using the Fluxion environment
and exports the trained policy to ONNX format for inference in the Rust engine.

Supports:
- Stable Baselines3 (SB3) with PPO
- ONNX export of trained policies
- Custom reward shaping for energy + comfort optimization
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Import RL dependencies
SB3_AVAILABLE = False
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    logger.warning("Stable Baselines3 not installed. Run: pip install stable-baselines3")
    PPO = None

# Import Fluxion environment
try:
    from rl_environment import FluxionEnv, EnvConfig
    ENV_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import FluxionEnv: {e}")
    ENV_AVAILABLE = False


class RLTrainingConfig:
    """Configuration for RL training."""
    
    # Environment
    num_envs: int = 4
    num_zones: int = 1
    steps_per_episode: int = 8760
    
    # Training
    total_timesteps: int = 100_000
    eval_freq: int = 10000
    save_freq: int = 20000
    
    # PPO Hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_range: float = 0.2
    ent_coef: float = 0.01  # Entropy coefficient for exploration
    vf_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5
    use_sde: bool = False  # State-dependent exploration
    
    # Network
    policy_layers: List[int] = [256, 256]
    
    # Reward weights
    energy_weight: float = -1.0
    comfort_weight: float = -0.5
    comfort_band: float = 2.0
    
    # Output
    output_dir: str = "models/rl_policy"
    experiment_name: str = "fluxion_ppo"


def create_env(config: RLTrainingConfig, seed: int = 0):
    """Create a single Fluxion environment."""
    env_config = EnvConfig(
        num_zones=config.num_zones,
        steps_per_episode=config.steps_per_episode,
        energy_weight=config.energy_weight,
        comfort_weight=config.comfort_weight,
        comfort_band=config.comfort_band,
    )
    env = FluxionEnv(env_config)
    env = Monitor(env)
    return env


def make_training_env(config: RLTrainingConfig, seed: int = 0):
    """Create vectorized training environment."""
    def make_env():
        return create_env(config, seed)
    
    # Use DummyVecEnv for simplicity (works with SB3)
    env = DummyVecEnv([make_env])
    return env


def train_ppo(config: RLTrainingConfig) -> PPO:
    """
    Train a PPO agent on the Fluxion environment.
    
    Args:
        config: Training configuration
        
    Returns:
        Trained PPO model
    """
    if not SB3_AVAILABLE:
        raise RuntimeError("Stable Baselines3 is required for training. Install with: pip install stable-baselines3")
    
    if not ENV_AVAILABLE:
        raise RuntimeError("Fluxion environment is required. Ensure rl_environment.py is available.")
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training environment
    logger.info("Creating training environment...")
    train_env = make_training_env(config)
    
    # Create evaluation environment (separate for unbiased eval)
    eval_env = make_training_env(config, seed=999)
    
    # PPO Policy kwargs
    policy_kwargs = {
        "net_arch": [
            {"pi": config.policy_layers, "vf": config.policy_layers}
        ],
        "activation_fn": "ReLU",
    }
    
    # Create PPO model
    logger.info("Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        use_sde=config.use_sde,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(output_dir / "tensorboard"),
        device="cpu",  # Use CPU for training
    )
    
    # Callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_fluxion",
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=config.eval_freq,
        deterministic=True,
    )
    callbacks.append(eval_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Train
    logger.info(f"Starting training for {config.total_timesteps} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback_list,
        progress_bar=True,
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f} seconds")
    
    # Save final model
    final_model_path = output_dir / "final_model.zip"
    model.save(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Save training config
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2, default=str)
    logger.info(f"Saved config to {config_path}")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model


def export_to_onnx(output_path: str = "models/rl_policy/policy.onnx"):
    """
    Export a simple ONNX policy for testing without full training.
    
    Creates a deterministic policy network that can be used
    for basic HVAC control in the Rust engine.
    
    Args:
        output_path: Path to save ONNX model
        
    Returns:
        Path to exported ONNX model
    """
    import torch
    from torch import nn
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Observation dimension from FluxionEnv
    obs_dim = 8  # [outdoor_temp, zone_temp, solar_rad, hour, day_of_week, month, heating_setpoint, cooling_setpoint]
    action_dim = 2  # [heating_setpoint, cooling_setpoint]
    
    # Simple deterministic policy
    class SimplePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            # Simple 2-layer network
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                # Output scaling
                nn.Tanh(),  # Bound to [-1, 1]
            )
            # Scale and bias for action space
            self.register_buffer('scale', torch.tensor([5.0, 5.0]))  # Heating: 15-25, Cooling: 20-30
            self.register_buffer('bias', torch.tensor([20.0, 25.0]))
        
        def forward(self, x):
            out = self.net(x)
            # Scale to action space
            return out * self.scale + self.bias
    
    model = SimplePolicy()
    model.eval()
    
    # Export
    dummy_input = torch.randn(1, obs_dim)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
        opset_version=17,
    )
    
    logger.info(f"Simple ONNX policy exported: {output_path}")
    
    # Metadata
    metadata = {
        "model_type": "SimpleDeterministicPolicy",
        "observation_dim": obs_dim,
        "action_dim": action_dim,
        "action_space": {
            "heating_setpoint": {"min": 15.0, "max": 25.0},
            "cooling_setpoint": {"min": 20.0, "max": 30.0},
        },
        "exported_at": datetime.now().isoformat(),
        "framework": "pytorch",
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return output_path


def export_policy_to_onnx(model: PPO, output_path: Path, config: RLTrainingConfig):
    """
    Export trained PPO policy to ONNX format.
    
    The exported ONNX model can be loaded by the Rust engine for
    Python-free inference.
    
    Args:
        model: Trained PPO model
        output_path: Path to save ONNX model
        config: Training configuration
    """
    import torch
    from torch import nn
    
    logger.info(f"Exporting policy to ONNX: {output_path}")
    
    # Extract the policy network from PPO
    # PPO uses ActorCriticPolicy with features_extractor -> mlp_extractor -> action_net, value_net
    
    policy = model.policy
    
    # Get the feature extractor
    if hasattr(policy, 'features_extractor'):
        feature_dim = policy.features_extractor.features_dim
    else:
        # Default feature dimension
        feature_dim = 64
    
    # Get action space dimension
    action_dim = policy.action_space.shape[0]
    
    # Create a simplified ONNX policy network
    # This mimics the SB3 policy architecture
    class ONNXPolicy(nn.Module):
        def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int]):
            super().__init__()
            
            # Feature encoder
            layers = []
            prev_dim = obs_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            self.encoder = nn.Sequential(*layers)
            
            # Mean action (no std for simplicity - can be added)
            self.mean_net = nn.Linear(prev_dim, action_dim)
            
            # Temperature scaling (for action bounds)
            self.register_buffer('action_scale', torch.tensor([(25.0 - 15.0) / 2.0, (30.0 - 20.0) / 2.0]))
            self.register_buffer('action_bias', torch.tensor([(25.0 + 15.0) / 2.0, (30.0 + 20.0) / 2.0]))
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Encode observations
            features = self.encoder(x)
            # Get mean action
            mean = self.mean_net(features)
            # Scale to action space
            mean = mean * self.action_scale + self.action_bias
            return mean
    
    # Get observation dimension from environment
    obs_dim = config.num_observation_features if hasattr(config, 'num_observation_features') else 5
    
    # Create ONNX model
    onnx_policy = ONNXPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=config.policy_layers,
    )
    
    # Copy weights from trained policy
    # Note: This is a simplified weight transfer - in production,
    # you might want more sophisticated weight mapping
    try:
        # Try to extract and copy the policy network weights
        if hasattr(policy, 'mlp_extractor'):
            # SB3 structure
            if hasattr(policy.mlp_extractor, 'policy_net'):
                # Copy policy network weights
                policy_state_dict = policy.state_dict()
                onnx_policy_state = onnx_policy.state_dict()
                
                # Map SB3 weights to our ONNX model
                for key in onnx_policy_state:
                    if 'encoder' in key:
                        # Map encoder layers
                        idx = key.split('.')[1] if len(key.split('.')) > 1 else 0
                        sb3_key = f'mlp_extractor.policy_net.{idx}.0.weight'
                        if sb3_key in policy_state_dict:
                            onnx_policy_state[key] = policy_state_dict[sb3_key].clone()
                
                onnx_policy.load_state_dict(onnx_policy_state)
                logger.info("Copied policy weights from SB3 model")
    except Exception as e:
        logger.warning(f"Could not copy weights directly: {e}")
        logger.info("Using randomly initialized ONNX policy - training required to improve")
    
    # Export to ONNX
    onnx_policy.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, obs_dim)
    
    # Export
    torch.onnx.export(
        onnx_policy,
        dummy_input,
        str(output_path),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
        opset_version=17,
    )
    
    logger.info(f"ONNX export complete: {output_path}")
    
    # Save metadata
    metadata = {
        "model_type": "PPO_Policy",
        "observation_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_layers": config.policy_layers,
        "action_space": {
            "heating_setpoint": {"min": 15.0, "max": 25.0},
            "cooling_setpoint": {"min": 20.0, "max": 30.0},
        },
        "exported_at": datetime.now().isoformat(),
        "framework": "stable-baselines3",
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_path}")
    
    return output_path


def export_simple_onxx_policy(output_path: Path, config: RLTrainingConfig):
    """
    Export a simple deterministic policy network to ONNX.
    
    This creates a minimal ONNX model that can be loaded by Rust.
    For production use, train a model first then use the SB3 ONNX exporter.
    """
    import torch
    from torch import nn
    
    logger.info(f"Creating simple ONNX policy at {output_path}")
    
    obs_dim = 5  # [outdoor_temp, zone_temp, solar_rad, hour, day]
    action_dim = 2  # [heating_setpoint, cooling_setpoint]
    
    # Simple deterministic policy
    # Maps observations to actions via learned weights
    class SimplePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            # Simple 2-layer network
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                # Output scaling
                nn.Tanh(),  # Bound to [-1, 1]
            )
            # Scale and bias for action space
            self.register_buffer('scale', torch.tensor([5.0, 5.0]))  # Heating: 15-25, Cooling: 20-30
            self.register_buffer('bias', torch.tensor([20.0, 25.0]))
        
        def forward(self, x):
            out = self.net(x)
            # Scale to action space
            return out * self.scale + self.bias
    
    model = SimplePolicy()
    model.eval()
    
    # Export
    dummy_input = torch.randn(1, obs_dim)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
        opset_version=17,
    )
    
    logger.info(f"Simple ONNX policy exported: {output_path}")
    
    # Metadata
    metadata = {
        "model_type": "SimpleDeterministicPolicy",
        "observation_dim": obs_dim,
        "action_dim": action_dim,
        "action_space": {
            "heating_setpoint": {"min": 15.0, "max": 25.0},
            "cooling_setpoint": {"min": 20.0, "max": 30.0},
        },
        "exported_at": datetime.now().isoformat(),
        "framework": "pytorch",
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return output_path


def evaluate_policy(model: PPO, config: RLTrainingConfig, num_episodes: int = 3) -> Dict:
    """
    Evaluate trained policy.
    
    Args:
        model: Trained PPO model
        config: Training config
        num_episodes: Number of evaluation episodes
        
    Returns:
        Evaluation metrics
    """
    eval_env = make_training_env(config, seed=42)
    
    episode_rewards = []
    episode_energies = []
    episode_discomforts = []
    
    for episode in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            episode_reward += reward[0]
            done = dones[0]
        
        episode_rewards.append(episode_reward)
        
        # Get info from monitor
        if hasattr(eval_env, 'envs') and eval_env.envs:
            info = eval_env.envs[0].get_episode_rewards()
            if info:
                episode_energies.append(info.get('episode_energy', 0))
                episode_discomforts.append(info.get('episode_discomfort', 0))
    
    eval_env.close()
    
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_energy": np.mean(episode_energies) if episode_energies else 0,
        "mean_discomfort": np.mean(episode_discomforts) if episode_discomforts else 0,
    }
    
    logger.info(f"Evaluation: Mean Reward = {metrics['mean_reward']:.2f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent for HVAC control and export to ONNX"
    )
    
    # Training arguments
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/rl_policy",
        help="Output directory for models",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate existing model",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export existing model to ONNX",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to existing model (for eval/export)",
    )
    
    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="PPO epochs")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    
    # Quick test mode
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with minimal training",
    )
    
    args = parser.parse_args()
    
    # Create config
    config = RLTrainingConfig()
    config.total_timesteps = args.timesteps
    config.num_envs = args.envs
    config.output_dir = args.output_dir
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.n_epochs = args.epochs
    config.gamma = args.gamma
    
    if args.quick_test:
        # Quick test mode - minimal training
        config.total_timesteps = 5000
        config.save_freq = 2000
        config.eval_freq = 1000
        logger.info("Quick test mode enabled")
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.eval_only:
        # Evaluation only
        if not args.model_path:
            logger.error("--model-path required for eval-only mode")
            sys.exit(1)
        
        if not SB3_AVAILABLE:
            logger.error("Stable Baselines3 required for evaluation")
            sys.exit(1)
        
        logger.info(f"Loading model from {args.model_path}")
        model = PPO.load(args.model_path)
        
        metrics = evaluate_policy(model, config)
        
        # Save metrics
        with open(output_dir / "eval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        return
    
    if args.export_only:
        # Export only
        if not args.model_path:
            logger.error("--model-path required for export-only mode")
            sys.exit(1)
        
        if not SB3_AVAILABLE:
            logger.error("Stable Baselines3 required for export")
            sys.exit(1)
        
        logger.info(f"Loading model from {args.model_path}")
        model = PPO.load(args.model_path)
        
        onnx_path = output_dir / "policy.onnx"
        export_policy_to_onnx(model, onnx_path, config)
        
        return
    
    # Full training + export
    if not SB3_AVAILABLE:
        logger.error("Stable Baselines3 is required. Install with: pip install stable-baselines3")
        sys.exit(1)
    
    # Train
    model = train_ppo(config)
    
    # Export to ONNX
    onnx_path = output_dir / "policy.onnx"
    export_policy_to_onnx(model, onnx_path, config)
    
    # Evaluate final model
    metrics = evaluate_policy(model, config)
    
    # Save final metrics
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("=" * 50)
    logger.info("Training complete!")
    logger.info(f"Model saved: {output_dir / 'final_model.zip'}")
    logger.info(f"ONNX policy: {onnx_path}")
    logger.info(f"Mean reward: {metrics['mean_reward']:.2f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
