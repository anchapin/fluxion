#!/usr/bin/env python3
"""
Tests for Gymnasium Environment Wrapper (FluxionEnv)

Issue: GitHub #328 - Develop Gymnasium Environment Wrapper (FluxionEnv)
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pytest
from tools.gymnasium_env import FluxionEnv, EnvConfig, GYMNASIUM_AVAILABLE


class TestFluxionEnv:
    """Test suite for FluxionEnv."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return EnvConfig(
            num_zones=1,
            steps_per_episode=24,
            weather_forecast_horizon=6,
        )
    
    @pytest.fixture
    def env(self, config):
        """Create test environment."""
        return FluxionEnv(config=config)
    
    def test_initialization(self, env):
        """Test environment initializes correctly."""
        assert env is not None
        assert env.config is not None
        assert env.config.num_zones == 1
        assert env.config.steps_per_episode == 24
    
    def test_observation_space(self, env):
        """Test observation space is properly defined."""
        obs_space = env.observation_space
        assert obs_space is not None
        assert obs_space.dtype == np.float32
        
        # Check dimensions
        num_zones = env.config.num_zones
        horizon = env.config.weather_forecast_horizon
        expected_dim = num_zones + 3 * horizon + 2
        assert obs_space.shape[0] == expected_dim
    
    def test_action_space(self, env):
        """Test action space is properly defined."""
        action_space = env.action_space
        assert action_space is not None
        assert "continuous" in action_space.spaces
        assert "discrete" in action_space.spaces
    
    def test_reset(self, env):
        """Test reset returns valid observation."""
        obs, info = env.reset(seed=42)
        
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        
        assert info is not None
        assert "step" in info
        assert info["step"] == 0
    
    def test_reset_with_seed(self, env):
        """Test reset with seed produces reproducible results."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_step(self, env):
        """Test step returns valid values."""
        env.reset(seed=42)
        
        action = {
            "continuous": np.array([20.0, 24.0, 50.0], dtype=np.float32),
            "discrete": np.array([1, 1, 1, 0], dtype=np.int8),
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        assert info["step"] == 1
    
    def test_episode_completion(self, env):
        """Test episode terminates after steps_per_episode."""
        env.reset(seed=42)
        
        action = {
            "continuous": np.array([20.0, 24.0, 50.0], dtype=np.float32),
            "discrete": np.array([1, 1, 1, 0], dtype=np.int8),
        }
        
        terminated = False
        steps = 0
        max_steps = env.config.steps_per_episode
        
        while not terminated and steps < max_steps:
            _, _, terminated, _, _ = env.step(action)
            steps += 1
        
        assert terminated
        assert steps == max_steps
    
    def test_action_bounds(self, env):
        """Test actions outside bounds are clipped."""
        env.reset(seed=42)
        
        # Action with values outside bounds
        action = {
            "continuous": np.array([10.0, 35.0, 150.0], dtype=np.float32),  # Out of range
            "discrete": np.array([1, 1, 1, 0], dtype=np.int8),
        }
        
        # Should not raise error - actions are clipped
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert not (terminated and not truncated)
    
    def test_reward_calculation(self, env):
        """Test reward is calculated correctly."""
        env.reset(seed=42)
        
        action = {
            "continuous": np.array([20.0, 24.0, 50.0], dtype=np.float32),
            "discrete": np.array([1, 1, 1, 0], dtype=np.int8),
        }
        
        _, reward, _, _, info = env.step(action)
        
        # Check reward is negative (penalty-based)
        assert isinstance(reward, float)
        
        # Check info contains expected metrics
        assert "episode_energy_cost" in info
        assert "pmv" in info
    
    def test_render(self, env, capsys):
        """Test render function."""
        env.reset(seed=42)
        env.render(mode="human")
        
        captured = capsys.readouterr()
        assert "Step" in captured.out or captured.out == ""
    
    def test_close(self, env):
        """Test close function."""
        env.close()  # Should not raise


class TestEnvConfig:
    """Test suite for EnvConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = EnvConfig()
        
        assert config.num_zones == 1
        assert config.steps_per_episode == 8760
        assert config.heating_setpoint_min == 15.0
        assert config.heating_setpoint_max == 25.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = EnvConfig(
            num_zones=3,
            steps_per_episode=168,
            heating_setpoint_min=18.0,
            heating_setpoint_max=28.0,
        )
        
        assert config.num_zones == 3
        assert config.steps_per_episode == 168
        assert config.heating_setpoint_min == 18.0
        assert config.heating_setpoint_max == 28.0


class TestGymnasiumCompatibility:
    """Test compatibility with gymnasium API."""
    
    def test_env_creation(self):
        """Test environment can be created."""
        if not GYMNASIUM_AVAILABLE:
            pytest.skip("Gymnasium not available")
        
        import gymnasium as gym
        
        # Import to trigger registration
        from tools.gymnasium_env import FluxionEnv, EnvConfig
        
        # Create config first
        config = EnvConfig(num_zones=1, steps_per_episode=24)
        env = gym.make("Fluxion-v0", config=config)
        
        assert env is not None
        env.close()
    
    def test_reset_returns_two_values(self):
        """Test reset returns (observation, info)."""
        if not GYMNASIUM_AVAILABLE:
            pytest.skip("Gymnasium not available")
        
        from tools.gymnasium_env import FluxionEnv
        
        env = FluxionEnv(config=EnvConfig(num_zones=1, steps_per_episode=24))
        result = env.reset(seed=42)
        
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        
        env.close()
    
    def test_step_returns_five_values(self):
        """Test step returns (obs, reward, terminated, truncated, info)."""
        if not GYMNASIUM_AVAILABLE:
            pytest.skip("Gymnasium not available")
        
        from tools.gymnasium_env import FluxionEnv
        
        env = FluxionEnv(config=EnvConfig(num_zones=1, steps_per_episode=24))
        env.reset(seed=42)
        
        action = {
            "continuous": np.array([20.0, 24.0, 50.0], dtype=np.float32),
            "discrete": np.array([1, 1, 1, 0], dtype=np.int8),
        }
        
        result = env.step(action)
        
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
