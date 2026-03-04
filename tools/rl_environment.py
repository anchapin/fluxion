#!/usr/bin/env python3
"""
Phase 3: Reinforcement Learning Environment

This module provides a Gymnasium-compatible environment for training RL agents
to control HVAC systems using the Fluxion building energy model.

The environment wraps the Rust engine via PyO3 bindings and implements
the Gymnasium API for compatibility with Stable Baselines3 and Ray RLlib.
"""

import logging
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Module-level flag
FLUXION_AVAILABLE = False

def _fluxion_available_global():
    """Get the global FLUXION_AVAILABLE flag."""
    return FLUXION_AVAILABLE

try:
    import fluxion
    FLUXION_AVAILABLE = True
except ImportError:
    logger.warning("Fluxion not available - using mock mode for testing")


@dataclass
class EnvConfig:
    """Configuration for the Fluxion RL environment."""
    # Action space
    heating_setpoint_min: float = 15.0  # °C
    heating_setpoint_max: float = 25.0  # °C
    cooling_setpoint_min: float = 20.0  # °C
    cooling_setpoint_max: float = 30.0  # °C
    
    # Window U-value (also controllable if desired)
    u_value_min: float = 0.5
    u_value_max: float = 3.0
    
    # Observation space
    # Features: [outdoor_temp, zone_temp, solar_radiation, hour_of_day, day_of_year]
    num_observation_features: int = 5
    
    # Simulation
    num_zones: int = 1
    steps_per_episode: int = 8760  # 1 year of hourly steps
    
    # Reward weights
    energy_weight: float = -1.0  # Minimize energy
    comfort_weight: float = -0.5  # Penalize thermal discomfort
    comfort_band: float = 2.0  # ±°C comfort band around setpoint


class FluxionEnv:
    """
    Gymnasium-compatible environment for HVAC control via RL.
    
    This environment wraps the Fluxion Rust engine and provides:
    - State: Outdoor temp, zone temp, solar radiation, time features
    - Action: HVAC setpoints (heating, cooling)
    - Reward: Energy minimization + comfort penalty
    
    Compatible with Stable Baselines3 and Ray RLlib.
    """
    
    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        use_surrogates: bool = False,
        weather_file: Optional[str] = None,
    ):
        """
        Initialize the Fluxion RL environment.
        
        Args:
            config: Environment configuration
            use_surrogates: Use AI surrogates for faster simulation
            weather_file: Path to weather file (EPW format)
        """
        self.config = config or EnvConfig()
        self.use_surrogates = use_surrogates
        
        # Initialize the Fluxion model
        self.model = None
        if _fluxion_available_global():
            try:
                self.model = fluxion.Model(num_zones=self.config.num_zones)
                logger.info(f"Fluxion model initialized with {self.config.num_zones} zone(s)")
            except Exception as e:
                logger.warning(f"Failed to initialize Fluxion model: {e}. Using mock mode.")
                self.model = None
        
        # Weather data (simplified - use dummy sinusoidal patterns)
        self._init_weather(weather_file)
        
        # State tracking
        self.current_step = 0
        self.episode_energy = 0.0
        self.episode_discomfort = 0.0
        
        # Action space bounds
        self.action_space_low = np.array([
            self.config.heating_setpoint_min,
            self.config.cooling_setpoint_min,
        ], dtype=np.float32)
        
        self.action_space_high = np.array([
            self.config.heating_setpoint_max,
            self.config.cooling_setpoint_max,
        ], dtype=np.float32)
        
        # Observation space bounds
        # [outdoor_temp, zone_temp, solar_radiation, hour_of_day, day_of_year_norm]
        obs_low = np.array([
            -20.0,  # Cold outdoor temp
            10.0,   # Cold zone temp
            0.0,    # No solar
            0.0,    # Hour 0
            0.0,    # Day 0
        ], dtype=np.float32)
        
        obs_high = np.array([
            45.0,   # Hot outdoor temp
            35.0,   # Hot zone temp
            1000.0, # High solar (W/m²)
            23.0,   # Hour 23
            364.0, # Day 364
        ], dtype=np.float32)
        
        self.observation_space = _MockSpace("box", low=obs_low, high=obs_high)
        self.action_space = _MockSpace("box", low=self.action_space_low, high=self.action_space_high)
        
        # Initial state
        self._reset_state()
    
    def _init_weather(self, weather_file: Optional[str]):
        """Initialize weather data."""
        # Generate synthetic weather if no file provided
        # Typical annual pattern: sinusoidal with yearly cycle
        hours = np.arange(self.config.steps_per_episode)
        
        # Temperature: -5°C to 30°C sinusoidal + noise
        base_temp = 12.5
        temp_amplitude = 17.5
        self._outdoor_temps = base_temp + temp_amplitude * np.sin(
            2 * np.pi * (hours - 1200) / 8760
        ) + np.random.normal(0, 2, self.config.steps_per_episode)
        
        # Solar radiation: 0 to ~800 W/m² (simplified daily pattern)
        hour_of_day = hours % 24
        solar_base = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
        self._solar_radiation = solar_base * (1 + 0.3 * np.sin(2 * np.pi * hours / 8760)) * 800
        self._solar_radiation = np.maximum(0, self._solar_radiation + np.random.normal(0, 50, self.config.steps_per_episode))
    
    def _reset_state(self):
        """Reset internal state for new episode."""
        self.current_step = 0
        self.episode_energy = 0.0
        self.episode_discomfort = 0.0
        
        # Initial zone temperature - start at a reasonable indoor temp
        self._zone_temperature = 20.0  # Starting temp (typical indoor)
        self._current_energy = 0.0
        
        # Default parameters
        self._heating_setpoint = 20.0
        self._cooling_setpoint = 24.0
        self._u_value = 1.5
        
        # Thermal parameters - using RC model
        # Thermal resistance (K/W)
        self._r_value = 0.01  # Low resistance = high insulation
        # Thermal capacity (J/K) - higher for more stability
        self._c_value = 5000000  # 5 MJ/K - large thermal mass
        # Internal heat gains from occupants, equipment, lights
        self._internal_gain = 800  # W
        
        # HVAC capacity (fixed power)
        self._hvac_heating_capacity = 5000  # 5 kW max heating
        self._hvac_cooling_capacity = 5000  # 5 kW max cooling
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Gymnasium API: Returns observation and info dict.
        """
        if seed is not None:
            np.random.seed(seed)
        
        self._reset_state()
        
        # Generate new weather for this episode
        self._init_weather(None)
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        logger.debug(f"Environment reset. Initial obs: {obs}")
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep.
        
        Args:
            action: [heating_setpoint, cooling_setpoint]
            
        Returns:
            observation: Current state
            reward: Reward for this step
            terminated: Episode ended (year complete)
            truncated: Episode truncated (max steps)
            info: Additional info
        """
        # Parse action
        self._heating_setpoint = float(np.clip(action[0], self.config.heating_setpoint_min, self.config.heating_setpoint_max))
        self._cooling_setpoint = float(np.clip(action[1], self.config.cooling_setpoint_min, self.config.cooling_setpoint_max))
        
        # Ensure valid setpoint range
        if self._heating_setpoint >= self._cooling_setpoint:
            self._heating_setpoint = self._cooling_setpoint - 0.5
        
        # Get current conditions
        outdoor_temp = self._outdoor_temps[self.current_step]
        solar_rad = self._solar_radiation[self.current_step]
        
        # Simulate one timestep
        energy, zone_temp = self._simulate_timestep(outdoor_temp, solar_rad)
        
        # Update state
        self._zone_temperature = zone_temp
        self._current_energy = energy
        self.episode_energy += energy
        
        # Calculate reward
        reward = self._calculate_reward(zone_temp, energy)
        
        # Track discomfort
        discomfort = self._calculate_discomfort(zone_temp)
        self.episode_discomfort += discomfort
        
        # Check termination
        self.current_step += 1
        terminated = self.current_step >= self.config.steps_per_episode
        truncated = False
        
        # Get observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _simulate_timestep(self, outdoor_temp: float, solar_rad: float) -> Tuple[float, float]:
        """
        Simulate one hour of building thermal dynamics.
        
        Uses a simple RC model for building thermal dynamics.
        For accurate simulation, use Fluxion Rust engine.
        """
        # RC model parameters
        R = self._r_value  # Thermal resistance
        C = self._c_value  # Thermal capacity
        
        # Current state
        Ti = self._zone_temperature
        To = outdoor_temp
        
        # Heat gains from solar (W)
        solar_gain = solar_rad * 0.2 * 10  # Approx window area * SHGF
        
        # Internal gains (occupants + equipment)
        internal_gain = self._internal_gain
        
        # HVAC control with fixed capacity
        hvac_power = 0
        if Ti < self._heating_setpoint:
            # Heating required - use fixed capacity
            hvac_power = self._hvac_heating_capacity
        elif Ti > self._cooling_setpoint:
            # Cooling required - use fixed capacity
            hvac_power = -self._hvac_cooling_capacity
        
        # Total heat input
        Q_total = solar_gain + internal_gain + hvac_power
        
        # RC model: C * dT/dt = (T_outdoor - T_zone) / R + Q
        # Simplified: C * dT/dt = (To - Ti) / R + Q
        dT = ((To - Ti) / R + Q_total) * 3600 / C  # 1 hour timestep
        
        # Update temperature
        Ti_next = Ti + dT
        
        # Bound temperature to reasonable range
        Ti_next = np.clip(Ti_next, -30, 60)
        
        # Calculate energy consumption (Wh)
        energy = max(0, abs(hvac_power))
        
        return energy, Ti_next
    
    def _calculate_reward(self, zone_temp: float, energy: float) -> float:
        """
        Calculate reward for current state.
        
        Reward = energy_weight * energy + comfort_weight * discomfort
        """
        discomfort = self._calculate_discomfort(zone_temp)
        
        reward = self.config.energy_weight * energy + self.config.comfort_weight * discomfort
        
        return reward
    
    def _calculate_discomfort(self, zone_temp: float) -> float:
        """
        Calculate thermal discomfort (deviation from setpoint range).
        """
        # Setpoint is the average of heating and cooling
        setpoint = (self._heating_setpoint + self._cooling_setpoint) / 2
        band = self.config.comfort_band
        
        if zone_temp < setpoint - band:
            return (setpoint - band - zone_temp) ** 2
        elif zone_temp > setpoint + band:
            return (zone_temp - setpoint - band) ** 2
        else:
            return 0.0
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        hour = self.current_step % 24
        day_of_year = (self.current_step // 24) % 365
        
        obs = np.array([
            self._outdoor_temps[self.current_step],  # Outdoor temperature
            self._zone_temperature,                   # Zone temperature
            self._solar_radiation[self.current_step], # Solar radiation
            hour / 24.0,                              # Hour of day (normalized)
            day_of_year / 365.0,                     # Day of year (normalized)
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> Dict:
        """Get info dictionary."""
        return {
            "heating_setpoint": self._heating_setpoint,
            "cooling_setpoint": self._cooling_setpoint,
            "zone_temperature": self._zone_temperature,
            "outdoor_temperature": self._outdoor_temps[self.current_step],
            "solar_radiation": self._solar_radiation[self.current_step],
            "current_energy": self._current_energy,
            "episode_energy": self.episode_energy,
            "episode_discomfort": self.episode_discomfort,
            "step": self.current_step,
        }
    
    def render(self):
        """Render the environment (not implemented)."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass
    
    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self


class _MockSpace:
    """Mock space for when Fluxion is not available."""
    
    def __init__(self, space_type: str, **kwargs):
        self.space_type = space_type
        self.low = kwargs.get("low")
        self.high = kwargs.get("high")
        self.shape = kwargs.get("shape")
    
    def sample(self) -> np.ndarray:
        """Sample random action/observation."""
        if self.space_type == "box":
            return np.random.uniform(self.low, self.high).astype(np.float32)
        raise NotImplementedError


# For Gymnasium v0.29+ compatibility
def make(env_id: str = "Fluxion-v0", **kwargs) -> FluxionEnv:
    """Create a Fluxion environment."""
    return FluxionEnv(**kwargs)


# Register with Gymnasium if available
try:
    import gymnasium as gym
    from gymnasium.envs.registration import register
    
    register(
        id="Fluxion-v0",
        entry_point="fluxion.rl_environment:FluxionEnv",
        max_episode_steps=8760,
    )
    logger.info("Registered Fluxion-v0 with Gymnasium")
except ImportError:
    logger.warning("Gymnasium not available - RL environment will work as standalone")
except Exception as e:
    logger.warning(f"Could not register environment: {e}")
