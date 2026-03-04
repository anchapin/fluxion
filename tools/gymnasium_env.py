#!/usr/bin/env python3
"""
Phase 3: Reinforcement Learning Environment - FluxionEnv

This module provides a Gymnasium-compatible environment for training RL agents
to control HVAC systems using the Fluxion building energy model.

The environment wraps the Rust engine via PyO3 bindings and implements
the Gymnasium API for compatibility with Stable Baselines3 and Ray RLlib.

Issue: GitHub #328 - Develop Gymnasium Environment Wrapper (FluxionEnv)

State Space:
- Indoor temperatures (per zone)
- Weather forecasts (outdoor temp, solar radiation)
- Electricity prices (time-of-use pricing)
- Time of day (hour, day of week)

Action Space:
- Continuous: HVAC heating/cooling setpoints, fan speeds
- Discrete: Equipment toggles (heater, cooler, fan, economizer)

Reward Function:
- Energy cost penalty (based on electricity prices)
- PMV (Predicted Mean Vote) thermal comfort penalty
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    logger.warning("Gymnasium not available - please install with: pip install gymnasium")
    # Create minimal stub for type hints
    spaces = None
    gym = None

try:
    import fluxion
    FLUXION_AVAILABLE = True
except ImportError:
    FLUXION_AVAILABLE = False
    logger.warning("Fluxion not available - using simplified thermal model")


# Default electricity price schedule (time-of-use in $/kWh)
# Peak: 16:00-21:00, Off-peak: 00:00-06:00, Shoulder: 6:00-16:00 & 21:00-24:00
DEFAULT_ELECTRICITY_PRICES = {
    "peak": 0.35,      # $/kWh
    "shoulder": 0.20,  # $/kWh
    "off_peak": 0.10,  # $/kWh
}


@dataclass
class EnvConfig:
    """Configuration for the Fluxion RL environment."""
    # Number of thermal zones
    num_zones: int = 1
    
    # Episode length (timesteps)
    steps_per_episode: int = 8760  # 1 year of hourly steps
    
    # Action space: HVAC setpoints (°C)
    heating_setpoint_min: float = 15.0
    heating_setpoint_max: float = 25.0
    cooling_setpoint_min: float = 20.0
    cooling_setpoint_max: float = 30.0
    
    # Action space: Fan speed (0-100%)
    fan_speed_min: float = 0.0
    fan_speed_max: float = 100.0
    
    # Discrete equipment
    num_equipment: int = 4  # [heater, cooler, fan, economizer]
    
    # State space: Weather forecast horizon (hours)
    weather_forecast_horizon: int = 24
    
    # Reward weights
    energy_weight: float = -1.0  # Minimize energy cost
    pmv_weight: float = -1.0     # Penalize thermal discomfort (PMV)
    
    # Thermal comfort parameters for PMV
    # Default indoor conditions
    indoor_rh: float = 50.0      # Relative humidity (%)
    air_velocity: float = 0.1     # m/s
    clothing_insulation: float = 0.5  # clo (typical indoor clothing)
    metabolic_rate: float = 1.0    # met (sedentary activity)
    
    # Electricity prices
    electricity_prices: Dict[str, float] = field(default_factory=lambda: DEFAULT_ELECTRICITY_PRICES.copy())


class FluxionEnv:
    """
    Gymnasium-compatible environment for HVAC control via RL.
    
    This environment wraps the Fluxion Rust engine (when available) and provides:
    - State: Indoor temps, weather forecasts, electricity prices, time features
    - Action: HVAC setpoints (heating, cooling), fan speeds, equipment toggles
    - Reward: Energy cost minimization + PMV thermal comfort penalty
    
    Compatible with Stable Baselines3 and Ray RLlib.
    
    Gymnasium API:
    - observation_space: gymnasium.spaces.Box
    - action_space: gymnasium.spaces.Box (continuous) + gymnasium.spaces.MultiBinary (discrete)
    - reset(seed): Returns initial observation and info
    - step(action): Returns observation, reward, terminated, truncated, info
    
    Usage:
        import gymnasium as gym
        env = gym.make("Fluxion-v0")
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
    """
    
    # Metadata for Gymnasium registration
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        use_surrogates: bool = False,
        weather_file: Optional[str] = None,
        electricity_price_file: Optional[str] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the Fluxion RL environment.
        
        Args:
            config: Environment configuration
            use_surrogates: Use AI surrogates for faster simulation
            weather_file: Path to weather file (EPW format)
            electricity_price_file: Path to electricity price schedule (CSV)
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        self.config = config or EnvConfig()
        self.use_surrogates = use_surrogates
        self.render_mode = render_mode
        
        # Initialize the Fluxion model if available
        self._init_fluxion_model()
        
        # Initialize weather data
        self._init_weather(weather_file)
        
        # Initialize electricity prices
        self._init_electricity_prices(electricity_price_file)
        
        # Define action and observation spaces
        self._define_spaces()
        
        # State tracking
        self.current_step = 0
        self.episode_energy_cost = 0.0
        self.episode_pmv_penalty = 0.0
        self.episode_energy = 0.0
        
        # Internal state
        self._reset_state()
    
    def _init_fluxion_model(self):
        """Initialize the Fluxion model if available."""
        if FLUXION_AVAILABLE:
            try:
                self.model = fluxion.Model(num_zones=self.config.num_zones)
                logger.info(f"Fluxion model initialized with {self.config.num_zones} zone(s)")
            except Exception as e:
                logger.warning(f"Failed to initialize Fluxion model: {e}. Using simplified model.")
                self.model = None
        else:
            self.model = None
    
    def _init_weather(self, weather_file: Optional[str]):
        """Initialize weather data from file or generate synthetic."""
        if weather_file is not None and False:  # TODO: Implement EPW parsing
            # Load from EPW file
            logger.info(f"Loading weather from {weather_file}")
            # Placeholder - would use epw library or custom parser
            self._outdoor_temps = np.zeros(self.config.steps_per_episode)
            self._solar_radiation = np.zeros(self.config.steps_per_episode)
        else:
            # Generate synthetic weather
            self._generate_synthetic_weather()
    
    def _generate_synthetic_weather(self):
        """Generate synthetic weather data for the episode."""
        hours = np.arange(self.config.steps_per_episode)
        
        # Outdoor temperature: sinusoidal with yearly cycle + daily cycle + noise
        base_temp = 12.5  # Annual average
        yearly_amplitude = 17.5  # Summer vs winter
        daily_amplitude = 5.0   # Day vs night
        
        yearly_cycle = np.sin(2 * np.pi * (hours - 1200) / 8760)
        daily_cycle = np.sin(2 * np.pi * hours / 24)
        
        self._outdoor_temps = (
            base_temp +
            yearly_amplitude * yearly_cycle +
            daily_amplitude * daily_cycle +
            np.random.normal(0, 2, self.config.steps_per_episode)
        )
        
        # Clip to realistic range
        self._outdoor_temps = np.clip(self._outdoor_temps, -20, 45)
        
        # Solar radiation: clear day pattern
        hour_of_day = hours % 24
        day_of_year = hours // 24
        
        # Solar elevation varies by day of year
        seasonal_factor = np.sin(2 * np.pi * day_of_year / 365)
        
        # Daytime pattern (sunrise ~6am, sunset ~6pm)
        daylight = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
        
        # Max solar ~1000 W/m²
        self._solar_radiation = (
            daylight * 
            (1 + 0.3 * seasonal_factor) * 
            800 * 
            (1 + np.random.normal(0, 0.1, self.config.steps_per_episode))
        )
        self._solar_radiation = np.maximum(0, self._solar_radiation)
        
        # Relative humidity (inverse of temperature pattern)
        self._relative_humidity = 50 + 20 * np.sin(yearly_cycle + np.pi/2)
    
    def _init_electricity_prices(self, price_file: Optional[str]):
        """Initialize electricity price schedule."""
        if price_file is not None:
            logger.info(f"Loading electricity prices from {price_file}")
            # TODO: Load from CSV file
            pass
        
        # Generate time-of-use prices
        hours = np.arange(self.config.steps_per_episode)
        hour_of_day = hours % 24
        
        prices = np.zeros(self.config.steps_per_episode)
        
        for i, h in enumerate(hour_of_day):
            if 16 <= h < 21:  # Peak: 4pm-9pm
                prices[i] = self.config.electricity_prices["peak"]
            elif 6 <= h < 16 or 21 <= h < 24:  # Shoulder
                prices[i] = self.config.electricity_prices["shoulder"]
            else:  # Off-peak
                prices[i] = self.config.electricity_prices["off_peak"]
        
        # Add some random variation
        prices = prices * (1 + np.random.normal(0, 0.05, self.config.steps_per_episode))
        self._electricity_prices = np.maximum(0.05, prices)
    
    def _define_spaces(self):
        """Define Gymnasium action and observation spaces."""
        if not GYMNASIUM_AVAILABLE:
            logger.error("Gymnasium is required but not available")
            return
        
        num_zones = self.config.num_zones
        forecast_horizon = self.config.weather_forecast_horizon
        
        # ========================
        # OBSERVATION SPACE
        # ========================
        # Per zone: indoor temperature (1)
        # Weather forecast: outdoor temps (forecast_horizon), solar rad (forecast_horizon)
        # Electricity prices: forecast horizon
        # Time: hour of day (1), day of year (1)
        
        obs_dim = (
            num_zones +  # Indoor temperatures
            forecast_horizon +  # Outdoor temp forecast
            forecast_horizon +  # Solar radiation forecast
            forecast_horizon +  # Electricity price forecast
            2  # Hour of day, day of year
        )
        
        # Define bounds
        obs_low = np.zeros(obs_dim, dtype=np.float32)
        obs_high = np.zeros(obs_dim, dtype=np.float32)
        
        idx = 0
        
        # Indoor temperatures: 10-35°C
        for _ in range(num_zones):
            obs_low[idx] = 10.0
            obs_high[idx] = 35.0
            idx += 1
        
        # Outdoor temp forecast: -20 to 45°C
        for _ in range(forecast_horizon):
            obs_low[idx] = -20.0
            obs_high[idx] = 45.0
            idx += 1
        
        # Solar radiation forecast: 0-1200 W/m²
        for _ in range(forecast_horizon):
            obs_low[idx] = 0.0
            obs_high[idx] = 1200.0
            idx += 1
        
        # Electricity price forecast: 0-1 $/kWh
        for _ in range(forecast_horizon):
            obs_low[idx] = 0.0
            obs_high[idx] = 1.0
            idx += 1
        
        # Hour of day: 0-23
        obs_low[idx] = 0.0
        obs_high[idx] = 23.0
        idx += 1
        
        # Day of year: 0-364
        obs_low[idx] = 0.0
        obs_high[idx] = 364.0
        idx += 1
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )
        
        # ========================
        # ACTION SPACE
        # ========================
        # Continuous: [heating_setpoint, cooling_setpoint, fan_speed]
        # Binary: [heater_on, cooler_on, fan_on, economizer_on]
        
        self.action_space = spaces.Dict({
            "continuous": spaces.Box(
                low=np.array([
                    self.config.heating_setpoint_min,
                    self.config.cooling_setpoint_min,
                    self.config.fan_speed_min,
                ], dtype=np.float32),
                high=np.array([
                    self.config.heating_setpoint_max,
                    self.config.cooling_setpoint_max,
                    self.config.fan_speed_max,
                ], dtype=np.float32),
            ),
            "discrete": spaces.MultiBinary(self.config.num_equipment),
        })
    
    def _reset_state(self):
        """Reset internal state for new episode."""
        self.current_step = 0
        self.episode_energy_cost = 0.0
        self.episode_pmv_penalty = 0.0
        self.episode_energy = 0.0
        
        # Initial zone temperatures (equilibrium with outdoor)
        self._zone_temperatures = np.ones(self.config.num_zones) * 20.0
        
        # Default HVAC settings
        self._heating_setpoint = 20.0
        self._cooling_setpoint = 24.0
        self._fan_speed = 50.0
        
        # Equipment states [heater, cooler, fan, economizer]
        self._equipment_states = np.array([1, 1, 1, 0], dtype=np.int8)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Gymnasium API: Returns observation and info dict.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            observation: Initial state
            info: Additional information
        """
        if seed is not None:
            np.random.seed(seed)
        
        self._reset_state()
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        logger.debug(f"Environment reset. Step: {self.current_step}")
        return obs, info
    
    def step(
        self, 
        action: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep.
        
        Gymnasium API.
        
        Args:
            action: Dict with:
                - "continuous": [heating_setpoint, cooling_setpoint, fan_speed]
                - "discrete": [heater_on, cooler_on, fan_on, economizer_on]
                
        Returns:
            observation: Current state
            reward: Reward for this step
            terminated: Episode ended (year complete)
            truncated: Episode truncated (max steps)
            info: Additional information
        """
        # Parse continuous actions
        continuous = action.get("continuous", np.array([20.0, 24.0, 50.0]))
        self._heating_setpoint = float(np.clip(
            continuous[0], 
            self.config.heating_setpoint_min, 
            self.config.heating_setpoint_max
        ))
        self._cooling_setpoint = float(np.clip(
            continuous[1], 
            self.config.cooling_setpoint_min, 
            self.config.cooling_setpoint_max
        ))
        self._fan_speed = float(np.clip(
            continuous[2], 
            self.config.fan_speed_min, 
            self.config.fan_speed_max
        ))
        
        # Parse discrete actions (equipment toggles)
        discrete = action.get("discrete", np.array([1, 1, 1, 0]))
        self._equipment_states = discrete.astype(np.int8)
        
        # Ensure valid setpoint range
        if self._heating_setpoint >= self._cooling_setpoint:
            self._cooling_setpoint = self._heating_setpoint + 0.5
        
        # Get current conditions
        outdoor_temp = self._outdoor_temps[self.current_step]
        solar_rad = self._solar_radiation[self.current_step]
        elec_price = self._electricity_prices[self.current_step]
        
        # Simulate one timestep
        energy, zone_temps = self._simulate_timestep(
            outdoor_temp, solar_rad, elec_price
        )
        
        # Update state
        self._zone_temperatures = zone_temps
        self.episode_energy += energy
        
        # Energy cost
        energy_cost = energy * elec_price / 1000.0  # Convert Wh to kWh
        self.episode_energy_cost += energy_cost
        
        # Calculate PMV and thermal comfort
        pmv = self._calculate_pmv(zone_temps[0], outdoor_temp)
        pmv_penalty = pmv ** 2  # Square to penalize deviation from 0
        self.episode_pmv_penalty += pmv_penalty
        
        # Calculate reward
        reward = self._calculate_reward(energy_cost, pmv)
        
        # Check termination
        self.current_step += 1
        terminated = self.current_step >= self.config.steps_per_episode
        truncated = False
        
        # Get observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _simulate_timestep(
        self, 
        outdoor_temp: float, 
        solar_rad: float,
        elec_price: float,
    ) -> Tuple[float, np.ndarray]:
        """
        Simulate one hour of building thermal dynamics.
        
        Uses either Fluxion engine or simplified 5R1C model.
        
        Returns:
            energy: Energy consumption (Wh)
            zone_temperatures: Updated zone temperatures
        """
        if self.model is not None and FLUXION_AVAILABLE:
            # Use Fluxion model
            return self._simulate_with_fluxion(outdoor_temp, solar_rad)
        else:
            # Use simplified thermal model
            return self._simulate_simplified(outdoor_temp, solar_rad)
    
    def _simulate_with_fluxion(
        self, 
        outdoor_temp: float, 
        solar_rad: float
    ) -> Tuple[float, np.ndarray]:
        """Simulate using Fluxion Rust engine."""
        # This would call the Fluxion model via PyO3
        # For now, fall back to simplified model
        return self._simulate_simplified(outdoor_temp, solar_rad)
    
    def _simulate_simplified(
        self, 
        outdoor_temp: float, 
        solar_rad: float
    ) -> Tuple[float, np.ndarray]:
        """
        Simulate using simplified 5R1C thermal model.
        
        For fast RL training when Fluxion is not available.
        Uses realistic building thermal parameters.
        """
        # Thermal parameters (realistic for typical building)
        # Based on standard 5R1C model
        thermal_mass = 1e7  # J/K (large thermal mass for stability)
        h_transmission = 500  # W/K (overall heat transfer coefficient)
        h_ventilation = 100   # W/K (ventilation)
        
        # Current state
        Ti = self._zone_temperatures[0]
        To = outdoor_temp
        
        # Solar gain (solar heat gain factor * window area * radiation)
        window_area = 20  # m²
        shgf = 0.8  # Solar heat gain factor
        solar_gain = solar_rad * window_area * shgf / 1000  # kW
        
        # Adjust for fan speed (increases ventilation)
        fan_factor = self._fan_speed / 100.0
        h_ventilation_effective = h_ventilation * (1 + fan_factor)
        
        # HVAC control based on equipment states and setpoints
        hvac_power = 0.0  # kW
        
        heater_on = bool(self._equipment_states[0])
        cooler_on = bool(self._equipment_states[1])
        fan_on = bool(self._equipment_states[2])
        
        # Calculate target temperature (midpoint of setpoints)
        target_temp = (self._heating_setpoint + self._cooling_setpoint) / 2
        
        # Only provide heating/cooling if equipment is on
        if heater_on and Ti < self._heating_setpoint:
            # Heating required - proportional control
            error = self._heating_setpoint - Ti
            hvac_power = min(5.0, max(0.5, error * 50))  # Max 5kW, min 0.5kW
        
        elif cooler_on and Ti > self._cooling_setpoint:
            # Cooling required - proportional control
            error = Ti - self._cooling_setpoint
            hvac_power = -min(5.0, max(0.5, error * 50))  # Max 5kW cooling
        
        # Fan baseline power when on
        if fan_on and abs(hvac_power) < 0.1:
            hvac_power = 0.1  # Fan baseline
        
        # Convert to watts
        hvac_power_w = hvac_power * 1000  # W
        
        # Energy balance
        Q_loss = (h_transmission + h_ventilation_effective) * (Ti - To)  # W
        Q_solar = solar_gain * 1000  # W
        Q_net = Q_solar + hvac_power_w - Q_loss
        
        # Update temperature (timestep = 3600 seconds = 1 hour)
        dT = Q_net * 3600 / thermal_mass
        Ti_next = Ti + dT
        
        # Clip temperature to realistic range
        Ti_next = np.clip(Ti_next, -30, 60)
        
        # Calculate energy consumption (Wh)
        energy = max(0, abs(hvac_power_w))  # Wh
        
        # Update zone temperatures
        zone_temps = np.array([Ti_next] * self.config.num_zones)
        
        return energy, zone_temps
    
    def _calculate_pmv(
        self, 
        zone_temp: float, 
        outdoor_temp: float
    ) -> float:
        """
        Calculate Predicted Mean Vote (PMV) for thermal comfort.
        
        Uses simplified PMV formula based on Fanger's equation.
        For a more accurate PMV, use the pythermalcomfort library.
        
        PMV scale:
        -3: Cold
        -2: Cool
        -1: Slightly cool
        0: Neutral
        +1: Slightly warm
        +2: Warm
        +3: Hot
        
        Args:
            zone_temp: Indoor air temperature (°C)
            outdoor_temp: Outdoor temperature (°C) - for radiant temp estimate
            
        Returns:
            PMV value
        """
        Ta = zone_temp  # Air temperature
        
        # Estimate mean radiant temperature (simplified)
        # In practice, this depends on wall temperatures, solar, etc.
        Tr = (Ta + outdoor_temp) / 2  # Simplified
        
        # Relative humidity
        RH = self.config.indoor_rh  # %
        
        # Air velocity
        v = self.config.air_velocity  # m/s
        
        # Clothing insulation
        Icl = self.config.clothing_insulation  # clo
        
        # Metabolic rate
        M = self.config.metabolic_rate  # met
        
        # Simplified PMV calculation (Fanger's equation)
        # This is a linearized approximation for real-time RL
        
        # Thermal sensation due to air temperature
        thermal_sensation = Ta - 22.5  # Neutral temperature ~22.5°C
        
        # Adjustment for radiant temperature
        thermal_sensation += 0.3 * (Tr - Ta)
        
        # Adjustment for humidity (positive = more uncomfortable when hot)
        humidity_effect = (RH - 50) / 50 * 0.2 * max(0, thermal_sensation)
        
        # Adjustment for air movement (positive = cooler when hot, warmer when cold)
        air_effect = 0.5 * (v - 0.1) * (25 - Ta)
        
        # Clothing adjustment
        clothing_effect = -0.5 * (Icl - 0.5)
        
        # Activity adjustment
        activity_effect = 0.5 * (M - 1.0)
        
        # Combined PMV
        pmv = (
            thermal_sensation + 
            humidity_effect + 
            air_effect + 
            clothing_effect + 
            activity_effect
        )
        
        # Clip to valid range [-3, 3]
        pmv = np.clip(pmv, -3.0, 3.0)
        
        return pmv
    
    def _calculate_reward(self, energy_cost: float, pmv: float) -> float:
        """
        Calculate reward for current state.
        
        Reward = energy_weight * energy_cost + pmv_weight * pmv^2
        
        The PMV is squared to penalize larger deviations more heavily.
        """
        # Energy cost penalty (negative reward to minimize)
        energy_penalty = self.config.energy_weight * energy_cost
        
        # PMV thermal comfort penalty (squared for quadratic penalty)
        pmv_penalty = self.config.pmv_weight * (pmv ** 2)
        
        reward = energy_penalty + pmv_penalty
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        num_zones = self.config.num_zones
        horizon = self.config.weather_forecast_horizon
        
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        idx = 0
        
        # Indoor temperatures
        obs[idx:idx + num_zones] = self._zone_temperatures
        idx += num_zones
        
        # Weather forecast (outdoor temps)
        start = self.current_step
        end = min(self.current_step + horizon, self.config.steps_per_episode)
        available = end - start
        
        if available > 0:
            obs[idx:idx + available] = self._outdoor_temps[start:end]
            # Fill remaining with last value
            if available < horizon:
                obs[idx + available:idx + horizon] = self._outdoor_temps[end - 1]
        idx += horizon
        
        # Weather forecast (solar radiation)
        if available > 0:
            obs[idx:idx + available] = self._solar_radiation[start:end]
            if available < horizon:
                obs[idx + available:idx + horizon] = self._solar_radiation[end - 1]
        idx += horizon
        
        # Electricity price forecast
        if available > 0:
            obs[idx:idx + available] = self._electricity_prices[start:end]
            if available < horizon:
                obs[idx + available:idx + horizon] = self._electricity_prices[end - 1]
        idx += horizon
        
        # Time features
        hour = self.current_step % 24
        day_of_year = (self.current_step // 24) % 365
        
        obs[idx] = float(hour)
        obs[idx + 1] = float(day_of_year)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        hour = self.current_step % 24
        day_of_year = (self.current_step // 24) % 365
        
        pmv = self._calculate_pmv(
            self._zone_temperatures[0],
            self._outdoor_temps[max(0, self.current_step - 1)]
        )
        
        return {
            "step": self.current_step,
            "hour": hour,
            "day_of_year": day_of_year,
            # HVAC settings
            "heating_setpoint": self._heating_setpoint,
            "cooling_setpoint": self._cooling_setpoint,
            "fan_speed": self._fan_speed,
            "equipment_states": self._equipment_states.tolist(),
            # Zone state
            "zone_temperatures": self._zone_temperatures.tolist(),
            "outdoor_temperature": self._outdoor_temps[self.current_step] if self.current_step < len(self._outdoor_temps) else 0,
            "solar_radiation": self._solar_radiation[self.current_step] if self.current_step < len(self._solar_radiation) else 0,
            "electricity_price": self._electricity_prices[self.current_step] if self.current_step < len(self._electricity_prices) else 0,
            # Metrics
            "pmv": pmv,
            "energy_this_step": 0,  # Would be from step results
            "episode_energy_cost": self.episode_energy_cost,
            "episode_pmv_penalty": self.episode_pmv_penalty,
            "episode_energy": self.episode_energy,
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(f"Step {self.current_step}: T_zone={self._zone_temperatures[0]:.1f}°C, "
                  f"T_out={self._outdoor_temps[self.current_step]:.1f}°C, "
                  f"Price=${self._electricity_prices[self.current_step]:.3f}/kWh")
        # TODO: Implement rgb_array rendering
    
    def close(self):
        """Clean up resources."""
        pass
    
    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self


# Gymnasium environment registration
def make(env_id: str = "Fluxion-v0", **kwargs) -> "FluxionEnv":
    """Create a Fluxion environment."""
    return FluxionEnv(**kwargs)


# Register with Gymnasium
if GYMNASIUM_AVAILABLE:
    from gymnasium.envs.registration import register
    
    register(
        id="Fluxion-v0",
        entry_point="fluxion.gymnasium_env:FluxionEnv",
        max_episode_steps=8760,
    )
    logger.info("Registered Fluxion-v0 with Gymnasium")
    
    # Also register with versioned ID
    register(
        id="Fluxion-v0.1",
        entry_point="fluxion.gymnasium_env:FluxionEnv",
        max_episode_steps=8760,
    )
    logger.info("Registered Fluxion-v0.1 with Gymnasium")


if __name__ == "__main__":
    # Test the environment
    if GYMNASIUM_AVAILABLE:
        import gymnasium as gym
        import sys
        sys.path.insert(0, '/home/alex/Projects/fluxion')
        
        print("Testing FluxionEnv...")
        
        # Import directly
        from tools.gymnasium_env import FluxionEnv
        env = FluxionEnv()
        
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        obs, info = env.reset(seed=42)
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial info: {info}")
        
        # Run a few steps
        for i in range(5):
            action = {
                "continuous": np.array([20.0, 24.0, 50.0], dtype=np.float32),
                "discrete": np.array([1, 1, 1, 0], dtype=np.int8),
            }
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: reward={reward:.4f}, zone_temp={info['zone_temperatures'][0]:.2f}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("Test completed!")
    else:
        print("Gymnasium not available. Install with: pip install gymnasium")
