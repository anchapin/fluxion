"""
Compliance Data Aggregation Module

This module extracts and aggregates building energy simulation data for compliance reporting.
It processes simulation outputs to generate metrics required for ASHRAE 90.1 and IECC compliance.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from pathlib import Path


@dataclass
class ComplianceMetrics:
    """Container for all compliance-related energy metrics."""
    
    # Building Information
    building_name: str = "Unknown Building"
    building_area_m2: float = 0.0
    building_type: str = "Commercial"
    climate_zone: str = "4A"
    
    # Energy Consumption (annual)
    total_electricity_kwh: float = 0.0
    total_natural_gas_kwh: float = 0.0
    total_energy_kwh: float = 0.0
    
    # Energy Use Intensity (EUI)
    electricity_eui_kwh_m2: float = 0.0
    total_eui_kwh_m2: float = 0.0
    
    # Peak Loads
    peak_heating_load_kw: float = 0.0
    peak_cooling_load_kw: float = 0.0
    peak_electric_demand_kw: float = 0.0
    
    # HVAC Performance
    heating_cop: float = 1.0
    cooling_cop: float = 3.0
    
    # Unmet Hours
    unmet_heating_hours: float = 0.0
    unmet_cooling_hours: float = 0.0
    total_unmet_hours: float = 0.0
    
    # End-use breakdown (kWh)
    heating_energy_kwh: float = 0.0
    cooling_energy_kwh: float = 0.0
    lighting_energy_kwh: float = 0.0
    plug_loads_kwh: float = 0.0
    ventilation_energy_kwh: float = 0.0
    
    # Financial
    electricity_rate_usd_kwh: float = 0.12
    gas_rate_usd_kwh: float = 0.08
    annual_energy_cost_usd: float = 0.0
    
    # Simulation metadata
    simulation_hours: int = 8760
    timestep_hours: float = 1.0
    
    # Hourly data (optional - for detailed analysis)
    hourly_temperatures: List[float] = field(default_factory=list)
    hourly_heating_loads: List[float] = field(default_factory=list)
    hourly_cooling_loads: List[float] = field(default_factory=list)
    hourly_total_loads: List[float] = field(default_factory=list)


class ComplianceDataAggregator:
    """
    Aggregates building energy simulation data into compliance-ready metrics.
    
    This class processes raw simulation outputs from Fluxion and transforms them
    into the structured format required for ASHRAE 90.1 Appendix G and IECC compliance reports.
    """
    
    # Default ASHRAE 90.1 baseline values (Appendix G)
    BASELINE_U_VALUES = {
        "wall": 0.59,  # W/m²K
        "roof": 0.35,  # W/m²K
        "floor": 0.96,  # W/m²K
        "window": 3.17,  # W/m²K
    }
    
    BASELINE_WINDOW_WWR = 0.40  # 40% window-to-wall ratio
    BASELINE_SHGC = 0.39  # Solar Heat Gain Coefficient
    BASELINE_LPD = 10.76  # Lighting Power Density W/m²
    
    def __init__(self, building_area_m2: float = 1000.0, building_type: str = "Commercial"):
        """
        Initialize the compliance data aggregator.
        
        Args:
            building_area_m2: Total conditioned floor area in square meters
            building_type: Type of building (Commercial, Residential, etc.)
        """
        self.building_area_m2 = building_area_m2
        self.building_type = building_type
        self.metrics = ComplianceMetrics(building_area_m2=building_area_m2, building_type=building_type)
    
    def process_simulation_results(
        self,
        hourly_temperatures: List[float],
        hourly_heating_loads: List[float],
        hourly_cooling_loads: List[float],
        hourly_internal_gains: Optional[List[float]] = None,
        hourly_solar_gains: Optional[List[float]] = None,
        hourly_lighting: Optional[List[float]] = None,
        hourly_plug_loads: Optional[List[float]] = None,
    ) -> ComplianceMetrics:
        """
        Process hourly simulation results and compute compliance metrics.
        
        Args:
            hourly_temperatures: Indoor zone temperatures (°C) for each hour
            hourly_heating_loads: Heating loads (W) for each hour
            hourly_cooling_loads: Cooling loads (W) for each hour
            hourly_internal_gains: Internal gains (W) for each hour
            hourly_solar_gains: Solar gains (W) for each hour
            hourly_lighting: Lighting loads (W) for each hour
            hourly_plug_loads: Plug loads (W) for each hour
        
        Returns:
            ComplianceMetrics with all computed values
        """
        n_hours = len(hourly_temperatures)
        self.metrics.simulation_hours = n_hours
        
        # Store hourly data
        self.metrics.hourly_temperatures = hourly_temperatures.copy()
        self.metrics.hourly_heating_loads = hourly_heating_loads.copy()
        self.metrics.hourly_cooling_loads = hourly_cooling_loads.copy()
        
        # Convert hourly loads from Watts to kWh
        heating_kwh = [max(0, load / 1000.0) for load in hourly_heating_loads]
        cooling_kwh = [max(0, -load / 1000.0) for load in hourly_cooling_loads]
        
        # Calculate total energy consumption
        self.metrics.heating_energy_kwh = sum(heating_kwh)
        self.metrics.cooling_energy_kwh = sum(cooling_kwh)
        
        # Add internal gains if provided
        if hourly_internal_gains:
            self.metrics.ventilation_energy_kwh = sum(hourly_internal_gains) / 1000.0
        
        # Add lighting and plug loads if provided
        if hourly_lighting:
            self.metrics.lighting_energy_kwh = sum(hourly_lighting) / 1000.0
        if hourly_plug_loads:
            self.metrics.plug_loads_kwh = sum(hourly_plug_loads) / 1000.0
        
        # Calculate total HVAC energy
        self.metrics.total_energy_kwh = (
            self.metrics.heating_energy_kwh + 
            self.metrics.cooling_energy_kwh + 
            self.metrics.ventilation_energy_kwh +
            self.metrics.lighting_energy_kwh +
            self.metrics.plug_loads_kwh
        )
        
        # Calculate EUI
        if self.building_area_m2 > 0:
            self.metrics.total_eui_kwh_m2 = self.metrics.total_energy_kwh / self.building_area_m2
            self.metrics.electricity_eui_kwh_m2 = (
                (self.metrics.total_energy_kwh - self.metrics.heating_energy_kwh) / 
                self.building_area_m2
            )
        
        # Calculate peak loads
        self.metrics.peak_heating_load_kw = max(hourly_heating_loads) / 1000.0 if hourly_heating_loads else 0.0
        self.metrics.peak_cooling_load_kw = max([-min(0, load) for load in hourly_cooling_loads]) if hourly_cooling_loads else 0.0
        
        # Calculate unmet hours (when temperature is outside acceptable range)
        heating_setpoint = 21.0  # Default
        cooling_setpoint = 24.0  # Default
        tolerance = 0.5  # ±0.5°C tolerance
        
        unmet_heat = sum(1 for t in hourly_temperatures if t < heating_setpoint - tolerance)
        unmet_cool = sum(1 for t in hourly_temperatures if t > cooling_setpoint + tolerance)
        
        self.metrics.unmet_heating_hours = float(unmet_heat)
        self.metrics.unmet_cooling_hours = float(unmet_cool)
        self.metrics.total_unmet_hours = float(unmet_heat + unmet_cool)
        
        # Calculate annual energy cost
        self._calculate_energy_cost()
        
        return self.metrics
    
    def _calculate_energy_cost(self):
        """Calculate annual energy cost based on rates."""
        # Assume electricity powers everything except heating (which uses natural gas)
        self.metrics.annual_energy_cost_usd = (
            self.metrics.total_energy_kwh * self.metrics.electricity_rate_usd_kwh
        )
    
    def add_baseline_metrics(
        self,
        baseline_metrics: "ComplianceMetrics",
        proposed_metrics: "ComplianceMetrics"
    ) -> Dict[str, Any]:
        """
        Create a comparison between baseline and proposed designs.
        
        This generates the data needed for the ASHRAE 90.1 Appendix G compliance table.
        
        Args:
            baseline_metrics: Metrics for the baseline design
            proposed_metrics: Metrics for the proposed design
        
        Returns:
            Dictionary with comparison data
        """
        comparison = {
            "building_area_m2": proposed_metrics.building_area_m2,
            "climate_zone": proposed_metrics.climate_zone,
            "building_type": proposed_metrics.building_type,
            "baseline": {
                "total_energy_kwh": baseline_metrics.total_energy_kwh,
                "total_eui_kwh_m2": baseline_metrics.total_eui_kwh_m2,
                "annual_cost_usd": baseline_metrics.annual_energy_cost_usd,
                "peak_heating_kw": baseline_metrics.peak_heating_load_kw,
                "peak_cooling_kw": baseline_metrics.peak_cooling_load_kw,
                "unmet_hours": baseline_metrics.total_unmet_hours,
            },
            "proposed": {
                "total_energy_kwh": proposed_metrics.total_energy_kwh,
                "total_eui_kwh_m2": proposed_metrics.total_eui_kwh_m2,
                "annual_cost_usd": proposed_metrics.annual_energy_cost_usd,
                "peak_heating_kw": proposed_metrics.peak_heating_load_kw,
                "peak_cooling_kw": proposed_metrics.peak_cooling_load_kw,
                "unmet_hours": proposed_metrics.total_unmet_hours,
            },
            "performance_improvement": {},
        }
        
        # Calculate improvements
        if baseline_metrics.total_energy_kwh > 0:
            comparison["performance_improvement"]["energy_reduction_percent"] = (
                (baseline_metrics.total_energy_kwh - proposed_metrics.total_energy_kwh) /
                baseline_metrics.total_energy_kwh * 100
            )
        
        if baseline_metrics.annual_energy_cost_usd > 0:
            comparison["performance_improvement"]["cost_savings_percent"] = (
                (baseline_metrics.annual_energy_cost_usd - proposed_metrics.annual_energy_cost_usd) /
                baseline_metrics.annual_energy_cost_usd * 100
            )
            comparison["performance_improvement"]["annual_savings_usd"] = (
                baseline_metrics.annual_energy_cost_usd - proposed_metrics.annual_energy_cost_usd
            )
        
        return comparison
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "building": {
                "name": self.metrics.building_name,
                "area_m2": self.metrics.building_area_m2,
                "type": self.metrics.building_type,
                "climate_zone": self.metrics.climate_zone,
            },
            "annual_energy": {
                "total_kwh": self.metrics.total_energy_kwh,
                "electricity_kwh": self.metrics.total_electricity_kwh,
                "natural_gas_kwh": self.metrics.total_natural_gas_kwh,
                "eui_kwh_m2": self.metrics.total_eui_kwh_m2,
            },
            "peak_loads": {
                "heating_kw": self.metrics.peak_heating_load_kw,
                "cooling_kw": self.metrics.peak_cooling_load_kw,
                "electric_demand_kw": self.metrics.peak_electric_demand_kw,
            },
            "unmet_hours": {
                "heating": self.metrics.unmet_heating_hours,
                "cooling": self.metrics.unmet_cooling_hours,
                "total": self.metrics.total_unmet_hours,
            },
            "end_uses": {
                "heating_kwh": self.metrics.heating_energy_kwh,
                "cooling_kwh": self.metrics.cooling_energy_kwh,
                "lighting_kwh": self.metrics.lighting_energy_kwh,
                "plug_loads_kwh": self.metrics.plug_loads_kwh,
                "ventilation_kwh": self.metrics.ventilation_energy_kwh,
            },
            "financial": {
                "annual_cost_usd": self.metrics.annual_energy_cost_usd,
                "electricity_rate_usd_kwh": self.metrics.electricity_rate_usd_kwh,
                "gas_rate_usd_kwh": self.metrics.gas_rate_usd_kwh,
            },
            "simulation": {
                "hours": self.metrics.simulation_hours,
                "timestep_hours": self.metrics.timestep_hours,
            },
        }
    
    def to_json(self, filepath: Optional[Path] = None) -> str:
        """
        Export metrics as JSON string.
        
        Args:
            filepath: Optional path to save JSON file
        
        Returns:
            JSON string
        """
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)
        
        if filepath:
            filepath.write_text(json_str)
        
        return json_str


def create_sample_metrics() -> ComplianceMetrics:
    """
    Create sample compliance metrics for testing.
    
    Returns:
        ComplianceMetrics with sample data
    """
    import random
    random.seed(42)
    
    # Create sample hourly data (one year = 8760 hours)
    n_hours = 8760
    
    # Simulate typical commercial building temperature profile
    base_temp = 22.0
    hourly_temps = [
        base_temp + 2 * (1 if h % 24 < 8 or h % 24 > 18 else 0)  # Night setback
        for h in range(n_hours)
    ]
    
    # Add some random variation
    hourly_temps = [t + random.uniform(-0.5, 0.5) for t in hourly_temps]
    
    # Heating loads in Watts (typical commercial building ~50-100 W/m²)
    # For 1000 m² building: 50,000-100,000 W = 50-100 kW
    hourly_heating = [
        max(0, 50000 + 30000 * (1 if h % 24 < 8 or h % 24 > 18 else 0))  # W
        for h in range(n_hours)
    ]
    
    # Cooling loads in Watts (typical commercial building ~40-80 W/m²)
    hourly_cooling = [
        -max(0, 40000 + 20000 * (1 if h % 24 > 8 and h % 24 < 18 else 0))  # W (negative)
        for h in range(n_hours)
    ]
    
    # Create aggregator
    aggregator = ComplianceDataAggregator(
        building_area_m2=1000.0,
        building_type="Commercial"
    )
    aggregator.metrics.building_name = "Sample Office Building"
    aggregator.metrics.climate_zone = "4A"
    
    # Process results
    return aggregator.process_simulation_results(
        hourly_temperatures=hourly_temps,
        hourly_heating_loads=hourly_heating,
        hourly_cooling_loads=hourly_cooling,
    )


if __name__ == "__main__":
    # Demo
    metrics = create_sample_metrics()
    print(json.dumps(metrics.__dict__, indent=2, default=list))
