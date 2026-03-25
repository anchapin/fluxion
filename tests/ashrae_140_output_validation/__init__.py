"""
ASHRAE 140 Output Validation Tests.

This module compares Fluxion simulation outputs against EnergyPlus
reference results at multiple levels:
- Annual energy (heating, cooling, total)
- Hourly temperatures (RMSE, NMBE, CV-RMSE, R²)
- Monthly energy profiles
- Peak loads (magnitude and timing)
"""

__version__ = "1.0.0"
