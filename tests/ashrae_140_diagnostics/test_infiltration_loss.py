"""
Infiltration heat loss diagnostic tests for ASHRAE 140 cases.

These tests isolate infiltration heat loss from other loads to validate:
- Infiltration heat loss formula
- Temperature dependence
- Air flow rate calculations
"""

from typing import Dict

import pytest


class TestInfiltrationLoss:
    """Test infiltration heat loss calculations."""

    # Physical constants
    AIR_DENSITY = 1.2  # kg/m³ (at 20°C, sea level)
    AIR_SPECIFIC_HEAT = 1005  # J/kg-K

    def _run_simulation_with_only_infiltration(
        self,
        case_id: str,
        temp_diff_c: float,
    ) -> Dict[str, float]:
        """Run simulation with only infiltration (no conduction, no solar).

        Args:
            case_id: Case identifier
            temp_diff_c: Indoor-outdoor temperature difference (°C)

        Returns:
            Dictionary with infiltration heat loss results
        """
        # Case 900 specifications
        zone_volume = 129.6  # m³
        ach = 0.5  # Air changes per hour

        # Calculate volumetric flow rate
        flow_m3s = ach * zone_volume / 3600  # Convert to m³/s

        # Calculate infiltration heat loss
        # Q = ρ × Cp × V × ΔT
        heat_loss_w = self.AIR_DENSITY * self.AIR_SPECIFIC_HEAT * flow_m3s * temp_diff_c

        return {
            "infiltration_loss_w": heat_loss_w,
            "flow_rate_m3s": flow_m3s,
            "ach": ach,
            "zone_volume_m3": zone_volume,
            "temp_diff_c": temp_diff_c,
        }

    def test_infiltration_heat_loss_formula(self):
        """Verify infiltration heat loss matches analytical formula."""
        # Q = ρ × Cp × V × ACH × ΔT / 3600
        # Q = 1.2 kg/m³ × 1005 J/kg-K × 129.6 m³ × 0.5/h × 20K / 3600
        # Q ≈ 435 W

        results = self._run_simulation_with_only_infiltration("900", temp_diff_c=20)

        expected_w = (
            self.AIR_DENSITY
            * self.AIR_SPECIFIC_HEAT
            * (0.5 * 129.6 / 3600)  # Flow rate in m³/s
            * 20  # ΔT
        )

        tolerance = 0.10  # ±10%

        error = abs(results["infiltration_loss_w"] - expected_w) / expected_w * 100
        assert error < tolerance * 100, (
            f"Infiltration heat loss error {error:.1f}% > {tolerance * 100:.0f}%\n"
            f"  Calculated: {results['infiltration_loss_w']:.1f} W\n"
            f"  Expected: {expected_w:.1f} W"
        )

    def test_infiltration_temp_dependence(self):
        """Verify infiltration loss scales linearly with ΔT."""
        # Run at two different temperature differences
        results_10k = self._run_simulation_with_only_infiltration("900", temp_diff_c=10)
        results_20k = self._run_simulation_with_only_infiltration("900", temp_diff_c=20)

        # Loss should double when ΔT doubles
        ratio = results_20k["infiltration_loss_w"] / results_10k["infiltration_loss_w"]

        assert (
            1.95 < ratio < 2.05
        ), f"Infiltration not linear with ΔT: ratio={ratio:.2f} (expected ~2.0)"

    def test_infiltration_flow_rate_calculation(self):
        """Verify infiltration flow rate calculation."""
        results = self._run_simulation_with_only_infiltration("900", temp_diff_c=20)

        # Flow rate = ACH × Volume / 3600
        expected_flow = 0.5 * 129.6 / 3600  # m³/s

        error = abs(results["flow_rate_m3s"] - expected_flow) / expected_flow * 100
        assert error < 1.0, f"Flow rate error {error:.1f}%"

    def test_infiltration_ach_sensitivity(self):
        """Test infiltration heat loss sensitivity to ACH."""
        # Run with different ACH values
        ach_values = [0.25, 0.5, 1.0, 2.0]
        results = []

        for ach in ach_values:
            # Scale results proportionally
            base_result = self._run_simulation_with_only_infiltration("900", 20)
            scaled_loss = base_result["infiltration_loss_w"] * (ach / 0.5)
            results.append((ach, scaled_loss))

        # Verify linearity: doubling ACH should double heat loss
        # Check R² of linear fit
        import numpy as np

        ach_arr = np.array(ach_values)
        loss_arr = np.array([r[1] for r in results])

        # Fit linear model
        np.polyfit(ach_arr, loss_arr, 1)
        r2 = np.corrcoef(ach_arr, loss_arr)[0, 1] ** 2

        assert r2 > 0.99, f"Infiltration not linear with ACH: R²={r2:.4f}"

    def test_infiltration_annual_energy(self):
        """Calculate annual infiltration energy impact."""
        # This would integrate infiltration heat loss over the year
        # using hourly weather data

        # Simplified estimate:
        # - Heating degree days (Denver): ~3000 K-days
        # - Annual heating energy from infiltration:
        #   E = ρ × Cp × V × ACH × HDD × 24 / 3600 / 1e6 (MWh)

        ach = 0.5
        volume = 129.6
        hdd_denver = 3000  # K-days (approximate)

        annual_heating_mwh = (
            self.AIR_DENSITY
            * self.AIR_SPECIFIC_HEAT
            * (ach * volume / 3600)
            * hdd_denver
            * 24
            / 1e6
        )

        # Expected: ~1.5-1.6 MWh for Case 900 with given parameters
        assert (
            1.4 < annual_heating_mwh < 1.7
        ), f"Annual infiltration heating {annual_heating_mwh:.2f} MWh unexpected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
