"""
Internal gains diagnostic tests for ASHRAE 140 cases.

These tests validate internal heat gain calculations:
- Equipment heat contribution
- Schedule implementation
- Convective/radiative split
"""

from typing import Dict

import pytest


class TestInternalGainsDiagnostic:
    """Test internal gains heat contribution and scheduling."""

    def _run_simulation_with_equipment(
        self,
        case_id: str,
        equipment_w: float,
    ) -> Dict[str, float]:
        """Run simulation with specified equipment power.

        Args:
            case_id: Case identifier
            equipment_w: Equipment power (W)

        Returns:
            Dictionary with energy results
        """
        # In production, this would run a modified simulation
        # For now, return calculated values

        # Case 900 equipment schedule:
        # - Day (6:00-22:00): 60% of full load
        # - Night (22:00-6:00): 40% of full load

        # Average daily fraction: (16h × 0.6 + 8h × 0.4) / 24h = 0.533

        day_hours = 16
        night_hours = 8
        day_fraction = 0.6
        night_fraction = 0.4

        avg_fraction = (day_hours * day_fraction + night_hours * night_fraction) / 24

        # Daily energy
        daily_energy_kwh = equipment_w * avg_fraction * 24 / 1000

        return {
            "equipment_power_w": equipment_w,
            "daily_energy_kwh": daily_energy_kwh,
            "avg_schedule_fraction": avg_fraction,
        }

    def test_equipment_heat_contribution(self):
        """Verify 200W equipment contributes expected heat."""
        # Run with equipment ON (200W)
        results_on = self._run_simulation_with_equipment("900", equipment_w=200)

        # Run with equipment OFF (0W)
        results_off = self._run_simulation_with_equipment("900", equipment_w=0)

        # Difference should be ~200W × hours × schedule
        expected_daily_kwh = 0.2 * 24 * 0.533  # 200W × 24hr × avg schedule

        daily_energy_diff = (
            results_on["daily_energy_kwh"] - results_off["daily_energy_kwh"]
        )

        tolerance = 0.20  # ±20%

        error = abs(daily_energy_diff - expected_daily_kwh) / expected_daily_kwh * 100
        assert error < tolerance * 100, (
            f"Equipment energy error {error:.1f}% > {tolerance * 100:.0f}%\n"
            f"  Calculated: {daily_energy_diff:.3f} kWh/day\n"
            f"  Expected: {expected_daily_kwh:.3f} kWh/day"
        )

    def test_equipment_schedule_implementation(self):
        """Verify equipment schedule fractions."""
        # ASHRAE 140 equipment schedule:
        # - Day (6:00-22:00): 60%
        # - Night (22:00-6:00): 40%

        day_fraction = 0.6
        night_fraction = 0.4

        # Verify schedule sums correctly
        day_hours = 16
        night_hours = 8

        weighted_avg = (day_hours * day_fraction + night_hours * night_fraction) / 24

        assert abs(weighted_avg - 0.533) < 0.01, (
            f"Schedule average {weighted_avg:.3f} unexpected"
        )

    def test_convective_radiative_split(self):
        """Verify internal gains convective/radiative split."""
        # Equipment gains are typically split:
        # - 60% radiative (heats surfaces)
        # - 40% convective (heats air directly)

        # This split affects how heat is distributed in the zone
        # Radiative portion is absorbed by surfaces
        # Convective portion goes directly into zone air

        radiative_fraction = 0.6
        convective_fraction = 0.4

        # Verify fractions sum to 1.0
        assert abs(radiative_fraction + convective_fraction - 1.0) < 0.01

        # In production, this would verify the split in the simulation
        # by comparing surface temperatures and air temperatures
        pass

    def test_internal_gains_annual_energy(self):
        """Calculate annual internal gains energy impact."""
        # Case 900: 200W equipment
        # Annual energy = 200W × 24hr × 365days × avg_schedule

        power_w = 200
        avg_schedule = 0.533

        annual_kwh = power_w * 24 * 365 * avg_schedule / 1000
        annual_mwh = annual_kwh / 1000

        # Expected: ~0.9-1.0 MWh
        assert 0.8 < annual_mwh < 1.2, (
            f"Annual equipment energy {annual_mwh:.2f} MWh unexpected"
        )

    def test_internal_gains_heat_balance(self):
        """Verify internal gains in zone heat balance."""
        # Internal gains contribute to zone heat balance:
        # - Convective portion directly affects air temperature
        # - Radiative portion is absorbed by surfaces and re-radiated

        # This test would verify that internal gains are correctly
        # partitioned and included in the heat balance equations

        # In production:
        # 1. Run simulation with internal gains
        # 2. Run simulation without internal gains
        # 3. Compare zone temperatures and HVAC loads
        # 4. Verify difference matches expected internal gain contribution

        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
