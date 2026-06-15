#!/usr/bin/env python3
"""
Dummy Thermostat FMU for FMI Co-Simulation Testing

This module provides a simple FMU-compatible thermostat that:
- Reads zone temperature from Fluxion
- Computes heating/cooling demand based on setpoints
- Returns heating/cooling signals to Fluxion

This is a lightweight stub for testing co-simulation without a full FMU.
"""

import json
import sys
from typing import Dict


class DummyThermostatFMU:
    """
    A simple thermostat FMU stub that implements FMI 2.0 co-simulation interface.

    Input variables:
        - zone_temperature (K): Current zone air temperature
        - outdoor_temperature (K): Current outdoor air temperature

    Output variables:
        - heating_signal (W): Heating demand (>0 when heating needed)
        - cooling_signal (W): Cooling demand (>0 when cooling needed)
        - energy_balance (W): Energy balance check (should be ~0)

    The thermostat implements a simple deadband control:
        - Heating activates when zone temp < heating_setpoint - deadband/2
        - Cooling activates when zone temp > cooling_setpoint + deadband/2
    """

    def __init__(
        self,
        heating_setpoint: float = 293.15,  # 20°C in K
        cooling_setpoint: float = 299.15,  # 26°C in K
        deadband: float = 2.0,  # 2 K deadband
        heating_capacity: float = 10000.0,  # 10 kW max heating
        cooling_capacity: float = 10000.0,  # 10 kW max cooling
    ):
        self.heating_setpoint = heating_setpoint
        self.cooling_setpoint = cooling_setpoint
        self.deadband = deadband
        self.heating_capacity = heating_capacity
        self.cooling_capacity = cooling_capacity

        # Internal state
        self.current_time: float = 0.0
        self.heating_signal: float = 0.0
        self.cooling_signal: float = 0.0

        # Track energy for balance checking
        self.total_heating_energy: float = 0.0
        self.total_cooling_energy: float = 0.0

        # Input values from Fluxion
        self.zone_temperature: float = 293.15
        self.outdoor_temperature: float = 283.15

    def setup_experiment(self, start_time: float, stop_time: float) -> None:
        """Called at simulation start to set time boundaries."""
        self.current_time = start_time
        print(
            f"[FMU] setup_experiment: start={start_time}, stop={stop_time}",
            file=sys.stderr,
        )

    def enter_initialization_mode(self) -> None:
        """Called to enter initialization mode."""
        print("[FMU] enter_initialization_mode", file=sys.stderr)
        self.heating_signal = 0.0
        self.cooling_signal = 0.0

    def exit_initialization_mode(self) -> None:
        """Called to exit initialization mode."""
        print("[FMU] exit_initialization_mode", file=sys.stderr)

    def do_step(
        self, current_time: float, step_size: float, no_step_prior: bool = False
    ) -> bool:
        """
        Execute one co-simulation timestep.

        Args:
            current_time: Current simulation time in seconds
            step_size: Timestep size in seconds
            no_step_prior: Whether step was rejected previously

        Returns:
            True if step was successful
        """
        self.current_time = current_time

        # Compute control signals based on zone temperature
        heating_demand = self._compute_heating_demand()
        cooling_demand = self._compute_cooling_demand()

        # Apply limits
        self.heating_signal = min(heating_demand, self.heating_capacity)
        self.cooling_signal = min(cooling_demand, self.cooling_capacity)

        # Track energy (in joules = watts * seconds)
        self.total_heating_energy += self.heating_signal * step_size
        self.total_cooling_energy += self.cooling_signal * step_size

        print(
            f"[FMU] do_step: t={current_time:.0f}s, dt={step_size:.0f}s, "
            f"zone_temp={self.zone_temperature:.2f}K, "
            f"heating={self.heating_signal:.2f}W, cooling={self.cooling_signal:.2f}W",
            file=sys.stderr,
        )

        return True

    def _compute_heating_demand(self) -> float:
        """Compute heating demand based on zone temperature and deadband."""
        if self.zone_temperature < (self.heating_setpoint - self.deadband / 2):
            # Temperature below heating setpoint - need heating
            error = self.heating_setpoint - self.zone_temperature
            # Proportional control with max at deadband
            demand = (error / self.deadband) * self.heating_capacity
            return max(0.0, demand)
        return 0.0

    def _compute_cooling_demand(self) -> float:
        """Compute cooling demand based on zone temperature and deadband."""
        if self.zone_temperature > (self.cooling_setpoint + self.deadband / 2):
            # Temperature above cooling setpoint - need cooling
            error = self.zone_temperature - self.cooling_setpoint
            # Proportional control with max at deadband
            demand = (error / self.deadband) * self.cooling_capacity
            return max(0.0, demand)
        return 0.0

    def get_real(self, vr: int) -> float:
        """
        Get value of a real variable by value reference.

        FMI 2.0 variable references:
            0: zone_temperature (input)
            1: outdoor_temperature (input)
            2: heating_signal (output)
            3: cooling_signal (output)
            4: energy_balance (output)
        """
        if vr == 0:
            return self.zone_temperature
        elif vr == 1:
            return self.outdoor_temperature
        elif vr == 2:
            return self.heating_signal
        elif vr == 3:
            return self.cooling_signal
        elif vr == 4:
            # Energy balance = heating - cooling (should be ~0 for ideal thermostat)
            return self.heating_signal - self.cooling_signal
        else:
            raise ValueError(f"Unknown value reference: {vr}")

    def set_real(self, vr: int, value: float) -> None:
        """
        Set value of a real variable by value reference.
        """
        if vr == 0:
            self.zone_temperature = value
        elif vr == 1:
            self.outdoor_temperature = value
        else:
            raise ValueError(f"Cannot set output variable with vr={vr}")

    def get_energy_balance(self) -> Dict[str, float]:
        """Return energy tracking for balance verification."""
        return {
            "total_heating_energy_j": self.total_heating_energy,
            "total_cooling_energy_j": self.total_cooling_energy,
            "net_energy_j": self.total_heating_energy - self.total_cooling_energy,
            "current_heating_signal_w": self.heating_signal,
            "current_cooling_signal_w": self.cooling_signal,
        }


def main():
    """
    Simple FMU stub runner for testing.

    Reads commands from stdin and writes responses to stdout.
    Protocol is JSON-based for simplicity in testing.
    """
    fmu = DummyThermostatFMU()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            command = json.loads(line.strip())
            action = command.get("action")

            if action == "setup_experiment":
                fmu.setup_experiment(
                    command.get("start_time", 0.0), command.get("stop_time", 31536000.0)
                )
                print(json.dumps({"status": "ok"}))

            elif action == "enter_initialization_mode":
                fmu.enter_initialization_mode()
                print(json.dumps({"status": "ok"}))

            elif action == "exit_initialization_mode":
                fmu.exit_initialization_mode()
                print(json.dumps({"status": "ok"}))

            elif action == "do_step":
                success = fmu.do_step(
                    command.get("current_time", 0.0),
                    command.get("step_size", 3600.0),
                    command.get("no_step_prior", False),
                )
                print(
                    json.dumps(
                        {
                            "status": "ok" if success else "failed",
                            "heating_signal": fmu.heating_signal,
                            "cooling_signal": fmu.cooling_signal,
                        }
                    )
                )

            elif action == "get_real":
                value = fmu.get_real(command.get("vr", 0))
                print(json.dumps({"status": "ok", "value": value}))

            elif action == "set_real":
                fmu.set_real(command.get("vr", 0), command.get("value", 0.0))
                print(json.dumps({"status": "ok"}))

            elif action == "get_energy_balance":
                balance = fmu.get_energy_balance()
                print(json.dumps({"status": "ok", "balance": balance}))

            elif action == "quit":
                print(json.dumps({"status": "ok"}))
                break

            else:
                print(
                    json.dumps(
                        {"status": "error", "message": f"Unknown action: {action}"}
                    )
                )

        except json.JSONDecodeError:
            print(json.dumps({"status": "error", "message": "Invalid JSON"}))
        except Exception as e:
            print(json.dumps({"status": "error", "message": str(e)}))


if __name__ == "__main__":
    main()
