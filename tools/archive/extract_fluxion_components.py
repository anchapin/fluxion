"""
Extract Component Data from Fluxion Simulation Logs.

This module parses Fluxion simulation output logs to extract component-level
heat balance data for comparison with EnergyPlus.

Supported components:
- CTF conductive flux (per surface)
- Convective heat transfer (per surface)
- Solar heat gain distribution (per surface)
- HVAC power (per zone)
- Zone air temperature (per zone)
- Surface temperatures (per surface)
- Infiltration heat transfer

Usage:
    python -m tools.extract_fluxion_components \
        --log fluxion_output.log \
        --output fluxion_components.csv
"""

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ComponentData:
    """Data for a single component at a single timestep."""

    timestep: int
    hour: float
    component_type: str  # ctf_flux, convective_flux, solar_flux, hvac_power, etc.
    zone_idx: int
    surface_idx: Optional[int]  # For per-surface data
    value: float
    units: str
    metadata: Dict[str, float] = field(default_factory=dict)


class FluxionLogParser:
    """Parser for Fluxion simulation logs with diagnostic output."""

    # Regex patterns for different diagnostic outputs
    PATTERNS = {
        # CTF_FLUX,t=0,surf=0,q_w_m2=12.34
        "ctf_flux": re.compile(r"CTF_FLUX,t=(\d+),surf=(\d+),q_w_m2=([+-]?\d+\.?\d*)"),
        # CONV_FLUX,t=0,surf=0,h_c=3.50,t_surf=20.5,t_zone=21.0,q_w_m2=1.75
        "convective_flux": re.compile(
            r"CONV_FLUX,t=(\d+),surf=(\d+),h_c=([+-]?\d+\.?\d*),t_surf=([+-]?\d+\.?\d*),"
            r"t_zone=([+-]?\d+\.?\d*),q_w_m2=([+-]?\d+\.?\d*)"
        ),
        # SOLAR,t=0,zone=0,total_w=1234.5,floor_w_m2=50.0,roof_w_m2=0.0,wall_avg_w_m2=10.0
        "solar_distribution": re.compile(
            r"SOLAR,t=(\d+),zone=(\d+),total_w=([+-]?\d+\.?\d*),"
            r"floor_w_m2=([+-]?\d+\.?\d*),roof_w_m2=([+-]?\d+\.?\d*),"
            r"wall_avg_w_m2=([+-]?\d+\.?\d*)"
        ),
        # HVAC_POWER,t=0,zone=0,power_w=1234.5,t_free=20.0,setpoint_heat=20.0,
        # setpoint_cool=25.0,sensitivity=0.002
        "hvac_power": re.compile(
            r"HVAC_POWER,t=(\d+),zone=(\d+),power_w=([+-]?\d+\.?\d*),"
            r"t_free=([+-]?\d+\.?\d*),setpoint_heat=([+-]?\d+\.?\d*),"
            r"setpoint_cool=([+-]?\d+\.?\d*),sensitivity=([+-]?\d+\.?\d*)"
        ),
        # SURF_TEMP,t=0,south_wall=20.5,floor=21.0,ceiling=20.0,avg=20.3
        "surface_temps": re.compile(
            r"SURF_TEMP,t=(\d+),south_wall=([+-]?\d+\.?\d*),"
            r"floor=([+-]?\d+\.?\d*),ceiling=([+-]?\d+\.?\d*),avg=([+-]?\d+\.?\d*)"
        ),
        # INFIL,t=0,t_out=10.0,t_zone=20.0,ach=0.50,q_w=123.4
        "infiltration": re.compile(
            r"INFIL,t=(\d+),t_out=([+-]?\d+\.?\d*),t_zone=([+-]?\d+\.?\d*),"
            r"ach=([+-]?\d+\.?\d*),q_w=([+-]?\d+\.?\d*)"
        ),
        # ZONE_BALANCE,t=0,q_conv=123.4,q_hvac=456.7,q_infil=89.0,
        # q_internal=12.3,residual=0.001
        "zone_balance": re.compile(
            r"ZONE_BALANCE,t=(\d+),q_conv=([+-]?\d+\.?\d*),"
            r"q_hvac=([+-]?\d+\.?\d*),q_infil=([+-]?\d+\.?\d*),"
            r"q_internal=([+-]?\d+\.?\d*),residual=([+-]?\d+\.?\d*)"
        ),
    }

    def __init__(self):
        """Initialize parser."""
        self.data: List[ComponentData] = []
        self.metadata: Dict[str, dict] = {}

    def parse_file(self, log_path: Path) -> List[ComponentData]:
        """Parse a Fluxion log file.

        Args:
            log_path: Path to log file

        Returns:
            List of ComponentData objects
        """
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                self._parse_line(line)

        return self.data

    def _parse_line(self, line: str):
        """Parse a single log line."""
        line = line.strip()

        for component_type, pattern in self.PATTERNS.items():
            match = pattern.match(line)
            if match:
                self._handle_match(component_type, match)
                break

    def _handle_match(self, component_type: str, match: re.Match):
        """Handle a regex match by creating ComponentData."""
        groups = match.groups()

        if component_type == "ctf_flux":
            timestep = int(groups[0])
            surface_idx = int(groups[1])
            value = float(groups[2])

            self.data.append(
                ComponentData(
                    timestep=timestep,
                    hour=timestep / 6.0,  # Assuming 10-minute timesteps
                    component_type="ctf_flux",
                    zone_idx=0,  # CTF is per-surface, zone inferred from surface
                    surface_idx=surface_idx,
                    value=value,
                    units="W/m²",
                )
            )

        elif component_type == "convective_flux":
            timestep = int(groups[0])
            surface_idx = int(groups[1])
            h_c = float(groups[2])
            t_surf = float(groups[3])
            t_zone = float(groups[4])
            value = float(groups[5])

            data = ComponentData(
                timestep=timestep,
                hour=timestep / 6.0,
                component_type="convective_flux",
                zone_idx=0,
                surface_idx=surface_idx,
                value=value,
                units="W/m²",
            )
            data.metadata = {
                "h_c": h_c,
                "t_surf": t_surf,
                "t_zone": t_zone,
            }
            self.data.append(data)

        elif component_type == "solar_distribution":
            timestep = int(groups[0])
            zone_idx = int(groups[1])
            total_w = float(groups[2])
            floor_w_m2 = float(groups[3])
            roof_w_m2 = float(groups[4])
            wall_avg_w_m2 = float(groups[5])

            # Create separate entries for each surface type
            for surf_idx, (surf_name, surf_value) in enumerate(
                [
                    ("floor", floor_w_m2),
                    ("roof", roof_w_m2),
                    ("wall_avg", wall_avg_w_m2),
                ]
            ):
                data = ComponentData(
                    timestep=timestep,
                    hour=timestep / 6.0,
                    component_type="solar_flux",
                    zone_idx=zone_idx,
                    surface_idx=surf_idx,
                    value=surf_value,
                    units="W/m²",
                )
                data.metadata = {
                    "total_w": total_w,
                    "surface_type": surf_name,
                }
                self.data.append(data)

        elif component_type == "hvac_power":
            timestep = int(groups[0])
            zone_idx = int(groups[1])
            power_w = float(groups[2])
            t_free = float(groups[3])
            setpoint_heat = float(groups[4])
            setpoint_cool = float(groups[5])
            sensitivity = float(groups[6])

            data = ComponentData(
                timestep=timestep,
                hour=timestep / 6.0,
                component_type="hvac_power",
                zone_idx=zone_idx,
                surface_idx=None,
                value=power_w,
                units="W",
            )
            data.metadata = {
                "t_free": t_free,
                "setpoint_heat": setpoint_heat,
                "setpoint_cool": setpoint_cool,
                "sensitivity": sensitivity,
            }
            self.data.append(data)

        elif component_type == "surface_temps":
            timestep = int(groups[0])
            south_wall = float(groups[1])
            floor = float(groups[2])
            ceiling = float(groups[3])
            avg = float(groups[4])

            # Create entries for each surface
            for surf_idx, (surf_name, temp) in enumerate(
                [
                    ("south_wall", south_wall),
                    ("floor", floor),
                    ("ceiling", ceiling),
                ]
            ):
                data = ComponentData(
                    timestep=timestep,
                    hour=timestep / 6.0,
                    component_type="surface_temp",
                    zone_idx=0,
                    surface_idx=surf_idx,
                    value=temp,
                    units="°C",
                )
                data.metadata = {
                    "surface_type": surf_name,
                    "avg_temp": avg,
                }
                self.data.append(data)

        elif component_type == "infiltration":
            timestep = int(groups[0])
            t_out = float(groups[1])
            t_zone = float(groups[2])
            ach = float(groups[3])
            q_w = float(groups[4])

            data = ComponentData(
                timestep=timestep,
                hour=timestep / 6.0,
                component_type="infiltration",
                zone_idx=0,
                surface_idx=None,
                value=q_w,
                units="W",
            )
            data.metadata = {
                "t_out": t_out,
                "t_zone": t_zone,
                "ach": ach,
            }
            self.data.append(data)

        elif component_type == "zone_balance":
            timestep = int(groups[0])
            q_conv = float(groups[1])
            q_hvac = float(groups[2])
            q_infil = float(groups[3])
            q_internal = float(groups[4])
            residual = float(groups[5])

            data = ComponentData(
                timestep=timestep,
                hour=timestep / 6.0,
                component_type="zone_balance_residual",
                zone_idx=0,
                surface_idx=None,
                value=residual,
                units="W",
            )
            data.metadata = {
                "q_conv": q_conv,
                "q_hvac": q_hvac,
                "q_infil": q_infil,
                "q_internal": q_internal,
            }
            self.data.append(data)

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame conversion")

        records = []
        for d in self.data:
            record = {
                "timestep": d.timestep,
                "hour": d.hour,
                "component_type": d.component_type,
                "zone_idx": d.zone_idx,
                "surface_idx": d.surface_idx if d.surface_idx is not None else -1,
                "value": d.value,
                "units": d.units,
            }
            # Add metadata fields
            for k, v in d.metadata.items():
                record[f"meta_{k}"] = v
            records.append(record)

        return pd.DataFrame(records)

    def to_csv(self, output_path: Path):
        """Export to CSV file.

        Args:
            output_path: Output CSV file path
        """
        if not self.data:
            print("No data to export")
            return

        # Get all metadata keys
        all_meta_keys = set()
        for d in self.data:
            all_meta_keys.update(d.metadata.keys())

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            # Write header
            header = [
                "timestep",
                "hour",
                "component_type",
                "zone_idx",
                "surface_idx",
                "value",
                "units",
            ] + [f"meta_{k}" for k in sorted(all_meta_keys)]
            writer = csv.writer(f)
            writer.writerow(header)

            # Write data
            for d in self.data:
                row = [
                    d.timestep,
                    d.hour,
                    d.component_type,
                    d.zone_idx,
                    d.surface_idx if d.surface_idx is not None else "",
                    d.value,
                    d.units,
                ]
                # Add metadata
                for k in sorted(all_meta_keys):
                    row.append(d.metadata.get(k, ""))
                writer.writerow(row)

        print(f"Exported {len(self.data)} records to {output_path}")

    def get_summary(self) -> dict:
        """Get summary statistics of parsed data."""
        from collections import defaultdict

        summary = {
            "total_records": len(self.data),
            "by_component": defaultdict(int),
            "timestep_range": (0, 0),
            "zones": set(),
        }

        if not self.data:
            return summary

        min_ts = min(d.timestep for d in self.data)
        max_ts = max(d.timestep for d in self.data)
        summary["timestep_range"] = (min_ts, max_ts)

        for d in self.data:
            summary["by_component"][d.component_type] += 1
            summary["zones"].add(d.zone_idx)

        summary["by_component"] = dict(summary["by_component"])
        summary["zones"] = sorted(summary["zones"])

        return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract component data from Fluxion simulation logs"
    )
    parser.add_argument(
        "--log", "-l", type=Path, required=True, help="Path to Fluxion log file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default="fluxion_components.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose output"
    )

    args = parser.parse_args()

    if not args.log.exists():
        print(f"Error: Log file not found: {args.log}")
        return 1

    # Parse log file
    print(f"Parsing Fluxion log: {args.log}")
    parser_obj = FluxionLogParser()
    data = parser_obj.parse_file(args.log)

    if not data:
        print("Warning: No component data found in log file")
        print("Ensure diagnostic logging is enabled in Fluxion")
        return 1

    # Export to CSV
    parser_obj.to_csv(args.output)

    # Print summary
    if args.verbose:
        summary = parser_obj.get_summary()
        print("\nSummary:")
        print(f"  Total records: {summary['total_records']}")
        print(f"  Timestep range: {summary['timestep_range']}")
        print(f"  Zones: {summary['zones']}")
        print("  By component:")
        for comp, count in summary["by_component"].items():
            print(f"    {comp}: {count}")

    return 0


if __name__ == "__main__":
    exit(main())
