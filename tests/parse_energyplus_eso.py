#!/usr/bin/env python3
"""
Parse EnergyPlus .eso output files and extract reference data for unit tests.

This script extracts hourly values for key variables:
- Zone Mean Air Temperature
- Zone Air System Sensible Heating/Cooling Energy
- Solar radiation through windows
"""

import sys
import json
from pathlib import Path
from typing import Dict, List


class ESOParser:
    """Parse EnergyPlus .eso (ESO) output files."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.variables: Dict[int, Dict] = {}  # variable ID -> {name, units}
        self.data: Dict[int, List[float]] = {}  # variable ID -> hourly values
        self.metadata: Dict = {}
        self._parse()

    def _parse(self):
        """Parse the .eso file."""
        in_data_section = False

        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip()

                if line.startswith('Program Version'):
                    # Parse version metadata
                    parts = line.split(',')
                    for part in parts:
                        if 'Version' in part:
                            self.metadata['version'] = part.split('Version')[1].strip()
                    for part in parts:
                        if 'YMD=' in part:
                            self.metadata['date'] = part.split('YMD=')[1]
                    continue

                elif line.startswith('End of Data Dictionary'):
                    in_data_section = True
                    continue

                elif not in_data_section and ',' in line and not line.startswith('Program'):
                    # Variable definition line
                    # Format: 20,1,ZONE ONE,Zone Mean Air Temperature [C] !Hourly
                    # Only split on FIRST two commas
                    first_comma = line.find(',')
                    if first_comma == -1:
                        continue
                    second_comma = line.find(',', first_comma + 1)
                    if second_comma == -1:
                        continue

                    # Extract variable ID (first field)
                    try:
                        var_id = int(line[:first_comma])
                    except ValueError:
                        continue

                    # Extract variable name (everything between 2nd and 3rd comma or end of line)
                    var_field_with_units = line[second_comma + 1:]  # Everything after 2nd comma

                    # Split by ! to separate from comment
                    if '!' in var_field_with_units:
                        var_name_units = var_field_with_units.split('!')[0].strip()
                    else:
                        var_name_units = var_field_with_units

                    # Extract units from [Units]
                    if '[' in var_name_units and ']' in var_name_units:
                        # Extract name (before brackets)
                        bracket_pos = var_name_units.rfind('[')
                        var_name = var_name_units[:bracket_pos].strip()
                        # Extract units (between brackets)
                        units = var_name_units[bracket_pos+1:-1].strip()
                    else:
                        var_name = var_name_units.strip()
                        units = ''

                    self.variables[var_id] = {
                        'name': var_name,
                        'units': units
                    }

                elif in_data_section and ',' in line:
                    # Data line
                    # Format: <variable_id>,<value>
                    first_comma = line.find(',')
                    if first_comma == -1:
                        continue
                    var_id_str = line[:first_comma]
                    try:
                        var_id = int(var_id_str)
                        value = float(line[first_comma + 1:])
                    except (ValueError, IndexError):
                        continue

                    if var_id not in self.data:
                        self.data[var_id] = []
                    self.data[var_id].append(value)


def extract_case_900_reference(eso_file: str) -> Dict:
    """
    Extract reference data for Case 900 from EnergyPlus .eso file.

    Returns dictionary with hourly data for key variables.
    """
    parser = ESOParser(eso_file)

    # Extract key variables
    reference_data = {
        'metadata': {
            'file': eso_file,
            'version': parser.metadata.get('version', 'unknown'),
            'date': parser.metadata.get('date', 'unknown'),
        },
        'variables': {},
        'hourly': {}
    }

    # Find variable IDs for key quantities
    for var_id, var_info in parser.variables.items():
        var_name = var_info['name'].lower()

        if 'zone mean air temperature' in var_name:
            reference_data['variables']['zone_air_temp'] = {
                'variable_id': var_id,
                'name': var_info['name'],
                'units': var_info['units']
            }
            if var_id in parser.data:
                reference_data['hourly']['zone_air_temp_c'] = parser.data[var_id]

        elif 'zone air system sensible heating energy' in var_name:
            reference_data['variables']['heating_energy'] = {
                'variable_id': var_id,
                'name': var_info['name'],
                'units': var_info['units']
            }
            if var_id in parser.data:
                # Convert from J to Wh (divide by 3600)
                reference_data['hourly']['heating_energy_wh'] = [
                    val / 3600.0 for val in parser.data[var_id]
                ]

        elif 'zone air system sensible cooling energy' in var_name:
            reference_data['variables']['cooling_energy'] = {
                'variable_id': var_id,
                'name': var_info['name'],
                'units': var_info['units']
            }
            if var_id in parser.data:
                # Convert from J to Wh (divide by 3600)
                reference_data['hourly']['cooling_energy_wh'] = [
                    val / 3600.0 for val in parser.data[var_id]
                ]

        elif 'enclosure windows total transmitted solar radiation rate' in var_name:
            reference_data['variables']['solar_rate_total'] = {
                'variable_id': var_id,
                'name': var_info['name'],
                'units': var_info['units']
            }
            if var_id in parser.data:
                reference_data['hourly']['solar_rate_total_w'] = parser.data[var_id]

    return reference_data


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python parse_energyplus_eso.py <eso_file>")
        print("Example: python parse_energyplus_eso.py benchmarks/outputs/bestest_gsr/case_900/run/eplusout.eso")
        sys.exit(1)

    eso_file = sys.argv[1]
    if not Path(eso_file).exists():
        print(f"Error: File not found: {eso_file}")
        sys.exit(1)

    # Extract reference data
    reference_data = extract_case_900_reference(eso_file)

    # Save to JSON
    output_file = Path(eso_file).parent / "reference_data.json"
    with open(output_file, 'w') as f:
        json.dump(reference_data, f, indent=2)

    print(f"✓ Extracted reference data from {eso_file}")
    print(f"  Saved to {output_file}")
    print(f"\nVariables found: {len(reference_data['variables'])}")
    print(f"  Variables:")
    for var_id, var_info in reference_data['variables'].items():
        print(f"    ID {var_id}: {var_info['name']}")
    print(f"\nHourly data points:")
    for var_name, data in reference_data['hourly'].items():
        print(f"  - {var_name}: {len(data)} values")


if __name__ == '__main__':
    main()
