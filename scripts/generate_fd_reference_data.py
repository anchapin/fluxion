#!/usr/bin/env python3
"""
Generate EnergyPlus reference data for FD step-response tests with controlled 20°C interior.

Each IDF is modified to add:
  - HVACTemplate:Thermostat            (constant 20°C heating, 100°C cooling = no cooling)
  - HVACTemplate:Zone:IdealLoadsAirSystem  (ideal air system controlled by thermostat)

Weather: tests/test_data/denver.epw
Output parsed from eplusout.eso (EnergyPlus standard output format).

Usage:
    python3 scripts/generate_fd_reference_data.py [--idf-dir tests/reference_data/energyplus_models]
"""

import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

EP_BINARY = os.environ.get("ENERGYPLUS", "/usr/local/bin/energyplus")
EP_IDD = "/usr/local/EnergyPlus-25-2-0/Energy+.idd"
EPW_FILE = "tests/test_data/denver.epw"

# CSV column order (matches load_reference_data expectation)
OUTPUT_COLS = ["hour", "T_ext", "T_zone", "T_surface_inside", "T_surface_outside", "q_inside", "q_outside"]

# Map: construction_key → surface_name used in E+
CONSTRUCTIONS = {
    "lightweight": "SOUTHWALL",
    "composite":  "SOUTHWALL",
    "roof":        "ROOF",
    "floor":       "FLOOR",
}

# HVAC template objects injected before Output:Variable in each IDF
HVAC_BLOCK = """HVACTemplate:Thermostat,
  ZONE1_Thermostat,  !- Name
  ,                    !- A2: Heating Setpoint Schedule Name (blank = constant)
  20.0,              !- N1: Constant Heating Setpoint {C}
  ,                    !- A3: Cooling Setpoint Schedule Name (blank = constant)
  100.0;              !- N2: Constant Cooling Setpoint {C}

HVACTemplate:Zone:IdealLoadsAirSystem,
  ZONE1,              !- A1: Zone Name
  ZONE1_Thermostat,  !- A2: Template Thermostat Name
  ,                    !- A3: System Availability Schedule Name
  50,                 !- N1: Max Heating Supply Air Temp {C}
  13;                 !- N2: Min Cooling Supply Air Temp {C}

"""


def expand_and_run_ep(idf_content: str, tmpdir: str) -> bool:
    """Expand HVACTemplate objects and run EnergyPlus. Returns True on success."""
    # Write IDF
    with open(os.path.join(tmpdir, "in.idf"), "w") as f:
        f.write(idf_content)

    # Create Energy+.ini for ExpandObjects
    with open(os.path.join(tmpdir, "Energy+.ini"), "w") as f:
        f.write("[EP]\nIDD File=Energy+.idd\n")

    # Copy IDD and weather
    shutil.copy(EP_IDD, os.path.join(tmpdir, "Energy+.idd"))
    shutil.copy(EPW_FILE, os.path.join(tmpdir, "denver.epw"))

    # Expand HVACTemplate objects - must run from tmpdir so it can write expanded.idf there
    r = subprocess.run(
        [EP_BINARY.replace("energyplus", "ExpandObjects"), os.path.join(tmpdir, "in.idf")],
        capture_output=True, text=True,
        cwd=tmpdir,  # ExpandObjects writes expanded.idf here
        env={"ENERGYPLUS": EP_BINARY, "EP_idd": EP_IDD}
    )
    if r.returncode != 0:
        print(f"  ExpandObjects failed: {r.stdout[:200]}")
        return False

    expanded = os.path.join(tmpdir, "expanded.idf")
    if not os.path.exists(expanded):
        print("  expanded.idf not created")
        return False

    # Run E+
    r = subprocess.run(
        [EP_BINARY, "-d", tmpdir, "-w", os.path.join(tmpdir, "denver.epw"),
         expanded],
        capture_output=True, text=True, cwd=tmpdir,
        env={"ENERGYPLUS": EP_BINARY, "EP_idd": EP_IDD}
    )
    return r.returncode == 0


def parse_eso(eso_path: str) -> list[dict]:
    """Parse eplusout.eso into list of timestep dicts."""
    with open(eso_path) as f:
        lines = f.readlines()

    # Parse dictionary (everything before "End of Data Dictionary")
    var_meta = {}  # idx → {"name": ..., "units": ...}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "End of Data Dictionary" in stripped:
            break
        parts = stripped.split(",")
        if len(parts) >= 3:
            try:
                var_id = int(parts[0].strip())
                # The variable description is in parts[3] (e.g. "Site Outdoor Air Drybulb Temperature [C]")
                # parts[2] is just the key/context (e.g. "Environment", "SOUTHWALL")
                name = parts[3].strip() if len(parts) >= 4 else parts[2].strip()
                var_meta[var_id] = {"name": name, "key": parts[2].strip()}
            except ValueError:
                pass

    # Map variable names to our column names
    var_map = {}  # our_col → var_id
    target_patterns = {
        # By variable name (meta["name"])
        "T_zone":             ["zone mean air temperature"],
        "T_surface_inside":   ["surface inside face temperature"],
        "T_surface_outside":  ["surface outside face temperature"],
        "q_inside":           ["surface inside face conduction heat transfer rate per area"],
        "q_outside":          ["surface outside face conduction heat transfer rate per area"],
    }
    # By key/context name (meta["key"]) — T_ext uses "Environment" key
    # NOTE: exact key match to avoid "Environment Title[]" matching "environment"
    target_keys = {
        "T_ext": ["Environment"],
    }
    for var_id, meta in var_meta.items():
        name_lower = meta["name"].lower()
        for our_col, patterns in target_patterns.items():
            if our_col not in var_map:
                for pat in patterns:
                    if pat in name_lower:
                        var_map[our_col] = var_id
                        break
        # Match by key for remaining columns (exact match to avoid partial matches)
        for our_col, key_patterns in target_keys.items():
            if our_col not in var_map:
                key = meta.get("key", "")
                for pat in key_patterns:
                    if pat.lower() == key.lower():
                        var_map[our_col] = var_id
                        break

    # Parse data (everything after "End of Data Dictionary")
    in_data = False
    rows = {}  # var_id → value (overwritten each timestep)
    result = []
    n_vars = len(var_map)

    for line in lines:
        stripped = line.strip()
        if "End of Data Dictionary" in stripped:
            in_data = True
            continue
        if not in_data:
            continue
        if not stripped or stripped.startswith("Program"):
            continue

        parts = stripped.split(",")
        if len(parts) < 2:
            continue

        try:
            var_id = int(parts[0].strip())
            value = float(parts[1].strip())
        except ValueError:
            continue

        if var_id in var_map.values():
            # Reverse map: var_id → our_col
            for our_col, vid in var_map.items():
                if vid == var_id:
                    rows[our_col] = value
                    break

            # When all vars collected for this timestep, store
            if len(rows) == n_vars:
                result.append({
                    "hour": round(len(result) * 0.25, 2),
                    **{k: round(v, 4) for k, v in rows.items()},
                })
                rows = {}

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate E+ reference data with controlled 20°C interior"
    )
    parser.add_argument("--idf-dir", default="tests/reference_data/energyplus_models")
    parser.add_argument("--output-dir", default="tests/reference_data/conduction")
    args = parser.parse_args()

    idf_dir = Path(args.idf_dir)
    output_dir = Path(args.output_dir)
    denver_epw = Path(EPW_FILE)
    if not denver_epw.is_absolute():
        denver_epw = Path.cwd() / denver_epw

    if not denver_epw.exists():
        print(f"ERROR: Weather file not found: {denver_epw}")
        return 1

    for key, surface_name in CONSTRUCTIONS.items():
        idf_file = idf_dir / f"step_change_{key}.idf"
        if not idf_file.exists():
            print(f"SKIP: {idf_file} not found")
            continue

        print(f"\nProcessing {idf_file.name} (surface: {surface_name})...")

        with open(idf_file) as f:
            idf_content = f.read()

        # Inject HVAC block before Output:Variable
        lines = idf_content.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("Output:Variable,"):
                lines.insert(i, HVAC_BLOCK)
                break
        modified = "\n".join(lines)

        with tempfile.TemporaryDirectory(prefix=f"ep_{key}_") as tmpdir:
            success = expand_and_run_ep(modified, tmpdir)
            if not success:
                print("  FAILED: E+ run failed")
                continue

            eso_path = os.path.join(tmpdir, "eplusout.eso")
            if not os.path.exists(eso_path):
                print("  FAILED: eplusout.eso not found")
                continue

            rows = parse_eso(eso_path)
            if not rows:
                print("  FAILED: no data extracted from ESO")
                continue

            output_file = output_dir / f"step_response_{key}.csv"
            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
                writer.writeheader()
                writer.writerows(rows)

            T_zones = [r["T_zone"] for r in rows]
            T_surfs = [r["T_surface_inside"] for r in rows]
            print(f"  Wrote {len(rows)} rows → {output_file}")
            print(f"  T_zone: min={min(T_zones):.1f}, max={max(T_zones):.1f}, mean={sum(T_zones)/len(T_zones):.2f}")
            print(f"  T_surface_inside: min={min(T_surfs):.1f}, max={max(T_surfs):.1f}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    exit(main() or 0)
