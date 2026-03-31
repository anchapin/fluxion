#!/usr/bin/env python3
"""
Phase 25: Parametric Sweep Script for OpenStudio MCP
Generates mass level, timestep, and construction variants for Case 900
"""

import json
import os
import sys

# Add OpenStudio to path (if needed)
try:
    import openstudio
except ImportError:
    print("OpenStudio Python module not found. Running in MCP mode...")
    print(
        "This script should be run via OpenStudio CLI or with openstudio-python installed."
    )
    sys.exit(1)


def create_mass_variant(base_osm_path, output_path, mass_multiplier):
    """
    Create a variant with modified thermal mass.

    Args:
        base_osm_path: Path to baseline OSM file
        output_path: Path to save modified OSM
        mass_multiplier: Multiplier for material density (0.5, 1.0, 1.5, 2.0, 3.0)

    Returns:
        dict: Variant info
    """
    # Load the model
    translator = openstudio.osversion.VersionTranslator()
    model = translator.loadModel(base_osm_path)
    if model.isEmpty():
        raise Exception(f"Failed to load model: {base_osm_path}")
    model = model.get()

    # Modify material densities
    materials = model.getMaterials()
    modified_count = 0
    for material in materials:
        if material.to_StandardOpaqueMaterial().is_initialized():
            mat = material.to_StandardOpaqueMaterial().get()
            original_density = mat.density()
            mat.setDensity(original_density * mass_multiplier)
            mat.setName(f"{mat.name().get()} (mass x{mass_multiplier})")
            modified_count += 1

    # Save modified model
    model.save(output_path, True)

    return {
        "mass_multiplier": mass_multiplier,
        "materials_modified": modified_count,
        "output_path": output_path,
    }


def create_timestep_variant(base_osm_path, output_path, timesteps_per_hour):
    """
    Create a variant with different simulation timestep.

    Args:
        base_osm_path: Path to baseline OSM file
        output_path: Path to save modified OSM
        timesteps_per_hour: Number of timesteps per hour (1, 2, 4, 10, 60)

    Returns:
        dict: Variant info
    """
    translator = openstudio.osversion.VersionTranslator()
    model = translator.loadModel(base_osm_path)
    if model.isEmpty():
        raise Exception(f"Failed to load model: {base_osm_path}")
    model = model.get()

    # Set timestep
    timestep = model.getTimestep()
    timestep.setNumberOfTimestepsPerHour(timesteps_per_hour)

    # Save modified model
    model.save(output_path, True)

    return {"timesteps_per_hour": timesteps_per_hour, "output_path": output_path}


def create_construction_variant(base_osm_path, output_path, construction_changes):
    """
    Create a variant with modified construction properties.

    Args:
        base_osm_path: Path to baseline OSM file
        output_path: Path to save modified OSM
        construction_changes: Dict of construction name -> property changes

    Returns:
        dict: Variant info
    """
    translator = openstudio.osversion.VersionTranslator()
    model = translator.loadModel(base_osm_path)
    if model.isEmpty():
        raise Exception(f"Failed to load model: {base_osm_path}")
    model = model.get()

    # Apply construction changes
    # (Implementation depends on specific changes needed)

    # Save modified model
    model.save(output_path, True)

    return {"construction_changes": construction_changes, "output_path": output_path}


def run_simulation(osm_path, name, output_dir):
    """
    Run EnergyPlus simulation for an OSM file.

    Args:
        osm_path: Path to OSM file
        name: Simulation name
        output_dir: Output directory

    Returns:
        str: Run ID or path to results
    """
    # Create workflow
    workflow = openstudio.workflow.Workflow()
    workflow.setOsmPath(openstudio.Path(osm_path))
    workflow.setOutputDirectory(openstudio.Path(output_dir))

    # Run workflow
    result = workflow.run()

    if result.value():
        return str(result.get())
    else:
        raise Exception(f"Simulation failed: {result.errorMessage()}")


def main():
    """Main execution function."""
    base_osm = "/runs/phase_25/case_900_baseline.osm"
    output_base = "/runs/phase_25/parametric_sweeps"

    # Create output directories
    os.makedirs(f"{output_base}/mass_sweep", exist_ok=True)
    os.makedirs(f"{output_base}/timestep_sweep", exist_ok=True)
    os.makedirs(f"{output_base}/construction_sweep", exist_ok=True)

    results = {"mass_sweep": [], "timestep_sweep": [], "construction_sweep": []}

    # Mass sweep: 50%, 100%, 150%, 200%, 300%
    mass_multipliers = [0.5, 1.0, 1.5, 2.0, 3.0]
    print(f"\n=== Mass Level Sweep ({len(mass_multipliers)} variants) ===")
    for mass_mult in mass_multipliers:
        output_path = f"{output_base}/mass_sweep/case_900_mass_{mass_mult:.1f}x.osm"
        print(f"  Creating mass variant: {mass_mult}x")
        variant = create_mass_variant(base_osm, output_path, mass_mult)
        results["mass_sweep"].append(variant)
        print(f"    ✓ Saved to {output_path}")

    # Timestep sweep: 1, 2, 4, 10, 60 timesteps/hour
    timesteps = [1, 2, 4, 10, 60]
    print(f"\n=== Timestep Sensitivity Sweep ({len(timesteps)} variants) ===")
    for ts in timesteps:
        output_path = f"{output_base}/timestep_sweep/case_900_timestep_{ts}tph.osm"
        print(f"  Creating timestep variant: {ts} timesteps/hour")
        variant = create_timestep_variant(base_osm, output_path, ts)
        results["timestep_sweep"].append(variant)
        print(f"    ✓ Saved to {output_path}")

    # Save sweep plan
    plan_path = f"{output_base}/sweep_plan.json"
    with open(plan_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Sweep Plan Saved ===")
    print(f"  Total variants created: {sum(len(v) for v in results.values())}")
    print(f"  Plan saved to: {plan_path}")

    return results


if __name__ == "__main__":
    results = main()
    print("\n=== Parametric Sweep Generation Complete ===")
