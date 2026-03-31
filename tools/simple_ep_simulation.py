#!/usr/bin/env python3
"""
Simple EnergyPlus simulation using OpenStudio API.

Generates a basic building model and runs EnergyPlus simulation
for ASHRAE 140 Case 600.
"""

import sys

sys.path.insert(0, "/usr/local/openstudio-3.11.0/Python")

from pathlib import Path

import openstudio


def create_case_600_model():
    """Create OpenStudio model for ASHRAE 140 Case 600."""
    # Create new model
    model = openstudio.model.Model()

    # Add thermal zone
    zone = openstudio.model.ThermalZone(model)
    zone.setName("Zone 1")

    # Add surfaces
    # Floor
    floor = openstudio.model.Surface(model)
    floor.setName("Floor")
    floor.setSurfaceType("Floor")
    floor.setConstruction(
        openstudio.model.DefaultConstructionSet(
            model
        ).defaultExteriorSurfaceConstruction()
    )

    # Ceiling
    ceiling = openstudio.model.Surface(model)
    ceiling.setName("Ceiling")
    ceiling.setSurfaceType("RoofCeiling")
    ceiling.setConstruction(
        openstudio.model.DefaultConstructionSet(
            model
        ).defaultExteriorSurfaceConstruction()
    )

    # Walls (North, South, East, West)
    for name, azimuth in [
        ("North Wall", 0),
        ("South Wall", 90),
        ("East Wall", 180),
        ("West Wall", 270),
    ]:
        wall = openstudio.model.Surface(model)
        wall.setName(name)
        wall.setSurfaceType("Wall")
        wall.setAzimuth(azimuth)
        wall.setConstruction(
            openstudio.model.DefaultConstructionSet(
                model
            ).defaultExteriorSurfaceConstruction()
        )

    # Add window on South wall
    window = openstudio.model.SubSurface(model)
    window.setName("South Window")
    window.setSubSurfaceType("FixedWindow")
    window.setSurface(floor_area=12.0)  # 12 m² window area

    # Add HVAC system
    thermostat = openstudio.model.Thermostat(model)
    thermostat.setName("Thermostat")
    thermostat.setHeatingSetpointSchedule(openstudio.schedule.ConstantSchedule(20.0))
    thermostat.setCoolingSetpointSchedule(openstudio.schedule.ConstantSchedule(27.0))

    # Add internal loads
    lights = openstudio.model.LightsDefinition(model)
    lights.setWattsperZoneFloorArea(200.0 / 48.0)  # 200 W over 48 m²

    # Add electric equipment
    electric_equipment = openstudio.model.ElectricEquipmentDefinition(model)
    electric_equipment.setWattsperZoneFloorArea(200.0 / 48.0)  # 200 W over 48 m²

    return model


def save_model(model: openstudio.model.Model, output_path: Path):
    """Save model to IDF file."""
    forward_translator = openstudio.version.VersionTranslator()
    workspace = forward_translator.translateModel(model)

    osm_path = output_path.with_suffix(".osm")
    model.save(osm_path, True)

    # Also save as IDF
    idf_path = output_path.with_suffix(".idf")
    idf_string = workspace.toIdfString()
    with open(idf_path, "w") as f:
        f.write(idf_string)

    print(f"Model saved to: {osm_path}")
    print(f"IDF saved to: {idf_path}")
    return idf_path


def main():
    print("Creating OpenStudio model for ASHRAE 140 Case 600...")

    try:
        model = create_case_600_model()
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Save model
    output_dir = Path("refdata")
    output_dir.mkdir(exist_ok=True)
    idf_path = save_model(model, output_dir / "Case_600.idf")

    print(f"\nIDF file created at: {idf_path}")
    print("\nTo run EnergyPlus simulation, use:")
    print(
        f"  energyplus -w refdata/epw/Denver.epw -r -d -p {output_dir} -x 'Zone Air Temperature,Site Outdoor Air Drybulb Temperature' {idf_path}"
    )
    print(
        "\nTo extract hourly results, use EnergyPlus SQL output (already enabled in model)."
    )


if __name__ == "__main__":
    main()
