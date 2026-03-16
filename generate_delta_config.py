#!/usr/bin/env python3
import yaml

from fluxion.validation import ASHRAE140Case

# Get Case600 spec
base_spec = ASHRAE140Case.Case600.spec()

# Convert to dict - we need to serialize the CaseSpec
# CaseSpec has __dict__ but contains complex types. Let's manually build a minimal valid spec.

# Actually, simpler: we can just use the existing spec and let PyYAML handle it through dataclass serialization
# But CaseSpec may not be directly serializable. Let me check if there's a to_dict method...

# Alternative: Use the existing CaseBuilder pattern to create a simple spec in YAML format directly
# The delta module accepts any CaseSpec, doesn't have to be from ASHRAE cases.

# Let's create a minimal valid CaseSpec in YAML manually, based on the structure from the delta test

delta_config = {
    "base": {
        "case_id": "600",
        "description": "Low mass baseline - standard construction with south windows",
        "geometry": [
            {
                "length": 8.0,
                "width": 6.0,
                "height": 2.7,
                "name": None,
            }
        ],
        "construction_type": "LowMass",
        "construction": {
            "type": "wood_frame",
            "insulation_r": 20.0,
            "stud_spacing": 0.6,
        },
        "windows": [[{"area": 12.0, "azimuth": 180.0, "tilt": 90.0}]],
        "window_properties": {
            "type": "double_clear",
            "u_value": 3.0,
            "shgc": 0.6,
            "visible_transmittance": 0.7,
        },
        "shading": None,
        "internal_loads": [{"people": 200.0, "plugs": 0.6, "lighting": 0.4}],
        "hvac": [{"heating_setpoint": 20.0, "cooling_setpoint": 27.0}],
        "night_ventilation": None,
        "common_walls": [],
        "infiltration_ach": 0.5,
        "opaque_absorptance": 0.6,
        "num_zones": 1,
        "weather_data": None,
    },
    "variants": [
        {
            "name": "high_infil",
            "patch": {"infiltration_ach": 1.5},
            "sweep": None,
        },
        {
            "name": "low_infil",
            "patch": {"infiltration_ach": 0.5},
            "sweep": None,
        },
        {
            "name": "u_sweep",
            "patch": None,
            "sweep": {"window_u_value": [2.0, 3.0, 4.0]},
        },
    ],
}

with open("delta_config.yaml", "w") as f:
    yaml.dump(delta_config, f, default_flow_style=False, sort_keys=False)

print("Delta config written to delta_config.yaml")
