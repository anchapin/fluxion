{
  "V1": {
    "version": "V1",
    "metadata": {
      "name": "Case 600 per-surface example (Issue #3989)",
      "description": "600-series-style single-zone case expressed with per-surface geometry, per-orientation fenestration and a named HVAC system (Ideal). Geometry: 8 m x 6 m x 2.7 m box; south wall carries a 12 m2 window, north wall uses a 0.2 window-to-wall ratio.",
      "author": null,
      "created_at": null,
      "schema_version": "V1"
    },
    "geometry": {
      "zones": [
        {
          "name": "Zone 1",
          "floor_area": 48.0,
          "volume": 129.6,
          "height": 2.7
        }
      ],
      "total_floor_area": 48.0,
      "total_volume": 129.6,
      "number_of_floors": 1,
      "floor_height": 2.7,
      "surfaces": [
        {
          "id": "south_wall",
          "zone_index": 0,
          "kind": "Wall",
          "area": 21.6,
          "orientation": "South",
          "tilt_deg": 90.0,
          "vertices": [
            {
              "x": 0.0,
              "y": 0.0,
              "z": 0.0
            },
            {
              "x": 8.0,
              "y": 0.0,
              "z": 0.0
            },
            {
              "x": 8.0,
              "y": 0.0,
              "z": 2.7
            },
            {
              "x": 0.0,
              "y": 0.0,
              "z": 2.7
            }
          ],
          "window": {
            "window_area": 12.0,
            "window_u_value": 3.0,
            "window_shgc": 0.79
          }
        },
        {
          "id": "north_wall",
          "zone_index": 0,
          "kind": "Wall",
          "area": 21.6,
          "orientation": "North",
          "tilt_deg": 90.0,
          "window": {
            "window_area": 0.0,
            "window_u_value": 3.0,
            "window_shgc": 0.79,
            "window_to_wall_ratio": 0.2
          }
        },
        {
          "id": "east_wall",
          "zone_index": 0,
          "kind": "Wall",
          "area": 16.2,
          "orientation": "East",
          "tilt_deg": 90.0
        },
        {
          "id": "west_wall",
          "zone_index": 0,
          "kind": "Wall",
          "area": 16.2,
          "orientation": "West",
          "tilt_deg": 90.0
        },
        {
          "id": "roof",
          "zone_index": 0,
          "kind": "Roof",
          "area": 48.0,
          "orientation": "Up",
          "tilt_deg": 0.0
        },
        {
          "id": "floor",
          "zone_index": 0,
          "kind": "Floor",
          "area": 48.0,
          "orientation": "Down",
          "tilt_deg": 0.0
        }
      ]
    },
    "constructions": {
      "wall": {
        "name": "Default Wall",
        "layers": [
          {
            "name": "Plasterboard",
            "conductivity": 0.16,
            "density": 950.0,
            "specific_heat": 840.0,
            "thickness": 0.012,
            "emissivity": 0.9,
            "absorptance": 0.7
          },
          {
            "name": "Fiberglass",
            "conductivity": 0.04,
            "density": 12.0,
            "specific_heat": 840.0,
            "thickness": 0.066,
            "emissivity": 0.9,
            "absorptance": 0.7
          },
          {
            "name": "Wood siding",
            "conductivity": 0.14,
            "density": 500.0,
            "specific_heat": 1300.0,
            "thickness": 0.009,
            "emissivity": 0.9,
            "absorptance": 0.7
          }
        ],
        "window": {
          "window_area": 12.0,
          "window_u_value": 1.5,
          "window_shgc": 0.3
        }
      },
      "roof": {
        "name": "Default Wall",
        "layers": [
          {
            "name": "Plasterboard",
            "conductivity": 0.16,
            "density": 950.0,
            "specific_heat": 840.0,
            "thickness": 0.012,
            "emissivity": 0.9,
            "absorptance": 0.7
          },
          {
            "name": "Fiberglass",
            "conductivity": 0.04,
            "density": 12.0,
            "specific_heat": 840.0,
            "thickness": 0.066,
            "emissivity": 0.9,
            "absorptance": 0.7
          },
          {
            "name": "Wood siding",
            "conductivity": 0.14,
            "density": 500.0,
            "specific_heat": 1300.0,
            "thickness": 0.009,
            "emissivity": 0.9,
            "absorptance": 0.7
          }
        ],
        "window": {
          "window_area": 12.0,
          "window_u_value": 1.5,
          "window_shgc": 0.3
        }
      },
      "floor": {
        "name": "Default Wall",
        "layers": [
          {
            "name": "Plasterboard",
            "conductivity": 0.16,
            "density": 950.0,
            "specific_heat": 840.0,
            "thickness": 0.012,
            "emissivity": 0.9,
            "absorptance": 0.7
          },
          {
            "name": "Fiberglass",
            "conductivity": 0.04,
            "density": 12.0,
            "specific_heat": 840.0,
            "thickness": 0.066,
            "emissivity": 0.9,
            "absorptance": 0.7
          },
          {
            "name": "Wood siding",
            "conductivity": 0.14,
            "density": 500.0,
            "specific_heat": 1300.0,
            "thickness": 0.009,
            "emissivity": 0.9,
            "absorptance": 0.7
          }
        ],
        "window": {
          "window_area": 12.0,
          "window_u_value": 1.5,
          "window_shgc": 0.3
        }
      },
      "interzone": null
    },
    "schedules": {
      "occupancy": {
        "name": "Occupancy",
        "schedule_type": "Weekly",
        "values": {
          "Weekly": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        }
      },
      "lighting": {
        "name": "Lighting",
        "schedule_type": "Weekly",
        "values": {
          "Weekly": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ]
        }
      },
      "hvac": {
        "heating": {
          "name": "Default Schedule",
          "schedule_type": "Constant",
          "values": {
            "Daily": [
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0
            ]
          }
        },
        "cooling": {
          "name": "Default Schedule",
          "schedule_type": "Constant",
          "values": {
            "Daily": [
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0,
              24.0
            ]
          }
        }
      },
      "infiltration": null
    },
    "weather": {
      "type": "tmy",
      "location": "Denver, CO"
    },
    "controls": {
      "zone_control": {
        "heating_setpoint": 20.0,
        "cooling_setpoint": 24.0,
        "deadband_tolerance": 0.5,
        "heating_capacity": 100000.0,
        "cooling_capacity": 100000.0
      },
      "global_control": null,
      "hvac_system": "Ideal"
    },
    "output": {
      "eui": 0.0,
      "total_energy": 0.0,
      "peak_heating_load": 0.0,
      "peak_cooling_load": 0.0,
      "heating_energy": 0.0,
      "cooling_energy": 0.0,
      "zone_temperatures": null,
      "hourly_zone_temperatures": null
    }
  }
}
