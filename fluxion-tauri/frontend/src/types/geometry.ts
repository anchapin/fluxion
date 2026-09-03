/**
 * Wire types mirroring `fluxion-tauri/src-tauri/src/geometry.rs`.
 * The serde JSON contract is pinned by the Rust unit test
 * `serde_json_contract_matches_frontend_expectations` — keep in sync.
 */

export interface Vertex {
  x: number;
  y: number;
  z: number;
}

export interface BoundingBox {
  min: Vertex;
  max: Vertex;
}

export interface Surface {
  id: string;
  vertices: Vertex[];
  normal: Vertex;
  area: number;
  surface_type: string;
}

export interface Space {
  id: string;
  name: string;
  surfaces: Surface[];
  bounding_box: BoundingBox;
  zone_id: string | null;
}

export interface BuildingLevel {
  id: string;
  name: string;
  elevation: number;
  height: number;
  spaces: Space[];
}

export interface ThermalZone {
  id: string;
  name: string;
  level_id: string;
  space_ids: string[];
  setpoint_heating: number | null;
  setpoint_cooling: number | null;
}

export interface BuildingGeometry {
  id: string;
  name: string;
  levels: BuildingLevel[];
  zones: ThermalZone[];
  bounding_box: BoundingBox;
}

/** Simulation parameter shape of the `get/update_simulation_parameters` commands. */
export interface SimulationParameters {
  zone_id: string | null;
  heating_setpoint: number | null;
  cooling_setpoint: number | null;
  lighting_load: number | null;
  equipment_load: number | null;
  occupancy: number | null;
  ventilation_rate: number | null;
  wall_u_value: number | null;
  roof_u_value: number | null;
}
