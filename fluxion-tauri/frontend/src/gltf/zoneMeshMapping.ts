import * as THREE from "three";
import { zoneColor } from "../lib/geometryAdapter";
import { thermalHexFromTemp, temperatureRange } from "../lib/thermal";
import type { ZoneState } from "../livetwin/protocol";

/**
 * Zone→mesh mapping system (issue #3175).
 *
 * Mesh naming convention: `Zone_{id}` where `{id}` is the numeric LiveTwin
 * `ZoneState.zone_id` (usize). A single zone may own several meshes via the
 * suffixed form `Zone_{id}_*` (e.g. `Zone_2_Roof`). Meshes that don't match
 * the convention are reported as unmatched and rendered with a neutral
 * color; telemetry for zones that own no mesh is surfaced as a warning —
 * neither case ever throws.
 *
 * Pure functions throughout (the `livetwin/reconnect.ts` pattern): every
 * entry point takes its inputs as arguments and is covered by
 * `tests/zoneMeshMapping.test.ts`.
 */

/**
 * `Zone_3`, `Zone_3_Roof`, `Zone_12-west` → 3 / 3 / 12.
 * `zone_3` (lowercase), `Zone_`, `Zone_x`, `Site_Pad`, `""` → null.
 */
export const ZONE_MESH_NAME_PATTERN = /^Zone_(\d+)(?:_.*)?$/;

/** Neutral gray (Tailwind gray-400) for meshes with no resolvable zone. */
export const NEUTRAL_MESH_COLOR = 0x9ca3af;

/** Extracts the numeric zone id from a mesh/node name, or null. */
export function zoneIdFromMeshName(name: string): number | null {
  const m = ZONE_MESH_NAME_PATTERN.exec(name);
  return m ? Number(m[1]) : null;
}

/** All renderable meshes under `root` (depth-first, deterministic order). */
export function collectMeshes(root: THREE.Object3D): THREE.Mesh[] {
  const meshes: THREE.Mesh[] = [];
  root.traverse((obj) => {
    if ((obj as THREE.Mesh).isMesh) meshes.push(obj as THREE.Mesh);
  });
  return meshes;
}

export interface MeshTemperatureEntry {
  /** The loaded glTF mesh (geometry reused verbatim by the R3F renderer). */
  mesh: THREE.Mesh;
  /** Numeric zone id resolved from the `Zone_{id}` name (never null here). */
  zoneId: number;
  /** Live zone air temperature, or null when no telemetry for this zone. */
  temperature: number | null;
}

export interface ZoneMeshResolution {
  /** Convention-matching meshes joined with their live zone temperature. */
  entries: MeshTemperatureEntry[];
  /** Mesh names that don't follow `Zone_{id}` (rendered neutral). */
  unmatchedMeshNames: string[];
  /** Live zone ids that own no mesh in the model (surfaced as a warning). */
  unmatchedZoneIds: number[];
}

/**
 * The core join: resolves every `Zone_{id}` mesh against the latest
 * LiveTwin zone states. Convention-matching meshes whose zone has no
 * telemetry keep `temperature: null` (→ neutral shading while thermal
 * coloring is active) — matching a mesh is not the same as having live
 * data for it.
 */
export function resolveZoneMeshTemperatures(
  scene: THREE.Object3D,
  liveZones: Map<number, ZoneState>,
): ZoneMeshResolution {
  const entries: MeshTemperatureEntry[] = [];
  const unmatchedMeshNames: string[] = [];
  const zoneIdsWithMesh = new Set<number>();

  for (const mesh of collectMeshes(scene)) {
    const zoneId = zoneIdFromMeshName(mesh.name);
    if (zoneId === null) {
      unmatchedMeshNames.push(mesh.name);
      continue;
    }
    zoneIdsWithMesh.add(zoneId);
    const state = liveZones.get(zoneId);
    entries.push({
      mesh,
      zoneId,
      temperature: state ? state.t_air : null,
    });
  }

  const unmatchedZoneIds: number[] = [];
  for (const zoneId of liveZones.keys()) {
    if (!zoneIdsWithMesh.has(zoneId)) unmatchedZoneIds.push(zoneId);
  }
  unmatchedZoneIds.sort((a, b) => a - b);

  return { entries, unmatchedMeshNames, unmatchedZoneIds };
}

export interface MeshColorOptions {
  thermal: boolean;
  zoneColoring: boolean;
  tempRange: { min: number; max: number };
}

/**
 * Per-mesh display color decision:
 * - thermal shading on + live temperature → 5-stop thermal colormap
 * - zone coloring on → deterministic per-zone palette (hashes the
 *   `zone-{n}` form so glTF zones share the BEM legend's palette)
 * - otherwise / no telemetry under thermal → neutral gray
 */
export function meshDisplayColor(
  zoneId: number | null,
  temperature: number | null,
  opts: MeshColorOptions,
): number {
  // Unmatched meshes are always neutral — temperature only exists through
  // the zone join, so a mesh without a zone can never claim one.
  if (zoneId === null) return NEUTRAL_MESH_COLOR;
  if (opts.thermal && temperature !== null) {
    return thermalHexFromTemp(temperature, opts.tempRange.min, opts.tempRange.max);
  }
  if (opts.zoneColoring) return zoneColor(`zone-${zoneId}`);
  return NEUTRAL_MESH_COLOR;
}

export interface BakedMeshDescriptor {
  mesh: THREE.Mesh;
  /** World transform of the mesh, baked (glTF node chains collapsed). */
  position: [number, number, number];
  quaternion: [number, number, number, number];
  scale: [number, number, number];
}

/**
 * Flattens the glTF scene graph for declarative R3F rendering: each mesh
 * keeps its geometry but its LOCAL transform is replaced by the baked WORLD
 * transform, so meshes can be rendered as siblings regardless of how the
 * authoring tool nested them. Assumes static geometry (no skinning or
 * animation), which holds for building envelopes.
 */
export function bakedMeshDescriptors(root: THREE.Object3D): BakedMeshDescriptor[] {
  return collectMeshes(root).map((mesh) => {
    // updateWorldMatrix(parents, children) recomputes ancestors + self.
    mesh.updateWorldMatrix(true, false);
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    mesh.matrixWorld.decompose(p, q, s);
    return {
      mesh,
      position: [p.x, p.y, p.z],
      quaternion: [q.x, q.y, q.z, q.w],
      scale: [s.x, s.y, s.z],
    };
  });
}

/**
 * Y-up axis-aligned world bounds of the scene (three.js ordering, the
 * shape `OrbitCam` fits the camera to). Empty scenes return a unit box
 * so OrbitCam's `maxDim` guard never sees a degenerate size.
 */
export function gltfSceneBounds(root: THREE.Object3D): {
  min: [number, number, number];
  max: [number, number, number];
} {
  const box = new THREE.Box3().setFromObject(root);
  if (box.isEmpty()) {
    return { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] };
  }
  return {
    min: [box.min.x, box.min.y, box.min.z],
    max: [box.max.x, box.max.y, box.max.z],
  };
}

/** Live temperature range across a resolution's zone entries. */
export function resolutionTempRange(
  resolution: ZoneMeshResolution,
): { min: number; max: number } {
  return temperatureRange(
    resolution.entries
      .map((e) => e.temperature)
      .filter((t): t is number => t !== null),
  );
}
