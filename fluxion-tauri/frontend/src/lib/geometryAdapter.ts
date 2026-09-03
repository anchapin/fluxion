import type { BuildingGeometry, ThermalZone } from "../types/geometry";
import { triangulatePolygon } from "./triangulate";

export interface RenderSurface {
  id: string;
  spaceId: string;
  zoneId: string | null;
  surfaceType: string;
  area: number;
  /** Flat Y-up xyz triangle buffer (triangleCount × 3 vertices × xyz). */
  positions: Float32Array;
  triangleCount: number;
}

export interface GeometryStats {
  buildingName: string;
  levelCount: number;
  spaceCount: number;
  zoneCount: number;
  totalFloorArea: number;
}

export interface RenderModel {
  surfaces: RenderSurface[];
  /** Y-up axis-aligned bounds, three.js ordering. */
  bounds: { min: [number, number, number]; max: [number, number, number] };
  stats: GeometryStats;
  zones: ThermalZone[];
}

/** Zone palette inherited from the preserved geometry viewer. */
const ZONE_PALETTE = [
  0x3b82f6, 0x22c55e, 0xeab308, 0xec4899, 0x8b5cf6, 0x14b8a6, 0xf97316,
  0x06b6d4, 0x84cc16, 0xf43f5e,
];

/** Deterministic per-zone palette color (stable across renders/sessions). */
export function zoneColor(zoneId: string): number {
  let h = 0;
  for (let i = 0; i < zoneId.length; i++) {
    h = (h * 31 + zoneId.charCodeAt(i)) >>> 0;
  }
  return ZONE_PALETTE[h % ZONE_PALETTE.length];
}

/**
 * Converts one `BuildingGeometry` (the `load_geometry` wire contract) into
 * the triangulated Y-up model the R3F scene dereferences, mirroring the
 * Rust `compute_geometry_summary` stats shown in the InfoPanel. BEM data
 * is Z-up; three.js is Y-up, so vertices map (x, y, z) → (x, z, y).
 */
export function toRenderModel(geometry: BuildingGeometry): RenderModel {
  const surfaces: RenderSurface[] = [];
  let totalFloorArea = 0;
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];

  for (const level of geometry.levels) {
    for (const space of level.spaces) {
      for (const surface of space.surfaces) {
        // Triangulate in BEM coordinates, then flip to Y-up in place.
        const tri = triangulatePolygon(surface.vertices, surface.normal);
        if (!tri) continue;
        const positions = new Float32Array(tri.length);
        for (let i = 0; i < tri.length; i += 3) {
          const x = tri[i];
          const y = tri[i + 1];
          const z = tri[i + 2];
          positions[i] = x;
          positions[i + 1] = z;
          positions[i + 2] = y;
          min[0] = Math.min(min[0], x);
          max[0] = Math.max(max[0], x);
          min[1] = Math.min(min[1], z);
          max[1] = Math.max(max[1], z);
          min[2] = Math.min(min[2], y);
          max[2] = Math.max(max[2], y);
        }
        if (surface.surface_type === "Floor") totalFloorArea += surface.area;
        surfaces.push({
          id: surface.id,
          spaceId: space.id,
          zoneId: space.zone_id,
          surfaceType: surface.surface_type,
          area: surface.area,
          positions,
          triangleCount: tri.length / 9,
        });
      }
    }
  }

  return {
    surfaces,
    bounds: { min, max },
    stats: {
      buildingName: geometry.name,
      levelCount: geometry.levels.length,
      spaceCount: geometry.levels.reduce((n, l) => n + l.spaces.length, 0),
      zoneCount: geometry.zones.length,
      totalFloorArea,
    },
    zones: geometry.zones,
  };
}
