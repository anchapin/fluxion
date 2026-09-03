import { useMemo } from "react";
import * as THREE from "three";
import type { RenderSurface } from "../lib/geometryAdapter";
import { zoneColor } from "../lib/geometryAdapter";
import { ThermalMaterial } from "./ThermalMaterial";

/** Neutral palette used when zone coloring is toggled off
 * (inherited from the preserved geometry viewer's surface materials). */
const SURFACE_TYPE_COLORS: Record<string, number> = {
  Wall: 0xcccccc,
  Floor: 0x555555,
  Roof: 0x8b4513,
};

export interface SurfaceMeshProps {
  surface: RenderSurface;
  zoneColoring: boolean;
  thermal: boolean;
  wireframe: boolean;
  temperature: number | null;
  tempRange: { min: number; max: number };
}

export function SurfaceMesh({
  surface,
  zoneColoring,
  thermal,
  wireframe,
  temperature,
  tempRange,
}: SurfaceMeshProps) {
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(surface.positions, 3));
    geo.computeVertexNormals();
    return geo;
  }, [surface.positions]);

  const color = zoneColoring && surface.zoneId
    ? zoneColor(surface.zoneId)
    : SURFACE_TYPE_COLORS[surface.surfaceType] ?? 0xcccccc;

  const thermalActive = thermal && temperature !== null;

  return (
    <mesh geometry={geometry}>
      {thermalActive ? (
        <ThermalMaterial
          temperature={temperature}
          minTemp={tempRange.min}
          maxTemp={tempRange.max}
          wireframe={wireframe}
        />
      ) : (
        <meshStandardMaterial
          color={color}
          transparent
          opacity={0.75}
          side={THREE.DoubleSide}
          wireframe={wireframe}
        />
      )}
    </mesh>
  );
}
