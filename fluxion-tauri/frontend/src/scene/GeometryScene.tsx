import * as THREE from "three";
import { Canvas } from "@react-three/fiber";
import { OrbitCam } from "./OrbitCam";
import type { OrbitControlsImpl } from "./OrbitCam";
import type { RenderModel, RenderSurface } from "../lib/geometryAdapter";
import { SurfaceMesh } from "./SurfaceMesh";
import { zoneNumber } from "../livetwin/protocol";
import type { ZoneState } from "../livetwin/protocol";

export interface GeometrySceneProps {
  model: RenderModel;
  zoneColoring: boolean;
  thermal: boolean;
  wireframe: boolean;
  /** Latest LiveTwin zone states keyed by numeric zone id. */
  liveZones: Map<number, ZoneState>;
  onControlsReady?: (controls: OrbitControlsImpl | null) => void;
}

/**
 * React Three Fiber scene rendering the building geometry as triangulated
 * surface meshes. Replaces the imperative three.js loops of the two legacy
 * viewers while reusing their visual vocabulary (zone palette, thermal
 * shader, wireframe toggle).
 */
export function GeometryScene({
  model,
  zoneColoring,
  thermal,
  wireframe,
  liveZones,
  onControlsReady,
}: GeometrySceneProps) {
  const temps = [...liveZones.values()].map((z) => z.t_air);
  const tempRange = thermalRange(temps);

  return (
    <Canvas
      camera={{ fov: 60, near: 0.1, far: 10000, position: [20, 14, 20] }}
      onCreated={({ scene }) => {
        scene.background = new THREE.Color(0x0a0a14);
      }}
    >
      <ambientLight intensity={0.6} />
      <directionalLight position={[50, 50, 50]} intensity={0.8} />
      <gridHelper args={[100, 50, 0x0f3460, 0x0f3460]} />

      {model.surfaces.map((surface) => (
        <SurfaceMesh
          key={surface.id}
          surface={surface}
          zoneColoring={zoneColoring}
          thermal={thermal}
          wireframe={wireframe}
          temperature={tempForSurface(surface, liveZones)}
          tempRange={tempRange}
        />
      ))}

      <OrbitCam bounds={model.bounds} onControlsReady={onControlsReady} />
    </Canvas>
  );
}

function tempForSurface(
  surface: RenderSurface,
  liveZones: Map<number, ZoneState>,
): number | null {
  if (!surface.zoneId) return null;
  const state = liveZones.get(zoneNumber(surface.zoneId));
  return state ? state.t_air : null;
}

function thermalRange(temps: number[]): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const t of temps) {
    if (Number.isFinite(t)) {
      min = Math.min(min, t);
      max = Math.max(max, t);
    }
  }
  if (!Number.isFinite(min)) return { min: 15, max: 30 };
  if (max - min < 0.5) return { min: min - 0.25, max: max + 0.25 };
  return { min, max };
}
