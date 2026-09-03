import * as THREE from "three";
import { Canvas } from "@react-three/fiber";
import { OrbitCam } from "./OrbitCam";
import type { OrbitControlsImpl } from "./OrbitCam";
import type { RenderModel, RenderSurface } from "../lib/geometryAdapter";
import { temperatureRange } from "../lib/thermal";
import { SurfaceMesh } from "./SurfaceMesh";
import { GltfBuilding } from "./GltfBuilding";
import {
  gltfSceneBounds,
  resolutionTempRange,
  type ZoneMeshResolution,
} from "../gltf/zoneMeshMapping";
import { zoneNumber } from "../livetwin/protocol";
import type { ZoneState } from "../livetwin/protocol";

export interface GeometrySceneProps {
  /** BEM wire-contract model (`tauri/geometryService`). Nullable in glTF mode. */
  model: RenderModel | null;
  /** Parsed glTF building + zone join (issue #3175); takes precedence over `model`. */
  gltfBuilding: { scene: THREE.Group; resolution: ZoneMeshResolution } | null;
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
 *
 * Two interchangeable geometry paths share this one Canvas (issue #3175):
 * the BEM `load_geometry` wire contract (`model`) and a glTF-loaded model
 * whose `Zone_{id}` meshes are colored through the LiveTwin zone join
 * (`gltfBuilding`).
 */
export function GeometryScene({
  model,
  gltfBuilding,
  zoneColoring,
  thermal,
  wireframe,
  liveZones,
  onControlsReady,
}: GeometrySceneProps) {
  const temps = [...liveZones.values()].map((z) => z.t_air);
  const tempRange = gltfBuilding
    ? resolutionTempRange(gltfBuilding.resolution)
    : temperatureRange(temps);

  const bounds: { min: [number, number, number]; max: [number, number, number] } =
    gltfBuilding
      ? gltfSceneBounds(gltfBuilding.scene)
      : model
        ? model.bounds
        : { min: [-5, -5, -5], max: [5, 5, 5] };

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

      {gltfBuilding ? (
        <GltfBuilding
          scene={gltfBuilding.scene}
          entries={gltfBuilding.resolution.entries}
          thermal={thermal}
          zoneColoring={zoneColoring}
          wireframe={wireframe}
          tempRange={tempRange}
        />
      ) : (
        model?.surfaces.map((surface) => (
          <SurfaceMesh
            key={surface.id}
            surface={surface}
            zoneColoring={zoneColoring}
            thermal={thermal}
            wireframe={wireframe}
            temperature={tempForSurface(surface, liveZones)}
            tempRange={tempRange}
          />
        ))
      )}

      <OrbitCam bounds={bounds} onControlsReady={onControlsReady} />
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
