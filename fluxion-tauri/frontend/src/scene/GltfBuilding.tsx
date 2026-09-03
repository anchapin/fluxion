import { useMemo } from "react";
import * as THREE from "three";
import {
  bakedMeshDescriptors,
  meshDisplayColor,
  type MeshTemperatureEntry,
} from "../gltf/zoneMeshMapping";

export interface GltfBuildingProps {
  /** Parsed glTF scene root (see `gltf/loadGltfBuilding.ts`). */
  scene: THREE.Group;
  /** Zone join computed by the caller (App) so the Legend shares it. */
  entries: MeshTemperatureEntry[];
  thermal: boolean;
  zoneColoring: boolean;
  wireframe: boolean;
  tempRange: { min: number; max: number };
}

/**
 * Renders the glTF-loaded building as declarative R3F meshes (issue #3175).
 *
 * The loaded scene graph is flattened once via `bakedMeshDescriptors`
 * (world transforms baked, so authoring-tool node nesting collapses) and
 * each mesh is re-emitted with a per-mesh standard material whose color is
 * the `meshDisplayColor` decision: thermal colormap → zone palette →
 * neutral gray for meshes that don't follow the `Zone_{id}` convention.
 *
 * The BEM wire-contract path (BEM surfaces + `ThermalMaterial` shader) stays
 * in `SurfaceMesh.tsx`; the two viewers share one Canvas via
 * `GeometryScene`.
 */
export function GltfBuilding({
  scene,
  entries,
  thermal,
  zoneColoring,
  wireframe,
  tempRange,
}: GltfBuildingProps) {
  const descriptors = useMemo(() => bakedMeshDescriptors(scene), [scene]);

  // mesh uuid → joined zone/temperature entry (convention-matching only).
  const entryByUuid = useMemo(() => {
    const map = new Map<string, MeshTemperatureEntry>();
    for (const entry of entries) map.set(entry.mesh.uuid, entry);
    return map;
  }, [entries]);

  return (
    <group name="gltf-building">
      {descriptors.map(({ mesh, position, quaternion, scale }) => {
        const entry = entryByUuid.get(mesh.uuid) ?? null;
        const color = meshDisplayColor(
          entry ? entry.zoneId : null,
          entry ? entry.temperature : null,
          { thermal, zoneColoring, tempRange },
        );
        return (
          <mesh
            key={mesh.uuid}
            name={mesh.name}
            geometry={mesh.geometry}
            position={position}
            quaternion={quaternion}
            scale={scale}
          >
            <meshStandardMaterial
              color={color}
              metalness={0}
              roughness={0.85}
              side={THREE.DoubleSide}
              wireframe={wireframe}
            />
          </mesh>
        );
      })}
    </group>
  );
}
