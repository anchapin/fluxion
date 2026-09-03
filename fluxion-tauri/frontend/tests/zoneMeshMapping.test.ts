import { describe, expect, it } from "vitest";
import * as THREE from "three";
import {
  NEUTRAL_MESH_COLOR,
  bakedMeshDescriptors,
  gltfSceneBounds,
  meshDisplayColor,
  resolutionTempRange,
  resolveZoneMeshTemperatures,
  zoneIdFromMeshName,
} from "../src/gltf/zoneMeshMapping";
import { thermalHexFromTemp } from "../src/lib/thermal";
import { syntheticZoneState } from "../src/livetwin/protocol";
import type { ZoneState } from "../src/livetwin/protocol";

const RANGE = { min: 15, max: 30 };

/** Builds a nested scene graph: group → subgroup → named meshes. */
function buildScene(
  meshNames: string[],
): { root: THREE.Group; meshes: THREE.Mesh[] } {
  const root = new THREE.Group();
  root.name = "root";
  const subgroup = new THREE.Group();
  subgroup.name = "Level1";
  subgroup.position.set(10, 0, 0); // nested transform must be baked
  const meshes = meshNames.map((name) => {
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(2, 2, 2),
      new THREE.MeshStandardMaterial(),
    );
    mesh.name = name;
    return mesh;
  });
  meshes.forEach((m) => subgroup.add(m));
  // A non-mesh Object3D with a Zone_ name must be ignored by traversal.
  const decoy = new THREE.Object3D();
  decoy.name = "Zone_99";
  subgroup.add(decoy);
  root.add(subgroup);
  return { root, meshes };
}

function liveZones(
  entries: Record<number, number>,
): Map<number, ZoneState> {
  const map = new Map<number, ZoneState>();
  for (const [zoneId, tAir] of Object.entries(entries)) {
    map.set(Number(zoneId), syntheticZoneState(Number(zoneId), tAir));
  }
  return map;
}

describe("zoneIdFromMeshName — `Zone_{id}` convention", () => {
  it("parses exact and suffixed forms", () => {
    expect(zoneIdFromMeshName("Zone_1")).toBe(1);
    expect(zoneIdFromMeshName("Zone_42")).toBe(42);
    expect(zoneIdFromMeshName("Zone_2_Roof")).toBe(2);
    expect(zoneIdFromMeshName("Zone_12_west_wing")).toBe(12);
  });

  it("rejects non-convention names", () => {
    expect(zoneIdFromMeshName("zone_3")).toBeNull(); // case-sensitive
    expect(zoneIdFromMeshName("Zone_")).toBeNull();
    expect(zoneIdFromMeshName("Zone_x")).toBeNull();
    expect(zoneIdFromMeshName("Site_Pad")).toBeNull();
    expect(zoneIdFromMeshName("")).toBeNull();
    expect(zoneIdFromMeshName("Space_3")).toBeNull();
  });
});

describe("resolveZoneMeshTemperatures — the LiveTwin join", () => {
  it("joins meshes with their zone's t_air", () => {
    const { root } = buildScene(["Zone_1", "Zone_2", "Zone_2_Roof"]);
    const res = resolveZoneMeshTemperatures(root, liveZones({ 1: 18.5, 2: 24 }));
    expect(res.entries.map((e) => [e.zoneId, e.temperature])).toEqual([
      [1, 18.5],
      [2, 24],
      [2, 24],
    ]);
    expect(res.unmatchedMeshNames).toEqual([]);
    expect(res.unmatchedZoneIds).toEqual([]);
  });

  it("reports meshes that don't follow the convention", () => {
    const { root } = buildScene(["Zone_1", "Site_Pad"]);
    const res = resolveZoneMeshTemperatures(root, liveZones({ 1: 20 }));
    expect(res.entries).toHaveLength(1);
    expect(res.unmatchedMeshNames).toEqual(["Site_Pad"]);
  });

  it("reports live zones that own no mesh", () => {
    const { root } = buildScene(["Zone_1"]);
    const res = resolveZoneMeshTemperatures(
      root,
      liveZones({ 1: 20, 7: 21, 3: 22 }),
    );
    expect(res.unmatchedZoneIds).toEqual([3, 7]); // sorted
  });

  it("matched mesh without telemetry gets temperature null (not dropped)", () => {
    const { root } = buildScene(["Zone_5"]);
    const res = resolveZoneMeshTemperatures(root, new Map());
    expect(res.entries).toHaveLength(1);
    expect(res.entries[0].temperature).toBeNull();
    expect(res.unmatchedZoneIds).toEqual([]);
  });

  it("never throws on empty scenes / empty telemetry", () => {
    const empty = resolveZoneMeshTemperatures(new THREE.Group(), new Map());
    expect(empty.entries).toEqual([]);
    expect(empty.unmatchedMeshNames).toEqual([]);
    expect(empty.unmatchedZoneIds).toEqual([]);
  });

  it("ignores non-mesh objects named like zones", () => {
    const { root } = buildScene([]);
    const res = resolveZoneMeshTemperatures(root, new Map());
    expect(res.entries).toEqual([]);
    expect(res.unmatchedMeshNames).toEqual([]);
  });
});

describe("meshDisplayColor — coloring decision", () => {
  it("thermal + live temperature → colormap hex", () => {
    const cold = meshDisplayColor(1, 15, { thermal: true, zoneColoring: true, tempRange: RANGE });
    const hot = meshDisplayColor(1, 30, { thermal: true, zoneColoring: true, tempRange: RANGE });
    expect(cold).toBe(thermalHexFromTemp(15, 15, 30));
    expect(hot).toBe(thermalHexFromTemp(30, 15, 30));
    expect(cold).not.toBe(hot);
  });

  it("thermal without telemetry → neutral, never colormap", () => {
    const color = meshDisplayColor(1, null, { thermal: true, zoneColoring: false, tempRange: RANGE });
    expect(color).toBe(NEUTRAL_MESH_COLOR);
  });

  it("unmatched mesh (zoneId null) → neutral regardless of mode", () => {
    const opts = { thermal: true, zoneColoring: true, tempRange: RANGE };
    expect(meshDisplayColor(null, 25, opts)).toBe(NEUTRAL_MESH_COLOR);
  });

  it("zone coloring (thermal off) → deterministic zone palette", () => {
    const a = meshDisplayColor(3, null, { thermal: false, zoneColoring: true, tempRange: RANGE });
    const b = meshDisplayColor(3, 25, { thermal: false, zoneColoring: true, tempRange: RANGE });
    expect(a).toBe(b); // temperature irrelevant when thermal is off
    // Zones 1-3 share the BEM legend palette (zone-{n} hash form).
    expect(a).not.toBe(
      meshDisplayColor(1, null, { thermal: false, zoneColoring: true, tempRange: RANGE }),
    );
  });

  it("all toggles off → neutral", () => {
    expect(
      meshDisplayColor(1, 22, { thermal: false, zoneColoring: false, tempRange: RANGE }),
    ).toBe(NEUTRAL_MESH_COLOR);
  });
});

describe("thermalHexFromTemp — GLSL colormap parity", () => {
  it("anchor stops map to the pinned 5-stop palette (0xRRGGBB)", () => {
    expect(thermalHexFromTemp(15, 15, 30)).toBe(0x3b82f5); // cold
    expect(thermalHexFromTemp(30, 15, 30)).toBe(0xf03d3d); // hot
  });

  it("clamps out-of-range temperatures", () => {
    expect(thermalHexFromTemp(-100, 15, 30)).toBe(thermalHexFromTemp(15, 15, 30));
    expect(thermalHexFromTemp(999, 15, 30)).toBe(thermalHexFromTemp(30, 15, 30));
  });

  it("degenerate near-constant range never divides by zero", () => {
    expect(() => thermalHexFromTemp(20, 20, 20)).not.toThrow();
  });
});

describe("bakedMeshDescriptors — scene flattening", () => {
  it("bakes nested local transforms into world transforms", () => {
    const { root, meshes } = buildScene(["Zone_1"]);
    const desc = bakedMeshDescriptors(root);
    expect(desc).toHaveLength(1);
    // Subgroup sits at x=10; the baked mesh position must include it.
    expect(desc[0].position[0]).toBeCloseTo(10, 6);
    expect(desc[0].mesh).toBe(meshes[0]);
    // Identity rotation/scale for untranslated children.
    expect(desc[0].quaternion).toEqual([0, 0, 0, 1]);
    expect(desc[0].scale).toEqual([1, 1, 1]);
  });

  it("covers every mesh in traversal order", () => {
    const { root } = buildScene(["Zone_1", "Zone_2", "Site_Pad"]);
    expect(bakedMeshDescriptors(root).map((d) => d.mesh.name)).toEqual([
      "Zone_1",
      "Zone_2",
      "Site_Pad",
    ]);
  });
});

describe("gltfSceneBounds / resolutionTempRange", () => {
  it("computes world bounds of nested geometry", () => {
    const { root } = buildScene(["Zone_1"]);
    const bounds = gltfSceneBounds(root);
    // Box 2×2×2 nested in a subgroup at x=10.
    expect(bounds.min[0]).toBeCloseTo(9, 6);
    expect(bounds.max[0]).toBeCloseTo(11, 6);
    expect(bounds.min[1]).toBeCloseTo(-1, 6);
    expect(bounds.max[1]).toBeCloseTo(1, 6);
  });

  it("empty scene → unit box (OrbitCam degenerate-size guard)", () => {
    const b = gltfSceneBounds(new THREE.Group());
    expect(b.min).toEqual([-0.5, -0.5, -0.5]);
    expect(b.max).toEqual([0.5, 0.5, 0.5]);
  });

  it("temperature range spans resolved live entries with nulls excluded", () => {
    const { root } = buildScene(["Zone_1", "Zone_2"]);
    const res = resolveZoneMeshTemperatures(root, liveZones({ 1: 18, 2: 26 }));
    expect(resolutionTempRange(res)).toEqual({ min: 18, max: 26 });
  });

  it("no telemetry → default 15-30 display range", () => {
    const { root } = buildScene(["Zone_1"]);
    const res = resolveZoneMeshTemperatures(root, new Map());
    expect(resolutionTempRange(res)).toEqual({ min: 15, max: 30 });
  });
});
