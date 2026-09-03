import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as THREE from "three";
import { buildSampleGltf, MAX_SAMPLE_GLTF_BYTES } from "../src/gltf/buildSampleGltf";
import { parseGltfBuilding } from "../src/gltf/loadGltfBuilding";
import { zoneIdFromMeshName } from "../src/gltf/zoneMeshMapping";

// Minimal DOM shim so three's FileLoader (GLTFLoader decodes the embedded
// data-URI buffer through it) can run in the node test environment.
// Browsers and the Tauri WebView provide ProgressEvent natively; the shim
// only needs to exist — three constructs it in its error-reporting path.
if (typeof globalThis.ProgressEvent === "undefined") {
  globalThis.ProgressEvent = class {} as unknown as typeof ProgressEvent;
}

const assetPath = fileURLToPath(
  new URL("../src/assets/sample-building.gltf", import.meta.url),
);
const assetText = readFileSync(assetPath, "utf8");
const asset = JSON.parse(assetText) as {
  asset: { version: string };
  nodes: Array<{ name: string; mesh: number }>;
  buffers: Array<{ byteLength: number; uri: string }>;
  accessors: Array<{ count: number; type: string }>;
  meshes: Array<{ primitives: Array<{ mode: number }> }>;
};

describe("committed sample glTF asset (structure pins)", () => {
  it("is valid glTF 2.0 JSON", () => {
    expect(asset.asset.version).toBe("2.0");
  });

  it("stays under the 1 MB budget from issue #3175", () => {
    const bytes = Buffer.byteLength(assetText, "utf8");
    expect(bytes).toBeLessThan(MAX_SAMPLE_GLTF_BYTES);
  });

  it("declares Zone_{id} mesh nodes, including the suffixed form", () => {
    const names = asset.nodes.map((n) => n.name);
    expect(names).toContain("Zone_1");
    expect(names).toContain("Zone_2");
    expect(names).toContain("Zone_2_Roof"); // Zone_2 + roof → same zone
    expect(names).toContain("Zone_3");
  });

  it("includes a deliberately unmatched mesh to exercise the neutral path", () => {
    const names = asset.nodes.map((n) => n.name);
    expect(names.filter((n) => zoneIdFromMeshName(n) === null)).toContain(
      "Site_Pad",
    );
  });

  it("embedded base64 buffer matches its declared byteLength", () => {
    const [buffer] = asset.buffers;
    expect(buffer.uri.startsWith("data:application/octet-stream;base64,")).toBe(
      true,
    );
    const decoded = Buffer.from(
      buffer.uri.slice("data:application/octet-stream;base64,".length),
      "base64",
    );
    expect(decoded.byteLength).toBe(buffer.byteLength);
  });

  it("every mesh primitive is TRIANGLES with POSITION accessors", () => {
    for (const mesh of asset.meshes) {
      for (const prim of mesh.primitives) {
        expect(prim.mode).toBe(4);
      }
    }
    const vec3s = asset.accessors.filter((a) => a.type === "VEC3");
    expect(vec3s.length).toBeGreaterThan(0);
    for (const acc of vec3s) {
      expect(acc.count % 3 === 0 || acc.count >= 4).toBe(true);
    }
  });

  it("committed asset matches the programmatic builder (sync pin)", () => {
    // Mirrors the `embedded web fallback stays in sync` pattern from
    // tests/geometryAdapter.test.ts: regenerate with `npm run gltf:sample`.
    expect(JSON.parse(assetText)).toEqual(buildSampleGltf());
  });
});

describe("parseGltfBuilding — real GLTFLoader round-trip", () => {
  it("parses the committed asset into Zone_{id} THREE.Mesh objects", async () => {
    const scene = await parseGltfBuilding(assetText);
    const meshes: THREE.Mesh[] = [];
    scene.traverse((obj) => {
      if ((obj as THREE.Mesh).isMesh) meshes.push(obj as THREE.Mesh);
    });
    expect(meshes.length).toBe(5);
    expect(meshes.map((m) => m.name).sort()).toEqual(
      ["Site_Pad", "Zone_1", "Zone_2", "Zone_2_Roof", "Zone_3"].sort(),
    );
    // Buffer geometry actually decoded: 24 verts / 36 indices per box.
    for (const mesh of meshes) {
      const geo = mesh.geometry as THREE.BufferGeometry;
      expect(geo.getAttribute("position").count).toBe(24);
      expect(geo.getIndex()?.count).toBe(36);
    }
    // glTF is Y-up natively — no axis swap expected on the root.
    expect(scene.rotation.x).toBe(0);
  });

  it("rejects malformed glTF text instead of throwing sync", async () => {
    await expect(parseGltfBuilding("{not json")).rejects.toBeInstanceOf(Error);
    await expect(parseGltfBuilding('{"asset":{"version":"2.0"}}')).rejects
      .toBeInstanceOf(Error);
  });
});
