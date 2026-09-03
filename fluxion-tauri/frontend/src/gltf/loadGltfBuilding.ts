import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import sampleGltfText from "../assets/sample-building.gltf?raw";

/**
 * glTF building-geometry loader (issue #3175).
 *
 * Uses the GLTFLoader that ships with the existing `three` dependency
 * (`three/examples/jsm/loaders/GLTFLoader.js`) — no extra packages. The
 * sample asset is inlined at build time via Vite's `?raw` import and fed
 * through `GLTFLoader.parse`, so the render path works offline with no
 * server MIME-type or fetch round-trip in either Tauri or web mode.
 *
 * This is the loader counterpart to the BEM wire contract
 * (`tauri/geometryService.ts`); both feed the same R3F canvas.
 */

/**
 * Parses glTF 2.0 JSON text into a `THREE.Group` scene root. Rejects with a
 * descriptive error (malformed JSON, unsupported buffers, …) — callers
 * decide whether to fall back to the BEM/sample-geometry path.
 */
export async function parseGltfBuilding(gltfText: string): Promise<THREE.Group> {
  const loader = new GLTFLoader();
  const gltf = await loader.parseAsync(gltfText, "");
  if (!gltf.scene) {
    // e.g. structurally-JSON documents without a `scenes` array.
    throw new Error("parseGltfBuilding: glTF document has no scene to render");
  }
  return gltf.scene;
}

/**
 * The committed sample building (`src/assets/sample-building.gltf`, kept
 * under 1 MB — see `src/gltf/buildSampleGltf.ts`). Loaded once and cached;
 * parse failures propagate to the caller's fallback handling.
 */
let sampleScene: Promise<THREE.Group> | null = null;

export function loadSampleGltfBuilding(): Promise<THREE.Group> {
  sampleScene ??= parseGltfBuilding(sampleGltfText);
  return sampleScene;
}
