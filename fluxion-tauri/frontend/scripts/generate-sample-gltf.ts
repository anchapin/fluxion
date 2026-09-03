/**
 * Regenerates `src/assets/sample-building.gltf` from the typed builder
 * (`src/gltf/buildSampleGltf.ts`). Run with `npm run gltf:sample`.
 *
 * Uses Node type stripping (Node >= 22.6) so no extra toolchain deps are
 * needed; the builder itself is imported by `tests/gltfAsset.test.ts`,
 * which fails if the committed asset and the builder ever drift apart.
 */
import { writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { buildSampleGltf, MAX_SAMPLE_GLTF_BYTES } from "../src/gltf/buildSampleGltf.ts";

const outPath = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "src",
  "assets",
  "sample-building.gltf",
);
const json = JSON.stringify(buildSampleGltf(), null, 1) + "\n";
const bytes = Buffer.byteLength(json, "utf8");
if (bytes > MAX_SAMPLE_GLTF_BYTES) {
  console.error(
    `generated asset is ${bytes} bytes (limit ${MAX_SAMPLE_GLTF_BYTES})`,
  );
  process.exit(1);
}
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, json);
console.log(`wrote ${outPath} (${bytes} bytes)`);
