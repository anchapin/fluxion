/**
 * Programmatic builder for the committed sample glTF 2.0 building asset
 * (`src/assets/sample-building.gltf`, issue #3175).
 *
 * The asset is a small box-y two-"storey" massing model whose mesh nodes
 * follow the `Zone_{id}` naming convention from the issue:
 *
 * - `Zone_1`, `Zone_3`  — plain zone boxes
 * - `Zone_2`, `Zone_2_Roof` — a zone box plus a parapet roof slab (exercises
 *   the tolerant `Zone_{id}_*` suffix form: both map to zone 2)
 * - `Site_Pad` — a ground slab that deliberately does NOT match the
 *   convention, exercising the unmatched-mesh/neutral-color path
 *
 * Zone numbers 1-3 intentionally align with the embedded BEM sample
 * (`zone-1..zone-3`) so an unmodified LiveTwin telemetry stream (usize
 * `zone_id` 1..3) drives the glTF coloring with no remapping.
 *
 * glTF is Y-up by specification — the same axis convention as three.js — so
 * no Z-up→Y-up swap is needed here (unlike the BEM wire contract, see
 * `lib/geometryAdapter.ts`).
 *
 * Regenerate the committed asset with `npm run gltf:sample`.
 */

interface BoxSpec {
  /** Node name (`Zone_{id}` convention or an intentionally unmatched name). */
  name: string;
  size: [number, number, number];
  center: [number, number, number];
  baseColor: [number, number, number, number];
}

const ZONE_BOXES: BoxSpec[] = [
  {
    name: "Zone_1",
    size: [8, 3, 6],
    center: [-5, 1.5, 0],
    baseColor: [0.72, 0.8, 0.92, 1],
  },
  {
    name: "Zone_2",
    size: [6, 4, 6],
    center: [2, 2, 0],
    baseColor: [0.85, 0.78, 0.65, 1],
  },
  {
    name: "Zone_2_Roof",
    size: [6.4, 0.3, 6.4],
    center: [2, 4.15, 0],
    baseColor: [0.45, 0.42, 0.4, 1],
  },
  {
    name: "Zone_3",
    size: [5, 2.5, 6],
    center: [7.5, 1.25, 0],
    baseColor: [0.78, 0.88, 0.76, 1],
  },
  {
    name: "Site_Pad",
    size: [24, 0.2, 10],
    center: [0, -0.1, 0],
    baseColor: [0.35, 0.37, 0.4, 1],
  },
];

interface BoxGeometryData {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint16Array;
}

/**
 * Builds a 24-vertex / 36-index axis-aligned box (three.js `BoxGeometry`
 * layout: one quad per face, face normals) centred on the origin.
 */
export function boxGeometryData(
  w: number,
  h: number,
  d: number,
): BoxGeometryData {
  const x = w / 2;
  const y = h / 2;
  const z = d / 2;
  // Each face: 4 corners in CCW order viewed from outside, plus its normal.
  const faces: Array<{
    n: [number, number, number];
    v: Array<[number, number, number]>;
  }> = [
    // +X
    { n: [1, 0, 0], v: [[x, -y, z], [x, -y, -z], [x, y, -z], [x, y, z]] },
    // -X
    { n: [-1, 0, 0], v: [[-x, -y, -z], [-x, -y, z], [-x, y, z], [-x, y, -z]] },
    // +Y
    { n: [0, 1, 0], v: [[-x, y, z], [x, y, z], [x, y, -z], [-x, y, -z]] },
    // -Y
    { n: [0, -1, 0], v: [[-x, -y, -z], [x, -y, -z], [x, -y, z], [-x, -y, z]] },
    // +Z
    { n: [0, 0, 1], v: [[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]] },
    // -Z
    { n: [0, 0, -1], v: [[x, -y, -z], [-x, -y, -z], [-x, y, -z], [x, y, -z]] },
  ];

  const positions = new Float32Array(faces.length * 4 * 3);
  const normals = new Float32Array(faces.length * 4 * 3);
  const indices = new Uint16Array(faces.length * 6);
  let pi = 0;
  let ii = 0;
  faces.forEach((face, fi) => {
    for (const [vx, vy, vz] of face.v) {
      positions[pi] = vx;
      positions[pi + 1] = vy;
      positions[pi + 2] = vz;
      normals[pi] = face.n[0];
      normals[pi + 1] = face.n[1];
      normals[pi + 2] = face.n[2];
      pi += 3;
    }
    const b = fi * 4;
    // Two CCW triangles: (0,1,2) and (0,2,3).
    indices[ii++] = b;
    indices[ii++] = b + 1;
    indices[ii++] = b + 2;
    indices[ii++] = b;
    indices[ii++] = b + 2;
    indices[ii++] = b + 3;
  });
  return { positions, normals, indices };
}

/** Size guard from issue #3175: the committed asset must stay under 1 MB. */
export const MAX_SAMPLE_GLTF_BYTES = 1_000_000;

/**
 * Assembles the complete glTF 2.0 JSON (single embedded base64 buffer,
 * one mesh + material per box). Deterministic — byte-identical output for
 * a given `ZONE_BOXES` table, so the `tests/gltfAsset.test.ts` sync pin
 * (`builder output matches the committed asset`) cannot drift silently.
 */
export function buildSampleGltf(): Record<string, unknown> {
  // 1. Concatenate every box's index/position/normal block into one buffer,
  //    4-byte aligning each bufferView (glTF bufferView byteOffsets should
  //    be a multiple of the component size; 4 covers float32 and uint16).
  const chunks: ArrayBuffer[] = [];
  let byteOffset = 0;
  const geoms = ZONE_BOXES.map((box) => {
    const geo = boxGeometryData(...box.size);
    const parts = [geo.indices.buffer, geo.positions.buffer, geo.normals.buffer];
    const partViews: Array<{ byteOffset: number; byteLength: number }> = [];
    for (const buf of parts) {
      // Uint16Array buffers are already 2-byte sized; round up to 4-align.
      const pad = (4 - (byteOffset % 4)) % 4;
      if (pad > 0) {
        chunks.push(new ArrayBuffer(pad));
        byteOffset += pad;
      }
      partViews.push({ byteOffset, byteLength: buf.byteLength });
      chunks.push(buf);
      byteOffset += buf.byteLength;
    }
    return { box, geo, partViews };
  });

  const buffer = new Uint8Array(byteOffset);
  let cursor = 0;
  for (const chunk of chunks) {
    buffer.set(new Uint8Array(chunk), cursor);
    cursor += chunk.byteLength;
  }
  // Node Buffer is unavailable in browsers; hand-roll base64 for btoa-free
  // deterministic encoding (also works in Node without imports).
  const base64 = bytesToBase64(buffer);

  // 2. bufferViews / accessors / meshes / materials per box.
  const bufferViews: Array<Record<string, unknown>> = [];
  const accessors: Array<Record<string, unknown>> = [];
  const meshes: Array<Record<string, unknown>> = [];
  const materials: Array<Record<string, unknown>> = [];
  const nodes: Array<Record<string, unknown>> = [];

  geoms.forEach(({ box, geo, partViews }, i) => {
    const [indexView, positionView, normalView] = partViews;

    bufferViews.push(
      {
        buffer: 0,
        byteOffset: indexView.byteOffset,
        byteLength: indexView.byteLength,
        target: 34963, // ELEMENT_ARRAY_BUFFER
      },
      {
        buffer: 0,
        byteOffset: positionView.byteOffset,
        byteLength: positionView.byteLength,
        target: 34962, // ARRAY_BUFFER
      },
      {
        buffer: 0,
        byteOffset: normalView.byteOffset,
        byteLength: normalView.byteLength,
        target: 34962,
      },
    );

    const vertexCount = geo.positions.length / 3;
    accessors.push(
      {
        bufferView: i * 3,
        componentType: 5123, // UNSIGNED_SHORT
        count: geo.indices.length,
        type: "SCALAR",
        min: [0],
        max: [vertexCount - 1],
      },
      {
        bufferView: i * 3 + 1,
        componentType: 5126, // FLOAT
        count: vertexCount,
        type: "VEC3",
        min: [-box.size[0] / 2, -box.size[1] / 2, -box.size[2] / 2],
        max: [box.size[0] / 2, box.size[1] / 2, box.size[2] / 2],
      },
      {
        bufferView: i * 3 + 2,
        componentType: 5126,
        count: geo.normals.length / 3,
        type: "VEC3",
      },
    );

    meshes.push({
      name: box.name,
      primitives: [
        {
          attributes: { POSITION: i * 3 + 1, NORMAL: i * 3 + 2 },
          indices: i * 3,
          material: i,
          mode: 4, // TRIANGLES
        },
      ],
    });

    materials.push({
      name: `${box.name}_material`,
      doubleSided: true,
      pbrMetallicRoughness: {
        baseColorFactor: box.baseColor,
        metallicFactor: 0,
        roughnessFactor: 0.85,
      },
    });

    nodes.push({
      name: box.name,
      mesh: i,
      translation: box.center,
    });
  });

  const gltf = {
    asset: { version: "2.0", generator: "fluxion generate-sample-gltf" },
    scene: 0,
    scenes: [{ name: "FluxionSampleBuilding", nodes: nodes.map((_, i) => i) }],
    nodes,
    meshes,
    materials,
    accessors,
    bufferViews,
    buffers: [
      {
        byteLength: byteOffset,
        uri: `data:application/octet-stream;base64,${base64}`,
      },
    ],
  };
  return gltf;
}

/** Dependency-free deterministic base64 encoder (btoa-free). */
export function bytesToBase64(bytes: Uint8Array): string {
  const chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  let out = "";
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i];
    const b1 = i + 1 < bytes.length ? bytes[i + 1] : 0;
    const b2 = i + 2 < bytes.length ? bytes[i + 2] : 0;
    out += chars[b0 >> 2];
    out += chars[((b0 & 3) << 4) | (b1 >> 4)];
    out += i + 1 < bytes.length ? chars[((b1 & 15) << 2) | (b2 >> 6)] : "=";
    out += i + 2 < bytes.length ? chars[b2 & 63] : "=";
  }
  return out;
}
