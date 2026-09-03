import type { Vertex } from "../types/geometry";

/**
 * Planar polygon fan triangulation (ear-clip equivalent for the convex
 * surfaces the geometry service emits). Ported from the preserved viewers'
 * `THREE.ShapeGeometry` usage so meshes can be built from raw BEM vertices.
 *
 * Returns a flat xyz Float32Array of `(n - 2)` triangles in the polygon's
 * own coordinate space, or `null` for degenerate inputs (<3 vertices,
 * collinear points / zero enclosed area).
 */
export function triangulatePolygon(
  vertices: Vertex[],
  normal: Vertex,
): Float32Array | null {
  const n = vertices.length;
  if (n < 3) return null;

  // Newell's method: plane normal whose magnitude is twice the enclosed
  // area — zero means the polygon is degenerate.
  let nx = 0;
  let ny = 0;
  let nz = 0;
  for (let i = 0; i < n; i++) {
    const a = vertices[i];
    const b = vertices[(i + 1) % n];
    nx += (a.y - b.y) * (a.z + b.z);
    ny += (a.z - b.z) * (a.x + b.x);
    nz += (a.x - b.x) * (a.y + b.y);
  }
  const len = Math.hypot(nx, ny, nz);
  if (len < 1e-9) return null;
  nx /= len;
  ny /= len;
  nz /= len;

  // Honor the caller's outward normal: reverse winding when Newell's
  // normal points the other way so lighting sees the front face.
  let ring = vertices;
  const alignment = normal.x * nx + normal.y * ny + normal.z * nz;
  if (alignment < 0) ring = [...vertices].reverse();

  // Orthonormal basis (u, v) spanning the polygon plane.
  const helper = Math.abs(ny) < 0.9 ? { x: 0, y: 1, z: 0 } : { x: 1, y: 0, z: 0 };
  let ux = helper.y * nz - helper.z * ny;
  let uy = helper.z * nx - helper.x * nz;
  let uz = helper.x * ny - helper.y * nx;
  const ul = Math.hypot(ux, uy, uz);
  if (ul < 1e-9) return null;
  ux /= ul;
  uy /= ul;
  uz /= ul;
  const vx = ny * uz - nz * uy;
  const vy = nz * ux - nx * uz;
  const vz = nx * uy - ny * ux;

  // Project onto (u, v) and fan-triangulate around vertex 0.
  const out = new Float32Array((n - 2) * 9);
  const p = ring.map((pt) => ({
    x: pt.x,
    y: pt.y,
    z: pt.z,
    u: pt.x * ux + pt.y * uy + pt.z * uz,
    v: pt.x * vx + pt.y * vy + pt.z * vz,
  }));
  let o = 0;
  for (let i = 1; i < n - 1; i++) {
    const a = p[0];
    const b = p[i];
    const c = p[i + 1];
    out[o++] = a.x;
    out[o++] = a.y;
    out[o++] = a.z;
    out[o++] = b.x;
    out[o++] = b.y;
    out[o++] = b.z;
    out[o++] = c.x;
    out[o++] = c.y;
    out[o++] = c.z;
  }
  return out;
}
