/**
 * Deterministic layout for the 2D topology section view (issue #3965).
 *
 * The layout reads like an architectural cross-section oriented along the
 * heat-transfer path, left → right:
 *
 *   [ outdoor ambient ] → [ exterior films ] → [ wall layers, in conduction
 *   depth order ] → [ interior films ] → [ zone air ] → [ zone attachments
 *   (internal mass / gains / HVAC) ]
 *
 * Within a zone band, assemblies stack vertically as a section silhouette:
 * roof on top, walls (ordered by azimuth) in the middle, floor at the bottom
 * (derived from each surface's tilt_deg). Multi-zone documents stack bands
 * vertically and air-exchange couplings route between the zone-air boxes.
 *
 * Pure module: same document in → identical layout out (unit-tested).
 */

import type { TopologyDocument, TopologyEdge, TopologyNode } from "./types";
import { ATTACHMENT_KINDS } from "./styling";

export interface PositionedNode {
  node: TopologyNode;
  x: number;
  y: number;
  w: number;
  h: number;
  /** x lane index (0 = ambient … laneCount-1 = attachments). */
  lane: number;
  /** zone band index (-1 for the global ambient column). */
  band: number;
  role: "ambient" | "assembly" | "zone_air" | "attachment";
}

export interface RoutedEdge {
  edge: TopologyEdge;
  /** SVG cubic-bezier path from source anchor to target anchor. */
  path: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface TopologyLayout {
  nodes: PositionedNode[];
  edges: RoutedEdge[];
  width: number;
  height: number;
  laneCount: number;
}

// ---- Geometry constants (px in diagram space) --------------------------------

const LANE_W = 168;
const NODE_W = 116;
const NODE_H = 46;
const ZONE_AIR_W = 132;
const ZONE_AIR_H_MAX = 200;
const AMBIENT_H = 120;
const ROW_GAP = 20;
const BAND_PAD = 32;
const BAND_GAP = 64;
const BAND_MIN_H = 300;
const MARGIN = 48;

function laneX(lane: number, w: number): number {
  return MARGIN + lane * LANE_W + (LANE_W - w) / 2;
}

// ---- Conduction-chain analysis ------------------------------------------------

interface Assembly {
  members: TopologyNode[];
  /** BFS hop distance from the exterior film; exterior film = 0. */
  depth: Map<string, number>;
  hasExterior: boolean;
  /** 0 = roof, 1 = wall, 2 = floor (fallback 1). */
  classRank: number;
  sortAzimuth: number;
  sortName: string;
}

function surfaceClassRank(n: TopologyNode): number {
  const tilt = n.tilt_deg;
  if (tilt !== undefined) {
    if (tilt <= 45) return 0; // roof / upward
    if (tilt >= 135) return 2; // floor / downward
    return 1; // wall
  }
  const name = n.name.toLowerCase();
  if (name.includes("roof") || name.includes("ceiling")) return 0;
  if (name.includes("floor")) return 2;
  return 1;
}

/** Connected components over the conduction subgraph = surface assemblies. */
function computeAssemblies(doc: TopologyDocument): Assembly[] {
  const conductionKinds = new Set(["exterior_surface", "wall_layer", "interior_surface"]);
  const members = doc.nodes.filter((n) => conductionKinds.has(n.kind));
  const adjacency = new Map<string, string[]>();
  for (const e of doc.edges) {
    if (e.coupling_type !== "conduction") continue;
    if (!members.some((n) => n.id === e.source_id) || !members.some((n) => n.id === e.target_id)) continue;
    push(adjacency, e.source_id, e.target_id);
    push(adjacency, e.target_id, e.source_id);
  }

  const byId = new Map(members.map((n) => [n.id, n]));
  const visited = new Set<string>();
  const assemblies: Assembly[] = [];

  for (const seed of members) {
    if (visited.has(seed.id)) continue;
    // BFS the component
    const component: string[] = [];
    const queue = [seed.id];
    visited.add(seed.id);
    while (queue.length > 0) {
      const id = queue.shift() as string;
      component.push(id);
      for (const next of adjacency.get(id) ?? []) {
        if (!visited.has(next) && byId.has(next)) {
          visited.add(next);
          queue.push(next);
        }
      }
    }
    const nodes = component.map((id) => byId.get(id) as TopologyNode).sort((a, b) => a.id.localeCompare(b.id));

    // Depth = BFS distance from the component's exterior film(s).
    const depth = new Map<string, number>();
    const exteriors = nodes.filter((n) => n.kind === "exterior_surface");
    const bfsSeeds = exteriors.length > 0 ? exteriors : [nodes[0]];
    const queue2: Array<{ id: string; d: number }> = bfsSeeds.map((n) => ({ id: n.id, d: 0 }));
    for (const s of queue2) depth.set(s.id, s.d);
    while (queue2.length > 0) {
      const { id, d } = queue2.shift() as { id: string; d: number };
      for (const next of adjacency.get(id) ?? []) {
        if (!depth.has(next) && byId.has(next)) {
          depth.set(next, d + 1);
          queue2.push({ id: next, d: d + 1 });
        }
      }
    }

    const orient = exteriors[0] ?? nodes[0];
    assemblies.push({
      members: nodes,
      depth,
      hasExterior: exteriors.length > 0,
      classRank: surfaceClassRank(orient),
      sortAzimuth: orient.azimuth_deg ?? 0,
      sortName: orient.name,
    });
  }

  assemblies.sort(
    (a, b) => a.classRank - b.classRank || a.sortAzimuth - b.sortAzimuth || a.sortName.localeCompare(b.sortName),
  );
  return assemblies;
}

function push<K, V>(map: Map<K, V[]>, key: K, value: V): void {
  const list = map.get(key);
  if (list) list.push(value);
  else map.set(key, [value]);
}

// ---- Main layout ---------------------------------------------------------------

export function computeLayout(doc: TopologyDocument): TopologyLayout {
  // Zone bands: one per zone_air node (ordered by id), plus a trailing band
  // for zone-scoped attachments whose zone has no zone_air, and unzoned nodes.
  const zoneAirs = doc.nodes.filter((n) => n.kind === "zone_air").sort((a, b) => a.id.localeCompare(b.id));
  const bandKeys: string[] = zoneAirs.map((z) => z.id);
  const extraBands = new Set<string>();
  for (const n of doc.nodes) {
    if (n.kind === "zone_air" || n.kind === "outdoor_ambient") continue;
    const key = n.zone_id ?? "(unzoned)";
    if (!bandKeys.includes(key) && !extraBands.has(key)) extraBands.add(key);
  }
  const allBands = [...bandKeys, ...[...extraBands].sort()];
  const bandIndex = new Map(allBands.map((k, i) => [k, i]));

  // Assemblies belong to the band of their zone_id (members inherit it).
  const assemblies = computeAssemblies(doc);

  // Lane geometry: 0 ambient | 1 exterior film | 2.. layers | interior film
  // (directly after the deepest layer) | zone air | attachments.
  // `maxDepth` includes interior films (depth = last layer + 1), so the
  // interior lane is maxDepth + 1, matching the natural chain position.
  let maxDepth = 0;
  for (const asm of assemblies) {
    for (const d of asm.depth.values()) maxDepth = Math.max(maxDepth, d);
  }
  const interiorLane = maxDepth + 1; // exterior film lane 1, layers 2..maxDepth
  const zoneAirLane = interiorLane + 1;
  const attachLane = zoneAirLane + 1;
  const laneCount = attachLane + 1;

  // ---- Vertical placement per band ----
  const positions = new Map<string, PositionedNode>();
  let y = MARGIN;

  for (const bandKey of allBands) {
    const bandTop = y;
    const bandAssemblies = assemblies.filter(
      (asm) => (asm.members[0].zone_id ?? "(unzoned)") === bandKey,
    );

    // Section silhouette rows: roof → walls → floor (assemblies pre-sorted).
    let rowY = bandTop + BAND_PAD;
    for (const asm of bandAssemblies) {
      const centerY = rowY + NODE_H / 2;
      for (const m of asm.members) {
        const d = asm.depth.get(m.id) ?? 0;
        const lane =
          m.kind === "exterior_surface" ? 1 : m.kind === "interior_surface" ? interiorLane : 1 + d;
        positions.set(m.id, {
          node: m,
          x: laneX(lane, NODE_W),
          y: centerY - NODE_H / 2,
          w: NODE_W,
          h: NODE_H,
          lane,
          band: bandIndex.get(bandKey) ?? -1,
          role: "assembly",
        });
      }
      rowY += NODE_H + ROW_GAP;
    }
    const rowsBottom = Math.max(rowY - ROW_GAP, bandTop);

    // Zone air box: vertically centered in the band.
    const zoneAir = zoneAirs.find((z) => z.id === bandKey);
    const attachments = doc.nodes
      .filter((n) => ATTACHMENT_KINDS.has(n.kind) && (n.zone_id ?? "(unzoned)") === bandKey)
      .sort((a, b) => a.kind.localeCompare(b.kind) || a.id.localeCompare(b.id));

    const rowsH = rowsBottom - bandTop;
    const bandH = Math.max(rowsH + BAND_PAD, BAND_MIN_H, attachments.length * (NODE_H + ROW_GAP) + 2 * BAND_PAD);
    const midY = bandTop + bandH / 2;

    if (zoneAir) {
      const zh = Math.min(ZONE_AIR_H_MAX, bandH - 2 * BAND_PAD);
      positions.set(zoneAir.id, {
        node: zoneAir,
        x: laneX(zoneAirLane, ZONE_AIR_W),
        y: midY - zh / 2,
        w: ZONE_AIR_W,
        h: zh,
        lane: zoneAirLane,
        band: bandIndex.get(bandKey) ?? -1,
        role: "zone_air",
      });
    }

    // Attachments stacked around the band middle.
    const attachTotal = attachments.length * NODE_H + (attachments.length - 1) * ROW_GAP;
    let ay = midY - attachTotal / 2;
    for (const n of attachments) {
      positions.set(n.id, {
        node: n,
        x: laneX(attachLane, NODE_W),
        y: ay,
        w: NODE_W,
        h: NODE_H,
        lane: attachLane,
        band: bandIndex.get(bandKey) ?? -1,
        role: "attachment",
      });
      ay += NODE_H + ROW_GAP;
    }

    y = bandTop + bandH + BAND_GAP;
  }

  const contentBottom = Math.max(y - BAND_GAP, MARGIN);
  const height = contentBottom + MARGIN;

  // Ambient column spans every band, vertically centered.
  for (const n of doc.nodes) {
    if (n.kind !== "outdoor_ambient") continue;
    positions.set(n.id, {
      node: n,
      x: laneX(0, NODE_W),
      y: height / 2 - AMBIENT_H / 2,
      w: NODE_W,
      h: AMBIENT_H,
      lane: 0,
      band: -1,
      role: "ambient",
    });
  }

  const width = MARGIN * 2 + laneCount * LANE_W;

  const nodes = [...positions.values()].sort((a, b) => a.node.id.localeCompare(b.node.id));

  // ---- Edge routing ----
  const edges: RoutedEdge[] = [];
  for (const e of doc.edges) {
    const s = positions.get(e.source_id);
    const t = positions.get(e.target_id);
    if (!s || !t) continue; // dangling edge — surfaced by diagnostics
    const forward = t.x >= s.x;
    const x1 = forward ? s.x + s.w : s.x;
    const y1 = s.y + s.h / 2;
    const x2 = forward ? t.x : t.x + t.w;
    const y2 = t.y + t.h / 2;
    const dx = Math.min(120, Math.max(30, Math.abs(x2 - x1) * 0.45));
    const c1x = forward ? x1 + dx : x1 - dx;
    const c2x = forward ? x2 - dx : x2 + dx;
    edges.push({
      edge: e,
      path: `M ${x1.toFixed(1)} ${y1.toFixed(1)} C ${c1x.toFixed(1)} ${y1.toFixed(1)}, ${c2x.toFixed(1)} ${y2.toFixed(1)}, ${x2.toFixed(1)} ${y2.toFixed(1)}`,
      x1,
      y1,
      x2,
      y2,
    });
  }

  return { nodes, edges, width, height, laneCount };
}

/** Adjacency index for selection highlighting (upstream / downstream). */
export interface Adjacency {
  upstream: Map<string, Set<string>>;
  downstream: Map<string, Set<string>>;
  edgesByNode: Map<string, Set<number>>;
  edgeEndpoints: Map<number, { source: string; target: string }>;
}

export function computeAdjacency(_doc: TopologyDocument, layout: TopologyLayout): Adjacency {
  const upstream = new Map<string, Set<string>>();
  const downstream = new Map<string, Set<string>>();
  const edgesByNode = new Map<string, Set<number>>();
  const edgeEndpoints = new Map<number, { source: string; target: string }>();
  for (const re of layout.edges) {
    const idx = edgeEndpoints.size;
    edgeEndpoints.set(idx, { source: re.edge.source_id, target: re.edge.target_id });
    const s = re.edge.source_id;
    const t = re.edge.target_id;
    pushToSet(downstream, s, t);
    pushToSet(upstream, t, s);
    pushToSet(edgesByNode, s, idx);
    pushToSet(edgesByNode, t, idx);
  }
  return { upstream, downstream, edgesByNode, edgeEndpoints };
}

function pushToSet(map: Map<string, Set<string | number>>, key: string, value: string | number): void {
  const set = map.get(key);
  if (set) set.add(value);
  else map.set(key, new Set([value]));
}
