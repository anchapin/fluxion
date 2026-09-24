/**
 * Kind-based styling maps for the topology viewer (issue #3965).
 * Pure data — fully unit-tested for kind-union exhaustiveness.
 *
 * Palette extends the app's dark-navy vocabulary (#1a1a2e / #16213e /
 * #0f3460 / #e94560, see styles.css) with per-kind hues chosen for
 * distinguishability on that background.
 */

import type { EdgeKind, NodeKind } from "./types";
import { EDGE_KINDS, NODE_KINDS } from "./types";

export interface NodeStyle {
  /** Short label for legends and the inspector. */
  label: string;
  /** Fill color for the node box. */
  fill: string;
  /** Stroke color for the node box. */
  stroke: string;
}

export interface EdgeStyle {
  /** Short label for legends and the inspector. */
  label: string;
  /** Stroke color for the edge path and arrowhead. */
  color: string;
  /** SVG stroke-dasharray; undefined = solid. */
  dash?: string;
}

export const NODE_STYLES: Readonly<Record<NodeKind, NodeStyle>> = {
  outdoor_ambient: { label: "Outdoor ambient", fill: "#3d1f2e", stroke: "#e94560" },
  exterior_surface: { label: "Exterior film", fill: "#42291b", stroke: "#f9844a" },
  wall_layer: { label: "Wall layer", fill: "#232c3d", stroke: "#8d99ae" },
  interior_surface: { label: "Interior film", fill: "#1b3a33", stroke: "#43aa8b" },
  internal_mass: { label: "Internal mass", fill: "#3a2b33", stroke: "#b5838d" },
  zone_air: { label: "Zone air", fill: "#12314a", stroke: "#4cc9f0" },
  internal_gain: { label: "Internal gain", fill: "#3d3418", stroke: "#f9c74e" },
  hvac_terminal: { label: "HVAC terminal", fill: "#263a22", stroke: "#90be6d" },
};

export const EDGE_STYLES: Readonly<Record<EdgeKind, EdgeStyle>> = {
  conduction: { label: "Conduction", color: "#f9844a" },
  convection_exterior: { label: "Ext. convection", color: "#5fa8d3" },
  convection_interior: { label: "Int. convection", color: "#43aa8b" },
  longwave_radiation: { label: "Longwave radiation", color: "#9d4edd", dash: "6 3" },
  shortwave_solar_direct: { label: "Direct solar", color: "#ffd166", dash: "2 3" },
  shortwave_solar_diffuse: { label: "Diffuse solar", color: "#ffe8a3", dash: "2 3" },
  air_exchange: { label: "Air exchange", color: "#4cc9f0", dash: "8 4" },
  internal_gain_split: { label: "Gain split", color: "#f9c74e" },
  hvac_sensible: { label: "HVAC sensible", color: "#90be6d" },
};

/** Node kinds rendered in the final "zone attachment" lane (right of zone air). */
export const ATTACHMENT_KINDS: ReadonlySet<NodeKind> = new Set<NodeKind>([
  "internal_mass",
  "internal_gain",
  "hvac_terminal",
]);

export function nodeStyle(kind: NodeKind): NodeStyle {
  return NODE_STYLES[kind];
}

export function edgeStyle(kind: EdgeKind): EdgeStyle {
  return EDGE_STYLES[kind];
}

/** Kinds actually present in a document, in canonical enum order (for legends). */
export function kindsPresent(nodes: readonly { kind: NodeKind }[]): NodeKind[] {
  const present = new Set(nodes.map((n) => n.kind));
  return NODE_KINDS.filter((k) => present.has(k));
}

export function edgeKindsPresent(edges: readonly { coupling_type: EdgeKind }[]): EdgeKind[] {
  const present = new Set(edges.map((e) => e.coupling_type));
  return EDGE_KINDS.filter((k) => present.has(k));
}

/**
 * Stroke width for an edge. Base uniform width; when the flux overlay is on,
 * the width is scaled by log-conductance so heavier thermal couplings (W/K)
 * read thicker. Edges without a conductance keep the base width.
 */
export function edgeStrokeWidth(
  conductance: number | undefined,
  fluxOverlay: boolean,
): number {
  const base = 1.6;
  if (!fluxOverlay || conductance === undefined || !Number.isFinite(conductance) || conductance <= 0) {
    return base;
  }
  // log10 mapping over the practical range [0.1, 1000] W/K → [1.0, 7.0] px.
  const t = (Math.log10(conductance) + 1) / 4; // 0.1 → 0, 1000 → 1
  const clamped = Math.min(1, Math.max(0, t));
  return Math.round((1.0 + clamped * 6.0) * 10) / 10;
}
