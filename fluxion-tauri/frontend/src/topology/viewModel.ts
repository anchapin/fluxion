/**
 * Topology document → view model mapping (issue #3965).
 * Merges layout, kind styling, flux-overlay widths, diagnostic badges and
 * selection-highlight sets into a single render-ready structure.
 *
 * Pure module.
 */

import type { LintReport, TopologyDocument } from "./types";
import type { Adjacency, PositionedNode, RoutedEdge, TopologyLayout } from "./layout";
import { computeLayout, computeAdjacency } from "./layout";
import { nodeStyle, edgeStyle, edgeStrokeWidth, kindsPresent, edgeKindsPresent } from "./styling";
import type { ViewerDiagnostic } from "./diagnostics";
import { computeClientDiagnostics, ingestLintReport, diagnosticsByNode } from "./diagnostics";
import type { EdgeKind, NodeKind } from "./types";

export interface NodeView extends PositionedNode {
  style: ReturnType<typeof nodeStyle>;
  badges: ViewerDiagnostic[];
}

export interface EdgeView extends RoutedEdge {
  style: ReturnType<typeof edgeStyle>;
  strokeWidth: number;
}

export interface TopologyViewModel {
  layout: TopologyLayout;
  adjacency: Adjacency;
  nodes: NodeView[];
  edges: EdgeView[];
  nodeViewsById: Map<string, NodeView>;
  diagnostics: ViewerDiagnostic[];
  diagnosticsByNodeId: Map<string, ViewerDiagnostic[]>;
  legendNodeKinds: NodeKind[];
  legendEdgeKinds: EdgeKind[];
  modelName: string;
  modelSource: string;
  nodeCount: number;
  edgeCount: number;
}

export interface ViewModelOptions {
  /** Scale edge strokes by log-conductance when true. */
  fluxOverlay: boolean;
  /** Optional lint report from `fluxion topology lint` (#3964). */
  lintReport?: LintReport;
}

export function buildViewModel(doc: TopologyDocument, opts: ViewModelOptions): TopologyViewModel {
  const layout = computeLayout(doc);
  const adjacency = computeAdjacency(doc, layout);
  const diagnostics = ingestLintReport(opts.lintReport, computeClientDiagnostics(doc));
  const byNode = diagnosticsByNode(diagnostics);

  const nodes: NodeView[] = layout.nodes.map((p: PositionedNode) => ({
    ...p,
    style: nodeStyle(p.node.kind),
    badges: byNode.get(p.node.id) ?? [],
  }));

  const edges: EdgeView[] = layout.edges.map((r: RoutedEdge) => ({
    ...r,
    style: edgeStyle(r.edge.coupling_type),
    strokeWidth: edgeStrokeWidth(r.edge.conductance_w_per_k, opts.fluxOverlay),
  }));

  return {
    layout,
    adjacency,
    nodes,
    edges,
    nodeViewsById: new Map(nodes.map((n) => [n.node.id, n])),
    diagnostics,
    diagnosticsByNodeId: byNode,
    legendNodeKinds: kindsPresent(doc.nodes),
    legendEdgeKinds: edgeKindsPresent(doc.edges),
    modelName: doc.metadata.model_name,
    modelSource: doc.metadata.model_source,
    nodeCount: doc.nodes.length,
    edgeCount: doc.edges.length,
  };
}

/** Selection: which nodes/edges stay highlighted (selected entity plus its
 *  upstream sources and downstream sinks). */
export function highlightFor(
  vm: TopologyViewModel,
  selection: { type: "node" | "edge"; id: string } | null,
): { nodeIds: Set<string>; edgeIdxs: Set<number>; upstream: string[]; downstream: string[] } {
  const empty = { nodeIds: new Set<string>(), edgeIdxs: new Set<number>(), upstream: [] as string[], downstream: [] as string[] };
  if (!selection) return empty;

  if (selection.type === "node") {
    if (!vm.nodeViewsById.has(selection.id)) return empty;
    const nodeIds = new Set<string>([selection.id]);
    const upstream = [...(vm.adjacency.upstream.get(selection.id) ?? [])];
    const downstream = [...(vm.adjacency.downstream.get(selection.id) ?? [])];
    for (const n of upstream) nodeIds.add(n);
    for (const n of downstream) nodeIds.add(n);
    const edgeIdxs = new Set<number>(vm.adjacency.edgesByNode.get(selection.id) ?? []);
    return { nodeIds, edgeIdxs, upstream, downstream };
  }

  // Edge selection: highlight the edge and its two endpoints.
  const idx = vm.edges.findIndex((e) => keyEdge(e.edge) === selection.id);
  if (idx < 0) return empty;
  const e = vm.edges[idx].edge;
  return {
    nodeIds: new Set([e.source_id, e.target_id]),
    edgeIdxs: new Set([idx]),
    upstream: [e.source_id],
    downstream: [e.target_id],
  };
}

export function keyEdge(e: { source_id: string; target_id: string; coupling_type: string }): string {
  return `${e.source_id}→${e.target_id}:${e.coupling_type}`;
}

/**
 * Parse and structurally validate a topology document from raw JSON text
 * (drag-and-drop, file input, `?model=` fetch). Kinds are checked against
 * the v1 enums; anything else is rejected with a human-readable message.
 */
export function parseTopologyDocument(text: string): { doc?: TopologyDocument; error?: string } {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (err) {
    return { error: `Invalid JSON: ${(err as Error).message}` };
  }
  if (typeof raw !== "object" || raw === null) return { error: "Document must be a JSON object." };
  const obj = raw as Record<string, unknown>;
  if (!Array.isArray(obj.nodes) || !Array.isArray(obj.edges) || typeof obj.metadata !== "object" || obj.metadata === null) {
    return { error: "Not a topology_v1 document: expected `metadata`, `nodes`, and `edges`." };
  }
  const validNodeKinds = new Set<string>([
    "outdoor_ambient", "exterior_surface", "wall_layer", "interior_surface",
    "internal_mass", "zone_air", "internal_gain", "hvac_terminal",
  ]);
  const validEdgeKinds = new Set<string>([
    "conduction", "convection_exterior", "convection_interior", "longwave_radiation",
    "shortwave_solar_direct", "shortwave_solar_diffuse", "air_exchange",
    "internal_gain_split", "hvac_sensible",
  ]);
  for (const n of obj.nodes as Record<string, unknown>[]) {
    if (typeof n.id !== "string" || typeof n.name !== "string" || typeof n.kind !== "string") {
      return { error: "Node missing id/name/kind." };
    }
    if (!validNodeKinds.has(n.kind)) return { error: `Unknown node kind "${String(n.kind)}" (node ${String(n.id)}).` };
  }
  for (const e of obj.edges as Record<string, unknown>[]) {
    if (typeof e.source_id !== "string" || typeof e.target_id !== "string") {
      return { error: "Edge missing source_id/target_id." };
    }
    if (typeof e.coupling_type !== "string" || !validEdgeKinds.has(e.coupling_type)) {
      return { error: `Unknown edge kind "${String(e.coupling_type)}" (${String(e.source_id)} → ${String(e.target_id)}).` };
    }
  }
  return { doc: obj as unknown as TopologyDocument };
}
