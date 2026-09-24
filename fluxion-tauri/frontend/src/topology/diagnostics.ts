/**
 * Structural diagnostics for the topology viewer (issue #3965).
 *
 * Two sources feed the on-node diagnostic badges:
 *  1. an external lint report from `fluxion topology lint` (issue #3964,
 *     codes E001–E007), ingested verbatim when supplied;
 *  2. cheap client-side recomputation (orphan nodes, dangling edges,
 *     non-positive capacitance) so drag-dropped documents get immediate
 *     feedback even without a lint report.
 *
 * Pure module.
 */

import type { LintReport, LintFinding, TopologyDocument } from "./types";

export interface ViewerDiagnostic {
  code: string;
  severity: "error" | "warning" | "info";
  nodeId?: string;
  message: string;
}

/** Merge an external lint report into the client-side findings (dedup by
 *  code+nodeId, external findings win). */
export function ingestLintReport(report: LintReport | undefined, client: ViewerDiagnostic[]): ViewerDiagnostic[] {
  const external: ViewerDiagnostic[] = (report?.findings ?? []).map((f: LintFinding) => ({
    code: f.code,
    severity: f.severity ?? "error",
    nodeId: f.node_id,
    message: f.message ?? f.code,
  }));
  const key = (d: ViewerDiagnostic) => `${d.code}|${d.nodeId ?? ""}`;
  const seen = new Set(external.map(key));
  const merged = [...external];
  for (const d of client) {
    if (!seen.has(key(d))) merged.push(d);
  }
  return merged;
}

/** Client-side recomputation of the cheap structural checks. Codes mirror
 *  the #3964 linter vocabulary: E001 orphans, dangling edge refs, E007
 *  non-positive capacity. */
export function computeClientDiagnostics(doc: TopologyDocument): ViewerDiagnostic[] {
  const findings: ViewerDiagnostic[] = [];
  const byId = new Set(doc.nodes.map((n) => n.id));
  const connected = new Set<string>();
  for (const e of doc.edges) {
    connected.add(e.source_id);
    connected.add(e.target_id);
  }

  for (const n of doc.nodes) {
    if (!connected.has(n.id)) {
      findings.push({
        code: "E001_ORPHAN_NODE",
        severity: "error",
        nodeId: n.id,
        message: `Node "${n.name}" (${n.kind}) has no incoming or outgoing edges.`,
      });
    }
    if (n.capacitance_j_per_k !== undefined && n.capacitance_j_per_k <= 0) {
      findings.push({
        code: "E007_NON_POSITIVE_CAPACITY",
        severity: "error",
        nodeId: n.id,
        message: `Node "${n.name}" has non-positive capacitance (${n.capacitance_j_per_k} J/K).`,
      });
    }
  }

  for (const e of doc.edges) {
    const missing: string[] = [];
    if (!byId.has(e.source_id)) missing.push(e.source_id);
    if (!byId.has(e.target_id)) missing.push(e.target_id);
    if (missing.length > 0) {
      findings.push({
        code: "E002_DANGLING_BOUNDARY",
        severity: "error",
        message: `Edge ${e.source_id} → ${e.target_id} (${e.coupling_type}) references missing node(s): ${missing.join(", ")}.`,
      });
    }
    if (e.fraction !== undefined && (e.fraction < 0 || e.fraction > 1)) {
      findings.push({
        code: "E005_INVALID_SPLIT_FRACTION",
        severity: "error",
        message: `Edge ${e.source_id} → ${e.target_id} has fraction ${e.fraction} outside [0, 1].`,
      });
    }
  }

  return findings;
}

/** Group diagnostics by node id for badge rendering. */
export function diagnosticsByNode(diags: readonly ViewerDiagnostic[]): Map<string, ViewerDiagnostic[]> {
  const map = new Map<string, ViewerDiagnostic[]>();
  for (const d of diags) {
    if (!d.nodeId) continue;
    const list = map.get(d.nodeId);
    if (list) list.push(d);
    else map.set(d.nodeId, [d]);
  }
  return map;
}
