/**
 * View-model tests (issue #3965): topology → view-model mapping, badge
 * attachment, highlight sets, document parsing/validation, and fixture↔schema
 * consistency guards.
 */

import { describe, expect, it } from "vitest";
import { buildViewModel, highlightFor, keyEdge, parseTopologyDocument } from "../src/topology/viewModel";
import { CASE_600, CASE_900, FIXTURES } from "../src/topology/fixtures";
import { NODE_KINDS } from "../src/topology/types";
import { syntheticTwoZoneDoc } from "./topology-layout.test";

describe("fixtures", () => {
  it("ship case 600 and case 900 with self-consistent metadata", () => {
    expect(FIXTURES.map((f) => f.id)).toEqual(["case600", "case900"]);
    for (const doc of [CASE_600, CASE_900]) {
      expect(doc.metadata.node_count).toBe(doc.nodes.length);
      expect(doc.metadata.edge_count).toBe(doc.edges.length);
      expect(doc.metadata.schema_version).toBe("1.0.0");
      for (const n of doc.nodes) expect(NODE_KINDS).toContain(n.kind);
    }
  });

  it("case 600 is low mass and case 900 is high mass", () => {
    const cap = (doc: typeof CASE_600): number =>
      doc.nodes.reduce((acc, n) => acc + (n.capacitance_j_per_k ?? 0), 0);
    expect(cap(CASE_900)).toBeGreaterThan(cap(CASE_600));
  });
});

describe("buildViewModel", () => {
  const vm = buildViewModel(CASE_600, { fluxOverlay: false });

  it("maps every node to a style and every edge to a style", () => {
    expect(vm.nodes).toHaveLength(CASE_600.nodes.length);
    expect(vm.edges).toHaveLength(CASE_600.edges.length);
    for (const n of vm.nodes) {
      expect(n.style.stroke).toMatch(/^#/);
      expect(n.badges).toEqual([]); // healthy export
    }
    for (const e of vm.edges) expect(e.style.color).toMatch(/^#/);
  });

  it("uses uniform stroke widths without the conductance overlay", () => {
    for (const e of vm.edges) expect(e.strokeWidth).toBe(1.6);
  });

  it("scales stroke widths by conductance with the overlay on", () => {
    const overlay = buildViewModel(CASE_600, { fluxOverlay: true });
    const widths = overlay.edges.map((e) => e.strokeWidth);
    expect(Math.max(...widths)).toBeGreaterThan(Math.min(...widths));
    // The heaviest coupling (window conduction, ~122 W/K) must be thick.
    const byK = [...overlay.edges].sort((a, b) => (b.edge.conductance_w_per_k ?? 0) - (a.edge.conductance_w_per_k ?? 0));
    expect(byK[0].strokeWidth).toBeGreaterThan(1.6);
  });

  it("attaches badges from a supplied lint report", () => {
    const vm2 = buildViewModel(CASE_600, {
      fluxOverlay: false,
      lintReport: { findings: [{ code: "E004_RECIPROCAL_MISMATCH", node_id: "zone-0:air", message: "mismatch" }] },
    });
    const zoneAir = vm2.nodeViewsById.get("zone-0:air");
    expect(zoneAir!.badges.map((b) => b.code)).toContain("E004_RECIPROCAL_MISMATCH");
    expect(vm2.diagnostics).toHaveLength(1);
  });

  it("carries document summary fields", () => {
    expect(vm.modelName).toBe(CASE_600.metadata.model_name);
    expect(vm.nodeCount).toBe(34);
    expect(vm.edgeCount).toBe(47);
    expect(vm.legendNodeKinds.length).toBeGreaterThan(0);
  });
});

describe("highlightFor (upstream/downstream highlighting)", () => {
  const vm = buildViewModel(CASE_600, { fluxOverlay: false });
  const zoneAirId = "zone-0:air";

  it("highlights a selected node plus its upstream sources and downstream sinks", () => {
    const h = highlightFor(vm, { type: "node", id: zoneAirId });
    expect(h.nodeIds.has(zoneAirId)).toBe(true);
    expect(h.upstream.length).toBeGreaterThan(0);
    expect(h.downstream.length).toBeGreaterThan(0);
    for (const id of h.upstream) expect(h.nodeIds.has(id)).toBe(true);
    for (const id of h.downstream) expect(h.nodeIds.has(id)).toBe(true);
    expect(h.edgeIdxs.size).toBeGreaterThan(0);
  });

  it("highlights the endpoints of a selected edge", () => {
    const ev = vm.edges[0];
    const h = highlightFor(vm, { type: "edge", id: keyEdge(ev.edge) });
    expect(h.nodeIds).toEqual(new Set([ev.edge.source_id, ev.edge.target_id]));
    expect(h.edgeIdxs.size).toBe(1);
  });

  it("returns empty sets for no selection or an unknown id", () => {
    expect(highlightFor(vm, null).nodeIds.size).toBe(0);
    expect(highlightFor(vm, { type: "node", id: "nope" }).nodeIds.size).toBe(0);
  });
});

describe("parseTopologyDocument", () => {
  it("accepts a real export", () => {
    const { doc, error } = parseTopologyDocument(JSON.stringify(CASE_600));
    expect(error).toBeUndefined();
    expect(doc!.metadata.model_name).toBe(CASE_600.metadata.model_name);
  });

  it("accepts the synthetic two-zone document", () => {
    const { doc, error } = parseTopologyDocument(JSON.stringify(syntheticTwoZoneDoc()));
    expect(error).toBeUndefined();
    expect(doc!.nodes).toHaveLength(11);
  });

  it("rejects invalid JSON with a readable error", () => {
    const { error } = parseTopologyDocument("{not json");
    expect(error).toMatch(/Invalid JSON/);
  });

  it("rejects documents that are not topology exports", () => {
    const { error } = parseTopologyDocument(JSON.stringify({ hello: "world" }));
    expect(error).toMatch(/Not a topology_v1 document/);
  });

  it("rejects unknown node kinds (schema drift guard)", () => {
    const doc = syntheticTwoZoneDoc();
    doc.nodes.push({ id: "x", kind: "solar_collector" as never, name: "X" });
    const { error } = parseTopologyDocument(JSON.stringify(doc));
    expect(error).toMatch(/Unknown node kind/);
  });

  it("rejects unknown edge kinds", () => {
    const doc = syntheticTwoZoneDoc();
    doc.edges.push({ source_id: "amb", target_id: "z0:air", coupling_type: "teleportation" as never, bidirectional: true });
    const { error } = parseTopologyDocument(JSON.stringify(doc));
    expect(error).toMatch(/Unknown edge kind/);
  });
});
