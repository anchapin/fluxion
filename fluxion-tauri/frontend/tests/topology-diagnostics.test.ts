/**
 * Diagnostics tests (issue #3965): client-side structural checks mirroring
 * the #3964 linter vocabulary, lint-report ingestion, badge grouping.
 */

import { describe, expect, it } from "vitest";
import {
  computeClientDiagnostics,
  diagnosticsByNode,
  ingestLintReport,
} from "../src/topology/diagnostics";
import { CASE_600 } from "../src/topology/fixtures";
import { syntheticTwoZoneDoc } from "./topology-layout.test";

describe("computeClientDiagnostics", () => {
  it("reports zero findings for the healthy case-600 export", () => {
    expect(computeClientDiagnostics(CASE_600)).toEqual([]);
  });

  it("flags orphan nodes (E001_ORPHAN_NODE)", () => {
    const doc = syntheticTwoZoneDoc();
    doc.nodes.push({ id: "lonely", kind: "internal_mass", name: "Lonely mass", zone_id: "z0:air" });
    const findings = computeClientDiagnostics(doc);
    const orphan = findings.find((f) => f.code === "E001_ORPHAN_NODE" && f.nodeId === "lonely");
    expect(orphan).toBeDefined();
    expect(orphan!.severity).toBe("error");
    expect(orphan!.message).toContain("Lonely mass");
  });

  it("flags dangling edge references (E002_DANGLING_BOUNDARY)", () => {
    const doc = syntheticTwoZoneDoc();
    doc.edges.push({ source_id: "amb", target_id: "ghost-node", coupling_type: "conduction", bidirectional: true });
    const findings = computeClientDiagnostics(doc);
    const dangling = findings.find((f) => f.code === "E002_DANGLING_BOUNDARY");
    expect(dangling).toBeDefined();
    expect(dangling!.message).toContain("ghost-node");
  });

  it("flags non-positive capacitance (E007_NON_POSITIVE_CAPACITY)", () => {
    const doc = syntheticTwoZoneDoc();
    doc.nodes[1] = { ...doc.nodes[1], capacitance_j_per_k: 0 };
    const findings = computeClientDiagnostics(doc);
    expect(findings.some((f) => f.code === "E007_NON_POSITIVE_CAPACITY" && f.nodeId === "z0:air")).toBe(true);
  });

  it("flags split fractions outside [0,1] (E005_INVALID_SPLIT_FRACTION)", () => {
    const doc = syntheticTwoZoneDoc();
    doc.edges[6] = { ...doc.edges[6], fraction: 1.5 };
    const findings = computeClientDiagnostics(doc);
    expect(findings.some((f) => f.code === "E005_INVALID_SPLIT_FRACTION" && f.message.includes("1.5"))).toBe(true);
  });
});

describe("ingestLintReport", () => {
  it("merges external findings with client findings and dedups", () => {
    const client = [{ code: "E001_ORPHAN_NODE", severity: "error" as const, nodeId: "a", message: "m" }];
    const external = { clean: false, findings: [{ code: "E004_RECIPROCAL_MISMATCH", node_id: "b" }], source: "lint", strict: true };
    const merged = ingestLintReport(external, client);
    expect(merged).toHaveLength(2);
    const dup = ingestLintReport({ findings: [{ code: "E001_ORPHAN_NODE", node_id: "a" }] }, client);
    expect(dup).toHaveLength(1);
  });

  it("tolerates a missing report", () => {
    const merged = ingestLintReport(undefined, []);
    expect(merged).toEqual([]);
  });
});

describe("diagnosticsByNode", () => {
  it("groups findings per node and skips node-less findings", () => {
    const grouped = diagnosticsByNode([
      { code: "E001_ORPHAN_NODE", severity: "error", nodeId: "a", message: "1" },
      { code: "E007_NON_POSITIVE_CAPACITY", severity: "error", nodeId: "a", message: "2" },
      { code: "E002_DANGLING_BOUNDARY", severity: "error", message: "3" },
    ]);
    expect(grouped.get("a")).toHaveLength(2);
    expect(grouped.has("3")).toBe(false);
  });
});
