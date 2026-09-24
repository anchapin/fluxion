/**
 * Layout engine tests (issue #3965): lane ordering, section silhouette,
 * multi-zone bands, determinism, edge routing anchors.
 */

import { describe, expect, it } from "vitest";
import { computeLayout } from "../src/topology/layout";
import type { TopologyDocument, TopologyEdge, TopologyNode } from "../src/topology/types";
import { CASE_600, CASE_900 } from "../src/topology/fixtures";

/** Synthetic two-zone document with an air-exchange coupling between zones. */
export function syntheticTwoZoneDoc(): TopologyDocument {
  const nodes: TopologyNode[] = [
    { id: "amb", kind: "outdoor_ambient", name: "Outdoor ambient" },
    { id: "z0:air", kind: "zone_air", name: "Zone 0 air", zone_id: "z0:air", volume_m3: 100 },
    { id: "z1:air", kind: "zone_air", name: "Zone 1 air", zone_id: "z1:air", volume_m3: 80 },
    { id: "z0:wall-s:ext", kind: "exterior_surface", name: "Z0 south ext film", zone_id: "z0:air", tilt_deg: 90, azimuth_deg: 180, area_m2: 12 },
    { id: "z0:wall-s:l0", kind: "wall_layer", name: "Z0 south layer 0", zone_id: "z0:air", capacitance_j_per_k: 5e6 },
    { id: "z0:wall-s:l1", kind: "wall_layer", name: "Z0 south layer 1", zone_id: "z0:air", capacitance_j_per_k: 2e6 },
    { id: "z0:wall-s:int", kind: "interior_surface", name: "Z0 south int film", zone_id: "z0:air" },
    { id: "z0:hvac", kind: "hvac_terminal", name: "Z0 HVAC terminal", zone_id: "z0:air" },
    { id: "z0:gain", kind: "internal_gain", name: "Z0 internal gains", zone_id: "z0:air" },
    { id: "z1:roof:ext", kind: "exterior_surface", name: "Z1 roof ext film", zone_id: "z1:air", tilt_deg: 0 },
    { id: "z1:roof:int", kind: "interior_surface", name: "Z1 roof int film", zone_id: "z1:air" },
  ];
  const edges: TopologyEdge[] = [
    { source_id: "amb", target_id: "z0:wall-s:ext", coupling_type: "shortwave_solar_direct", bidirectional: false },
    { source_id: "z0:wall-s:ext", target_id: "z0:wall-s:l0", coupling_type: "conduction", conductance_w_per_k: 30, bidirectional: true },
    { source_id: "z0:wall-s:l0", target_id: "z0:wall-s:l1", coupling_type: "conduction", conductance_w_per_k: 25, bidirectional: true },
    { source_id: "z0:wall-s:l1", target_id: "z0:wall-s:int", coupling_type: "conduction", conductance_w_per_k: 20, bidirectional: true },
    { source_id: "z0:wall-s:int", target_id: "z0:air", coupling_type: "convection_interior", conductance_w_per_k: 12, bidirectional: true },
    { source_id: "z0:air", target_id: "z0:hvac", coupling_type: "hvac_sensible", conductance_w_per_k: 50, bidirectional: true },
    { source_id: "z0:gain", target_id: "z0:air", coupling_type: "internal_gain_split", fraction: 0.6, bidirectional: false },
    { source_id: "amb", target_id: "z1:roof:ext", coupling_type: "convection_exterior", conductance_w_per_k: 80, bidirectional: true },
    { source_id: "z1:roof:ext", target_id: "z1:roof:int", coupling_type: "conduction", conductance_w_per_k: 60, bidirectional: true },
    { source_id: "z1:roof:int", target_id: "z1:air", coupling_type: "convection_interior", conductance_w_per_k: 9, bidirectional: true },
    { source_id: "z0:air", target_id: "z1:air", coupling_type: "air_exchange", conductance_w_per_k: 5, bidirectional: true },
  ];
  return {
    metadata: {
      schema_version: "1.0.0",
      tool_version: "test",
      model_name: "synthetic-2zone",
      model_source: "unit-test",
      node_count: nodes.length,
      edge_count: edges.length,
    },
    nodes,
    edges,
  };
}

describe("computeLayout", () => {
  it("orders lanes along the heat-transfer path (case 600)", () => {
    const layout = computeLayout(CASE_600);
    const ambient = layout.nodes.find((n) => n.node.kind === "outdoor_ambient");
    const exterior = layout.nodes.find((n) => n.node.kind === "exterior_surface");
    const layer = layout.nodes.find((n) => n.node.kind === "wall_layer");
    const interior = layout.nodes.find((n) => n.node.kind === "interior_surface");
    const zoneAir = layout.nodes.find((n) => n.node.kind === "zone_air");
    const attachment = layout.nodes.find((n) => n.role === "attachment");
    expect(ambient && exterior && layer && interior && zoneAir && attachment).toBeTruthy();

    expect(ambient!.lane).toBe(0);
    expect(exterior!.lane).toBe(1);
    expect(layer!.lane).toBeGreaterThan(exterior!.lane);
    expect(layer!.lane).toBeLessThan(interior!.lane);
    expect(zoneAir!.lane).toBe(interior!.lane + 1);
    expect(attachment!.lane).toBe(zoneAir!.lane + 1);
    expect(layout.laneCount).toBe(attachment!.lane + 1);
  });

  it("stacks the section silhouette: roof above walls above floor (case 600)", () => {
    const layout = computeLayout(CASE_600);
    const yOf = (pred: (n: ReturnType<typeof computeLayout>["nodes"][number]) => boolean): number => {
      const n = layout.nodes.find(pred);
      if (!n) throw new Error("no matching node");
      return n.y;
    };
    const roofY = yOf((n) => n.node.kind === "exterior_surface" && n.node.tilt_deg === 0);
    const wallY = yOf((n) => n.node.kind === "exterior_surface" && n.node.tilt_deg === 90);
    const floorY = yOf((n) => n.node.kind === "exterior_surface" && n.node.tilt_deg === 180);
    expect(roofY).toBeLessThan(wallY);
    expect(wallY).toBeLessThan(floorY);
  });

  it("orders walls by azimuth within the wall band (case 600)", () => {
    const layout = computeLayout(CASE_600);
    const walls = layout.nodes
      .filter((n) => n.node.kind === "exterior_surface" && n.node.tilt_deg === 90)
      .sort((a, b) => a.y - b.y);
    expect(walls.length).toBeGreaterThanOrEqual(2);
    for (let i = 1; i < walls.length; i++) {
      expect(walls[i].node.azimuth_deg ?? 0).toBeGreaterThanOrEqual(walls[i - 1].node.azimuth_deg ?? 0);
    }
  });

  it("places deeper conduction layers further right (case 900)", () => {
    const layout = computeLayout(CASE_900);
    const find = (id: string): number => {
      const n = layout.nodes.find((p) => p.node.id === id);
      if (!n) throw new Error(`missing ${id}`);
      return n.lane;
    };
    expect(find("zone-0:floor:layer-0")).toBeLessThan(find("zone-0:floor:layer-1"));
    expect(find("zone-0:floor:layer-1")).toBeLessThan(find("zone-0:floor:int"));
    expect(find("zone-0:roof:ext")).toBe(1);
  });

  it("stacks zone bands for multi-zone documents and routes inter-zone edges", () => {
    const layout = computeLayout(syntheticTwoZoneDoc());
    const z0 = layout.nodes.find((n) => n.node.id === "z0:air");
    const z1 = layout.nodes.find((n) => n.node.id === "z1:air");
    expect(z0 && z1).toBeTruthy();
    expect(z0!.band).toBe(0);
    expect(z1!.band).toBe(1);
    expect(z1!.y).toBeGreaterThan(z0!.y);

    const airExchange = layout.edges.find((e) => e.edge.coupling_type === "air_exchange");
    expect(airExchange).toBeDefined();
    expect(airExchange!.path).toMatch(/^M /);

    // Zone-1 assembly members live in band 1.
    const z1Roof = layout.nodes.find((n) => n.node.id === "z1:roof:ext");
    expect(z1Roof!.band).toBe(1);
  });

  it("is deterministic (identical output for identical input)", () => {
    const a = computeLayout(CASE_600);
    const b = computeLayout(CASE_600);
    expect(a).toEqual(b);
  });

  it("routes forward edges from the source right anchor to the target left anchor", () => {
    const layout = computeLayout(syntheticTwoZoneDoc());
    const re = layout.edges.find((e) => e.edge.source_id === "z0:wall-s:l0" && e.edge.target_id === "z0:wall-s:l1");
    expect(re).toBeDefined();
    const s = layout.nodes.find((n) => n.node.id === "z0:wall-s:l0")!;
    const t = layout.nodes.find((n) => n.node.id === "z0:wall-s:l1")!;
    expect(re!.x1).toBeCloseTo(s.x + s.w, 1);
    expect(re!.y1).toBeCloseTo(s.y + s.h / 2, 1);
    expect(re!.x2).toBeCloseTo(t.x, 1);
    expect(re!.y2).toBeCloseTo(t.y + t.h / 2, 1);
  });

  it("skips routing for edges referencing missing nodes", () => {
    const doc = syntheticTwoZoneDoc();
    doc.edges.push({ source_id: "amb", target_id: "ghost", coupling_type: "conduction", bidirectional: true });
    const layout = computeLayout(doc);
    expect(layout.edges.some((e) => e.edge.target_id === "ghost")).toBe(false);
  });
});
