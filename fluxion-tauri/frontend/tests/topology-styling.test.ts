/**
 * Kind-based styling tests (issue #3965): style-map exhaustiveness against
 * the schema kind unions, visual distinguishability, flux-overlay scaling.
 */

import { describe, expect, it } from "vitest";
import { EDGE_KINDS, NODE_KINDS } from "../src/topology/types";
import {
  EDGE_STYLES,
  NODE_STYLES,
  edgeStrokeWidth,
  edgeStyle,
  kindsPresent,
  nodeStyle,
} from "../src/topology/styling";
import { CASE_600 } from "../src/topology/fixtures";

function presetOrder(kinds: string[]): boolean {
  const idx = kinds.map((k) => NODE_KINDS.indexOf(k as (typeof NODE_KINDS)[number]));
  return idx.every((v, i) => i === 0 || idx[i - 1] < v);
}

describe("styling", () => {
  it("has a style entry for every node kind in the schema union", () => {
    for (const kind of NODE_KINDS) {
      expect(NODE_STYLES[kind], `missing node style for ${kind}`).toBeDefined();
      expect(NODE_STYLES[kind].label.length).toBeGreaterThan(0);
      expect(NODE_STYLES[kind].fill).toMatch(/^#[0-9a-f]{6}$/i);
      expect(NODE_STYLES[kind].stroke).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });

  it("has a style entry for every edge kind in the schema union", () => {
    for (const kind of EDGE_KINDS) {
      expect(EDGE_STYLES[kind], `missing edge style for ${kind}`).toBeDefined();
      expect(EDGE_STYLES[kind].label.length).toBeGreaterThan(0);
      expect(EDGE_STYLES[kind].color).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });

  it("uses visually distinguishable (unique) colors per kind", () => {
    const nodeStrokes = NODE_KINDS.map((k) => NODE_STYLES[k].stroke);
    expect(new Set(nodeStrokes).size).toBe(nodeStrokes.length);
    const edgeColors = EDGE_KINDS.map((k) => EDGE_STYLES[k].color);
    expect(new Set(edgeColors).size).toBe(edgeColors.length);
  });

  it("nodeStyle/edgeStyle resolve by kind", () => {
    expect(nodeStyle("zone_air").label).toBe("Zone air");
    expect(edgeStyle("conduction").label).toBe("Conduction");
  });

  it("kindsPresent returns present kinds in canonical order", () => {
    const present = kindsPresent(CASE_600.nodes);
    expect(presetOrder(present)).toBe(true);
    // Case 600 exercises every node kind the schema defines.
    expect(present).toHaveLength(NODE_KINDS.length);
  });

  it("kindsPresent omits kinds a document does not use", () => {
    const synthetic = {
      nodes: [
        { kind: "zone_air" as const },
        { kind: "hvac_terminal" as const },
        { kind: "zone_air" as const },
      ],
    };
    expect(kindsPresent(synthetic.nodes)).toEqual(["zone_air", "hvac_terminal"]);
  });
});

describe("edgeStrokeWidth (flux overlay scaling)", () => {
  it("returns the uniform base width when the overlay is off", () => {
    expect(edgeStrokeWidth(1000, false)).toBe(1.6);
    expect(edgeStrokeWidth(undefined, true)).toBe(1.6);
  });

  it("scales monotonically with conductance when the overlay is on", () => {
    const low = edgeStrokeWidth(0.1, true);
    const mid = edgeStrokeWidth(10, true);
    const high = edgeStrokeWidth(1000, true);
    expect(low).toBeLessThan(mid);
    expect(mid).toBeLessThan(high);
  });

  it("clamps to the [1, 7] px band", () => {
    expect(edgeStrokeWidth(1e-9, true)).toBeGreaterThanOrEqual(1);
    expect(edgeStrokeWidth(1e12, true)).toBeLessThanOrEqual(7);
    expect(edgeStrokeWidth(-5, true)).toBe(1.6); // invalid conductance → base
  });
});
