/**
 * Pan/zoom math tests (issue #3965): zoom-about-cursor stability, clamping,
 * fit-to-view centering.
 */

import { describe, expect, it } from "vitest";
import { fit, MAX_SCALE, MIN_SCALE, pan, zoomAt } from "../src/topology/panZoom";

describe("panZoom", () => {
  it("pan translates without touching scale", () => {
    const t = pan({ x: 10, y: -5, scale: 2 }, 3, 4);
    expect(t).toEqual({ x: 13, y: -1, scale: 2 });
  });

  it("zoomAt keeps the diagram point under the cursor fixed", () => {
    const t = { x: 40, y: -20, scale: 1 };
    const px = 331;
    const py = 97;
    const before = (px - t.x) / t.scale;
    const t2 = zoomAt(t, px, py, 1.25);
    const after = (px - t2.x) / t2.scale;
    expect(after).toBeCloseTo(before, 6);
  });

  it("clamps scale to [MIN_SCALE, MAX_SCALE]", () => {
    const t = { x: 0, y: 0, scale: 7.9 };
    expect(zoomAt(t, 0, 0, 2).scale).toBe(MAX_SCALE);
    const t2 = { x: 0, y: 0, scale: 0.2 };
    expect(zoomAt(t2, 0, 0, 0.5).scale).toBe(MIN_SCALE);
  });

  it("fit centers the box in the viewport at the largest fitting scale", () => {
    const box = { width: 1600, height: 900 };
    const viewport = { width: 800, height: 600 };
    const t = fit(box, viewport);
    expect(t.scale).toBeCloseTo((800 - 48) / 1600, 5); // width-constrained
    const scaledW = box.width * t.scale;
    expect(t.x).toBeCloseTo((viewport.width - scaledW) / 2, 5);
  });

  it("fit never leaves the clamp band", () => {
    const tiny = fit({ width: 1, height: 1 }, { width: 800, height: 600 });
    expect(tiny.scale).toBe(MAX_SCALE);
    const huge = fit({ width: 1e9, height: 1e9 }, { width: 800, height: 600 });
    expect(huge.scale).toBe(MIN_SCALE);
  });
});
