import { describe, expect, it } from "vitest";
import {
  temperatureRange,
  thermalColor,
  thermalCss,
  thermalGradientCss,
} from "../src/lib/thermal";

describe("thermalColor — GLSL colormap mirror", () => {
  it("hits the five anchor stops exactly", () => {
    const expectStop = (
      t: number,
      expected: { r: number; g: number; b: number },
    ) => {
      const c = thermalColor(t);
      expect(c.r).toBeCloseTo(expected.r, 10);
      expect(c.g).toBeCloseTo(expected.g, 10);
      expect(c.b).toBeCloseTo(expected.b, 10);
    };
    expectStop(0, { r: 0.23, g: 0.51, b: 0.96 }); // cold
    expectStop(0.25, { r: 0.13, g: 0.77, b: 0.37 }); // cool
    expectStop(0.5, { r: 0.92, g: 0.75, b: 0.03 }); // neutral
    expectStop(0.75, { r: 0.98, g: 0.45, b: 0.09 }); // warm
    expectStop(1, { r: 0.94, g: 0.24, b: 0.24 }); // hot
  });

  it("interpolates midpoint between cold and cool", () => {
    const mid = thermalColor(0.125);
    expect(mid.r).toBeCloseTo((0.23 + 0.13) / 2);
    expect(mid.g).toBeCloseTo((0.51 + 0.77) / 2);
    expect(mid.b).toBeCloseTo((0.96 + 0.37) / 2);
  });

  it("clamps out-of-range inputs", () => {
    expect(thermalColor(-1)).toEqual(thermalColor(0));
    expect(thermalColor(2)).toEqual(thermalColor(1));
  });
});

describe("thermalCss / thermalGradientCss", () => {
  it("formats rgb() with 0-255 channels", () => {
    expect(thermalCss(0)).toBe("rgb(59, 130, 245)");
  });

  it("builds a multi-stop CSS gradient", () => {
    const css = thermalGradientCss(4);
    expect(css.startsWith("linear-gradient(to right, rgb(")).toBe(true);
    expect(css.endsWith("100.0%)")).toBe(true);
  });
});

describe("temperatureRange", () => {
  it("returns 15-30 default with no data", () => {
    expect(temperatureRange([])).toEqual({ min: 15, max: 30 });
  });

  it("tracks live min/max", () => {
    expect(temperatureRange([18.2, 24.6, 21.0])).toEqual({
      min: 18.2,
      max: 24.6,
    });
  });

  it("widens a degenerate single-temperature range", () => {
    const { min, max } = temperatureRange([20]);
    expect(min).toBeLessThan(20);
    expect(max).toBeGreaterThan(20);
  });
});
