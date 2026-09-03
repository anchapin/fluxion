/**
 * Thermal colormap — TS mirror of the GLSL `thermalColormap` in
 * `scene/ThermalMaterial.tsx` (ported from the preserved thermal viewer,
 * issue #3249). Same five anchor stops, same piecewise-lerp behaviour, so
 * the sidebar legend/gradient always matches the 3D shading.
 */

export interface RGB {
  r: number;
  g: number;
  b: number;
}

const STOPS: RGB[] = [
  { r: 0.23, g: 0.51, b: 0.96 }, // cold
  { r: 0.13, g: 0.77, b: 0.37 }, // cool
  { r: 0.92, g: 0.75, b: 0.03 }, // neutral
  { r: 0.98, g: 0.45, b: 0.09 }, // warm
  { r: 0.94, g: 0.24, b: 0.24 }, // hot
];

/** Maps t ∈ [0, 1] (clamped) through the 5-stop colormap. */
export function thermalColor(t: number): RGB {
  const c = Math.min(1, Math.max(0, t)) * 4;
  const i = Math.min(3, Math.floor(c));
  const f = c - i;
  const a = STOPS[i];
  const b = STOPS[i + 1];
  return {
    r: a.r + (b.r - a.r) * f,
    g: a.g + (b.g - a.g) * f,
    b: a.b + (b.b - a.b) * f,
  };
}

/** Formats a colormap sample as a CSS `rgb()` string (0-255 channels). */
export function thermalCss(t: number): string {
  const { r, g, b } = thermalColor(t);
  const to255 = (c: number) => Math.round(c * 255);
  return `rgb(${to255(r)}, ${to255(g)}, ${to255(b)})`;
}

/** Multi-stop CSS gradient of the colormap for the legend bar. */
export function thermalGradientCss(stops = 5): string {
  const points: string[] = [];
  for (let i = 0; i < stops; i++) {
    const t = stops === 1 ? 0 : i / (stops - 1);
    points.push(`${thermalCss(t)} ${(t * 100).toFixed(1)}%`);
  }
  return `linear-gradient(to right, ${points.join(", ")})`;
}

/**
 * Live temperature display range: falls back to the 15-30 °C default with
 * no data and widens degenerate (near-constant) ranges so the colormap
 * never divides by zero.
 */
export function temperatureRange(temps: number[]): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const t of temps) {
    if (Number.isFinite(t)) {
      min = Math.min(min, t);
      max = Math.max(max, t);
    }
  }
  if (!Number.isFinite(min)) return { min: 15, max: 30 };
  if (max - min < 0.5) return { min: min - 0.25, max: max + 0.25 };
  return { min, max };
}
