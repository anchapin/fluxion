/**
 * Pure pan/zoom transform math for the topology viewer (issue #3965).
 * Scale is clamped; zoomAtPoint keeps the diagram point under the cursor
 * fixed, matching standard map-style interaction.
 */

export interface PanZoom {
  x: number;
  y: number;
  scale: number;
}

export const MIN_SCALE = 0.15;
export const MAX_SCALE = 8;

export const IDENTITY: PanZoom = { x: 0, y: 0, scale: 1 };

export function pan(t: PanZoom, dx: number, dy: number): PanZoom {
  return { ...t, x: t.x + dx, y: t.y + dy };
}

/** Zoom by `factor` about the diagram-space point (px, py) in viewport px. */
export function zoomAt(t: PanZoom, px: number, py: number, factor: number): PanZoom {
  const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, t.scale * factor));
  const eff = scale / t.scale;
  return {
    scale,
    x: px - (px - t.x) * eff,
    y: py - (py - t.y) * eff,
  };
}

/** Fit a diagram box into a viewport, centered, with padding. */
export function fit(box: { width: number; height: number }, viewport: { width: number; height: number }, pad = 24): PanZoom {
  const vw = Math.max(1, viewport.width - 2 * pad);
  const vh = Math.max(1, viewport.height - 2 * pad);
  const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, Math.min(vw / box.width, vh / box.height)));
  return {
    scale,
    x: (viewport.width - box.width * scale) / 2,
    y: (viewport.height - box.height * scale) / 2,
  };
}
