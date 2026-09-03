/**
 * Runtime platform detection for the dual-mode GUI (Tauri desktop vs plain
 * web). Tauri v2 injects `__TAURI_INTERNALS__` on the global window object
 * before any frontend code runs; the legacy `__TAURI__` global is kept as a
 * fallback for older shells.
 */
export type Platform = "tauri" | "web";

export function detectPlatform(): Platform {
  if (typeof window === "undefined") return "web";
  const w = window as unknown as Record<string, unknown>;
  const isTauri =
    "__TAURI_INTERNALS__" in w || "__TAURI__" in w || "__TAURI_METADATA__" in w;
  return isTauri ? "tauri" : "web";
}
