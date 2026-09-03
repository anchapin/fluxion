import { invoke } from "@tauri-apps/api/core";
import type { BuildingGeometry } from "../types/geometry";
import { detectPlatform } from "../lib/platform";
import { sampleGeometry } from "../lib/sampleGeometry";

export interface GeometrySource {
  geometry: BuildingGeometry;
  /** "ipc" = Tauri command, "sample" = embedded fallback. */
  source: "ipc" | "sample";
}

/**
 * Loads building geometry. Native mode invokes the `load_geometry` command
 * (see `fluxion-tauri/src-tauri/src/commands.rs`); web mode falls back to the
 * embedded mirror of the Rust sample so the R3F canvas renders in both modes.
 */
export async function loadGeometry(): Promise<GeometrySource> {
  if (detectPlatform() === "tauri") {
    try {
      const geometry = await invoke<BuildingGeometry>("load_geometry");
      return { geometry, source: "ipc" };
    } catch (err) {
      console.warn("load_geometry IPC failed, falling back to sample:", err);
    }
  }
  return { geometry: sampleGeometry, source: "sample" };
}
