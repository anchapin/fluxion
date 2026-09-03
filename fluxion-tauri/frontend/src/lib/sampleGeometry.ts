import sampleJson from "./sample-geometry.json";
import type { BuildingGeometry } from "../types/geometry";

/**
 * Embedded mirror of the Rust sample building (`cargo run --example
 * dump_sample -p fluxion-tauri`) used as the web-mode fallback when the
 * `load_geometry` IPC is unavailable. Kept in sync with
 * `tests/fixtures/rust-sample-geometry.json` — the
 * `embedded web fallback stays in sync` test fails if they drift.
 */
export const sampleGeometry = sampleJson as BuildingGeometry;
