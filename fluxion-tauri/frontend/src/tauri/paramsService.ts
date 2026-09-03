import { invoke } from "@tauri-apps/api/core";
import type { SimulationParameters } from "../types/geometry";
import { detectPlatform } from "../lib/platform";

/**
 * Read/write the GUI simulation parameters held by the
 * `get_simulation_parameters` / `update_simulation_parameters` Tauri commands.
 * Web mode has no IPC backend — callers show a disabled notice instead.
 */
export function paramsAvailable(): boolean {
  return detectPlatform() === "tauri";
}

export async function getSimulationParameters(): Promise<SimulationParameters | null> {
  if (!paramsAvailable()) return null;
  try {
    return await invoke<SimulationParameters>("get_simulation_parameters");
  } catch (err) {
    console.warn("get_simulation_parameters failed:", err);
    return null;
  }
}

export async function updateSimulationParameters(
  params: SimulationParameters,
): Promise<SimulationParameters | null> {
  if (!paramsAvailable()) return null;
  try {
    return await invoke<SimulationParameters>(
      "update_simulation_parameters",
      { params },
    );
  } catch (err) {
    console.warn("update_simulation_parameters failed:", err);
    return null;
  }
}
