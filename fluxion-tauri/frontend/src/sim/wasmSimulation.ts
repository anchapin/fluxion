import type { ZoneState } from "../livetwin/protocol";
import { syntheticZoneState } from "../livetwin/protocol";

/**
 * Optional in-browser simulation via the `fluxion-wasm` crate (issue #3178 web
 * fallback path). The wasm-pack `--target web` output is NOT bundled — it is
 * generated into `frontend/public/wasm/` (gitignored) and loaded with a
 * runtime dynamic import so the app builds and works without it:
 *
 *   wasm-pack build --target web --out-dir ../fluxion-tauri/frontend/public/wasm ../fluxion-wasm
 *
 * When present, a lightweight FluidSimulation steps client-side and feeds
 * zone temperatures into the same rendering path as LiveTwin.
 */

const WASM_MODULE_URL = "/wasm/fluxion_wasm.js";

export interface WasmSimulation {
  numZones: number;
  step(dtHours: number): void;
  zoneTemperatures(): number[];
  currentHour(): number;
}

type WasmModule = {
  default: (path?: string) => Promise<unknown>;
  FluidSimulation: new (configJson: string) => {
    step: (dtHours: number) => void;
    get_zone_temps: () => Float64Array;
    current_hour: () => number;
    num_zones: () => number;
  };
};

let loadPromise: Promise<WasmModule | null> | null = null;

/** Resolves the wasm module, or null when the pkg has not been generated. */
export function loadWasmModule(): Promise<WasmModule | null> {
  if (!loadPromise) {
    loadPromise = (async () => {
      try {
        // @vite-ignore keeps the bundler from resolving this at build time —
        // the module only exists after the optional wasm-pack step.
        const mod = (await import(/* @vite-ignore */ WASM_MODULE_URL)) as unknown as WasmModule;
        await mod.default();
        return mod;
      } catch (err) {
        console.info("fluxion-wasm not available (optional):", err);
        return null;
      }
    })();
  }
  return loadPromise;
}

/**
 * Starts a FluidSimulation matching the sample building's zone count so its
 * temperatures map onto the geometry's `zone-N` ids.
 */
export async function createWasmSimulation(
  numZones: number,
): Promise<WasmSimulation | null> {
  const mod = await loadWasmModule();
  if (!mod) return null;
  const config = JSON.stringify({
    building: "fluxion_gui_sample",
    num_zones: numZones,
    initial_temps: Array.from({ length: numZones }, () => 21.0),
    heating_setpoint: 20.0,
    cooling_setpoint: 24.0,
  });
  try {
    const sim = new mod.FluidSimulation(config);
    return {
      numZones: sim.num_zones(),
      step: (dtHours: number) => sim.step(dtHours),
      zoneTemperatures: () => Array.from(sim.get_zone_temps()),
      currentHour: () => sim.current_hour(),
    };
  } catch (err) {
    console.warn("FluidSimulation construction failed:", err);
    return null;
  }
}

/** Converts wasm zone temperatures into LiveTwin-shaped ZoneStates. */
export function wasmTempsToZoneStates(temps: number[]): Map<number, ZoneState> {
  const map = new Map<number, ZoneState>();
  for (let i = 0; i < temps.length; i++) {
    // fluxion-wasm zones are 0-based; LiveTwin zone_ids match `zone-(i+1)`.
    map.set(i + 1, syntheticZoneState(i + 1, temps[i]));
  }
  return map;
}
