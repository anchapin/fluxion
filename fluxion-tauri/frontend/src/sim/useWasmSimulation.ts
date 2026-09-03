import { useCallback, useEffect, useRef, useState } from "react";
import type { ZoneState } from "../livetwin/protocol";
import {
  createWasmSimulation,
  loadWasmModule,
  wasmTempsToZoneStates,
  type WasmSimulation,
} from "./wasmSimulation";

export interface WasmSimulationState {
  /** null = not probed yet; false = wasm pkg not generated. */
  available: boolean | null;
  running: boolean;
  zones: Map<number, ZoneState>;
  hour: number;
  start: () => void;
  stop: () => void;
}

/**
 * Drives an optional in-browser `fluxion-wasm` FluidSimulation (see
 * `wasmSimulation.ts`). When running, it steps the model on a timer and
 * publishes LiveTwin-shaped zone states so the thermal view renders from
 * client-side physics in web mode — no backend required.
 */
export function useWasmSimulation(zoneCount: number): WasmSimulationState {
  const [available, setAvailable] = useState<boolean | null>(null);
  const [running, setRunning] = useState(false);
  const [zones, setZones] = useState(() => new Map<number, ZoneState>());
  const [hour, setHour] = useState(0);
  const simRef = useRef<WasmSimulation | null>(null);

  useEffect(() => {
    loadWasmModule().then((mod) => setAvailable(mod !== null));
  }, []);

  const stop = useCallback(() => {
    setRunning(false);
    simRef.current = null;
  }, []);

  const start = useCallback(() => {
    if (simRef.current) return;
    createWasmSimulation(zoneCount)
      .then((sim) => {
        if (!sim) {
          setAvailable(false);
          return;
        }
        simRef.current = sim;
        setRunning(true);
        // Seed the view immediately.
        setZones(wasmTempsToZoneStates(sim.zoneTemperatures()));
      })
      .catch((err) => {
        console.warn("wasm simulation start failed:", err);
        setAvailable(false);
      });
  }, [zoneCount]);

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => {
      const sim = simRef.current;
      if (!sim) return;
      sim.step(1.0);
      setZones(wasmTempsToZoneStates(sim.zoneTemperatures()));
      setHour(sim.currentHour());
    }, 700);
    return () => window.clearInterval(id);
  }, [running]);

  return { available, running, zones, hour, start, stop };
}
