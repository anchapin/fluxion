import { useEffect, useState } from "react";
import type { SimulationParameters } from "../types/geometry";
import {
  getSimulationParameters,
  paramsAvailable,
  updateSimulationParameters,
} from "../tauri/paramsService";

interface SliderSpec {
  key: keyof SimulationParameters;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
}

const SLIDERS: SliderSpec[] = [
  { key: "heating_setpoint", label: "Heating Setpoint", unit: "°C", min: 10, max: 25, step: 0.5, format: (v) => `${v.toFixed(1)} °C` },
  { key: "cooling_setpoint", label: "Cooling Setpoint", unit: "°C", min: 18, max: 35, step: 0.5, format: (v) => `${v.toFixed(1)} °C` },
  { key: "lighting_load", label: "Lighting Load", unit: "W/m²", min: 0, max: 20, step: 0.5, format: (v) => `${v.toFixed(1)} W/m²` },
  { key: "equipment_load", label: "Equipment Load", unit: "W/m²", min: 0, max: 30, step: 0.5, format: (v) => `${v.toFixed(1)} W/m²` },
  { key: "occupancy", label: "Occupancy", unit: "pers/m²", min: 0, max: 0.5, step: 0.01, format: (v) => `${v.toFixed(2)} pers/m²` },
  { key: "ventilation_rate", label: "Ventilation Rate", unit: "ach", min: 0, max: 5, step: 0.1, format: (v) => `${v.toFixed(1)} ach` },
  { key: "wall_u_value", label: "Wall U-Value", unit: "W/m²K", min: 0.1, max: 2.0, step: 0.05, format: (v) => `${v.toFixed(2)} W/m²K` },
  { key: "roof_u_value", label: "Roof U-Value", unit: "W/m²K", min: 0.1, max: 2.0, step: 0.05, format: (v) => `${v.toFixed(2)} W/m²K` },
];

/**
 * Simulation-parameter sliders — the React port of the geometry viewer's
 * Controls tab (issue #3177), wired to the `get/update_simulation_parameters`
 * Tauri commands that were previously defined but never registered.
 * Disabled in web mode (no IPC backend to talk to).
 */
export function ParamsPanel() {
  const available = paramsAvailable();
  const [params, setParams] = useState<SimulationParameters | null>(null);
  const [draft, setDraft] = useState<Record<string, number>>({});
  const [status, setStatus] = useState("");

  useEffect(() => {
    if (!available) return;
    getSimulationParameters().then((p) => {
      if (!p) return;
      setParams(p);
      const d: Record<string, number> = {};
      for (const s of SLIDERS) {
        const v = p[s.key];
        d[s.key] = typeof v === "number" ? v : (s.min + s.max) / 2;
      }
      setDraft(d);
    });
  }, [available]);

  if (!available) {
    return (
      <div className="params-panel">
        <h3>Simulation Controls</h3>
        <p className="params-unavailable">
          Parameter adjustment requires the native Tauri backend. Start the
          desktop app to enable these controls.
        </p>
      </div>
    );
  }

  const apply = async () => {
    if (!params) return;
    setStatus("Applying…");
    const next: SimulationParameters = { ...params };
    for (const s of SLIDERS) {
      (next[s.key] as number) = draft[s.key];
    }
    const result = await updateSimulationParameters(next);
    if (result) {
      setParams(result);
      setStatus("✓ Applied");
    } else {
      setStatus("✗ IPC failed");
    }
    setTimeout(() => setStatus(""), 3000);
  };

  return (
    <div className="params-panel">
      <h3>Simulation Controls</h3>
      {SLIDERS.map((s) => (
        <div key={s.key} className="param-group">
          <label>
            {s.label}
            <span>{s.format(draft[s.key] ?? 0)}</span>
          </label>
          <input
            type="range"
            min={s.min}
            max={s.max}
            step={s.step}
            value={draft[s.key] ?? 0}
            onChange={(e) =>
              setDraft((d) => ({ ...d, [s.key]: parseFloat(e.target.value) }))
            }
          />
        </div>
      ))}
      <button className="apply" onClick={apply} disabled={!params}>
        Apply Parameters
      </button>
      <div className="param-status">{status}</div>
    </div>
  );
}
