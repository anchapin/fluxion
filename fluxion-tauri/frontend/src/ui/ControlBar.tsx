import { useState } from "react";
import type { LiveTwinState } from "../livetwin/useLiveTwin";
import type { WasmSimulationState } from "../sim/useWasmSimulation";

export interface ControlBarProps {
  onResetView: () => void;
  wireframe: boolean;
  onToggleWireframe: () => void;
  zoneColoring: boolean;
  onToggleZoneColoring: () => void;
  thermal: boolean;
  onToggleThermal: () => void;
  livetwin: LiveTwinState;
  wasm: WasmSimulationState;
}

/**
 * Floating viewer controls — the union of both preserved viewers' control
 * rows: Reset View / Wireframe / zone coloring / Thermal, plus the LiveTwin
 * connection affordance with a live status dot.
 */
export function ControlBar({
  onResetView,
  wireframe,
  onToggleWireframe,
  zoneColoring,
  onToggleZoneColoring,
  thermal,
  onToggleThermal,
  livetwin,
  wasm,
}: ControlBarProps) {
  const [urlDraft, setUrlDraft] = useState(livetwin.url);
  const connected = livetwin.status === "open";

  const wasmTitle =
    wasm.available === false
      ? "WASM pkg not generated (run wasm-pack, see README)"
      : "Step an in-browser FluidSimulation via fluxion-wasm";

  return (
    <div className="control-bar">
      <button onClick={onResetView}>Reset View</button>
      <button
        className={wireframe ? "active" : ""}
        onClick={onToggleWireframe}
      >
        Wireframe
      </button>
      <button
        className={zoneColoring && !thermal ? "active" : ""}
        onClick={onToggleZoneColoring}
        disabled={thermal}
        title={thermal ? "Zone colors are replaced by thermal shading" : ""}
      >
        Zone Colors
      </button>
      <button className={thermal ? "active" : ""} onClick={onToggleThermal}>
        Thermal
      </button>

      <button
        className={wasm.running ? "active" : ""}
        onClick={() => (wasm.running ? wasm.stop() : wasm.start())}
        disabled={wasm.available === false}
        title={wasmTitle}
      >
        WASM Sim{wasm.running ? ` · ${wasm.hour.toFixed(0)}h` : ""}
      </button>

      <span className="livetwin-controls">
        <span className={`status-dot ${connected ? "live" : livetwin.status}`} />
        <input
          className="livetwin-url"
          value={urlDraft}
          onChange={(e) => setUrlDraft(e.target.value)}
          placeholder="ws://localhost:8080/live-twin"
          spellCheck={false}
        />
        {connected ? (
          <button onClick={() => livetwin.disconnect()}>Disconnect</button>
        ) : (
          <button onClick={() => livetwin.connect(urlDraft)}>LiveTwin</button>
        )}
      </span>
    </div>
  );
}
