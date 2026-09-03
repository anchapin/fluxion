import { useState } from "react";
import type { LiveTwinState } from "../livetwin/useLiveTwin";
import type { WasmSimulationState } from "../sim/useWasmSimulation";
import type { ModelSource } from "../App";

export interface ControlBarProps {
  onResetView: () => void;
  wireframe: boolean;
  onToggleWireframe: () => void;
  zoneColoring: boolean;
  onToggleZoneColoring: () => void;
  thermal: boolean;
  onToggleThermal: () => void;
  /** Geometry path feeding the canvas (issue #3175). */
  modelSource: ModelSource;
  onModelSourceChange: (source: ModelSource) => void;
  livetwin: LiveTwinState;
  wasm: WasmSimulationState;
}

/**
 * Floating viewer controls — the union of both preserved viewers' control
 * rows: Reset View / Wireframe / zone coloring / Thermal, plus the LiveTwin
 * connection affordance with a live status dot. The BEM/glTF segmented
 * control switches the geometry path (issue #3175): BEM wire contract vs
 * the glTF sample with `Zone_{id}` mesh auto-mapping.
 */
export function ControlBar({
  onResetView,
  wireframe,
  onToggleWireframe,
  zoneColoring,
  onToggleZoneColoring,
  thermal,
  onToggleThermal,
  modelSource,
  onModelSourceChange,
  livetwin,
  wasm,
}: ControlBarProps) {
  const [urlDraft, setUrlDraft] = useState(livetwin.url);
  const connected = livetwin.status === "open";
  const reconnecting = livetwin.status === "reconnecting";
  const liveTwinTitle = reconnecting
    ? `Reconnecting (attempt ${livetwin.reconnectAttempt}) with exponential backoff — click to stop`
    : "Connect to the LiveTwin MessagePack stream";

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

      <span
        className="model-source-toggle"
        title="Geometry path: BEM wire contract vs glTF sample (issue #3175)"
      >
        <button
          className={modelSource === "bem" ? "active" : ""}
          onClick={() => onModelSourceChange("bem")}
        >
          BEM
        </button>
        <button
          className={modelSource === "gltf" ? "active" : ""}
          onClick={() => onModelSourceChange("gltf")}
        >
          glTF
        </button>
      </span>

      <button
        className={wasm.running ? "active" : ""}
        onClick={() => (wasm.running ? wasm.stop() : wasm.start())}
        disabled={wasm.available === false}
        title={wasmTitle}
      >
        WASM Sim{wasm.running ? ` · ${wasm.hour.toFixed(0)}h` : ""}
      </button>

      <span className="livetwin-controls" title={liveTwinTitle}>
        <span className={`status-dot ${connected ? "live" : livetwin.status}`} />
        <input
          className="livetwin-url"
          value={urlDraft}
          onChange={(e) => setUrlDraft(e.target.value)}
          placeholder="ws://localhost:8080/live-twin"
          spellCheck={false}
        />
        {connected || reconnecting ? (
          <button
            onClick={() => livetwin.disconnect()}
            title={liveTwinTitle}
          >
            {reconnecting
              ? `Stop Retry ${livetwin.reconnectAttempt || ""}`.trim()
              : "Disconnect"}
          </button>
        ) : (
          <button onClick={() => livetwin.connect(urlDraft)}>LiveTwin</button>
        )}
      </span>
    </div>
  );
}
