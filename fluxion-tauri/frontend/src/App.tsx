import { useCallback, useMemo, useRef, useState } from "react";
import type { OrbitControlsImpl } from "./scene/OrbitCam";
import { GeometryScene } from "./scene/GeometryScene";
import { toRenderModel } from "./lib/geometryAdapter";
import { detectPlatform } from "./lib/platform";
import { loadGeometry } from "./tauri/geometryService";
import { useLiveTwin } from "./livetwin/useLiveTwin";
import { useWasmSimulation } from "./sim/useWasmSimulation";
import type { ZoneState } from "./livetwin/protocol";
import { ControlBar } from "./ui/ControlBar";
import { InfoPanel } from "./ui/InfoPanel";
import { Legend } from "./ui/Legend";
import { ParamsPanel } from "./ui/ParamsPanel";
import type { BuildingGeometry } from "./types/geometry";
import { useEffect } from "react";

/**
 * Fluxion GUI shell (issue #3178). Unifies the two preserved legacy viewers —
 * the geometry viewer (`src-tauri/index.html`, issues #3177/#3179) and the
 * thermal viewer (`src-tauri/src/index.html`, issue #3249) — into one React +
 * React Three Fiber app that runs in Tauri desktop mode and as a plain web
 * app.
 */
export default function App() {
  const platform = useMemo(detectPlatform, []);
  const [geometry, setGeometry] = useState<BuildingGeometry | null>(null);
  const [source, setSource] = useState<"ipc" | "sample" | null>(null);

  const [wireframe, setWireframe] = useState(false);
  const [zoneColoring, setZoneColoring] = useState(true);
  const [thermal, setThermal] = useState(false);

  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const livetwin = useLiveTwin();

  useEffect(() => {
    loadGeometry().then(({ geometry, source }) => {
      setGeometry(geometry);
      setSource(source);
    });
  }, []);

  const model = useMemo(
    () => (geometry ? toRenderModel(geometry) : null),
    [geometry],
  );

  const wasm = useWasmSimulation(model?.stats.zoneCount ?? 3);

  // LiveTwin streaming takes precedence; the in-browser fluxion-wasm
  // simulation is the web-fallback temperature source.
  const liveZones: Map<number, ZoneState> =
    livetwin.zones.size > 0 ? livetwin.zones : wasm.zones;
  const tempSource =
    livetwin.zones.size > 0
      ? "LiveTwin stream"
      : wasm.running
        ? `fluxion-wasm sim (${wasm.hour.toFixed(0)}h)`
        : "none";

  const onControlsReady = useCallback((controls: OrbitControlsImpl | null) => {
    controlsRef.current = controls;
  }, []);

  const resetView = useCallback(() => {
    controlsRef.current?.reset();
  }, []);

  return (
    <div className="app">
      <aside className="sidebar">
        <header className="sidebar-header">
          <h1>Fluxion</h1>
          <p>Building Energy Model Viewer</p>
        </header>
        <InfoPanel stats={model?.stats ?? null} source={source} platform={platform} />
        {model && (
          <Legend zones={model.zones} thermal={thermal} liveZones={liveZones} />
        )}
        <ParamsPanel />
      </aside>

      <main className="viewer">
        {model ? (
          <GeometryScene
            model={model}
            zoneColoring={zoneColoring}
            thermal={thermal}
            wireframe={wireframe}
            liveZones={liveZones}
            onControlsReady={onControlsReady}
          />
        ) : (
          <div className="loading">
            <div className="spinner" />
            <p>Loading geometry…</p>
          </div>
        )}

        <ControlBar
          onResetView={resetView}
          wireframe={wireframe}
          onToggleWireframe={() => setWireframe((w) => !w)}
          zoneColoring={zoneColoring}
          onToggleZoneColoring={() => setZoneColoring((z) => !z)}
          thermal={thermal}
          onToggleThermal={() => setThermal((t) => !t)}
          livetwin={livetwin}
          wasm={wasm}
        />

        <footer className="status-bar">
          <span>
            Mode: {platform === "tauri" ? "Tauri Desktop" : "Web Browser"}
          </span>
          <span>
            Geometry: {source === "ipc" ? "load_geometry IPC" : "embedded sample"}
          </span>
          <span>
            LiveTwin: {livetwin.status}
            {livetwin.payloadCount > 0 && ` (${livetwin.payloadCount} frames)`}
            {livetwin.lastError && ` — ${livetwin.lastError}`}
          </span>
          <span>Temps: {tempSource}</span>
        </footer>
      </main>
    </div>
  );
}
