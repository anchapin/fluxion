import { useCallback, useMemo, useRef, useState } from "react";
import type { OrbitControlsImpl } from "./scene/OrbitCam";
import { GeometryScene } from "./scene/GeometryScene";
import { toRenderModel } from "./lib/geometryAdapter";
import { detectPlatform } from "./lib/platform";
import { loadGeometry } from "./tauri/geometryService";
import { loadSampleGltfBuilding } from "./gltf/loadGltfBuilding";
import { resolveZoneMeshTemperatures } from "./gltf/zoneMeshMapping";
import type { ZoneMeshResolution } from "./gltf/zoneMeshMapping";
import { useLiveTwin } from "./livetwin/useLiveTwin";
import { useWasmSimulation } from "./sim/useWasmSimulation";
import type { ZoneState } from "./livetwin/protocol";
import type { ThermalZone } from "./types/geometry";
import { ControlBar } from "./ui/ControlBar";
import { InfoPanel } from "./ui/InfoPanel";
import { Legend } from "./ui/Legend";
import { ParamsPanel } from "./ui/ParamsPanel";
import type { BuildingGeometry } from "./types/geometry";
import type * as THREE from "three";
import { useEffect } from "react";

/** Which geometry path feeds the R3F canvas (issue #3175). */
export type ModelSource = "bem" | "gltf";

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
  const [modelSource, setModelSource] = useState<ModelSource>("bem");
  const [gltfScene, setGltfScene] = useState<THREE.Group | null>(null);
  const [gltfError, setGltfError] = useState<string | null>(null);

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

  // The glTF sample is parsed lazily on first switch and cached by the
  // loader module; a parse failure surfaces in the status bar, never a crash.
  useEffect(() => {
    if (modelSource !== "gltf" || gltfScene || gltfError) return;
    let cancelled = false;
    loadSampleGltfBuilding()
      .then((scene) => {
        if (!cancelled) setGltfScene(scene);
      })
      .catch((err: unknown) => {
        if (!cancelled) setGltfError(String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [modelSource, gltfScene, gltfError]);

  const model = useMemo(
    () => (geometry ? toRenderModel(geometry) : null),
    [geometry],
  );

  const wasm = useWasmSimulation(model?.stats.zoneCount ?? 3);

  // LiveTwin streaming takes precedence; the in-browser fluxion-wasm
  // simulation is the web-fallback temperature source.
  const liveZones: Map<number, ZoneState> =
    livetwin.zones.size > 0 ? livetwin.zones : wasm.zones;

  // Zone→mesh join for the glTF path, computed here so the scene and the
  // Legend/status warnings share one resolution (issue #3175).
  const gltfResolution: ZoneMeshResolution | null = useMemo(
    () =>
      gltfScene && modelSource === "gltf"
        ? resolveZoneMeshTemperatures(gltfScene, liveZones)
        : null,
    [gltfScene, modelSource, liveZones],
  );

  // Legend zone list for glTF mode: synthesize display zones from the join
  // (BEM zone metadata only exists on the wire-contract path).
  const legendZones: ThermalZone[] = useMemo(() => {
    if (modelSource !== "gltf") return model?.zones ?? [];
    if (!gltfResolution) return [];
    const seen = new Set<number>();
    const zones: ThermalZone[] = [];
    for (const entry of gltfResolution.entries) {
      if (seen.has(entry.zoneId)) continue;
      seen.add(entry.zoneId);
      zones.push({
        id: `zone-${entry.zoneId}`,
        name: `Zone ${entry.zoneId}`,
        level_id: "",
        space_ids: [],
        setpoint_heating: null,
        setpoint_cooling: null,
      });
    }
    return zones;
  }, [modelSource, model, gltfResolution]);

  // Unmatched meshes/zones are surfaced as warnings, never fatal.
  useEffect(() => {
    if (!gltfResolution) return;
    if (gltfResolution.unmatchedMeshNames.length > 0) {
      console.warn(
        "[gltf] meshes without a Zone_{id} name (rendered neutral):",
        gltfResolution.unmatchedMeshNames,
      );
    }
    if (gltfResolution.unmatchedZoneIds.length > 0) {
      console.warn(
        "[gltf] live zones with no mesh in the model:",
        gltfResolution.unmatchedZoneIds,
      );
    }
  }, [gltfResolution]);

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

  const hasRenderable = modelSource === "gltf" ? gltfScene !== null : model !== null;

  return (
    <div className="app">
      <aside className="sidebar">
        <header className="sidebar-header">
          <h1>Fluxion</h1>
          <p>Building Energy Model Viewer</p>
        </header>
        <InfoPanel stats={model?.stats ?? null} source={source} platform={platform} />
        {model && (
          <Legend
            zones={legendZones}
            thermal={thermal}
            liveZones={liveZones}
            mappingWarnings={
              gltfResolution && modelSource === "gltf"
                ? {
                    unmatchedZoneIds: gltfResolution.unmatchedZoneIds,
                    unmatchedMeshNames: gltfResolution.unmatchedMeshNames,
                  }
                : null
            }
          />
        )}
        <ParamsPanel />
      </aside>

      <main className="viewer">
        {hasRenderable ? (
          <GeometryScene
            model={model}
            gltfBuilding={
              modelSource === "gltf" && gltfScene && gltfResolution
                ? { scene: gltfScene, resolution: gltfResolution }
                : null
            }
            zoneColoring={zoneColoring}
            thermal={thermal}
            wireframe={wireframe}
            liveZones={liveZones}
            onControlsReady={onControlsReady}
          />
        ) : (
          <div className="loading">
            <div className="spinner" />
            <p>
              {modelSource === "gltf" && gltfError
                ? `glTF load failed: ${gltfError}`
                : "Loading geometry…"}
            </p>
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
          modelSource={modelSource}
          onModelSourceChange={setModelSource}
          livetwin={livetwin}
          wasm={wasm}
        />

        <footer className="status-bar">
          <span>
            Mode: {platform === "tauri" ? "Tauri Desktop" : "Web Browser"}
          </span>
          <span>
            Geometry:{" "}
            {modelSource === "gltf"
              ? "glTF sample (Zone_{id} meshes)"
              : source === "ipc"
                ? "load_geometry IPC"
                : "embedded sample"}
          </span>
          <span>
            LiveTwin:{" "}
            {livetwin.status === "reconnecting"
              ? `reconnecting (attempt ${livetwin.reconnectAttempt})`
              : livetwin.status}
            {livetwin.payloadCount > 0 && ` (${livetwin.payloadCount} frames)`}
            {livetwin.lastError && ` — ${livetwin.lastError}`}
          </span>
          <span>Temps: {tempSource}</span>
        </footer>
      </main>
    </div>
  );
}
