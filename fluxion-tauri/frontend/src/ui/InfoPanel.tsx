import type { GeometryStats } from "../lib/geometryAdapter";

export interface InfoPanelProps {
  stats: GeometryStats | null;
  source: "ipc" | "sample" | null;
  platform: "tauri" | "web";
}

/** Sidebar building summary — port of the preserved viewers' info panels. */
export function InfoPanel({ stats, source, platform }: InfoPanelProps) {
  return (
    <div className="info-panel">
      <h3>Building Summary</h3>
      {stats ? (
        <>
          <div className="info-grid">
            <div className="info-item">
              <div className="info-label">Building</div>
              <div className="info-value">{stats.buildingName}</div>
            </div>
            <div className="info-item">
              <div className="info-label">Floor Area</div>
              <div className="info-value">{stats.totalFloorArea.toFixed(0)} m²</div>
            </div>
            <div className="info-item">
              <div className="info-label">Levels</div>
              <div className="info-value">{stats.levelCount}</div>
            </div>
            <div className="info-item">
              <div className="info-label">Spaces</div>
              <div className="info-value">{stats.spaceCount}</div>
            </div>
            <div className="info-item">
              <div className="info-label">Zones</div>
              <div className="info-value">{stats.zoneCount}</div>
            </div>
            <div className="info-item">
              <div className="info-label">Geometry</div>
              <div className="info-value">
                {source === "ipc" ? "Tauri IPC" : "Sample"}
              </div>
            </div>
          </div>
          <div className="info-footnote">
            Running in {platform === "tauri" ? "Tauri desktop" : "web fallback"} mode.
          </div>
        </>
      ) : (
        <p className="info-loading">Loading geometry…</p>
      )}
    </div>
  );
}
