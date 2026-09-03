import { zoneColor } from "../lib/geometryAdapter";
import { temperatureRange, thermalGradientCss } from "../lib/thermal";
import type { ThermalZone } from "../types/geometry";
import type { ZoneState } from "../livetwin/protocol";

/** Zone-mapping diagnostics surfaced from the glTF join (issue #3175). */
export interface MappingWarnings {
  /** Live zone ids that own no `Zone_{id}` mesh in the model. */
  unmatchedZoneIds: number[];
  /** Mesh names that don't follow the `Zone_{id}` convention. */
  unmatchedMeshNames: string[];
}

export interface LegendProps {
  zones: ThermalZone[];
  thermal: boolean;
  liveZones: Map<number, ZoneState>;
  /** Present in glTF mode: surfaces unmatched meshes/zones as warnings. */
  mappingWarnings?: MappingWarnings | null;
}

/**
 * Sidebar legend combining the two preserved viewers' vocabularies: the zone
 * palette list (geometry viewer) and the thermal temperature gradient bar
 * (thermal viewer) with live min/max labels. In glTF mode the mapping
 * warnings paragraph makes the zone→mesh join's unmatched cases explicit
 * (issue #3175).
 */
export function Legend({ zones, thermal, liveZones, mappingWarnings }: LegendProps) {
  const warnings = <MappingWarningList warnings={mappingWarnings} />;
  if (thermal) {
    const temps = [...liveZones.values()].map((z) => z.t_air);
    const { min, max } = temperatureRange(temps);
    return (
      <div className="legend">
        <h3>Temperature (°C)</h3>
        <div
          className="thermal-gradient"
          style={{ background: thermalGradientCss() }}
        />
        <div className="thermal-labels">
          <span>{min.toFixed(1)}</span>
          <span>{max.toFixed(1)}</span>
        </div>
        {liveZones.size === 0 && (
          <p className="legend-hint">
            Connect LiveTwin to stream live zone temperatures.
          </p>
        )}
        {warnings}
      </div>
    );
  }

  return (
    <div className="legend">
      <h3>Thermal Zones</h3>
      <ul className="zone-list">
        {zones.map((zone) => {
          const state = liveZones.get(Number(zone.id.replace(/^\D+/g, "")));
          return (
            <li key={zone.id} className="zone-item">
              <span
                className="zone-swatch"
                style={{
                  background: `#${zoneColor(zone.id).toString(16).padStart(6, "0")}`,
                }}
              />
              <span className="zone-name">{zone.name}</span>
              {state && (
                <span className="zone-temp">{state.t_air.toFixed(1)}°C</span>
              )}
            </li>
          );
        })}
      </ul>
      {warnings}
    </div>
  );
}

/** Unmatched cases of the glTF zone→mesh join (issue #3175). */
function MappingWarningList({
  warnings = null,
}: {
  warnings?: MappingWarnings | null;
}) {
  if (!warnings) return null;
  return (
    <>
      {warnings.unmatchedZoneIds.length > 0 && (
        <p className="legend-warning">
          ⚠ Live zones without meshes (no geometry to shade): ids{" "}
          {warnings.unmatchedZoneIds.join(", ")}
        </p>
      )}
      {warnings.unmatchedMeshNames.length > 0 && (
        <p className="legend-warning">
          ⚠ Meshes without a Zone_&#123;id&#125; name (rendered neutral):{" "}
          {warnings.unmatchedMeshNames.join(", ")}
        </p>
      )}
    </>
  );
}
