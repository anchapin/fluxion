import { zoneColor } from "../lib/geometryAdapter";
import { temperatureRange, thermalGradientCss } from "../lib/thermal";
import type { ThermalZone } from "../types/geometry";
import type { ZoneState } from "../livetwin/protocol";

export interface LegendProps {
  zones: ThermalZone[];
  thermal: boolean;
  liveZones: Map<number, ZoneState>;
}

/**
 * Sidebar legend combining the two preserved viewers' vocabularies: the zone
 * palette list (geometry viewer) and the thermal temperature gradient bar
 * (thermal viewer) with live min/max labels.
 */
export function Legend({ zones, thermal, liveZones }: LegendProps) {
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
    </div>
  );
}
