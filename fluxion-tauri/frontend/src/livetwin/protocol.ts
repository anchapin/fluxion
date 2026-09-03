import { decode as msgpackDecode } from "@msgpack/msgpack";

/**
 * LiveTwin wire protocol — mirrors `src/twin/live_twin_broadcaster.rs`:
 * MessagePack-encoded `LiveTwinPayload { timestamp, simulation_id,
 * zone_states: Vec<ZoneState> }` frames over WebSocket at /live-twin (port
 * 8080). The preserved thermal viewer's legacy JSON shapes are still accepted
 * for backwards compatibility.
 */

export interface ZoneState {
  zone_id: number;
  t_air: number;
  t_mass: number;
  t_surface: number;
  rh: number;
  heating_setpoint: number;
  cooling_setpoint: number;
  heating_demand: number;
  cooling_demand: number;
  hvac_power_kw: number;
  energy_heating_kwh: number;
  energy_cooling_kwh: number;
  occupancy: number;
}

export interface LiveTwinPayload {
  timestamp: string;
  simulation_id: string;
  zone_states: ZoneState[];
}

export const DEFAULT_LIVE_TWIN_URL = "ws://localhost:8080/live-twin";

function isZoneState(v: unknown): v is ZoneState {
  if (typeof v !== "object" || v === null) return false;
  const z = v as Record<string, unknown>;
  return (
    typeof z.zone_id === "number" &&
    typeof z.t_air === "number" &&
    typeof z.t_mass === "number" &&
    typeof z.t_surface === "number"
  );
}

function isLiveTwinPayload(v: unknown): v is LiveTwinPayload {
  if (typeof v !== "object" || v === null) return false;
  const p = v as Record<string, unknown>;
  return Array.isArray(p.zone_states);
}

/**
 * Decodes one LiveTwin frame. Accepts:
 * - MessagePack binaries (the production broadcaster, rmp-serde encoded)
 * - JSON strings/objects in the legacy shapes understood by the preserved
 *   thermal viewer (`{zone_temperatures: {...}}` or
 *   `{type: "thermal_update", temperatures: [{zone_id, temperature}]}`)
 */
export function decodeLiveTwinFrame(data: unknown): LiveTwinPayload | null {
  let parsed: unknown = data;
  try {
    if (data instanceof ArrayBuffer) {
      parsed = msgpackDecode(new Uint8Array(data));
    } else if (data instanceof Uint8Array) {
      parsed = msgpackDecode(data);
    } else if (typeof data === "string") {
      parsed = JSON.parse(data);
    }
  } catch {
    return null;
  }

  if (isLiveTwinPayload(parsed)) {
    if (parsed.zone_states.every(isZoneState)) return parsed;
    return null;
  }

  // Legacy JSON fallback shapes from the preserved viewer.
  if (typeof parsed === "object" && parsed !== null) {
    const legacy = parsed as Record<string, unknown>;
    const zoneTemps = legacy.zone_temperatures;
    if (zoneTemps && typeof zoneTemps === "object") {
      const states: ZoneState[] = Object.entries(
        zoneTemps as Record<string, unknown>,
      ).map(([zoneKey, val]) => {
        const temperature =
          typeof val === "number"
            ? val
            : typeof val === "object" && val !== null &&
                typeof (val as Record<string, unknown>).temperature === "number"
              ? (val as Record<string, unknown>).temperature as number
              : 20;
        return syntheticZoneState(zoneNumber(zoneKey), temperature);
      });
      if (states.length > 0) return wrapLegacy(states);
    }
    if (
      legacy.type === "thermal_update" &&
      Array.isArray(legacy.temperatures)
    ) {
      const states: ZoneState[] = [];
      for (const t of legacy.temperatures) {
        if (
          typeof t === "object" && t !== null &&
          typeof (t as Record<string, unknown>).zone_id !== "undefined" &&
          typeof (t as Record<string, unknown>).temperature === "number"
        ) {
          const rec = t as Record<string, unknown>;
          states.push(
            syntheticZoneState(
              zoneNumber(String(rec.zone_id)),
              rec.temperature as number,
            ),
          );
        }
      }
      if (states.length > 0) return wrapLegacy(states);
    }
  }
  return null;
}

function wrapLegacy(zone_states: ZoneState[]): LiveTwinPayload {
  return {
    timestamp: new Date().toISOString(),
    simulation_id: "legacy",
    zone_states,
  };
}

export function syntheticZoneState(zoneId: number, tAir: number): ZoneState {
  return {
    zone_id: zoneId,
    t_air: tAir,
    t_mass: tAir,
    t_surface: tAir,
    rh: 50,
    heating_setpoint: 21,
    cooling_setpoint: 26,
    heating_demand: 0,
    cooling_demand: 0,
    hvac_power_kw: 0,
    energy_heating_kwh: 0,
    energy_cooling_kwh: 0,
    occupancy: 0,
  };
}

/** Extracts a numeric zone index from ids like "zone-3" (LiveTwin uses usize). */
export function zoneNumber(zoneId: string | number): number {
  if (typeof zoneId === "number") return zoneId;
  const m = zoneId.match(/(\d+)\s*$/);
  if (m) return Number(m[1]);
  return -1;
}
