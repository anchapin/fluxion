import { describe, expect, it } from "vitest";
import { encode } from "@msgpack/msgpack";
import {
  DEFAULT_LIVE_TWIN_URL,
  decodeLiveTwinFrame,
  zoneNumber,
  type LiveTwinPayload,
} from "../src/livetwin/protocol";

function samplePayload(): LiveTwinPayload {
  return {
    timestamp: "2026-09-02T12:00:00Z",
    simulation_id: "01234567-89ab-cdef-0123-456789abcdef",
    zone_states: [
      {
        zone_id: 1,
        t_air: 21.5,
        t_mass: 20.9,
        t_surface: 21.1,
        rh: 48,
        heating_setpoint: 21,
        cooling_setpoint: 26,
        heating_demand: 0.4,
        cooling_demand: 0,
        hvac_power_kw: 1.2,
        energy_heating_kwh: 3.4,
        energy_cooling_kwh: 0,
        occupancy: 2,
      },
    ],
  };
}

describe("decodeLiveTwinFrame — production MessagePack frames", () => {
  it("decodes an rmp-serde style binary frame", () => {
    const bytes = encode(samplePayload());
    const buf = bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength,
    );
    const decoded = decodeLiveTwinFrame(buf);
    expect(decoded).not.toBeNull();
    expect(decoded!.zone_states.length).toBe(1);
    expect(decoded!.zone_states[0].zone_id).toBe(1);
    expect(decoded!.zone_states[0].t_air).toBeCloseTo(21.5);
  });

  it("accepts Uint8Array frames directly", () => {
    const decoded = decodeLiveTwinFrame(encode(samplePayload()));
    expect(decoded).not.toBeNull();
    expect(decoded!.simulation_id).toBe(samplePayload().simulation_id);
  });

  it("rejects binary garbage", () => {
    expect(decodeLiveTwinFrame(new Uint8Array([0xc1, 0xff, 0x00]))).toBeNull();
  });
});

describe("decodeLiveTwinFrame — legacy JSON shapes (preserved viewer)", () => {
  it("decodes {zone_temperatures: {…}} maps", () => {
    const decoded = decodeLiveTwinFrame(
      JSON.stringify({ zone_temperatures: { "1": 22.1, "2": { temperature: 23.4 } } }),
    );
    expect(decoded).not.toBeNull();
    const byZone = new Map(decoded!.zone_states.map((z) => [z.zone_id, z.t_air]));
    expect(byZone.get(1)).toBeCloseTo(22.1);
    expect(byZone.get(2)).toBeCloseTo(23.4);
  });

  it("decodes thermal_update arrays", () => {
    const decoded = decodeLiveTwinFrame(
      JSON.stringify({
        type: "thermal_update",
        temperatures: [
          { zone_id: 1, temperature: 20.0 },
          { zone_id: 2, temperature: 25.5 },
        ],
      }),
    );
    expect(decoded).not.toBeNull();
    expect(decoded!.zone_states.length).toBe(2);
    expect(decoded!.zone_states[1].t_air).toBeCloseTo(25.5);
  });

  it("returns null for unrelated JSON", () => {
    expect(decodeLiveTwinFrame(JSON.stringify({ hello: "world" }))).toBeNull();
  });
});

describe("zoneNumber", () => {
  it("extracts numeric suffixes from geometry zone ids", () => {
    expect(zoneNumber("zone-1")).toBe(1);
    expect(zoneNumber("zone-42")).toBe(42);
    expect(zoneNumber(7)).toBe(7);
    expect(zoneNumber("no-number")).toBe(-1);
  });
});

describe("default endpoint", () => {
  it("matches the LiveTwin broadcaster route (port 8080)", () => {
    expect(DEFAULT_LIVE_TWIN_URL).toBe("ws://localhost:8080/live-twin");
  });
});
