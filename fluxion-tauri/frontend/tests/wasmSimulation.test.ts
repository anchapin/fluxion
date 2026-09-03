import { describe, expect, it } from "vitest";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { wasmTempsToZoneStates } from "../src/sim/wasmSimulation";

const wasmDir = fileURLToPath(new URL("../public/wasm", import.meta.url));
const pkgPresent = existsSync(`${wasmDir}/fluxion_wasm.js`);

describe("wasmTempsToZoneStates", () => {
  it("maps 0-based wasm zones onto 1-based geometry zone ids", () => {
    const zones = wasmTempsToZoneStates([20.5, 22.0, 24.5]);
    expect(zones.size).toBe(3);
    expect(zones.get(1)!.t_air).toBeCloseTo(20.5);
    expect(zones.get(2)!.t_air).toBeCloseTo(22.0);
    expect(zones.get(3)!.t_air).toBeCloseTo(24.5);
    // LiveTwin-compatible ZoneState shape for the rendering path.
    expect(zones.get(1)).toHaveProperty("t_mass");
    expect(zones.get(1)).toHaveProperty("hvac_power_kw");
  });
});

describe("fluxion-wasm FluidSimulation (optional pkg)", () => {
  it(
    "loads and steps the real wasm module when the pkg is generated",
    { timeout: 20000 },
    async () => {
      if (!pkgPresent) {
        console.warn("public/wasm pkg absent — skipping live wasm check");
        return;
      }

      // Serve the .wasm binary over http for the web-target loader.
      const server = createServer(async (_req, res) => {
        const data = await readFile(`${wasmDir}/fluxion_wasm_bg.wasm`);
        res.setHeader("content-type", "application/wasm");
        res.end(data);
      });
      await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
      const port = (server.address() as { port: number }).port;

      try {
        const mod = (await import(
          /* @vite-ignore */ `${wasmDir}/fluxion_wasm.js`
        )) as unknown as {
          default: (path: string) => Promise<unknown>;
          FluidSimulation: new (config: string) => {
            step: (dt: number) => void;
            get_zone_temps: () => Float64Array;
            num_zones: () => number;
            current_hour: () => number;
          };
        };
        await mod.default(`http://127.0.0.1:${port}/fluxion_wasm_bg.wasm`);

        const sim = new mod.FluidSimulation(
          JSON.stringify({
            building: "fluxion_gui_test",
            num_zones: 3,
            initial_temps: [20, 21, 22],
            heating_setpoint: 20,
            cooling_setpoint: 24,
          }),
        );
        expect(sim.num_zones()).toBe(3);

        const before = Array.from(sim.get_zone_temps());
        for (let i = 0; i < 24; i++) sim.step(1.0);
        const after = Array.from(sim.get_zone_temps());
        expect(after.length).toBe(3);
        // A day of physics must move at least one zone temperature.
        expect(after.some((t, i) => Math.abs(t - before[i]) > 1e-6)).toBe(true);
        expect(sim.current_hour()).toBeGreaterThan(0);
      } finally {
        server.close();
      }
    },
  );
});
