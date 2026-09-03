# fluxion-tauri — Fluxion Cross-Platform GUI

Tauri v2 desktop shell with a React + React Three Fiber (R3F) frontend, scaffolded in
issue #3178 as the cross-platform GUI foundation. The same frontend runs as a plain web
app (web fallback), with optional in-browser physics via the `fluxion-wasm` crate.

## Layout

```
fluxion-tauri/
├── src-tauri/              # Rust crate (workspace member, package `fluxion-tauri`)
│   ├── src/main.rs         #   Tauri app + invoke_handler registration
│   ├── src/commands.rs     #   IPC commands (geometry, summary, zone info, sim params)
│   ├── src/geometry.rs     #   BuildingGeometry sample + JSON contract (unit-tested)
│   ├── examples/dump_sample.rs  # regenerates the frontend contract fixture
│   ├── index.html          # LEGACY geometry viewer (superseded — kept as reference)
│   └── src/index.html      # LEGACY thermal viewer (superseded — kept as reference)
└── frontend/               # Vite + React 18 + TypeScript + R3F app (Tauri frontendDist)
    └── src/
        ├── scene/          # R3F Canvas, orbit camera, thermal GLSL material, surfaces
        ├── lib/            # geometry adapter/triangulation, thermal colormap, platform
        ├── livetwin/       # MessagePack LiveTwin WebSocket protocol + hook
        ├── sim/            # optional fluxion-wasm in-browser simulation
        ├── tauri/          # IPC service wrappers with web-mode fallbacks
        └── ui/             # sidebar, legends, control bar, params panel
```

Note: `fluxion-tauri/Cargo.toml` (the manifest directly under `fluxion-tauri/`, with its
own `[profile.release]`) is a leftover duplicate that is **not** a workspace member and
does not build; the canonical crate is `fluxion-tauri/src-tauri` (see root
`Cargo.toml` → `members`). Do not add code to the outer manifest.

## Legacy viewers (preserved)

Two hand-written viewers landed before the React frontend and are preserved in place as
reference implementations (do not serve them; `frontendDist` points at
`frontend/dist`):

- `src-tauri/index.html` — geometry viewer (issues #3177/#3179): parameter sliders,
  `load_geometry` IPC, zone overlay boxes.
- `src-tauri/src/index.html` — thermal viewer (issue #3249): GLSL thermal colormap,
  temperature legend, LiveTwin WebSocket, wireframe toggle.

The React app ports both feature sets (sliders → `ui/ParamsPanel.tsx`, thermal shader →
`scene/ThermalMaterial.tsx`, LiveTwin → `livetwin/`, wireframe/zone coloring →
`ui/ControlBar.tsx`). Known legacy gaps that remain open:

- The thermal viewer's `load_geometry_file` command was never implemented in
  `commands.rs`; file loading is not part of the React app either.
- The thermal viewer connected to `ws://localhost:8765/livetwin` with a JSON protocol.
  The React app implements the real broadcaster contract instead: MessagePack
  `LiveTwinPayload` frames on `ws://localhost:8080/live-twin`
  (`src/twin/live_twin_broadcaster.rs`), with the legacy JSON shapes still accepted.

## Commands

All npm commands assume Node ≥ 18. Rust commands assume the usual Fluxion toolchain.

```bash
# Frontend (from fluxion-tauri/frontend)
npm install
npm run dev                # Vite dev server on :1420 (Tauri devUrl)
npm run test               # vitest (29 tests incl. optional live wasm check)
npm run build              # tsc + vite build -> frontend/dist

# Desktop (from fluxion-tauri/frontend; scripts cd to fluxion-tauri/)
npm run tauri:dev          # dev app (runs vite + cargo run)
npm run tauri:build        # release bundle (use `-- --debug` for fast builds)

# Rust (from repo root)
cargo test -p fluxion-tauri          # 11 unit tests (geometry + commands)
cargo check -p fluxion-tauri

# Regenerate the cross-language contract fixture after changing geometry.rs
cargo run --example dump_sample -p fluxion-tauri \
  > fluxion-tauri/frontend/tests/fixtures/rust-sample-geometry.json
```

**Run `tauri` builds from `fluxion-tauri/` (the npm scripts do this for you).** Launched
from the repo root, the Tauri CLI misidentifies the sibling `npm/` package as the
frontend and runs its `napi build` instead.

## Web fallback

`npm run build` produces a browser-ready bundle (`frontend/dist`). In a browser the app
detects the absence of Tauri IPC (`src/lib/platform.ts`), renders the embedded sample
geometry (a TS mirror of the Rust `BuildingGeometry::sample()`), and disables the
parameter panel. LiveTwin still connects to a running simulation backend, and the
optional wasm module adds client-side physics:

```bash
# Optional: build fluxion-wasm into the frontend (gitignored artifact)
wasm-pack build --target web --out-dir ../fluxion-tauri/frontend/public/wasm ../fluxion-wasm
```

When `public/wasm/` exists, the **WASM Sim** button steps a `FluidSimulation`
(3 zones matching the sample building) in-browser and drives the thermal view without
any backend. The geometry command itself stays behind Tauri IPC; exposing building
geometry through `fluxion-wasm` is future work (the crate currently exports simulation,
not geometry — see `fluxion-wasm/WASM_STATUS.md`).

## Verification status

| Path | Status |
|------|--------|
| `cargo test -p fluxion-tauri` (Linux) | ✅ 11/11 |
| `npm run test` (vitest) | ✅ 29/29 (incl. live wasm module when the pkg is built) |
| `npm run build` (web bundle) | ✅ |
| Web-mode 3D canvas render | ✅ headless Chrome (SwiftShader WebGL) screenshot |
| `tauri build --debug` (Linux) | ✅ ELF binary + `.deb` + `.rpm` |
| AppImage bundling | ⚠️ blocked by `linuxdeploy` in this environment (FUSE/user-ns) |
| macOS / Windows | ❌ untested — no hardware; toolchain is standard Tauri v2 |

## Issue

#3178 — feat(gui): Scaffold Tauri + R3F workspace for cross-platform GUI.
