import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // Tauri dev server conventions: fixed port, fail fast if taken.
  server: {
    port: 1420,
    strictPort: true,
    host: true,
  },
  build: {
    target: "es2021",
    // Tauri embeds these assets; keep output predictable and sourcemapped.
    sourcemap: true,
  },
  // Vitest picks up its own config from tests/setup; keep Vite lean.
});
