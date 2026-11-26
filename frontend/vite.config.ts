import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import fs from "node:fs";
import path from "node:path";

const SIMULATE_TARGET = process.env.SIMULATE_TARGET || "http://127.0.0.1:8000";
const DASHBOARD_DIR = path.resolve(__dirname, "../dashboard");
const DASHBOARD_DATA_DIR = path.join(DASHBOARD_DIR, "data");

/**
 * Serves ../dashboard/data/ under /data during `vite dev` and `vite preview`,
 * and clears stale hashed assets in ../dashboard/assets before writing a build
 * while preserving ../dashboard/data/demo_windows.json and ../dashboard/README.md.
 */
function dashboardIntegrationPlugin(): Plugin {
  const serveDataMiddleware = (
    req: { url?: string },
    res: { statusCode: number; setHeader: (k: string, v: string) => void; end: (b?: Buffer | string) => void },
    next: () => void,
  ) => {
    const cleanUrl = (req.url ?? "").split("?")[0];
    if (!cleanUrl.startsWith("/data/")) return next();
    const rel = cleanUrl.slice("/data/".length);
    if (!rel || rel.includes("..")) return next();
    const filePath = path.join(DASHBOARD_DATA_DIR, rel);
    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) return next();
    res.setHeader("Content-Type", filePath.endsWith(".json") ? "application/json; charset=utf-8" : "application/octet-stream");
    res.end(fs.readFileSync(filePath));
  };

  return {
    name: "dashboard-integration",
    configureServer(server) {
      server.middlewares.use(serveDataMiddleware);
    },
    configurePreviewServer(server) {
      server.middlewares.use(serveDataMiddleware);
    },
    buildStart() {
      if (process.env.npm_lifecycle_event === "build" || process.argv.includes("build")) {
        fs.rmSync(path.join(DASHBOARD_DIR, "assets"), { recursive: true, force: true });
      }
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss(), dashboardIntegrationPlugin()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: false,
    allowedHosts: true,
    proxy: {
      "/simulate": { target: SIMULATE_TARGET, changeOrigin: true },
      "/health": { target: SIMULATE_TARGET, changeOrigin: true },
      "/meta": { target: SIMULATE_TARGET, changeOrigin: true },
    },
  },
  build: {
    outDir: DASHBOARD_DIR,
    emptyOutDir: false,
    target: "es2022",
    chunkSizeWarningLimit: 1600,
  },
});
