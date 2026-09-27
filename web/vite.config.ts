import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// VITE_API_PROXY points the dev server at another API, e.g. a mock-provider one for UI tests.
const api = { target: process.env.VITE_API_PROXY ?? "http://127.0.0.1:8000", changeOrigin: true };

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    watch: { usePolling: true },
    allowedHosts: true,
    proxy: { "/auth": api, "/photos": api, "/products": api, "/files": api, "/health": api },
  },
});
