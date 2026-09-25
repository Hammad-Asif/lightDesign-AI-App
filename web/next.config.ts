import type { NextConfig } from "next";

// The Python backend (FastAPI, ../app.py) owns every /api route. Proxying it through Next keeps one
// origin, so the HttpOnly session cookie the backend sets works with no CORS.
const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8765";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${BACKEND_URL}/api/:path*` }];
  },
  experimental: {
    proxyTimeout: 180_000, // SAM 3 calls with extra passes can take well over the 30 s default
  },
};

export default nextConfig;
