import { defineConfig, devices } from "@playwright/test";

// Smoke test of the deployed site:            npm run test:live
// The same test on the local production build: npm run build, then LIVE_URL=http://localhost:4173 npm run test:live
const baseURL = process.env.LIVE_URL ?? "https://fabricscan-ai.vercel.app";
const local = baseURL.includes("localhost");

export default defineConfig({
  testDir: "./e2e",
  testMatch: "live.spec.ts",
  timeout: 240_000,
  expect: { timeout: 180_000 },
  reporter: [["list"]],
  use: { baseURL, channel: "msedge" },
  projects: [{ name: "live", use: { ...devices["Desktop Edge"], channel: "msedge", viewport: { width: 1366, height: 900 } } }],
  webServer: local
    ? { command: "npx vite preview --port 4173 --strictPort", url: "http://localhost:4173", reuseExistingServer: false, timeout: 60_000 }
    : undefined,
});
