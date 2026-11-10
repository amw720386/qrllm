import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  testMatch: "**/*.spec.mjs",
  use: { baseURL: "http://127.0.0.1:4174" },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "android-emulation", use: { ...devices["Pixel 7"] } },
    { name: "ios-emulation", use: { ...devices["iPhone 13"] } },
  ],
  webServer: {
    command: "python -m http.server 4174 --bind 127.0.0.1 --directory dist",
    url: "http://127.0.0.1:4174",
  },
});
