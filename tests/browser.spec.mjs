import { execFileSync } from "node:child_process";
import { readFileSync, readdirSync } from "node:fs";
import { expect, test } from "@playwright/test";

const demo = new URL("../demo/", import.meta.url);
const scan = readFileSync(new URL("model-url.txt", demo), "ascii").trim();
const { hash } = new URL(scan);
const staticFiles = ["/", "/app.mjs", "/runtime.mjs", "/style.css"];

test("site contains only the static interface and inference code", async ({ request }) => {
  const files = readdirSync(new URL("../dist/", import.meta.url)).sort();
  expect(files).toEqual(["_headers", "app.mjs", "index.html", "runtime.mjs", "style.css"]);
  for (const file of files) {
    const source = readFileSync(new URL("../dist/" + file, import.meta.url), "utf8");
    expect(source).not.toContain(hash.slice(4));
    expect(source).not.toMatch(/[A-Za-z0-9_$*+./:-]{180}/);
    expect(source).not.toMatch(/\b(fetch\s*\(|XMLHttpRequest|WebSocket|sendBeacon|indexedDB|localStorage)\b/);
  }
  expect((await request.get("/model.qrl")).status()).toBe(404);
  expect((await request.get("/qrllm.png")).status()).toBe(404);
});

test("a scanned link loads through static requests only", async ({ page, baseURL }) => {
  const seen = [];
  const errors = [];
  page.on("request", request => seen.push(request.url()));
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/" + hash);
  await expect(page.getByLabel("Starting letters")).toBeVisible();
  await expect(page.getByLabel("Generated name")).toHaveText(/^[A-Z][a-z]*$/);
  await expect(page.getByRole("link", { name: "Back to ahamedwajibu.com" })).toHaveAttribute("href", "https://ahamedwajibu.com/");
  expect(seen.length).toBeGreaterThanOrEqual(4);
  for (const request of seen) {
    const url = new URL(request);
    expect(url.origin).toBe(baseURL);
    expect(staticFiles).toContain(url.pathname);
    expect(url.search + url.hash).toBe("");
  }
  expect(errors).toEqual([]);
});

test("removing and restoring the QR state controls offline inference", async ({ page, context }) => {
  const seen = [];
  const errors = [];
  await page.goto("/");
  const noPayload = page.getByText("what are you doing here?", { exact: true });
  const prefix = page.getByLabel("Starting letters");
  const result = page.getByLabel("Generated name");
  await expect(noPayload).toBeVisible();
  await expect(prefix).toBeHidden();
  await expect(result).toBeHidden();
  await expect(page.getByRole("link")).toHaveCount(0);
  await context.setOffline(true);
  // Explicitly block HTTP even on engines that treat loopback as reachable while offline.
  await context.route("**/*", route => route.abort());
  page.on("request", request => seen.push(request.url()));
  page.on("pageerror", error => errors.push(error.message));
  await page.evaluate(value => { location.hash = value; }, hash);
  await expect(prefix).toBeVisible();
  await prefix.fill("el");
  await prefix.press("Enter");
  await expect(result).toHaveAttribute("data-state", "ready");
  await expect(result).toHaveText(/^El[a-z]*$/);
  await page.evaluate(() => { location.hash = ""; });
  await expect(noPayload).toBeVisible();
  await expect(prefix).toBeHidden();
  await page.evaluate(() => { location.hash = "#m=A000000"; });
  await expect(page.getByText("Couldn’t load the model.", { exact: true })).toBeVisible();
  await expect(prefix).toBeHidden();
  await page.evaluate(value => { location.hash = value; }, hash);
  await expect(prefix).toBeVisible();
  await expect(result).toHaveAttribute("data-state", "ready");
  await expect(result).toHaveText(/^[A-Z][a-z]*$/);
  expect(seen).toEqual([]);
  expect(errors).toEqual([]);
});

test("a screenshot of the resized PNG preserves the complete URL", async ({ page }, info) => {
  test.skip(info.project.name !== "chromium", "One independent screenshot round trip is sufficient.");
  const png = readFileSync(new URL("qrllm.png", demo)).toString("base64");
  await page.setContent('<img width="720" alt="QR code" src="data:image/png;base64,' + png + '">');
  const file = info.outputPath("qr-screenshot.png");
  await page.getByAltText("QR code").screenshot({ path: file });
  execFileSync("python", ["tools/check_qr.py", "--screenshot", file]);
});
