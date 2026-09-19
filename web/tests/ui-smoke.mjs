import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const OUT = process.env.SHOT_DIR ?? "./test-output";
// Defaults are the usual dev servers; set both to point at a mock-provider pair on other ports.
const API = process.env.API_URL ?? "http://127.0.0.1:8000";
const UI = process.env.UI_URL ?? "http://127.0.0.1:5173/";
mkdirSync(OUT, { recursive: true });
const errors = [];
let failed = false;

function check(label, ok) {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
  if (!ok) failed = true;
}


const health = await fetch(`${API}/health`).then((r) => r.json());
if (health.segmentation_provider !== "mock") {
  console.error(
    `This suite asserts against the mock provider's fixed regions, but the API is running ` +
    `"${health.segmentation_provider}". Restart it without SEGMENTATION_PROVIDER set, then re-run.`,
  );
  process.exit(2);
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(String(e)));

await page.goto(UI, { waitUntil: "networkidle" });
check("sign-in screen shows first", (await page.locator("h1").innerText()) === "Sign in");
await page.getByRole("button", { name: "New here? Create an account" }).click();
await page.getByLabel("Name").fill("UI Smoke");
await page.getByLabel("Email").fill(`ui-smoke-${Date.now()}@example.com`);
await page.getByLabel("Password").fill("correct horse battery");
await page.getByRole("button", { name: "Create account" }).click();
await page.waitForSelector("h1:has-text('New mock-up')", { timeout: 15000 });
check("app mounts", (await page.locator("h1").innerText()) === "New mock-up");

const png = await page.evaluate(async () => {
  const canvas = document.createElement("canvas");
  canvas.width = 1200;
  canvas.height = 800;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#7fa6cc";
  ctx.fillRect(0, 0, 1200, 800);
  ctx.fillStyle = "#8a6a52";
  ctx.beginPath();
  ctx.moveTo(120, 340);
  ctx.lineTo(600, 150);
  ctx.lineTo(1080, 340);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "#3f6b3a";
  ctx.fillRect(80, 560, 400, 120);
  const blob = await new Promise((r) => canvas.toBlob(r, "image/png"));
  return [...new Uint8Array(await blob.arrayBuffer())];
});
writeFileSync(`${OUT}/house.png`, Buffer.from(png));

await page.setInputFiles('input[type="file"]', `${OUT}/house.png`);
await page.fill("textarea", "C9 on the front roofline, mini lights in the hedges and palm trunks");
await page.getByRole("button", { name: "Continue" }).click();

await page.waitForSelector("text=Check the instruction", { timeout: 15000 });
check("mode chooser appears", true);
// The instruction is read before any detection: one row per clause, resolved to a product.
const preview = await page.locator(".spec-row").allInnerTexts();
check(`instruction read as requests before detection (${preview.length})`, preview.length === 2);
check("roofline request resolved to a C9 line with the outline edges",
  preview[0].includes("Roofline") && preview[0].includes("eaves, rakes, hips") && preview[0].includes("C9 LED"));
check("hedges and trunks share one strand request, warm white assumed",
  preview[1].includes("Hedge") && preview[1].includes("Palm trunk") && preview[1].includes("Mini lights, warm white")
  && preview[1].includes("Assumed: warm white"));
await page.screenshot({ path: `${OUT}/01-mode.png` });
// Re-reading a corrected instruction costs nothing and needs no detection.
await page.fill("textarea", "red and green C9 on the front roofline at 15 inch spacing, mini lights in the hedges and palm trunks");
await page.getByRole("button", { name: "Read it again" }).click();
await page.waitForFunction(() => document.querySelector(".spec-row")?.textContent?.includes("Green"), null, { timeout: 15000 });
const reread = await page.locator(".spec-row").first().innerText();
check("a corrected instruction is re-read in place", reread.includes("Red") && reread.includes("15 in spacing"));

await page.getByRole("button", { name: /Automatic/ }).click();
await page.waitForSelector(".sidebar", { timeout: 120000 });
await page.waitForTimeout(1200);

const banner = await page.locator(".interpreted").innerText();
check("interpreted banner counts the requests and says nothing is confirmed",
  banner.includes("2 requests") && banner.includes("not confirmed"));

const kinds = await page.locator(".region[data-geometry='area'][data-role='target'] select").evaluateAll(
  (nodes) => [...new Set(nodes.map((n) => n.value))],
);
check(`only requested area kinds masked (${kinds.join(",")})`,
  kinds.every((k) => ["hedge", "palm_trunk"].includes(k)));

const regionCount = await page.locator(".region").count();
check(`regions listed in sidebar (${regionCount})`, regionCount >= 3);
const lineRegions = await page.locator('.region[data-geometry="line"]').count();
check(`roofline spans listed (${lineRegions})`, lineRegions >= 1);
check("eave spans present", (await page.locator('.region[data-edge="eave"]').count()) >= 1);
const options = await page.locator(".region select option").allInnerTexts();
check("side selector offers front/left/right", options.includes("front"));
check("edge selector offers eave/ridge", options.includes("eave") && options.includes("ridge"));
check("kind selector offers the house catalogue", options.includes("Driveway edge") && options.includes("Garage door"));
await page.screenshot({ path: `${OUT}/02-editor.png` });

const canvas = page.locator("canvas.canvas");
check("canvas present", (await canvas.count()) === 1);

const before = await canvas.screenshot();

const hedge = page.locator('.region[data-geometry="area"]').first();
await hedge.click();
await page.waitForTimeout(300);
check("brush tool enabled for area region", await page.getByRole("button", { name: "Brush" }).isEnabled());

const box = await canvas.boundingBox();
await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.75);
await page.mouse.down();
for (let i = 0; i <= 12; i += 1) {
  await page.mouse.move(box.x + box.width * (0.3 + i * 0.02), box.y + box.height * 0.75);
}
await page.mouse.up();
await page.waitForTimeout(900);

const after = await canvas.screenshot();
check("brush stroke changed the canvas", !before.equals(after));
await page.screenshot({ path: `${OUT}/03-brushed.png` });

const undoBtn = page.getByRole("button", { name: "Undo" });
check("undo became available", await undoBtn.isEnabled());
await undoBtn.click();
await page.waitForTimeout(900);
check("redo became available after undo", await page.getByRole("button", { name: "Redo" }).isEnabled());

const roofRegion = page.locator('.region[data-geometry="line"]').first();
await roofRegion.click();
await page.waitForTimeout(400);
check("vertex tool enabled for line region", await page.getByRole("button", { name: "Vertices" }).isEnabled());
check("brush disabled for line region", !(await page.getByRole("button", { name: "Brush" }).isEnabled()));
await page.screenshot({ path: `${OUT}/04-roofline.png` });

const sideSelect = roofRegion.locator("select").nth(1);
check("roofline has kind, side and edge selectors", (await roofRegion.locator("select").count()) === 3);
await sideSelect.selectOption("front");
await page.waitForTimeout(600);
check("roofline side set to front", (await sideSelect.inputValue()) === "front");

// Confirming the lighting: each request against the regions it will light, editable, then saved
// as the photo's design. Nothing renders from the instruction directly.
await page.getByRole("button", { name: "Review lighting" }).click();
await page.waitForSelector(".spec-bar .spec-row", { timeout: 10000 });
const rows = await page.locator(".spec-bar .spec-row").allInnerTexts();
check("lighting rows count the regions each request lights", rows.every((row) => /\d+ regions?|no regions yet/.test(row)));
check("roofline request lights at least one region", /[1-9]\d* regions?/.test(rows[0]));
const spacing = page.locator(".spec-bar .spec-row").first().getByTitle("Spacing");
await spacing.selectOption("18");
check("spacing can be corrected before confirming", (await spacing.inputValue()) === "18");
await page.getByRole("button", { name: "Confirm lighting" }).click();
await page.waitForFunction(() => document.querySelector(".interpreted")?.textContent?.includes("lighting confirmed"), null, { timeout: 10000 });
const confirmedBanner = await page.locator(".interpreted").innerText();
check(`design confirmed with regions (${confirmedBanner.match(/confirmed for (\d+)/)?.[1]})`,
  /confirmed for [1-9]\d* regions?/.test(confirmedBanner));
check("confirm button rests once confirmed", !(await page.getByRole("button", { name: "Confirm changes" }).isEnabled()));
await page.screenshot({ path: `${OUT}/04b-lighting.png` });
await page.getByRole("button", { name: "Hide lighting" }).click();

await page.getByRole("button", { name: "Approve mask set" }).click();
await page.waitForSelector(".badge.ok", { timeout: 10000 });
// Case-insensitive: the badge reads "Locked"; what matters is that it locks.
check("mask set locks on approve",
  (await page.locator(".badge.ok").innerText()).toLowerCase().includes("locked"));
await page.screenshot({ path: `${OUT}/05-approved.png` });

check("no console errors", errors.length === 0);
if (errors.length) console.log(errors.slice(0, 5));

await browser.close();
console.log(failed ? "\nUI smoke FAILED" : "\nAll UI smoke checks passed.");
process.exit(failed ? 1 : 0);
