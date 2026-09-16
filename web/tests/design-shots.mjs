// Design QA: capture every screen in both themes, for review.
// Uses manual masking only, so it never triggers detection and never spends a billed call.
//   node tests/design-shots.mjs
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const OUT = process.env.SHOT_DIR ?? "./test-output/design";
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
const shot = (name) => page.screenshot({ path: `${OUT}/${name}.png` });
const setTheme = async (theme) => {
  await page.evaluate((value) => {
    document.documentElement.dataset.theme = value;
    localStorage.setItem("ld-theme", value);
  }, theme);
  await page.waitForTimeout(250);
};

await page.goto("http://127.0.0.1:5173/", { waitUntil: "networkidle" });

for (const theme of ["dark", "light"]) {
  await setTheme(theme);
  await shot(`01-signin-${theme}`);
}

await setTheme("dark");
await page.getByRole("button", { name: "New here? Create an account" }).click();
await page.getByLabel("Name").fill("Design Review");
await page.getByLabel("Email").fill(`design-${Date.now()}@example.com`);
await page.getByLabel("Password").fill("correct horse battery");
await shot("02-register-dark");
await page.getByRole("button", { name: "Create account" }).click();
await page.waitForSelector("h1:has-text('New mock-up')", { timeout: 15000 });

const png = await page.evaluate(async () => {
  const c = document.createElement("canvas");
  c.width = 1200;
  c.height = 700;
  const x = c.getContext("2d");
  const sky = x.createLinearGradient(0, 0, 0, 700);
  sky.addColorStop(0, "#9ec5e8");
  sky.addColorStop(1, "#dfe9f2");
  x.fillStyle = sky; x.fillRect(0, 0, 1200, 700);
  x.fillStyle = "#6b5546";
  x.beginPath(); x.moveTo(140, 330); x.lineTo(600, 170); x.lineTo(1060, 330); x.closePath(); x.fill();
  x.fillStyle = "#efe7db"; x.fillRect(160, 330, 880, 250);
  x.fillStyle = "#4a3b30"; x.fillRect(760, 400, 220, 180);
  x.fillStyle = "#3f6b3a"; x.fillRect(180, 600, 300, 60);
  const blob = await new Promise((r) => c.toBlob(r, "image/png"));
  return [...new Uint8Array(await blob.arrayBuffer())];
});
writeFileSync(`${OUT}/house.png`, Buffer.from(png));

await page.setInputFiles('input[type="file"]', `${OUT}/house.png`);
await page.fill("textarea", "C9 warm white on the front roofline, mini lights in the hedges");
for (const theme of ["dark", "light"]) {
  await setTheme(theme);
  await shot(`03-upload-${theme}`);
}

await setTheme("dark");
await page.getByRole("button", { name: "Continue" }).click();
await page.waitForSelector("text=Choose masking mode", { timeout: 20000 });
for (const theme of ["dark", "light"]) {
  await setTheme(theme);
  await shot(`04-mode-${theme}`);
}

await setTheme("dark");
await page.getByRole("button", { name: /Manual/ }).click();
await page.waitForSelector(".sidebar", { timeout: 20000 });
await page.getByRole("button", { name: "Roofline", exact: false }).first().click();
const canvas = page.locator("canvas.canvas");
const box = await canvas.boundingBox();
for (const [fx, fy] of [[0.14, 0.47], [0.5, 0.25], [0.87, 0.47]]) {
  await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
  await page.waitForTimeout(120);
}
await page.keyboard.press("Enter");
await page.waitForTimeout(1200);

for (const theme of ["dark", "light"]) {
  await setTheme(theme);
  await shot(`05-workspace-${theme}`);
}

await browser.close();
console.log(`screenshots in ${OUT}`);
