import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const OUT = process.env.SHOT_DIR ?? "./test-output";
mkdirSync(OUT, { recursive: true });
const errors = [];
let failed = false;

function check(label, ok) {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
  if (!ok) failed = true;
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1500, height: 980 } });
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(String(e)));

await page.goto("http://127.0.0.1:5173/", { waitUntil: "networkidle" });

// Every photo belongs to an account, so start by registering a fresh one.
await page.getByRole("button", { name: "New here? Create an account" }).click();
await page.getByLabel("Name").fill("Manual Mask Test");
await page.getByLabel("Email").fill(`manual-mask-${Date.now()}@example.com`);
await page.getByLabel("Password").fill("correct horse battery");
await page.getByRole("button", { name: "Create account" }).click();
await page.waitForSelector("h1:has-text('New mock-up')", { timeout: 15000 });

const png = await page.evaluate(async () => {
  const c = document.createElement("canvas");
  c.width = 1200;
  c.height = 700;
  const x = c.getContext("2d");
  x.fillStyle = "#8fb8dd"; x.fillRect(0, 0, 1200, 700);
  x.fillStyle = "#6b5546";
  x.beginPath(); x.moveTo(140, 330); x.lineTo(600, 170); x.lineTo(1060, 330); x.closePath(); x.fill();
  x.fillStyle = "#e8e2d8"; x.fillRect(160, 330, 880, 250);
  x.fillStyle = "#4a3b30"; x.fillRect(760, 400, 220, 180);
  x.fillStyle = "#3f6b3a"; x.fillRect(180, 600, 300, 60);
  const blob = await new Promise((r) => c.toBlob(r, "image/png"));
  return [...new Uint8Array(await blob.arrayBuffer())];
});
const { writeFileSync } = await import("node:fs");
writeFileSync(`${OUT}/house.png`, Buffer.from(png));

await page.setInputFiles('input[type="file"]', `${OUT}/house.png`);
await page.fill("textarea", "manual masking demo");
await page.getByRole("button", { name: "Continue" }).click();
await page.waitForSelector("text=Choose masking mode", { timeout: 20000 });

await page.getByRole("button", { name: /Manual/ }).click();
await page.waitForSelector(".sidebar", { timeout: 20000 });
check("manual mode opens the editor with no regions", (await page.locator(".region").count()) === 0);

const groups = await page.locator(".add-group h3").allInnerTexts();
check(`add panel is grouped (${groups.join(" / ")})`, groups.length >= 3);
for (const name of ["Roofline", "Driveway edge", "Window", "Garage door", "Column / post", "Hedge"]) {
  check(`"${name}" available to add`, (await page.getByRole("button", { name, exact: false }).count()) > 0);
}
const swatches = await page.locator(".add-kind .swatch").evaluateAll((n) =>
  n.map((e) => getComputedStyle(e).backgroundColor),
);
check(`every add button has its own colour (${new Set(swatches).size} distinct)`,
  new Set(swatches).size === swatches.length);
await page.screenshot({ path: `${OUT}/10-add-panel.png` });

await page.getByRole("button", { name: "Roofline", exact: false }).first().click();
check("drawing hint appears", (await page.locator(".drawing-hint").count()) === 1);

const canvas = page.locator("canvas.canvas");
const box = await canvas.boundingBox();
const pts = [[0.14, 0.47], [0.5, 0.25], [0.87, 0.47]];
for (const [fx, fy] of pts) {
  await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
  await page.waitForTimeout(120);
}
await page.screenshot({ path: `${OUT}/11-drawing.png` });
await page.keyboard.press("Enter");
await page.waitForTimeout(1200);

check("roofline created without any detection", (await page.locator(".region[data-geometry='line']").count()) === 1);
const firstLabel = await page.locator(".region[data-geometry='line'] input").first().inputValue();
check(`created region is a roofline ("${firstLabel}")`, firstLabel.toLowerCase().includes("roof"));
check("it is marked manual", (await page.locator(".region[data-geometry='line']").innerText()).includes("manual"));
await page.screenshot({ path: `${OUT}/12-roofline-added.png` });

// Paint an obstruction across the roofline: the stretch behind it must be reported as hidden.
await page.getByRole("button", { name: "Obstruction", exact: false }).first().click();
await page.waitForTimeout(900);
const lineBox = await canvas.boundingBox();
await page.mouse.move(lineBox.x + lineBox.width * 0.28, lineBox.y + lineBox.height * 0.31);
await page.mouse.down();
await page.mouse.move(lineBox.x + lineBox.width * 0.32, lineBox.y + lineBox.height * 0.43, { steps: 10 });
await page.mouse.up();
await page.waitForTimeout(1800);
const hiddenNote = await page.locator(".region[data-geometry='line'] .hidden-note").first()
  .innerText().catch(() => "");
check(`painting an obstruction across the roofline marks it hidden ("${hiddenNote}")`,
  /passes behind objects/.test(hiddenNote));
await page.screenshot({ path: `${OUT}/12b-obstruction.png` });

await page.getByRole("button", { name: "Custom region", exact: false }).first().click();
await page.waitForTimeout(900);
const colour = page.locator('.region input[type="color"]');
check("custom region exposes a colour picker", (await colour.count()) === 1);
await colour.fill("#ff0066");
await page.waitForTimeout(900);
const swatch = await page
  .locator(".region[data-geometry='area']", { has: page.locator('input[type="color"]') })
  .locator(".swatch").first()
  .evaluate((e) => getComputedStyle(e).backgroundColor);
check(`custom colour applied (${swatch})`, swatch === "rgb(255, 0, 102)");

const roofSwatch = await page.locator(".region[data-geometry='line'] .swatch").first()
  .evaluate((e) => getComputedStyle(e).backgroundColor);
check("non-custom region keeps its own kind colour", roofSwatch !== swatch);
await page.screenshot({ path: `${OUT}/13-custom-colour.png` });

const beforeClick = await page.locator(".region").count();
await page.getByRole("button", { name: "Click object", exact: true }).click();
const cbox = await canvas.boundingBox();
await page.mouse.click(cbox.x + cbox.width * 0.72, cbox.y + cbox.height * 0.68);
await page.waitForTimeout(1400);
const afterClick = await page.locator(".region").count();
check(`click on the photo adds an object (${beforeClick} -> ${afterClick})`, afterClick === beforeClick + 1);
await page.screenshot({ path: `${OUT}/14-click-add.png` });

await page.keyboard.down("Alt");
await page.mouse.click(cbox.x + cbox.width * 0.72, cbox.y + cbox.height * 0.68);
await page.keyboard.up("Alt");
await page.waitForTimeout(1400);
const afterAlt = await page.locator(".region").count();
check(`alt-click removes the object under the cursor (${afterClick} -> ${afterAlt})`, afterAlt === afterClick - 1);
await page.screenshot({ path: `${OUT}/15-click-remove.png` });

check("no console errors", errors.length === 0);
if (errors.length) console.log(errors.slice(0, 5));

await browser.close();
console.log(failed ? "\nManual-mask suite FAILED" : "\nAll manual-mask checks passed.");
process.exit(failed ? 1 : 0);
