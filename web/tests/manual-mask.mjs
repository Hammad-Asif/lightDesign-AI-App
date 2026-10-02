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
// Tool guides (components/ToolGuide.tsx) open on a tool's first use; these checks are about the tools.
await page.addInitScript(() => localStorage.setItem("ld-tool-guides-seen",
  JSON.stringify(["select", "brush", "erase", "vertex", "move", "box", "click", "find", "measure", "wreath", "house"])));
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(String(e)));

await page.goto(process.env.UI_URL ?? "http://127.0.0.1:5173/", { waitUntil: "networkidle" });

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
await page.waitForSelector("text=Check the instruction", { timeout: 20000 });

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
// A hand-drawn roofline gets a suggested type, shown in the picker for the salesperson to confirm.
check("hand-drawn roofline opens the type picker", (await page.locator(".edge-picker").count()) === 1);
check("a type is suggested for it",
  (await page.locator(".edge-picker .chip.on").count()) === 1 &&
  (await page.locator(".region[data-geometry='line'][data-edge='unknown']").count()) === 0);
await page.locator(".edge-picker").getByRole("button", { name: "Done" }).click();
check("Done closes the picker", (await page.locator(".edge-picker").count()) === 0);
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

// Move and extend: the two operations that did not exist before Day 3.
const polylines = async () =>
  page.evaluate(async () => {
    const list = await (await fetch("/photos", { credentials: "include" })).json();
    const detail = await (await fetch(`/photos/${list[0].id}`, { credentials: "include" })).json();
    return detail.regions.filter((r) => r.geometry === "line").map((r) => r.polyline);
  });
const toClient = (box, p) => [box.x + (p[0] / 1200) * box.width, box.y + (p[1] / 700) * box.height];
const same = (a, b, slack = 8) => Math.abs(a - b) <= slack;

// Clicking a region on the photo selects it, instead of hunting for its row in the panel.
await page.getByRole("button", { name: "Select" }).click();
const selectBox = await canvas.boundingBox();
const firstLine = (await polylines())[0];
const mid = toClient(selectBox, [
  (firstLine[0][0] + firstLine[1][0]) / 2,
  (firstLine[0][1] + firstLine[1][1]) / 2,
]);
await page.mouse.click(mid[0], mid[1]);
await page.waitForTimeout(400);
check("clicking a line on the photo selects it",
  (await page.locator(".region.active[data-geometry='line']").count()) === 1);
// Below the roofline, which would otherwise win the click: a line takes priority over a mask.
await page.mouse.click(selectBox.x + selectBox.width * 0.315, selectBox.y + selectBox.height * 0.42);
await page.waitForTimeout(400);
check("clicking a painted area on the photo selects it",
  (await page.locator(".region.active[data-geometry='area']").count()) === 1);
await page.screenshot({ path: `${OUT}/12c-select.png` });


// Painting the obstruction made that area active; Move / extend needs a line selected.
// Click the swatch, not the row: the row's label input stops propagation, and Playwright
// clicks an element's centre, which lands on that input.
await page.locator(".region[data-geometry='line'] .swatch").first().click();
await page.waitForSelector(".region.active[data-geometry='line']", { timeout: 10000 });
await page.getByRole("button", { name: "Move / extend" }).click();
const before = (await polylines())[0];
const moveBox = await canvas.boundingBox();
const grab = toClient(moveBox, [(before[0][0] + before[1][0]) / 2, (before[0][1] + before[1][1]) / 2]);
const shift = [90, 60];
await page.mouse.move(grab[0], grab[1]);
await page.mouse.down();
await page.mouse.move(grab[0] + shift[0], grab[1] + shift[1], { steps: 12 });
await page.mouse.up();
await page.waitForTimeout(1200);

const moved = (await polylines())[0];
const expected = [(shift[0] / moveBox.width) * 1200, (shift[1] / moveBox.height) * 700];
check(`dragging a line moves the whole line (${before.length} points)`,
  moved.length === before.length &&
  moved.every((p, i) => same(p[0] - before[i][0], expected[0]) && same(p[1] - before[i][1], expected[1])));
await page.screenshot({ path: `${OUT}/12c-moved.png` });

const endBox = await canvas.boundingBox();
const tail = moved[moved.length - 1];
const startAt = toClient(endBox, tail);
await page.mouse.move(startAt[0], startAt[1]);
await page.mouse.down();
await page.mouse.move(startAt[0] + 70, startAt[1] + 55, { steps: 12 });
await page.mouse.up();
await page.waitForTimeout(1200);

const extended = (await polylines())[0];
const head = moved[moved.length - 2];
const was = Math.hypot(tail[0] - head[0], tail[1] - head[1]);
const now = Math.hypot(extended[extended.length - 1][0] - head[0], extended[extended.length - 1][1] - head[1]);
const cross = Math.abs((tail[0] - head[0]) * (extended[extended.length - 1][1] - head[1]) -
                       (tail[1] - head[1]) * (extended[extended.length - 1][0] - head[0])) / (was || 1);
check(`dragging an end lengthens the run (${was.toFixed(0)}px -> ${now.toFixed(0)}px)`, now > was + 10);
check("the rest of the line stays put",
  extended.slice(0, -1).every((p, i) => same(p[0], moved[i][0], 2) && same(p[1], moved[i][1], 2)));
check(`extending keeps the line's angle (${cross.toFixed(1)}px off axis)`, cross < 6);
await page.screenshot({ path: `${OUT}/12d-extended.png` });

// Undo has to cover the new operations, not only brush strokes and vertex edits.
await page.keyboard.press("Meta+z");
await page.waitForTimeout(1500);
const undone = (await polylines())[0];
check("undo reverses an extend",
  undone.length === moved.length &&
  undone.every((p, i) => same(p[0], moved[i][0], 3) && same(p[1], moved[i][1], 3)));
await page.keyboard.press("Meta+z");
await page.waitForTimeout(1500);
const undoneTwice = (await polylines())[0];
check("a second undo reverses the move",
  undoneTwice.every((p, i) => same(p[0], before[i][0], 3) && same(p[1], before[i][1], 3)));
await page.screenshot({ path: `${OUT}/12e-undone.png` });

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
