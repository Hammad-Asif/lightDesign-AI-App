// The product library screen, in a real browser: grouping, the per-category forms, the API's
// own refusals shown inline, duplicating a built-in, retiring a product, and the one thing
// that is easy to get wrong -- the confirmation screen seeing a new product without a reload.
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const OUT = process.env.SHOT_DIR ?? "./test-output";
const API = process.env.API_URL ?? "http://127.0.0.1:8000";
const UI = process.env.UI_URL ?? "http://127.0.0.1:5173/";
mkdirSync(OUT, { recursive: true });
const errors = [];
let failed = false;

function check(label, ok) {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
  if (!ok) failed = true;
}

// Products sits in the account menu, which has to be opened first.
async function openProducts(page) {
  await page.locator(".avatar-trigger").click();
  await page.getByTestId("open-products").click();
  await page.waitForSelector("[data-testid=products-screen]", { timeout: 10000 });
}

const health = await fetch(`${API}/health`).then((r) => r.json());
if (health.segmentation_provider !== "mock") {
  console.error(`Run the API with SEGMENTATION_PROVIDER=mock; it is "${health.segmentation_provider}".`);
  process.exit(2);
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(String(e)));

await page.goto(UI, { waitUntil: "networkidle" });
await page.getByRole("button", { name: "New here? Create an account" }).click();
await page.getByLabel("Name").fill("Products UI");
await page.getByLabel("Email").fill(`products-ui-${Date.now()}@example.com`);
await page.getByLabel("Password").fill("correct horse battery");
await page.getByRole("button", { name: "Create account" }).click();
await page.waitForSelector("h1:has-text('New mock-up')", { timeout: 15000 });

// ---- the library, grouped -------------------------------------------------------------
await openProducts(page);
await page.waitForSelector(".product-group-head", { timeout: 10000 });
check("the products screen opens", await page.getByTestId("products-screen").isVisible());
for (const group of ["Bulb lines", "Strands", "Accents"]) {
  check(`${group} is a group`, await page.locator(`.product-group-head h2:text-is("${group}")`).isVisible());
}
const seeded = await page.locator(".product").count();
check(`the seeded catalogue is listed (${seeded} products)`, seeded > 0);
check("seeded products are marked built in", (await page.locator(".product .badge").count()) > 0);
check("a built-in offers duplication, not editing",
  (await page.locator("[data-testid^=duplicate-]").count()) > 0 &&
  (await page.locator("[data-testid^=edit-]").count()) === 0);
await page.screenshot({ path: `${OUT}/p1-library.png`, fullPage: true });

// ---- the add form shows only what the category needs ----------------------------------
await page.getByTestId("add-c9").click();
check("the form says what it is doing",
  (await page.locator(".product-form-title").innerText()) === "New C9");
check("a bulb line asks for spacings as chips", await page.getByTestId("spacing-12").isVisible());
check("a bulb line does not ask for colours", (await page.locator("[data-testid^=colour-]").count()) === 0);
check("a bulb line says why it has no colours",
  await page.locator(".product-form p.muted").first().innerText()
    .then((text) => text.includes("any bulb colour")));

// Switching the kind switches the fields.
await page.getByTestId("product-category").selectOption("mini");
check("a strand asks for colours", await page.getByTestId("colour-warm_white").isVisible());
check("a strand does not offer socket spacings", (await page.locator("[data-testid^=spacing-]").count()) === 0);
await page.getByTestId("product-category").selectOption("wreath");
check("a wreath asks for neither bulb size nor spacing",
  !(await page.locator(".product-form").getByText("Bulb size (in)").isVisible().catch(() => false)));
await page.getByTestId("product-category").selectOption("c9");

// ---- the API's own refusals, shown inline ---------------------------------------------
await page.getByTestId("product-name").fill("Incomplete C9");
await page.getByTestId("product-save").click();
await page.waitForSelector("[data-testid=product-problems]", { timeout: 10000 });
const problems = await page.getByTestId("product-problems").innerText();
check(`a product missing its fields is refused in words (${problems.trim()})`,
  /bulb size/.test(problems) && /spacing/.test(problems)
  && !/422|Value error|Unprocessable|bulb_diameter_in/i.test(problems));
await page.screenshot({ path: `${OUT}/p2-problems.png`, fullPage: true });

// ---- adding one properly ---------------------------------------------------------------
await page.getByTestId("product-name").fill("Lukas C9");
await page.getByTestId("spacing-12").click();
await page.getByTestId("spacing-18").click();
await page.getByTestId("product-bulb").fill("1.125");
await page.getByTestId("product-save").click();
await page.waitForSelector("[data-testid=products-screen] .product-form", { state: "detached", timeout: 10000 });
const added = page.locator('.product:has-text("Lukas C9")');
await added.waitFor({ timeout: 10000 });
check("the new product is in the library", await added.isVisible());
check("the new product shows its spacings",
  (await page.locator('.product:has-text("Lukas C9")').innerText()).includes("12, 18 in spacing"));
check("an own product can be edited and removed",
  (await page.locator('.product:has-text("Lukas C9") [data-testid^=edit-]').count()) === 1 &&
  (await page.locator('.product:has-text("Lukas C9") [data-testid^=delete-]').count()) === 1);

// ---- editing it ------------------------------------------------------------------------
await page.locator('.product:has-text("Lukas C9") [data-testid^=edit-]').click();
await page.getByTestId("product-name").fill("Lukas C9 LED");
await page.getByTestId("product-save").click();
await page.waitForSelector("[data-testid=products-screen] .product-form", { state: "detached", timeout: 10000 });
const edited = page.locator('.product:has-text("Lukas C9 LED")');
await edited.waitFor({ timeout: 10000 });
check("an edit is kept", await edited.isVisible());

// ---- duplicating a built-in ------------------------------------------------------------
const builtinName = await page.locator(".product:has(.badge)").first().locator(".product-name").innerText();
await page.locator(".product:has(.badge)").first().locator("[data-testid^=duplicate-]").click();
const copyName = await page.getByTestId("product-name").inputValue();
check(`duplicating a built-in opens an editable copy (${copyName})`, copyName === `${builtinName} (copy)`);
await page.getByTestId("product-save").click();
await page.waitForSelector("[data-testid=products-screen] .product-form", { state: "detached", timeout: 10000 });
const copy = page.locator(`.product:has-text("${builtinName} (copy)")`);
await copy.waitFor({ timeout: 10000 });
check("the copy is the user's own, not built in",
  (await copy.locator(".badge").count()) === 0 && (await copy.locator("[data-testid^=edit-]").count()) === 1);
await page.screenshot({ path: `${OUT}/p3-duplicated.png`, fullPage: true });

// ---- finding one in a long library -----------------------------------------------------
const all = await page.locator(".product").count();
await page.getByTestId("product-search").fill("icicle");
await page.waitForFunction((n) => document.querySelectorAll(".product").length < n, all, { timeout: 5000 });
const found = await page.locator(".product").count();
check(`searching narrows the library (${all} -> ${found})`, found > 0 && found < all);
check("the search says how much it narrowed it",
  /of \d+/.test(await page.getByTestId("product-count").innerText()));
check("groups with no match are hidden while searching",
  (await page.locator('.product-group-head h2:text-is("Accents")').count()) === 0);
await page.getByTestId("product-search").fill("zzzz no such product");
await page.waitForSelector("[data-testid=no-matches]", { timeout: 5000 });
check("a search with no matches says so", await page.getByTestId("no-matches").isVisible());
await page.getByRole("button", { name: "Clear the search" }).click();
await page.waitForFunction((n) => document.querySelectorAll(".product").length === n, all, { timeout: 5000 });
check("clearing the search brings the library back", (await page.locator(".product").count()) === all);

// ---- a picture of the product ----------------------------------------------------------
const swatchPng = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAHElEQVQoz2P8z8Dwn4GKgIlhFIxaMGrB" +
  "sLAAAKgoAxOBmSzBAAAAAElFTkSuQmCC", "base64");

check("a product with no picture still holds the space",
  (await page.locator(".product .product-thumb.empty").count()) > 0);

await page.locator('.product:has-text("Lukas C9 LED") [data-testid^=edit-]').click();
check("the form offers a picture",
  await page.locator(".product-form .button-like").getByText("Add a picture").isVisible());
await page.setInputFiles("[data-testid=product-picture]", {
  name: "c9.png", mimeType: "image/png", buffer: swatchPng,
});
check("the chosen file is shown before it is saved",
  await page.locator(".product-form img.product-thumb").isVisible());
await page.getByTestId("product-save").click();
await page.waitForSelector("[data-testid=products-screen] .product-form", { state: "detached", timeout: 10000 });

const pictured = page.locator('.product:has-text("Lukas C9 LED")');
await pictured.locator("img.product-thumb").waitFor({ timeout: 10000 });
check("the picture is shown in the library", await pictured.locator("img.product-thumb").isVisible());
// Attached is not loaded: wait for the browser to have decoded it before asking its size.
const loaded = await pictured.locator("img.product-thumb")
  .evaluate((img) => img.complete && img.naturalWidth > 0
    ? true
    : new Promise((done) => {
        img.addEventListener("load", () => done(img.naturalWidth > 0), { once: true });
        img.addEventListener("error", () => done(false), { once: true });
      }))
  .catch(() => false);
check("the picture actually loads", loaded === true);
await page.screenshot({ path: `${OUT}/p3b-pictured.png`, fullPage: true });

// Re-opening shows the stored picture, and it can be taken off again.
await pictured.locator("[data-testid^=edit-]").click();
await page.locator(".product-form img.product-thumb").waitFor({ timeout: 10000 });
check("re-opening shows the stored picture", await page.locator(".product-form img.product-thumb").isVisible());
await page.getByTestId("product-picture-remove").click();
check("removing clears it in the form", (await page.locator(".product-form img.product-thumb").count()) === 0);
await page.getByTestId("product-save").click();
await page.waitForSelector("[data-testid=products-screen] .product-form", { state: "detached", timeout: 10000 });
await page.locator('.product:has-text("Lukas C9 LED") img.product-thumb')
  .waitFor({ state: "detached", timeout: 10000 });
check("the picture is gone from the library",
  (await page.locator('.product:has-text("Lukas C9 LED") img.product-thumb').count()) === 0);

// ---- removing one ----------------------------------------------------------------------
await copy.locator("[data-testid^=delete-]").click();
check("removing asks before it deletes for good", await copy.getByText("Remove it?").isVisible());
await copy.getByRole("button", { name: "Keep" }).click();
check("keeping leaves the product alone",
  (await page.locator(`.product:has-text("${builtinName} (copy)")`).count()) === 1);
await copy.locator("[data-testid^=delete-]").click();
await copy.locator("[data-testid^=confirm-]").click();
await page.waitForFunction(
  (name) => !document.body.innerText.includes(name), `${builtinName} (copy)`, { timeout: 10000 });
check("an unused product is removed outright",
  (await page.locator(`.product:has-text("${builtinName} (copy)")`).count()) === 0);
check("nothing was retired, so no retired section appears",
  (await page.getByTestId("toggle-retired").count()) === 0);

// ---- the confirmation screen sees the library change, with no reload -------------------
await page.getByRole("button", { name: "Done" }).click();
await page.waitForSelector("h1:has-text('New mock-up')", { timeout: 10000 });
const png = await page.evaluate(async () => {
  const canvas = document.createElement("canvas");
  canvas.width = 1200; canvas.height = 800;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#7fa6cc"; ctx.fillRect(0, 0, 1200, 800);
  ctx.fillStyle = "#8a6a52";
  ctx.beginPath(); ctx.moveTo(120, 340); ctx.lineTo(600, 150); ctx.lineTo(1080, 340); ctx.closePath(); ctx.fill();
  const blob = await new Promise((r) => canvas.toBlob(r, "image/png"));
  return [...new Uint8Array(await blob.arrayBuffer())];
});
await page.setInputFiles("input[type=file]", {
  name: "house.png", mimeType: "image/png", buffer: Buffer.from(png),
});
// "Blue mini lights" resolves to nothing in the seeded catalogue, so the reading carries a
// problem instead of a product. Adding the missing strand must clear it where it stands.
await page.getByLabel(/Where do the lights go/i).fill("blue mini lights in the hedges");
await page.getByRole("button", { name: "Continue" }).click();
await page.waitForSelector("h1:has-text('Check the instruction')", { timeout: 20000 });
const before = await page.locator(".spec-row").first().innerText();
check(`an unstocked colour is a problem, not a guess (${before.split("\n").pop()})`,
  (await page.locator(".spec-problem").count()) === 1 && !before.includes("Blue minis"));
await page.screenshot({ path: `${OUT}/p4-unresolved.png`, fullPage: true });

await openProducts(page);
await page.getByTestId("add-mini").click();
await page.getByTestId("product-name").fill("Blue minis");
await page.getByTestId("colour-blue").click();
await page.getByTestId("product-bulb").fill("0.25");
await page.getByTestId("product-length").fill("6");
await page.getByTestId("product-spacing").fill("4");
await page.getByTestId("product-save").click();
await page.waitForSelector("[data-testid=products-screen] .product-form", { state: "detached", timeout: 10000 });
await page.getByRole("button", { name: "Done" }).click();
await page.waitForSelector("h1:has-text('Check the instruction')", { timeout: 10000 });

await page.waitForFunction(
  () => document.body.innerText.includes("Blue minis"), null, { timeout: 10000 });
const after = await page.locator(".spec-row").first().innerText();
check("a product added just now reaches the confirmation screen without a reload",
  after.includes("Blue minis"));
check("and the reading's problem is gone with it",
  (await page.locator(".spec-problem").count()) === 0);
await page.screenshot({ path: `${OUT}/p5-resolved.png`, fullPage: true });

const unexpected = errors.filter((e) => !/422/.test(e));
check("no console errors beyond the refusal the suite asks for", unexpected.length === 0);
if (unexpected.length) console.log(unexpected.slice(0, 5));

await browser.close();
console.log(failed ? "\nProducts UI FAILED" : "\nAll products UI checks passed.");
process.exit(failed ? 1 : 0);
