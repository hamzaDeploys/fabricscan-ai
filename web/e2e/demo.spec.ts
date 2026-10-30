import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));
const SHOTS = process.env.SCREENSHOT_DIR ?? here("./screenshots");

test.beforeEach(async ({ page }) => {
  page.on("pageerror", (err) => {
    throw new Error(`Uncaught error in the page: ${err.message}`);
  });
  await page.goto("/");
});

test("page loads with SEO basics and the model becomes ready", async ({ page }) => {
  await expect(page).toHaveTitle(/FabricScan AI/);
  await expect(page.locator('meta[name="description"]')).toHaveAttribute("content", /fabric/i);
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await expect(page.getByTestId("model-status")).toContainText("ready", { timeout: 60_000 });
});

test("browser inference matches ONNX Runtime in Python", async ({ page }) => {
  const fixture = JSON.parse(readFileSync(here("../src/test/fixtures/parity.json"), "utf-8"));
  await page.waitForFunction(() => Boolean(window.__fabricscan));
  const probs: number[] = await page.evaluate((b64) => window.__fabricscan!.runPixelsBase64(b64), fixture.image_uint8_base64);
  const maxDiff = Math.max(...probs.map((p, i) => Math.abs(p - fixture.probs[i])));
  expect(probs).toHaveLength(fixture.probs.length);
  expect(maxDiff).toBeLessThan(1e-4);
});

test("analysing a sample produces a full report", async ({ page }, info) => {
  await page.locator("#demo").scrollIntoViewIfNeeded();
  await page.getByRole("button", { name: /Analyse sample image: Hole/ }).first().click();
  const result = page.getByTestId("result");
  await expect(result).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("predicted-label")).not.toBeEmpty();
  await expect(page.getByTestId("summary")).toContainText(/Recommended action/);
  await expect(page.getByTestId("confidence-bars").locator("li")).toHaveCount(4);
  await expect(page.getByTestId("severity")).toBeVisible();
  await page.waitForTimeout(900); // let the bar and ring animations finish
  await page.locator("#demo").screenshot({ path: `${SHOTS}/${info.project.name}-demo.png` });
  await page.getByRole("button", { name: /Switch to (dark|light) mode/ }).click();
  await page.waitForTimeout(400);
  await page.locator("#demo").screenshot({ path: `${SHOTS}/${info.project.name}-demo-alt-theme.png` });
});

test("every sample image is recognised as its own class", async ({ page }) => {
  const samples: { label: string }[] = JSON.parse(readFileSync(here("../public/samples/samples.json"), "utf-8"));
  await page.locator("#demo").scrollIntoViewIfNeeded();
  await expect(page.getByTestId("model-status")).toContainText("ready", { timeout: 60_000 });
  const seen: Record<string, number> = {};
  for (const s of samples) {
    seen[s.label] = (seen[s.label] ?? -1) + 1;
    await page.getByRole("button", { name: `Analyse sample image: ${s.label}` }).nth(seen[s.label]).click();
    await expect(page.getByTestId("predicted-label")).toContainText(s.label, { timeout: 60_000 });
  }
});

test("the model dropdown switches model and checks the same photo again", async ({ page }) => {
  const select = page.getByTestId("model-select");
  await expect(select.locator("option")).toHaveCount(2);
  await page.getByRole("button", { name: /Analyse sample image: Hole/ }).first().click();
  await expect(page.getByTestId("result-model")).toContainText("MobileNetV2", { timeout: 60_000 });
  await expect(page.getByTestId("model-facts")).toContainText("81.3%");

  await select.selectOption({ label: "Custom CNN (from scratch)" });
  await expect(page.getByTestId("result-model")).toContainText("Custom CNN", { timeout: 60_000 });
  await expect(page.getByTestId("predicted-label")).toContainText("Hole");
  await expect(page.getByTestId("model-facts")).toContainText("68.4%");
  await expect(page.getByTestId("model-status")).toContainText("Custom CNN");
  await expect(page.getByTestId("model-status")).toContainText("ready");

  await select.selectOption({ label: "MobileNetV2 (transfer learning)" });
  await expect(page.getByTestId("result-model")).toContainText("MobileNetV2");
});

test("report can be copied and downloaded as PNG and PDF", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.getByRole("button", { name: /Analyse sample image: Stain/ }).first().click();
  await expect(page.getByTestId("result")).toBeVisible({ timeout: 60_000 });

  await page.getByRole("button", { name: "Copy report" }).click();
  await expect(page.getByText("Report copied")).toBeVisible();
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  expect(copied).toContain("FabricScan AI · Inspection report");

  const [png] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "PNG" }).click()]);
  expect(png.suggestedFilename()).toMatch(/^fabricscan-.+\.png$/);
  const [pdf] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "PDF" }).click()]);
  expect(pdf.suggestedFilename()).toMatch(/^fabricscan-.+\.pdf$/);
  await png.saveAs(`${SHOTS}/report-card.png`);
  await pdf.saveAs(`${SHOTS}/report.pdf`);
});

test("uploading a photo works and a non-image is rejected", async ({ page }) => {
  await page.getByTestId("file-input").setInputFiles(here("../public/samples/stain-1.jpg"));
  await expect(page.getByTestId("result")).toBeVisible({ timeout: 60_000 });

  await page.getByRole("button", { name: "New image" }).click();
  await page.getByTestId("file-input").setInputFiles(here("../public/samples/samples.json"));
  await expect(page.getByRole("alert")).toContainText("is not an image");
});

test("about section shows the GitHub, LinkedIn and YouTube buttons", async ({ page }, info) => {
  const about = page.locator("#about");
  await about.scrollIntoViewIfNeeded();
  const socials = page.getByTestId("social-links").locator("li");
  await expect(socials).toHaveCount(3);
  for (const name of ["GitHub", "LinkedIn", "YouTube"]) await expect(socials.filter({ hasText: name })).toHaveCount(1);
  await page.waitForTimeout(700); // fade-in
  await about.screenshot({ path: `${SHOTS}/${info.project.name}-about.png` });
});

test("the video plays in a popup and closes again", async ({ page }, info) => {
  const link = page.getByTestId("link-video");
  const dialog = page.getByTestId("video-dialog");
  const player = page.getByTestId("video-player");
  await link.scrollIntoViewIfNeeded();
  await expect(player).toHaveCount(0); // nothing is loaded before the click

  await link.click();
  await expect(dialog).toBeVisible();
  await expect.poll(() => player.evaluate((v: HTMLVideoElement) => v.duration), { timeout: 20_000 }).toBeGreaterThan(300);
  await expect.poll(() => player.evaluate((v: HTMLVideoElement) => v.currentTime), { timeout: 20_000 }).toBeGreaterThan(0);
  await page.screenshot({ path: `${SHOTS}/${info.project.name}-video.png` });

  await page.getByRole("button", { name: "Close video" }).click();
  await expect(dialog).toBeHidden();
  await expect(player).toHaveCount(0); // closing stops the sound

  await link.click(); // Esc, then the dark area around the player
  await expect(dialog).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await link.click();
  await expect(dialog).toBeVisible();
  await page.mouse.click(4, 4);
  await expect(dialog).toBeHidden();
});

test("project links, dataset download and links readable without JavaScript", async ({ page, request }, info) => {
  const section = page.locator("#dataset");
  await section.scrollIntoViewIfNeeded();
  await expect(page.getByTestId("dataset-download")).toContainText("MB");
  await page.waitForTimeout(900); // fade-in
  await section.screenshot({ path: `${SHOTS}/${info.project.name}-dataset.png` });
  await expect(page.getByTestId("project-links").locator(":scope > li")).toHaveCount(6);
  for (const id of ["dataset-download", "link-dataset", "link-weights"]) {
    const href = await page.getByTestId(id).getAttribute("href");
    const res = await request.head(href!);
    expect(res.ok(), `${id} -> ${href}`).toBeTruthy();
  }
  const html = await (await request.get("/")).text();
  expect(html).toContain('id="project-links"');
  expect(html).toContain("/downloads/fabricscan-dataset.zip");
  const project = await (await request.get("/project.json")).json();
  expect(project.dataset).toBe("/downloads/fabricscan-dataset.zip");
  expect(project.video).toBe("/video/fabricscan-presentation.mp4");
});

test("nothing overflows the screen horizontally", async ({ page }) => {
  await page.getByTestId("model-status").waitFor();
  await revealAll(page);
  const offenders = await page.evaluate(() => {
    const vw = document.documentElement.clientWidth;
    return Array.from(document.querySelectorAll("body *"))
      .filter((el) => {
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.right > vw + 1 && !el.closest(".overflow-x-auto, .overflow-hidden");
      })
      .slice(0, 8)
      .map((el) => `${el.tagName.toLowerCase()}.${String(el.className).slice(0, 60)} right=${Math.round(el.getBoundingClientRect().right)} vw=${vw}`);
  });
  const extra = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(offenders, offenders.join("\n")).toEqual([]);
  expect(extra).toBeLessThanOrEqual(0);
});

/** Scroll through the page so every "fade in on scroll" section has appeared before a full-page screenshot. */
async function revealAll(page: import("@playwright/test").Page) {
  const height = await page.evaluate(() => document.body.scrollHeight);
  for (let y = 0; y < height; y += 500) {
    await page.evaluate((top) => window.scrollTo(0, top), y);
    await page.waitForTimeout(120);
  }
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(500);
}

test("dark mode toggles and full-page screenshots", async ({ page }, info) => {
  await page.getByTestId("model-status").waitFor();
  await revealAll(page);
  await page.screenshot({ path: `${SHOTS}/${info.project.name}-light.png`, fullPage: true });
  await page.getByRole("button", { name: /Switch to dark mode/ }).click();
  await expect(page.locator("html")).toHaveClass(/dark/);
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${SHOTS}/${info.project.name}-dark.png`, fullPage: true });
});
