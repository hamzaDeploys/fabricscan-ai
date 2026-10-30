import { expect, test } from "@playwright/test";

// Run against the deployed site with playwright.live.config.ts. Logs how long each step takes on this connection.
test("deployed site loads the model and analyses a photo", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(err.message));
  page.on("console", (msg) => msg.type() === "error" && errors.push(msg.text()));
  const start = Date.now();
  const seconds = () => ((Date.now() - start) / 1000).toFixed(1);

  await page.goto("/");
  console.log(`page shown after ${seconds()} s, crossOriginIsolated=${await page.evaluate(() => self.crossOriginIsolated)}`);
  await expect(page.getByTestId("model-status")).toContainText("ready");
  console.log(`model ready after ${seconds()} s`);

  await page.getByRole("button", { name: /Analyse sample image: Hole/ }).first().click();
  await expect(page.getByTestId("predicted-label")).toContainText("Hole");
  console.log(`sample analysed after ${seconds()} s`);

  await page.getByRole("button", { name: "New image" }).click();
  await page.getByTestId("file-input").setInputFiles("public/samples/stain-1.jpg");
  await expect(page.getByTestId("predicted-label")).toContainText("Stain");
  console.log(`uploaded photo analysed after ${seconds()} s`);

  await page.getByTestId("model-select").selectOption({ label: "Custom CNN (from scratch)" });
  await expect(page.getByTestId("result-model")).toContainText("Custom CNN");
  await expect(page.getByTestId("model-status")).toContainText("ready");
  console.log(`second model answered after ${seconds()} s`);
  expect(errors, errors.join("\n")).toHaveLength(0);
});
