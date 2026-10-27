import { describe, expect, it } from "vitest";

import type { MetricsFile, ModelEntry } from "../types";
import { compareModels } from "./compare";
import { areaText, compact, joinOr } from "./format";
import { meanColor, occlude } from "./occlusion";
import { cropBox, rgbaToRgb } from "./preprocess";

describe("preprocess helpers", () => {
  it("crops the centred square", () => {
    expect(cropBox(4000, 3000)).toEqual({ sx: 500, sy: 0, side: 3000 });
    expect(cropBox(300, 500)).toEqual({ sx: 0, sy: 100, side: 300 });
  });
  it("drops the alpha channel", () => {
    expect(Array.from(rgbaToRgb([1, 2, 3, 255, 4, 5, 6, 255]))).toEqual([1, 2, 3, 4, 5, 6]);
  });
});

describe("occlusion helpers", () => {
  it("fills only the patch with the mean colour", () => {
    const size = 4;
    const px = new Float32Array(size * size * 3).fill(100);
    px[0] = 200;
    expect(meanColor(px)[0]).toBeCloseTo((200 + 15 * 100) / 16);
    const out = occlude(px, size, 1, 1, 2, [0, 0, 0]);
    expect(out[0]).toBe(0); // (0,0) inside the patch
    expect(out[(3 * size + 3) * 3]).toBe(100); // (3,3) outside
    expect(px[0]).toBe(200); // input untouched
  });
});

describe("format", () => {
  it("formats areas, counts and lists", () => {
    expect(areaText(0.05)).toBe("less than 0.1%");
    expect(areaText(4.14)).toBe("4.1%");
    expect(areaText(23.6)).toBe("24%");
    expect(compact(2_265_670)).toBe("2.27 M");
    expect(joinOr(["a", "b", "c"])).toBe("a, b or c");
  });
});

function entry(key: string, acc: number, f1: number, size: number): ModelEntry {
  return {
    key,
    name: key === "transfer" ? "MobileNetV2 (transfer learning)" : "Custom CNN (from scratch)",
    architecture: "",
    params: 1,
    keras_size_mb: size,
    onnx_size_mb: size,
    keras_latency_ms: 10,
    onnx_latency_ms: 5,
    epochs: 10,
    train_time_s: 100,
    fine_tune_start_epoch: null,
    best_val_accuracy: acc,
    history: { accuracy: [0.5, 0.95], val_accuracy: [0.5, acc] },
    test: {
      n: 100, accuracy: acc, precision_macro: f1, recall_macro: f1, f1_macro: f1,
      precision_weighted: f1, recall_weighted: f1, f1_weighted: f1, per_class: {}, confusion_matrix: [],
    },
  };
}

describe("compareModels", () => {
  const base = { demo: false, generated: "", class_names: [], display_names: {}, images: {}, selection_rule: "" };
  it("explains a transfer-learning win with the real numbers", () => {
    const m: MetricsFile = { ...base, best_model: "transfer", models: { scratch: entry("scratch", 0.8, 0.78, 4.7), transfer: entry("transfer", 0.92, 0.91, 8.9) } };
    const c = compareModels(m);
    expect(c.winner.key).toBe("transfer");
    expect(c.headline).toContain("92.0% test accuracy");
    expect(c.points.join(" ")).toContain("+12.0 points");
    expect(c.points.join(" ")).toContain("1.9× larger");
    expect(c.points.at(-1)).toContain("ImageNet");
  });
  it("adapts the explanation when the scratch CNN wins", () => {
    const m: MetricsFile = { ...base, best_model: "scratch", models: { scratch: entry("scratch", 0.9, 0.9, 4.7), transfer: entry("transfer", 0.85, 0.84, 8.9) } };
    expect(compareModels(m).points.at(-1)).toContain("custom CNN");
  });
});
