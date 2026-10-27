/** Data-driven explanation of which model is better and why (no hard-coded numbers). */
import type { MetricsFile, ModelEntry } from "../types";
import { fmtMB, fmtMs } from "./format";

export interface Comparison {
  winner: ModelEntry;
  other: ModelEntry | null;
  headline: string;
  points: string[];
}

const pts = (a: number, b: number) => `${Math.abs((a - b) * 100).toFixed(1)} points`;

function gap(m: ModelEntry): number | null {
  const tr = m.history.accuracy;
  const va = m.history.val_accuracy;
  if (!tr?.length || !va?.length) return null;
  return tr[tr.length - 1] - va[va.length - 1];
}

export function compareModels(metrics: MetricsFile): Comparison {
  const winner = metrics.models[metrics.best_model];
  const otherKey = Object.keys(metrics.models).find((k) => k !== metrics.best_model);
  const other = otherKey ? metrics.models[otherKey] : null;
  if (!other) {
    return {
      winner,
      other: null,
      headline: `${winner.name} reaches ${(winner.test.accuracy * 100).toFixed(1)}% test accuracy.`,
      points: [],
    };
  }

  const w = winner.test;
  const o = other.test;
  const headline =
    `${winner.name} is the better model on this dataset: ${(w.accuracy * 100).toFixed(1)}% test accuracy and ` +
    `${(w.f1_macro * 100).toFixed(1)}% macro F1, versus ${(o.accuracy * 100).toFixed(1)}% and ` +
    `${(o.f1_macro * 100).toFixed(1)}% for the ${other.name}.`;

  const points: string[] = [];
  const dAcc = w.accuracy - o.accuracy;
  points.push(
    Math.abs(dAcc) < 0.01
      ? "Both models reach almost the same accuracy (less than 1 point apart); the choice is decided by macro F1, which weighs every class equally."
      : `Accuracy: ${dAcc > 0 ? "+" : "−"}${pts(w.accuracy, o.accuracy)} for ${winner.name}; macro F1 differs by ${pts(w.f1_macro, o.f1_macro)}.`,
  );

  const sizeRatio = winner.onnx_size_mb / other.onnx_size_mb;
  points.push(
    `Size: ${fmtMB(winner.onnx_size_mb)} vs ${fmtMB(other.onnx_size_mb)} (${sizeRatio >= 1 ? `${sizeRatio.toFixed(1)}× larger` : `${(1 / sizeRatio).toFixed(1)}× smaller`}). ` +
      `Speed: ${fmtMs(winner.onnx_latency_ms)} vs ${fmtMs(other.onnx_latency_ms)} per image (ONNX Runtime, CPU).`,
  );

  const gw = gap(winner);
  const go = gap(other);
  if (gw !== null && go !== null) {
    points.push(
      `Generalisation gap (final train − validation accuracy): ${(gw * 100).toFixed(1)} points for ${winner.name}, ` +
        `${(go * 100).toFixed(1)} points for the ${other.name}${Math.max(gw, go) > 0.1 ? "; the larger gap indicates overfitting" : ""}.`,
    );
  }

  if (winner.key === "transfer") {
    points.push(
      "Why: MobileNetV2 starts from features learned on 1.2 million ImageNet photos (edges, textures, shapes), so a few hundred fabric photos are enough to adapt it. The scratch CNN has to learn those low-level features from the small fabric dataset alone, which needs many more epochs and overfits sooner.",
    );
  } else {
    points.push(
      "Why: the custom CNN keeps a finer 14×14 feature map, which helps with small defects, and the dataset was consistent enough to learn from scratch. Transfer learning would be expected to win on smaller or more varied datasets.",
    );
  }
  return { winner, other, headline, points };
}
