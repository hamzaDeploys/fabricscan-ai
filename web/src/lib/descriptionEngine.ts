/**
 * Template-based automatic defect description.
 *
 * Inputs: the class probabilities, the predicted class, and the localisation (zone + area). Output: a structured
 * report plus a one-paragraph summary such as:
 *   "A hole-type defect was detected near the upper-left region, covering about 4.1% of the visible fabric area.
 *    Severity: High. Likely cause: needle damage or snagging. Recommended action: reject the piece ..."
 */

import { DEFECTS, type ClassKey, type Severity, isClassKey, severityFor } from "./defectKnowledge";
import { areaText, capitalize, formatDateTime, joinOr, pct } from "./format";
import type { Localization } from "./localization";

/** Below this top probability, or when the top two are closer than LOW_MARGIN, the report asks for a manual check. */
export const LOW_CONFIDENCE = 0.6;
export const LOW_MARGIN = 0.2;

export interface Ranked {
  key: string;
  label: string;
  prob: number;
}

export interface DefectReport {
  classKey: ClassKey;
  label: string;
  isDefect: boolean;
  confidence: number;
  severity: Severity;
  areaPct: number | null;
  zone: string | null;
  zoneLabel: string | null;
  summary: string;
  confidenceNote: string;
  lowConfidence: boolean;
  causes: string[];
  action: string;
  ranked: Ranked[];
  method: string | null;
  modelName: string;
  demo: boolean;
  createdAt: string;
}

export interface DescribeInput {
  probs: ArrayLike<number>;
  classNames: string[];
  displayNames: Record<string, string>;
  localization: Localization | null;
  modelName: string;
  demo?: boolean;
  now?: Date;
}

export function rankClasses(probs: ArrayLike<number>, classNames: string[], displayNames: Record<string, string>): Ranked[] {
  return classNames
    .map((key, i) => ({ key, label: displayNames[key] ?? key, prob: probs[i] }))
    .sort((a, b) => b.prob - a.prob);
}

function sentence(text: string): string {
  const t = text.trim();
  return /[.!?]$/.test(t) ? t : `${t}.`;
}

export function describe({ probs, classNames, displayNames, localization, modelName, demo = false, now = new Date() }: DescribeInput): DefectReport {
  const ranked = rankClasses(probs, classNames, displayNames);
  const [top, second] = ranked;
  if (!isClassKey(top.key)) throw new Error(`Unknown class "${top.key}"`);
  const info = DEFECTS[top.key];
  const isDefect = top.key !== "good";
  const lowConfidence = top.prob < LOW_CONFIDENCE || (second !== undefined && top.prob - second.prob < LOW_MARGIN);
  const loc = isDefect ? localization : null;
  const areaPct = loc ? loc.areaPct : null;
  const severity: Severity = isDefect ? severityFor(top.key, areaPct ?? 0) : "None";
  const action = info.actions[severity];

  let summary: string;
  if (isDefect) {
    const where = loc ? (loc.zone === "center" ? "near the center of the fabric" : `near the ${loc.zoneLabel}`) : "in the image";
    const covering = loc ? `, covering about ${areaText(loc.areaPct)} of the visible fabric area` : "";
    summary =
      `${info.subject} was detected ${where}${covering}. Severity: ${severity}. ` +
      `Likely cause: ${joinOr(info.causesShort)}. Recommended action: ${sentence(action)}`;
  } else {
    summary = `No defect was detected; the fabric surface looks uniform. Recommended action: ${sentence(action)}`;
  }

  let confidenceNote: string;
  if (!lowConfidence) {
    confidenceNote = `Model confidence: ${pct(top.prob)}.`;
  } else if (second) {
    confidenceNote =
      `The model is not certain: ${top.label} ${pct(top.prob)} vs ${second.label} ${pct(second.prob)}. ` +
      "Manual inspection is recommended.";
  } else {
    confidenceNote = `Low confidence (${pct(top.prob)}). Manual inspection is recommended.`;
  }

  return {
    classKey: top.key,
    label: info.label,
    isDefect,
    confidence: top.prob,
    severity,
    areaPct,
    zone: loc?.zone ?? null,
    zoneLabel: loc?.zoneLabel ?? null,
    summary,
    confidenceNote,
    lowConfidence,
    causes: info.causes,
    action: capitalize(sentence(action)),
    ranked,
    method: loc?.method ?? null,
    modelName,
    demo,
    createdAt: formatDateTime(now),
  };
}

/** Plain-text version for "Copy report". */
export function reportToText(r: DefectReport): string {
  const lines = [
    "FabricScan AI · Inspection report",
    `Date: ${r.createdAt}`,
    `Result: ${r.label} (confidence ${pct(r.confidence)})`,
    `Severity: ${r.severity}`,
  ];
  if (r.isDefect && r.zoneLabel !== null && r.areaPct !== null) {
    lines.push(`Location: ${r.zoneLabel} · Affected area ≈ ${areaText(r.areaPct)}`);
  }
  lines.push("", `Summary: ${r.summary}`, r.confidenceNote);
  if (r.causes.length) lines.push("", "Possible causes:", ...r.causes.map((c) => `  - ${c}`));
  lines.push("", `Recommended action: ${r.action}`, "", "All classes:");
  lines.push(...r.ranked.map((c) => `  ${c.label.padEnd(26)} ${pct(c.prob)}`));
  lines.push("", `Model: ${r.modelName}${r.demo ? " (DEMO model, synthetic training data)" : ""}`);
  return lines.join("\n");
}
