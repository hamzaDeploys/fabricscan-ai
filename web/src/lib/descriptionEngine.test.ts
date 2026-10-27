import { describe as suite, expect, it } from "vitest";

import { DEFECTS, severityFor } from "./defectKnowledge";
import { describe, rankClasses, reportToText } from "./descriptionEngine";
import type { Localization } from "./localization";

const CLASSES = ["good", "hole", "stain", "broken_thread"];
const NAMES = Object.fromEntries(CLASSES.map((c) => [c, DEFECTS[c as keyof typeof DEFECTS].label]));

function loc(zone: Localization["zone"], areaPct: number): Localization {
  return {
    size: 4,
    heat: new Float32Array(16),
    mask: new Uint8Array(16),
    centroid: { x: 0.2, y: 0.2 },
    zone,
    zoneLabel: zone === "center" ? "center of the fabric" : `${zone} region`,
    bbox: { x0: 0, y0: 0, x1: 0.3, y1: 0.3 },
    areaPct,
    method: "cam+contrast",
  };
}

suite("severityFor", () => {
  it("never rates a hole below Medium", () => {
    expect(severityFor("hole", 0.05)).toBe("Medium");
    expect(severityFor("hole", 2)).toBe("High");
  });
  it("grades surface defects by area", () => {
    expect(severityFor("stain", 1)).toBe("Low");
    expect(severityFor("stain", 3)).toBe("Medium");
    expect(severityFor("stain", 10)).toBe("High");
  });
  it("returns None for good fabric", () => {
    expect(severityFor("good", 50)).toBe("None");
  });
});

suite("describe", () => {
  it("writes the full sentence for a defect", () => {
    const r = describe({ probs: [0.02, 0.93, 0.02, 0.01, 0.01, 0.01], classNames: CLASSES, displayNames: NAMES, localization: loc("upper-left", 4.1), modelName: "MobileNetV2" });
    expect(r.classKey).toBe("hole");
    expect(r.severity).toBe("High");
    expect(r.lowConfidence).toBe(false);
    expect(r.summary).toBe(
      "A hole-type defect was detected near the upper-left region, covering about 4.1% of the visible fabric area. " +
        "Severity: High. Likely cause: needle damage or snagging. " +
        "Recommended action: reject the piece or cut out the damaged section; holes weaken the fabric.",
    );
    expect(r.confidenceNote).toBe("Model confidence: 93.0%.");
  });

  it("says 'center of the fabric' for the middle zone", () => {
    const r = describe({ probs: [0, 0, 0.9, 0.05, 0.05, 0], classNames: CLASSES, displayNames: NAMES, localization: loc("center", 2), modelName: "m" });
    expect(r.summary).toContain("near the center of the fabric");
    expect(r.severity).toBe("Medium");
  });

  it("reports good fabric without location or causes", () => {
    const r = describe({ probs: [0.97, 0.01, 0.01, 0.005, 0.003, 0.002], classNames: CLASSES, displayNames: NAMES, localization: loc("upper-left", 3), modelName: "m" });
    expect(r.isDefect).toBe(false);
    expect(r.severity).toBe("None");
    expect(r.areaPct).toBeNull();
    expect(r.causes).toEqual([]);
    expect(r.summary).toMatch(/^No defect was detected/);
  });

  it("flags low confidence and names the runner-up", () => {
    const r = describe({ probs: [0.05, 0.45, 0.38, 0.04, 0.04, 0.04], classNames: CLASSES, displayNames: NAMES, localization: loc("lower-right", 1), modelName: "m" });
    expect(r.lowConfidence).toBe(true);
    expect(r.confidenceNote).toContain("Stain / Oil Spot 38.0%");
    expect(r.confidenceNote).toContain("Manual inspection is recommended");
  });

  it("ranks classes by probability and produces a text report", () => {
    const ranked = rankClasses([0.1, 0.2, 0.7, 0, 0, 0], CLASSES, NAMES);
    expect(ranked.map((r) => r.key).slice(0, 3)).toEqual(["stain", "hole", "good"]);
    const r = describe({ probs: [0.1, 0.2, 0.7, 0, 0, 0], classNames: CLASSES, displayNames: NAMES, localization: loc("upper-right", 0.05), modelName: "Custom CNN", demo: true, now: new Date(2026, 9, 26, 14, 5) });
    const text = reportToText(r);
    expect(text).toContain("Date: 2026-10-26 14:05");
    expect(text).toContain("Affected area ≈ less than 0.1%");
    expect(text).toContain("DEMO model");
  });

  it("has knowledge for every class", () => {
    for (const c of CLASSES) {
      const info = DEFECTS[c as keyof typeof DEFECTS];
      expect(info.label.length).toBeGreaterThan(0);
      if (c !== "good") {
        expect(info.causes.length).toBeGreaterThanOrEqual(3);
        expect(info.causesShort.length).toBeGreaterThanOrEqual(1);
      }
    }
  });
});
