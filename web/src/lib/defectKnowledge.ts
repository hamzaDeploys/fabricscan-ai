/**
 * Domain knowledge used by the description engine: what each class means, how severity scales with the affected
 * area, likely causes and the recommended action per severity level.
 *
 * Severity thresholds follow the spirit of the industry "4-point" fabric inspection system: openings (holes)
 * are serious even when small, while surface defects (stains, broken threads) are graded by size.
 */

export type ClassKey = "good" | "hole" | "stain" | "broken_thread";
export type Severity = "None" | "Low" | "Medium" | "High";

export interface DefectInfo {
  key: ClassKey;
  label: string;
  /** Sentence subject, e.g. "A hole-type defect". */
  subject: string;
  /** One-line definition shown in the dataset section. */
  definition: string;
  /** Upper area limits (% of image) for Low and Medium; above the second value the severity is High. */
  thresholds: [number, number];
  /** Severity never goes below this level for a detected defect of this type. */
  minSeverity: Severity;
  /** Short causes used inside the summary sentence. */
  causesShort: string[];
  /** Fuller causes listed in the report. */
  causes: string[];
  /** Recommended action per severity, written to continue "Recommended action: ...". */
  actions: Record<Severity, string>;
}

export const SEVERITY_ORDER: Severity[] = ["None", "Low", "Medium", "High"];

export const DEFECTS: Record<ClassKey, DefectInfo> = {
  good: {
    key: "good",
    label: "Good (No Defect)",
    subject: "No defect",
    definition: "Defect-free fabric: plain, printed, striped or textured, including normal folds, seams and wrinkles.",
    thresholds: [0, 0],
    minSeverity: "None",
    causesShort: [],
    causes: [],
    actions: {
      None: "pass the fabric; no action needed",
      Low: "pass the fabric; no action needed",
      Medium: "pass the fabric; no action needed",
      High: "pass the fabric; no action needed",
    },
  },
  hole: {
    key: "hole",
    label: "Hole",
    subject: "A hole-type defect",
    definition: "A compact opening through the fabric where the surface underneath shows through.",
    thresholds: [0, 1.0],
    minSeverity: "Medium",
    causesShort: ["needle damage", "snagging"],
    causes: [
      "Needle or sinker damage during knitting or sewing",
      "Snagging on sharp machine parts, hooks or jewellery",
      "Yarn breakage at a weak spot in the fabric",
      "Abrasion, burns or insect damage during storage",
    ],
    actions: {
      None: "no action needed",
      Low: "mark the spot and repair it before dispatch",
      Medium: "repair (darn) the hole, or cut out the section and re-inspect",
      High: "reject the piece or cut out the damaged section; holes weaken the fabric",
    },
  },
  stain: {
    key: "stain",
    label: "Stain / Oil Spot",
    subject: "A stain-type defect (oil or dirt spot)",
    definition:
      "A foreign substance or discolouration on the surface: oil, grease, tea, coffee, ink, food, mud, rust, dye transferred from another garment or bleach spots.",
    thresholds: [1.5, 6.0],
    minSeverity: "Low",
    causesShort: ["machine oil", "contamination during handling"],
    causes: [
      "Machine oil or grease from looms, needles or rollers",
      "Contact with dirty surfaces, floors or hands during handling",
      "Food, drink or ink splashes",
      "Rust from metal parts or containers",
      "Dye transfer from a wet, darker garment, or a bleach splash",
    ],
    actions: {
      None: "no action needed",
      Low: "spot-clean the area and re-inspect; pass if the mark is gone",
      Medium: "send for stain removal or washing; downgrade if it persists",
      High: "reject or downgrade to second quality, and trace the source of contamination",
    },
  },
  broken_thread: {
    key: "broken_thread",
    label: "Broken Thread / Slub",
    subject: "A broken-thread defect",
    definition: "A missing, broken, pulled or snagged yarn, or a thick uneven section of yarn (slub).",
    thresholds: [1.0, 4.0],
    minSeverity: "Low",
    causesShort: ["excessive yarn tension", "a weak, uneven yarn"],
    causes: [
      "Yarn breakage caused by excessive tension on the loom or knitting machine",
      "Weak, uneven or slubby yarn",
      "A snag that pulled a yarn out of the structure",
      "Worn heddles, reeds or needles",
    ],
    actions: {
      None: "no action needed",
      Low: "mend the thread and pass the fabric",
      Medium: "repair (mend or darn) the affected area before dispatch",
      High: "reject the section and check yarn quality and machine tension",
    },
  },
};

export function isClassKey(key: string): key is ClassKey {
  return key in DEFECTS;
}

/** Severity from the defect type and the affected area (% of the image). */
export function severityFor(key: ClassKey, areaPct: number): Severity {
  if (key === "good") return "None";
  const info = DEFECTS[key];
  const [low, medium] = info.thresholds;
  const byArea: Severity = areaPct <= low ? "Low" : areaPct <= medium ? "Medium" : "High";
  return SEVERITY_ORDER.indexOf(byArea) >= SEVERITY_ORDER.indexOf(info.minSeverity) ? byArea : info.minSeverity;
}
