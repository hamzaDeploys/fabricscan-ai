import { CircleCheck, Info, OctagonAlert, TriangleAlert } from "lucide-react";

import type { Severity } from "../../lib/defectKnowledge";

const STYLE: Record<Severity, { icon: typeof Info; color: string; ink: string; text: string }> = {
  None: { icon: CircleCheck, color: "var(--sev-none)", ink: "var(--sev-none-ink)", text: "No defect" },
  Low: { icon: Info, color: "var(--sev-low)", ink: "var(--sev-low-ink)", text: "Severity: Low" },
  Medium: { icon: TriangleAlert, color: "var(--sev-medium)", ink: "var(--sev-medium-ink)", text: "Severity: Medium" },
  High: { icon: OctagonAlert, color: "var(--sev-high)", ink: "var(--sev-high-ink)", text: "Severity: High" },
};

/** Colour is never the only signal: every level has its own icon and label. */
export function SeverityBadge({ severity, short = false }: { severity: Severity; short?: boolean }) {
  const s = STYLE[severity];
  const Icon = s.icon;
  const text = short && severity !== "None" ? severity : s.text;
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold"
      style={{
        background: `color-mix(in oklab, ${s.color} 16%, transparent)`,
        color: s.ink,
        boxShadow: `inset 0 0 0 1px color-mix(in oklab, ${s.color} 45%, transparent)`,
      }}
      data-testid="severity"
    >
      <Icon className="size-3.5" aria-hidden="true" />
      {text}
    </span>
  );
}
