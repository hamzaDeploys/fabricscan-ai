import { motion } from "motion/react";

import type { Ranked } from "../../lib/descriptionEngine";
import { pct } from "../../lib/format";

/** One compact row per class: name, bar, value. The predicted class is drawn in the brand colour. */
export function ConfidenceBars({ ranked }: { ranked: Ranked[] }) {
  return (
    <ul className="space-y-2" data-testid="confidence-bars">
      {ranked.map((r, i) => (
        <li
          key={r.key}
          aria-label={`${r.label}: ${pct(r.prob)}`}
          className="grid grid-cols-[minmax(0,10rem)_minmax(0,1fr)_3.25rem] items-center gap-3 text-sm"
        >
          <span className={`truncate ${i === 0 ? "font-semibold text-ink" : "text-ink-2"}`}>{r.label}</span>
          <span className="h-2 overflow-hidden rounded-full bg-surface-3">
            <motion.span
              className="block h-full rounded-full"
              style={{ background: i === 0 ? "var(--btn)" : "color-mix(in oklab, var(--muted) 55%, transparent)" }}
              initial={{ width: 0 }}
              animate={{ width: `${Math.max(r.prob * 100, 0.8)}%` }}
              transition={{ duration: 0.6, delay: i * 0.04 }}
            />
          </span>
          <span className={`text-right tabular-nums ${i === 0 ? "font-semibold text-ink" : "text-muted"}`}>{pct(r.prob)}</span>
        </li>
      ))}
    </ul>
  );
}
