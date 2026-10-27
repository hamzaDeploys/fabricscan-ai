import type { SampleImage } from "../../types";

const SHORT: Record<string, string> = {
  good: "Good",
  hole: "Hole",
  stain: "Stain",
  broken_thread: "Thread",
};

/** A single scrollable row of sample thumbnails with readable labels underneath. */
export function SampleStrip({ samples, onPick, disabled }: { samples: SampleImage[]; onPick: (s: SampleImage) => void; disabled?: boolean }) {
  if (!samples.length) return null;
  const synthetic = samples.some((s) => s.source.includes("synthetic"));
  return (
    <div className="min-w-0 flex-1">
      <p className="text-xs font-medium text-muted">
        Try a sample
        {synthetic ? " · computer-generated demo images" : " · clear examples, not used in training"}
      </p>
      <ul className="mt-2 flex gap-1.5 overflow-x-auto pb-1 [mask-image:linear-gradient(to_right,black_88%,transparent)] lg:[mask-image:none]">
        {samples.map((s) => (
          <li key={s.file} className="shrink-0">
            <button
              type="button"
              disabled={disabled}
              onClick={() => onPick(s)}
              title={`Analyse sample: ${s.label}`}
              aria-label={`Analyse sample image: ${s.label}`}
              className="group flex w-[52px] flex-col items-center gap-1 rounded-lg disabled:opacity-50"
            >
              <span className="block size-11 overflow-hidden rounded-lg border border-line bg-surface-3 ring-brand ring-offset-2 ring-offset-surface-2 transition group-hover:ring-2">
                <img src={s.file} alt="" loading="lazy" className="size-full object-cover" />
              </span>
              <span className="w-full truncate text-center text-[10.5px] font-medium text-ink-2">{SHORT[s.class] ?? s.class}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
