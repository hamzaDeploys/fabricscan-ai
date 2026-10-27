import { ImagePlus, ScanLine } from "lucide-react";
import type { RefObject } from "react";

import type { AnalysisResult, Phase } from "../../hooks/useAnalyzer";
import { fmtMs } from "../../lib/format";
import type { OverlayMode } from "../../lib/overlay";
import { HeatmapView } from "./HeatmapView";

interface Props {
  phase: Phase;
  preview: string | null;
  result: AnalysisResult | null;
  mode: OverlayMode;
  onModeChange: (m: OverlayMode) => void;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  onBrowse: () => void;
  busy: boolean;
}

/** Left side of the workspace: drop zone → photo being scanned → analysed photo with heatmap. */
export function ImagePanel({ phase, preview, result, mode, onModeChange, canvasRef, onBrowse, busy }: Props) {
  if (phase === "done" && result) {
    const { loc } = result;
    const method = loc ? (loc.method.includes("occlusion") ? "occlusion map" : "activation map") + (loc.method.includes("contrast") ? " + contrast" : "") : null;
    return (
      <figure className="mx-auto w-full max-w-[560px] lg:sticky lg:top-24">
        <HeatmapView image={result.image.display} loc={loc} mode={mode} onModeChange={onModeChange} canvasRef={canvasRef} />
        <figcaption className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted">
          <span className="truncate">{result.sourceLabel}</span>
          <span>· {fmtMs(result.inferenceMs)}</span>
          {method && <span>· {method}</span>}
        </figcaption>
      </figure>
    );
  }

  if (busy && preview) {
    return (
      <div className="relative mx-auto aspect-square w-full max-w-[560px] overflow-hidden rounded-2xl border border-line bg-surface-3">
        <img src={preview} alt="Photo being analysed" className="size-full object-cover" onError={(e) => (e.currentTarget.style.display = "none")} />
        <div className="absolute inset-0 bg-black/25" />
        <div className="scan-line absolute inset-x-0 top-0 h-16 bg-gradient-to-b from-transparent via-[color-mix(in_oklab,var(--accent-bright)_60%,transparent)] to-transparent" />
        <p className="absolute bottom-3 left-3 inline-flex items-center gap-1.5 rounded-lg bg-black/55 px-2.5 py-1.5 text-xs font-medium text-white backdrop-blur-md">
          <ScanLine className="size-3.5" aria-hidden="true" /> Scanning…
        </p>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={onBrowse}
      disabled={busy}
      className="group weave mx-auto flex aspect-square w-full max-w-[560px] flex-col items-center justify-center rounded-2xl border-2 border-dashed border-line bg-surface px-6 text-center transition-colors hover:border-brand hover:bg-brand-soft/40 disabled:opacity-60"
    >
      <span className="flex size-16 items-center justify-center rounded-2xl bg-brand-soft text-brand transition-transform group-hover:scale-105">
        <ImagePlus className="size-8" aria-hidden="true" />
      </span>
      <span className="mt-5 font-display text-xl font-semibold text-ink">Drop a fabric photo here</span>
      <span className="mt-1.5 text-sm text-ink-2">
        or <span className="font-semibold text-brand underline-offset-4 group-hover:underline">click to choose a file</span>
      </span>
      <span className="mt-6 text-xs text-muted">JPG, PNG or WebP · up to 25 MB · Ctrl+V to paste</span>
    </button>
  );
}
