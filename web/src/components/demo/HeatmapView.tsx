import { CircleCheck, SquareDashed } from "lucide-react";
import { useEffect, type RefObject } from "react";

import type { Localization } from "../../lib/localization";
import { drawOverlay, type OverlayMode } from "../../lib/overlay";

const MODES: { mode: OverlayMode; label: string }[] = [
  { mode: "spotlight", label: "Spotlight" },
  { mode: "heatmap", label: "Heatmap" },
  { mode: "original", label: "Original" },
];

/** The analysed photo with the localisation drawn on top; the overlay switch floats on the image. */
export function HeatmapView({
  image,
  loc,
  mode,
  onModeChange,
  canvasRef,
}: {
  image: HTMLCanvasElement;
  loc: Localization | null;
  mode: OverlayMode;
  onModeChange: (m: OverlayMode) => void;
  canvasRef: RefObject<HTMLCanvasElement | null>;
}) {
  useEffect(() => {
    if (canvasRef.current) drawOverlay(canvasRef.current, image, loc, loc ? mode : "original");
  }, [image, loc, mode, canvasRef]);

  return (
    <div className="relative overflow-hidden rounded-2xl border border-line bg-surface-3 shadow-sm">
      <canvas
        ref={canvasRef}
        width={640}
        height={640}
        className="block aspect-square w-full"
        role="img"
        aria-label={loc ? `Analysed fabric with the estimated defect area highlighted (${loc.zoneLabel})` : "Analysed fabric photo"}
      />
      {loc ? (
        <>
          <div
            className="absolute top-3 left-3 inline-flex rounded-xl bg-black/55 p-1 text-white shadow-lg backdrop-blur-md"
            role="radiogroup"
            aria-label="Overlay style"
          >
            {MODES.map((m) => (
              <button
                key={m.mode}
                type="button"
                role="radio"
                aria-checked={mode === m.mode}
                onClick={() => onModeChange(m.mode)}
                className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors ${
                  mode === m.mode ? "bg-white text-[#12141c]" : "text-white/85 hover:bg-white/15"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
          {mode !== "original" && (
            <p className="absolute bottom-3 left-3 inline-flex items-center gap-1.5 rounded-lg bg-black/55 px-2.5 py-1.5 text-xs font-medium text-white backdrop-blur-md">
              <SquareDashed className="size-3.5 text-[#ff8a6b]" aria-hidden="true" /> Estimated defect area
            </p>
          )}
        </>
      ) : (
        <p className="absolute bottom-3 left-3 inline-flex items-center gap-1.5 rounded-lg bg-black/55 px-2.5 py-1.5 text-xs font-medium text-white backdrop-blur-md">
          <CircleCheck className="size-3.5 text-[#5fd35f]" aria-hidden="true" /> No defect area to highlight
        </p>
      )}
    </div>
  );
}
