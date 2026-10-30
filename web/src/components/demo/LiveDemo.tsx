import { BrainCircuit, Camera, ChevronDown, CircleAlert, Crosshair, FileText, Gauge, Layers, RotateCcw, ShieldCheck, Upload } from "lucide-react";
import { useEffect, useRef, useState, type DragEvent } from "react";

import { useAnalyzer } from "../../hooks/useAnalyzer";
import { useSiteData } from "../../hooks/useSiteData";
import { fmtMB, pct } from "../../lib/format";
import type { OverlayMode } from "../../lib/overlay";
import type { SampleImage } from "../../types";
import { SectionHeader } from "../layout/SectionHeader";
import { AnalysisProgress } from "./AnalysisProgress";
import { CameraCapture } from "./CameraCapture";
import { ImagePanel } from "./ImagePanel";
import { ResultSummary } from "./ResultSummary";
import { SampleStrip } from "./SampleStrip";

const YOU_GET = [
  { icon: Layers, title: "Class and confidence", text: "Scores for all four fabric classes" },
  { icon: Crosshair, title: "Where the defect is", text: "Heatmap, zone and bounding box" },
  { icon: Gauge, title: "Area and severity", text: "Graded Low, Medium or High" },
  { icon: FileText, title: "A written report", text: "Causes, action, PDF and PNG" },
];

const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files");

export function LiveDemo() {
  const { meta, metrics, models, samples } = useSiteData();
  const { phase, modelState, error, result, refining, preview, metaFile, analyze, chooseModel, refine, reset, preload } = useAnalyzer();
  const [cameraOpen, setCameraOpen] = useState(false);
  const [mode, setMode] = useState<OverlayMode>("spotlight");
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);
  const fileInput = useRef<HTMLInputElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const busy = phase === "preparing" || phase === "loading-model" || phase === "analyzing";

  // Start downloading the model shortly after the page loads, so the first analysis is fast.
  useEffect(() => {
    const t = setTimeout(preload, 1200);
    return () => clearTimeout(t);
  }, [preload]);

  // Paste an image from the clipboard (Ctrl+V) anywhere on the page.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const file = Array.from(e.clipboardData?.files ?? []).find((f) => f.type.startsWith("image/"));
      if (file) {
        analyze(file, "Pasted image");
        document.getElementById("demo")?.scrollIntoView({ behavior: "smooth" });
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [analyze]);

  const pickSample = async (s: SampleImage) => {
    const res = await fetch(s.file);
    analyze(await res.blob(), `Sample: ${s.label}`);
  };

  const dropHandlers = {
    onDragEnter: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      dragDepth.current += 1;
      setDragging(true);
    },
    onDragLeave: () => {
      dragDepth.current = Math.max(0, dragDepth.current - 1);
      if (dragDepth.current === 0) setDragging(false);
    },
    onDragOver: (e: DragEvent) => e.preventDefault(),
    onDrop: (e: DragEvent) => {
      e.preventDefault();
      dragDepth.current = 0;
      setDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file && !busy) analyze(file, file.name);
    },
  };

  const chosen = models.find((m) => m.meta === metaFile);
  const chosenStats = chosen && metrics?.models[chosen.key];

  const modelText =
    modelState === "ready" ? "ready" : modelState === "loading" ? "loading…" : modelState === "error" ? "failed to load" : "not loaded yet";
  const dot =
    modelState === "ready" ? "var(--sev-none)" : modelState === "loading" ? "var(--sev-low)" : modelState === "error" ? "var(--sev-high)" : "var(--muted)";

  return (
    <section id="demo" aria-labelledby="demo-title" className="border-t border-line bg-surface-3/40 py-20 sm:py-24">
      <div className="container-page">
        <SectionHeader eyebrow="Live demo" title="Analyse a fabric photo" id="demo-title">
          Choose a model, then upload a photo, take one with your camera or pick a sample. Switching the model checks the
          same photo again. Everything runs inside your browser, so your photo is never uploaded anywhere.
        </SectionHeader>

        <div className="card relative mt-10 overflow-clip" {...dropHandlers}>
          {/* Which model answers */}
          {models.length > 1 && (
            <div className="flex flex-col gap-2 border-b border-line bg-surface-3/50 p-4 sm:flex-row sm:items-center sm:gap-3 sm:px-5 sm:py-3">
              <label htmlFor="model-select" className="flex shrink-0 items-center gap-2 text-sm font-semibold text-ink">
                <BrainCircuit className="size-4 text-brand" aria-hidden="true" /> Model
              </label>
              <span className="relative block sm:shrink-0">
                <select
                  id="model-select"
                  className="select w-full"
                  data-testid="model-select"
                  value={metaFile}
                  disabled={busy}
                  onChange={(e) => chooseModel(e.target.value)}
                >
                  {models.map((m) => (
                    <option key={m.key} value={m.meta}>
                      {m.name}
                    </option>
                  ))}
                </select>
                <ChevronDown className="pointer-events-none absolute top-1/2 right-3 size-4 -translate-y-1/2 text-muted" aria-hidden="true" />
              </span>
              {chosenStats && (
                <p className="text-xs leading-relaxed text-muted sm:ml-1" data-testid="model-facts">
                  {pct(chosenStats.test.accuracy)} test accuracy · macro F1 {chosenStats.test.f1_macro.toFixed(2)} · {fmtMB(chosenStats.onnx_size_mb)}
                  {chosen.key === metrics?.best_model ? " · the better model" : " · the weaker model, expect more mistakes"}
                </p>
              )}
            </div>
          )}

          {/* Toolbar: input options */}
          <div className="flex flex-col gap-4 border-b border-line p-4 sm:p-5 lg:flex-row lg:items-center lg:gap-6">
            <div className="flex shrink-0 flex-wrap items-center gap-2">
              <button type="button" className="btn btn-primary" onClick={() => fileInput.current?.click()} disabled={busy}>
                <Upload className="size-4" aria-hidden="true" /> Upload photo
              </button>
              <button type="button" className="btn btn-secondary" onClick={() => setCameraOpen(true)} disabled={busy}>
                <Camera className="size-4" aria-hidden="true" /> Camera
              </button>
            </div>
            <div className="hidden h-12 w-px shrink-0 bg-line lg:block" aria-hidden="true" />
            <SampleStrip samples={samples} onPick={pickSample} disabled={busy} />
            <input
              ref={fileInput}
              type="file"
              accept="image/*,.heic,.heif"
              className="sr-only"
              aria-label="Choose a fabric image to analyse"
              data-testid="file-input"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) analyze(file, file.name);
                e.target.value = "";
              }}
            />
          </div>

          {/* Workspace: image on the left, result on the right */}
          <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)]">
            <div className="border-b border-line bg-surface p-4 sm:p-6 lg:border-r lg:border-b-0">
              <ImagePanel
                phase={phase}
                preview={preview}
                result={result}
                mode={mode}
                onModeChange={setMode}
                canvasRef={canvas}
                onBrowse={() => fileInput.current?.click()}
                busy={busy}
              />
            </div>
            <div className="min-h-[420px] p-5 sm:p-7" aria-live="polite" aria-busy={busy}>
              {phase === "idle" && (
                <div className="flex h-full flex-col justify-center">
                  <p className="font-display text-xl font-semibold text-ink">What you will get</p>
                  <ul className="mt-5 space-y-4">
                    {YOU_GET.map(({ icon: Icon, title, text }) => (
                      <li key={title} className="flex items-start gap-3">
                        <span className="icon-tile size-9 shrink-0">
                          <Icon className="size-[18px]" aria-hidden="true" />
                        </span>
                        <span>
                          <span className="block text-sm font-semibold text-ink">{title}</span>
                          <span className="block text-sm text-ink-2">{text}</span>
                        </span>
                      </li>
                    ))}
                  </ul>
                  <p className="mt-7 border-t border-line pt-4 text-xs leading-relaxed text-muted">
                    Best results: fill the frame with fabric, use soft daylight, no flash, and hold the phone 20–35 cm above
                    the fabric.
                  </p>
                </div>
              )}
              {busy && <AnalysisProgress phase={phase} />}
              {phase === "error" && (
                <div className="flex h-full flex-col items-start justify-center gap-4" role="alert">
                  <span className="flex size-11 items-center justify-center rounded-xl bg-[color-mix(in_oklab,var(--sev-high)_14%,transparent)]">
                    <CircleAlert className="size-6 text-[var(--sev-high-ink)]" aria-hidden="true" />
                  </span>
                  <p className="font-display text-lg font-semibold text-ink">That did not work</p>
                  <p className="max-w-md text-sm text-ink-2">{error}</p>
                  <button type="button" className="btn btn-secondary" onClick={reset}>
                    <RotateCcw className="size-4" aria-hidden="true" /> Try another image
                  </button>
                </div>
              )}
              {phase === "done" && result && (
                <ResultSummary result={result} canvasRef={canvas} refining={refining} onRefine={refine} onReset={reset} />
              )}
            </div>
          </div>

          {/* Status bar */}
          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 border-t border-line bg-surface px-5 py-2.5 text-xs text-muted">
            <span className="flex items-center gap-2" data-testid="model-status">
              <span className={`size-2 rounded-full ${modelState === "loading" ? "animate-pulse" : ""}`} style={{ background: dot }} aria-hidden="true" />
              {chosen?.name ?? meta?.architecture ?? "Model"} · {modelText}
              {meta?.demo ? " · demo" : ""}
            </span>
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="size-3.5" aria-hidden="true" /> Runs on your device · nothing is uploaded
            </span>
          </div>

          {dragging && (
            <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center rounded-2xl border-2 border-dashed border-brand bg-[color-mix(in_oklab,var(--surface-2)_82%,transparent)] backdrop-blur-sm">
              <p className="flex items-center gap-2 font-display text-xl font-semibold text-brand">
                <Upload className="size-6" aria-hidden="true" /> Drop the photo to analyse it
              </p>
            </div>
          )}
        </div>
      </div>
      {cameraOpen && <CameraCapture onClose={() => setCameraOpen(false)} onCapture={(blob) => analyze(blob, "Camera capture")} />}
    </section>
  );
}
