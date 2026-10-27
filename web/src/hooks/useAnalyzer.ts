import { useCallback, useEffect, useRef, useState } from "react";

import { describe, type DefectReport } from "../lib/descriptionEngine";
import { classMap, localize, type Localization } from "../lib/localization";
import { loadModel, predict, type LoadedModel, type Prediction } from "../lib/model";
import { occlusionMap } from "../lib/occlusion";
import { ImageInputError, prepareImage, validateFile, type PreparedImage } from "../lib/preprocess";

export type Phase = "idle" | "preparing" | "loading-model" | "analyzing" | "done" | "error";
export type ModelState = "idle" | "loading" | "ready" | "error";

export interface AnalysisResult {
  image: PreparedImage;
  prediction: Prediction;
  loc: Localization | null;
  report: DefectReport;
  inferenceMs: number;
  sourceLabel: string;
  refined: boolean;
}

const argmax = (a: ArrayLike<number>) => {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
};

function buildReport(model: LoadedModel, probs: Float32Array, loc: Localization | null): DefectReport {
  const { meta } = model;
  return describe({
    probs,
    classNames: meta.class_names,
    displayNames: meta.display_names,
    localization: loc,
    modelName: meta.architecture,
    demo: meta.demo,
  });
}

function messageFor(err: unknown): string {
  if (err instanceof ImageInputError) return err.message;
  const text = err instanceof Error ? err.message : String(err);
  if (/model_meta|fabricscan\.onnx|HTTP 404|failed to load model|protobuf/i.test(text)) {
    return "The AI model could not be loaded. Make sure web/public/models/ contains fabricscan.onnx (run the training script), then reload the page.";
  }
  return `Something went wrong while analysing the image: ${text}`;
}

export function useAnalyzer() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [refining, setRefining] = useState<{ done: number; total: number } | null>(null);
  const [preview, setPreview] = useState<string | null>(null); // object URL of the photo being analysed
  const previewUrl = useRef<string | null>(null);
  const run = useRef(0);

  const showPreview = useCallback((blob: Blob | null) => {
    if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    previewUrl.current = blob ? URL.createObjectURL(blob) : null;
    setPreview(previewUrl.current);
  }, []);
  useEffect(() => () => showPreview(null), [showPreview]);

  const ensureModel = useCallback(async () => {
    setModelState((s) => (s === "ready" ? s : "loading"));
    try {
      const model = await loadModel();
      setModelState("ready");
      return model;
    } catch (err) {
      setModelState("error");
      throw err;
    }
  }, []);

  const preload = useCallback(() => {
    ensureModel().catch(() => undefined);
  }, [ensureModel]);

  const analyze = useCallback(
    async (blob: Blob, sourceLabel: string) => {
      const id = ++run.current;
      setError(null);
      setRefining(null);
      showPreview(blob);
      setPhase("preparing");
      try {
        if (blob instanceof File) {
          const problem = validateFile(blob);
          if (problem) throw new ImageInputError(problem);
        }
        const image = await prepareImage(blob);
        setPhase("loading-model");
        const model = await ensureModel();
        setPhase("analyzing");
        const start = performance.now();
        const [prediction] = await predict(model, [image.pixels]);
        const inferenceMs = performance.now() - start;
        const cls = argmax(prediction.probs);
        const [h, w, k] = prediction.camShape;
        const loc =
          model.meta.class_names[cls] === "good"
            ? null
            : localize(classMap(prediction.cam, h, w, k, cls), h, w, image.rgba, image.size, "cam");
        const report = buildReport(model, prediction.probs, loc);
        if (id !== run.current) return;
        setResult({ image, prediction, loc, report, inferenceMs, sourceLabel, refined: false });
        setPhase("done");
      } catch (err) {
        if (id !== run.current) return;
        setError(messageFor(err));
        setPhase("error");
      }
    },
    [ensureModel, showPreview],
  );

  /** Re-localise with the slower, model-agnostic occlusion map. */
  const refine = useCallback(async () => {
    if (!result || !result.report.isDefect) return;
    const id = run.current;
    const model = await ensureModel();
    const { image, prediction } = result;
    const cls = argmax(prediction.probs);
    const grid = 7;
    setRefining({ done: 0, total: grid * grid });
    try {
      const map = await occlusionMap(
        image.pixels,
        image.size,
        cls,
        prediction.probs[cls],
        async (batch) => (await predict(model, batch)).map((p) => p.probs),
        { grid },
        (done, total) => setRefining({ done, total }),
      );
      if (id !== run.current) return;
      const loc = localize(map, grid, grid, image.rgba, image.size, "occlusion");
      setResult({ ...result, loc, report: buildReport(model, prediction.probs, loc), refined: true });
    } finally {
      setRefining(null);
    }
  }, [ensureModel, result]);

  const reset = useCallback(() => {
    run.current++;
    showPreview(null);
    setResult(null);
    setError(null);
    setPhase("idle");
  }, [showPreview]);

  return { phase, modelState, error, result, refining, preview, analyze, refine, reset, preload };
}
