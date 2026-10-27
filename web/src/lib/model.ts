/**
 * Loads the ONNX model with onnxruntime-web (WebAssembly backend) and runs batched inference.
 * Uses the wasm-only build (~3.5 MB gzipped) instead of the WebGPU bundle; MobileNetV2 runs in tens of ms on a CPU.
 */
import * as ort from "onnxruntime-web/wasm";

import type { ModelMeta } from "../types";

export interface LoadedModel {
  session: ort.InferenceSession;
  meta: ModelMeta;
}

export interface Prediction {
  probs: Float32Array; // [k]
  cam: Float32Array; // [h * w * k]
  camShape: [number, number, number];
}

const BASE = import.meta.env.BASE_URL;
let loading: Promise<LoadedModel> | null = null;

export async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`Could not load ${path} (HTTP ${res.status})`);
  return (await res.json()) as T;
}

export function loadModel(): Promise<LoadedModel> {
  if (!loading) {
    loading = (async () => {
      // Several threads need cross-origin isolation (COOP/COEP headers); otherwise fall back to one thread.
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      const meta = await fetchJson<ModelMeta>("models/model_meta.json");
      const session = await ort.InferenceSession.create(`${BASE}models/${meta.model_file}`, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      return { session, meta };
    })().catch((err: unknown) => {
      loading = null; // allow a retry
      throw err;
    });
  }
  return loading;
}

/** Run a batch of preprocessed images (each size*size*3 floats, RGB 0-255, NHWC). */
export async function predict(model: LoadedModel, images: Float32Array[]): Promise<Prediction[]> {
  const { meta, session } = model;
  const size = meta.input.size;
  const per = size * size * 3;
  const data = new Float32Array(images.length * per);
  images.forEach((img, i) => data.set(img, i * per));
  const input = new ort.Tensor("float32", data, [images.length, size, size, 3]);
  const out = await session.run({ [meta.input.name]: input });
  const probs = out[meta.outputs.probs];
  const cam = out[meta.outputs.cam];
  const k = probs.dims[1];
  const [, h, w] = cam.dims;
  const probsData = probs.data as Float32Array;
  const camData = cam.data as Float32Array;
  const camPer = h * w * k;
  return images.map((_, i) => ({
    probs: probsData.slice(i * k, (i + 1) * k),
    cam: camData.slice(i * camPer, (i + 1) * camPer),
    camShape: [h, w, k] as [number, number, number],
  }));
}
