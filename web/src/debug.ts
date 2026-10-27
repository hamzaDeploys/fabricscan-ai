/**
 * Development-only hook used by the end-to-end test (e2e/demo.spec.ts) to check that the browser engine gives the
 * same probabilities as ONNX Runtime in Python for the same preprocessed pixels. Not included in production builds.
 */
import { loadModel, predict } from "./lib/model";

declare global {
  interface Window {
    __fabricscan?: { runPixelsBase64: (b64: string) => Promise<number[]> };
  }
}

window.__fabricscan = {
  async runPixelsBase64(b64: string) {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const model = await loadModel();
    const [out] = await predict(model, [Float32Array.from(bytes)]);
    return Array.from(out.probs);
  },
};
