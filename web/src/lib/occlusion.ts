/**
 * Occlusion sensitivity map: grey out one patch at a time and measure how much the predicted class's probability
 * drops. Model-agnostic and slower (49 extra inferences) than the class activation map, so it is offered as an
 * optional "refine" step that double-checks where the model is looking.
 */

export interface OcclusionOptions {
  grid?: number; // patches per side
  batch?: number; // images per inference call
}

/** Copy of `pixels` with a square patch filled by the image's mean colour. */
export function occlude(pixels: Float32Array, size: number, cx: number, cy: number, patch: number, fill: number[]): Float32Array {
  const out = pixels.slice();
  const x0 = Math.max(0, Math.round(cx - patch / 2));
  const y0 = Math.max(0, Math.round(cy - patch / 2));
  const x1 = Math.min(size, x0 + patch);
  const y1 = Math.min(size, y0 + patch);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * size + x) * 3;
      out[i] = fill[0];
      out[i + 1] = fill[1];
      out[i + 2] = fill[2];
    }
  }
  return out;
}

export function meanColor(pixels: Float32Array): number[] {
  const sum = [0, 0, 0];
  for (let i = 0; i < pixels.length; i += 3) {
    sum[0] += pixels[i];
    sum[1] += pixels[i + 1];
    sum[2] += pixels[i + 2];
  }
  const n = pixels.length / 3;
  return sum.map((s) => s / n);
}

/**
 * @returns grid*grid map of probability drops (>= 0) for `classIndex`.
 */
export async function occlusionMap(
  pixels: Float32Array,
  size: number,
  classIndex: number,
  baseProb: number,
  predictProbs: (batch: Float32Array[]) => Promise<Float32Array[]>,
  { grid = 7, batch = 7 }: OcclusionOptions = {},
  onProgress?: (done: number, total: number) => void,
): Promise<Float32Array> {
  const cell = size / grid;
  const patch = Math.round(cell * 1.5);
  const fill = meanColor(pixels);
  const total = grid * grid;
  const map = new Float32Array(total);
  for (let start = 0; start < total; start += batch) {
    const ids = Array.from({ length: Math.min(batch, total - start) }, (_, j) => start + j);
    const images = ids.map((id) => occlude(pixels, size, ((id % grid) + 0.5) * cell, (Math.floor(id / grid) + 0.5) * cell, patch, fill));
    const probs = await predictProbs(images);
    ids.forEach((id, j) => {
      map[id] = Math.max(0, baseProb - probs[j][classIndex]);
    });
    onProgress?.(Math.min(total, start + batch), total);
    await new Promise((r) => setTimeout(r, 0)); // keep the page responsive between batches
  }
  return map;
}
