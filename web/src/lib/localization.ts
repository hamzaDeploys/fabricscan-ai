/**
 * Turn a coarse class-activation map (7x7 or 14x14) into a defect location, a bounding box and an area estimate.
 *
 * 1. Min-max normalise the map and upsample it (bilinear) to the image size: the "heat".
 * 2. Hot area = heat >= 0.5. Its heat-weighted centroid gives the 3x3 zone ("upper-left", ...).
 * 3. Contrast refinement: the fabric's expected colour at every pixel comes from a smooth illumination model, a
 *    quadratic surface per colour channel fitted to the cold area (heat < 0.25), so uneven lighting and lens
 *    vignetting are not mistaken for defects. Inside the warm area (heat >= 0.2), pixels that differ from the
 *    expected colour by more than a robust threshold (median + 4 x MAD of the cold area's residuals, at least 20)
 *    count as defect pixels. Isolated speckles and blobs that do not touch the hot area are discarded. This is
 *    much finer than the 7x7 map, so small holes or stains are not overestimated.
 * 4. If the defect has almost no colour contrast (fewer than 0.05% of pixels pass), half of the hot area is used
 *    as a rough estimate instead.
 */

export type Zone =
  | "upper-left"
  | "upper-center"
  | "upper-right"
  | "middle-left"
  | "center"
  | "middle-right"
  | "lower-left"
  | "lower-center"
  | "lower-right";

export type LocalizationMethod = "cam" | "occlusion";

export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface Localization {
  size: number;
  heat: Float32Array; // size*size, 0..1
  mask: Uint8Array; // size*size, 1 = defect pixel (contrast refinement)
  centroid: { x: number; y: number }; // 0..1
  zone: Zone;
  zoneLabel: string;
  bbox: Box; // 0..1
  areaPct: number;
  method: string; // e.g. "cam+contrast"
}

const HOT = 0.5;
const COLD = 0.25;
const WARM = 0.2;
const MIN_CONTRAST = 20;
const MIN_CONTRAST_SHARE = 0.0005;
const MIN_BLOB_SHARE = 0.0002; // ~10 px at 224 x 224: smaller blobs are texture speckle, not defects

export function zoneOf(x: number, y: number): Zone {
  const col = x < 1 / 3 ? "left" : x < 2 / 3 ? "center" : "right";
  const row = y < 1 / 3 ? "upper" : y < 2 / 3 ? "middle" : "lower";
  if (row === "middle" && col === "center") return "center";
  return `${row}-${col}` as Zone;
}

export function zoneLabel(zone: Zone): string {
  return zone === "center" ? "center of the fabric" : `${zone} region`;
}

/** Extract one class channel from a [h, w, k] activation tensor (row-major, as returned by ONNX Runtime). */
export function classMap(cam: ArrayLike<number>, h: number, w: number, k: number, cls: number): Float32Array {
  const out = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) out[i] = cam[i * k + cls];
  return out;
}

export function normalizeMap(map: ArrayLike<number>): Float32Array {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < map.length; i++) {
    if (map[i] < min) min = map[i];
    if (map[i] > max) max = map[i];
  }
  const out = new Float32Array(map.length);
  const range = max - min;
  if (!(range > 1e-9)) return out;
  for (let i = 0; i < map.length; i++) out[i] = (map[i] - min) / range;
  return out;
}

/** Bilinear upsampling with half-pixel centres (same convention as tf.image.resize). */
export function upsample(map: ArrayLike<number>, h: number, w: number, size: number): Float32Array {
  const out = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    const sy = Math.min(h - 1, Math.max(0, ((y + 0.5) * h) / size - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(h - 1, y0 + 1);
    const fy = sy - y0;
    for (let x = 0; x < size; x++) {
      const sx = Math.min(w - 1, Math.max(0, ((x + 0.5) * w) / size - 0.5));
      const x0 = Math.floor(sx);
      const x1 = Math.min(w - 1, x0 + 1);
      const fx = sx - x0;
      const top = (1 - fx) * map[y0 * w + x0] + fx * map[y0 * w + x1];
      const bottom = (1 - fx) * map[y1 * w + x0] + fx * map[y1 * w + x1];
      out[y * size + x] = (1 - fy) * top + fy * bottom;
    }
  }
  return out;
}

/** Solve the small dense system M x = b (Gaussian elimination with partial pivoting). */
function solve(M: number[][], b: number[]): number[] {
  const n = b.length;
  const a = M.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    [a[c], a[p]] = [a[p], a[c]];
    const pivot = a[c][c] || 1e-12;
    for (let r = c + 1; r < n; r++) {
      const f = a[r][c] / pivot;
      for (let k = c; k <= n; k++) a[r][k] -= f * a[c][k];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = a[r][n];
    for (let k = r + 1; k < n; k++) s -= a[r][k] * x[k];
    x[r] = s / (a[r][r] || 1e-12);
  }
  return x;
}

/**
 * Least-squares illumination model per RGB channel over the pixels where `use[i]` is true: a quadratic surface
 * (1, u, v, u², v², uv) when they cover enough of the image, otherwise a plane (1, u, v), which extrapolates
 * more safely. Returns the expected colour of every pixel as size*size*3 floats.
 */
export function fitIllumination(rgba: ArrayLike<number>, size: number, use: (i: number) => boolean): Float32Array {
  const n = size * size;
  let coverage = 0;
  for (let i = 0; i < n; i += 4) if (use(i)) coverage++;
  const quadratic = coverage * 4 >= n * 0.4;
  const terms = (u: number, v: number) => (quadratic ? [1, u, v, u * u, v * v, u * v] : [1, u, v]);
  const k = quadratic ? 6 : 3;
  const ata = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const atb = [0, 1, 2].map(() => new Array<number>(k).fill(0));
  for (let y = 0; y < size; y += 2) {
    for (let x = 0; x < size; x += 2) {
      const i = y * size + x;
      if (!use(i)) continue;
      const t = terms(x / size - 0.5, y / size - 0.5);
      for (let r = 0; r < k; r++) {
        for (let c = 0; c < k; c++) ata[r][c] += t[r] * t[c];
        for (let ch = 0; ch < 3; ch++) atb[ch][r] += t[r] * rgba[i * 4 + ch];
      }
    }
  }
  for (let r = 0; r < k; r++) ata[r][r] += 1e-6; // keeps the system solvable for degenerate masks
  const coef = atb.map((b) => solve(ata, b));
  const out = new Float32Array(n * 3);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const t = terms(x / size - 0.5, y / size - 0.5);
      const i = y * size + x;
      for (let ch = 0; ch < 3; ch++) {
        let s = 0;
        for (let r = 0; r < k; r++) s += coef[ch][r] * t[r];
        out[i * 3 + ch] = s;
      }
    }
  }
  return out;
}

/**
 * Remove texture speckle from the contrast mask: keep only 4-connected blobs that are at least MIN_BLOB_SHARE of
 * the image and touch the hot area (heat >= 0.5). Modifies `mask` in place.
 */
export function keepDefectComponents(mask: Uint8Array, heat: Float32Array, size: number): void {
  const n = size * size;
  const minPixels = Math.max(8, Math.round(n * MIN_BLOB_SHARE));
  const seen = new Uint8Array(n);
  const queue = new Int32Array(n);
  for (let start = 0; start < n; start++) {
    if (!mask[start] || seen[start]) continue;
    let head = 0;
    let tail = 0;
    let touchesHot = false;
    queue[tail++] = start;
    seen[start] = 1;
    while (head < tail) {
      const i = queue[head++];
      if (heat[i] >= HOT) touchesHot = true;
      const x = i % size;
      const neighbours = [x > 0 ? i - 1 : -1, x < size - 1 ? i + 1 : -1, i - size, i + size];
      for (const j of neighbours) {
        if (j >= 0 && j < n && mask[j] && !seen[j]) {
          seen[j] = 1;
          queue[tail++] = j;
        }
      }
    }
    if (tail < minPixels || !touchesHot) {
      for (let k = 0; k < tail; k++) mask[queue[k]] = 0;
    }
  }
}

function median(values: Float32Array): number {
  if (values.length === 0) return 0;
  const sorted = Float32Array.from(values).sort();
  return sorted[Math.floor(sorted.length / 2)];
}

/**
 * @param rawMap class activation (or occlusion) map, h*w values
 * @param rgba   the model-input image as RGBA bytes, size*size*4
 */
export function localize(
  rawMap: ArrayLike<number>,
  h: number,
  w: number,
  rgba: ArrayLike<number>,
  size: number,
  method: LocalizationMethod = "cam",
): Localization {
  const n = size * size;
  const heat = upsample(normalizeMap(rawMap), h, w, size);
  const mask = new Uint8Array(n);

  let sx = 0;
  let sy = 0;
  let sw = 0;
  let hotCount = 0;
  let hx0 = size;
  let hy0 = size;
  let hx1 = 0;
  let hy1 = 0;
  for (let i = 0; i < n; i++) {
    if (heat[i] >= HOT) {
      const x = i % size;
      const y = (i - x) / size;
      sx += (x + 0.5) * heat[i];
      sy += (y + 0.5) * heat[i];
      sw += heat[i];
      hotCount++;
      if (x < hx0) hx0 = x;
      if (y < hy0) hy0 = y;
      if (x > hx1) hx1 = x;
      if (y > hy1) hy1 = y;
    }
  }
  if (sw === 0) {
    return {
      size, heat, mask, centroid: { x: 0.5, y: 0.5 }, zone: "center", zoneLabel: zoneLabel("center"),
      bbox: { x0: 0, y0: 0, x1: 1, y1: 1 }, areaPct: 0, method,
    };
  }

  // Expected fabric colour at every pixel, fitted on the cold area (or the whole image if that area is tiny).
  let coldCount = 0;
  for (let i = 0; i < n; i++) if (heat[i] < COLD) coldCount++;
  const useAll = coldCount < n * 0.05;
  const isBackground = (i: number) => useAll || heat[i] < COLD;
  const expected = fitIllumination(rgba, size, isBackground);

  const dist = new Float32Array(n);
  const coldDist = new Float32Array(useAll ? n : coldCount);
  let c = 0;
  for (let i = 0; i < n; i++) {
    const dr = rgba[i * 4] - expected[i * 3];
    const dg = rgba[i * 4 + 1] - expected[i * 3 + 1];
    const db = rgba[i * 4 + 2] - expected[i * 3 + 2];
    dist[i] = Math.sqrt(dr * dr + dg * dg + db * db);
    if (isBackground(i)) coldDist[c++] = dist[i];
  }
  const med = median(coldDist);
  const mad = median(coldDist.map((d) => Math.abs(d - med)));
  const threshold = Math.max(MIN_CONTRAST, med + 4 * 1.4826 * mad);

  for (let i = 0; i < n; i++) if (heat[i] >= WARM && dist[i] > threshold) mask[i] = 1;
  keepDefectComponents(mask, heat, size);

  let count = 0;
  let mx = 0;
  let my = 0;
  let bx0 = size;
  let by0 = size;
  let bx1 = 0;
  let by1 = 0;
  for (let i = 0; i < n; i++) {
    if (mask[i]) {
      count++;
      const x = i % size;
      const y = (i - x) / size;
      mx += x + 0.5;
      my += y + 0.5;
      if (x < bx0) bx0 = x;
      if (y < by0) by0 = y;
      if (x > bx1) bx1 = x;
      if (y > by1) by1 = y;
    }
  }

  let areaPct: number;
  let centroid: { x: number; y: number };
  let bbox: Box;
  let used: string;
  if (count >= n * MIN_CONTRAST_SHARE) {
    areaPct = (count / n) * 100;
    centroid = { x: mx / count / size, y: my / count / size };
    bbox = { x0: bx0 / size, y0: by0 / size, x1: (bx1 + 1) / size, y1: (by1 + 1) / size };
    used = `${method}+contrast`;
  } else {
    mask.fill(0);
    areaPct = ((hotCount / n) * 100) / 2;
    centroid = { x: sx / sw / size, y: sy / sw / size };
    bbox = { x0: hx0 / size, y0: hy0 / size, x1: (hx1 + 1) / size, y1: (hy1 + 1) / size };
    used = method;
  }
  const zone = zoneOf(centroid.x, centroid.y);
  return { size, heat, mask, centroid, zone, zoneLabel: zoneLabel(zone), bbox, areaPct, method: used };
}
