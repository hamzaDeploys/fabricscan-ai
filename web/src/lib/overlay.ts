/**
 * Draws the analysed image with its localisation.
 *  - "spotlight": darkens everything except the hot area (readable on any fabric colour)
 *  - "heatmap":   single-hue tint whose opacity follows the activation (no rainbow colour map)
 *  - "original":  the photo only
 * A bounding box marks the estimated defect area in the first two modes.
 */
import type { Localization } from "./localization";

export type OverlayMode = "spotlight" | "heatmap" | "original";

const HEAT_RGB = [255, 72, 40];

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

function roundedRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

export function drawOverlay(
  target: HTMLCanvasElement,
  base: HTMLCanvasElement,
  loc: Localization | null,
  mode: OverlayMode,
  intensity = 0.75,
): void {
  const S = target.width;
  const ctx = target.getContext("2d")!;
  ctx.clearRect(0, 0, S, S);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(base, 0, 0, S, S);
  if (!loc || mode === "original") return;

  const n = loc.size;
  const layer = new ImageData(n, n);
  for (let i = 0; i < n * n; i++) {
    const h = loc.heat[i];
    const j = i * 4;
    if (mode === "spotlight") {
      layer.data[j + 3] = Math.round((1 - smoothstep(0.3, 0.65, h)) * 0.68 * intensity * 255);
    } else {
      layer.data[j] = HEAT_RGB[0];
      layer.data[j + 1] = HEAT_RGB[1];
      layer.data[j + 2] = HEAT_RGB[2];
      layer.data[j + 3] = Math.round(smoothstep(0.2, 1, h) * 0.8 * intensity * 255);
    }
  }
  const off = document.createElement("canvas");
  off.width = off.height = n;
  off.getContext("2d")!.putImageData(layer, 0, 0);
  ctx.drawImage(off, 0, 0, S, S);

  if (loc.areaPct > 0) {
    const pad = 0.02;
    const x = Math.max(0, loc.bbox.x0 - pad) * S;
    const y = Math.max(0, loc.bbox.y0 - pad) * S;
    const w = (Math.min(1, loc.bbox.x1 + pad) - Math.max(0, loc.bbox.x0 - pad)) * S;
    const h = (Math.min(1, loc.bbox.y1 + pad) - Math.max(0, loc.bbox.y0 - pad)) * S;
    const r = Math.min(12, w / 4, h / 4);
    ctx.lineWidth = Math.max(3, S / 160);
    ctx.strokeStyle = "rgba(255,255,255,0.95)";
    roundedRect(ctx, x, y, w, h, r);
    ctx.stroke();
    ctx.lineWidth = Math.max(1.5, S / 320);
    ctx.strokeStyle = "rgb(255,72,40)";
    roundedRect(ctx, x, y, w, h, r);
    ctx.stroke();
  }
}
