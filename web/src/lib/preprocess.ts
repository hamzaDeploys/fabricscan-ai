/**
 * Browser-side preprocessing that mirrors the Python pipeline:
 *   EXIF orientation -> centre square crop -> downscale (in halving steps, i.e. antialiased) to 256 -> 224.
 * The model receives raw RGB values 0-255; normalisation is inside the model.
 */

export const MAX_FILE_MB = 25;
export const MIN_SIDE = 64;

export interface PreparedImage {
  pixels: Float32Array; // size*size*3, RGB 0-255
  rgba: Uint8ClampedArray; // size*size*4, same image (used by the localiser)
  size: number;
  display: HTMLCanvasElement; // square crop for display (<= displaySize px)
  sourceWidth: number;
  sourceHeight: number;
}

export class ImageInputError extends Error {}

/** Largest centred square inside a w x h image. */
export function cropBox(w: number, h: number): { sx: number; sy: number; side: number } {
  const side = Math.min(w, h);
  return { sx: Math.floor((w - side) / 2), sy: Math.floor((h - side) / 2), side };
}

/** RGBA bytes -> RGB float32. */
export function rgbaToRgb(rgba: ArrayLike<number>): Float32Array {
  const n = rgba.length / 4;
  const out = new Float32Array(n * 3);
  for (let i = 0, j = 0; i < n; i++) {
    out[j++] = rgba[i * 4];
    out[j++] = rgba[i * 4 + 1];
    out[j++] = rgba[i * 4 + 2];
  }
  return out;
}

export function validateFile(file: File): string | null {
  const name = file.name.toLowerCase();
  const heic = /\.(heic|heif)$/.test(name) || /hei[cf]/.test(file.type);
  if (!file.type.startsWith("image/") && !heic) {
    return `"${file.name}" is not an image. Please choose a JPG, PNG or WebP photo.`;
  }
  if (file.size > MAX_FILE_MB * 1024 * 1024) {
    return `This file is ${(file.size / 1024 / 1024).toFixed(1)} MB; the limit is ${MAX_FILE_MB} MB.`;
  }
  return null;
}

async function decode(blob: Blob): Promise<ImageBitmap> {
  try {
    return await createImageBitmap(blob, { imageOrientation: "from-image" });
  } catch {
    // Some browsers only decode certain formats through <img>.
    const url = URL.createObjectURL(blob);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      return await createImageBitmap(img);
    } catch {
      const heic = /hei[cf]/i.test(blob.type) || (blob instanceof File && /\.(heic|heif)$/i.test(blob.name));
      throw new ImageInputError(
        heic
          ? "This browser cannot open HEIC photos. Convert it to JPG, or set the iPhone camera to 'Most Compatible'."
          : "The image could not be decoded. The file may be damaged or in an unsupported format.",
      );
    } finally {
      URL.revokeObjectURL(url);
    }
  }
}

function canvas(size: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Canvas 2D is not available in this browser.");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  return [c, ctx];
}

function resized(src: HTMLCanvasElement, size: number): HTMLCanvasElement {
  const [c, ctx] = canvas(size);
  ctx.drawImage(src, 0, 0, size, size);
  return c;
}

/** Downscale in halving steps (each step averages 2x2 pixels), which avoids aliasing on big phone photos. */
export function downscale(src: HTMLCanvasElement, target: number): HTMLCanvasElement {
  let cur = src;
  while (cur.width / 2 >= target) cur = resized(cur, Math.floor(cur.width / 2));
  return cur.width === target ? cur : resized(cur, target);
}

export async function prepareImage(blob: Blob, size = 224, displaySize = 640): Promise<PreparedImage> {
  const bitmap = await decode(blob);
  try {
    const { width, height } = bitmap;
    if (Math.min(width, height) < MIN_SIDE) {
      throw new ImageInputError(`The image is only ${width}x${height} px; please use a photo at least ${MIN_SIDE} px wide.`);
    }
    const { sx, sy, side } = cropBox(width, height);
    const first = Math.min(side, 2048);
    const [square, ctx] = canvas(first);
    ctx.drawImage(bitmap, sx, sy, side, side, 0, 0, first, first);

    const input = downscale(downscale(square, Math.max(256, size)), size);
    const rgba = input.getContext("2d", { willReadFrequently: true })!.getImageData(0, 0, size, size).data;
    const display = first > displaySize ? downscale(square, displaySize) : square;
    return { pixels: rgbaToRgb(rgba), rgba, size, display, sourceWidth: width, sourceHeight: height };
  } finally {
    bitmap.close();
  }
}
