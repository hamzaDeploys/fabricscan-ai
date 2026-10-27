import { describe, expect, it } from "vitest";

import { classMap, localize, normalizeMap, upsample, zoneOf } from "./localization";

const SIZE = 224;

/** Plain fabric (RGB 150,120,90) with a dark square defect at [x0,y0]-[x1,y1). */
function fabricWithSquare(x0: number, y0: number, x1: number, y1: number): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(SIZE * SIZE * 4);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const i = (y * SIZE + x) * 4;
      const inside = x >= x0 && x < x1 && y >= y0 && y < y1;
      const noise = ((x * 7 + y * 13) % 5) - 2; // mild texture
      rgba[i] = inside ? 20 : 150 + noise;
      rgba[i + 1] = inside ? 20 : 120 + noise;
      rgba[i + 2] = inside ? 25 : 90 + noise;
      rgba[i + 3] = 255;
    }
  }
  return rgba;
}

/** 7x7 activation map peaking at cell (cx, cy). */
function peakMap(cx: number, cy: number): Float32Array {
  const m = new Float32Array(49);
  for (let y = 0; y < 7; y++) for (let x = 0; x < 7; x++) m[y * 7 + x] = Math.exp(-((x - cx) ** 2 + (y - cy) ** 2));
  return m;
}

describe("zoneOf", () => {
  it("maps the 3x3 grid", () => {
    expect(zoneOf(0.1, 0.1)).toBe("upper-left");
    expect(zoneOf(0.5, 0.5)).toBe("center");
    expect(zoneOf(0.9, 0.5)).toBe("middle-right");
    expect(zoneOf(0.5, 0.95)).toBe("lower-center");
  });
});

describe("map helpers", () => {
  it("min-max normalises and handles flat maps", () => {
    expect(Array.from(normalizeMap([2, 4, 6]))).toEqual([0, 0.5, 1]);
    expect(Array.from(normalizeMap([3, 3]))).toEqual([0, 0]);
  });
  it("upsamples a constant map to a constant map", () => {
    const up = upsample(new Float32Array(49).fill(0.4), 7, 7, 28);
    expect(up.every((v) => Math.abs(v - 0.4) < 1e-6)).toBe(true);
  });
  it("extracts one class channel from an HWK tensor", () => {
    const cam = new Float32Array([1, 10, 2, 20, 3, 30, 4, 40]); // 2x2 spatial, k = 2
    expect(Array.from(classMap(cam, 2, 2, 2, 1))).toEqual([10, 20, 30, 40]);
  });
});

describe("localize", () => {
  it("finds an upper-left defect and estimates its area from contrast", () => {
    const rgba = fabricWithSquare(30, 30, 60, 60); // 30x30 px = 1.79 % of the image
    const loc = localize(peakMap(1.4, 1.4), 7, 7, rgba, SIZE);
    expect(loc.zone).toBe("upper-left");
    expect(loc.method).toBe("cam+contrast");
    expect(loc.areaPct).toBeGreaterThan(1.5);
    expect(loc.areaPct).toBeLessThan(2.1);
    expect(loc.bbox.x0).toBeCloseTo(30 / SIZE, 2);
    expect(loc.bbox.x1).toBeCloseTo(60 / SIZE, 2);
  });

  it("is not fooled by uneven lighting and vignetting", () => {
    const rgba = fabricWithSquare(150, 150, 180, 180);
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const i = (y * SIZE + x) * 4;
        if (rgba[i] === 20) continue; // keep the defect dark
        const u = x / SIZE - 0.5;
        const v = y / SIZE - 0.5;
        const light = 0.75 + 0.5 * (x / SIZE) - 0.5 * (u * u + v * v); // 80+ levels of gradient plus vignette
        for (let ch = 0; ch < 3; ch++) rgba[i + ch] = Math.min(255, rgba[i + ch] * light);
      }
    }
    const loc = localize(peakMap(4.9, 4.9), 7, 7, rgba, SIZE);
    expect(loc.zone).toBe("lower-right");
    expect(loc.areaPct).toBeGreaterThan(1.5);
    expect(loc.areaPct).toBeLessThan(2.3); // true area: 30 x 30 px = 1.79 %
  });

  it("falls back to the activation area when the defect has no colour contrast", () => {
    const rgba = fabricWithSquare(0, 0, 0, 0); // no defect pixels at all
    const loc = localize(peakMap(5, 5), 7, 7, rgba, SIZE);
    expect(loc.method).toBe("cam");
    expect(loc.zone).toBe("lower-right");
    expect(loc.areaPct).toBeGreaterThan(0);
  });

  it("returns an empty localisation for a flat map", () => {
    const loc = localize(new Float32Array(49), 7, 7, fabricWithSquare(0, 0, 0, 0), SIZE);
    expect(loc.areaPct).toBe(0);
    expect(loc.zone).toBe("center");
  });
});
