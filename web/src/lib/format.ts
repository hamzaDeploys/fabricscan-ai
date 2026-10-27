export const pct = (p: number, digits = 1): string => `${(p * 100).toFixed(digits)}%`;

/** Area as a human phrase: "less than 0.1%", "4.2%", "23%". */
export function areaText(areaPct: number): string {
  if (areaPct < 0.1) return "less than 0.1%";
  if (areaPct < 10) return `${areaPct.toFixed(1)}%`;
  return `${Math.round(areaPct)}%`;
}

export const fmtMs = (ms: number): string => (ms < 10 ? `${ms.toFixed(1)} ms` : `${Math.round(ms)} ms`);
export const fmtMB = (mb: number): string => `${mb.toFixed(1)} MB`;

export function compact(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} K`;
  return String(n);
}

export function capitalize(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

export function joinOr(items: string[]): string {
  if (items.length <= 1) return items[0] ?? "";
  return `${items.slice(0, -1).join(", ")} or ${items[items.length - 1]}`;
}

export function formatDateTime(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
