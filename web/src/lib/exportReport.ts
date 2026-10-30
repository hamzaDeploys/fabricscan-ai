/** "Copy report", "Download PNG" and "Download PDF" for an analysis result. */
import type { DefectReport } from "./descriptionEngine";
import { areaText, pct } from "./format";

export async function copyText(text: string): Promise<void> {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  const ok = document.execCommand("copy");
  area.remove();
  if (!ok) throw new Error("Copying is blocked in this browser.");
}

let logoPromise: Promise<HTMLImageElement | null> | null = null;

/** The site logo for report headers; resolves to null if it cannot be loaded (the report is drawn without it). */
function loadLogo(): Promise<HTMLImageElement | null> {
  logoPromise ??= new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = `${import.meta.env.BASE_URL}logo.png`;
  });
  return logoPromise;
}

const SEVERITY_COLORS: Record<string, string> = {
  None: "#0ca30c",
  Low: "#fab219",
  Medium: "#ec835a",
  High: "#d03b3b",
};

function wrap(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  const lines: string[] = [];
  let line = "";
  for (const word of text.split(/\s+/)) {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = test;
    }
  }
  if (line) lines.push(line);
  return lines;
}

/** A shareable 1400 px wide report card: annotated image on the left, the written report on the right. */
export async function renderReportCanvas(report: DefectReport, annotated: HTMLCanvasElement): Promise<HTMLCanvasElement> {
  await document.fonts?.ready;
  const W = 1400;
  const pad = 48;
  const imgSize = 560;
  const textX = pad + imgSize + 48;
  const textW = W - textX - pad;
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d")!;
  const sans = '"Inter Variable", "Segoe UI", system-ui, sans-serif';
  const display = '"Space Grotesk Variable", "Inter Variable", "Segoe UI", sans-serif';

  // Measure the text block first to size the canvas.
  ctx.font = `20px ${sans}`;
  const summary = wrap(ctx, report.summary, textW);
  const note = wrap(ctx, report.confidenceNote, textW);
  const causes = report.causes.slice(0, 3).flatMap((cause) => wrap(ctx, `• ${cause}`, textW));
  const textHeight = 210 + (summary.length + note.length) * 30 + (causes.length ? 60 + causes.length * 28 : 0) + 140;
  const H = Math.max(imgSize + pad * 2 + 70, textHeight + pad);
  c.width = W;
  c.height = H;

  ctx.fillStyle = "#f5f8fd";
  ctx.fillRect(0, 0, W, H);
  const band = ctx.createLinearGradient(0, 0, W, 0);
  band.addColorStop(0, "#1466f2");
  band.addColorStop(1, "#00d0e0");
  ctx.fillStyle = band;
  ctx.fillRect(0, 0, W, 8);
  ctx.drawImage(annotated, pad, pad + 16, imgSize, imgSize);
  ctx.strokeStyle = "#dbe3f0";
  ctx.lineWidth = 2;
  ctx.strokeRect(pad, pad + 16, imgSize, imgSize);

  let y = pad + 40;
  const logo = await loadLogo();
  let eyebrowX = textX;
  if (logo) {
    ctx.drawImage(logo, textX, y - 30, 40, 40);
    eyebrowX = textX + 52;
  }
  ctx.fillStyle = "#08807a";
  ctx.font = `600 16px ${display}`;
  ctx.fillText("FABRICSCAN AI · INSPECTION REPORT", eyebrowX, y - 4);
  y += 54;
  ctx.fillStyle = "#0b1533";
  ctx.font = `700 42px ${display}`;
  ctx.fillText(report.label, textX, y);
  y += 44;

  // Severity pill + confidence
  const pill = `Severity: ${report.severity}`;
  ctx.font = `600 18px ${sans}`;
  const pw = ctx.measureText(pill).width + 36;
  ctx.fillStyle = SEVERITY_COLORS[report.severity];
  ctx.beginPath();
  ctx.roundRect(textX, y - 24, pw, 36, 18);
  ctx.fill();
  ctx.fillStyle = report.severity === "Low" ? "#0b1533" : "#ffffff";
  ctx.fillText(pill, textX + 18, y);
  ctx.fillStyle = "#3c4a6b";
  ctx.font = `500 18px ${sans}`;
  ctx.fillText(`Confidence ${pct(report.confidence)}`, textX + pw + 18, y);
  y += 44;
  if (report.isDefect && report.zoneLabel && report.areaPct !== null) {
    ctx.fillText(`Location: ${report.zoneLabel} · Area ≈ ${areaText(report.areaPct)}`, textX, y);
    y += 40;
  }

  ctx.fillStyle = "#0b1533";
  ctx.font = `20px ${sans}`;
  for (const line of summary) {
    ctx.fillText(line, textX, y);
    y += 30;
  }
  ctx.fillStyle = "#3c4a6b";
  for (const line of note) {
    ctx.fillText(line, textX, y);
    y += 30;
  }
  if (causes.length) {
    y += 18;
    ctx.fillStyle = "#0b1533";
    ctx.font = `600 18px ${sans}`;
    ctx.fillText("Possible causes", textX, y);
    y += 32;
    ctx.font = `18px ${sans}`;
    ctx.fillStyle = "#3c4a6b";
    for (const line of causes) {
      ctx.fillText(line, textX, y);
      y += 28;
    }
  }

  ctx.fillStyle = "#5d6a88";
  ctx.font = `15px ${sans}`;
  const footer = `${report.createdAt} · ${report.modelName}${report.demo ? " · DEMO model (synthetic training data)" : ""} · runs in the browser`;
  ctx.fillText(footer, pad, H - 28);
  return c;
}

function download(url: string, filename: string): void {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

export function fileStem(report: DefectReport): string {
  return `fabricscan-${report.classKey}-${report.createdAt.replace(/[^0-9]/g, "").slice(0, 12)}`;
}

export async function downloadPng(report: DefectReport, annotated: HTMLCanvasElement): Promise<void> {
  const c = await renderReportCanvas(report, annotated);
  const blob = await new Promise<Blob | null>((resolve) => c.toBlob(resolve, "image/png"));
  if (!blob) throw new Error("Could not create the PNG.");
  const url = URL.createObjectURL(blob);
  download(url, `${fileStem(report)}.png`);
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export async function downloadPdf(report: DefectReport, annotated: HTMLCanvasElement): Promise<void> {
  const { jsPDF } = await import("jspdf"); // loaded only when needed
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const left = 18;
  const width = 174;
  let y = 22;
  doc.setFillColor(52, 70, 196);
  doc.rect(0, 0, 210, 4, "F");
  const logo = await loadLogo();
  let eyebrowX = left;
  if (logo) {
    const c = document.createElement("canvas");
    c.width = c.height = 256;
    c.getContext("2d")!.drawImage(logo, 0, 0, 256, 256);
    doc.addImage(c.toDataURL("image/png"), "PNG", left, y - 7.5, 11, 11);
    eyebrowX = left + 14;
  }
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(201, 130, 31);
  doc.text("FABRICSCAN AI · INSPECTION REPORT", eyebrowX, y - 0.5);
  y += 10;
  doc.setTextColor(18, 20, 28);
  doc.setFontSize(22);
  doc.text(report.label, left, y);
  y += 8;
  doc.setFont("helvetica", "normal");
  doc.setFontSize(11);
  doc.setTextColor(68, 73, 89);
  // jsPDF's built-in fonts only cover Latin-1; a character such as "≈" garbles the whole line, so avoid them.
  const facts = [`Severity: ${report.severity}`, `Confidence: ${pct(report.confidence)}`];
  if (report.isDefect && report.zoneLabel && report.areaPct !== null) {
    facts.push(`Location: ${report.zoneLabel}`, `Area: approx. ${areaText(report.areaPct)}`);
  }
  const factLines = doc.splitTextToSize(facts.join("  ·  "), width) as string[];
  doc.text(factLines, left, y);
  y += factLines.length * 5 + 2;
  doc.addImage(annotated.toDataURL("image/jpeg", 0.92), "JPEG", left, y, 90, 90);
  y += 98;

  const block = (title: string, body: string[]) => {
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(18, 20, 28);
    doc.text(title, left, y);
    y += 6;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10.5);
    doc.setTextColor(68, 73, 89);
    for (const paragraph of body) {
      const lines = doc.splitTextToSize(paragraph, width) as string[];
      doc.text(lines, left, y);
      y += lines.length * 5 + 1.5;
    }
    y += 4;
  };
  block("Summary", [report.summary, report.confidenceNote]);
  if (report.causes.length) block("Possible causes", report.causes.map((cause) => `• ${cause}`));
  block("Recommended action", [report.action]);
  block("All classes", [report.ranked.map((r) => `${r.label}: ${pct(r.prob)}`).join("   ·   ")]);

  doc.setFontSize(8.5);
  doc.setTextColor(100, 106, 123);
  doc.text(
    `${report.createdAt} · ${report.modelName}${report.demo ? " · DEMO model (synthetic training data)" : ""} · Generated in the browser by FabricScan AI`,
    left,
    287,
  );
  doc.save(`${fileStem(report)}.pdf`);
}
