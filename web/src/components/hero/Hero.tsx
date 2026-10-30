import { ArrowDown, ArrowUpRight, OctagonAlert } from "lucide-react";
import { motion } from "motion/react";

import { useSiteData } from "../../hooks/useSiteData";
import { pct } from "../../lib/format";

/** What this app really reports with MobileNetV2 for the close-up in public/hero/photo.webp. */
const EXAMPLE = { label: "Hole", confidence: 0.818, severity: "High", location: "center" };

function ModelWindow() {
  return (
    <motion.figure
      className="mx-auto w-full max-w-lg lg:mr-0 lg:ml-auto"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.15 }}
    >
      <div className="hero-tilt card overflow-hidden shadow-2xl shadow-black/50">
        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3 px-4 py-3">
          <span className="flex gap-1.5" aria-hidden="true">
            <i className="size-2 rounded-full bg-[#ff6b6b]" />
            <i className="size-2 rounded-full bg-[#ffc857]" />
            <i className="size-2 rounded-full bg-[#77e4bd]" />
          </span>
          <span className="font-mono text-xs text-ink-2">fabricscan.onnx</span>
          <span className="text-right text-[10px] font-semibold tracking-[0.14em] text-accent uppercase">Model output</span>
        </div>
        <div className="relative mx-3 aspect-[6/5] overflow-hidden rounded-xl bg-surface-3">
          <img src="/hero/photo.webp" alt="Close-up of dark woven fabric with a hole" width={800} height={667} className="absolute inset-0 size-full object-cover" />
          <div className="scan-corners absolute inset-x-[6%] top-[7%] bottom-[23%]" aria-hidden="true">
            <i />
            <i />
            <i />
            <i />
          </div>
          <div className="hero-beam absolute inset-x-0 top-0 h-[12.5%] bg-gradient-to-b from-transparent via-[color-mix(in_oklab,var(--accent-bright)_22%,transparent)] to-transparent" aria-hidden="true" />
          <figcaption className="absolute inset-x-2.5 bottom-2.5 flex items-center gap-3 rounded-xl border border-white/10 bg-[#0d1120]/90 px-3 py-2.5">
            <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-[color-mix(in_oklab,var(--sev-high)_20%,transparent)] text-[var(--sev-high-ink)]">
              <OctagonAlert className="size-[18px]" aria-hidden="true" />
            </span>
            <span className="min-w-0">
              <span className="block font-display text-sm font-semibold text-ink">{EXAMPLE.label}</span>
              <span className="block text-xs text-ink-2">
                {pct(EXAMPLE.confidence)} confidence · {EXAMPLE.severity} severity · {EXAMPLE.location}
              </span>
            </span>
          </figcaption>
        </div>
        <div className="px-4 py-3">
          <p className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-xs text-ink-2">
            <span className="flex items-center gap-2">
              <i className="size-1.5 rounded-full bg-accent" aria-hidden="true" /> Real output, MobileNetV2
            </span>
            <span>Preprocess → Predict → Report</span>
          </p>
          <p className="mt-1.5 text-[10.5px] leading-snug text-muted">
            Close-up crop of a photo found on the web (Google Images), outside the dataset. Original source not confirmed.
          </p>
        </div>
      </div>
    </motion.figure>
  );
}

export function Hero() {
  const { meta, metrics } = useSiteData();
  const best = metrics?.models[metrics.best_model];
  const stats = [
    [best ? best.name.split(" (")[0] : "—", "Default model"],
    [best ? pct(best.test.accuracy) : "—", "Test accuracy"],
    [metrics ? String(metrics.class_names.length) : "—", "Fabric classes"],
    [meta ? `${meta.input.size} × ${meta.input.size}` : "—", "Input image size"],
  ];
  return (
    <section id="top" className="hero-bg dark relative overflow-hidden">
      <div className="pointer-events-none absolute -top-72 -right-44 size-[680px] rounded-full border border-white/[0.05]" aria-hidden="true" />
      <div className="pointer-events-none absolute -top-96 -right-64 size-[980px] rounded-full border border-white/[0.04]" aria-hidden="true" />
      <div className="container-page relative grid grid-cols-1 items-center gap-12 pt-14 pb-16 md:pt-20 lg:grid-cols-[minmax(0,1.08fr)_minmax(0,0.92fr)] lg:gap-10 lg:pb-24">
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
          <p className="eyebrow">Deep learning · Fabric inspection</p>
          <h1 className="mt-5 font-display text-5xl leading-[1.04] font-semibold tracking-[-0.03em] text-ink sm:text-6xl lg:text-[4.6rem]">
            Spot fabric defects <span className="text-gradient whitespace-nowrap">in seconds.</span>
          </h1>
          <p className="mt-6 max-w-xl text-lg leading-relaxed text-ink-2">
            An image classification project that sorts a photo of fabric or clothing into four classes, shows where the
            defect is and writes the inspection report. Pick either of two trained models; everything runs in your browser.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a href="#demo" className="btn btn-primary h-13 rounded-2xl px-6 text-base">
              Try the live demo <ArrowDown className="size-4" aria-hidden="true" />
            </a>
            <a href="#links" className="btn btn-secondary h-13 rounded-2xl bg-transparent px-6 text-base">
              Project links <ArrowUpRight className="size-4" aria-hidden="true" />
            </a>
          </div>
          <dl className="mt-10 flex flex-wrap gap-x-10 gap-y-5">
            {stats.map(([value, label], i) => (
              <motion.div
                key={label}
                className="flex flex-col-reverse"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.3 + i * 0.08 }}
              >
                <dt className="mt-1.5 text-xs text-muted">{label}</dt>
                <dd className="font-display text-xl leading-none font-semibold text-ink">{value}</dd>
              </motion.div>
            ))}
          </dl>
        </motion.div>
        <ModelWindow />
      </div>
    </section>
  );
}
