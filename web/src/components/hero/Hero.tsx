import { ArrowRight, OctagonAlert } from "lucide-react";
import { motion } from "motion/react";

import { useSiteData } from "../../hooks/useSiteData";
import { pct } from "../../lib/format";

/** What this app really reports for samples/broken_thread-1.jpg with MobileNetV2; the images are its own output. */
const EXAMPLE = { label: "Broken Thread / Slub", confidence: 0.997, severity: "High", location: "Center", area: "≈ 6.0%" };

function HeroVisual() {
  return (
    <motion.figure
      className="scan-frame relative mx-auto w-full max-w-md lg:mr-0 lg:ml-auto"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.15 }}
    >
      <i />
      <i />
      <i />
      <i />
      <div className="card overflow-hidden shadow-2xl shadow-black/40">
        <div className="relative aspect-[5/4] overflow-hidden bg-surface-3">
          <img src="/hero/photo.jpg" alt="Dark fabric with a broken thread" width={520} height={416} className="absolute inset-0 size-full object-cover" />
          <img src="/hero/spot.jpg" alt="" className="hero-step-2 absolute inset-0 size-full object-cover" />
          <img src="/hero/heat.jpg" alt="" className="hero-step-3 absolute inset-0 size-full object-cover" />
          <div className="hero-beam absolute inset-x-0 top-0 h-[14%] bg-gradient-to-b from-transparent via-[color-mix(in_oklab,var(--accent-bright)_45%,transparent)] to-transparent" />
          <span className="absolute top-3 left-3 rounded-lg bg-black/70 px-2.5 py-1 text-xs font-medium text-white" aria-hidden="true">
            <span className="hero-step-1 absolute inset-0 grid place-items-center">Scanning…</span>
            <span className="hero-step-2 absolute inset-0 grid place-items-center">Defect found</span>
            <span className="hero-step-3 absolute inset-0 grid place-items-center">Heatmap</span>
            <span className="invisible">Defect found</span>
          </span>
        </div>
        <figcaption className="p-4 sm:p-5">
          <div className="flex items-end justify-between gap-3">
            <div className="min-w-0">
              <p className="flex items-center gap-1.5 text-[11px] font-semibold tracking-wider text-[var(--sev-high-ink)] uppercase">
                <OctagonAlert className="size-3.5" aria-hidden="true" /> Defect detected
              </p>
              <p className="mt-1 font-display text-xl leading-tight font-bold text-ink">{EXAMPLE.label}</p>
            </div>
            <p className="shrink-0 text-right">
              <span className="block font-display text-2xl leading-none font-bold text-ink">{pct(EXAMPLE.confidence)}</span>
              <span className="mt-1 block text-[11px] text-muted">confidence</span>
            </p>
          </div>
          <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-surface-3" aria-hidden="true">
            <motion.div
              className="h-full origin-left rounded-full"
              style={{ width: `${EXAMPLE.confidence * 100}%`, background: "var(--grad-line)" }}
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              transition={{ duration: 1, delay: 0.5, ease: "easeOut" }}
            />
          </div>
          <dl className="mt-3.5 grid grid-cols-3 gap-2 text-sm">
            {[
              ["Severity", EXAMPLE.severity],
              ["Location", EXAMPLE.location],
              ["Affected area", EXAMPLE.area],
            ].map(([term, value]) => (
              <div key={term}>
                <dt className="text-[11px] text-muted">{term}</dt>
                <dd className="mt-0.5 font-semibold text-ink">{value}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 border-t border-line pt-2.5 text-[11px] text-muted">Real output of MobileNetV2 on a test photo from StitchingNet (CC BY-NC 4.0).</p>
        </figcaption>
      </div>
    </motion.figure>
  );
}

export function Hero() {
  const { metrics, dataset } = useSiteData();
  const best = metrics?.models[metrics.best_model];
  const stats = [
    [best ? pct(best.test.accuracy) : "—", "test accuracy of the better model"],
    [metrics ? String(Object.keys(metrics.models).length) : "—", "trained models you can compare"],
    [dataset ? dataset.total_images.toLocaleString("en-US") : "—", `real images in ${dataset?.class_names.length ?? "four"} classes`],
    ["0", "photos uploaded: it runs in your browser"],
  ];
  return (
    <section id="top" className="hero-bg dark relative overflow-hidden">
      <div className="weave pointer-events-none absolute inset-0 opacity-60 [mask-image:radial-gradient(ellipse_at_top,black,transparent_75%)]" aria-hidden="true" />
      <div className="aurora -top-52 left-[2%] size-[620px] bg-[radial-gradient(closest-side,rgba(20,102,242,0.42),transparent)]" aria-hidden="true" />
      <div className="aurora -right-40 bottom-[-180px] size-[560px] bg-[radial-gradient(closest-side,rgba(0,224,208,0.22),transparent)] [animation-delay:-12s]" aria-hidden="true" />
      <div className="container-page relative grid grid-cols-1 items-center gap-12 pt-14 pb-16 md:pt-20 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)] lg:gap-14 lg:pb-24">
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
          <p className="eyebrow">AI textile inspection</p>
          <h1 className="mt-4 font-display text-4xl leading-[1.08] font-bold tracking-tight text-balance text-ink sm:text-5xl lg:text-[3.5rem]">
            Spot fabric defects in seconds, <span className="text-gradient">with the report written for you.</span>
          </h1>
          <p className="mt-6 max-w-xl text-lg leading-relaxed text-ink-2">
            Upload or photograph a fabric and choose one of two trained neural networks. It sorts the photo into four
            classes, shows where the defect is and writes a severity-graded inspection note with likely causes and the
            recommended action.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a href="#demo" className="btn btn-primary px-5 py-3 text-base">
              Try the live demo <ArrowRight className="size-4" aria-hidden="true" />
            </a>
            <a href="#how" className="btn btn-secondary px-5 py-3 text-base">
              How it works
            </a>
          </div>
          <dl className="mt-10 grid max-w-2xl grid-cols-2 gap-x-6 gap-y-5 border-t border-line pt-6 sm:grid-cols-4">
            {stats.map(([value, label], i) => (
              <motion.div
                key={label}
                className="flex flex-col-reverse"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.3 + i * 0.08 }}
              >
                <dt className="mt-1.5 text-xs leading-snug text-muted">{label}</dt>
                <dd className="font-display text-2xl leading-none font-bold text-ink">{value}</dd>
              </motion.div>
            ))}
          </dl>
        </motion.div>
        <HeroVisual />
      </div>
    </section>
  );
}
