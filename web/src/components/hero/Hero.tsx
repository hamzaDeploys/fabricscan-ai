import { ArrowRight, Cpu, Layers, ShieldCheck, Sparkles } from "lucide-react";
import { motion } from "motion/react";

import { useSiteData } from "../../hooks/useSiteData";

const CHIPS = [
  { icon: Layers, text: "6 fabric classes" },
  { icon: Cpu, text: "Runs 100% in your browser" },
  { icon: ShieldCheck, text: "Photos never leave your device" },
  { icon: Sparkles, text: "Automatic written report" },
];

function HeroVisual() {
  const { samples } = useSiteData();
  const sample = samples.find((s) => s.class === "hole") ?? samples.find((s) => s.class !== "good") ?? samples[0];
  return (
    <motion.div
      className="relative mx-auto w-full max-w-md lg:max-w-none"
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6, delay: 0.15 }}
      aria-hidden="true"
    >
      <div className="card relative overflow-hidden p-3">
        <div className="relative aspect-square overflow-hidden rounded-xl bg-surface-3">
          {sample ? (
            <img src={sample.file} alt="" className="size-full object-cover" />
          ) : (
            <div className="weave size-full" />
          )}
          <div className="scan-line absolute inset-x-0 top-0 h-10 bg-gradient-to-b from-transparent via-[color-mix(in_oklab,var(--accent-bright)_55%,transparent)] to-transparent" />
          <div className="absolute inset-0 rounded-xl ring-1 ring-black/5 ring-inset" />
        </div>
      </div>
      <div className="card absolute -bottom-6 -left-4 w-[78%] p-4 shadow-xl sm:-left-8">
        <p className="text-[11px] font-semibold tracking-wider text-accent uppercase">Example output</p>
        <p className="mt-1.5 text-sm leading-relaxed text-ink">
          “A hole-type defect was detected near the upper-left region, covering about 1.4% of the visible fabric area.
          Severity: High. Recommended action: reject or repair.”
        </p>
      </div>
    </motion.div>
  );
}

export function Hero() {
  return (
    <section id="top" className="relative overflow-hidden">
      <div className="weave pointer-events-none absolute inset-0 [mask-image:radial-gradient(ellipse_at_top,black,transparent_70%)]" aria-hidden="true" />
      <div className="pointer-events-none absolute -top-48 left-1/2 h-[520px] w-[900px] -translate-x-1/2 rounded-full bg-brand/15 blur-3xl" aria-hidden="true" />
      <div className="container-page relative grid grid-cols-1 items-center gap-14 pt-14 pb-24 md:pt-20 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)] lg:pb-28">
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
          <p className="eyebrow">AI textile inspection</p>
          <h1 className="mt-3 font-display text-4xl leading-[1.08] font-bold tracking-tight text-balance text-ink sm:text-5xl lg:text-[3.4rem]">
            Spot fabric defects in seconds, <span className="text-brand">with the report written for you.</span>
          </h1>
          <p className="mt-6 max-w-xl text-lg leading-relaxed text-ink-2">
            Upload or photograph a fabric. A neural network sorts it into four classes, highlights where the defect is,
            and writes a severity-graded inspection note with likely causes and the recommended action.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a href="#demo" className="btn btn-primary px-5 py-3 text-base">
              Try the live demo <ArrowRight className="size-4" aria-hidden="true" />
            </a>
            <a href="#how" className="btn btn-secondary px-5 py-3 text-base">
              How it works
            </a>
          </div>
          <ul className="mt-9 flex flex-wrap gap-2">
            {CHIPS.map(({ icon: Icon, text }) => (
              <li key={text} className="chip">
                <Icon className="size-3.5 text-brand" aria-hidden="true" />
                {text}
              </li>
            ))}
          </ul>
        </motion.div>
        <HeroVisual />
      </div>
    </section>
  );
}
