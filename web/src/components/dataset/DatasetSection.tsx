import { ArrowUpRight, Download, FileSpreadsheet, FileText, Folder, FolderArchive, type LucideIcon } from "lucide-react";
import { motion } from "motion/react";

import { DOWNLOADS } from "../../config/site";
import { useSiteData } from "../../hooks/useSiteData";
import { DEFECTS, type ClassKey } from "../../lib/defectKnowledge";
import { SectionHeader } from "../layout/SectionHeader";
import { SHORT_NAMES } from "../performance/charts";

const ORDER: ClassKey[] = ["good", "hole", "stain", "broken_thread"];
const SPLITS = ["train", "val", "test"];

const reveal = {
  initial: { opacity: 0, y: 12 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-40px" },
};

/** One line of the "inside the ZIP" file tree. */
function TreeRow({ icon: Icon, name, note, depth = 0 }: { icon: LucideIcon; name: string; note?: string; depth?: number }) {
  return (
    <li className="flex items-start gap-2" style={{ paddingLeft: `${depth * 1.25}rem` }}>
      <Icon className="mt-0.5 size-4 shrink-0 text-accent" aria-hidden="true" />
      <span className="min-w-0 font-medium break-words text-ink">{name}</span>
      {note && <span className="ml-auto shrink-0 pl-3 text-right font-sans text-xs text-muted">{note}</span>}
    </li>
  );
}

export function DatasetSection() {
  const { dataset, download, samples } = useSiteData();
  const total = download?.images ?? dataset?.total_images;
  const perSplit = (s: string) => (download ? Object.values(download.splits[s] ?? {}).reduce((a, b) => a + b, 0) : null);
  const sizeMb = download ? (download.size_bytes / 1e6).toFixed(1) : null;
  const facts: [string, string][] = [
    ["Images", total !== undefined ? total.toLocaleString("en-US") : "–"],
    ["Classes", String(ORDER.length)],
    ["Image size", `${download?.image_size ?? 256} px`],
    ["Sources", download ? String(download.sources.length) : "–"],
  ];

  return (
    <section id="dataset" aria-labelledby="dataset-title" className="border-t border-line bg-surface-3/40 py-20 sm:py-24">
      <div className="container-page">
        <SectionHeader eyebrow="Dataset" title="Four classes of real fabric photos" id="dataset-title">
          Real photos from five public Kaggle datasets: close-ups of woven and knitted fabric, garment seams, stained
          clothes and clean clothes as people wear them. Every image was cleaned, de-duplicated and split by its source image, so the test set holds only
          images the model has never seen.
        </SectionHeader>

        <motion.div {...reveal} transition={{ duration: 0.45 }} className="card mt-10 overflow-hidden" data-testid="dataset-download-panel">
          <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
            <div className="relative p-6 sm:p-8">
              <div className="weave pointer-events-none absolute inset-0 opacity-60" aria-hidden="true" />
              <div className="relative">
                <div className="flex items-start gap-4">
                  <span className="flex size-12 shrink-0 items-center justify-center rounded-2xl bg-[image:var(--grad-btn)] text-[var(--btn-ink)] shadow-md">
                    <FolderArchive className="size-6" aria-hidden="true" />
                  </span>
                  <div className="min-w-0">
                    <h3 className="font-display text-xl font-semibold text-ink sm:text-2xl">Download the dataset</h3>
                    <p className="mt-1 leading-relaxed text-ink-2">
                      The exact images both models were trained and tested on, ready to use: one folder per class in
                      train, val and test.
                    </p>
                  </div>
                </div>

                <dl className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
                  {facts.map(([k, v]) => (
                    <div key={k} className="rounded-xl border border-line bg-surface px-4 py-3">
                      <dt className="text-[11px] font-medium text-muted">{k}</dt>
                      <dd className="mt-0.5 font-display text-xl font-bold text-ink">{v}</dd>
                    </div>
                  ))}
                </dl>

                <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-4">
                  <a href={DOWNLOADS.dataset} download className="btn btn-primary shrink-0 self-start px-6 py-3 text-base whitespace-nowrap shadow-md" data-testid="dataset-download">
                    <Download className="size-5" aria-hidden="true" />
                    Download ZIP{sizeMb ? ` · ${sizeMb} MB` : ""}
                  </a>
                  <p className="text-xs leading-relaxed text-muted">
                    Free for study and research. Each image keeps the licence of its source (listed below and in
                    README.txt).
                  </p>
                </div>
              </div>
            </div>

            <div className="border-t border-line bg-surface-3/60 p-6 sm:p-8 lg:border-t-0 lg:border-l">
              <p className="eyebrow">Inside the ZIP</p>
              <ul className="mt-4 space-y-2.5 font-mono text-[13px]" aria-label="Files in the ZIP">
                <TreeRow icon={Folder} name="fabricscan-dataset/" />
                {SPLITS.map((s) => {
                  const n = perSplit(s);
                  return <TreeRow key={s} icon={Folder} name={`${s}/`} note={n !== null ? `${n} images` : undefined} depth={1} />;
                })}
                <li className="pl-10 font-sans text-xs leading-relaxed text-muted">
                  each with <span className="font-mono text-ink-2">{ORDER.join("/ ")}/</span>
                </li>
                <TreeRow icon={FileSpreadsheet} name="SOURCES.csv" note="origin + licence" depth={1} />
                <TreeRow icon={FileText} name="README.txt" note="counts + licences" depth={1} />
              </ul>
            </div>
          </div>
        </motion.div>

        {download && download.sources.length > 0 && (
          <div className="mt-12">
            <h3 className="font-display text-lg font-semibold text-ink">Sources and licences</h3>
            <ul className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2" aria-label="Image sources">
              {download.sources.map((s, i) => (
                <motion.li key={s.ref || s.title} {...reveal} transition={{ duration: 0.4, delay: (i % 2) * 0.06 }} className="card card-hover flex flex-col p-5">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="font-display font-semibold text-ink">{s.title}</p>
                      {s.ref && <p className="truncate font-mono text-xs text-muted">kaggle.com/datasets/{s.ref}</p>}
                    </div>
                    {s.licence && <span className="chip shrink-0 border-accent/40 text-ink">{s.licence}</span>}
                  </div>
                  <ul className="mt-4 flex flex-wrap gap-1.5" aria-label={`Images from ${s.title} per class`}>
                    {Object.entries(s.classes).map(([c, n]) => (
                      <li key={c} className="chip">
                        <span className="font-semibold text-ink">{n}</span> {SHORT_NAMES[c] ?? c}
                      </li>
                    ))}
                  </ul>
                  <div className="mt-auto flex items-center justify-between gap-3 pt-4">
                    <span className="text-sm text-ink-2">
                      <span className="font-display font-bold text-ink">{s.images}</span> images used
                    </span>
                    {s.url && (
                      <a href={s.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 rounded text-sm font-semibold text-brand hover:underline">
                        View on Kaggle <ArrowUpRight className="size-4" aria-hidden="true" />
                      </a>
                    )}
                  </div>
                </motion.li>
              ))}
            </ul>
          </div>
        )}

        <h3 className="mt-12 font-display text-lg font-semibold text-ink">The four classes</h3>
        <ul className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {ORDER.map((key, i) => {
            const info = DEFECTS[key];
            const sample = samples.find((s) => s.class === key);
            const count = dataset?.totals[key];
            return (
              <motion.li key={key} {...reveal} transition={{ duration: 0.4, delay: (i % 4) * 0.06 }} className="card card-hover group overflow-hidden">
                <div className="aspect-[4/3] overflow-hidden bg-surface-3">
                  {sample && <img src={sample.file} alt={`Example of the class ${info.label}`} loading="lazy" className="size-full object-cover transition-transform duration-500 group-hover:scale-105" />}
                </div>
                <div className="p-5">
                  <h4 className="font-display text-lg font-semibold text-ink">{info.label}</h4>
                  <p className="mt-1.5 text-sm leading-relaxed text-ink-2">{info.definition}</p>
                  {count !== undefined && <p className="mt-3 text-xs font-medium text-muted">{count} images</p>}
                </div>
              </motion.li>
            );
          })}
        </ul>
      </div>
    </section>
  );
}
