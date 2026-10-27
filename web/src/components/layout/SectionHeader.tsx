import { motion } from "motion/react";
import type { ReactNode } from "react";

export function SectionHeader({ eyebrow, title, children, id }: { eyebrow: string; title: string; children?: ReactNode; id?: string }) {
  return (
    <motion.div
      className="max-w-3xl"
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.45 }}
    >
      <p className="eyebrow">{eyebrow}</p>
      <h2 id={id} className="section-title mt-2">
        {title}
      </h2>
      {children && <div className="mt-4 text-lg leading-relaxed text-ink-2">{children}</div>}
    </motion.div>
  );
}
