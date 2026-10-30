import { Menu, Moon, Sun, X } from "lucide-react";
import { motion, useScroll, useSpring } from "motion/react";
import { useEffect, useState } from "react";

import { SITE } from "../../config/site";
import { useTheme } from "../../hooks/useTheme";
import { GitHubIcon } from "./icons";
import { Logo } from "./Logo";

const LINKS = [
  ["#demo", "Live demo"],
  ["#how", "How it works"],
  ["#performance", "Performance"],
  ["#dataset", "Dataset"],
  ["#links", "Links"],
  ["#about", "About"],
] as const;

export function Navbar() {
  const { dark, toggle } = useTheme();
  const [open, setOpen] = useState(false);
  const { scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, { stiffness: 200, damping: 30, restDelta: 0.001 });

  useEffect(() => {
    if (!open) return;
    const close = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [open]);

  return (
    <header className="sticky top-0 z-40 border-b border-line/70 bg-surface/90 backdrop-blur-md">
      <motion.div className="absolute inset-x-0 bottom-0 h-0.5 origin-left" style={{ scaleX: progress, background: "var(--grad-line)" }} aria-hidden="true" />
      <nav className="container-page flex h-16 items-center justify-between gap-4" aria-label="Main">
        <a href="#top" className="rounded-lg" aria-label="FabricScan AI, back to top">
          <Logo />
        </a>
        <ul className="hidden items-center gap-1 md:flex">
          {LINKS.map(([href, label]) => (
            <li key={href}>
              <a href={href} className="relative rounded-lg px-3 py-2 text-sm font-medium text-ink-2 transition-colors after:absolute after:inset-x-3 after:bottom-1 after:h-0.5 after:origin-left after:scale-x-0 after:rounded-full after:bg-[image:var(--grad-line)] after:transition-transform after:duration-300 hover:text-ink hover:after:scale-x-100">
                {label}
              </a>
            </li>
          ))}
        </ul>
        <div className="flex items-center gap-1">
          {SITE.githubUrl && (
            <a href={SITE.githubUrl} target="_blank" rel="noreferrer" className="btn btn-ghost px-2.5" aria-label="Source code on GitHub">
              <GitHubIcon className="size-5" />
            </a>
          )}
          <button type="button" onClick={toggle} className="btn btn-ghost px-2.5" aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}>
            {dark ? <Sun className="size-5" /> : <Moon className="size-5" />}
          </button>
          <button
            type="button"
            className="btn btn-ghost px-2.5 md:hidden"
            aria-expanded={open}
            aria-controls="mobile-menu"
            aria-label={open ? "Close menu" : "Open menu"}
            onClick={() => setOpen((o) => !o)}
          >
            {open ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>
        </div>
      </nav>
      {open && (
        <ul id="mobile-menu" className="container-page border-t border-line pb-3 md:hidden">
          {LINKS.map(([href, label]) => (
            <li key={href}>
              <a href={href} onClick={() => setOpen(false)} className="block rounded-lg px-3 py-3 text-base font-medium text-ink-2 hover:bg-surface-3 hover:text-ink">
                {label}
              </a>
            </li>
          ))}
        </ul>
      )}
    </header>
  );
}
