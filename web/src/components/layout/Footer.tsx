import { ShieldCheck } from "lucide-react";

import { SITE } from "../../config/site";
import { GitHubIcon, LinkedInIcon, YouTubeIcon } from "./icons";
import { Logo } from "./Logo";

export function Footer() {
  return (
    <footer className="dark relative border-t border-line bg-surface">
      <div className="absolute inset-x-0 -top-px h-0.5 opacity-70" style={{ background: "var(--grad-line)" }} aria-hidden="true" />
      <div className="container-page flex flex-col gap-8 py-12 md:flex-row md:items-start md:justify-between">
        <div className="max-w-md space-y-3">
          <Logo />
          <p className="text-sm leading-relaxed text-ink-2">{SITE.tagline}.</p>
          <p className="flex items-center gap-2 text-sm text-ink-2">
            <ShieldCheck className="size-4 text-[var(--sev-none-ink)]" aria-hidden="true" />
            Images are analysed on your device and never uploaded.
          </p>
        </div>
        <div className="flex flex-col gap-3 text-sm md:items-end">
          <div className="flex gap-2">
            {SITE.githubUrl && (
              <a href={SITE.githubUrl} target="_blank" rel="noreferrer" className="btn btn-secondary">
                <GitHubIcon className="size-4" /> Source code
              </a>
            )}
            {SITE.linkedinUrl && (
              <a href={SITE.linkedinUrl} target="_blank" rel="noreferrer" className="btn btn-secondary" aria-label="LinkedIn profile">
                <LinkedInIcon className="size-4" />
              </a>
            )}
            {SITE.youtubeUrl && (
              <a href={SITE.youtubeUrl} target="_blank" rel="noreferrer" className="btn btn-secondary" aria-label="Demo video on YouTube">
                <YouTubeIcon className="size-4" />
              </a>
            )}
          </div>
          <p className="text-muted">
            © {SITE.year} {SITE.author || SITE.name}. Built with TensorFlow, ONNX Runtime Web and React.
          </p>
        </div>
      </div>
    </footer>
  );
}
