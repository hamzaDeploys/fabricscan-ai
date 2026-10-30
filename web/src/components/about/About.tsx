import { BookOpen, GraduationCap, Rocket, Scale, UserRound, type LucideIcon } from "lucide-react";
import type { ComponentType, CSSProperties, SVGProps } from "react";

import { DOWNLOADS, SITE } from "../../config/site";
import { DEFECTS } from "../../lib/defectKnowledge";
import { GitHubIcon, LinkedInIcon, YouTubeIcon } from "../layout/icons";
import { LogoMark } from "../layout/Logo";
import { SectionHeader } from "../layout/SectionHeader";
import { useVideoLink } from "../layout/VideoDialog";

const HIGHLIGHTS = [
  { value: String(Object.keys(DEFECTS).length), label: "fabric classes" },
  { value: "2", label: "models compared" },
  { value: "100%", label: "runs in the browser" },
  { value: "0", label: "photos uploaded to a server" },
];

const STACK: [string, string[]][] = [
  ["Machine learning", ["Python", "TensorFlow / Keras", "MobileNetV2", "Grad-CAM"]],
  ["Deployment", ["ONNX", "ONNX Runtime Web", "Vercel"]],
  ["Website", ["React", "TypeScript", "Tailwind CSS", "Motion", "Vite"]],
];

const LIMITS = [
  "The model knows four classes from public datasets; other defect types (tears, colour bleeding), very different fabrics or lighting can lead to wrong answers.",
  "One label per image: a photo with two different defects is reported as the more likely one.",
  "Area and location come from a coarse activation map plus colour contrast, so they are estimates, not measurements.",
];

const FUTURE = [
  "Pixel-level segmentation for exact defect outlines and sizes",
  "More fabrics, defect types and a larger field-test set from real production lines",
  "A conveyor-camera mode that inspects video frames continuously",
];

interface Social {
  label: string;
  action: string;
  url: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  color: string; // hover colour (brand colour of the platform)
  video?: boolean; // plays in the popup instead of opening a new tab
}

const SOCIALS: Social[] = [
  { label: "GitHub", action: "Source code", url: SITE.githubUrl, icon: GitHubIcon, color: "var(--ink)" },
  { label: "LinkedIn", action: "Connect", url: SITE.linkedinUrl, icon: LinkedInIcon, color: "#0a66c2" },
  { label: "YouTube", action: "Play video", url: DOWNLOADS.video, icon: YouTubeIcon, color: "#ff0033", video: true },
];

function initials(name: string): string {
  return name
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0].toUpperCase())
    .join("");
}

/** A platform button. Without a URL it stays visible but inactive ("Coming soon"). */
function SocialButton({ label, action, url, icon: Icon, color, video }: Social) {
  const style = { "--social": color } as CSSProperties;
  const videoLink = useVideoLink();
  const body = (
    <>
      <Icon className="size-6 transition-colors" />
      <span className="mt-2 text-sm font-semibold text-ink">{label}</span>
      <span className="text-[11px] text-muted">{url ? action : "Coming soon"}</span>
    </>
  );
  const base = "flex flex-col items-center rounded-xl border border-line bg-surface px-2 py-3.5 text-ink-2 transition-colors";
  return url ? (
    <a
      {...(video ? videoLink : { href: url, target: "_blank", rel: "noopener noreferrer" })}
      style={style}
      aria-label={`${label}: ${action}${video ? "" : " (opens in a new tab)"}`}
      className={`${base} hover:border-[var(--social)] hover:text-[var(--social)] focus-visible:border-[var(--social)] focus-visible:text-[var(--social)]`}
    >
      {body}
    </a>
  ) : (
    <span role="link" aria-disabled="true" title={`${label} link coming soon`} aria-label={`${label}: link coming soon`} className={`${base} cursor-not-allowed opacity-70`}>
      {body}
    </span>
  );
}

function ProfileCard() {
  const details: [LucideIcon, string, string][] = [
    [GraduationCap, "University", SITE.university],
    [BookOpen, "Course", SITE.course],
    [UserRound, "Instructor", SITE.instructor],
  ];
  const filled = details.filter(([, , value]) => value);
  return (
    <div className="card overflow-hidden lg:sticky lg:top-24 lg:self-start" data-testid="profile-card">
      <div className="relative h-24 bg-[linear-gradient(120deg,#231a5c,#6d4aff_50%,#22b8cf)]">
        <div className="weave absolute inset-0 opacity-40" aria-hidden="true" />
      </div>
      <div className="px-6 pb-6 sm:px-7">
        {SITE.author ? (
          <div className="relative -mt-10 flex size-20 items-center justify-center rounded-2xl bg-[image:var(--grad-btn)] font-display text-2xl font-bold text-[var(--btn-ink)] shadow-lg ring-4 ring-surface-2" aria-hidden="true">
            {initials(SITE.author)}
          </div>
        ) : (
          <div className="relative -mt-10 flex size-20 items-center justify-center rounded-2xl bg-surface-2 p-2 shadow-lg ring-4 ring-surface-2" aria-hidden="true">
            <LogoMark className="size-full" />
          </div>
        )}
        <h3 className="mt-4 font-display text-xl font-bold text-ink">{SITE.author || "Project author"}</h3>
        <p className="text-sm text-ink-2">{SITE.authorRole}</p>

        {filled.length > 0 && (
          <dl className="mt-5 space-y-3">
            {filled.map(([Icon, label, value]) => (
              <div key={label} className="flex items-center gap-3">
                <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-brand-soft text-brand">
                  <Icon className="size-4" aria-hidden="true" />
                </span>
                <div className="min-w-0">
                  <dt className="text-[11px] text-muted">{label}</dt>
                  <dd className="text-sm font-medium text-ink">{value}</dd>
                </div>
              </div>
            ))}
          </dl>
        )}

        <p className="mt-6 text-[11px] font-semibold tracking-wider text-muted uppercase">Find the project online</p>
        <ul className="mt-3 grid grid-cols-3 gap-2" data-testid="social-links">
          {SOCIALS.map((s) => (
            <li key={s.label}>
              <SocialButton {...s} />
            </li>
          ))}
        </ul>
        {SITE.reportUrl && (
          <a href={SITE.reportUrl} target="_blank" rel="noopener noreferrer" className="btn btn-secondary mt-3 w-full">
            <BookOpen className="size-4" aria-hidden="true" /> Read the project report
          </a>
        )}
      </div>
    </div>
  );
}

export function About() {
  return (
    <section id="about" aria-labelledby="about-title" className="border-t border-line py-20 sm:py-24">
      <div className="container-page">
        <SectionHeader eyebrow="About" title="About the project" id="about-title">
          A university project that turns a phone photo of fabric into a quality-inspection report.
        </SectionHeader>

        <div className="mt-10 grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
          <div className="card p-6 sm:p-8">
            <h3 className="font-display text-xl font-semibold text-ink">Why this project</h3>
            <p className="mt-3 leading-relaxed text-ink">
              Fabric inspection is still mostly done by eye. It is slow, tiring and inconsistent from one inspector to the
              next, and defects that slip through become returns and waste.
            </p>
            <p className="mt-3 leading-relaxed text-ink-2">
              FabricScan AI tests how far a small convolutional neural network, trained on real fabric and garment photos from public datasets, can
              automate the first step: deciding whether a fabric is defective, what kind of defect it has, where it is and
              how serious it is. It compares a CNN trained from scratch with a transfer-learning model, deploys both as a
              static website that needs no server (the better one by default), and writes the kind of note a quality inspector would write.
            </p>

            <dl className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
              {HIGHLIGHTS.map((h) => (
                <div key={h.label} className="rounded-xl border border-line bg-surface px-4 py-3">
                  <dt className="sr-only">{h.label}</dt>
                  <dd className="font-display text-2xl font-bold text-brand">{h.value}</dd>
                  <dd className="text-xs text-ink-2">{h.label}</dd>
                </div>
              ))}
            </dl>

            <div className="mt-7 space-y-3">
              <p className="text-[11px] font-semibold tracking-wider text-muted uppercase">Built with</p>
              {STACK.map(([group, items]) => (
                <div key={group} className="flex flex-col gap-2 sm:flex-row sm:items-baseline">
                  <span className="w-36 shrink-0 text-sm font-medium text-ink">{group}</span>
                  <ul className="flex flex-wrap gap-1.5">
                    {items.map((s) => (
                      <li key={s} className="chip">
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          <ProfileCard />
        </div>

        <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
          {[
            { icon: Scale, title: "Limitations", items: LIMITS },
            { icon: Rocket, title: "Future work", items: FUTURE },
          ].map(({ icon: Icon, title, items }) => (
            <div key={title} className="card p-6">
              <h3 className="flex items-center gap-2.5 font-display text-lg font-semibold text-ink">
                <span className="flex size-8 items-center justify-center rounded-lg bg-brand-soft text-brand">
                  <Icon className="size-4" aria-hidden="true" />
                </span>
                {title}
              </h3>
              <ul className="mt-4 space-y-2.5">
                {items.map((item) => (
                  <li key={item} className="flex gap-2.5 text-sm leading-relaxed text-ink-2">
                    <span className="mt-2 size-1.5 shrink-0 rounded-full bg-accent" aria-hidden="true" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
