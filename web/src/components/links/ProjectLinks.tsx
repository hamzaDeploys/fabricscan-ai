import { BrainCircuit, Clapperboard, Database, Download, ExternalLink, FileText, Play, Presentation } from "lucide-react";
import { motion } from "motion/react";
import type { ComponentType, SVGProps } from "react";

import { DOWNLOADS, SITE } from "../../config/site";
import { useSiteData } from "../../hooks/useSiteData";
import { GitHubIcon } from "../layout/icons";
import { SectionHeader } from "../layout/SectionHeader";
import { useVideoLink } from "../layout/VideoDialog";

interface Resource {
  id: string;
  title: string;
  text: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  url: string;
  action: string;
  download?: boolean; // a file served by this site, saved instead of opened
  video?: boolean; // plays in the popup
}

/** One resource card. Without a URL it stays visible but inactive ("Coming soon"). */
function ResourceCard({ resource, index }: { resource: Resource; index: number }) {
  const { id, title, text, icon: Icon, url, action, download, video } = resource;
  const videoLink = useVideoLink();
  const ActionIcon = video ? Play : download ? Download : ExternalLink;
  return (
    <motion.li
      className="card card-hover flex flex-col p-5"
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.4, delay: (index % 3) * 0.06 }}
    >
      <span className="icon-tile size-10">
        <Icon className="size-5" aria-hidden="true" />
      </span>
      <h3 className="mt-4 font-display text-lg font-semibold text-ink">{title}</h3>
      <p className="mt-1.5 flex-1 text-sm leading-relaxed text-ink-2">{text}</p>
      {url ? (
        <a
          data-testid={`link-${id}`}
          className={`btn mt-4 ${download || video ? "btn-primary" : "btn-secondary"}`}
          {...(video ? videoLink : download ? { href: url, download: "" } : { href: url, target: "_blank", rel: "noopener noreferrer" })}
        >
          <ActionIcon className="size-4" aria-hidden="true" /> {action}
        </a>
      ) : (
        <span role="link" aria-disabled="true" data-testid={`link-${id}`} className="btn btn-secondary mt-4 cursor-not-allowed opacity-60">
          Coming soon
        </span>
      )}
    </motion.li>
  );
}

/** Everything the course asks the website to link: code, dataset, weights, report, slides and video. */
export function ProjectLinks() {
  const { dataset, download } = useSiteData();
  const total = download?.images ?? dataset?.total_images;
  const images = total !== undefined ? `${total.toLocaleString("en-US")} cleaned images` : "The cleaned images";
  const zip = download ? `ZIP, ${(download.size_bytes / 1e6).toFixed(1)} MB` : "ZIP";
  const resources: Resource[] = [
    {
      id: "github",
      title: "Source code",
      text: "The full GitHub repository: data pipeline, training scripts, notebooks, this website and the model weights.",
      icon: GitHubIcon,
      url: SITE.githubUrl,
      action: "Open on GitHub",
    },
    {
      id: "dataset",
      title: "Dataset",
      text: `${images} in train / val / test folders, with the source and licence of every image (${zip}).`,
      icon: Database,
      url: DOWNLOADS.dataset,
      action: "Download dataset",
      download: true,
    },
    {
      id: "weights",
      title: "Model weights",
      text: "The default model on this page (ONNX). Both models, in ONNX and Keras format, are in the GitHub repository.",
      icon: BrainCircuit,
      url: DOWNLOADS.model,
      action: "Download model",
      download: true,
    },
    {
      id: "report",
      title: "Project report",
      text: "Problem statement, dataset, architectures, training details, results and the problems we solved.",
      icon: FileText,
      url: SITE.reportUrl,
      action: "Read the report",
    },
    {
      id: "presentation",
      title: "Presentation",
      text: "The project slides.",
      icon: Presentation,
      url: SITE.presentationUrl,
      action: "Open the slides",
    },
    {
      id: "video",
      title: "Video",
      text: "The presentation explained with a voice-over. It plays right here on this page.",
      icon: Clapperboard,
      url: DOWNLOADS.video,
      action: "Play video",
      video: true,
    },
  ];

  return (
    <section id="links" aria-labelledby="links-title" className="border-t border-line py-20 sm:py-24">
      <div className="container-page">
        <SectionHeader eyebrow="Project links" title="Code, data, model, report and video" id="links-title">
          Everything behind this page, in one place.
        </SectionHeader>
        <ul className="mt-10 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3" data-testid="project-links">
          {resources.map((r, i) => (
            <ResourceCard key={r.id} resource={r} index={i} />
          ))}
        </ul>
      </div>
    </section>
  );
}
