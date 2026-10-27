import { TriangleAlert } from "lucide-react";

import { useSiteData } from "../../hooks/useSiteData";

/** Shown while the site runs the demo model trained on code-generated images. */
export function DemoBanner() {
  const { meta } = useSiteData();
  if (!meta?.demo) return null;
  return (
    <div role="status" className="border-b border-[color-mix(in_oklab,var(--sev-low)_45%,transparent)] bg-[color-mix(in_oklab,var(--sev-low)_16%,var(--surface))]">
      <p className="container-page flex items-start gap-2 py-2 text-sm text-[var(--sev-low-ink)] sm:items-center">
        <TriangleAlert className="mt-0.5 size-4 shrink-0 sm:mt-0" aria-hidden="true" />
        <span>
          <strong className="font-semibold">Demo model.</strong> It was trained on computer-generated fabric images to test
          the website. Results on real photos are not meaningful until it is trained on the real dataset.
        </span>
      </p>
    </div>
  );
}
