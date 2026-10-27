import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

import { DOWNLOADS, SITE } from "./src/config/site";

// Cross-origin isolation lets onnxruntime-web use several CPU threads (SharedArrayBuffer).
// Everything the page loads is same-origin (fonts are bundled), so these strict headers break nothing.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

/**
 * The page is rendered by JavaScript, so a crawler that only reads HTML would find no links. This plugin copies the
 * project links from src/config/site.ts into a hidden plain-HTML list in index.html and into /project.json.
 */
function projectLinks(): Plugin {
  const links: [string, string][] = [
    ["GitHub repository", SITE.githubUrl],
    ["Dataset (ZIP)", DOWNLOADS.dataset],
    ["Model weights (ONNX)", DOWNLOADS.model],
    ["Model details (JSON)", DOWNLOADS.modelMeta],
    ["Project report", SITE.reportUrl],
    ["Presentation", SITE.presentationUrl],
    ["YouTube video", SITE.youtubeUrl],
    ["LinkedIn", SITE.linkedinUrl],
  ];
  const active = links.filter(([, url]) => url);
  const escape = (s: string) => s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
  const html = [
    '<nav id="project-links" aria-label="Project links" hidden>',
    "  <ul>",
    ...active.map(([label, url]) => `    <li><a href="${escape(url)}">${escape(label)}</a></li>`),
    "  </ul>",
    "</nav>",
  ].join("\n    ");
  const json = JSON.stringify(
    {
      name: SITE.name,
      description: SITE.tagline,
      author: SITE.author,
      github: SITE.githubUrl,
      dataset: DOWNLOADS.dataset,
      model: { onnx: DOWNLOADS.model, meta: DOWNLOADS.modelMeta },
      report: SITE.reportUrl,
      presentation: SITE.presentationUrl,
      video: SITE.youtubeUrl,
      linkedin: SITE.linkedinUrl,
    },
    null,
    2,
  );
  return {
    name: "project-links",
    transformIndexHtml: (page) => page.replace("</body>", `  ${html}\n  </body>`),
    configureServer(server) {
      server.middlewares.use("/project.json", (_req, res) => {
        res.setHeader("Content-Type", "application/json");
        res.end(json);
      });
    },
    generateBundle() {
      this.emitFile({ type: "asset", fileName: "project.json", source: json });
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), projectLinks()],
  server: { port: 5173, headers: isolation },
  preview: { port: 4173, headers: isolation },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { target: "es2022", chunkSizeWarningLimit: 1200 },
});
