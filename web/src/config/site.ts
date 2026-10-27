/**
 * Personal details and links. Fill these in before publishing (see MY_TASKS.md).
 *
 * - Empty text fields (author, university, ...) are simply hidden on the page.
 * - Empty links stay visible as "Coming soon" buttons (About and Project links sections). Paste the full link
 *   (https://...) here to activate them; the build also copies every link into plain HTML and /project.json so that
 *   tools that read the page without running JavaScript find them.
 */
export const SITE = {
  name: "FabricScan AI",
  tagline: "AI-based textile defect detection and automatic defect description",
  author: "Hamza",
  authorRole: "Student, project author",
  university: "Superior University, Gold Campus",
  course: "Deep Learning (semester project)",
  instructor: "Dr. Munir Ahmed",
  githubUrl: "https://github.com/hamzaDeploys/fabricscan-ai",
  linkedinUrl: "",
  youtubeUrl: "",
  reportUrl: "https://github.com/hamzaDeploys/fabricscan-ai/blob/main/docs/REPORT.md",
  presentationUrl: "https://github.com/hamzaDeploys/fabricscan-ai/blob/main/docs/FabricScan_AI_Presentation.pptx",
  year: 2026,
};

/** Files served by this website itself (from web/public). */
export const DOWNLOADS = {
  dataset: "/downloads/fabricscan-dataset.zip",
  model: "/models/fabricscan.onnx",
  modelMeta: "/models/model_meta.json",
};
