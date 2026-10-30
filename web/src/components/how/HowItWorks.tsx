import { Camera, Cpu, FileText, Layers, ScanSearch } from "lucide-react";
import { motion } from "motion/react";

import { SectionHeader } from "../layout/SectionHeader";

const STEPS = [
  {
    icon: Camera,
    title: "Capture",
    text: "Upload, drag and drop, paste or use the camera. The photo is centre-cropped to a square.",
  },
  {
    icon: Layers,
    title: "Preprocess",
    text: "Resized to 224 × 224 pixels exactly as in training. The pixels never leave your device.",
  },
  {
    icon: Cpu,
    title: "Classify",
    text: "A convolutional neural network, exported to ONNX and run with WebAssembly, scores all four classes.",
  },
  {
    icon: ScanSearch,
    title: "Localise",
    text: "A class activation map (equivalent to Grad-CAM for this architecture), refined by colour contrast, gives the location and area.",
  },
  {
    icon: FileText,
    title: "Describe",
    text: "A rule engine turns class, confidence, location and area into a severity-graded report with causes and actions.",
  },
];

const PIPELINE = [
  ["Real fabric photos", "Five public Kaggle datasets of fabric, garments and their defects (licences cited), cut into close-ups."],
  ["Cleaning", "Corrupt files, duplicates and tiny images removed; EXIF orientation applied and metadata stripped."],
  ["Leakage-free split", "70 / 15 / 15, keeping every shot of a physical sample in the same split."],
  ["Augmentation", "Rotation, flips, zoom, brightness, contrast and noise; no colour shifts, because colour is evidence."],
  ["Two models", "A custom CNN trained from scratch and MobileNetV2 fine-tuned from ImageNet, compared on the same test set."],
  ["Export", "Both models are converted to ONNX and verified to give the same outputs as TensorFlow. The better one is the default in the demo."],
];

export function HowItWorks() {
  return (
    <section id="how" aria-labelledby="how-title" className="border-t border-line py-20 sm:py-24">
      <div className="container-page">
        <SectionHeader eyebrow="How it works" title="From photo to inspection report" id="how-title">
          Five steps, all inside the browser. No server, no upload, no waiting in a queue.
        </SectionHeader>
        <ol className="mt-12 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {STEPS.map(({ icon: Icon, title, text }, i) => (
            <motion.li
              key={title}
              className="card card-hover relative p-5"
              initial={{ opacity: 0, y: 14 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ duration: 0.4, delay: i * 0.07 }}
            >
              <span className="icon-tile size-10">
                <Icon className="size-5" aria-hidden="true" />
              </span>
              <p className="mt-4 text-xs font-semibold text-muted">Step {i + 1}</p>
              <h3 className="font-display text-lg font-semibold text-ink">{title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-2">{text}</p>
            </motion.li>
          ))}
        </ol>

        <div className="card mt-8 p-6 sm:p-8">
          <h3 className="font-display text-xl font-semibold text-ink">How the model was built</h3>
          <ol className="mt-6 grid grid-cols-1 gap-x-8 gap-y-5 md:grid-cols-2 lg:grid-cols-3">
            {PIPELINE.map(([title, text], i) => (
              <li key={title} className="flex gap-3">
                <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-[image:var(--grad-btn)] text-xs font-bold text-white">{i + 1}</span>
                <div>
                  <p className="font-semibold text-ink">{title}</p>
                  <p className="mt-1 text-sm leading-relaxed text-ink-2">{text}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  );
}
