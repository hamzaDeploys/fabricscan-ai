/** Shapes of the JSON files written by the Python pipeline (fabricscan/workflow.py, scripts/dataset_report.py). */

export interface ModelMeta {
  model_file: string;
  model_key: string;
  architecture: string;
  demo: boolean;
  created: string;
  input: { name: string; layout: "NHWC"; size: number; range: [number, number]; channels: "RGB" };
  outputs: { probs: string; cam: string };
  cam_shape: [number, number, number];
  class_names: string[];
  display_names: Record<string, string>;
  test_accuracy: number | null;
}

/** web/public/models/models.json, written by fabricscan/workflow.py. The first entry is the default model. */
export interface ModelChoice {
  key: string; // key in metrics.json
  name: string;
  meta: string; // its model_meta file in models/
}

export interface ClassScores {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface TestMetrics {
  n: number;
  accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  precision_weighted: number;
  recall_weighted: number;
  f1_weighted: number;
  per_class: Record<string, ClassScores>;
  confusion_matrix: number[][];
}

export interface ModelEntry {
  key: string;
  name: string;
  architecture: string;
  params: number;
  keras_size_mb: number;
  onnx_size_mb: number;
  keras_latency_ms: number;
  onnx_latency_ms: number;
  epochs: number;
  train_time_s: number;
  fine_tune_start_epoch: number | null;
  best_val_accuracy: number;
  history: Record<string, number[]>;
  test: TestMetrics;
}

export interface MetricsFile {
  demo: boolean;
  generated: string;
  class_names: string[];
  display_names: Record<string, string>;
  images: Record<string, number>;
  best_model: string;
  selection_rule: string;
  models: Record<string, ModelEntry>;
}

export interface DatasetSummary {
  class_names: string[];
  display_names: Record<string, string>;
  images: Record<string, Record<string, number>>;
  totals: Record<string, number>;
  total_images: number;
  physical_samples: Record<string, number>;
  imbalance_ratio: number;
  sources?: Record<string, number>; // e.g. {"Kaggle: owner/dataset (CC BY 4.0)": 318}
}

/** web/public/downloads/fabricscan-dataset.json, written by scripts/package_dataset.py. */
export interface DatasetSource {
  title: string;
  ref: string; // Kaggle owner/dataset
  licence: string;
  url: string;
  images: number;
  classes: Record<string, number>;
}

export interface DatasetDownload {
  file: string;
  size_bytes: number;
  images: number;
  image_size: number;
  classes: string[];
  splits: Record<string, Record<string, number>>;
  sources: DatasetSource[];
}

export interface SampleImage {
  file: string;
  class: string;
  label: string;
  source: string;
}
