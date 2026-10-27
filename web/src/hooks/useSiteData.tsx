import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

import { fetchJson } from "../lib/model";
import type { DatasetDownload, DatasetSummary, MetricsFile, ModelMeta, SampleImage } from "../types";

export interface SiteData {
  meta: ModelMeta | null;
  metrics: MetricsFile | null;
  dataset: DatasetSummary | null;
  download: DatasetDownload | null;
  samples: SampleImage[];
  loaded: boolean;
}

const EMPTY: SiteData = { meta: null, metrics: null, dataset: null, download: null, samples: [], loaded: false };
const SiteDataContext = createContext<SiteData>(EMPTY);

/** Loads the JSON files produced by the training pipeline. Missing files simply leave a section empty. */
export function SiteDataProvider({ children }: { children: ReactNode }) {
  const [data, setData] = useState<SiteData>(EMPTY);
  useEffect(() => {
    Promise.allSettled([
      fetchJson<ModelMeta>("models/model_meta.json"),
      fetchJson<MetricsFile>("data/metrics.json"),
      fetchJson<DatasetSummary>("data/dataset_summary.json"),
      fetchJson<SampleImage[]>("samples/samples.json"),
      fetchJson<DatasetDownload>("downloads/fabricscan-dataset.json"),
    ]).then(([meta, metrics, dataset, samples, download]) => {
      setData({
        meta: meta.status === "fulfilled" ? meta.value : null,
        metrics: metrics.status === "fulfilled" ? metrics.value : null,
        dataset: dataset.status === "fulfilled" ? dataset.value : null,
        download: download.status === "fulfilled" ? download.value : null,
        samples: samples.status === "fulfilled" ? samples.value : [],
        loaded: true,
      });
    });
  }, []);
  return <SiteDataContext.Provider value={data}>{children}</SiteDataContext.Provider>;
}

export const useSiteData = () => useContext(SiteDataContext);
