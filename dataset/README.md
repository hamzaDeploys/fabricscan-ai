# dataset/

The raw images, one folder per class: `good`, `hole`, `stain`, `broken_thread`.

The images come from five public Kaggle datasets (rebuilt with `python scripts/import_sources.py`):

| Kaggle dataset | Licence | Used for |
|---|---|---|
| [nexuswho/fabric-defects-dataset](https://www.kaggle.com/datasets/nexuswho/fabric-defects-dataset) | CC BY 4.0 | good, hole, stain, broken_thread |
| [pulakdroy/fabric-defect-without-augmentation](https://www.kaggle.com/datasets/pulakdroy/fabric-defect-without-augmentation) | CC BY 4.0 | hole, stain (spot), broken_thread (missing yarn, slub) |
| [anmol055/clothes-staindefect-detection](https://www.kaggle.com/datasets/anmol055/clothes-staindefect-detection) | CC BY 4.0 | stain on clothes, good (clean clothes) |
| [hyungjung/stitchingnet-dataset](https://www.kaggle.com/datasets/hyungjung/stitchingnet-dataset) | CC BY-NC 4.0 | broken_thread (broken stitch), good (normal seam) |
| [rajkumarl/people-clothing-segmentation](https://www.kaggle.com/datasets/rajkumarl/people-clothing-segmentation) | CC0 | good (clean shirts, tops and jackets as worn, cropped with the clothing mask) |

[SOURCES.csv](SOURCES.csv) lists the original file and licence of every image. Detection boxes were cut into square
close-ups around the defect. The cleaned and split dataset is published as a ZIP on the website
(`web/public/downloads/fabricscan-dataset.zip`, built by `scripts/package_dataset.py`).

The raw images are not committed to GitHub. Only this README, `SOURCES.csv`, `capture_log.csv` and the empty class
folders are tracked.

```
dataset/
├── SOURCES.csv         origin and licence of every image
├── good/<group>/*.jpg  one sub-folder per source image, so near-copies stay in the same split
├── hole/ stain/ broken_thread/
└── _rejected/          folders starting with "_" are ignored by all scripts
```
