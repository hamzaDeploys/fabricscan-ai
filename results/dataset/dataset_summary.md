# Dataset summary

Folder: `data/splits`. Origin and licence of every image: `dataset/SOURCES.csv`.

## Images per class

| Class | Train | Val | Test | Total | Physical samples |
|---|---:|---:|---:|---:|---:|
| Good (No Defect) | 242 | 52 | 52 | 346 | 346 |
| Hole | 120 | 26 | 26 | 172 | 172 |
| Stain / Oil Spot | 197 | 43 | 43 | 283 | 283 |
| Broken Thread / Slub | 156 | 34 | 34 | 224 | 224 |
| **Total** | **715** | **155** | **155** | **1025** | **1025** |

Split share: train 69.8%, val 15.1%, test 15.1% (group-aware, seed 42).

Class balance (largest / smallest): **2.01**.

## Image sources

| Class | Kaggle: anmol055/clothes-staindefect-detection (CC BY 4.0) | Kaggle: hyungjung/stitchingnet-dataset (CC BY-NC 4.0) | Kaggle: nexuswho/fabric-defects-dataset (CC BY 4.0) | Kaggle: pulakdroy/fabric-defect-without-augmentation (CC BY 4.0) | Kaggle: rajkumarl/people-clothing-segmentation (CC0) |
|---|---:|---:|---:|---:|---:|
| Good (No Defect) | 98 | 39 | 99 | 0 | 110 |
| Hole | 0 | 0 | 15 | 157 | 0 |
| Stain / Oil Spot | 140 | 0 | 100 | 43 | 0 |
| Broken Thread / Slub | 0 | 99 | 4 | 121 | 0 |
| **Total** | **238** | **138** | **218** | **321** | **110** |

## Cleaning

1078 files found, 1025 kept, 53 removed, 81 flagged for manual review.

| Removal reason | Files |
|---|---:|
| exact_duplicate_of | 3 |
| near_duplicate_of | 3 |
| too_small | 47 |

## Capture conditions (from capture_log.csv)

_capture_log.csv is empty. Fill it in to get fabric, lighting and size statistics._
