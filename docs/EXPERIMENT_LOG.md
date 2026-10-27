# Experiment Log

Engineering decisions backed by measurements. These runs used the **synthetic demo dataset**
(`scripts/build_demo_model.py`: 600 code-generated images, 408 train / 96 val / 96 test, group-aware split), so the
absolute numbers say nothing about real fabric photos. The *differences* between settings are what matter. All runs
used seed 42 on a laptop CPU (Intel i7-1165G7).

## E1: Short training schedule (2026-10-26)

| Model | Epochs | Test accuracy |
|---|---|---|
| Custom CNN (GAP head, L2 1e-4, spatial dropout 0.1/0.15, batch 32) | 10 (early-stopped, patience 5) | 28.1 % |
| MobileNetV2 (GAP head) | 4 frozen + 4 fine-tune | 69.8 % |

Validation accuracy was still rising when training stopped, so the schedule was **too short (underfitting)**.
Longer schedules were used afterwards (E2).

## E2: Longer schedule, same architecture (2026-10-26)

| Model | Epochs | Test accuracy |
|---|---|---|
| Custom CNN (GAP head, batch 32) | 23 (early-stopped, patience 12) | 15.6 % |
| MobileNetV2 (GAP head) | 10 frozen + 7 fine-tune (early-stopped) | 70.8 % |

The custom CNN stayed at chance level (1/6 = 16.7 %). A diagnostic run showed it reached only 26 % accuracy **on its
own training images** in evaluation mode, which means it was failing to learn at all (underfitting), not overfitting.

**Hypothesis:** a defect covers only a few of the 14 × 14 positions of the last feature map. Global *average*
pooling divides its signal by about 196 / 4 ≈ 50, so the classifier barely sees it.

## E3: Pooling ablation (2026-10-26)

Controlled comparison: the same lighter regularisation for both CNN runs (L2 1e-5, spatial dropout 0.05, head
dropout 0.3), batch 16, 25 epochs, no early stopping. **Only the pooling layer differs.**

| Model | Global pooling | Test accuracy | Macro F1 | Final train acc |
|---|---|---|---|---|
| Custom CNN | average | 15.6 % | 10.5 % | 34 % |
| Custom CNN | **max** | **56.2 %** | **53.2 %** | 71 % (still rising) |
| MobileNetV2 | **average** (E2) | **70.8 %** | **69.2 %** | — |
| MobileNetV2 | max | 67.7 % | 63.2 % | 82 % |

**Decision:** the custom CNN uses global **max** pooling; MobileNetV2 keeps global **average** pooling. Its
ImageNet features already respond strongly to the defect, and it was slightly better that way.

**Why the website heatmap is still exact:** for a "global pooling → Dense" head, the channel-averaged gradient of
a class score is `w[c, k] / Z` for both average and max pooling. Grad-CAM therefore equals the class activation map
`Σ_c w[c, k] · A_c` up to a positive factor. `tests/test_models.py::test_cam_equals_grad_cam_for_max_pooling` checks
this numerically.

## Notes for the real dataset

- Expect the same pattern: a scratch CNN needs many more epochs than transfer learning. Use at least 60 epochs
  with patience 10–15.
- If the custom CNN underfits on the real photos (training accuracy well below 90 %), first train longer; next,
  try batch 16; only then reduce regularisation further.
