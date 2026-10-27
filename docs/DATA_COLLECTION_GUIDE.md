# Dataset Collection Guide — FabricScan AI

> **Data provenance statement.** The dataset combines two sources: photos taken by the author with a phone camera
> (collected as described in this guide), and the carpet category of the public MVTec AD dataset (Bergmann et al.,
> CVPR 2019; CC BY-NC-SA 4.0), imported with `scripts/import_mvtec_carpet.py`. Every imported file is listed in
> `dataset/SOURCES.csv`. No Kaggle data is used at any stage.

This guide explains exactly how to build the dataset so that the model learns **defects**, not accidental
patterns in how the photos were taken. Follow it before you take the first photo. A 30-minute pilot (Section 12) will
catch most mistakes early.

---

## 1. Targets at a glance

| Item | Minimum | Target | Notes |
|---|---|---|---|
| Images per class | 150 | **250** | Stop at about 300; more physical samples beat more shots |
| Classes | 6 | 6 | See Section 2 |
| Total images | 900 | **1,500** | |
| Physical samples per class | 40 | **60** | One real defect (or one clean patch) = one sample |
| Shots per physical sample | 3 | **4** | Never more than 5 |
| Different fabrics | 8 | **12+** | Each fabric should appear in **every** class |
| Class balance | — | largest ≤ 1.5 × smallest | Training also uses class weights to compensate |

**Why physical samples matter:** 250 photos of 10 holes teach the model 10 holes. 250 photos of 60 different holes on
12 different fabrics teach it what *a hole* looks like.

---

## 2. The four classes

Folder names are fixed. The scripts and the website depend on them.

| Folder | Display name | Counts as | Does **not** count | How to create it on scrap fabric |
|---|---|---|---|---|
| `good` | Good (No Defect) | Clean fabric of any kind: plain, printed, striped, checked, textured, wrinkled, folded, seams, normal edges | Any visible defect | Photograph every fabric **before** damaging it. Deliberately include wrinkles, folds, seams and busy prints, because these are the "hard negatives" that stop false alarms |
| `hole` | Hole | A compact, roughly round opening through the fabric, from about 2 mm to about 3 cm, where the surface underneath shows through | Long slits or rips (not a class here) | Puncture with an awl, skewer or closed scissor tip, then widen; snip a small piece out with nail scissors; vary the size |
| `stain` | Stain / Oil Spot | A foreign substance or discolouration on the surface: cooking oil, grease, tea, coffee, ink, food, mud, rust, lipstick, dye transferred from another garment, bleach spots; fresh or dried | Printed patterns, shadows | Drop or smear oil, tea, coffee, ink, ketchup, turmeric, mud or lipstick; let some dry; lay damp red cotton on white fabric for dye transfer; vary size, shape and colour |
| `broken_thread` | Broken Thread / Slub | Missing, broken, pulled or snagged yarn; a visible line where a thread is missing; thick uneven yarn sections (slubs) | A cut through the whole fabric (→ `hole`) | Hook a yarn with a needle and pull a loop out (snag); cut a single thread with a seam ripper and pull it out a few cm (missing end/pick); slub-cotton fabrics show natural slubs |

### Decision rules for confusing pairs

- **Broken thread vs good:** a normal stripe or a woven pattern line is `good`. A single yarn that is missing, pulled
  out or looped where the pattern says it should not be → `broken_thread`.
- **Still unsure?** Do not guess. Move the photo to `dataset/_rejected/`. A wrong label hurts more than a missing photo.

### One label per image

Every photo must show **exactly one defect type** (or none, for `good`). If a frame contains two defects, reframe the
shot or reject it. Several defects on one piece of fabric are fine, as long as each photo shows only one.

---

## 3. The golden rule: make the defect the only difference

A CNN learns whatever separates the classes most easily. If all your holes are on blue denim and all your stains are on
white cotton, the model learns **"blue = hole, white = stain"**. This is called *shortcut learning*. It gives high test
accuracy and then fails on any new fabric. To prevent it:

1. **Same fabrics in every class.** Photograph each fabric clean (`good`) first, then create defects on it.
   If you use a phone stand, a *before/after pair* of the exact same spot (clean, then damaged) is especially
   valuable. Keep the "before" shot in `good/` and the "after" shot in the defect class.
2. **Same lighting setups in every class.** Use the lighting mix in Section 6 for every class, not "daylight for holes,
   lamp for stains".
3. **Same surfaces underneath in every class.** A hole shows whatever is under the fabric. Rotate through 3–4 surfaces
   (white paper, dark cloth or table, wood, coloured card) and use the same surfaces for `good` photos too. Otherwise
   "white dot" becomes the model's definition of a hole.
4. **Same phone, same settings** for all classes.
5. **Same distance range** for all classes (Section 6).
6. **Spread the defect position across the frame.** The website reports *where* the defect is ("upper-left region"),
   so the training data must not always have it in the centre. Use the 3 × 3 grid below and rotate through the zones.

```
 ┌──────────┬──────────┬──────────┐
 │ upper-   │  upper-  │  upper-  │
 │ left     │  centre  │  right   │
 ├──────────┼──────────┼──────────┤
 │ middle-  │  centre  │  middle- │
 │ left     │          │  right   │
 ├──────────┼──────────┼──────────┤
 │ lower-   │  lower-  │  lower-  │
 │ left     │  centre  │  right   │
 └──────────┴──────────┴──────────┘
  Turn on the camera's grid lines to see these zones.
  Keep the defect at least ~10% away from the outer border.
```

---

## 4. Equipment and materials

- **Phone** with a decent main camera (use the 1× lens)
- **10–15 scrap fabrics**, about 30 × 30 cm or larger. Aim for a mix of:
  plain cotton, denim, polyester, linen, knit/jersey, printed, striped or checked, satin, fleece or towel;
  white or light, dark, and bright colours
- **Tools:** sharp scissors, nail scissors, seam ripper, sewing needle, awl or skewer, fine sandpaper
- **Staining:** cooking oil, tea, coffee, ink or markers, ketchup, turmeric, mud, lipstick
- **Dye-transfer stains:** a red or dark cotton that bleeds, water, cotton buds, rubber gloves
- **3–4 background surfaces** to place under the fabric (Section 3)
- **Optional:** phone stand or tripod, a white LED desk lamp and a warm (yellow) bulb
- Clips or tape to keep fabric flat (always **out of frame**)

---

## 5. Phone settings (set once, keep for the whole shoot)

| Setting | iPhone | Android |
|---|---|---|
| File format **JPEG** | Settings › Camera › Formats › **Most Compatible** | Camera settings › turn **off** "High efficiency pictures" / HEIF |
| Aspect ratio **1:1 (square)** | Camera app › tap **4:3** › choose **Square** (older iOS: swipe to *Square* mode) | Camera › ratio icon › **1:1** |
| Resolution | Default (highest) | Highest available for 1:1 |
| Flash | **Off** | **Off** |
| Live Photos / Motion photos | Off | Off |
| Night mode | Off | Off |
| AI scene optimiser, beauty, filters | No filters | **Off** (e.g. Samsung *Scene optimizer*, Xiaomi *AI camera*) |
| Watermark / date stamp | — | **Off** (e.g. Xiaomi *Add watermark*) |
| Grid lines | Settings › Camera › **Grid** on | Camera settings › **Grid lines** on |
| Lens | **1×** main lens (not 0.5× ultra-wide) | **1×** main lens |

**Why square?** The models see square images (224 × 224 pixels). A 1:1 photo loses nothing when resized. If your phone
has no 1:1 mode, shoot 4:3 and keep the defect inside the central square: the cleaning script centre-crops to a
square, and anything near the long edges will be cut off.

**Why no watermark or filters?** A watermark that appears only in some sessions becomes a shortcut the model can learn.
Filters change colours, which matters a lot for `stain`.

---

## 6. Capture protocol

### Framing

- Shoot **top-down**, phone parallel to the fabric, **20–35 cm** away.
- The **fabric fills the whole frame**: no table edges, hands, tools, tape, labels or text.
- Smooth the fabric flat, but leave natural wrinkles in some `good` photos on purpose.
- **Tap the fabric to focus.** Long-press to lock focus and exposure (AE/AF lock) if your phone supports it.
- Before moving on, **zoom into the preview**. The weave or texture must look sharp.
- Keep the defect **at least ~10% away from the border**, so the later crop and zoom augmentation never cuts it out.

### Defect size mix (per class)

| Size | Share of the frame | Target share of photos |
|---|---|---|
| Small | about 1–3% | ~40% |
| Medium | about 3–10% | ~40% |
| Large | over 10% | ~20% |

This mix lets the system learn to estimate the affected area and severity.

### Lighting mix (the same for every class)

| Light | Share |
|---|---|
| Diffuse daylight (near a window, **no direct sun**) | ~50% |
| Indoor white / cool LED | ~25% |
| Indoor warm / yellow light | ~25% |

- **No flash, ever.** A flash glare spot looks like a stain or a hole.
- Avoid your own shadow on the fabric: keep the light source to your side, not behind you.

### The 4-shot recipe (per physical sample)

| Shot | Light | Camera | What changes |
|---|---|---|---|
| A | Daylight | Top-down | Defect in one grid zone |
| B | Same as A | Top-down, **fabric rotated 90°** | Defect moves to a different zone |
| C | **Indoor** (LED or warm) | Top-down | Different lighting |
| D | Any | **Tilted 15–25°**, or noticeably closer/farther | Perspective and scale |

For `good`, use the same recipe on a clean region. Move to a new region of the fabric for each new sample.

---

## 7. Shot plan

For the 250-image target: **about 60 physical samples × 4 shots = 240 photos per class**, plus a few extras. Spread
them across fabrics, for example 12 fabrics × 5 samples each.

**Capture order (least destructive first)**, so that every fabric can serve every class:

1. `good`: every fabric, several regions each, including wrinkles, folds, seams and prints
2. `stain`
3. `broken_thread`
4. `hole`

Keep a running tally (the capture log in Section 8 does this for you) so the classes stay balanced.

---

## 8. Organising the files on your PC

### Transferring photos

Copy photos with a **USB cable**, **Google Drive** (upload in original quality) or **AirDrop**.
**Never** send them through WhatsApp, Telegram or Messenger: these apps compress and resize images.

### Folder layout

```
dataset/
├── capture_log.csv          ← optional but recommended (one row per sample)
├── good/
│   ├── s001/                ← one folder per physical sample
│   │   ├── IMG_1023.jpg     ← its 3–5 shots (original phone file names are fine)
│   │   ├── IMG_1024.jpg
│   │   ├── IMG_1025.jpg
│   │   └── IMG_1026.jpg
│   ├── s002/
│   └── ...
├── hole/
│   ├── s001/
│   └── ...
├── stain/
├── broken_thread/
└── _rejected/               ← ambiguous or bad photos; ignored by every script
```

- Sample IDs restart at `s001` in each class. Use three digits (`s001` … `s060`).
- Any folder whose name starts with `_` is ignored by the scripts.

**Why one folder per sample?** The 4 shots of the same hole are near-copies. If one lands in the training set and
another in the test set, the model is tested on something it has effectively already seen, and the reported accuracy is
inflated. This is called **data leakage**. The split script keeps each sample folder entirely inside one split
(train, val or test). This detail matters for an honest evaluation, and the report will point it out.

*Flat files also work.* If you put photos directly in `dataset/hole/` without sub-folders, the scripts group them
by the capture time stored in each photo: shots taken less than 30 seconds apart count as one sample. That only
works if you pause between samples and the phone keeps the time stamp, so sub-folders are preferred.

### Capture log (optional, recommended)

`dataset/capture_log.csv` has one row per **sample** and takes about 10 seconds to fill in. It produces the dataset
statistics table in the report (fabrics, lighting and size distribution) for free.

| Column | Allowed values / example |
|---|---|
| `sample_id` | `s001` |
| `class` | `good`, `hole`, `stain`, `broken_thread` |
| `fabric_id` | `F01` … `F15` (number your fabric pieces once) |
| `fabric_type` | `cotton`, `denim`, `polyester`, `linen`, `knit`, `satin`, `fleece`, … |
| `fabric_color` | `white`, `dark blue`, `red`, `multicolour`, … |
| `pattern` | `plain`, `printed`, `striped`, `checked`, `textured` |
| `defect_method` | `awl puncture`, `coffee dried 2h`, `seam ripper pull`, … (`none` for good) |
| `defect_size` | `S`, `M`, `L` (`-` for good) |
| `lighting` | `daylight`, `led`, `warm`, `mixed` (`mixed` when shots A–D differ) |
| `background` | `white paper`, `dark cloth`, `wood`, `card` |
| `date` | `2026-10-28` |
| `notes` | free text |

Example rows:

```csv
sample_id,class,fabric_id,fabric_type,fabric_color,pattern,defect_method,defect_size,lighting,background,date,notes
s001,good,F03,denim,dark blue,plain,none,-,mixed,wood,2026-10-28,includes a fold
s001,hole,F03,denim,dark blue,plain,awl puncture widened,S,mixed,white paper,2026-10-28,
s014,stain,F07,cotton,white,striped,coffee dried 2h,M,mixed,dark cloth,2026-10-29,
```

---

## 9. Quality control

Reject a photo (delete it or move it to `_rejected/`) if:

- it is **blurry**: zoom to 100% and check that the weave is sharp
- it shows **more than one defect type**, or the defect is ambiguous
- the defect is **cut off** by the frame edge
- **hands, fingers, tools, tape, table edges, labels or text** are visible
- there is **heavy glare** or a hard shadow over the defect
- it is **edited, filtered, a screenshot**, or a burst duplicate

The cleaning script (`scripts/clean_dataset.py`) then automatically:
removes corrupt and unreadable files; removes exact and near duplicates; drops images that are too small; flags very
blurry, very dark and over-exposed images for review; flags near-identical images filed under two different classes
(`check_label`); fixes rotation from EXIF; strips EXIF metadata (including GPS); centre-crops to square; resizes;
and writes a CSV report of everything it did. Details: [PREPROCESSING.md](PREPROCESSING.md).
Your own eye is still the first and most important filter.

---

## 10. Optional: field test set (recommended for a top grade)

After the main dataset is finished, capture **15–20 extra photos per class**:

- in a **different room**, on a **different day**
- ideally with a **different phone** (borrow one)
- on **fabric pieces that were never used** in the main dataset

Store them in `field_test/<class>/` (same class folder names, no sample sub-folders needed). These images are **never**
used for training or tuning. They are reported separately as *real-world generalisation*. It is usually lower than the
test-set accuracy, and that honest gap is exactly what examiners like to see discussed.

---

## 11. Safety and privacy

- **Sharp tools:** always cut away from your body. Keep the awl and seam ripper capped when not in use.
- **Bleach:** wear gloves, work in a ventilated room, and **never mix bleach with other cleaners** (ammonia, vinegar or
  acids release toxic gas).
- **No burning.** Do not use flames or cigarettes to make holes. Synthetic fabrics melt and release toxic fumes.
- Only damage **scrap fabric you own**.
- **Privacy:** no faces, people, ID cards, documents or screens in any frame. The repository is public. The cleaning
  script strips GPS and other EXIF metadata anyway.

---

## 12. Suggested timeline

| Step | Time |
|---|---|
| Gather fabrics, set up the phone (Section 5) | ~1 h |
| **Pilot:** 3 samples × 4 shots for every class (≈ 48 photos) → run `python scripts/clean_dataset.py` → check `data/reports/flagged/` → fix any issues | ~1 h |
| `good` + `stain` | 2–3 h |
| `broken_thread` | 1–2 h |
| `hole` | ~1 h |
| Field test set (optional) | ~1 h |
| **Total** | **about 1–2 days** |

Do the pilot first. Finding out after 1,500 photos that the phone saved HEIC files, or that every stain is on white
cotton, costs a day.

---

## 13. Things to record for the report and LinkedIn

- Take **3–4 "behind the scenes" photos** of your setup (fabric, phone stand, lighting). They are useful for the
  report's *Dataset Collection* section and the LinkedIn post.
- Write down problems as they happen: lighting changes during the day, classes that were hard to create (usually
  `broken_thread`), blurry batches, confusing hole/broken-thread cases, and what you did about each one.
  These notes become the report's **Problems Faced** section.
- Keep the final count per class. `scripts/dataset_report.py` produces the table, but your notes explain *why* the
  numbers look the way they do.
