# Image dataset needed: FabricScan AI (clothing and fabric defects)

I am building an AI model that looks at a photo of a garment or a piece of fabric and says which defect it has.
I need real photos for 4 classes.

## The 4 classes

| Folder | What it is | What it is NOT |
|---|---|---|
| `good` | No defect. Shirts, T-shirts, kurtas, trousers, bedsheets, plain fabric. Include plain, printed, striped and checked fabric with normal seams, buttons, wrinkles and folds. | Anything with a visible defect |
| `hole` | A small, roughly round opening where fabric is missing and you can see through it: moth holes, burn holes, worn-through holes | Buttonholes, eyelets, mesh or lace patterns; long rips |
| `stain` | A mark from something that got onto the fabric: oil, grease, food, tea, coffee, ink, sweat, rust, mud, lipstick, dye transferred from another garment in the wash, bleach spots | Printed patterns, shadows |
| `broken_thread` | The fabric is still whole, but a yarn is pulled, snagged, broken or missing: a pulled loop, a snag line, a ladder in knitwear, a line where a thread is missing | An opening through the fabric (→ `hole`) |

## How many

- Target **150 photos per class** (600 in total). Minimum **100 per class**.
- Keep the classes roughly equal: the biggest class should have at most 1.5 times as many photos as the smallest.

## What a usable photo looks like

- **Real photographs only.** No drawings, 3D renders or AI-generated images.
- **The defect is clearly visible.** Close-up or medium shot: the garment or fabric fills most of the picture.
  Quick test: shrink the photo to a small thumbnail (about 224 × 224 px). If the defect disappears, the photo was
  taken from too far away.
- **One defect type per photo** (none for `good`). If a photo shows a stain and a hole, skip it.
- Shorter side at least 256 px (500 px or more is better). JPG or PNG; WEBP and HEIC are also fine.
- No watermarks, text, logos, arrows or circles drawn on the image, no before/after collages, no screenshots.
- **Variety in every class:** garment types, light and dark colours, plain and patterned fabric, backgrounds and
  lighting. For example, do not take all stains on white shirts and all good photos on dark shirts, or the model
  will learn the colour instead of the defect.
- The `good` photos should look like the defect photos (same kinds of garments, photo style and sources), only
  without the defect.
- No duplicates: not the same photo twice, and not a resized or cropped copy of a photo already included.

## How to send them

One zip file with this layout:

```
dataset/
├── good/
├── hole/
├── stain/
├── broken_thread/
└── sources.csv
```

- Put each photo in its class folder.
- If several photos show the **same garment or the same defect** (different angles or lighting), put them together in
  a sub-folder, for example `stain/shirt01/1.jpg`, `stain/shirt01/2.jpg`. This keeps near-copies out of both the
  training and the test set at the same time.
- `sources.csv` has one row per photo with the columns `file, source, link, licence`. Example:
  `stain/shirt01/1.jpg, own photo, , own` or `hole/img07.jpg, Wikimedia Commons, https://..., CC BY-SA 4.0`.
- Use only own photos, photos from people who gave permission, or images under an open licence (CC0, CC BY,
  CC BY-SA, the Pexels / Unsplash / Pixabay licence, or public datasets that allow academic use).
