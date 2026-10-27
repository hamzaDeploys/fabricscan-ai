import csv
from pathlib import Path

import pytest
from PIL import Image

from fabricscan.cleaning import CleanConfig, clean_dataset
from fabricscan.synthetic import generate_dataset

SAMPLES = 7


def _manifest(reports: Path) -> dict[str, dict]:
    with open(reports / "clean_manifest.csv", newline="", encoding="utf-8") as fh:
        return {row["source"]: row for row in csv.DictReader(fh)}


def _quiet(_: str) -> None:
    pass


@pytest.fixture(scope="module")
def cleaned(tmp_path_factory):
    root = tmp_path_factory.mktemp("clean")
    generated = generate_dataset(root / "raw", samples_per_class=SAMPLES, shots=3, size=320, seed=1)
    cfg = CleanConfig(root / "raw", root / "processed", root / "reports", workers=1)
    summary = clean_dataset(cfg, log=_quiet)
    return root, generated, summary, _manifest(root / "reports")


def test_every_planted_fault_gets_the_expected_outcome(cleaned):
    _, generated, _, manifest = cleaned
    for rel, outcome in generated["expected"].items():
        row = manifest.get(rel)
        if outcome == "ignored":
            assert row is None, rel
            continue
        assert row is not None, rel
        if outcome == "kept":
            assert row["status"] == "kept", (rel, row["reason"])
        elif outcome.startswith("flag:"):
            assert row["status"] == "kept" and outcome[5:] in row["flags"].split(";"), (rel, row)
        else:
            assert row["status"] == "removed" and row["reason"].startswith(outcome), (rel, row["reason"])


def test_only_the_planted_near_duplicate_is_removed(cleaned):
    _, _, _, manifest = cleaned
    removed = [k for k, r in manifest.items() if r["reason"].startswith("near_duplicate_of")]
    assert removed == ["good/s008/zz_resaved.jpg"]


def test_outputs_are_square_rgb_jpegs_without_exif(cleaned):
    root, _, summary, _ = cleaned
    files = list((root / "processed").rglob("*.jpg"))
    assert len(files) == summary["totals"]["kept"]
    for path in files[:25]:
        with Image.open(path) as img:
            assert img.size == (256, 256) and img.mode == "RGB" and not img.getexif()


def test_sample_folders_become_groups_and_flat_files_are_grouped_by_time(cleaned):
    root, _, summary, manifest = cleaned
    flat_groups = {r["group"] for r in manifest.values() if r["class"] == "broken_thread" and r["status"] == "kept"}
    assert len(flat_groups) == SAMPLES
    names = [p.name for p in (root / "processed" / "hole").glob("*.jpg")]
    assert all(n.split("__")[0].startswith("s0") for n in names)
    assert summary["per_class"]["hole"]["groups"] == SAMPLES


def test_flagged_images_are_copied_for_review(cleaned):
    root, _, _, _ = cleaned
    assert any((root / "reports" / "flagged" / "blurry").glob("*.jpg"))
    assert any((root / "reports" / "flagged" / "check_label").glob("*.jpg"))


def test_drop_flagged_removes_quality_outliers(cleaned):
    root, _, _, _ = cleaned
    cfg = CleanConfig(root / "raw", root / "processed_strict", root / "reports_strict", workers=1, drop_flagged=True)
    clean_dataset(cfg, log=_quiet)
    row = _manifest(root / "reports_strict")["good/s005/zz_blurry.jpg"]
    assert row["status"] == "removed" and row["reason"].startswith("quality:blurry")


def test_multiprocessing_gives_the_same_result(cleaned):
    root, _, summary, manifest = cleaned
    cfg = CleanConfig(root / "raw", root / "processed_mp", root / "reports_mp", workers=2)
    summary_mp = clean_dataset(cfg, log=_quiet)
    assert summary_mp["per_class"] == summary["per_class"]
    assert {k: r["status"] for k, r in _manifest(root / "reports_mp").items()} == {
        k: r["status"] for k, r in manifest.items()
    }


def test_exif_orientation_is_applied(tmp_path):
    # Stored pixels: top half red, bottom half blue. Orientation 6 means "rotate 90 degrees clockwise to
    # display", so after cleaning the red half must be on the right.
    img = Image.new("RGB", (300, 200), (0, 0, 255))
    img.paste((255, 0, 0), (0, 0, 300, 100))
    exif = Image.Exif()
    exif[0x0112] = 6
    (tmp_path / "raw/good/s001").mkdir(parents=True)
    img.save(tmp_path / "raw/good/s001/rotated.jpg", exif=exif.tobytes())
    cfg = CleanConfig(tmp_path / "raw", tmp_path / "out", tmp_path / "rep", out_size=64, min_side=100, workers=1,
                      class_names=("good",))
    clean_dataset(cfg, log=_quiet)
    (out,) = (tmp_path / "out/good").glob("*.jpg")
    with Image.open(out) as result:
        left, right = result.getpixel((8, 32)), result.getpixel((56, 32))
    assert left[2] > 200 and left[0] < 60
    assert right[0] > 200 and right[2] < 60


def test_refuses_to_wipe_a_folder_it_did_not_create(tmp_path):
    (tmp_path / "raw/good/s001").mkdir(parents=True)
    Image.new("RGB", (300, 300)).save(tmp_path / "raw/good/s001/a.jpg")
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "keep.txt").write_text("do not delete")
    cfg = CleanConfig(tmp_path / "raw", precious, tmp_path / "rep", workers=1, class_names=("good",))
    with pytest.raises(RuntimeError, match="will not be"):
        clean_dataset(cfg, log=_quiet)
    assert (precious / "keep.txt").exists()
