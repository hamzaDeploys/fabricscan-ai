import csv
import random

import numpy as np
import pytest
from PIL import Image

from fabricscan.config import CLASS_NAMES, SPLIT_RATIOS
from fabricscan.splitting import SplitConfig, assign_groups_to_splits, group_of, split_dataset


def test_group_of_reads_the_prefix():
    assert group_of("s001__IMG_1234.jpg") == "s001"
    assert group_of("auto003__IMG_9.jpg") == "auto003"
    assert group_of("plain.jpg") == "plain"


def test_equal_groups_hit_the_exact_ratio():
    sizes = {f"s{i:03d}": 4 for i in range(60)}  # 240 images -> 168 / 36 / 36
    assignment = assign_groups_to_splits(sizes, SPLIT_RATIOS, random.Random(0))
    counts = {s: sum(sizes[g] for g, a in assignment.items() if a == s) for s in SPLIT_RATIOS}
    assert counts == {"train": 168, "val": 36, "test": 36}


def test_uneven_groups_stay_close_to_the_ratio():
    rng = np.random.default_rng(3)
    sizes = {f"g{i}": int(rng.integers(2, 7)) for i in range(70)}
    assignment = assign_groups_to_splits(sizes, SPLIT_RATIOS, random.Random(1))
    total = sum(sizes.values())
    for split, ratio in SPLIT_RATIOS.items():
        share = sum(sizes[g] for g, a in assignment.items() if a == split) / total
        assert abs(share - ratio) < 0.03


def test_assignment_is_reproducible_and_seed_dependent():
    sizes = {f"s{i:03d}": 3 + i % 3 for i in range(40)}
    a = assign_groups_to_splits(sizes, SPLIT_RATIOS, random.Random(42))
    b = assign_groups_to_splits(sizes, SPLIT_RATIOS, random.Random(42))
    c = assign_groups_to_splits(sizes, SPLIT_RATIOS, random.Random(7))
    assert a == b
    assert a != c


def test_too_few_groups_is_an_error():
    with pytest.raises(ValueError):
        assign_groups_to_splits({"a": 3, "b": 3}, SPLIT_RATIOS, random.Random(0))


def _fake_processed(root, groups_per_class=10, shots=3):
    rng = np.random.default_rng(0)
    for c in CLASS_NAMES:
        (root / c).mkdir(parents=True)
        for g in range(groups_per_class):
            for k in range(shots):
                arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(arr).save(root / c / f"s{g:03d}__IMG_{k}.jpg")


def test_split_dataset_keeps_groups_together(tmp_path):
    _fake_processed(tmp_path / "processed")
    cfg = SplitConfig(tmp_path / "processed", tmp_path / "splits", tmp_path / "reports", seed=5)
    summary = split_dataset(cfg, log=lambda _: None)
    assert summary["total_images"] == len(CLASS_NAMES) * 30
    assert summary["audit"] == {"shared_groups": [], "cross_split_near_duplicates": []}
    with open(tmp_path / "reports" / "split_manifest.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    split_of_group = {}
    for row in rows:
        key = (row["class"], row["group"])
        assert split_of_group.setdefault(key, row["split"]) == row["split"]
        assert (tmp_path / "splits" / row["path"]).is_file()
    for split in SPLIT_RATIOS:
        for c in CLASS_NAMES:
            assert summary["images"][split][c] > 0


def test_audit_catches_a_duplicate_planted_across_splits(tmp_path):
    _fake_processed(tmp_path / "processed")
    cfg = SplitConfig(tmp_path / "processed", tmp_path / "splits", tmp_path / "reports", seed=5)
    split_dataset(cfg, log=lambda _: None)
    with open(tmp_path / "reports" / "split_manifest.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    train = next(r for r in rows if r["split"] == "train" and r["class"] == "hole")
    # Copy a training image into a different group of the same class: it must land in another split's audit.
    src = tmp_path / "splits" / train["path"]
    victim = next(r for r in rows if r["split"] == "test" and r["class"] == "hole")
    (tmp_path / "processed" / "hole" / victim["path"].split("/")[-1]).write_bytes(src.read_bytes())
    summary = split_dataset(cfg, log=lambda _: None)
    assert summary["audit"]["cross_split_near_duplicates"]
