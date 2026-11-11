from pathlib import Path

import torch

from dg_rtdiff.data import FixedCompositeDataset, RoadEmptyDataset, WeakMDataset


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT.parent / "data"
FIXED = DATA / "fixed_composite_dataset_v2_gain18_lower"


def test_fixed_dataset_decodes_real_pngs() -> None:
    dataset = FixedCompositeDataset(FIXED, "train", max_samples=16)
    sample = dataset[0]
    assert sample["image"].shape == (1, 320, 320)
    assert sample["target"].shape == sample["image"].shape
    assert sample["mask"].shape == sample["image"].shape
    assert sample["image"].dtype == torch.float32
    assert torch.isfinite(sample["target"]).all()


def test_empty_composite_is_exact_zero() -> None:
    dataset = FixedCompositeDataset(FIXED, "train", max_samples=32)
    empty = next(sample for sample in (dataset[index] for index in range(len(dataset))) if sample["kind"] == "empty")
    assert torch.count_nonzero(empty["target"]) == 0
    assert torch.count_nonzero(empty["mask"]) == 0


def test_m_and_r2_splits_are_readable() -> None:
    splits = FIXED / "source_splits.json"
    m = WeakMDataset(DATA / "M", splits, "train", max_samples=1)[0]
    r2 = RoadEmptyDataset(DATA / "R2", splits, "train", max_samples=1)[0]
    assert m["image"].shape == (1, 128, 128)
    assert m["box_mask"].sum() > 0
    assert r2["image"].shape == (1, 320, 320)
    assert r2["is_empty"].item() == 1.0
