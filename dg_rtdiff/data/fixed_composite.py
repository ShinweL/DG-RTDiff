from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .common import center_crop_tensor, read_input_png, read_mask_png, read_target_residual_png


class FixedCompositeDataset(Dataset[dict[str, Any]]):
    """Read the frozen M-to-R2 composite PNG dataset without online synthesis."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        horizontal_flip: bool = False,
        debug_crop_size: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.horizontal_flip = horizontal_flip
        self.debug_crop_size = debug_crop_size
        manifest_path = self.root / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError(f"No rows for split={split!r} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = read_input_png(self.root / Path(row["input_png"]))
        target = read_target_residual_png(self.root / Path(row["target_residual_png"]))
        mask = read_mask_png(self.root / Path(row["mask_png"]))
        image = center_crop_tensor(image, self.debug_crop_size)
        target = center_crop_tensor(target, self.debug_crop_size)
        mask = center_crop_tensor(mask, self.debug_crop_size)
        if self.horizontal_flip and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(-1,))
            target = torch.flip(target, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        is_empty = float(row["kind"] == "empty")
        return {
            "image": image.contiguous(),
            "target": target.contiguous(),
            "mask": mask.contiguous(),
            "is_empty": torch.tensor(is_empty, dtype=torch.float32),
            "sample_id": row["sample_id"],
            "kind": row["kind"],
            "domain": "composite",
        }
