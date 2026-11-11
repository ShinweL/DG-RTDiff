from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .common import center_crop_tensor, read_input_png


class RoadEmptyDataset(Dataset[dict[str, Any]]):
    """Author-curated R2 empty backgrounds using the frozen source split."""

    def __init__(
        self,
        root: str | Path,
        source_splits: str | Path,
        split: str,
        horizontal_flip: bool = False,
        debug_crop_size: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.horizontal_flip = horizontal_flip
        self.debug_crop_size = debug_crop_size
        with Path(source_splits).expanduser().resolve().open("r", encoding="utf-8") as handle:
            split_data = json.load(handle)
        files = [self.root / filename for filename in split_data["r2_background_files"][split]]
        self.files = files[:max_samples] if max_samples is not None else files
        missing = [path for path in self.files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing R2 file: {missing[0]}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        image = center_crop_tensor(read_input_png(path), self.debug_crop_size)
        if self.horizontal_flip and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(-1,))
        zeros = torch.zeros_like(image)
        return {
            "image": image.contiguous(),
            "target": zeros,
            "mask": zeros,
            "is_empty": torch.tensor(1.0),
            "sample_id": path.stem,
            "kind": "empty",
            "domain": "r2",
        }


class R1WindowDataset(Dataset[dict[str, Any]]):
    """Frozen windows from wide Shenzhen screenshots.

    The CSV must contain: window_id,image_path,left,top,right,bottom. Coordinates
    are created only after the author confirms the valid radar ROI.
    """

    REQUIRED_COLUMNS = {"window_id", "image_path", "left", "top", "right", "bottom"}

    def __init__(self, root: str | Path, window_manifest: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest = Path(window_manifest).expanduser().resolve()
        if not manifest.exists():
            raise FileNotFoundError(
                f"R1 window manifest is required; do not resize the full screenshot: {manifest}"
            )
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not self.REQUIRED_COLUMNS.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"R1 manifest requires columns: {sorted(self.REQUIRED_COLUMNS)}")
            self.rows = list(reader)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = self.root / image_path
        with Image.open(image_path) as source:
            crop = source.convert("L").crop(
                (int(row["left"]), int(row["top"]), int(row["right"]), int(row["bottom"]))
            )
            if crop.size != (320, 320):
                raise ValueError(f"R1 window must be 320x320, got {crop.size} for {row['window_id']}")
            import numpy as np

            array = np.asarray(crop, dtype=np.float32)
        image = torch.from_numpy(array / 127.5 - 1.0).unsqueeze(0)
        zeros = torch.zeros_like(image)
        return {
            "image": image,
            "target": zeros,
            "mask": zeros,
            "is_empty": torch.tensor(1.0),
            "sample_id": row["window_id"],
            "kind": "empty",
            "domain": "r1",
        }
