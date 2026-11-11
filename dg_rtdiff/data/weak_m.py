from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .common import read_input_png


class WeakMDataset(Dataset[dict[str, Any]]):
    """Sandbox PNGs with loose target boxes and optional apex points."""

    def __init__(
        self,
        root: str | Path,
        source_splits: str | Path,
        split: str,
        horizontal_flip: bool = False,
        max_objects: int = 8,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.horizontal_flip = horizontal_flip
        self.max_objects = max_objects
        with Path(source_splits).expanduser().resolve().open("r", encoding="utf-8") as handle:
            split_data = json.load(handle)
        allowed = {int(value) for value in split_data["m_source_ids"][split]}
        records: list[tuple[Path, Path]] = []
        for image_path in sorted(self.root.glob("*.png"), key=lambda path: int(path.stem)):
            if int(image_path.stem) not in allowed:
                continue
            json_path = image_path.with_suffix(".json")
            if not json_path.exists():
                continue
            with json_path.open("r", encoding="utf-8") as handle:
                annotation = json.load(handle)
            if any(shape.get("label") == "target" for shape in annotation.get("shapes", [])):
                records.append((image_path, json_path))
        self.records = records[:max_samples] if max_samples is not None else records
        if not self.records:
            raise ValueError(f"No annotated M samples found for split={split!r}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, json_path = self.records[index]
        image = read_input_png(image_path)
        _, height, width = image.shape
        with json_path.open("r", encoding="utf-8") as handle:
            annotation = json.load(handle)

        boxes: list[list[float]] = []
        apexes: list[list[float]] = []
        for shape in annotation.get("shapes", []):
            label = shape.get("label")
            points = shape.get("points", [])
            if label == "target" and len(points) >= 2:
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
            elif label == "apex" and points:
                apexes.append([float(points[0][0]), float(points[0][1])])

        box_mask = torch.zeros((1, height, width), dtype=torch.float32)
        padded_boxes = torch.zeros((self.max_objects, 4), dtype=torch.float32)
        box_valid = torch.zeros((self.max_objects,), dtype=torch.bool)
        for object_index, box in enumerate(boxes[: self.max_objects]):
            x1, y1, x2, y2 = box
            left = max(0, min(width - 1, math.floor(x1)))
            top = max(0, min(height - 1, math.floor(y1)))
            right = max(left + 1, min(width, math.ceil(x2)))
            bottom = max(top + 1, min(height, math.ceil(y2)))
            box_mask[:, top:bottom, left:right] = 1.0
            padded_boxes[object_index] = torch.tensor([x1, y1, x2, y2])
            box_valid[object_index] = True

        padded_apexes = torch.zeros((self.max_objects, 2), dtype=torch.float32)
        apex_valid = torch.zeros((self.max_objects,), dtype=torch.bool)
        for object_index, apex in enumerate(apexes[: self.max_objects]):
            padded_apexes[object_index] = torch.tensor(apex)
            apex_valid[object_index] = True

        if self.horizontal_flip and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(-1,))
            box_mask = torch.flip(box_mask, dims=(-1,))
            valid_boxes = padded_boxes[box_valid].clone()
            padded_boxes[box_valid, 0] = width - 1 - valid_boxes[:, 2]
            padded_boxes[box_valid, 2] = width - 1 - valid_boxes[:, 0]
            padded_apexes[apex_valid, 0] = width - 1 - padded_apexes[apex_valid, 0]

        zeros = torch.zeros_like(image)
        return {
            "image": image.contiguous(),
            "target": zeros,
            "mask": box_mask.contiguous(),
            "box_mask": box_mask.contiguous(),
            "boxes": padded_boxes,
            "box_valid": box_valid,
            "apexes": padded_apexes,
            "apex_valid": apex_valid,
            "is_empty": torch.tensor(0.0),
            "sample_id": image_path.stem,
            "kind": "weak_target",
            "domain": "m",
        }
