from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def read_input_png(path: str | Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return torch.from_numpy(array / 127.5 - 1.0).unsqueeze(0)


def read_target_residual_png(path: str | Path) -> torch.Tensor:
    array = np.asarray(Image.open(path), dtype=np.int32)
    residual_gray = (array.astype(np.float32) - 32768.0) / 128.0
    # One normalized input unit is approximately 127.5 grayscale levels.
    return torch.from_numpy(residual_gray / 128.0).unsqueeze(0)


def read_mask_png(path: str | Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return torch.from_numpy(array / 255.0).unsqueeze(0)


def center_crop_tensor(tensor: torch.Tensor, size: int | None) -> torch.Tensor:
    if size is None:
        return tensor
    height, width = tensor.shape[-2:]
    if size > height or size > width:
        raise ValueError(f"Debug crop {size} exceeds tensor size {(height, width)}")
    top = (height - size) // 2
    left = (width - size) // 2
    return tensor[..., top : top + size, left : left + size]
