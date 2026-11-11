from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def choose_device(requested: str = "auto", require_cuda: bool = False) -> torch.device:
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required by this config, but no CUDA device is available.")
    return device


def to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=device.type == "cuda")
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(to_device(value, device) for value in batch)
    return batch


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(data: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


class CSVLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict[str, Any]) -> None:
        exists = self.path.exists()
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def cycle_loader(loader: Iterable[Any]) -> Iterable[Any]:
    while True:
        yield from loader


def tensor_to_uint8_preview(tensor: torch.Tensor, symmetric: bool = True) -> np.ndarray:
    array = tensor.detach().float().cpu().squeeze().numpy()
    if symmetric:
        array = np.clip((array + 1.0) * 127.5, 0, 255)
    else:
        low, high = np.percentile(array, [1, 99])
        if high <= low:
            high = low + 1.0
        array = np.clip((array - low) / (high - low) * 255.0, 0, 255)
    return array.astype(np.uint8)


def save_preview(tensor: torch.Tensor, path: str | Path, symmetric: bool = True) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_preview(tensor, symmetric=symmetric), mode="L").save(output)


def save_signed16(tensor: torch.Tensor, path: str | Path) -> None:
    """Store normalized residual using the dataset's signed 16-bit encoding."""
    residual_gray = tensor.detach().float().cpu().squeeze().numpy() * 128.0
    encoded = np.clip(np.rint(residual_gray * 128.0 + 32768.0), 0, 65535).astype(np.uint16)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded, mode="I;16").save(output)


def environment_summary(device: torch.device) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "python_pid": os.getpid(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        summary.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory": props.total_memory,
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return summary
