from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .diffusion import GaussianDiffusion
from .metrics import (
    aggregate_records,
    composite_batch_metrics,
    empty_batch_metrics,
    weak_m_batch_metrics,
)
from .utils import save_preview, save_signed16, to_device, write_json


@torch.no_grad()
def predict_target(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion | None,
    image: torch.Tensor,
    method: str,
    sampling_steps: int,
    eta: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if method == "unet":
        output = model(image)
        return output["pred"], output.get("prior_logits")
    if diffusion is None:
        raise ValueError("Diffusion object is required for diffusion methods")
    generator = torch.Generator(device=image.device)
    generator.manual_seed(seed)
    return diffusion.ddim_sample(
        model,
        image,
        steps=sampling_steps,
        eta=eta,
        generator=generator,
    )


def _rows_from_tensor_metrics(metrics: dict[str, torch.Tensor]) -> list[dict[str, float]]:
    batch_size = next(iter(metrics.values())).shape[0]
    return [
        {key: float(value[index].detach().cpu()) for key, value in metrics.items()}
        for index in range(batch_size)
    ]


def _save_csv(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in records for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


@torch.no_grad()
def evaluate_composite(
    *,
    model: torch.nn.Module,
    diffusion: GaussianDiffusion | None,
    loader: DataLoader,
    device: torch.device,
    method: str,
    sampling_steps: int,
    eta: float,
    seed: int,
    output_dir: str | Path | None = None,
    save_predictions: bool = False,
) -> dict[str, float]:
    model.eval()
    records: list[dict[str, Any]] = []
    output_path = Path(output_dir) if output_dir is not None else None
    sample_index = 0
    for batch in tqdm(loader, desc="composite-eval", leave=False):
        batch = to_device(batch, device)
        prediction, prior_logits = predict_target(
            model, diffusion, batch["image"], method, sampling_steps, eta, seed + sample_index
        )
        target_records = _rows_from_tensor_metrics(
            composite_batch_metrics(prediction, batch["target"], batch["mask"])
        )
        empty_records = _rows_from_tensor_metrics(
            empty_batch_metrics(prediction, threshold=0.1)
        )
        ids = batch["sample_id"]
        for local_index in range(prediction.shape[0]):
            record = empty_records[local_index] if bool(batch["is_empty"][local_index]) else target_records[local_index]
            record.update({"sample_id": ids[local_index], "domain": "composite"})
            records.append(record)
            if save_predictions and output_path is not None:
                sample_dir = output_path / ids[local_index]
                save_preview(batch["image"][local_index], sample_dir / "input.png")
                save_signed16(prediction[local_index], sample_dir / "target_signed16.png")
                save_preview(prediction[local_index], sample_dir / "target_preview.png")
                save_preview(batch["target"][local_index], sample_dir / "target_truth.png")
                if prior_logits is not None:
                    save_preview(prior_logits[local_index].sigmoid() * 2 - 1, sample_dir / "prior.png")
        sample_index += prediction.shape[0]
    if output_path is not None:
        _save_csv(records, output_path / "per_image.csv")
        write_json(aggregate_records(records), output_path / "summary.json")
    return aggregate_records(records)


@torch.no_grad()
def evaluate_empty(
    *,
    model: torch.nn.Module,
    diffusion: GaussianDiffusion | None,
    loader: DataLoader,
    device: torch.device,
    method: str,
    sampling_steps: int,
    eta: float,
    seed: int,
    threshold: float,
    domain: str,
    output_dir: str | Path | None = None,
    save_predictions: bool = False,
) -> dict[str, float]:
    model.eval()
    records: list[dict[str, Any]] = []
    output_path = Path(output_dir) if output_dir is not None else None
    sample_index = 0
    for batch in tqdm(loader, desc=f"{domain}-eval", leave=False):
        batch = to_device(batch, device)
        prediction, _ = predict_target(
            model, diffusion, batch["image"], method, sampling_steps, eta, seed + sample_index
        )
        batch_records = _rows_from_tensor_metrics(empty_batch_metrics(prediction, threshold))
        ids = batch["sample_id"]
        for local_index, record in enumerate(batch_records):
            record.update({"sample_id": ids[local_index], "domain": domain})
            records.append(record)
            if save_predictions and output_path is not None:
                sample_dir = output_path / ids[local_index]
                save_preview(batch["image"][local_index], sample_dir / "input.png")
                save_signed16(prediction[local_index], sample_dir / "target_signed16.png")
                save_preview(prediction[local_index], sample_dir / "target_preview.png")
        sample_index += prediction.shape[0]
    if output_path is not None:
        _save_csv(records, output_path / "per_image.csv")
        write_json(aggregate_records(records), output_path / "summary.json")
    return aggregate_records(records)


@torch.no_grad()
def evaluate_m(
    *,
    model: torch.nn.Module,
    diffusion: GaussianDiffusion | None,
    loader: DataLoader,
    device: torch.device,
    method: str,
    sampling_steps: int,
    eta: float,
    seed: int,
    output_dir: str | Path | None = None,
    save_predictions: bool = False,
) -> dict[str, float]:
    model.eval()
    records: list[dict[str, Any]] = []
    output_path = Path(output_dir) if output_dir is not None else None
    sample_index = 0
    for batch in tqdm(loader, desc="m-eval", leave=False):
        batch = to_device(batch, device)
        prediction, prior_logits = predict_target(
            model, diffusion, batch["image"], method, sampling_steps, eta, seed + sample_index
        )
        prior_probability = prior_logits.sigmoid() if prior_logits is not None else None
        batch_records = weak_m_batch_metrics(
            prediction,
            prior_probability,
            batch["box_mask"],
            batch["boxes"],
            batch["box_valid"],
            batch["apexes"],
            batch["apex_valid"],
        )
        ids = batch["sample_id"]
        for local_index, record in enumerate(batch_records):
            record.update({"sample_id": ids[local_index], "domain": "m"})
            records.append(record)
            if save_predictions and output_path is not None:
                sample_dir = output_path / ids[local_index]
                save_preview(batch["image"][local_index], sample_dir / "input.png")
                save_signed16(prediction[local_index], sample_dir / "target_signed16.png")
                save_preview(prediction[local_index], sample_dir / "target_preview.png")
                if prior_probability is not None:
                    save_preview(prior_probability[local_index] * 2 - 1, sample_dir / "prior.png")
        sample_index += prediction.shape[0]
    if output_path is not None:
        _save_csv(records, output_path / "per_image.csv")
        write_json(aggregate_records(records), output_path / "summary.json")
    return aggregate_records(records)