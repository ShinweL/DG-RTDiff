from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


def _sample_mean(values: torch.Tensor) -> torch.Tensor:
    return values.flatten(1).mean(dim=1)


def _sample_top_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    flat = values.flatten(1)
    count = max(1, int(round(flat.shape[1] * fraction)))
    return flat.topk(count, dim=1).values.mean(dim=1)


def windowed_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 4.0,
    window_size: int = 11,
) -> torch.Tensor:
    """Dependency-free, uniform-window SSIM computed per image."""
    padding = window_size // 2
    mu_prediction = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    variance_prediction = F.avg_pool2d(prediction.square(), window_size, 1, padding) - mu_prediction.square()
    variance_target = F.avg_pool2d(target.square(), window_size, 1, padding) - mu_target.square()
    covariance = F.avg_pool2d(prediction * target, window_size, 1, padding) - mu_prediction * mu_target
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_prediction * mu_target + c1) * (2.0 * covariance + c2)
    denominator = (
        (mu_prediction.square() + mu_target.square() + c1)
        * (variance_prediction + variance_target + c2)
    )
    return (numerator / denominator.clamp_min(1e-12)).flatten(1).mean(dim=1)


def composite_batch_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    absolute_error = (prediction - target).abs()
    roi_denominator = (mask * target.abs()).flatten(1).sum(dim=1) + eps
    nmae = (mask * absolute_error).flatten(1).sum(dim=1) / roi_denominator
    trr = (mask * prediction.abs()).flatten(1).sum(dim=1) / roi_denominator
    target_energy = (mask * prediction.square()).flatten(1).sum(dim=1) / (
        mask.flatten(1).sum(dim=1) + eps
    )
    background = 1.0 - mask
    background_energy = (background * prediction.square()).flatten(1).sum(dim=1) / (
        background.flatten(1).sum(dim=1) + eps
    )
    tbr = 10.0 * torch.log10((target_energy + eps) / (background_energy + eps))
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    psnr = 10.0 * torch.log10(torch.full_like(mse, 16.0) / (mse + eps))
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    return {
        "nmae_roi": nmae,
        "trr": trr,
        "trr_error": (trr - 1.0).abs(),
        "tbr_db": tbr,
        "mae": _sample_mean(absolute_error),
        "psnr": psnr,
        "ssim": windowed_ssim(prediction, target, data_range=4.0),
        "gradient_y_mae": _sample_mean((pred_y - target_y).abs()),
        "gradient_x_mae": _sample_mean((pred_x - target_x).abs()),
    }


def empty_batch_metrics(
    prediction: torch.Tensor, threshold: float = 0.1
) -> dict[str, torch.Tensor]:
    absolute = prediction.abs()
    return {
        "empty_mean": _sample_mean(absolute),
        "empty_top1": _sample_top_mean(absolute, 0.01),
        "pfr": _sample_mean((absolute > threshold).float()),
    }


def weak_m_batch_metrics(
    prediction: torch.Tensor,
    prior_probability: torch.Tensor | None,
    box_mask: torch.Tensor,
    boxes: torch.Tensor,
    box_valid: torch.Tensor,
    apexes: torch.Tensor,
    apex_valid: torch.Tensor,
    eps: float = 1e-8,
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    if prior_probability is not None and prior_probability.shape[-2:] != prediction.shape[-2:]:
        prior_probability = torch.nn.functional.interpolate(
            prior_probability, size=prediction.shape[-2:], mode="bilinear", align_corners=False
        )
    for batch_index in range(prediction.shape[0]):
        absolute = prediction[batch_index, 0].abs()
        union = box_mask[batch_index, 0] > 0.5
        inside = absolute[union]
        outside = absolute[~union]
        inside_top = (
            inside.topk(max(1, int(round(inside.numel() * 0.1)))).values.mean()
            if inside.numel()
            else absolute.new_zeros(())
        )
        outside_mean = outside.mean() if outside.numel() else absolute.new_zeros(())
        apex_distances: list[float] = []
        if prior_probability is not None:
            prior = prior_probability[batch_index, 0]
            height, width = prior.shape
            for object_index in range(boxes.shape[1]):
                if not bool(box_valid[batch_index, object_index]):
                    continue
                x1, y1, x2, y2 = boxes[batch_index, object_index]
                left = max(0, min(width - 1, int(torch.floor(x1).item())))
                top = max(0, min(height - 1, int(torch.floor(y1).item())))
                right = max(left + 1, min(width, int(torch.ceil(x2).item())))
                bottom = max(top + 1, min(height, int(torch.ceil(y2).item())))
                region = prior[top:bottom, left:right]
                flat_index = int(region.argmax().item())
                peak_y = top + flat_index // region.shape[1]
                peak_x = left + flat_index % region.shape[1]
                if bool(apex_valid[batch_index, object_index]):
                    apex_x, apex_y = apexes[batch_index, object_index]
                    distance = float(torch.sqrt((apex_x - peak_x) ** 2 + (apex_y - peak_y) ** 2))
                    apex_distances.append(distance)
        results.append(
            {
                "box_top10": float(inside_top),
                "outside_mean": float(outside_mean),
                "box_background_ratio": float((inside_top + eps) / (outside_mean + eps)),
                "apex_distance": float(np.mean(apex_distances)) if apex_distances else float("nan"),
            }
        )
    return results


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    keys = sorted({key for record in records for key in record})
    result: dict[str, float] = {}
    for key in keys:
        try:
            values = np.asarray([record[key] for record in records if key in record], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        finite = values[np.isfinite(values)]
        if finite.size:
            result[key] = float(finite.mean())
            result[f"{key}_std"] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
            result[f"{key}_count"] = float(finite.size)
    result["count"] = float(len(records))
    return result