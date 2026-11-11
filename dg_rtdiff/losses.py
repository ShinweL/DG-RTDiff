from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def _top_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    flat = values.flatten(1)
    count = max(1, int(round(flat.shape[1] * fraction)))
    return flat.topk(count, dim=1).values.mean(dim=1)


def dense_prior_loss(logits: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    probability = logits.sigmoid()
    intersection = (probability * mask).flatten(1).sum(dim=1)
    denominator = probability.flatten(1).sum(dim=1) + mask.flatten(1).sum(dim=1)
    dice = 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()
    return bce + dice


def directional_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    return F.l1_loss(pred_y, target_y) + 0.5 * F.l1_loss(pred_x, target_x)


def empty_response_loss(prediction: torch.Tensor, empty_selector: torch.Tensor) -> torch.Tensor:
    selector = empty_selector.bool().flatten()
    if not selector.any():
        return prediction.new_zeros(())
    empty_prediction = prediction[selector].abs()
    return empty_prediction.mean() + _top_mean(empty_prediction, 0.01).mean()


def diffusion_training_losses(
    *,
    v_prediction: torch.Tensor,
    v_target: torch.Tensor,
    x0_prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    prior_logits: torch.Tensor | None,
    is_empty: torch.Tensor,
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    kappa = float(weights.get("target_weight_kappa", 4.0))
    weighted_diffusion = ((1.0 + kappa * mask) * (v_prediction - v_target).square()).mean()
    reconstruction = F.l1_loss(x0_prediction, target)
    nonempty = ~is_empty.bool().flatten()
    direction = (
        directional_loss(x0_prediction[nonempty], target[nonempty])
        if nonempty.any()
        else target.new_zeros(())
    )
    prior = (
        dense_prior_loss(prior_logits, mask)
        if prior_logits is not None
        else target.new_zeros(())
    )
    empty = empty_response_loss(x0_prediction, is_empty)
    total = (
        weighted_diffusion
        + float(weights.get("lambda_rec", 1.0)) * reconstruction
        + float(weights.get("lambda_dir", 0.2)) * direction
        + float(weights.get("lambda_prior", 0.1)) * prior
        + float(weights.get("lambda_empty", 0.2)) * empty
    )
    return {
        "total": total,
        "diffusion": weighted_diffusion.detach(),
        "reconstruction": reconstruction.detach(),
        "direction": direction.detach(),
        "prior": prior.detach(),
        "empty": empty.detach(),
    }


def unet_training_losses(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    is_empty: torch.Tensor,
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    kappa = float(weights.get("target_weight_kappa", 4.0))
    reconstruction = ((1.0 + kappa * mask) * (prediction - target).abs()).mean()
    nonempty = ~is_empty.bool().flatten()
    direction = (
        directional_loss(prediction[nonempty], target[nonempty])
        if nonempty.any()
        else target.new_zeros(())
    )
    empty = empty_response_loss(prediction, is_empty)
    total = (
        float(weights.get("lambda_rec", 1.0)) * reconstruction
        + float(weights.get("lambda_dir", 0.2)) * direction
        + float(weights.get("lambda_empty", 0.2)) * empty
    )
    return {
        "total": total,
        "diffusion": target.new_zeros(()),
        "reconstruction": reconstruction.detach(),
        "direction": direction.detach(),
        "prior": target.new_zeros(()),
        "empty": empty.detach(),
    }


def weak_mil_prior_loss(
    logits: torch.Tensor,
    box_mask: torch.Tensor,
    top_fraction: float = 0.1,
    outside_margin: int = 9,
    eps: float = 1e-6,
) -> torch.Tensor:
    probability = logits.sigmoid()
    if probability.shape[-2:] != box_mask.shape[-2:]:
        probability = F.interpolate(probability, size=box_mask.shape[-2:], mode="bilinear", align_corners=False)
    kernel = outside_margin if outside_margin % 2 else outside_margin + 1
    expanded = F.max_pool2d(box_mask, kernel_size=kernel, stride=1, padding=kernel // 2)
    losses: list[torch.Tensor] = []
    for sample_probability, sample_box, sample_expanded in zip(probability, box_mask, expanded):
        positive = sample_probability[sample_box > 0.5]
        negative = sample_probability[sample_expanded < 0.5]
        if positive.numel() == 0:
            continue
        count = max(1, int(round(positive.numel() * top_fraction)))
        positive_score = positive.topk(count).values.mean()
        negative_score = negative.mean() if negative.numel() else positive_score.new_zeros(())
        losses.append(-torch.log(positive_score + eps) - torch.log(1.0 - negative_score + eps))
    if not losses:
        raise ValueError("M MIL batch contains no valid target-box pixels")
    return torch.stack(losses).mean()


def detach_loss_dict(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in losses.items()}
