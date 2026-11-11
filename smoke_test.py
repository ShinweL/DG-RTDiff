from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from dg_rtdiff.config import load_config, resolve_path
from dg_rtdiff.data import FixedCompositeDataset, RoadEmptyDataset, WeakMDataset
from dg_rtdiff.diffusion import GaussianDiffusion
from dg_rtdiff.losses import diffusion_training_losses, weak_mil_prior_loss
from dg_rtdiff.models import build_model
from dg_rtdiff.utils import count_parameters, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small CPU smoke test; never runs formal training.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 20260716))
    seed_everything(seed)
    device = torch.device("cpu")
    fixed_root = resolve_path(config["data"]["fixed_root"])
    split_path = resolve_path(config["data"]["source_splits"])
    crop = int(config["data"].get("debug_crop_size", 64))
    composite = FixedCompositeDataset(fixed_root, "train", debug_crop_size=crop, max_samples=2)
    r2 = RoadEmptyDataset(
        resolve_path(config["data"]["r2_root"]), split_path, "train", debug_crop_size=crop, max_samples=1
    )
    weak_m = WeakMDataset(resolve_path(config["data"]["m_root"]), split_path, "train", max_samples=1)
    composite_batch = next(iter(DataLoader(composite, batch_size=1)))
    r2_batch = next(iter(DataLoader(r2, batch_size=1)))
    m_batch = next(iter(DataLoader(weak_m, batch_size=1)))

    model = build_model(config["model"]).to(device)
    diffusion = GaussianDiffusion(
        timesteps=int(config["diffusion"]["timesteps"]),
        schedule=config["diffusion"]["schedule"],
        clip_x0=float(config["diffusion"]["clip_x0"]),
    ).to(device)
    image = composite_batch["image"]
    target = composite_batch["target"]
    time = torch.tensor([3], dtype=torch.long)
    noisy, noise = diffusion.q_sample(target, time)
    output = model(noisy, image, time)
    v_target = diffusion.v_target(target, noise, time)
    x0 = diffusion.predict_x0(noisy, output["v"], time)
    losses = diffusion_training_losses(
        v_prediction=output["v"],
        v_target=v_target,
        x0_prediction=x0,
        target=target,
        mask=composite_batch["mask"],
        prior_logits=output["prior_logits"],
        is_empty=composite_batch["is_empty"],
        weights=config["loss"],
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])

    model.zero_grad(set_to_none=True)
    cache = model.encode_condition(m_batch["image"])
    mil = weak_mil_prior_loss(cache["prior_logits"], m_batch["box_mask"])
    mil.backward()
    assert torch.isfinite(mil)

    with torch.no_grad():
        sample, _ = diffusion.ddim_sample(
            model,
            r2_batch["image"],
            steps=int(config["diffusion"]["sampling_steps"]),
            eta=0.0,
            generator=torch.Generator().manual_seed(seed),
        )
    assert sample.shape == r2_batch["image"].shape
    assert torch.isfinite(sample).all()
    print(
        {
            "status": "SMOKE_OK",
            "parameters": count_parameters(model),
            "composite_shape": tuple(image.shape),
            "m_shape": tuple(m_batch["image"].shape),
            "r2_shape": tuple(r2_batch["image"].shape),
            "loss": float(losses["total"].detach()),
            "mil": float(mil.detach()),
            "sample_range": (float(sample.min()), float(sample.max())),
        }
    )


if __name__ == "__main__":
    main()
