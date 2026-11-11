from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dg_rtdiff.config import load_config, resolve_path
from dg_rtdiff.data import FixedCompositeDataset, R1WindowDataset, RoadEmptyDataset, WeakMDataset
from dg_rtdiff.engine import EMA, build_diffusion, load_weights
from dg_rtdiff.evaluation import evaluate_composite, evaluate_empty, evaluate_m
from dg_rtdiff.models import build_model
from dg_rtdiff.utils import choose_device, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen DG-RTDiff checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datasets", nargs="+", choices=["composite", "m", "r2", "r1"], default=["composite", "m", "r2"])
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-ema", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 20260716))
    seed_everything(seed)
    device = choose_device(config.get("device", "auto"), bool(config.get("require_cuda", False)))
    model = build_model(config["model"]).to(device)
    diffusion = build_diffusion(config, device)
    ema = EMA(model, float(config["training"].get("ema_decay", 0.9999)))
    load_weights(args.checkpoint, model, ema)
    if not args.no_ema:
        ema.apply(model)
    model.eval()
    method = config["model"]["method"]
    diffusion_config = config.get("diffusion", {})
    sampling_steps = int(diffusion_config.get("sampling_steps", 1))
    eta = float(diffusion_config.get("eta", 0.0))
    output = Path(args.output).resolve() if args.output else Path(args.checkpoint).resolve().parents[1] / "predictions" / args.split
    output.mkdir(parents=True, exist_ok=True)
    data = config["data"]
    fixed_root = resolve_path(data["fixed_root"])
    source_splits = resolve_path(data["source_splits"])
    batch_size = int(config["evaluation"].get("batch_size", 1))
    loader_args = {"batch_size": batch_size, "shuffle": False, "num_workers": min(4, int(config["loader"].get("num_workers", 0)))}
    results = {}
    if "composite" in args.datasets:
        loader = DataLoader(FixedCompositeDataset(fixed_root, args.split), **loader_args)
        results["composite"] = evaluate_composite(
            model=model, diffusion=diffusion, loader=loader, device=device, method=method,
            sampling_steps=sampling_steps, eta=eta, seed=seed,
            output_dir=output / "composite", save_predictions=True,
        )
    if "m" in args.datasets:
        loader = DataLoader(WeakMDataset(resolve_path(data["m_root"]), source_splits, args.split), **loader_args)
        results["m"] = evaluate_m(
            model=model, diffusion=diffusion, loader=loader, device=device, method=method,
            sampling_steps=sampling_steps, eta=eta, seed=seed + 200_000,
            output_dir=output / "m", save_predictions=True,
        )
    if "r2" in args.datasets:
        loader = DataLoader(RoadEmptyDataset(resolve_path(data["r2_root"]), source_splits, args.split), **loader_args)
        results["r2"] = evaluate_empty(
            model=model, diffusion=diffusion, loader=loader, device=device, method=method,
            sampling_steps=sampling_steps, eta=eta, seed=seed + 100_000,
            threshold=float(config["evaluation"].get("pfr_threshold", 0.1)), domain="r2",
            output_dir=output / "r2", save_predictions=True,
        )
    if "r1" in args.datasets:
        manifest = data.get("r1_window_manifest")
        if not manifest:
            raise ValueError("Set data.r1_window_manifest before R1 evaluation")
        loader = DataLoader(R1WindowDataset(resolve_path(data["r1_root"]), resolve_path(manifest)), **loader_args)
        results["r1"] = evaluate_empty(
            model=model, diffusion=diffusion, loader=loader, device=device, method=method,
            sampling_steps=sampling_steps, eta=eta, seed=seed + 300_000,
            threshold=float(config["evaluation"].get("pfr_threshold", 0.1)), domain="r1",
            output_dir=output / "r1", save_predictions=True,
        )
    write_json(results, output / "summary.json")
    print(results)


if __name__ == "__main__":
    main()
