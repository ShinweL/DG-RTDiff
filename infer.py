from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dg_rtdiff.config import load_config
from dg_rtdiff.engine import EMA, build_diffusion, load_weights
from dg_rtdiff.evaluation import predict_target
from dg_rtdiff.models import build_model
from dg_rtdiff.utils import choose_device, save_preview, save_signed16, seed_everything


def positions(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DG-RTDiff inference on cropped radar PNGs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("L", "T", "R", "B"))
    parser.add_argument("--tile-size", type=int, default=320)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--allow-full-image", action="store_true", help="Allow tiling a non-320 image without an explicit ROI.")
    parser.add_argument("--no-ema", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def tiled_prediction(
    image: torch.Tensor,
    model: torch.nn.Module,
    diffusion,
    method: str,
    steps: int,
    eta: float,
    seed: int,
    tile_size: int,
    overlap: int,
) -> torch.Tensor:
    _, _, height, width = image.shape
    pad_bottom = max(0, tile_size - height)
    pad_right = max(0, tile_size - width)
    if pad_bottom or pad_right:
        image = torch.nn.functional.pad(image, (0, pad_right, 0, pad_bottom), mode="replicate")
    padded_height, padded_width = image.shape[-2:]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile-size")
    output = torch.zeros_like(image)
    weight = torch.zeros_like(image)
    tile_index = 0
    for top in positions(padded_height, tile_size, stride):
        for left in positions(padded_width, tile_size, stride):
            tile = image[..., top : top + tile_size, left : left + tile_size]
            prediction, _ = predict_target(
                model, diffusion, tile, method, steps, eta, seed + tile_index
            )
            output[..., top : top + tile_size, left : left + tile_size] += prediction
            weight[..., top : top + tile_size, left : left + tile_size] += 1.0
            tile_index += 1
    return (output / weight.clamp_min(1.0))[..., :height, :width]


def input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted({item.resolve() for item in [*path.glob("*.png"), *path.glob("*.PNG")]})


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
    steps = int(diffusion_config.get("sampling_steps", 1))
    eta = float(diffusion_config.get("eta", 0.0))
    args.output.mkdir(parents=True, exist_ok=True)

    files = input_files(args.input)
    if not files:
        raise FileNotFoundError(f"No PNG files found: {args.input}")
    for file_index, path in enumerate(files):
        with Image.open(path) as source:
            source = source.convert("L")
            if args.roi:
                source = source.crop(tuple(args.roi))
            elif source.size != (args.tile_size, args.tile_size) and not args.allow_full_image:
                raise ValueError(
                    f"{path.name} is {source.size}. Supply --roi for wide screenshots or explicitly pass --allow-full-image."
                )
            array = np.asarray(source, dtype=np.float32).copy()
        image = torch.from_numpy(array / 127.5 - 1.0).unsqueeze(0).unsqueeze(0).to(device)
        prediction = tiled_prediction(
            image,
            model,
            diffusion,
            method,
            steps,
            eta,
            seed + file_index * 10_000,
            args.tile_size,
            args.overlap,
        )
        sample_dir = args.output / path.stem
        save_preview(image[0], sample_dir / "input.png")
        save_signed16(prediction[0], sample_dir / "target_signed16.png")
        save_preview(prediction[0], sample_dir / "target_preview.png")
        save_preview((image - prediction)[0], sample_dir / "removed_preview.png", symmetric=False)
    print(f"Processed {len(files)} image(s) into {args.output}")


if __name__ == "__main__":
    main()