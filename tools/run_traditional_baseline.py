from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dg_rtdiff.config import load_config, resolve_path
from dg_rtdiff.data import FixedCompositeDataset, R1WindowDataset, RoadEmptyDataset
from dg_rtdiff.metrics import aggregate_records, composite_batch_metrics, empty_batch_metrics
from dg_rtdiff.utils import save_preview, save_signed16, write_json


def soft_threshold(array: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(array) * np.maximum(np.abs(array) - threshold, 0.0)


def robust_pca(
    matrix: np.ndarray,
    lam: float | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Inexact augmented-Lagrange Principal Component Pursuit."""
    rows, cols = matrix.shape
    lam = 1.0 / np.sqrt(max(rows, cols)) if lam is None else lam
    spectral_norm = np.linalg.norm(matrix, 2)
    infinity_norm = np.linalg.norm(matrix.ravel(), np.inf) / lam
    dual = matrix / max(spectral_norm, infinity_norm, 1e-12)
    low_rank = np.zeros_like(matrix)
    sparse = np.zeros_like(matrix)
    mu = 1.25 / max(spectral_norm, 1e-12)
    mu_max = mu * 1e7
    rho = 1.5
    normalizer = max(np.linalg.norm(matrix, "fro"), 1e-12)
    for _ in range(max_iterations):
        u, singular, vt = np.linalg.svd(matrix - sparse + dual / mu, full_matrices=False)
        kept = singular > 1.0 / mu
        if kept.any():
            low_rank = (u[:, kept] * (singular[kept] - 1.0 / mu)) @ vt[kept]
        else:
            low_rank.fill(0.0)
        sparse = soft_threshold(matrix - low_rank + dual / mu, lam / mu)
        residual = matrix - low_rank - sparse
        dual = dual + mu * residual
        mu = min(mu * rho, mu_max)
        if np.linalg.norm(residual, "fro") / normalizer < tolerance:
            break
    return low_rank, sparse


def predict(image: torch.Tensor, method: str, rank: int, rpca_lambda: float | None) -> torch.Tensor:
    if method == "mean_scan":
        return image - image.mean(dim=-1, keepdim=True)
    if method == "svd":
        matrix = image[0, 0]
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        used_rank = min(rank, singular.numel())
        background = (u[:, :used_rank] * singular[:used_rank]) @ vh[:used_rank]
        return (matrix - background).unsqueeze(0).unsqueeze(0)
    if method == "rpca":
        matrix = image[0, 0].cpu().numpy().astype(np.float64)
        _, sparse = robust_pca(matrix, lam=rpca_lambda)
        return torch.from_numpy(sparse.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    raise ValueError(method)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PNG-domain traditional clutter baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True, choices=["mean_scan", "svd", "rpca"])
    parser.add_argument("--dataset", required=True, choices=["composite", "r2", "r1"])
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--rpca-lambda", type=float, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="Development-only sample cap.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = config["data"]
    splits = resolve_path(data["source_splits"])
    if args.dataset == "composite":
        dataset = FixedCompositeDataset(resolve_path(data["fixed_root"]), args.split, max_samples=args.max_samples)
    elif args.dataset == "r2":
        dataset = RoadEmptyDataset(resolve_path(data["r2_root"]), splits, args.split, max_samples=args.max_samples)
    else:
        manifest = data.get("r1_window_manifest")
        if not manifest:
            raise ValueError("Set data.r1_window_manifest before R1 evaluation")
        dataset = R1WindowDataset(resolve_path(data["r1_root"]), resolve_path(manifest))
        if args.max_samples is not None:
            dataset.rows = dataset.rows[: args.max_samples]
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for batch in tqdm(loader, desc=f"{args.method}-{args.dataset}"):
        prediction = predict(batch["image"], args.method, args.rank, args.rpca_lambda)
        if args.dataset == "composite" and not bool(batch["is_empty"][0]):
            metric_tensors = composite_batch_metrics(prediction, batch["target"], batch["mask"])
        else:
            metric_tensors = empty_batch_metrics(
                prediction, threshold=float(config["evaluation"].get("pfr_threshold", 0.1))
            )
        record = {key: float(value[0]) for key, value in metric_tensors.items()}
        record.update({"sample_id": batch["sample_id"][0], "domain": args.dataset, "method": args.method})
        records.append(record)
        if args.save_predictions:
            sample_dir = args.output / "predictions" / batch["sample_id"][0]
            save_preview(batch["image"][0], sample_dir / "input.png")
            save_signed16(prediction[0], sample_dir / "target_signed16.png")
            save_preview(prediction[0], sample_dir / "target_preview.png")
    fields = sorted({key for record in records for key in record})
    with (args.output / "per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = aggregate_records(records)
    summary.update({"method": args.method, "dataset": args.dataset, "split": args.split, "rank": args.rank, "rpca_lambda": args.rpca_lambda})
    write_json(summary, args.output / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()