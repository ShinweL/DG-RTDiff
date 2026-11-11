from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import resolve_path, save_config
from .data import FixedCompositeDataset, RoadEmptyDataset, WeakMDataset
from .diffusion import GaussianDiffusion
from .evaluation import evaluate_composite, evaluate_empty, evaluate_m
from .losses import (
    detach_loss_dict,
    diffusion_training_losses,
    unet_training_losses,
    weak_mil_prior_loss,
)
from .models import DiffusionUNet, build_model
from .utils import (
    CSVLogger,
    choose_device,
    count_parameters,
    cycle_loader,
    environment_summary,
    seed_everything,
    seed_worker,
    sha256_file,
    to_device,
    write_json,
)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
            if tensor.is_floating_point()
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for name, shadow in self.shadow.items():
            shadow.lerp_(state[name].detach(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = state["shadow"]

    def to(self, device: torch.device) -> None:
        self.shadow = {name: tensor.to(device) for name, tensor in self.shadow.items()}

    @torch.no_grad()
    def apply(self, model: torch.nn.Module) -> None:
        self.backup = {}
        state = model.state_dict()
        for name, shadow in self.shadow.items():
            self.backup[name] = state[name].detach().clone()
            state[name].copy_(shadow.to(state[name].device))

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for name, value in self.backup.items():
            state[name].copy_(value)
        self.backup = {}


def _loader_kwargs(config: dict[str, Any], seed: int, shuffle: bool) -> dict[str, Any]:
    workers = int(config["loader"].get("num_workers", 0))
    generator = torch.Generator().manual_seed(seed)
    return {
        "batch_size": int(config["loader"]["batch_size"]),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": bool(config["loader"].get("pin_memory", False)),
        "persistent_workers": bool(config["loader"].get("persistent_workers", False)) and workers > 0,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "drop_last": shuffle,
    }


def build_training_loaders(config: dict[str, Any], seed: int, dry_run: bool = False) -> dict[str, DataLoader]:
    data = config["data"]
    fixed_root = resolve_path(data["fixed_root"])
    source_splits = resolve_path(data["source_splits"])
    debug_crop = data.get("debug_crop_size")
    max_samples = 2 if dry_run else None
    loaders: dict[str, DataLoader] = {}
    composite = FixedCompositeDataset(
        fixed_root,
        "train",
        horizontal_flip=bool(config["loader"].get("horizontal_flip", False)),
        debug_crop_size=debug_crop,
        max_samples=max_samples,
    )
    loaders["composite"] = DataLoader(
        composite, **_loader_kwargs(config, seed, shuffle=True)
    )
    if config["mode"] == "stage_b":
        weak_m = WeakMDataset(
            resolve_path(data["m_root"]),
            source_splits,
            "train",
            horizontal_flip=bool(config["loader"].get("horizontal_flip", False)),
            max_samples=max_samples,
        )
        r2 = RoadEmptyDataset(
            resolve_path(data["r2_root"]),
            source_splits,
            "train",
            horizontal_flip=bool(config["loader"].get("horizontal_flip", False)),
            debug_crop_size=debug_crop,
            max_samples=max_samples,
        )
        loaders["m"] = DataLoader(weak_m, **_loader_kwargs(config, seed + 1, shuffle=True))
        loaders["r2"] = DataLoader(r2, **_loader_kwargs(config, seed + 2, shuffle=True))
    return loaders


def build_validation_loaders(config: dict[str, Any], seed: int) -> dict[str, DataLoader]:
    data = config["data"]
    evaluation = config["evaluation"]
    fixed_root = resolve_path(data["fixed_root"])
    source_splits = resolve_path(data["source_splits"])
    common = {
        "batch_size": int(evaluation.get("batch_size", 1)),
        "shuffle": False,
        "num_workers": min(4, int(config["loader"].get("num_workers", 0))),
        "pin_memory": bool(config["loader"].get("pin_memory", False)),
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed),
    }
    return {
        "composite": DataLoader(
            FixedCompositeDataset(
                fixed_root,
                "val",
                max_samples=evaluation.get("max_composite_samples"),
            ),
            **common,
        ),
        "r2": DataLoader(
            RoadEmptyDataset(
                resolve_path(data["r2_root"]),
                source_splits,
                "val",
                max_samples=evaluation.get("max_empty_samples"),
            ),
            **common,
        ),
        "m": DataLoader(
            WeakMDataset(
                resolve_path(data["m_root"]),
                source_splits,
                "val",
                max_samples=evaluation.get("max_empty_samples"),
            ),
            **common,
        ),
    }


def build_diffusion(config: dict[str, Any], device: torch.device) -> GaussianDiffusion | None:
    if config["model"]["method"] == "unet":
        return None
    diffusion_config = config["diffusion"]
    return GaussianDiffusion(
        timesteps=int(diffusion_config["timesteps"]),
        schedule=diffusion_config.get("schedule", "cosine"),
        clip_x0=diffusion_config.get("clip_x0", 2.0),
    ).to(device)


def _forward_supervised(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion | None,
    batch: dict[str, Any],
    method: str,
    loss_config: dict[str, float],
) -> dict[str, torch.Tensor]:
    image = batch["image"]
    target = batch["target"]
    if method == "unet":
        prediction = model(image)["pred"]
        return unet_training_losses(
            prediction=prediction,
            target=target,
            mask=batch["mask"],
            is_empty=batch["is_empty"],
            weights=loss_config,
        )
    if diffusion is None:
        raise ValueError("Diffusion object missing")
    time_steps = torch.randint(0, diffusion.timesteps, (image.shape[0],), device=image.device)
    noisy, noise = diffusion.q_sample(target, time_steps)
    output = model(noisy, image, time_steps)
    v_target = diffusion.v_target(target, noise, time_steps)
    x0_prediction = diffusion.predict_x0(noisy, output["v"], time_steps)
    return diffusion_training_losses(
        v_prediction=output["v"],
        v_target=v_target,
        x0_prediction=x0_prediction,
        target=target,
        mask=batch["mask"],
        prior_logits=output.get("prior_logits"),
        is_empty=batch["is_empty"],
        weights=loss_config,
    )


def _forward_m_weak(
    model: torch.nn.Module, batch: dict[str, Any], loss_config: dict[str, float]
) -> dict[str, torch.Tensor]:
    if not isinstance(model, DiffusionUNet) or model.conditioner is None:
        raise ValueError("M weak supervision requires the DG-RTDiff directional conditioner")
    cache = model.encode_condition(batch["image"])
    if cache is None:
        raise ValueError("Condition cache is missing")
    prior_loss = weak_mil_prior_loss(
        cache["prior_logits"],
        batch["box_mask"],
        top_fraction=float(loss_config.get("mil_top_fraction", 0.1)),
    )
    total = float(loss_config.get("lambda_prior", 0.1)) * prior_loss
    zero = total.detach().new_zeros(())
    return {
        "total": total,
        "diffusion": zero,
        "reconstruction": zero,
        "direction": zero,
        "prior": prior_loss.detach(),
        "empty": zero,
    }


def _autocast_context(device: torch.device, config: dict[str, Any]):
    enabled = bool(config["training"].get("amp", False)) and device.type == "cuda"
    dtype_name = config["training"].get("amp_dtype", "float16")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled), enabled, dtype


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
        },
        path,
    )


def load_weights(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    ema: EMA | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    resume_optimizer: bool = False,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
        ema.to(next(model.parameters()).device)
    if resume_optimizer:
        if optimizer is None or scheduler is None:
            raise ValueError("Optimizer and scheduler are required for full resume")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def _evaluate_for_training(
    config: dict[str, Any],
    model: torch.nn.Module,
    ema: EMA,
    diffusion: GaussianDiffusion | None,
    loaders: dict[str, DataLoader],
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    method = config["model"]["method"]
    diffusion_cfg = config.get("diffusion", {})
    steps = int(diffusion_cfg.get("sampling_steps", 1))
    eta = float(diffusion_cfg.get("eta", 0.0))
    ema.apply(model)
    try:
        composite = evaluate_composite(
            model=model,
            diffusion=diffusion,
            loader=loaders["composite"],
            device=device,
            method=method,
            sampling_steps=steps,
            eta=eta,
            seed=seed,
        )
        r2 = evaluate_empty(
            model=model,
            diffusion=diffusion,
            loader=loaders["r2"],
            device=device,
            method=method,
            sampling_steps=steps,
            eta=eta,
            seed=seed + 100_000,
            threshold=float(config["evaluation"].get("pfr_threshold", 0.1)),
            domain="r2",
        )
        m_metrics = evaluate_m(
            model=model,
            diffusion=diffusion,
            loader=loaders["m"],
            device=device,
            method=method,
            sampling_steps=steps,
            eta=eta,
            seed=seed + 200_000,
        )
    finally:
        ema.restore(model)
    result = {"composite": composite, "r2": r2, "m": m_metrics}
    write_json(result, output_dir / "metrics" / "val.json")
    return result


def _empty_limit(config: dict[str, Any]) -> float | None:
    path_value = config["evaluation"].get("baseline_metrics_json")
    if not path_value:
        return None
    path = resolve_path(path_value)
    if not path.exists():
        print(f"[warning] baseline metrics not found; selection uses nMAE only: {path}")
        return None
    with path.open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    value = float(baseline["r2"]["empty_top1"])
    return value * float(config["evaluation"].get("empty_tolerance", 1.05))


def check_dataset_qc(config: dict[str, Any], allow_pending_qc: bool, dry_run: bool) -> None:
    status_path = resolve_path(config["data"]["fixed_root"]) / "dataset_status.json"
    with status_path.open("r", encoding="utf-8") as handle:
        status = json.load(handle).get("status", "UNKNOWN")
    if status == "PENDING_USER_QC" and not (allow_pending_qc or dry_run):
        raise RuntimeError(
            "Dataset is still PENDING_USER_QC. Finish author QC or pass --allow-pending-qc explicitly."
        )


def run_training(
    config: dict[str, Any],
    *,
    seed: int | None = None,
    resume: str | Path | None = None,
    resume_optimizer: bool = False,
    dry_run: bool = False,
    allow_pending_qc: bool = False,
) -> dict[str, Any]:
    actual_seed = int(config.get("seed", 20260716) if seed is None else seed)
    seed_everything(actual_seed)
    check_dataset_qc(config, allow_pending_qc, dry_run)
    device = choose_device(config.get("device", "auto"), bool(config.get("require_cuda", False)))
    model = build_model(config["model"]).to(device)
    diffusion = build_diffusion(config, device)
    loaders = build_training_loaders(config, actual_seed, dry_run=dry_run)
    validation_loaders = None if dry_run else build_validation_loaders(config, actual_seed)

    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
        betas=tuple(float(value) for value in optimizer_config.get("betas", [0.9, 0.999])),
    )
    if config["mode"] == "stage_b":
        stage_b_pattern = list(config["training"].get("stage_b_pattern", ["composite", "composite", "m", "r2"]))
        if dry_run:
            stage_b_pattern = list(dict.fromkeys(stage_b_pattern))
            steps_per_epoch = len(stage_b_pattern)
        else:
            steps_per_epoch = int(config["training"]["steps_per_epoch"])
    else:
        stage_b_pattern = ["composite"]
        requested_steps = config["training"].get("max_steps_per_epoch")
        steps_per_epoch = min(len(loaders["composite"]), int(requested_steps)) if requested_steps else len(loaders["composite"])
        if dry_run:
            steps_per_epoch = 1
    epochs = int(config["training"]["epochs"])
    accumulation = int(config["training"].get("grad_accum_steps", 1))
    optimizer_steps = max(1, math.ceil(steps_per_epoch * epochs / accumulation))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimizer_steps)
    ema = EMA(model, float(config["training"].get("ema_decay", 0.9999)))

    start_epoch = 0
    global_step = 0
    if resume is not None:
        checkpoint = load_weights(
            resume,
            model,
            ema,
            optimizer,
            scheduler,
            resume_optimizer=resume_optimizer,
        )
        if resume_optimizer:
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            global_step = int(checkpoint.get("global_step", 0))

    run_dir = resolve_path(config.get("output_root", "runs")) / f"{config['experiment']}_s{actual_seed}"
    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, run_dir / "config_frozen.yaml")
        write_json(environment_summary(device), run_dir / "environment.json")
        source_splits = resolve_path(config["data"]["source_splits"])
        manifest = resolve_path(config["data"]["fixed_root"]) / "manifest.csv"
        write_json(
            {"source_splits_sha256": sha256_file(source_splits), "manifest_sha256": sha256_file(manifest)},
            run_dir / "source_split_hashes.json",
        )
    print(
        f"method={config['model']['method']} device={device} parameters={count_parameters(model):,} "
        f"dry_run={dry_run}"
    )

    autocast, amp_enabled, amp_dtype = _autocast_context(device, config)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled and amp_dtype == torch.float16,
    )
    log = CSVLogger(run_dir / "logs" / "train.csv")
    empty_limit = _empty_limit(config) if not dry_run else None
    best_nmae = float("inf")
    evaluations_without_improvement = 0
    optimizer.zero_grad(set_to_none=True)

    iterators: dict[str, Iterable[Any]] = {name: cycle_loader(loader) for name, loader in loaders.items()}
    final_summary: dict[str, Any] = {}
    training_started = time.time()
    max_hours = float(config["training"].get("max_hours", 48.0))
    timed_out = False
    for epoch in range(start_epoch, epochs):
        if isinstance(model, DiffusionUNet) and config["mode"] == "stage_b":
            model.set_denoiser_trainable(epoch >= int(config["training"].get("freeze_denoiser_epochs", 0)))
        model.train()
        running: dict[str, float] = {}
        started = time.time()
        progress = tqdm(range(steps_per_epoch), desc=f"epoch {epoch + 1}/{epochs}")
        for step in progress:
            source = "composite" if config["mode"] == "stage_a" else stage_b_pattern[step % len(stage_b_pattern)]
            batch = to_device(next(iterators[source]), device)
            with autocast:
                losses = (
                    _forward_m_weak(model, batch, config["loss"])
                    if source == "m"
                    else _forward_supervised(
                        model, diffusion, batch, config["model"]["method"], config["loss"]
                    )
                )
                scaled_loss = losses["total"] / accumulation
            scaler.scale(scaled_loss).backward()
            if (step + 1) % accumulation == 0 or step + 1 == steps_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["training"].get("grad_clip_norm", 1.0))
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
                global_step += 1
            values = detach_loss_dict(losses)
            for key, value in values.items():
                running[key] = running.get(key, 0.0) + value
            progress.set_postfix(loss=f"{values['total']:.4f}", source=source)
            if not dry_run and time.time() - training_started >= max_hours * 3600.0:
                timed_out = True
                print(f"Hard timeout reached after {max_hours:.2f} hours; saving checkpoint.")
                break
        epoch_row = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
            **{key: value / (step + 1) for key, value in running.items()},
        }
        final_summary = epoch_row
        if dry_run:
            print("dry-run losses:", epoch_row)
            break
        log.append(epoch_row)
        write_json(
            {
                "pid": environment_summary(device)["python_pid"],
                "epoch": epoch,
                "global_step": global_step,
                "updated_unix": time.time(),
                "timed_out": timed_out,
            },
            run_dir / "heartbeat.json",
        )
        _save_checkpoint(
            run_dir / "checkpoints" / "latest.pt",
            model,
            ema,
            optimizer,
            scheduler,
            epoch,
            global_step,
            config,
        )
        if (epoch + 1) % int(config["training"].get("save_every", 5)) == 0:
            _save_checkpoint(
                run_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pt",
                model,
                ema,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
            )
        if timed_out:
            break
        if (epoch + 1) % int(config["training"].get("val_every", 5)) == 0:
            if validation_loaders is None:
                raise RuntimeError("Validation loaders were not built")
            metrics = _evaluate_for_training(
                config, model, ema, diffusion, validation_loaders, device, actual_seed, run_dir
            )
            nmae = float(metrics["composite"].get("nmae_roi", float("inf")))
            empty_top1 = float(metrics["r2"].get("empty_top1", float("inf")))
            allowed = empty_limit is None or empty_top1 <= empty_limit
            if allowed and nmae < best_nmae:
                best_nmae = nmae
                evaluations_without_improvement = 0
                _save_checkpoint(
                    run_dir / "checkpoints" / "best_ema.pt",
                    model,
                    ema,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    config,
                )
            else:
                evaluations_without_improvement += 1
            if evaluations_without_improvement >= int(
                config["training"].get("patience_evaluations", 3)
            ):
                print("Early stopping: validation did not improve under the empty-background constraint.")
                break
    return {
        "status": "DRY_RUN_OK" if dry_run else ("TIMEOUT" if timed_out else "COMPLETED"),
        "run_dir": str(run_dir),
        "device": str(device),
        "parameters": count_parameters(model),
        "last": final_summary,
    }