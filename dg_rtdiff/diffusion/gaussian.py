from __future__ import annotations

import math

import torch
from torch import nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    cumulative = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    cumulative = cumulative / cumulative[0]
    betas = 1.0 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(0.0001, 0.9999).float()


def extract(values: torch.Tensor, time: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    gathered = values.gather(0, time)
    return gathered.reshape(time.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """Cosine diffusion with v-prediction and deterministic DDIM support."""

    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",
        clip_x0: float | None = 2.0,
    ) -> None:
        super().__init__()
        if schedule != "cosine":
            raise ValueError(f"Only cosine schedule is implemented, got {schedule!r}")
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.clip_x0 = clip_x0
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    def q_sample(
        self, clean: torch.Tensor, time: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean) if noise is None else noise
        noisy = extract(self.sqrt_alpha_bars, time, clean.shape) * clean + extract(
            self.sqrt_one_minus_alpha_bars, time, clean.shape
        ) * noise
        return noisy, noise

    def v_target(self, clean: torch.Tensor, noise: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_alpha_bars, time, clean.shape) * noise - extract(
            self.sqrt_one_minus_alpha_bars, time, clean.shape
        ) * clean

    def predict_x0(self, noisy: torch.Tensor, v_prediction: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        clean = extract(self.sqrt_alpha_bars, time, noisy.shape) * noisy - extract(
            self.sqrt_one_minus_alpha_bars, time, noisy.shape
        ) * v_prediction
        if self.clip_x0 is not None:
            clean = clean.clamp(-self.clip_x0, self.clip_x0)
        return clean

    def predict_noise(
        self, noisy: torch.Tensor, v_prediction: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        return extract(self.sqrt_one_minus_alpha_bars, time, noisy.shape) * noisy + extract(
            self.sqrt_alpha_bars, time, noisy.shape
        ) * v_prediction

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        image: torch.Tensor,
        steps: int = 20,
        eta: float = 0.0,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not 1 <= steps <= self.timesteps:
            raise ValueError(f"steps must be in [1, {self.timesteps}], got {steps}")
        current = (
            torch.randn(image.shape, device=image.device, dtype=image.dtype, generator=generator)
            if initial_noise is None
            else initial_noise.to(device=image.device, dtype=image.dtype)
        )
        condition_cache = model.encode_condition(image) if hasattr(model, "encode_condition") else None
        prior_logits = condition_cache["prior_logits"] if condition_cache is not None else None
        times = torch.linspace(self.timesteps - 1, 0, steps, device=image.device).round().long()
        for index, scalar_time in enumerate(times):
            time = torch.full(
                (image.shape[0],), int(scalar_time.item()), device=image.device, dtype=torch.long
            )
            output = model(current, image, time, condition_cache=condition_cache)
            v_prediction = output["v"]
            clean = self.predict_x0(current, v_prediction, time)
            noise = self.predict_noise(current, v_prediction, time)
            if index == len(times) - 1:
                current = clean
                break
            previous_time = times[index + 1]
            alpha = self.alpha_bars[scalar_time]
            alpha_previous = self.alpha_bars[previous_time]
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_previous) / (1.0 - alpha))
                * (1.0 - alpha / alpha_previous)
            ).clamp(min=0.0)
            direction_scale = torch.sqrt((1.0 - alpha_previous - sigma**2).clamp(min=0.0))
            if eta > 0:
                random_noise = torch.randn(
                    current.shape,
                    device=current.device,
                    dtype=current.dtype,
                    generator=generator,
                )
            else:
                random_noise = torch.zeros_like(current)
            current = alpha_previous.sqrt() * clean + direction_scale * noise + sigma * random_noise
        return current, prior_logits
