from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def valid_groups(channels: int, requested: int) -> int:
    groups = min(channels, requested)
    while channels % groups != 0:
        groups -= 1
    return groups


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=time.device, dtype=torch.float32) * -scale
        )
        angles = time.float()[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.dim % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class DiffusionResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        condition_channels: int | None,
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(valid_groups(in_channels, groups), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(valid_groups(out_channels, groups), out_channels)
        self.time_affine = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels * 2))
        self.condition_affine = (
            nn.Conv2d(condition_channels, out_channels * 2, kernel_size=1)
            if condition_channels is not None
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(
        self, x: torch.Tensor, time_embedding: torch.Tensor, condition: torch.Tensor | None
    ) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        hidden = self.norm2(hidden)
        time_scale, time_shift = self.time_affine(time_embedding).chunk(2, dim=1)
        scale = time_scale[:, :, None, None]
        shift = time_shift[:, :, None, None]
        if self.condition_affine is not None:
            if condition is None:
                raise ValueError("This block requires a directional condition tensor")
            cond_scale, cond_shift = self.condition_affine(condition).chunk(2, dim=1)
            scale = scale + cond_scale
            shift = shift + cond_shift
        hidden = hidden * (1.0 + scale) + shift
        hidden = self.conv2(self.dropout(torch.nn.functional.silu(hidden)))
        return residual + hidden


class PlainResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(valid_groups(in_channels, groups), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(valid_groups(out_channels, groups), out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.body(x)


class BottleneckAttention(nn.Module):
    def __init__(self, channels: int, groups: int = 8, heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(valid_groups(channels, groups), channels)
        self.attention = nn.MultiheadAttention(channels, num_heads=heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        sequence = self.norm(x).flatten(2).transpose(1, 2)
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        return x + attended.transpose(1, 2).reshape(batch, channels, height, width)


def run_diffusion_block(
    block: DiffusionResBlock,
    x: torch.Tensor,
    time_embedding: torch.Tensor,
    condition: torch.Tensor | None,
    enabled: bool,
) -> torch.Tensor:
    if enabled and torch.is_grad_enabled() and x.requires_grad:
        if condition is None:
            return checkpoint(
                lambda a, b: block(a, b, None), x, time_embedding, use_reentrant=False
            )
        return checkpoint(block, x, time_embedding, condition, use_reentrant=False)
    return block(x, time_embedding, condition)


def run_plain_block(block: PlainResBlock, x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and torch.is_grad_enabled() and x.requires_grad:
        return checkpoint(block, x, use_reentrant=False)
    return block(x)
