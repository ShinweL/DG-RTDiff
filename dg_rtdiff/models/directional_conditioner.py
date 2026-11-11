from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import valid_groups


class DirectionalBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(valid_groups(channels, groups), channels)
        self.horizontal = nn.Conv2d(
            channels, channels, kernel_size=(1, 9), padding=(0, 4), groups=channels
        )
        self.vertical = nn.Conv2d(
            channels, channels, kernel_size=(9, 1), padding=(4, 0), groups=channels
        )
        self.local = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.horizontal_mix = nn.Conv2d(channels, channels, kernel_size=1)
        self.vertical_mix = nn.Conv2d(channels, channels, kernel_size=1)
        self.local_mix = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Conv2d(channels, 3, kernel_size=1)
        self.output = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = F.silu(self.norm(x))
        horizontal = self.horizontal_mix(self.horizontal(normalized))
        vertical = self.vertical_mix(self.vertical(normalized))
        local = self.local_mix(self.local(normalized))
        gates = self.gate(normalized).softmax(dim=1)
        horizontal_feature = gates[:, 0:1] * horizontal + 0.25 * gates[:, 2:3] * local
        detail_feature = gates[:, 1:2] * vertical + gates[:, 2:3] * local
        fused = x + self.output(F.silu(horizontal_feature + detail_feature))
        return horizontal_feature, detail_feature, fused


class DirectionalConditioner(nn.Module):
    """Multiscale horizontal/detail decomposition and weak target prior."""

    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, ...],
        groups: int = 8,
        horizontal_kernel: int = 31,
        use_prior_gate: bool = True,
    ) -> None:
        super().__init__()
        if horizontal_kernel % 2 == 0:
            raise ValueError("horizontal_kernel must be odd")
        self.horizontal_kernel = horizontal_kernel
        self.use_prior_gate = use_prior_gate
        self.stem = nn.Conv2d(in_channels * 2, channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([DirectionalBlock(ch, groups=groups) for ch in channels])
        self.downsamples = nn.ModuleList(
            [
                nn.Conv2d(channels[index], channels[index + 1], kernel_size=3, stride=2, padding=1)
                for index in range(len(channels) - 1)
            ]
        )
        self.prior_projections = nn.ModuleList(
            [nn.Conv2d(ch, channels[0], kernel_size=1) for ch in channels]
        )
        self.prior_head = nn.Sequential(
            nn.GroupNorm(valid_groups(channels[0], groups), channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels[0], 1, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> dict[str, object]:
        horizontal_prior = F.avg_pool2d(
            image,
            kernel_size=(1, self.horizontal_kernel),
            stride=1,
            padding=(0, self.horizontal_kernel // 2),
        )
        detail_prior = image - horizontal_prior
        hidden = self.stem(torch.cat((horizontal_prior, detail_prior), dim=1))
        horizontal_features: list[torch.Tensor] = []
        detail_features: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            horizontal, detail, hidden = block(hidden)
            horizontal_features.append(horizontal)
            detail_features.append(detail)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)

        target_size = detail_features[0].shape[-2:]
        prior_feature = torch.zeros_like(self.prior_projections[0](detail_features[0]))
        for projection, feature in zip(self.prior_projections, detail_features):
            projected = projection(feature)
            if projected.shape[-2:] != target_size:
                projected = F.interpolate(projected, size=target_size, mode="bilinear", align_corners=False)
            prior_feature = prior_feature + projected
        prior_logits = self.prior_head(prior_feature)

        conditions: list[torch.Tensor] = []
        probability = prior_logits.sigmoid()
        for horizontal, detail in zip(horizontal_features, detail_features):
            if self.use_prior_gate:
                gate = F.interpolate(probability, size=detail.shape[-2:], mode="bilinear", align_corners=False)
                detail = (0.25 + 0.75 * gate) * detail
            # Keep the two physical-image priors separable for FiLM instead of
            # collapsing them into one tensor before the denoiser sees them.
            conditions.append(torch.cat((horizontal, detail), dim=1))
        return {
            "conditions": conditions,
            "prior_logits": prior_logits,
            "horizontal_prior": horizontal_prior,
            "detail_prior": detail_prior,
        }