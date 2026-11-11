from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import (
    BottleneckAttention,
    DiffusionResBlock,
    SinusoidalTimeEmbedding,
    run_diffusion_block,
    valid_groups,
)
from .directional_conditioner import DirectionalConditioner


class DiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        blocks_per_level: int = 2,
        dropout: float = 0.0,
        time_dim: int = 128,
        groups: int = 8,
        use_checkpoint: bool = False,
        bottleneck_attention: bool = False,
        use_direction: bool = True,
        use_prior_gate: bool = True,
        horizontal_kernel: int = 31,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("At least two U-Net levels are required")
        self.in_channels = in_channels
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        self.use_direction = use_direction
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.conditioner = (
            DirectionalConditioner(
                in_channels=in_channels,
                channels=channels,
                groups=groups,
                horizontal_kernel=horizontal_kernel,
                use_prior_gate=use_prior_gate,
            )
            if use_direction
            else None
        )
        self.input_conv = nn.Conv2d(in_channels * 2, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, channel in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(blocks_per_level):
                blocks.append(
                    DiffusionResBlock(
                        channel,
                        channel,
                        time_dim,
                        channel * 2 if use_direction else None,
                        groups=groups,
                        dropout=dropout,
                    )
                )
            self.down_blocks.append(blocks)
            if index < len(channels) - 1:
                self.downsamples.append(
                    nn.Conv2d(channel, channels[index + 1], kernel_size=3, stride=2, padding=1)
                )

        condition_channels = channels[-1] * 2 if use_direction else None
        self.mid_block1 = DiffusionResBlock(
            channels[-1], channels[-1], time_dim, condition_channels, groups, dropout
        )
        self.mid_attention = (
            BottleneckAttention(channels[-1], groups=groups) if bottleneck_attention else nn.Identity()
        )
        self.mid_block2 = DiffusionResBlock(
            channels[-1], channels[-1], time_dim, condition_channels, groups, dropout
        )

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for index in reversed(range(len(channels))):
            channel = channels[index]
            blocks = nn.ModuleList(
                [
                    DiffusionResBlock(
                        channel * 2,
                        channel,
                        time_dim,
                        channel * 2 if use_direction else None,
                        groups,
                        dropout,
                    )
                ]
            )
            for _ in range(blocks_per_level - 1):
                blocks.append(
                    DiffusionResBlock(
                        channel,
                        channel,
                        time_dim,
                        channel * 2 if use_direction else None,
                        groups,
                        dropout,
                    )
                )
            self.up_blocks.append(blocks)
            if index > 0:
                self.upsamples.append(
                    nn.Conv2d(channel, channels[index - 1], kernel_size=3, padding=1)
                )
            else:
                self.upsamples.append(nn.Identity())

        self.output = nn.Sequential(
            nn.GroupNorm(valid_groups(channels[0], groups), channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1),
        )

    def encode_condition(self, image: torch.Tensor) -> dict[str, object] | None:
        return self.conditioner(image) if self.conditioner is not None else None

    def set_denoiser_trainable(self, trainable: bool) -> None:
        for name, parameter in self.named_parameters():
            if not name.startswith("conditioner."):
                parameter.requires_grad = trainable

    def forward(
        self,
        noisy_target: torch.Tensor,
        image: torch.Tensor,
        time: torch.Tensor,
        condition_cache: dict[str, object] | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if noisy_target.shape != image.shape:
            raise ValueError(f"noisy_target and image must match, got {noisy_target.shape} and {image.shape}")
        factor = 2 ** (len(self.channels) - 1)
        if image.shape[-2] % factor or image.shape[-1] % factor:
            raise ValueError(f"Input spatial dimensions must be divisible by {factor}")
        if self.use_direction and condition_cache is None:
            condition_cache = self.encode_condition(image)
        conditions = condition_cache["conditions"] if condition_cache is not None else [None] * len(self.channels)
        prior_logits = condition_cache["prior_logits"] if condition_cache is not None else None
        time_embedding = self.time_embedding(time)
        hidden = self.input_conv(torch.cat((noisy_target, image), dim=1))
        skips: list[torch.Tensor] = []
        for index, blocks in enumerate(self.down_blocks):
            condition = conditions[index]
            for block in blocks:
                hidden = run_diffusion_block(
                    block, hidden, time_embedding, condition, self.use_checkpoint
                )
            skips.append(hidden)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)

        hidden = run_diffusion_block(
            self.mid_block1, hidden, time_embedding, conditions[-1], self.use_checkpoint
        )
        hidden = self.mid_attention(hidden)
        hidden = run_diffusion_block(
            self.mid_block2, hidden, time_embedding, conditions[-1], self.use_checkpoint
        )

        for decoder_index, level_index in enumerate(reversed(range(len(self.channels)))):
            skip = skips[level_index]
            if hidden.shape[-2:] != skip.shape[-2:]:
                hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = torch.cat((hidden, skip), dim=1)
            condition = conditions[level_index]
            for block in self.up_blocks[decoder_index]:
                hidden = run_diffusion_block(
                    block, hidden, time_embedding, condition, self.use_checkpoint
                )
            if level_index > 0:
                hidden = F.interpolate(hidden, scale_factor=2.0, mode="nearest")
                hidden = self.upsamples[decoder_index](hidden)
        return {"v": self.output(hidden), "prior_logits": prior_logits}