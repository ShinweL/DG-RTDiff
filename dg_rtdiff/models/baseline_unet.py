from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import PlainResBlock, run_plain_block, valid_groups


class ResidualUNet(nn.Module):
    """Deterministic target-residual U-Net baseline."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        blocks_per_level: int = 2,
        dropout: float = 0.0,
        groups: int = 8,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        self.input_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, channel in enumerate(channels):
            self.down_blocks.append(
                nn.ModuleList(
                    [PlainResBlock(channel, channel, groups, dropout) for _ in range(blocks_per_level)]
                )
            )
            if index < len(channels) - 1:
                self.downsamples.append(
                    nn.Conv2d(channel, channels[index + 1], kernel_size=3, stride=2, padding=1)
                )
        self.mid = nn.ModuleList(
            [PlainResBlock(channels[-1], channels[-1], groups, dropout) for _ in range(2)]
        )
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for index in reversed(range(len(channels))):
            channel = channels[index]
            blocks = [PlainResBlock(channel * 2, channel, groups, dropout)]
            blocks.extend(
                PlainResBlock(channel, channel, groups, dropout)
                for _ in range(blocks_per_level - 1)
            )
            self.up_blocks.append(nn.ModuleList(blocks))
            self.upsamples.append(
                nn.Conv2d(channel, channels[index - 1], kernel_size=3, padding=1)
                if index > 0
                else nn.Identity()
            )
        self.output = nn.Sequential(
            nn.GroupNorm(valid_groups(channels[0], groups), channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | None]:
        factor = 2 ** (len(self.channels) - 1)
        if image.shape[-2] % factor or image.shape[-1] % factor:
            raise ValueError(f"Input spatial dimensions must be divisible by {factor}")
        hidden = self.input_conv(image)
        skips: list[torch.Tensor] = []
        for index, blocks in enumerate(self.down_blocks):
            for block in blocks:
                hidden = run_plain_block(block, hidden, self.use_checkpoint)
            skips.append(hidden)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)
        for block in self.mid:
            hidden = run_plain_block(block, hidden, self.use_checkpoint)
        for decoder_index, level_index in enumerate(reversed(range(len(self.channels)))):
            skip = skips[level_index]
            if hidden.shape[-2:] != skip.shape[-2:]:
                hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = torch.cat((hidden, skip), dim=1)
            for block in self.up_blocks[decoder_index]:
                hidden = run_plain_block(block, hidden, self.use_checkpoint)
            if level_index > 0:
                hidden = F.interpolate(hidden, scale_factor=2.0, mode="nearest")
                hidden = self.upsamples[decoder_index](hidden)
        return {"pred": self.output(hidden), "prior_logits": None}
