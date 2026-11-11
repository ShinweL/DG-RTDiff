from .baseline_unet import ResidualUNet
from .diffusion_unet import DiffusionUNet


def build_model(config: dict):
    method = config["method"]
    if method == "unet":
        return ResidualUNet(
            in_channels=int(config.get("in_channels", 1)),
            channels=tuple(config["channels"]),
            blocks_per_level=int(config.get("blocks_per_level", 2)),
            dropout=float(config.get("dropout", 0.0)),
            groups=int(config.get("group_norm_groups", 8)),
            use_checkpoint=bool(config.get("use_checkpoint", False)),
        )
    if method not in {"dg_rtdiff", "vanilla_ddpm"}:
        raise ValueError(f"Unknown model method: {method}")
    return DiffusionUNet(
        in_channels=int(config.get("in_channels", 1)),
        channels=tuple(config["channels"]),
        blocks_per_level=int(config.get("blocks_per_level", 2)),
        dropout=float(config.get("dropout", 0.0)),
        time_dim=int(config.get("time_dim", 128)),
        groups=int(config.get("group_norm_groups", 8)),
        use_checkpoint=bool(config.get("use_checkpoint", False)),
        bottleneck_attention=bool(config.get("bottleneck_attention", False)),
        use_direction=method == "dg_rtdiff",
        use_prior_gate=bool(config.get("use_prior_gate", True)),
        horizontal_kernel=int(config.get("horizontal_kernel", 31)),
    )


__all__ = ["ResidualUNet", "DiffusionUNet", "build_model"]
