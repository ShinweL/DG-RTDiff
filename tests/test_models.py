import torch

from dg_rtdiff.diffusion import GaussianDiffusion
from dg_rtdiff.models import DiffusionUNet, ResidualUNet


def test_directional_diffusion_forward_backward_and_ddim() -> None:
    model = DiffusionUNet(
        channels=(8, 16, 32, 64), blocks_per_level=1, time_dim=32, use_direction=True,
        use_checkpoint=False, horizontal_kernel=15,
    )
    image = torch.randn(1, 1, 32, 32)
    noisy = torch.randn_like(image)
    time = torch.tensor([3])
    output = model(noisy, image, time)
    assert output["v"].shape == image.shape
    assert output["prior_logits"].shape == image.shape
    output["v"].mean().backward()
    diffusion = GaussianDiffusion(timesteps=8)
    sampled, prior = diffusion.ddim_sample(model, image, steps=2, eta=0.0)
    assert sampled.shape == image.shape
    assert prior is not None and prior.shape == image.shape


def test_vanilla_and_unet_shapes() -> None:
    image = torch.randn(1, 1, 32, 32)
    vanilla = DiffusionUNet(
        channels=(8, 16, 32, 64), blocks_per_level=1, time_dim=32, use_direction=False
    )
    assert vanilla(torch.randn_like(image), image, torch.tensor([1]))["v"].shape == image.shape
    unet = ResidualUNet(channels=(8, 16, 32, 64), blocks_per_level=1)
    assert unet(image)["pred"].shape == image.shape
