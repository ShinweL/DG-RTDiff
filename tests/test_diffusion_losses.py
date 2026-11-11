import torch

from dg_rtdiff.diffusion import GaussianDiffusion
from dg_rtdiff.losses import diffusion_training_losses, weak_mil_prior_loss


def test_v_prediction_round_trip() -> None:
    diffusion = GaussianDiffusion(timesteps=16, clip_x0=None)
    clean = torch.randn(2, 1, 16, 16)
    time = torch.tensor([0, 12])
    noisy, noise = diffusion.q_sample(clean, time)
    target_v = diffusion.v_target(clean, noise, time)
    reconstructed = diffusion.predict_x0(noisy, target_v, time)
    assert torch.allclose(clean, reconstructed, atol=1e-5)


def test_losses_are_finite() -> None:
    shape = (2, 1, 16, 16)
    target = torch.randn(shape) * 0.1
    mask = torch.zeros(shape)
    mask[:, :, 4:12, 4:12] = 1
    losses = diffusion_training_losses(
        v_prediction=torch.randn(shape), v_target=torch.randn(shape),
        x0_prediction=torch.randn(shape), target=target, mask=mask,
        prior_logits=torch.randn(shape), is_empty=torch.tensor([0.0, 1.0]),
        weights={"target_weight_kappa": 4, "lambda_rec": 1, "lambda_dir": 0.2, "lambda_prior": 0.1, "lambda_empty": 0.2},
    )
    assert all(torch.isfinite(value) for value in losses.values())


def test_mil_has_gradients() -> None:
    logits = torch.zeros(1, 1, 16, 16, requires_grad=True)
    mask = torch.zeros_like(logits)
    mask[..., 4:12, 4:12] = 1
    loss = weak_mil_prior_loss(logits, mask)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
