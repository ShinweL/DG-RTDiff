from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dg_rtdiff.data import FixedCompositeDataset
from dg_rtdiff.diffusion import GaussianDiffusion
from dg_rtdiff.engine import EMA, load_weights
from dg_rtdiff.evaluation import evaluate_composite
from dg_rtdiff.models import DiffusionUNet


PROJECT = Path(__file__).resolve().parents[1]
FIXED = PROJECT.parent / "data" / "fixed_composite_dataset_v2_gain18_lower"


def small_model() -> DiffusionUNet:
    return DiffusionUNet(
        channels=(8, 16, 32, 64), blocks_per_level=1, time_dim=32,
        use_direction=True, horizontal_kernel=15,
    )


def test_evaluator_writes_summary(tmp_path: Path) -> None:
    model = small_model()
    diffusion = GaussianDiffusion(timesteps=8)
    loader = DataLoader(
        FixedCompositeDataset(FIXED, "train", debug_crop_size=32, max_samples=2),
        batch_size=1,
    )
    result = evaluate_composite(
        model=model, diffusion=diffusion, loader=loader, device=torch.device("cpu"),
        method="dg_rtdiff", sampling_steps=1, eta=0.0, seed=7,
        output_dir=tmp_path, save_predictions=False,
    )
    assert result["count"] == 2.0
    assert (tmp_path / "per_image.csv").exists()
    assert (tmp_path / "summary.json").exists()


def test_checkpoint_and_ema_round_trip(tmp_path: Path) -> None:
    model = small_model()
    ema = EMA(model, 0.9)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, checkpoint_path)
    restored = small_model()
    restored_ema = EMA(restored, 0.5)
    load_weights(checkpoint_path, restored, restored_ema)
    for original, loaded in zip(model.parameters(), restored.parameters()):
        assert torch.equal(original, loaded)
    restored_ema.apply(restored)
    restored_ema.restore(restored)
