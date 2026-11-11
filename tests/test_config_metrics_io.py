from pathlib import Path

import torch

from dg_rtdiff.config import load_config
from dg_rtdiff.data.common import read_target_residual_png
from dg_rtdiff.metrics import aggregate_records
from dg_rtdiff.utils import save_signed16


PROJECT = Path(__file__).resolve().parents[1]


def test_ablation_config_inheritance() -> None:
    config = load_config(PROJECT / "configs" / "ablations" / "a2_direction_only.yaml")
    assert config["model"]["channels"] == [32, 64, 128, 256]
    assert config["model"]["use_prior_gate"] is False
    assert config["loss"]["lambda_prior"] == 0.0


def test_aggregate_ignores_string_metadata() -> None:
    summary = aggregate_records([
        {"sample_id": "a", "domain": "x", "metric": 1.0},
        {"sample_id": "b", "domain": "x", "metric": 3.0},
    ])
    assert summary["metric"] == 2.0
    assert "sample_id" not in summary


def test_signed16_round_trip(tmp_path: Path) -> None:
    original = torch.linspace(-1.5, 1.5, 64).reshape(1, 8, 8)
    path = tmp_path / "signed.png"
    save_signed16(original, path)
    decoded = read_target_residual_png(path)
    assert torch.allclose(original, decoded, atol=1 / 16384)
