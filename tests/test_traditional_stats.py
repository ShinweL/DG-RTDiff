import numpy as np
import torch

from tools.compare_methods import average_ranks, signed_rank_permutation_test
from tools.run_traditional_baseline import predict, robust_pca


def test_traditional_predictions_have_expected_shape() -> None:
    image = torch.randn(1, 1, 8, 8)
    for method in ("mean_scan", "svd", "rpca"):
        result = predict(image, method, rank=2, rpca_lambda=None)
        assert result.shape == image.shape
        assert torch.isfinite(result).all()


def test_robust_pca_reconstruction() -> None:
    matrix = np.ones((6, 6), dtype=np.float64)
    matrix[2, 3] += 5
    low_rank, sparse = robust_pca(matrix, max_iterations=200)
    assert np.linalg.norm(matrix - low_rank - sparse) / np.linalg.norm(matrix) < 1e-5


def test_signed_rank_helpers() -> None:
    ranks = average_ranks(np.asarray([1.0, 2.0, 2.0, 4.0]))
    assert np.allclose(ranks, [1.0, 2.5, 2.5, 4.0])
    statistic, p_value, used, method = signed_rank_permutation_test(
        np.asarray([-1.0, -2.0, -3.0, -4.0]), 1000, np.random.default_rng(1)
    )
    assert statistic == 0.0
    assert 0.0 <= p_value <= 1.0
    assert used == 16
    assert method == "exact sign enumeration"
