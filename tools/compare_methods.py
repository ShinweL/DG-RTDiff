from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from dg_rtdiff.utils import write_json


def read_metric(path: Path, metric: str) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        if row.get(metric, "") != "":
            result[row["sample_id"]] = float(row[metric])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired bootstrap and Wilcoxon comparison.")
    parser.add_argument("--method-a", required=True, type=Path)
    parser.add_argument("--method-b", required=True, type=Path)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def signed_rank_permutation_test(
    differences: np.ndarray, permutations: int, rng: np.random.Generator
) -> tuple[float, float, int, str]:
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0, 1.0, 0, "all differences zero"
    ranks = average_ranks(np.abs(nonzero))
    observed_positive = float(ranks[nonzero > 0].sum())
    rank_sum = float(ranks.sum())
    observed_distance = abs(observed_positive - rank_sum / 2.0)
    if nonzero.size <= 20:
        total = 1 << nonzero.size
        extreme = 0
        for pattern in range(total):
            positive = sum(ranks[index] for index in range(nonzero.size) if pattern & (1 << index))
            extreme += abs(positive - rank_sum / 2.0) >= observed_distance - 1e-12
        p_value = extreme / total
        used = total
        method = "exact sign enumeration"
    else:
        used = permutations
        extreme = 0
        remaining = permutations
        while remaining:
            chunk = min(10_000, remaining)
            signs = rng.integers(0, 2, size=(chunk, nonzero.size), dtype=np.int8)
            positive = signs @ ranks
            extreme += int(np.count_nonzero(np.abs(positive - rank_sum / 2.0) >= observed_distance - 1e-12))
            remaining -= chunk
        p_value = (extreme + 1) / (permutations + 1)
        method = "Monte Carlo sign permutation"
    statistic = min(observed_positive, rank_sum - observed_positive)
    return float(statistic), float(p_value), used, method


def main() -> None:
    args = parse_args()
    method_a = read_metric(args.method_a, args.metric)
    method_b = read_metric(args.method_b, args.metric)
    common = sorted(set(method_a) & set(method_b))
    if len(common) < 2:
        raise ValueError("At least two paired samples are required")
    a = np.asarray([method_a[key] for key in common], dtype=np.float64)
    b = np.asarray([method_b[key] for key in common], dtype=np.float64)
    delta = b - a
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(common), size=(args.bootstrap, len(common)))
    bootstrap_delta = delta[indices].mean(axis=1)
    confidence = np.percentile(bootstrap_delta, [2.5, 97.5])
    statistic, p_value, used_permutations, p_method = signed_rank_permutation_test(
        delta, args.permutations, rng
    )
    relative = float(delta.mean() / (abs(a.mean()) + 1e-12))
    b_better = delta.mean() < 0 if args.lower_is_better else delta.mean() > 0
    result = {
        "metric": args.metric,
        "paired_count": len(common),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_delta_b_minus_a": float(delta.mean()),
        "relative_delta": relative,
        "bootstrap_95_ci": [float(confidence[0]), float(confidence[1])],
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p_two_sided": float(p_value),
        "wilcoxon_p_method": p_method,
        "wilcoxon_permutations": used_permutations,
        "lower_is_better": args.lower_is_better,
        "method_b_better_by_mean": bool(b_better),
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap,
    }
    write_json(result, args.output)
    print(result)


if __name__ == "__main__":
    main()