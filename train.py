from __future__ import annotations

import argparse

from dg_rtdiff.config import load_config
from dg_rtdiff.engine import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DG-RTDiff or a registered baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="One short forward/backward check; no formal training.")
    parser.add_argument("--allow-pending-qc", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training(
        load_config(args.config),
        seed=args.seed,
        resume=args.resume,
        resume_optimizer=args.resume_optimizer,
        dry_run=args.dry_run,
        allow_pending_qc=args.allow_pending_qc,
    )
    print(result)


if __name__ == "__main__":
    main()
