from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduce.src.variance_experiment import default_variance_config, run_variance_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cleaned variance/entanglement experiment.")
    parser.add_argument("--case", choices=["typical", "atypical"], required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--delta-t", type=float, default=0.01)
    parser.add_argument("--trotter-order", type=int, choices=[1, 2], default=1)
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_variance_config(
        case=args.case,
        seed=args.seed,
        samples=args.samples,
        depth=args.depth,
    )
    config = replace(
        config,
        n=args.n,
        delta_t=args.delta_t,
        trotter_order=args.trotter_order,
        trotter_steps=args.trotter_steps,
        start_index=args.start_index,
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "reproduce" / "outputs" / (
            f"variance_{args.case}_n{config.n}_depth{config.depth}_"
            f"samples{config.samples}_seed{config.seed}"
        )

    result_dir = run_variance_experiment(config, output_dir)
    print(f"Saved results to {result_dir}")


if __name__ == "__main__":
    main()
