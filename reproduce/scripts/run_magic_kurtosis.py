from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduce.src.magic_experiment import MagicKurtosisConfig, run_magic_kurtosis_experiment
from reproduce.src.resources import GLOBAL_CLIFFORD_SAMPLER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cleaned magic/kurtosis experiment.")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--state-seed", type=int, default=4321)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2468)
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--trotter-order", type=int, choices=[1, 2], default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--trotter-steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--magic-batch-size", type=int, default=4096)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = args.trotter_steps if args.trotter_steps is not None else args.steps
    config = MagicKurtosisConfig(
        n=args.n,
        samples=args.samples,
        seed=args.seed,
        state_seed=args.state_seed,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        delta_t=args.delta_t,
        trotter_order=args.trotter_order,
        steps=steps,
        magic_batch_size=args.magic_batch_size,
        append=args.append,
    )

    output_dir = args.output_dir
    if output_dir is None:
        sampler_label = GLOBAL_CLIFFORD_SAMPLER.replace("_", "-")
        output_dir = REPO_ROOT / "reproduce" / "outputs" / (
            f"magic_kurtosis_n{config.n}_state_seed{config.state_seed}_"
            f"dt{config.delta_t}_order{config.trotter_order}_steps{config.steps}_"
            f"{sampler_label}"
        )

    result_dir = run_magic_kurtosis_experiment(config, output_dir)
    print(f"Saved results to {result_dir}")


if __name__ == "__main__":
    main()
