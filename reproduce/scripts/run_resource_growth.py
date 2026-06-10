from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduce.src.resource_growth import default_resource_growth_config, run_resource_growth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute magic and entanglement growth over time.")
    parser.add_argument("--case", choices=["typical", "atypical", "both"], default="both")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--time-max", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--subsystem-size", type=int, default=4)
    parser.add_argument("--magic-batch-size", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_cases = ["typical", "atypical"] if args.case == "both" else [args.case]
    configs = [
        default_resource_growth_config(
            case=case,
            n=args.n,
            time_max=args.time_max,
            steps=args.steps,
            subsystem_size=args.subsystem_size,
            magic_batch_size=args.magic_batch_size,
        )
        for case in selected_cases
    ]

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "reproduce" / "outputs" / (
            f"resource_growth_{args.case}_n{args.n}_steps{args.steps}_tmax{args.time_max}"
        )

    result_dir = run_resource_growth(configs, output_dir)
    print(f"Saved results to {result_dir}")


if __name__ == "__main__":
    main()
