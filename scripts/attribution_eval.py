"""Cheap prediction-level attribution CLI.

Score held-out one-step prediction quality (MSE + NLL), the residual contribution fraction, and
the parameter count for one or more trained variant runs -- BEFORE any planner eval -- and write
a comparison table (markdown + JSON). All variants are scored on a FIXED held-out transition set
(deterministic seed, built from the first run's env), so the numbers are directly comparable.

Usage (CPU, tiny):
    uv run python scripts/attribution_eval.py \\
        --run runs/EnvironmentI/cdl/<full_run> \\
        --run runs/EnvironmentI/cdl/<structure_only_run> \\
        --run runs/EnvironmentI/cdl/<oracle_run> \\
        --run runs/EnvironmentI/mlp/<dense_run> \\
        --device cpu --n-transitions 512 --seed 0 \\
        --out-md .temp/new_arch/reports/attribution_table.md \\
        --out-json .temp/new_arch/reports/attribution_table.json

Variant names are inferred from each run's config (full | structure_only | oracle | dense);
pass --name once per --run (same order) to override.
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdd_oran.analysis.attribution import run_attribution  # noqa: E402
from cdd_oran.utils.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Trained run dir (repeatable)")
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        help="Optional variant label per --run (same order); inferred from config otherwise",
    )
    parser.add_argument("--config", default=None, help="Override config path for every run")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument("--n-transitions", type=int, default=512, help="Held-out transition count")
    parser.add_argument("--seed", type=int, default=0, help="Held-out set seed (same for all variants)")
    parser.add_argument(
        "--attribution-seed",
        type=int,
        default=0,
        help="Structure-draw seed, reused for every CDL model (full & structure_only share it)",
    )
    parser.add_argument(
        "--held-out-ranges",
        choices=("train", "ood"),
        default=None,
        help="Held-out param ranges (default: each run's evaluation_param_ranges)",
    )
    parser.add_argument(
        "--require-full-set",
        action="store_true",
        help="Require all four named variants (full, structure_only, oracle, dense) to be present",
    )
    parser.add_argument("--out-md", default=None, help="Write the markdown comparison table here")
    parser.add_argument("--out-json", default=None, help="Write the full JSON comparison here")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    rows, markdown = run_attribution(
        args.run,
        names=args.name,
        device=args.device,
        n_transitions=args.n_transitions,
        seed=args.seed,
        attribution_seed=args.attribution_seed,
        held_out_ranges=args.held_out_ranges,
        require_full_set=args.require_full_set,
        out_md=args.out_md,
        out_json=args.out_json,
        config_path=args.config,
    )
    logger.info("Scored %d variant(s) on %d held-out transitions", len(rows), args.n_transitions)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
