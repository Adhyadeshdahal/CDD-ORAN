"""E6-P v2 stage 2, Kaggle half of a Kaggle+Colab split: kernel shard s/12 -> global part s/24 (units 0-11)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e6p_v2  # noqa: E402

TOTAL, KAGGLE_SHARDS = 24, 12

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    argv = sys.argv[1:]
    i, k = map(int, argv[argv.index("--part") + 1].split("/"))
    assert k == KAGGLE_SHARDS, f"launch with 3 Kaggle kernels (12 shards), got {k}"
    argv[argv.index("--part") + 1] = f"{i}/{TOTAL}"
    e6p_v2.cli(argv, stage="2")
