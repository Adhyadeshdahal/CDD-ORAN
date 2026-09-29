"""E6-P v2 stage 2, Colab half of a Kaggle+Colab split: Colab part i/4 -> global parts 12+i, 16+i, 20+i of 24."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e6p_v2  # noqa: E402

TOTAL, OFFSET, COLAB_PARTS = 24, 12, 4

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    argv = sys.argv[1:]
    j = argv.index("--part") + 1
    i, k = map(int, argv[j].split("/"))
    assert k == COLAB_PARTS, f"launch with 4 Colab parts, got {k}"
    for g in range(OFFSET + i, TOTAL, COLAB_PARTS):
        argv[j] = f"{g}/{TOTAL}"
        e6p_v2.cli(list(argv), stage="2")
