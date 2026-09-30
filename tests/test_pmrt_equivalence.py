"""Label-only PMRT rename: the renamed code + E6P_PMRT_V4.json reproduces the frozen (4fc2cd9, "MSCR+") code +
E6P_MSCRPLUS_V4_FROZEN.json bit for bit (scratchpad/e6_dev/pmrt_equivalence.py)."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scratchpad", "e6_dev"))

import pmrt_equivalence as EQ  # noqa: E402


def _small():
    for base in (ROOT, os.path.join(ROOT, "..", "..", "..")):          # worktree -> main repo (.tmp is untracked)
        d = os.path.join(base, ".tmp", "mscr_plus", "infra", "cache")
        if all(os.path.exists(os.path.join(d, f"{n}.npz")) for n, _ in EQ.SMALL):
            return os.path.abspath(d)
    return None


def _has_commit(c):
    return subprocess.run(["git", "-C", ROOT, "cat-file", "-e", f"{c}^{{commit}}"], capture_output=True).returncode == 0


def test_diff_detects_changes():
    a = {"x": 1.0, "y": [1, float("nan")], "artifact": "a", "z": {"p": .5}}
    assert EQ.diff(a, {"x": 1.0, "y": [1, float("nan")], "artifact": "b", "z": {"p": .5}}, ignore={"artifact"}) == []
    d = EQ.diff(a, {"x": 1.0 + 1e-16 * 3, "y": [1, 2.0], "artifact": "a", "z": {}})
    assert {p for p, _, _ in d} == {"/x", "/y[1]", "/z/p"}
    assert EQ.max_abs(d) == float("inf")


@pytest.mark.skipif(_small() is None or not _has_commit(EQ.FROZEN_COMMIT), reason="small caches / frozen commit")
def test_local_equivalence_small():
    rep = EQ.cmd_local(type("A", (), {"small": _small(), "B": 199, "frozen_commit": EQ.FROZEN_COMMIT})())
    assert rep["equivalent"] and rep["byte_identical"] and rep["n_diff"] == 0, rep["first_diffs"]
    assert rep["n_leaves"] > 10000
