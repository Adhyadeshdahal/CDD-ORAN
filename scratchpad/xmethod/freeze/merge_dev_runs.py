"""Selective merge recipe for xm/dev-runs into the freeze tree (FREEZE_CHECKLIST C1; freeze-prep F21).

xm/dev-runs branched before feat/v2 was squashed and copied only some feat/v2 files, so a plain merge would bring
back stale copies (eval_analysis.py, the EVAL spec, PROTOCOL_A, CONTRACT, FIDELITY docs, test files) and delete
files it never had. This script classifies every path that differs between the destination (default HEAD = feat/v2
with xm/freeze-prep merged) and the source branch, and with --apply checks out the TAKE paths from the source. It
never commits, never deletes and never touches a path it cannot classify (REVIEW: listed, exit 2).

  uv run python scratchpad/xmethod/freeze/merge_dev_runs.py [--src xm/dev-runs] [--dst HEAD] [--apply]

Rules, per path (git diff --name-status DST SRC):
- only in SRC (A): TAKE if it matches TAKE_GLOBS (dev-runs' own deliverables), DROP if DROP_GLOBS (reference copies),
  else REVIEW.
- only in DST (D): KEEP (dev-runs is behind; e.g. cdl / pmrt_nl tests, briefs, other workers' results).
- in both (M): KEEP_ALWAYS paths are never taken (freeze-prep owns them; REVIEW if SRC looks newer). Otherwise by
  lineage: DST's blob in SRC's history of the path -> SRC is newer -> TAKE; SRC's blob in DST's history (or the
  pre-squash tag backup/xmethod-full-history) -> KEEP; else REVIEW.
After --apply: run the checks in FREEZE_CHECKLIST C5 (tests incl. tests/test_xmethod_campaign.py) and commit.
"""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys

TAKE_GLOBS = (
    "cdd_oran/xmethod/campaign.py", "cdd_oran/xmethod/dev_power.py", "tests/test_xmethod_campaign.py",
    "scratchpad/e6_dev/*.py", "scratchpad/e6_dev/runs/xm-dev-*", "scratchpad/xmethod/dev_*.py",
    "scratchpad/xmethod/specs/dev/*", "scratchpad/xmethod/results/dev/*", "scratchpad/xmethod/status/dev-runs.md",
)
DROP_GLOBS = ("scratchpad/xmethod/_ref/*",)
KEEP_ALWAYS = (
    "scratchpad/xmethod/eval_analysis.py", "scratchpad/xmethod/specs/eval/*", "docs/xmethod/PROTOCOL_A.md",
    "scratchpad/xmethod/CONTRACT.md", "tests/test_xmethod_eval_analysis.py", "scratchpad/xmethod/freeze/*",
)
HISTORY_TAGS = ("backup/xmethod-full-history",)


def git(*a: str, check: bool = True) -> str:
    p = subprocess.run(["git", *a], capture_output=True, text=True, encoding="utf-8")
    if check and p.returncode != 0:
        raise SystemExit(f"git {' '.join(a)} failed: {p.stderr.strip()}")
    return p.stdout


def match(path: str, globs) -> bool:
    return any(fnmatch.fnmatch(path, g) for g in globs)       # fnmatch '*' also crosses '/'


def blob(ref: str, path: str) -> str | None:
    out = subprocess.run(["git", "rev-parse", f"{ref}:{path}"], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def history_blobs(refs, path: str) -> set[str]:
    out = set()
    for ref in refs:
        if subprocess.run(["git", "rev-parse", "--verify", "-q", ref], capture_output=True).returncode != 0:
            continue
        for c in git("log", "--format=%H", ref, "--", path).split():
            b = blob(c, path)
            if b:
                out.add(b)
    return out


def classify(src: str, dst: str) -> dict[str, list[tuple[str, str]]]:
    res: dict[str, list[tuple[str, str]]] = {"TAKE": [], "KEEP": [], "DROP": [], "REVIEW": []}
    for line in git("diff", "--no-renames", "--name-status", dst, src).splitlines():
        st, path = line.split("\t", 1)
        if st == "A":
            k = "TAKE" if match(path, TAKE_GLOBS) else "DROP" if match(path, DROP_GLOBS) else "REVIEW"
            res[k].append((path, "only in source"))
        elif st == "D":
            res["KEEP"].append((path, "only in destination (source is behind)"))
        else:
            db, sb = blob(dst, path), blob(src, path)
            src_newer = db in history_blobs([src], path)
            dst_newer = sb in history_blobs([dst, *HISTORY_TAGS], path)
            if match(path, KEEP_ALWAYS):
                res["REVIEW" if src_newer and not dst_newer else "KEEP"].append(
                    (path, "owned by freeze-prep" + ("; SOURCE LOOKS NEWER" if src_newer and not dst_newer else "")))
            elif src_newer and not dst_newer:
                res["TAKE"].append((path, "source descends from destination's version"))
            elif dst_newer and not src_newer:
                res["KEEP"].append((path, "destination descends from source's version"))
            else:
                res["REVIEW"].append((path, "diverged (no lineage either way)"))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="xm/dev-runs")
    ap.add_argument("--dst", default="HEAD")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    res = classify(a.src, a.dst)
    for k in ("TAKE", "KEEP", "DROP", "REVIEW"):
        print(f"== {k} ({len(res[k])})")
        for p, why in res[k]:
            if k != "KEEP" or "only in destination" not in why:
                print(f"  {p}  [{why}]")
        if k == "KEEP":
            print(f"  (+ {sum('only in destination' in w for _, w in res[k])} paths only in the destination)")
    if res["REVIEW"]:
        print("REVIEW paths present: resolve them by hand (or extend the globs) before --apply.", file=sys.stderr)
        return 2
    if a.apply:
        if git("status", "--porcelain", "--untracked-files=no").strip():
            raise SystemExit("working tree has tracked changes: commit or set them aside first")
        take = [p for p, _ in res["TAKE"]]
        for i in range(0, len(take), 100):
            git("checkout", a.src, "--", *take[i:i + 100])
        print(f"checked out {len(take)} paths from {a.src}; nothing committed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
