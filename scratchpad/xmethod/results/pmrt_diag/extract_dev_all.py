"""All arms' E2 R2 n 1000 truth-null raw-p (p <= .05) indicators per seed from the DEV merge (read-only)."""
import gzip, json, sys, collections
SRC = "D:/academia/major-project/CDD-ORAN-wt/xm-pmrt/scratchpad/xmethod/results/dev/full/merged.jsonl.gz"
sys.path.insert(0, ".")
from cdd_oran.xmethod.worlds.generate import truth_for
T = truth_for("E2", "R2")
out = collections.defaultdict(dict)
with gzip.open(SRC, "rt") as f:
    for line in f:
        if "|E2|R2|" not in line or "|n1000|" not in line:
            continue
        d = json.loads(line)
        if d["role"] != "measure":
            continue
        E = [x if isinstance(x, list) else [x["source"], x["target"], x.get("score"), x.get("p"), x.get("declared")] for x in d["edges"]]
        e = [x for x in E if (x[0], x[1]) in T.null_edges and x[0].startswith("P") and x[0] != "P_placebo"]
        if not e or all(x[3] is None for x in e):
            dec = [bool(x[4]) for x in e]
            out[d["key"].split("|s")[0]][d["key"].split("|s")[1]] = {"decl": sum(dec), "m": len(dec), "raw": None}
            continue
        ps = [x[3] for x in e if x[3] is not None]
        out[d["key"].split("|s")[0]][d["key"].split("|s")[1]] = {"raw": sum(p <= .05 for p in ps), "m": len(ps)}
json.dump(out, open(sys.argv[1], "w"))
for k, v in sorted(out.items()):
    r = [x["raw"] for x in v.values() if x["raw"] is not None]
    if r:
        print(k, len(v), round(sum(r) / sum(x["m"] for x in v.values()), 4))
