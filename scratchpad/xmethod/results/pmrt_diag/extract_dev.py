"""Extract pmrt_eq / pmrt_r3 E2 R2 records (all kappa, n) from the DEV full merge (xm/dev-runs, read-only)."""
import gzip, json, sys
SRC = "D:/academia/major-project/CDD-ORAN-wt/xm-pmrt/scratchpad/xmethod/results/dev/full/merged.jsonl.gz"
out = []
with gzip.open(SRC, "rt") as f:
    for line in f:
        if '"E2|R2' not in line and "|E2|R2|" not in line:
            continue
        d = json.loads(line)
        if d["arm"] not in ("pmrt_eq", "pmrt_r3") or "|E2|R2|" not in d["key"]:
            continue
        out.append({"key": d["key"], "arm": d["arm"], "role": d["role"],
                    "edges": [[e[0], e[1], e[3]] if isinstance(e, list) else [e["source"], e["target"], e["p"]] for e in d["edges"]],
                    "z": d["notes"].get("z")})
json.dump(out, open(sys.argv[1], "w"))
print(len(out), out[0]["key"], out[0]["edges"][:2])
