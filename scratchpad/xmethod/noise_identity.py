"""R-24 check: generate_dataset(kappa=None) is byte-identical to the pre-R-24 generator (feat/v2 1191678).

    git show 1191678:cdd_oran/xmethod/worlds/generate.py > REF.py
    PYTHONPATH=. uv run python scratchpad/xmethod/noise_identity.py REF.py
"""
import importlib.util
import json
import sys

from cdd_oran.xmethod.worlds import REGIMES_OF, dataset_hash, generate_dataset

spec = importlib.util.spec_from_file_location("gen_ref", sys.argv[1])
REF = importlib.util.module_from_spec(spec)
sys.modules["gen_ref"] = REF
spec.loader.exec_module(REF)
n_ok = n = 0
for w, rs in REGIMES_OF.items():
    for r in rs:
        for lam in ((0.0, 1.5) if w == "E4" else (1.0,)):
            for nn, s in ((300, 3_000_000), (1000, 3_000_017)):
                a = dataset_hash(REF.generate_dataset(w, r, nn, s, lam=lam)[0])
                b = dataset_hash(generate_dataset(w, r, nn, s, lam=lam)[0])
                n += 1
                n_ok += a == b
                if a != b:
                    print(json.dumps({"MISMATCH": [w, r, lam, nn, s]}))
print(json.dumps({"datasets": n, "identical_hash": n_ok, "all_identical": n_ok == n}))
