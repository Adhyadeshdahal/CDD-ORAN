"""Old (feat/v2 a25b0a4) vs refactored pmrt_core (shared covariates helper): full Result equality.

    git show a25b0a4:cdd_oran/xmethod/methods/pmrt_core.py > REF.py
    PYTHONPATH=. uv run python scratchpad/xmethod/cov_equiv.py REF.py > scratchpad/xmethod/results/cov_equiv.jsonl
"""
import dataclasses, importlib.util, json, sys, time
import numpy as np
from cdd_oran.xmethod.worlds import generate_dataset
from cdd_oran.xmethod.methods import pmrt_core as NEW

spec = importlib.util.spec_from_file_location("pmrt_ref", sys.argv[1])
OLD = importlib.util.module_from_spec(spec); sys.modules["pmrt_ref"] = OLD; spec.loader.exec_module(OLD)

def perturb(ds, seed):
    rng = np.random.default_rng(seed)
    n = ds.n
    t = np.arange(n) + np.cumsum(rng.random(n) < 0.03) * 3          # time gaps
    perm = rng.permutation(n)                                        # row order != information order
    xk = ds.X_kpi_lag.copy(); xk[rng.random(xk.shape) < 0.02] = np.nan  # partly missing lagged KPIs
    def P(a): return None if a is None else np.asarray(a)[perm]
    des = tuple(dataclasses.replace(d, random_part=P(d.random_part), fixed_part=P(d.fixed_part),
                                    propensity=P(d.propensity)) for d in ds.designs)
    return dataclasses.replace(ds, X_action=ds.X_action[perm], X_kpi_lag=xk[perm], Y=ds.Y[perm], designs=des,
                               time_index=t[perm], context=P(ds.context))

cases = [("E1","R1",1.0),("E2","R2",1.0),("E3","R2",1.0),("E4","R1",1.5),("E4","R2",1.5),("E4","R3",1.5),
         ("E4","R4",1.5),("E5","R2",1.0)]
out = []
for w, r, lam in cases:
    for pert in (False, True):
        ds, _ = generate_dataset(w, r, 1000, 3000000, lam=lam)
        if pert:
            ds = perturb(ds, 7)
        t0 = time.process_time()
        a = OLD.PmrtCore().run(ds, {}); b = NEW.PmrtCore().run(ds, {})
        ea = [(e.source, e.target, e.score, e.p, e.sign, e.declared) for e in a.edges]
        eb = [(e.source, e.target, e.score, e.p, e.sign, e.declared) for e in b.edges]
        same_edges = repr(ea) == repr(eb)                               # repr: NaN == NaN, exact floats
        same_notes = all(repr(a.notes[k]) == repr(b.notes[k]) for k in ("z", "p_plus", "p_minus", "adjust",
                                                                        "not_applicable"))
        Xo = OLD.covariates(ds, OLD.info_order(ds), 2)
        Xn = NEW.covariates(ds, NEW.info_order(ds), 2)
        rec = {"cell": f"{w} {r} lam{lam}", "perturbed": pert, "edges_identical": same_edges,
               "notes_identical": same_notes, "X_identical": bool(Xo.shape == Xn.shape and np.array_equal(Xo, Xn)),
               "d": Xn.shape[1], "n_decl": sum(e[5] for e in eb), "cpu_s": round(time.process_time() - t0, 1)}
        print(json.dumps(rec), flush=True); out.append(rec)
print("ALL IDENTICAL:", all(r["edges_identical"] and r["notes_identical"] and r["X_identical"] for r in out))
