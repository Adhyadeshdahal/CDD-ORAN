"""``cdl``: Causal Dynamics Learning, the authors' NaNA 2026 conference method (ruling R-50; Wang et al. 2022 CDL).

Model: ``_cdl_model.CDL``, a verbatim port of branch ``refactor/codebase`` ``cdd_oran/models/cdl.py`` @65b56f3 (see its
header): one masked predictor per target (per-node linear feature extractors, max-pool, MLP head -> Gaussian mean and
std), trained on full + one-node-masked NLL; per edge CMI = mean(masked NLL - full NLL), averaged over ``eval_steps``
batches and smoothed by an EMA (tau .99) started at 0; the conference declares CMI >= threshold.

Offline mapping (status Q1): CDL state nodes = every action column at t (the conference's NCPs were state nodes, its
edges NCP_t -> KPI_t+1) + every non-NaN lagged KPI + every context column; targets = Y (KPIs at t+1) only; the CDL
action input = a constant 0 (Study A rows have no meta-action); mask = uniformly random node (``informed_drop`` False:
there is no single changed param). Scaling as the conference: actions / context min-max to [0, 1], lagged KPIs and Y
z-scored (per dataset).
Training budget (ruling Q2, orchestrator: a fixed EPOCH budget equal to the conference's, so the data-to-update ratio
matches the paper's regime): gradient steps = min(``max_gradient_steps`` 16 000, ceil(``epochs`` 130 * n / 128)).
(Conference Env I: 16 000 Adam steps of 128 rows on a ~16 000-transition train buffer = ~130 epochs; 16 000 is its
step count.) Each step = one Adam step on 128 rows drawn with replacement from the dataset (offline the whole dataset
is the buffer); every ``eval_steps``-th step (step % 10 == 0) one CMI batch; the EMA moves every ``eval_steps`` CMI
batches, as in the conference loop. So n 500 / 1000 / 4000 / >= 15 754: 508 / 1 016 / 4 063 / 16 000 steps. The EMA
starts at 0 (conference), so after U = steps / 100 updates every CMI is shrunk by about 1 - .99^U (.05 / .10 / .33 /
.80 at those n): the R-29 tau is per (cell, n), so the primary rule is unaffected; the fixed .16 threshold is not
comparable across n (secondary only). The brief's first rule (16 000 steps for every n) overfits at small n: null
CMIs .2-.5 nats at n 1000 (FIDELITY_CDL.md).
Hyper-parameters (conference config): lr .001, batch 128, grad clip 10, feature 64, predictor [64, 64], EMA .99.
Score = final EMA CMI of (source node, target); sign = R-4 ``pcorr_given_Z`` (native set). Primary declaration =
score > R-29 conformal placebo tau (``ClassicBase.tune``); the conference's fixed threshold (CMI >= .16,
configs/env_i_cdl.yaml; the paper text says .2) is the ``native`` rule (``notes['native_declared']``), secondary.
No conditioning interface: native arm only. R-37: the primary fit has no ``P_placebo_conf`` node; its candidates are
read from a second fit with every action (``notes['diagnostic_fit']``).
RNG: torch seeded from ``default_rng([7804, dataset seed, fit index])`` (tag 7804 = cdl), global torch RNG state
restored afterwards; ``torch.use_deterministic_algorithms`` on during the fit (CPU runs are bitwise reproducible).
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch  # imported at module load so ClassicBase._threads pins its thread count

from cdd_oran.xmethod import api

from ._classic_common import (
    PLACEBO_CONF,
    ClassicBase,
    Scored,
    columns,
    resolve,
    sign_pcorr_given_Z,
    target_index,
)

RNG_TAG_CDL = 7804
SOURCE = "refactor/codebase@65b56f3 cdd_oran/models/cdl.py"


def fit_seed(data: api.Dataset, fit_idx: int) -> int:
    return int(np.random.default_rng([RNG_TAG_CDL, int(data.seed), int(fit_idx)]).integers(0, 2**31 - 1))


def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = a.min(0), a.max(0)
    return (a - lo) / np.where(hi > lo, hi - lo, 1.0)


def _z(a: np.ndarray) -> np.ndarray:
    sd = a.std(0)
    return (a - a.mean(0)) / np.where(sd > 0, sd, 1.0)


def gradient_steps(n: int, config: dict[str, Any]) -> int:
    """Q2 rule: min(max_gradient_steps, ceil(epochs * n / batch_size))."""
    return int(min(int(config["max_gradient_steps"]), math.ceil(float(config["epochs"]) * n / int(config["batch_size"]))))


def fit_cmi(S: np.ndarray, Y: np.ndarray, seed: int, config: dict[str, Any]) -> dict[str, Any]:
    """Train CDL offline on rows (S [n, d] state nodes, Y [n, k] targets); returns the final EMA CMI [k, d + 1] (last
    column = the constant action node) and run facts."""
    from ._cdl_model import CDL

    dev = torch.device(config.get("device", "cpu"))
    det = torch.are_deterministic_algorithms_enabled()
    t0 = time.perf_counter()
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        torch.use_deterministic_algorithms(True, warn_only=dev.type == "cuda")
        try:
            torch.manual_seed(seed)
            model = CDL(state_dim=S.shape[1], action_dim=1, kpi_start=0,
                        feature_fc_dims=tuple(config["feature_fc_dims"]),
                        generative_fc_dims=tuple(config["generative_fc_dims"]), lr=config["lr"],
                        cmi_threshold=config["cmi_threshold"], eval_tau=config["eval_tau"],
                        grad_clip=config["grad_clip"], device=dev, node_names=None,
                        eval_steps=config["eval_steps"], target_dim=Y.shape[1], informed_drop=False)
            s_all = torch.as_tensor(S, dtype=torch.float32, device=dev)
            y_all = torch.as_tensor(Y, dtype=torch.float32, device=dev)
            n, bs = S.shape[0], int(config["batch_size"])
            a0 = torch.zeros(bs, 1, device=dev)
            every = int(config["eval_steps"])
            loss = float("nan")
            for step in range(gradient_steps(n, config)):
                idx = torch.randint(n, (bs,)).to(dev)
                loss = model.fit_step(s_all[idx], y_all[idx], a0)
                if step % every == 0:
                    idx = torch.randint(n, (bs,)).to(dev)
                    model.cmi_step(s_all[idx], y_all[idx], a0)
            cmi = model.mask_CMI.detach().cpu().numpy().astype(float)
            loss = float(loss)
        finally:
            torch.use_deterministic_algorithms(det)
    return {"cmi": cmi, "final_loss": loss, "wall_s": time.perf_counter() - t0, "device": str(dev),
            "torch": torch.__version__}


class CDLMethod(ClassicBase):
    name = "cdl"
    version = f"{SOURCE} port, offline 130-epoch rule (cap 16k steps)"
    uses_p = False
    method_idx = 7
    defaults: dict[str, Any] = {"lr": 0.001, "batch_size": 128, "grad_clip": 10.0, "feature_fc_dims": [64, 64],
                                "generative_fc_dims": [64, 64], "eval_tau": 0.99, "eval_steps": 10,
                                "epochs": 130, "max_gradient_steps": 16000,
                                "cmi_threshold": 0.16, "device": "cpu"}

    def _score(self, data: api.Dataset, config: dict[str, Any]):
        cols = columns(data)
        Y = cols.Y
        ok_y = [k for k in range(Y.shape[1]) if np.std(Y[:, k]) > 0]

        def fit(with_conf: bool, fit_idx: int):
            """State nodes = actions (minus P_placebo_conf unless ``with_conf``; R-37) + lagged KPIs + context."""
            use = [j for j, a in enumerate(data.action_names) if with_conf or a != PLACEBO_CONF]
            S = np.column_stack([_minmax(cols.actions[:, use]), _z(cols.lags), _minmax(cols.context)])
            seed = fit_seed(data, fit_idx)
            r = fit_cmi(S, _z(Y[:, ok_y]), seed, config)
            r.update(use=use, seed=seed)
            return r

        prim = fit(False, 0)
        diag = fit(True, 1) if any(s == PLACEBO_CONF for s, _ in data.candidates) else None
        thr = float(config["cmi_threshold"])
        out = []
        for s, t in data.candidates:
            fam, j = resolve(data, cols, s)
            ti = target_index(data, t)
            r = diag if s == PLACEBO_CONF else prim
            if j is None or ti not in ok_y or (fam == "action" and j not in r["use"]):
                out.append(Scored(s, t, fam, float("nan"), 0, None, None))
                continue
            node = r["use"].index(j) if fam == "action" else len(r["use"]) + j
            v = float(r["cmi"][ok_y.index(ti), node])
            out.append(Scored(s, t, fam, v, sign_pcorr_given_Z(data, cols, fam, j, ti), None, bool(v >= thr)))

        def facts(r):
            names = [data.action_names[j] for j in r["use"]] + list(cols.lag_names) + [
                f"ctx:{c}" for c in range(cols.context.shape[1])] + ["action_node"]
            return {"seed": r["seed"], "final_loss": r["final_loss"], "wall_s": r["wall_s"], "device": r["device"],
                    "torch": r["torch"], "cmi": {data.kpi_names[k]: dict(zip(names, map(float, r["cmi"][i]),
                                                                               strict=True))
                                                for i, k in enumerate(ok_y)}}

        gsteps = gradient_steps(data.n, config)
        notes = {"sign_rule": "pcorr_given_Z", "source": SOURCE, "gradient_steps": gsteps,
                 "native_rule": f"conference fixed threshold CMI >= {thr} (secondary)", "fit": facts(prim)}
        if diag is not None:
            notes["diagnostic_fit"] = {"rule": f"R-37: {PLACEBO_CONF} candidates read from a second fit with every "
                                               "action; primary fit without it", **facts(diag)}
        return out, notes
