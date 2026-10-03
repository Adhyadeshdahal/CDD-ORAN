"""cdl adapter (ruling R-50): the port of the conference CDL model and its offline Study A adapter.

Fast: tiny training budgets (``total_steps`` just above ``init_steps``) and small nets where the test is about the
plumbing; the golden check uses the conference sizes. The full-length F2 / F3 checks against refactor/codebase are in
``scratchpad/xmethod/cdl_fidelity.py`` (they need that branch's code).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from cdd_oran.xmethod.methods import _classic_synth as SYN
from cdd_oran.xmethod.methods._cdl_model import CDL
from cdd_oran.xmethod.methods.classic import METHODS

FAST = {"max_gradient_steps": 200, "feature_fc_dims": [8, 8], "generative_fc_dims": [8, 8]}   # 200 steps, 2 EMA updates

# refactor/codebase tests/golden/EnvironmentI_CDL.json: initial state and the untrained CDL's predict_next_state probe
GOLDEN_STATE = [0.9646475911, 0.7236853242, 0.642475307, 0.717453599, 0.4675990045, 0.3255846798, 0.4396446049,
                -1.5278024673, -0.1549416035, -1.6454441547, -1.7974038124]
GOLDEN_PROBE = [[0.11801817, 0.00609978, 0.26027608, 0.15531325], [1.00701094, 1.1368022, 1.06562746, 1.10233963]]


def test_port_reproduces_golden_probe():
    """F2: tests/_harness.build_model (seed 12345, golden MODEL_KWARGS) + _model_probe, Environment I."""
    torch.manual_seed(12345)
    m = CDL(state_dim=11, action_dim=3, kpi_start=7, device=torch.device("cpu"), node_names=[f"n{i}" for i in range(11)],
            cmi_threshold=0.2, eval_tau=0.99, grad_clip=10.0, generative_fc_dims=[64, 64], feature_fc_dims=[64, 64],
            lr=1e-3)
    torch.manual_seed(12345)
    with torch.no_grad():
        dist = m.predict_next_state(torch.tensor([GOLDEN_STATE]), torch.tensor([[0.0, 1.0, 2.0]]))
    np.testing.assert_allclose([dist.mean.flatten().tolist(), dist.scale.flatten().tolist()], GOLDEN_PROBE,
                               rtol=1e-5, atol=1e-7)


def test_port_target_dim_and_uninformed_mask():
    """PORT changes: target_dim rows of the CMI; informed_drop switches the mask (same RNG draws, so with every
    a[:, 0] = 0 the informed model drops node 0 on about half the rows and the uninformed one does not)."""
    def make(**kw):
        torch.manual_seed(0)
        return CDL(state_dim=4, action_dim=1, kpi_start=0, feature_fc_dims=(8, 8), generative_fc_dims=(8, 8), lr=1e-3,
                   cmi_threshold=0.2, eval_tau=0.9, grad_clip=10.0, device="cpu", node_names=None, eval_steps=1, **kw)
    s, y = torch.rand(16, 4), torch.rand(16, 2)
    m = make(target_dim=2, informed_drop=False)
    m.fit_step(s, y, torch.zeros(16, 1))
    m.cmi_step(s, y, torch.zeros(16, 1))
    assert m.get_causal_graph().shape == (2, 5)
    a0 = torch.zeros(16, 1)
    losses = []
    for informed in (False, False, True):
        m = make(target_dim=2, informed_drop=informed)
        torch.manual_seed(1)
        losses.append(float(m.fit_step(s, y, a0)))
    assert losses[0] == losses[1] and losses[0] != losses[2]


@pytest.fixture(scope="module")
def syn():
    return SYN.make(400, 3_000_000, b=1.0)


@pytest.fixture(scope="module")
def res(syn):
    return METHODS["cdl"]().run(syn[0], dict(FAST, tau=0.01))


def test_contract(syn, res):
    d, _ = syn
    assert [(e.source, e.target) for e in res.edges] == list(d.candidates)
    assert all(np.isfinite(e.score) and e.p is None and e.sign in (-1, 0, 1) for e in res.edges)
    assert res.notes["sign_rule"] == "pcorr_given_Z" and res.notes["arm"] == "native"
    assert res.notes["gradient_steps"] == 200 and res.notes["fit"]["device"] == "cpu"
    assert [e.declared for e in res.edges] == [e.score > 0.01 for e in res.edges]
    cmi = res.notes["fit"]["cmi"]
    assert set(cmi) == set(d.kpi_names) and list(cmi["Y0"])[-1] == "action_node"
    e = next(x for x in res.edges if (x.source, x.target) == ("A1", "Y2"))
    assert e.score == cmi["Y2"]["A1"]
    assert set(res.notes["native_declared"]) == {f"{x.source}->{x.target}" for x in res.edges if x.score >= 0.16}


def test_gradient_step_rule():
    """Q2 (orchestrator): min(16 000, ceil(130 n / 128)) gradient steps."""
    from cdd_oran.xmethod.methods.cdl import CDLMethod, gradient_steps
    cfg = CDLMethod().default_config()
    assert [gradient_steps(n, cfg) for n in (500, 1000, 4000, 15754, 24000)] == [508, 1016, 4063, 16000, 16000]


def test_deterministic(syn, res):
    again = METHODS["cdl"]().run(syn[0], dict(FAST, tau=0.01))
    assert [e.score for e in again.edges] == [e.score for e in res.edges]


def test_untuned_declares_nothing(syn):
    r = METHODS["cdl"]().run(syn[0], dict(FAST, max_gradient_steps=20))
    assert not any(e.declared for e in r.edges) and "untuned" in r.notes["declare_rule"]


def test_tune_sets_placebo_tau():
    dev = [SYN.make(300, 3_000_000 + i, b=1.0)[0] for i in range(2)]
    cfg = METHODS["cdl"]().tune(dev, config=dict(FAST, max_gradient_steps=20))
    assert np.isfinite(cfg["tau"]) and cfg["n_placebo_scores"] == 6 and cfg["dev_seeds"] == [3_000_000, 3_000_001]


def _conf_scrambled(ds):
    j = ds.action_names.index("P_placebo_conf")
    X = ds.X_action.copy()
    X[:, j] = np.random.default_rng(11).permutation(X[:, j])[::-1]
    des = list(ds.designs)
    if des[j].propensity is not None:
        des[j] = dataclasses.replace(des[j], propensity=des[j].propensity[::-1].copy())
    return dataclasses.replace(ds, X_action=X, designs=tuple(des))


@pytest.mark.parametrize("regime", ["R3", "R4"])
def test_placebo_conf_does_not_change_primary(regime):
    """R-37: scrambling P_placebo_conf leaves every other candidate unchanged; its own come from a second fit."""
    from cdd_oran.xmethod.worlds import generate_dataset
    ds, _ = generate_dataset("E4", regime, 300, 3_000_001, lam=1.5, kappa=0.25)
    m = METHODS["cdl"]()
    a, b = m.run(ds, FAST), m.run(_conf_scrambled(ds), FAST)

    def rows(r):
        return repr([(e.source, e.target, e.score, e.sign) for e in r.edges if e.source != "P_placebo_conf"])
    assert rows(a) == rows(b)
    assert all("P_placebo_conf" not in c for c in a.notes["fit"]["cmi"]["K0"])
    assert "R-37" in a.notes["diagnostic_fit"]["rule"] and "P_placebo_conf" in a.notes["diagnostic_fit"]["cmi"]["K0"]
    assert all(np.isfinite(e.score) for e in a.edges if e.source == "P_placebo_conf")


def test_runner_load_method():
    from cdd_oran.xmethod.runner import load_method
    m = load_method("cdd_oran.xmethod.methods.cdl:CDLMethod")
    assert m.name == "cdl" and not m.uses_p
