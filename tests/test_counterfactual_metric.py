from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.distributions import Normal

from cdd_oran.analysis.counterfactual_metrics import (
    aggregate_counterfactual_records,
    autoregressive_rollout_rmse,
    counterfactual_action_response_error,
    decision_regret,
    one_step_mse,
    study_correlations,
)
from cdd_oran.config import DEFAULT_CONFIG
from cdd_oran.experiments import evaluate


def test_known_counterfactual_values_and_member_shape():
    # These are the action-response arrays used by CF_ARRE: d_pred=[.5, 0].
    pred_action = np.array([[[0.5], [0.0]]])
    pred_reference = np.zeros((1, 2, 1))
    true_action = np.array([[1.0], [0.0]])
    true_reference = np.zeros((2, 1))

    arre = counterfactual_action_response_error(
        pred_action, pred_reference, true_action, true_reference
    )
    assert arre["per_member"].shape == (1,)
    assert arre["per_member"][0] == pytest.approx(0.125)
    assert arre["mean"] == pytest.approx(0.125)

    rmse = autoregressive_rollout_rmse(np.array([[[0.5], [0.5]]]), np.ones((2, 1)))
    assert rmse["per_member"][0] == pytest.approx(0.5)

    mse = one_step_mse(np.array([[[0.5], [0.0]]]), np.ones((2, 1)))
    assert mse["per_member"][0] == pytest.approx(0.625)

    two_members = counterfactual_action_response_error(
        np.array([[[0.5], [0.0]], [[1.0], [0.0]]]),
        np.zeros((2, 2, 1)),
        true_action,
        true_reference,
    )
    assert two_members["per_member"].tolist() == pytest.approx([0.125, 0.0])
    assert two_members["mean"] == pytest.approx(0.0625)

    with pytest.raises(ValueError, match="shape"):
        one_step_mse(np.zeros((2, 2, 1)), np.zeros((3, 1)))


def test_decision_regret_zero_oracle_and_seed_level_study():
    xapp = SimpleNamespace(threshold=0.0, mean=0.0, std=1.0, direction=0)
    utility_fns = [lambda params: float(params[0])]
    regret = decision_regret(
        [0.5], 0, [-1.0, 0.0, 1.0], utility_fns, [xapp], [1.0], planner_value=0.0
    )
    assert regret["oracle_action"] == pytest.approx(0.0)
    assert regret["decision_regret"] == pytest.approx(0.0)

    records = []
    for seed in range(5):
        for conflict in range(2):
            records.append(
                {
                    "environment": "tiny",
                    "training_seed": seed,
                    "planner": "stub",
                    "global_step": conflict,
                    "conflict_index": conflict,
                    "cf_arre": float(seed + conflict * 0.1),
                    "cf_rmse": float(seed * 0.5 + conflict * 0.1),
                    "one_step_mse": float(seed * 0.25 + conflict * 0.1),
                    "decision_regret": float(seed + conflict * 0.2),
                }
            )
    seed_rows = aggregate_counterfactual_records(records)
    assert len(seed_rows) == 5
    study = study_correlations(records, n_boot=100, n_permutations=100, seed=7)
    comparison = study["comparisons"][0]
    assert comparison["n_seeds"] == 5
    assert {row["metric"] for row in comparison["correlations"]} == {
        "cf_arre",
        "cf_rmse",
        "one_step_mse",
    }
    assert "holm_p_value" in comparison["delta_abs_rho"]["cf_arre"]


class _TinyParam:
    def __init__(self):
        self.value = 0.5

    def get_threshold(self):
        return (0.0, 2.0)

    def get_param(self):
        return self.value

    def set_param(self, value, params=None):
        self.value = float(np.clip(value, 0.0, 2.0))


class _TinyKpi:
    def __init__(self):
        self.value = 0.0

    def compute_utility_value(self):
        return self.value


class _TinyXApp:
    threshold = 0.0
    mean = 0.0
    std = 1.0
    direction = 0

    def __init__(self, params=None):
        self.params = tuple(params or ())


class _TinyEnv:
    num_params = 1
    num_kpis = 1
    num_bins = 2
    action_space = [0, 1, 0]

    def __init__(self):
        self.params = [_TinyParam()]
        self.kpis = [_TinyKpi()]
        self.xapps = [_TinyXApp(self.params)]
        self.step_calls = 0

    def action_to_param(self, action):
        value = float(action[1]) * 2.0 + float(action[2])
        return 0, float(np.clip(value, 0.0, 2.0))

    def step(self, action):
        self.step_calls += 1
        self.params[0].set_param(self.action_to_param(action)[1])
        self.kpis[0].value = self.params[0].get_param()
        return self.get_state(), 0.0, False, {}

    def get_state(self):
        return {"param0": np.array([self.params[0].get_param() / 2], dtype=np.float32), "kpi0": np.array([self.kpis[0].value], dtype=np.float32)}


class _StubModel:
    dynamics_mode = "hard_mask"
    device = torch.device("cpu")

    def predict_next_state(self, state, action):
        # The model emits a deterministic normalized KPI from the action's bin.
        mean = action[:, 1:2]
        return Normal(mean, torch.ones_like(mean))


class _StubPlanner:
    name = "stub"

    def __init__(self):
        self.calls = 0

    def act(self, **kwargs):
        self.calls += 1
        return [0, 0, 0]


def test_evaluator_collector_isolated_and_disabled_path(tmp_path, monkeypatch):
    env = _TinyEnv()
    planner = _StubPlanner()
    state = torch.tensor([0.25, 0.0])
    record = evaluate.collect_counterfactual_trajectory(
        model=_StubModel(),
        planner=planner,
        env=env,
        state_t=state,
        first_action=[0, 1, 0],
        param_id=0,
        xapps_under_conflict=env.xapps,
        weights_per_xapps=[1.0],
        scaling_term=10.0,
        raw_params=[0.5],
        utility_fns=[lambda params: float(params[0])],
        global_step=3,
        conflict_index=4,
        base_seed=9,
        planner_name="stub",
        horizon=2,
    )
    assert record["actions"][0].tolist() == [0, 1, 0]
    assert env.step_calls == 0
    assert planner.calls == 1
    record.update(
        {
            "training_seed": 9,
            "planner": "stub",
            "global_step": 3,
            "conflict_index": 4,
            "pairing_key": [9, "stub", 3, 4],
        }
    )
    assert record["pairing_key"] == [9, "stub", 3, 4]

    class _Writer:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_scalars(self, *args, **kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass

    class _EvalModel(_StubModel):
        def load_model(self, path):
            pass

        def get_binary_graph(self):
            return torch.zeros((2, 2), dtype=torch.bool)

    class _EvalEnv(_TinyEnv):
        def reset(self):
            self.params[0].value = 0.5
            self.kpis[0].value = 0.0

        def get_action_dim(self):
            return 3

        def get_utility_fns(self):
            return [lambda params: float(params[0])]

        def get_thresholds_stds(self):
            return [0.0], [(0.0, 1.0)]

        def get_kpi_to_xapp_mapping(self):
            return {}

    eval_dir = tmp_path / "disabled"
    eval_dir.mkdir()
    (eval_dir / "checkpoint.pt").write_bytes(b"stub")
    monkeypatch.setattr(evaluate, "SummaryWriter", _Writer)
    monkeypatch.setattr(evaluate, "get_env", lambda cfg: _EvalEnv())
    monkeypatch.setattr(evaluate, "get_model", lambda cfg, env, sampler=None: _EvalModel())
    monkeypatch.setattr(evaluate, "get_planners", lambda cfg, model, env: [])
    cfg = DEFAULT_CONFIG
    cfg = cfg.__class__(
        **{**cfg.__dict__, "device": "cpu", "num_steps": 1, "model_kind": "cdl"}
    )
    assert evaluate.main(cfg=cfg, run_dir=eval_dir) == 0
    assert not (eval_dir / "counterfactuals.json").exists()
