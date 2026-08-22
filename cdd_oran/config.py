from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml


@dataclass(frozen=True)
class ModelConfig:
    lr: float
    cmi_threshold: float
    eval_tau: float
    grad_clip: float
    generative_fc_dims: tuple[int, ...]
    feature_fc_dims: tuple[int, ...]
    batch_size: int
    # Interventional-CMI reweight (CDL only). 1.0 = OFF, bit-identical to baseline.
    interv_weight: float = 1.0
    # Phase 2 structure-conditioned dynamics (CDL only). "hard_mask" is the legacy
    # default and stays byte-identical; "structure_conditioned" is the new world model.
    dynamics_mode: str = "hard_mask"
    residual_bound: float = 0.25
    residual_l2: float = 1e-2
    residual_l1: float = 1e-3
    residual_hidden: tuple[int, ...] = (64, 64)
    residual_alert_fraction: float = 0.25
    # Trained CDL run whose P1 bootstrap posterior supplies the structure sampler for
    # structure_conditioned mode. Legacy/offline only; the train path uses the artifacts below.
    posterior_run: str | None = None
    # Precomputed CALIBRATED posterior artifact (GraphPosterior.save) loaded at train/eval time
    # for the structure sampler. Required for structure_conditioned; no online bootstrap.
    posterior_artifact: str | None = None
    # Frozen crisp enumeration-graph artifact (freeze_enumeration_graph) used for conflict
    # enumeration, kept separate from sampled prediction structures.
    enumeration_graph: str | None = None
    # Phase 3: ensemble members returned by predict_next_state (structure_conditioned).
    # 1 = single (batch,k) prediction; >1 exposes the member axis for robust planning.
    predict_members: int = 1


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int
    init_steps: int
    inference_gradient_steps: int
    eval_steps: int
    plot_freq: int
    test_batch_size: int


@dataclass(frozen=True)
class CEMConfig:
    n_candidate: int
    n_top: int
    n_iter: int


@dataclass(frozen=True)
class MPPIConfig:
    n_samples: int
    temperature: float
    noise_sigma: float


@dataclass(frozen=True)
class MCTSConfig:
    n_simulations: int
    ucb_c: float


@dataclass(frozen=True)
class PlannerConfig:
    n_horizon: int
    cem: CEMConfig
    mppi: MPPIConfig
    mcts: MCTSConfig
    joint: bool = False
    risk_kappa: float = 0.0
    # Phase 3 robust planning over per-model returns (only active when the world model
    # returns an ensemble, m>1). m=1 is byte-identical regardless of these.
    ensemble_aggregator: str = "quantile"  # "mean" | "quantile" | "kappa"
    ensemble_quantile: float = 0.9  # upper cost quantile (risk-averse) for "quantile"
    ensemble_kappa: float | None = None  # for "kappa"; None -> reuse risk_kappa
    disagreement_penalty: float = 0.0  # additive OOD penalty on cost; 0 = off
    ood_threshold: float | None = None  # disagreement OOD warning threshold; None = off


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    mitigation_seed: int
    environment: Literal["EnvironmentI", "EnvironmentII", "EnvironmentIII", "EnvironmentIV"]
    model_kind: Literal["cdl", "mlp"]
    param_ranges: Literal["train", "ood"]
    evaluation_param_ranges: Literal["train", "ood"]
    num_steps: int
    device: str
    deterministic: bool
    model: ModelConfig
    train: TrainConfig
    planner: PlannerConfig


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> ExperimentConfig:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).parent.parent / "configs" / path
    with path.open() as config_file:
        values = cast(dict[str, Any], yaml.safe_load(config_file))

    for override in overrides:
        try:
            key, value = override.split("=", 1)
        except ValueError as error:
            raise ValueError(f"Invalid override {override!r}; expected key=value") from error
        target = values
        *parents, leaf = key.split(".")
        for parent in parents:
            if parent not in target or not isinstance(target[parent], dict):
                raise ValueError(f"Unknown config path: {key}")
            target = target[parent]
        if leaf not in target:
            raise ValueError(f"Unknown config path: {key}")
        target[leaf] = yaml.safe_load(value)

    device = values["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return ExperimentConfig(
        seed=values["seed"],
        mitigation_seed=values.get("mitigation_seed", values["seed"]),
        environment=values["environment"],
        model_kind=values["model_kind"],
        param_ranges=values["param_ranges"],
        evaluation_param_ranges=values.get("evaluation_param_ranges", values["param_ranges"]),
        num_steps=values["num_steps"],
        device=device,
        deterministic=values.get("deterministic", False),
        model=ModelConfig(
            lr=cast(float, values["model"]["lr"]),
            cmi_threshold=cast(float, values["model"]["cmi_threshold"]),
            eval_tau=cast(float, values["model"]["eval_tau"]),
            grad_clip=cast(float, values["model"]["grad_clip"]),
            generative_fc_dims=tuple(cast(list[int], values["model"]["generative_fc_dims"])),
            feature_fc_dims=tuple(cast(list[int], values["model"]["feature_fc_dims"])),
            batch_size=cast(int, values["model"]["batch_size"]),
            interv_weight=cast(float, values["model"].get("interv_weight", 1.0)),
            dynamics_mode=cast(str, values["model"].get("dynamics_mode", "hard_mask")),
            residual_bound=cast(float, values["model"].get("residual_bound", 0.25)),
            residual_l2=cast(float, values["model"].get("residual_l2", 1e-2)),
            residual_l1=cast(float, values["model"].get("residual_l1", 1e-3)),
            residual_hidden=tuple(cast(list[int], values["model"].get("residual_hidden", [64, 64]))),
            residual_alert_fraction=cast(
                float, values["model"].get("residual_alert_fraction", 0.25)
            ),
            posterior_run=cast(
                "str | None", values["model"].get("posterior_run", None)
            ),
            posterior_artifact=cast(
                "str | None", values["model"].get("posterior_artifact", None)
            ),
            enumeration_graph=cast(
                "str | None", values["model"].get("enumeration_graph", None)
            ),
            predict_members=cast(int, values["model"].get("predict_members", 1)),
        ),
        train=TrainConfig(**values["train"]),
        planner=PlannerConfig(
            n_horizon=values["planner"]["n_horizon"],
            cem=CEMConfig(**values["planner"]["cem"]),
            mppi=MPPIConfig(**values["planner"]["mppi"]),
            mcts=MCTSConfig(**values["planner"]["mcts"]),
            joint=values["planner"].get("joint", False),
            risk_kappa=values["planner"].get("risk_kappa", 0.0),
            ensemble_aggregator=values["planner"].get("ensemble_aggregator", "quantile"),
            ensemble_quantile=values["planner"].get("ensemble_quantile", 0.9),
            ensemble_kappa=values["planner"].get("ensemble_kappa", None),
            disagreement_penalty=values["planner"].get("disagreement_penalty", 0.0),
            ood_threshold=values["planner"].get("ood_threshold", None),
        ),
    )


DEFAULT_CONFIG = load_config("env_i_mlp.yaml")


def config_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return asdict(cfg)
