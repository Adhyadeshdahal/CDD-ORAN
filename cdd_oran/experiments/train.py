import logging
import random
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cdd_oran.config import DEFAULT_CONFIG, ExperimentConfig
from cdd_oran.conflicts import state_to_tensor
from cdd_oran.envs import get_env
from cdd_oran.models import get_model
from cdd_oran.policies.random_policy import RandomPolicy
from cdd_oran.utils.runs import create_run_dir, write_metrics
from cdd_oran.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Transitions originate in the CPU-only environment. Keep them contiguous
        # there and move one sampled batch to the training device.
        self.data = torch.empty((capacity, 2 * state_dim + action_dim))
        self.size = 0
        self._next_index = 0

    def __len__(self):
        return self.size

    def add(self, s, a, s_next):
        row = self.data[self._next_index]
        row[: self.state_dim].copy_(torch.as_tensor(s, dtype=torch.float32))
        row[self.state_dim : self.state_dim + self.action_dim].copy_(
            torch.as_tensor(a, dtype=torch.float32)
        )
        row[self.state_dim + self.action_dim :].copy_(
            torch.as_tensor(s_next, dtype=torch.float32)
        )
        self._next_index = (self._next_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(self.size, (batch_size,))
        batch = self.data[idx]
        if self.device.type != "cpu":
            batch = batch.to(self.device)
        action_end = self.state_dim + self.action_dim
        return batch[:, : self.state_dim], batch[:, self.state_dim : action_end], batch[:, action_end:]


def main(cfg: ExperimentConfig = DEFAULT_CONFIG, resume=False, run_dir=None):
    seed_everything(cfg.seed, cfg.deterministic)
    env = get_env(cfg)

    ground_truth_causal_graph = env.true_adj_matrix
    run_dir = create_run_dir(cfg) if run_dir is None else run_dir
    checkpoint_path = run_dir / "checkpoint.pt"
    state_dim = env.get_state_dim()
    action_dim = env.action_dim

    random_policy = RandomPolicy(action_dim=env.action_dim, action_space=env.action_space)
    sampler = None
    artifact_manifest = None
    if cfg.model_kind == "cdl" and cfg.model.posterior_artifact:
        from dataclasses import replace as _replace

        from cdd_oran.models import (
            make_structure_sampler,
            stage_run_artifacts,
            verify_checkpoint_manifest,
            verify_run_artifacts,
        )

        if resume:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            # Resume is bound to the artifacts already staged with this run. Never recopy
            # mutable external paths over them before checking the old checkpoint.
            artifact_manifest, resolved = verify_run_artifacts(
                run_dir, env=env, environment=cfg.environment
            )
            verify_checkpoint_manifest(checkpoint_path, artifact_manifest)
        else:
            # Fresh runs stage and validate external artifacts before constructing the sampler.
            artifact_manifest = stage_run_artifacts(cfg, run_dir, env=env)
            resolved = {
                "posterior": str(run_dir / "artifacts" / "posterior.json"),
                "enumeration_graph": str(run_dir / "artifacts" / "enumeration_graph.json"),
            }
        cfg = _replace(
            cfg,
            model=_replace(
                cfg.model,
                posterior_artifact=resolved["posterior"],
                enumeration_graph=resolved["enumeration_graph"],
            ),
        )
        # Load the calibrated posterior artifact (pure I/O, no online bootstrap, no global RNG
        # mutation -- review blocker #4). Then reseed explicitly so P2 model init + data gen
        # start from cfg.seed regardless of the load.
        sampler = make_structure_sampler(cfg, seed=cfg.seed, env=env)
        seed_everything(cfg.seed, cfg.deterministic)
    model = get_model(cfg, env, sampler=sampler)
    if artifact_manifest is not None:
        model.artifact_manifest = artifact_manifest  # persisted in the checkpoint

    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model.load_model(checkpoint_path)
        logger.info("Resumed from checkpoint: %s", checkpoint_path)

    writer = SummaryWriter(run_dir / "tensorboard")

    train_buffer = ReplayBuffer(cfg.train.total_steps, state_dim, action_dim, cfg.device)
    test_buffer = ReplayBuffer(cfg.train.total_steps, state_dim, action_dim, cfg.device)
    obs = env.reset()
    expected_device = torch.device(cfg.device)
    parameter_device = next(model.models.parameters()).device
    if parameter_device.type != expected_device.type or (
        expected_device.index is not None and parameter_device.index != expected_device.index
    ):
        raise RuntimeError(
            f"Model is on {parameter_device}, but config requested {expected_device}"
        )
    logger.info(
        "Training device=%s parameter_device=%s torch=%s CUDA=%s",
        expected_device,
        parameter_device,
        torch.__version__,
        torch.version.cuda,
    )
    if cfg.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {cfg.device}")
        logger.info(
            "GPU=%s",
            torch.cuda.get_device_name(parameter_device),
        )
    episode_reward = 0
    episode_rewards = []
    metrics: dict[str, Any] = {"prediction_mse": None}
    for step in range(cfg.train.total_steps):
        action = random_policy.act()

        next_obs, reward, done, info = env.step(action)

        s_tensor = state_to_tensor(obs)
        s_next_tensor = state_to_tensor(next_obs)
        train_buffer.add(
            s_tensor, action, s_next_tensor
        ) if random.random() > 0.2 else test_buffer.add(s_tensor, action, s_next_tensor)
        episode_reward += reward

        obs = next_obs
        if done:
            obs = env.reset()
            episode_rewards.append(episode_reward)
            logger.info("Episode %d reward: %s", len(episode_rewards), episode_rewards[-1])

            episode_reward = 0

        if step < cfg.train.init_steps:
            continue

        for _ in range(cfg.train.inference_gradient_steps):
            s, a, s_next = train_buffer.sample(cfg.model.batch_size)
            a = a.reshape(-1, action_dim)
            s_pair = torch.stack([s, s_next], dim=1)
            loss = model.train_step(s_pair, a)

        # The structure-conditioned world model draws its structures from the injected P1
        # sampler; the CMI graph it consumes for conflict enumeration is a frozen offline
        # artifact, so there is no in-loop CMI (update_mask) estimation here.

        if step % cfg.train.plot_freq == 0:
            if cfg.model_kind == "cdl":
                if len(test_buffer) >= cfg.train.test_batch_size:
                    s_b, a_b, s_1b = test_buffer.sample(cfg.train.test_batch_size)
                    s_b, a_b, s_1b = (
                        s_b,
                        a_b.reshape(-1, action_dim),
                        s_1b,
                    )
                    mse = model.evaluate_predictions(s_b, a_b, s_1b)
                    metrics["prediction_mse"] = mse
                    logger.info("Next-step prediction MSE: %s", mse)
                    writer.add_scalar("Predictions/MSE", mse, step)

                pred = model.get_binary_graph()[:, :-1].cpu().detach().numpy()
                gt = ground_truth_causal_graph

                tp = np.sum((pred == 1) & (gt == 1))
                fp = np.sum((pred == 1) & (gt == 0))
                fn = np.sum((pred == 0) & (gt == 1))
                tn = np.sum((pred == 0) & (gt == 0))

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                writer.add_scalar("graph_eval/precision", precision, step)
                writer.add_scalar("graph_eval/recall", recall, step)
                writer.add_scalar("graph_eval/f1", f1, step)
                writer.add_scalar("graph_eval/accuracy", accuracy, step)

                metrics["graph"] = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "accuracy": float(accuracy),
                }
                logger.info(
                    "Step %d loss=%.4f precision=%s recall=%s f1=%s accuracy=%s edges=%s",
                    step,
                    loss.item(),
                    precision,
                    recall,
                    f1,
                    accuracy,
                    pred.sum(),
                )

                # Residual reporting (CDL world model only): the contribution fraction is the
                # anti-collapse measurement, reported at every eval.
                if (
                    hasattr(model, "residual_diagnostics")
                    and len(test_buffer) >= cfg.train.test_batch_size
                ):
                    # Reuse the SAME held-out batch as the MSE report above (review MINOR #8),
                    # so prediction error and residual fraction are directly comparable.
                    diag = model.residual_diagnostics(s_b, a_b)
                    frac = float(diag["fraction_aggregate"])
                    writer.add_scalar("Residual/fraction", frac, step)
                    writer.add_scalar("Residual/abs", float(diag["abs_mean_residual"].mean()), step)
                    metrics["residual"] = {
                        "fraction_per_kpi": [float(x) for x in diag["fraction_per_kpi"]],
                        "fraction_aggregate": frac,
                        "fraction_max_member": float(diag["fraction_max_member"]),
                        "abs_mean_residual": [float(x) for x in diag["abs_mean_residual"]],
                        "signed_mean_residual": [float(x) for x in diag["signed_mean_residual"]],
                        "alert": bool(diag["alert"]),
                        "alert_fraction": diag["alert_fraction"],
                    }
                    if diag["alert"]:
                        logger.warning(
                            "Residual contribution fraction %.3f exceeds alert threshold "
                            "%.3f (possible MLP collapse; tune the bound/penalty, do NOT "
                            "loosen it to chase utility)",
                            frac,
                            diag["alert_fraction"],
                        )

            elif len(test_buffer) > cfg.train.test_batch_size:
                s_b, a_b, s_1b = test_buffer.sample(cfg.train.test_batch_size)
                s_b, a_b, s_1b = (
                    s_b,
                    a_b.reshape(-1, action_dim),
                    s_1b,
                )
                mse = model.evaluate_predictions(s_b, a_b, s_1b)
                metrics["prediction_mse"] = mse
                logger.info("Next-step prediction MSE: %s", mse)
                writer.add_scalar("Predictions/MSE", mse, step)

    for episode, reward in enumerate(episode_rewards):
        writer.add_scalar("policy_stat/episode_reward", reward, episode)

    if metrics["prediction_mse"] is None and test_buffer:
        s_b, a_b, s_1b = test_buffer.sample(min(len(test_buffer), cfg.train.test_batch_size))
        s_b, a_b, s_1b = (
            s_b,
            a_b.reshape(-1, action_dim),
            s_1b,
        )
        metrics["prediction_mse"] = model.evaluate_predictions(s_b, a_b, s_1b)

    writer.close()

    model.save_model(filepath=checkpoint_path)
    write_metrics(run_dir, metrics)
    logger.info("Training run saved to %s", run_dir)
    return run_dir
