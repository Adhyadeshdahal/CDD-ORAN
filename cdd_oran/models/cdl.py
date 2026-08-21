import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.func import functional_call, vmap

from cdd_oran.models.base import CausalModel


def masked_max_from_top_two(features):
    """Return each single-source-ablated max pool without materializing masks."""
    values, indices = torch.topk(features, k=2, dim=2)
    largest = values[:, :, 0, :]
    second = values[:, :, 1, :]
    largest_indices = indices[:, :, 0, :]
    drop_indices = torch.arange(features.shape[2], device=features.device).view(-1, 1, 1, 1)
    return torch.where(
        largest_indices.unsqueeze(0) == drop_indices,
        second.unsqueeze(0),
        largest.unsqueeze(0),
    )


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim, pred_hidden):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_feature_extractors = nn.ModuleList(
            [MLP(1, feature_dim, []) for _ in range(state_dim)]
        )
        self.action_feature_extractor = MLP(action_dim, feature_dim, [])
        self.predictor = MLP(feature_dim, 2, pred_hidden)

    def features(self, s, a):
        """
        s: (bs, state_dim), a: (bs, action_dim) -> (bs, state_dim+1, feature_dim)
        """
        feats = []
        for i in range(self.state_dim):
            feats.append(self.state_feature_extractors[i](s[:, i : i + 1]))

        feats.append(self.action_feature_extractor(a))
        return torch.stack(feats, dim=1)

    def head(self, feats, pooled=False):
        """
        feats: (..., state_dim+1, feature_dim) -> mu (..., 1), std (..., 1)
        """
        h = feats if pooled else feats.max(dim=-2).values

        out = self.predictor(h)
        mu = out[..., 0:1]
        log_std = out[..., 1:2]
        std = torch.exp(torch.clamp(log_std, -5, 2)) + 1e-4

        return mu, std

    def forward(
        self,
        s,
        a,
        mask=None,
        features=None,
        return_features=False,
        pooled_features=None,
    ):
        """
        s:    (bs, state_dim)
        a:    (bs, action_dim)
        mask: (bs, state_dim+1) bool
        returns: mu (bs, 1), std (bs, 1)
        """
        if pooled_features is not None:
            return self.head(pooled_features, pooled=True)

        feats = self.features(s, a) if features is None else features

        if return_features:
            return feats

        if mask is not None:
            feats = feats.masked_fill(mask.unsqueeze(-1), float("-inf"))

        return self.head(feats)


class CDL(CausalModel):
    def __init__(
        self,
        state_dim,
        action_dim,
        kpi_start,
        feature_fc_dims,
        generative_fc_dims,
        lr,
        cmi_threshold,
        eval_tau,
        grad_clip,
        device,
        node_names,
        eval_steps=10,
        interv_weight=1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cmi_threshold = cmi_threshold
        # Interventional-CMI reweight. 1.0 = OFF (no-op, bit-identical to baseline).
        self.interv_weight = interv_weight
        self.eval_tau = eval_tau
        self.eval_steps = eval_steps
        self.grad_clip = grad_clip
        self.kpi_start = kpi_start
        self.node_names = node_names

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        feature_dim = feature_fc_dims[-1]

        self.models = nn.ModuleList(
            [
                StatePredictor(state_dim, action_dim, feature_dim, list(generative_fc_dims))
                for _ in range(state_dim)
            ]
        ).to(self.device)

        prototype = self.models[0]
        parameter_names = tuple(dict(prototype.named_parameters()))
        parameter_maps = [dict(model.named_parameters()) for model in self.models]
        self._parameter_names = parameter_names
        self._parameter_refs = tuple(
            tuple(parameters[name] for parameters in parameter_maps) for name in parameter_names
        )

        def forward_one(parameters, model_features, model_pooled, s, a, mask, return_features):
            return functional_call(
                prototype,
                parameters,
                (s, a, mask, model_features, return_features, model_pooled),
            )

        self._vmap_parameters = vmap(
            forward_one,
            in_dims=(0, None, None, None, None, None, None),
        )
        self._vmap_features = vmap(
            forward_one,
            in_dims=(0, 0, None, None, None, None, None),
        )
        self._vmap_pooled_drop = vmap(
            forward_one,
            in_dims=(0, None, 1, None, None, None, None),
        )

        optimizer_kwargs = {"fused": True} if self.device.type == "cuda" else {}
        self.opt = optim.Adam(self.models.parameters(), lr=lr, **optimizer_kwargs)

        fd = state_dim
        self.mask_CMI = torch.zeros(fd, fd + 1, device=self.device)
        self._eval_cmi_acc = torch.zeros(fd, fd + 1, device=self.device)
        self._eval_step_count = 0

    def _stacked_parameters(self):
        return {
            name: torch.stack(parameters)
            for name, parameters in zip(self._parameter_names, self._parameter_refs, strict=True)
        }

    def _batched_forward(
        self,
        parameters,
        s=None,
        a=None,
        mask=None,
        features=None,
        features_in_dim=None,
        pooled_features=None,
        pooled_in_dim=None,
        return_features=False,
    ):
        if pooled_in_dim is not None:
            forward = self._vmap_pooled_drop
        elif features_in_dim == 0:
            forward = self._vmap_features
        else:
            forward = self._vmap_parameters
        return forward(parameters, features, pooled_features, s, a, mask, return_features)

    def _nll(self, mu, std, target):
        return -Normal(mu, std).log_prob(target)

    def train_step(self, s_batch, a_batch):
        s_t = s_batch[:, 0]
        s_tp1 = s_batch[:, 1]
        bs = s_t.shape[0]
        fd = self.state_dim

        self.opt.zero_grad()
        # a_batch[:, 0] is param_id — the param that was changed
        # bias: drop the changed param more often so model learns to predict without it
        changed_param_ids = a_batch[:, 0].long()  # (bs,) which param changed

        # 50% of the time drop the changed param, 50% drop random
        use_informed = torch.rand(bs, device=self.device) > 0.5
        random_drop = torch.randint(fd + 1, (bs,), device=self.device)
        informed_drop = changed_param_ids  # drop the changed param

        drop_idx = torch.where(use_informed, informed_drop, random_drop)
        mask = F.one_hot(drop_idx, fd + 1).bool()

        self.models.train()
        parameters = self._stacked_parameters()
        feats = self._batched_forward(
            parameters,
            s_t,
            a_batch,
            features=None,
            features_in_dim=None,
            return_features=True,
        )
        mu, std = self._batched_forward(parameters, features=feats, features_in_dim=0)
        targets = s_tp1.transpose(0, 1).unsqueeze(-1)
        full_loss = self._nll(mu, std, targets).mean()

        masked_feats = feats.masked_fill(mask.unsqueeze(0).unsqueeze(-1), float("-inf"))
        mu_m, std_m = self._batched_forward(
            parameters,
            features=masked_feats,
            features_in_dim=0,
        )
        masked_loss = self._nll(mu_m, std_m, targets).mean()
        loss = full_loss + masked_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.models.parameters(), self.grad_clip)
        self.opt.step()

        return loss.detach()

    def update_mask(self, s_batch, a_batch):
        s_t = s_batch[:, 0]
        s_tp1 = s_batch[:, 1]

        with torch.no_grad():
            self.models.eval()
            parameters = self._stacked_parameters()
            feats = self._batched_forward(
                parameters,
                s_t,
                a_batch,
                features=None,
                features_in_dim=None,
                return_features=True,
            )
            mu, std = self._batched_forward(parameters, features=feats, features_in_dim=0)
            targets = s_tp1.transpose(0, 1).unsqueeze(-1)
            full_nll = self._nll(mu, std, targets)

            pooled = masked_max_from_top_two(feats)
            mu_m, std_m = self._batched_forward(
                parameters,
                pooled_features=pooled,
                pooled_in_dim=1,
            )
            masked_nll = self._nll(mu_m, std_m, targets.unsqueeze(1))
            # diff: (fd_child, fd+1_source, bs, 1) per-sample per-source CMI contribution.
            diff = masked_nll - full_nll.unsqueeze(1)
            if self.interv_weight != 1.0:
                # A do() on NCP param_id is strong evidence for its param_id -> KPI
                # edges. Upweight the intervened SOURCE column of each sample before
                # the batch mean. param_id (a_batch[:, 0]) is always an NCP column, so
                # only NCP->KPI edges are affected; KPI->KPI recovery is unchanged.
                changed = a_batch[:, 0].long()  # (bs,) intervened source column
                n_sources = diff.shape[1]  # fd + 1
                columns = torch.arange(n_sources, device=self.device).unsqueeze(1)  # (fd+1, 1)
                weight = torch.where(
                    columns == changed.unsqueeze(0),  # (fd+1, bs)
                    torch.tensor(self.interv_weight, device=self.device, dtype=diff.dtype),
                    torch.tensor(1.0, device=self.device, dtype=diff.dtype),
                )
                diff = diff * weight.view(1, n_sources, changed.shape[0], 1)
            step_cmi = diff.mean(dim=(2, 3))

        self._eval_cmi_acc += step_cmi
        self._eval_step_count += 1

        if self._eval_step_count >= self.eval_steps:
            avg_cmi = self._eval_cmi_acc / self.eval_steps
            self.mask_CMI = self.eval_tau * self.mask_CMI + (1 - self.eval_tau) * avg_cmi
            self._eval_cmi_acc.zero_()
            self._eval_step_count = 0

    def get_causal_graph(self):
        return self.mask_CMI

    def get_binary_graph(self, threshold=None):
        if threshold is None:
            threshold = self.cmi_threshold
        graph = self.mask_CMI >= threshold
        fd = graph.shape[0]
        if graph.shape[1] > fd:
            graph[:fd, :fd].fill_diagonal_(0)
        return graph

    def predict_next_state(self, s, a):
        s = s.to(self.device)
        a = a.to(self.device)
        with torch.no_grad():
            self.models.eval()
            parameters = self._stacked_parameters()
            feats = self._batched_forward(
                parameters,
                s,
                a,
                features=None,
                features_in_dim=None,
                return_features=True,
            )
            targets = torch.arange(self.kpi_start, self.state_dim, device=self.device)
            parameters = {name: value[targets] for name, value in parameters.items()}
            feats = feats[targets]
            graph_mask = self.get_binary_graph()[targets].clone()
            graph_mask[torch.arange(len(targets), device=self.device), targets] = True
            graph_mask[:, -1] = True
            masked_feats = feats.masked_fill(~graph_mask.unsqueeze(1).unsqueeze(-1), float("-inf"))
            mu, std = self._batched_forward(
                parameters,
                features=masked_feats,
                features_in_dim=0,
            )
        mu = mu.squeeze(-1).transpose(0, 1)
        std = std.squeeze(-1).transpose(0, 1)
        return Normal(mu, std)

    def evaluate_predictions(self, s, a, s_1):
        """
        s:   (bs, state_dim)
        a:   (bs, 1)
        s_1: (bs, state_dim)
        returns: scalar MSE
        """
        dist = self.predict_next_state(s, a)
        pred = dist.sample()  # (bs, state_dim - kpi_start)
        target = s_1[:, self.kpi_start :]  # (bs, state_dim - kpi_start)
        return ((pred - target) ** 2).mean().item()  # scalar

    def save_model(self, filepath):
        """
        Save the model state, optimizer state, and other necessary attributes.
        """
        state = {
            "models_state_dict": [model.state_dict() for model in self.models],
            "optimizer_state_dict": self.opt.state_dict(),
            "mask_CMI": self.mask_CMI,
            "eval_cmi_acc": self._eval_cmi_acc,
            "eval_step_count": self._eval_step_count,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "cmi_threshold": self.cmi_threshold,
            "eval_tau": self.eval_tau,
            "eval_steps": self.eval_steps,
            "grad_clip": self.grad_clip,
            "device": self.device,
        }
        torch.save(state, filepath)

    def load_model(self, filepath):
        """
        Load the model state, optimizer state, and other attributes.
        """
        state = torch.load(filepath, map_location=self.device)
        for i, model in enumerate(self.models):
            model.load_state_dict(state["models_state_dict"][i])
        self.opt.load_state_dict(state["optimizer_state_dict"])
        if self.device.type == "cuda":
            for group in self.opt.param_groups:
                group["fused"] = True
                group["foreach"] = None
        self.mask_CMI = state["mask_CMI"]
        self._eval_cmi_acc = state["eval_cmi_acc"]
        self._eval_step_count = state["eval_step_count"]


def _self_check():
    """Verify the interventional-CMI reweight in ``update_mask``:

    - ``interv_weight=1.0`` is a bit-identical no-op (two builds agree exactly).
    - ``interv_weight=w`` scales ONLY the intervened source column and leaves
      every other column bit-identical. With a constant intervened column c0,
      the accumulated CMI on c0 is exactly ``w x`` the baseline (every sample of
      that column is weighted), proving the reweight raises the intervened edges.
    """
    state_dim, action_dim, kpi_start = 5, 3, 3  # sources 0..2 = NCP, 3..4 = KPI
    c0 = 1  # constant intervened NCP source column (< kpi_start)

    def build(w):
        torch.manual_seed(0)  # identical predictor init across builds
        return CDL(
            state_dim=state_dim,
            action_dim=action_dim,
            kpi_start=kpi_start,
            feature_fc_dims=[8],
            generative_fc_dims=[8],
            lr=1e-3,
            cmi_threshold=0.1,
            eval_tau=0.99,
            grad_clip=10.0,
            device="cpu",
            node_names=None,
            eval_steps=10,
            interv_weight=w,
        )

    torch.manual_seed(1)
    bs = 16
    s_batch = torch.randn(bs, 2, state_dim)
    a_batch = torch.randint(0, state_dim, (bs, action_dim)).float()
    a_batch[:, 0] = float(c0)  # every sample intervenes on column c0

    base = build(1.0)
    base.update_mask(s_batch, a_batch)
    base_acc = base._eval_cmi_acc.clone()

    identical = build(1.0)
    identical.update_mask(s_batch, a_batch)
    assert torch.equal(identical._eval_cmi_acc, base_acc), "weight=1.0 is not a no-op"

    w = 5.0
    up = build(w)
    up.update_mask(s_batch, a_batch)
    up_acc = up._eval_cmi_acc

    other = [c for c in range(state_dim + 1) if c != c0]
    assert torch.equal(up_acc[:, other], base_acc[:, other]), (
        "non-intervened source columns changed under reweight"
    )
    assert torch.allclose(up_acc[:, c0], w * base_acc[:, c0]), (
        "intervened column not scaled by interv_weight"
    )
    assert not torch.equal(up_acc[:, c0], base_acc[:, c0]) or torch.all(base_acc[:, c0] == 0), (
        "intervened column unchanged despite reweight"
    )

    print(
        "cdl interventional-CMI self-check passed: "
        f"weight=1.0 bit-identical; weight={w} scales only column {c0} "
        f"(x{w}), other columns unchanged"
    )


if __name__ == "__main__":
    _self_check()
