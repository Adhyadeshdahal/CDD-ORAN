from typing import cast

import torch
import torch.nn as nn
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


class DenseResidual(nn.Module):
    """Bounded, zero-initialized dense residual on the raw ``concat(s, a)`` input.

    Phase 2 safety net (see .temp/new_arch/reports/p2_impl_spec.md sec 3): a strongly
    regularized dense path so an omitted true edge stays recoverable, WITHOUT letting the
    model collapse into an MLP. The output is bounded by ``bound * tanh(.)`` so the
    per-KPI residual magnitude can never exceed ``bound``; the final layer starts at zero
    so a fresh model begins exactly at the structure-conditioned graph mean.
    """

    def __init__(self, input_dim, out_dim, hidden, bound):
        super().__init__()
        self.bound = float(bound)
        self.net = MLP(input_dim, out_dim, list(hidden))
        last = cast(nn.Linear, self.net.net[-1])  # final nn.Linear
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, s, a):
        r_raw = self.net(torch.cat([s, a], dim=-1))
        return self.bound * torch.tanh(r_raw)  # (batch, out_dim), |delta| < bound


class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim, pred_hidden):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_feature_extractors = nn.ModuleList(
            [MLP(1, feature_dim, []) for _ in range(state_dim)]
        )
        self.action_feature_extractor = MLP(action_dim, feature_dim, [])
        # Max-pool head, kept for the offline P1 posterior bootstrap: update_mask recomputes
        # CMI through this head (masked_max_from_top_two) on a frozen predictor. It is NOT the
        # prediction path -- predictions go through the structure-conditioned head below.
        self.predictor = MLP(feature_dim, 2, pred_hidden)
        # Structure-conditioned child predictor (the sole prediction head): input is the
        # flattened per-source slot tensor (feature + one presence bit per source). The
        # source axis is NEVER reduced before this MLP, so source j cannot cancel source j'
        # by being pooled together -- per-parent identity is preserved.
        ns = state_dim + 1
        self.structure_predictor = MLP(ns * (feature_dim + 1), 2, pred_hidden)

    def features(self, s, a):
        """
        s: (bs, state_dim), a: (bs, action_dim) -> (bs, state_dim+1, feature_dim)
        """
        feats = []
        for i in range(self.state_dim):
            feats.append(self.state_feature_extractors[i](s[:, i : i + 1]))

        feats.append(self.action_feature_extractor(a))
        return torch.stack(feats, dim=1)

    def head(self, feats, structure=None, *, pooled=False):
        """
        feats: (..., state_dim+1, feature_dim) -> mu (..., 1), std (..., 1)

        ``structure`` (..., state_dim+1) selects the structure-conditioned prediction path:
        each source keeps its own feature slot and a presence bit is concatenated so an
        absent parent is distinguishable from a present parent whose feature happens to be
        zero. The source axis is NOT reduced. ``structure=None`` uses the max-pool head,
        which exists only for the CMI/update_mask bootstrap, not for prediction.
        """
        if structure is not None:
            structure_f = structure.to(feats.dtype).unsqueeze(-1)  # (..., ns, 1)
            structure_f = structure_f.expand(*feats.shape[:-1], 1)
            # GATE each source feature by its presence bit BEFORE flattening: an absent
            # parent contributes an exact zero (not just a 0 bit), so it cannot drive the
            # child's mean/variance through its own feature weights. The separate presence
            # bit is still concatenated so an absent parent stays distinguishable from a
            # present parent whose feature happens to be zero. (review blocker #1)
            gated_feats = feats * structure_f  # zero out absent-source features
            slot_input = torch.cat([gated_feats, structure_f], dim=-1)  # (..., ns, f + 1)
            flat_input = slot_input.flatten(start_dim=-2)  # (..., ns * (f + 1))
            out = self.structure_predictor(flat_input)
            mu = out[..., 0:1]
            log_std = out[..., 1:2]
            std = torch.exp(torch.clamp(log_std, -5, 2)) + 1e-4
            return mu, std

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
        structure=None,
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

        if structure is not None:
            # A mask + structure combination would silently -inf-fill the flattened path;
            # the caller must convert a mask to a structure explicitly (spec sec 2.3).
            if mask is not None:
                raise ValueError("pass either mask (legacy) or structure, not both")
            return self.head(feats, structure)

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
        *,
        residual_bound=0.25,
        residual_l2=1e-2,
        residual_l1=1e-3,
        residual_hidden=(64, 64),
        residual_alert_fraction=0.25,
        sampler=None,
        m_train=1,
        predict_members=1,
        enumeration_graph=None,
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
        # Structure-conditioned dynamics config (the sole world model).
        self.residual_bound = float(residual_bound)
        self.residual_l2 = float(residual_l2)
        self.residual_l1 = float(residual_l1)
        self.residual_hidden = tuple(residual_hidden)
        self.residual_alert_fraction = float(residual_alert_fraction)
        # P1 structure sampler (adapter injected by the model factory). Never a point graph.
        self.sampler = sampler
        self.m_train = int(m_train)
        # Ensemble size returned by predict_next_state when no structures are passed. 1 =>
        # a plain (batch, k) Normal (planner-transparent); >1 => the Phase 3 member axis.
        self.predict_members = int(predict_members)
        # Structure lifecycle (review OPEN finding): structures drawn ONCE per planner.act
        # call are cached here and reused for every candidate + rollout step in that call.
        self._decision_structures = None
        # Self-contained-run artifact manifest (hashes + posterior identity), set by the
        # trainer and persisted in the checkpoint (review v2 MAJOR).
        self.artifact_manifest = None

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        feature_dim = feature_fc_dims[-1]

        self.models = nn.ModuleList(
            [
                StatePredictor(
                    state_dim,
                    action_dim,
                    feature_dim,
                    list(generative_fc_dims),
                )
                for _ in range(state_dim)
            ]
        ).to(self.device)

        # Bounded dense residual (KPI children only) is always part of the world model.
        self.residual = DenseResidual(
            state_dim + action_dim,
            state_dim - kpi_start,
            self.residual_hidden,
            self.residual_bound,
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
        trainable = list(self.models.parameters()) + list(self.residual.parameters())
        self.opt = optim.Adam(trainable, lr=lr, **optimizer_kwargs)

        fd = state_dim
        self.mask_CMI = torch.zeros(fd, fd + 1, device=self.device)
        self._eval_cmi_acc = torch.zeros(fd, fd + 1, device=self.device)
        self._eval_step_count = 0
        # Frozen crisp ENUMERATION graph (review blocker #3). Separate artifact from the
        # sampled PREDICTION structures: conflicts are enumerated from this hard graph while
        # dynamics predict under sampled structures. structure_conditioned runs never learn
        # mask_CMI, so get_binary_graph would otherwise return zeros -> zero conflicts.
        self.enumeration_graph = None
        if enumeration_graph is not None:
            self.enumeration_graph = self._validate_enumeration_graph(enumeration_graph)

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

    def train_step(self, s_batch, a_batch, structures=None):
        """Structure-conditioned dynamics training (the sole training path).

        ``structures`` may be passed explicitly; otherwise the injected P1 sampler draws
        ``m_train`` of them. There is no max-pool training path -- the max-pool head is kept
        only so the offline CMI bootstrap can run on a frozen predictor.
        """
        return self._structure_train_step(s_batch, a_batch, structures)

    # ------------------------------------------------------------------ #
    # Structure-conditioned dynamics + bounded residual.
    # ------------------------------------------------------------------ #
    def _sample_structures(self, m, device):
        """Draw ``m`` structures from the injected P1 sampler. Fails loudly if none is
        supplied -- the world model must never silently fall back to a point graph."""
        if self.sampler is None:
            raise ValueError(
                "structure-conditioned prediction requires a P1 structure sampler; "
                "none was supplied (pass structures= explicitly or inject a sampler)"
            )
        structures = self.sampler.sample_structures(m, device=device)
        return structures.to(device=device)

    def _force_slots(self, structures, child_rows):
        """Force the invariant slots the legacy path always keeps on: each child's own
        source (self slot) and the action source. ``structures`` is ``(m, n_children,
        ns)`` aligned to ``child_rows`` (the state indices of those children)."""
        structures = structures.clone().bool()
        m, n_children, _ = structures.shape
        rows = torch.arange(n_children, device=structures.device)
        structures[:, rows, child_rows] = True  # self slot
        structures[:, :, -1] = True  # action slot
        return structures

    def _normalize_structures(self, structures, child_rows):
        """Normalize a sampled structure to ``(m, len(child_rows), ns)`` and force slots.

        Accepts a full ``(m, fd, ns)`` graph (KPI rows are selected) or an already-sliced
        ``(m, k, ns)`` tensor. This exact layout is shared by train_step, predict, and
        diagnostics (spec risk 3)."""
        structures = torch.as_tensor(structures, device=self.device)
        if structures.dim() != 3 or structures.shape[-1] != self.state_dim + 1:
            raise ValueError(
                f"structures must be (m, n_children, {self.state_dim + 1}); got {tuple(structures.shape)}"
            )
        n_children = child_rows.shape[0]
        if structures.shape[1] == self.state_dim:
            structures = structures[:, child_rows, :]
        elif structures.shape[1] != n_children:
            raise ValueError(
                f"structure child axis {structures.shape[1]} matches neither fd={self.state_dim} "
                f"nor n_children={n_children}"
            )
        return self._force_slots(structures, child_rows)

    def _graph_means(self, s, a, child_rows, structures):
        """Structure-conditioned graph mean/std for the given children, per member.

        ``structures`` is ``(m, n_children, ns)`` (one structure per member, broadcast over
        the batch) OR ``(m, n_children, batch, ns)`` (a per-batch-item structure, used by the
        augmentation). Returns ``mu, std`` of shape ``(m, n_children, batch, 1)``. The source
        axis is never reduced before each child's predictor (per-parent identity)."""
        m = structures.shape[0]
        batch = s.shape[0]
        mus, stds = [], []
        for i in range(child_rows.shape[0]):
            sp = cast(StatePredictor, self.models[int(child_rows[i])])
            feats = sp.features(s, a)  # (batch, ns, f)
            feats_m = feats.unsqueeze(0).expand(m, batch, feats.shape[-2], feats.shape[-1])
            structure_c = structures[:, i]  # (m, ns) or (m, batch, ns)
            if structure_c.dim() == 2:
                structure_c = structure_c.view(m, 1, -1)  # broadcast over batch
            mu_c, std_c = sp.head(feats_m, structure_c)  # (m, batch, 1)
            mus.append(mu_c.unsqueeze(1))
            stds.append(std_c.unsqueeze(1))
        return torch.cat(mus, dim=1), torch.cat(stds, dim=1)  # (m, n_children, batch, 1)

    def sample_decision_structures(self):
        """Draw and CACHE one member set for the current decision (one ``planner.act`` call).

        Every ``predict_next_state`` call with ``structures=None`` afterwards reuses this set
        until ``clear_decision_structures`` -- so all candidate actions and the full rollout
        within a single act call are scored under the IDENTICAL structures, and a fresh set is
        drawn at the next decision."""
        self._decision_structures = self._sample_structures(self.predict_members, self.device)
        return self._decision_structures

    def clear_decision_structures(self):
        self._decision_structures = None

    def _structure_train_step(self, s_batch, a_batch, structures=None):
        s_t = s_batch[:, 0].to(self.device)
        s_tp1 = s_batch[:, 1].to(self.device)
        a_batch = a_batch.to(self.device)
        fd = self.state_dim
        all_rows = torch.arange(fd, device=self.device)

        self.models.train()
        if self.residual is not None:
            self.residual.train()
        self.opt.zero_grad()

        if structures is None:
            structures = self._sample_structures(self.m_train, self.device)
        struct = self._normalize_structures(structures, all_rows)  # (m, fd, ns)
        m = struct.shape[0]

        mu_graph, std_graph = self._graph_means(s_t, a_batch, all_rows, struct)  # (m, fd, batch, 1)

        # Bounded residual on KPI children only; broadcast over structure members.
        delta = self.residual(s_t, a_batch)  # (batch, k)
        delta_rows = delta.transpose(0, 1).unsqueeze(-1)  # (k, batch, 1)
        mu_total = mu_graph.clone()
        mu_total[:, self.kpi_start :, :, :] = mu_graph[:, self.kpi_start :, :, :] + delta_rows.unsqueeze(0)

        targets = s_tp1.transpose(0, 1).unsqueeze(-1).unsqueeze(0)  # (1, fd, batch, 1)
        graph_nll = self._nll(mu_total, std_graph, targets).mean()

        # Single-source dropout augmentation (review MAJOR #5), matching the legacy intent:
        # PER batch item, drop the intervention-informed source (a_batch[:,0]) half the time
        # and a random source otherwise. The pool is STATE sources 0..fd-1 only -- the action
        # slot (fd) is never a target, so a forced-back no-op is impossible. Self slots may be
        # cleared (a meaningful drop). The bit is cleared per item across all children.
        ns = fd + 1
        bs = s_t.shape[0]
        informed = a_batch[:, 0].long().clamp_(0, fd - 1)  # NCP param col = a valid state source
        use_informed = torch.rand(bs, device=self.device) > 0.5
        random_drop = torch.randint(fd, (bs,), device=self.device)  # exclude action slot fd
        drop_idx = torch.where(use_informed, informed, random_drop)  # (bs,)
        struct_aug = struct.unsqueeze(2).expand(m, fd, bs, ns).clone()  # (m, fd, bs, ns)
        struct_aug[:, :, torch.arange(bs, device=self.device), drop_idx] = False
        mu_aug, std_aug = self._graph_means(s_t, a_batch, all_rows, struct_aug)
        mu_aug_total = mu_aug.clone()
        mu_aug_total[:, self.kpi_start :, :, :] = mu_aug[:, self.kpi_start :, :, :] + delta_rows.unsqueeze(0)
        aug_nll = self._nll(mu_aug_total, std_aug, targets).mean()

        scaled = delta / self.residual_bound
        residual_penalty = self.residual_l2 * scaled.pow(2).mean() + self.residual_l1 * scaled.abs().mean()

        loss = graph_nll + aug_nll + residual_penalty
        loss.backward()
        params = list(self.models.parameters()) + list(self.residual.parameters())
        nn.utils.clip_grad_norm_(params, self.grad_clip)
        self.opt.step()
        return loss.detach()

    def residual_diagnostics(self, s, a, structures=None):
        """Anti-collapse measurement (spec sec 3). Returns graph mean, bounded residual,
        total mean and the unsigned per-KPI / aggregate contribution fractions. The tanh
        bound is the safety bound; this fraction is the collapse guard.

        With ``m > 1`` the fraction is computed per member (transitions are NEVER averaged
        first) then reported as both the member-mean and the max-member fraction."""
        s = s.to(self.device)
        a = a.to(self.device)
        kpi_rows = torch.arange(self.kpi_start, self.state_dim, device=self.device)
        with torch.no_grad():
            self.models.eval()
            self.residual.eval()
            if structures is None:
                structures = self._sample_structures(1, self.device)
            struct = self._normalize_structures(structures, kpi_rows)  # (m, k, ns)
            mu_graph, _ = self._graph_means(s, a, kpi_rows, struct)  # (m, k, batch, 1)
            delta = self.residual(s, a)  # (batch, k)
            # mu_total = graph mean + bounded residual (KPI rows), broadcast over members.
            delta_rows = delta.transpose(0, 1).unsqueeze(-1).unsqueeze(0)  # (1, k, batch, 1)
            mu_total = mu_graph + delta_rows  # (m, k, batch, 1)

            abs_delta = delta.abs().mean(dim=0)  # (k,)
            # per-member per-KPI fraction: (m, k)
            abs_mu = mu_graph.abs().mean(dim=(2, 3))  # (m, k)
            frac_mk = abs_delta.unsqueeze(0) / (abs_mu + abs_delta.unsqueeze(0) + 1e-8)
            frac_per_kpi = frac_mk.mean(dim=0)  # (k,)
            frac_aggregate = frac_mk.mean()
            frac_max_member = frac_mk.mean(dim=1).max()  # worst member, aggregated over KPI
        return {
            # Components (review MINOR #8): callers can audit the fractions against these.
            "mu_graph": mu_graph,  # (m, k, batch, 1)
            "delta": delta,  # (batch, k), bounded by +/- residual_bound
            "mu_total": mu_total,  # (m, k, batch, 1) == mu_graph + delta
            "fraction_per_kpi": frac_per_kpi,
            "fraction_aggregate": frac_aggregate,
            "fraction_max_member": frac_max_member,
            "signed_mean_residual": delta.mean(dim=0),
            "abs_mean_residual": abs_delta,
            "residual_bound": torch.tensor(self.residual_bound, device=self.device),
            "alert": bool(frac_aggregate.item() > self.residual_alert_fraction),
            "alert_fraction": self.residual_alert_fraction,
        }

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

    def _validate_enumeration_graph(self, graph):
        """Structural validation for the frozen enumeration graph (review v2 BLOCKER B):
        a (fd, fd+1) bool matrix whose SQUARE state-edge block has a NONZERO edge count.
        (Full env/node-name identity is validated in the model factory against the env.)"""
        tensor = torch.as_tensor(graph, dtype=torch.bool, device=self.device)
        fd = self.state_dim
        if tuple(tensor.shape) != (fd, fd + 1):
            raise ValueError(
                f"enumeration_graph must be (fd, fd+1)=({fd},{fd + 1}); got {tuple(tensor.shape)}"
            )
        if not bool(tensor[:, :fd].any()):
            raise ValueError(
                "enumeration_graph has zero state edges -> would enumerate no conflicts"
            )
        return tensor

    def get_binary_graph(self, threshold=None):
        # Frozen enumeration graph wins when present (structure_conditioned runs): it is the
        # crisp hard graph used for conflict enumeration, independent of the sampled
        # prediction structures. It is already binary, so the threshold does not apply.
        if self.enumeration_graph is not None:
            return self.enumeration_graph.clone()
        if threshold is None:
            threshold = self.cmi_threshold
        graph = self.mask_CMI >= threshold
        fd = graph.shape[0]
        if graph.shape[1] > fd:
            graph[:fd, :fd].fill_diagonal_(0)
        return graph

    def predict_next_state(self, s, a, structures=None):
        return self._structure_predict_next_state(s, a, structures)

    def _structure_predict_next_state(self, s, a, structures=None):
        """Structure-conditioned prediction over KPI children with the bounded residual.

        ``structures`` is ``(m, fd, ns)`` or ``(m, k, ns)``; if omitted the injected P1
        sampler is used for ``m=1`` (never ``get_binary_graph``). Returns a Normal with
        mean/std shaped ``(m, batch, k)``, squeezed to the planner-compatible ``(batch,
        k)`` when ``m == 1``. The multi-member axis is an API for Phase 3; Phase 2 does not
        average transitions across members."""
        s = s.to(self.device)
        a = a.to(self.device)
        kpi_rows = torch.arange(self.kpi_start, self.state_dim, device=self.device)
        with torch.no_grad():
            self.models.eval()
            self.residual.eval()
            if structures is None:
                # Reuse the structures fixed for this decision (planner.act), if any;
                # otherwise draw a fresh set (a bare predict outside a decision scope).
                structures = (
                    self._decision_structures
                    if self._decision_structures is not None
                    else self._sample_structures(self.predict_members, self.device)
                )
            struct = self._normalize_structures(structures, kpi_rows)  # (m, k, ns)
            mu_graph, std_graph = self._graph_means(s, a, kpi_rows, struct)  # (m, k, batch, 1)
            delta = self.residual(s, a)  # (batch, k)
            # (m, k, batch, 1): broadcast the structure-independent residual over members.
            delta_rows = delta.transpose(0, 1).unsqueeze(-1).unsqueeze(0)  # (1, k, batch, 1)
            mu_total = mu_graph + delta_rows
        # -> (m, batch, k)
        mu = mu_total.squeeze(-1).permute(0, 2, 1)
        std = std_graph.squeeze(-1).permute(0, 2, 1)
        if mu.shape[0] == 1:
            mu = mu.squeeze(0)
            std = std.squeeze(0)
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
            # Structure-conditioned residual state + config.
            "residual_state_dict": self.residual.state_dict(),
            "residual_bound": self.residual_bound,
            "residual_l2": self.residual_l2,
            "residual_l1": self.residual_l1,
            "residual_hidden": list(self.residual_hidden),
            "residual_alert_fraction": self.residual_alert_fraction,
            # Frozen enumeration graph travels with the P2 run (review blocker #3).
            "enumeration_graph": self.enumeration_graph,
            # Self-contained-run artifact manifest (hashes + posterior identity).
            "artifact_manifest": self.artifact_manifest,
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
        if state.get("residual_state_dict") is not None:
            self.residual.load_state_dict(state["residual_state_dict"])
        saved_enum = state.get("enumeration_graph")
        if saved_enum is not None:
            restored = self._validate_enumeration_graph(saved_enum)  # shape + nonzero edges
            if self.enumeration_graph is not None and not torch.equal(
                self.enumeration_graph, restored
            ):
                raise ValueError(
                    "checkpoint enumeration graph differs from the one the model was built "
                    "with (factory/env validation vs checkpoint mismatch)"
                )
            self.enumeration_graph = restored
        self.artifact_manifest = state.get("artifact_manifest", self.artifact_manifest)


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
