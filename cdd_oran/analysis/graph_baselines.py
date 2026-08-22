"""Graph-source baselines for the causal-structure ablation (no retraining).

Swap the binary graph used at inference to isolate the contribution of the
learned causal STRUCTURE while holding the trained CDL predictor fixed. Both
readers of the graph (conflict detection and the predictor mask) go through
``cdl.get_binary_graph()``; overriding that one method ablates the structure.
"""

import numpy as np
import torch

from cdd_oran.conflicts import state_to_tensor


def collect_transitions(env, num, act_dim):
    """Roll out ``num`` (s_t, s_{t+1}) pairs from a fresh env for correlation stats.

    Returns a tensor shaped (num, 2, state_dim) matching the s_batch layout.
    """
    neutral = np.zeros(act_dim)
    pairs = []
    for _ in range(num):
        env.reset()  # random params -> KPI response next step
        s_t = state_to_tensor(env.get_state())
        env.step(neutral)
        s_tp1 = state_to_tensor(env.get_state())
        pairs.append(torch.stack([s_t, s_tp1]))
    return torch.stack(pairs)


def _correlation_graph(causal, transitions, num_params, device):
    """|Pearson corr| of each s_t feature (NCP or KPI) vs each s_{t+1} KPI, top-K
    rank-matched to the FULL causal edge set (NCP->KPI + KPI->KPI).

    Both blocks are scored so the correlation baseline can express KPI->KPI edges
    (e.g. Env III K0->K4, K4->K5) instead of being handicapped to NCP->KPI only.
    """
    st = transitions[:, 0].to(device).float()
    stp1 = transitions[:, 1].to(device).float()
    sd = causal.shape[0]
    src = st  # (N, sd) every feature is a candidate source (NCP + KPI)
    kpi_next = stp1[:, num_params:]  # (N, K) KPI targets
    src_c = src - src.mean(0, keepdim=True)
    kpi_c = kpi_next - kpi_next.mean(0, keepdim=True)
    num = kpi_c.t() @ src_c  # (K, sd)
    den = kpi_c.pow(2).sum(0).sqrt().unsqueeze(1) * src_c.pow(2).sum(0).sqrt().unsqueeze(0)
    corr = (num / (den + 1e-8)).abs()  # (K, sd)

    n_edges = int(causal[num_params:, :].sum().item())  # full causal edge count (both blocks)
    graph = torch.zeros((sd, sd), dtype=torch.bool, device=device)
    n_edges = min(n_edges, corr.numel())
    if n_edges > 0:
        flat = torch.topk(corr.flatten(), n_edges).indices
        graph[num_params + flat // sd, flat % sd] = True
    return graph


def build_override_graph(kind, cdl, env, transitions=None):
    """Binary graph in the layout of ``cdl.get_binary_graph()[:, :-1]`` (state_dim, state_dim).

    - ``causal``: the trained CDL thresholded graph (unchanged).
    - ``full``: dense, every feature a parent (tests whether sparsity matters).
    - ``correlation``: non-causal |Pearson corr| structure, edge count matched to the
      FULL causal edge set (NCP->KPI + KPI->KPI).
    """
    device = cdl.device
    causal = cdl.get_binary_graph()[:, :-1].bool()
    sd = causal.shape[0]
    if kind == "causal":
        return causal.clone()
    if kind == "full":
        return torch.ones((sd, sd), dtype=torch.bool, device=device)
    if kind == "correlation":
        if transitions is None:
            raise ValueError("correlation graph requires env transitions")
        return _correlation_graph(causal, transitions, env.num_params, device)
    raise ValueError(f"Unknown graph override: {kind}")


def _node_index(token, env):
    """State index of an edge endpoint written the way ``envs/env_i.py`` names it.

    ``P<n>`` is a 1-based NCP (``P1`` -> column 0, matching ``prev_params[0]``);
    ``KPI<n>`` / ``K<n>`` is a 1-based KPI (``KPI1`` -> row ``num_params``, matching
    ``kpi1`` / ``prev_kpis[0]``). Returns the index into the ``get_binary_graph()``
    ``[:, :-1]`` block.
    """
    token = token.strip().upper()
    state_dim = env.num_params + env.num_kpis
    if token.startswith("KPI"):
        number, kind = token[3:], "kpi"
    elif token.startswith("K"):
        number, kind = token[1:], "kpi"
    elif token.startswith("P"):
        number, kind = token[1:], "ncp"
    else:
        raise ValueError(f"Cannot parse edge endpoint {token!r}; expected P<n>, KPI<n> or K<n>")
    if not number.isdigit() or int(number) < 1:
        raise ValueError(f"Edge endpoint {token!r} needs a 1-based index (e.g. P2, KPI1)")
    index = int(number) - 1
    node = index if kind == "ncp" else env.num_params + index
    if not 0 <= node < state_dim:
        raise ValueError(f"Edge endpoint {token!r} is out of range for this environment")
    return node


def parse_edge_spec(spec, env):
    """Parse one edge into ``(child_row, parent_col)`` in the ``graph[child, parent]``
    layout that ``get_binary_graph`` and ``detect_conflict_edges`` read.

    Accepts child<-parent (``KPI1<-P2``) and parent->child (``P2->KPI1``); both name
    the same graph cell. This is the same orientation as ``env.true_adj_matrix``,
    where ``[num_params + kpi_index, param_index]`` marks a true edge.
    """
    if "<-" in spec:
        child, parent = spec.split("<-", 1)
    elif "->" in spec:
        parent, child = spec.split("->", 1)
    else:
        raise ValueError(f"Edge {spec!r} needs '<-' or '->' (e.g. 'KPI1<-P2' or 'P2->KPI1')")
    return _node_index(child, env), _node_index(parent, env)


def remove_edges(graph, specs, env):
    """Return a COPY of a binary graph with each named edge removed (set False).

    ``graph`` is a bool tensor in the ``get_binary_graph()`` layout (``(state_dim,
    state_dim)`` or ``(state_dim, state_dim + 1)`` with a trailing action column).
    Only the ``[child, parent]`` cell is cleared; the action column is untouched.
    Returns ``(corrupted_graph, removed)`` where ``removed`` lists
    ``(spec, child_row, parent_col)`` for the handoff/log.
    """
    corrupted = graph.clone()
    removed = []
    for spec in specs:
        child, parent = parse_edge_spec(spec, env)
        corrupted[child, parent] = False
        removed.append((spec.strip(), child, parent))
    return corrupted, removed


def _demo():
    """Self-check: full sets all NCP->KPI edges; correlation rank-matches the FULL
    causal edge set (NCP->KPI + KPI->KPI) and targets KPI rows only."""
    import types

    sd, num_params = 5, 3  # 3 NCPs, 2 KPIs (cols/rows 3,4)
    torch.manual_seed(0)
    graph = torch.zeros((sd, sd + 1), dtype=torch.bool)
    graph[3, 0] = True
    graph[4, 1] = True
    graph[4, 2] = True  # 3 causal NCP->KPI edges
    graph[4, 3] = True  # 1 causal KPI->KPI edge (K0 -> K1)
    cdl = types.SimpleNamespace(
        device=torch.device("cpu"), get_binary_graph=lambda threshold=None: graph
    )
    env = types.SimpleNamespace(num_params=num_params)

    full = build_override_graph("full", cdl, env)
    assert full[num_params:, :num_params].all(), "full must set all NCP->KPI edges"

    causal_edges = int(graph[:, :-1][num_params:, :].sum())  # both blocks
    transitions = torch.randn(64, 2, sd)
    corr = build_override_graph("correlation", cdl, env, transitions)
    assert int(corr[num_params:, :].sum()) == causal_edges, "must rank-match full causal count"
    assert corr[:num_params].sum() == 0, "correlation edges must target KPI rows only"
    print("graph_baselines self-check OK")


if __name__ == "__main__":
    _demo()
