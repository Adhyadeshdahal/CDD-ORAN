"""MSCR-v2 permutation bank on torch (CUDA, or any torch device), bit-identical to the numpy bank.

Same method, same RNG stream, same bits: an accelerator for the frozen MSCR-v2, not a new version.
Selected at run time through ``discover_mscr(device=...)`` / ``$MSCR_DEVICE`` / auto-detection
(``mscr.resolve_device``); ``'torch:cpu'`` runs this kernel on the CPU (used for validation).

Why the output is identical rather than "close":

1. Uniforms: the same numpy PCG64 streams, same draws, same order (``mscr._stream_tasks``); generated on
   the host, in parallel by jump-ahead when several threads are available.
2. Permutation: ``torch.sort`` of those uniforms. With distinct keys the ascending order is unique, so
   the index vector equals ``np.argsort``. A row with an exact tie among its uniforms (about 1e-9 per
   row at ns=667, 4e-8 at ns=4000) is re-sorted with ``np.argsort`` on the host, which is how v2 ordered it.
3. Group sums: ``np.add.reduceat`` on a contiguous row computes each segment as
   ``a[lo] + pairwise_sum(a[lo+1:hi])`` (numpy's 8-accumulator pairwise summation, block 128; checked by
   scratchpad/perf_mscr/check_reduceat_order2.py). ``_np_pairwise`` replays that association order with
   elementwise float64 adds, which IEEE-754 rounds identically on every device (a pure add has no FMA).
4. The per-draw tail ``sum_k gs_k^2 / size_k`` and everything after it run in numpy, as in v2.

Targets are batched: all targets share the strata plan (it depends on x only) and the stream positions,
so one device sort covers every target's block. A small bit-identity self-check runs once per device
before auto-selection trusts it (``device_selfcheck``).
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cdd_oran.discovery.mscr import (
    MSCRConfig,
    _finish_stream,
    _jumpable,
    _rowsums_cpu,
    _strata_plan,
    _stream_tasks,
    _uniforms_at,
)

_BYTES_PER_ELEM = 56  # device bytes per (target, row, ns) element: uniforms, sorted keys, index, gather, segments
_SELFCHECK: dict[str, bool] = {}


def _torch_device(device: str):
    import torch

    return torch.device(device.removeprefix("torch:"))


def _np_pairwise(t):
    """numpy's pairwise_sum over the last axis, same association order (numpy loops_utils.h.src)."""
    import torch

    n = t.shape[-1]
    if n < 8:
        res = torch.full(t.shape[:-1], -0.0, dtype=t.dtype, device=t.device)
        for i in range(n):
            res = res + t[..., i]
        return res
    if n <= 128:
        r = t[..., 0:8]
        for i in range(8, n - n % 8, 8):
            r = r + t[..., i:i + 8]
        res = ((r[..., 0] + r[..., 1]) + (r[..., 2] + r[..., 3])) + ((r[..., 4] + r[..., 5]) + (r[..., 6] + r[..., 7]))
        for i in range(n - n % 8, n):
            res = res + t[..., i]
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return _np_pairwise(t[..., :n2]) + _np_pairwise(t[..., n2:])


def _reduceat_last(yp, offsets: np.ndarray):
    """``np.add.reduceat(yp, offsets, axis=-1)`` bit-for-bit; equal-length segments reduce together."""
    import torch

    ns = yp.shape[-1]
    bounds = [int(o) for o in offsets] + [ns]
    out = torch.empty((*yp.shape[:-1], len(offsets)), dtype=yp.dtype, device=yp.device)
    by_len: dict[int, list[int]] = {}
    for k in range(len(offsets)):
        by_len.setdefault(bounds[k + 1] - bounds[k], []).append(k)
    for length, ks in by_len.items():
        idx = torch.as_tensor(np.array([np.arange(bounds[k], bounds[k] + length) for k in ks]), device=yp.device)
        seg = yp[..., idx]  # (..., len(ks), length)
        out[..., ks] = seg[..., 0] + _np_pairwise(seg[..., 1:])
    return out


_MAX_BLOCK_ELEMS = 1 << 23  # all targets together: 64 MiB of host uniforms per block (x3 with prefetch)


def _auto_block_elems(device, n_targets: int) -> int:
    """Per-target elements per device block: at most 1/4 of free device memory and at most
    ``_MAX_BLOCK_ELEMS`` over all targets (host RAM: the uniforms of ~3 blocks are alive at once)."""
    import torch

    budget = _MAX_BLOCK_ELEMS * _BYTES_PER_ELEM
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        budget = min(budget, free // 4)
    return int(max(budget // (_BYTES_PER_ELEM * max(1, n_targets)), 1 << 12))


def _groupsums(u: np.ndarray, y_dev, offsets: np.ndarray, dev) -> np.ndarray:
    """(n_t, rows, nb) permutation group sums of block ``u`` (n_t, rows, ns), bit-identical to v2."""
    import torch

    ut = torch.from_numpy(u).to(dev)
    vals, perm = torch.sort(ut, dim=-1)
    tie = (vals[..., 1:] == vals[..., :-1]).any(dim=-1)
    if bool(tie.any()):
        for j, r in tie.nonzero().cpu().tolist():
            perm[j, r] = torch.from_numpy(np.argsort(u[j, r])).to(dev)
    del vals, ut
    yp = torch.gather(y_dev[:, None, :].expand(-1, u.shape[1], -1), 2, perm)
    del perm
    return _reduceat_last(yp, offsets).cpu().numpy()


def _groupsums_split(u: np.ndarray, y_dev, offsets: np.ndarray, dev) -> np.ndarray:
    """``_groupsums``, retried on row halves when the device runs out of memory (other processes may
    share the GPU). Rows are independent draws, so the split does not change a bit."""
    import torch

    try:
        return _groupsums(u, y_dev, offsets, dev)
    except torch.OutOfMemoryError:
        if u.shape[1] <= 1:
            raise
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        h = u.shape[1] // 2
        return np.concatenate([_groupsums_split(u[:, :h], y_dev, offsets, dev),
                               _groupsums_split(u[:, h:], y_dev, offsets, dev)], axis=1)


def rowsums_torch(ys, plan, cfg: MSCRConfig, rngs, device: str, n_jobs: int, block_elems: int | None):
    """Same contract as ``mscr._rowsums_cpu``: rowsums[j][g][k] = per-draw sum_k gs^2/size for target j."""
    import torch

    dev = _torch_device(device)
    n_t = len(rngs)
    block = block_elems or _auto_block_elems(dev, n_t)
    tasks, total = _stream_tasks(plan, cfg.n_perm, block)
    rowsums = [[[np.empty(cfg.n_perm) for _ in kept] for kept in plan] for _ in rngs]

    def shape(t):
        return (t[3] - t[2], len(plan[t[0]][t[1]][0]))

    jump = n_jobs > 1 and all(_jumpable(r) for r in rngs)
    states = [(r.bit_generator.state, type(r.bit_generator)) for r in rngs] if jump else None
    pool = ThreadPoolExecutor(n_jobs) if jump else None

    def host_uniforms(t):  # (n_t, rows, ns) for task t, every target's own stream segment
        if pool is not None and states is not None:
            return np.stack(list(pool.map(lambda sb: _uniforms_at(sb[0], sb[1], t[4], shape(t)), states)))
        return np.stack([r.random(shape(t)) for r in rngs])

    try:
        ahead: deque = deque()
        feeder = ThreadPoolExecutor(1) if jump else None  # overlap host RNG of the next block with the device
        it = iter(tasks)
        cached_gk, y_dev = None, None
        for _ in range(len(tasks)):
            while feeder is not None and len(ahead) < 2:
                nxt = next(it, None)
                if nxt is None:
                    break
                ahead.append((nxt, feeder.submit(host_uniforms, nxt)))
            if feeder is not None:
                t, fut = ahead.popleft()
                u = fut.result()
            else:
                t = next(it)
                u = host_uniforms(t)
            g, k, lo, hi, _ = t
            _, sizes, offsets = plan[g][k]
            if cached_gk != (g, k):
                y_dev = torch.from_numpy(np.stack([ys[j][g][k] for j in range(n_t)])).to(dev)
                cached_gk = (g, k)
            assert y_dev is not None
            gs = _groupsums_split(u, y_dev, offsets, dev)
            for j in range(n_t):
                rowsums[j][g][k][lo:hi] = np.sum(gs[j] ** 2 / sizes[None, :], axis=1)
        if feeder is not None:
            feeder.shutdown()
    finally:
        if pool is not None:
            pool.shutdown()
    if jump:
        for r, (state, _) in zip(rngs, states or [], strict=True):
            _finish_stream(r, state, total)
    return rowsums


def device_selfcheck(device: str) -> bool:
    """Bit-identity of this kernel vs the numpy kernel on ``device`` (tiny, cached per process).
    Covers the short-segment (8..128) and recursive (>128) pairwise paths and multi-target batching."""
    if device in _SELFCHECK:
        return _SELFCHECK[device]
    ok = True
    try:
        rng = np.random.default_rng(12345)
        for n, cfg in ((520, MSCRConfig(nc=6, nb=8, min_stratum=40, n_perm=40)),
                       (700, MSCRConfig(nc=2, nb=2, min_stratum=10, n_perm=24))):
            x = rng.uniform(size=(n, 3))
            y = rng.normal(size=(n, 2))
            plan = _strata_plan(x, cfg)
            ys = [[[y[rows, j] for rows, _, _ in kept] for kept in plan] for j in range(2)]
            a = _rowsums_cpu(ys, plan, cfg, [np.random.default_rng([7, j]) for j in range(2)], 1, 1 << 12)
            b = rowsums_torch(ys, plan, cfg, [np.random.default_rng([7, j]) for j in range(2)], device, 1, 1 << 12)
            ok &= all(np.array_equal(a[j][g][k], b[j][g][k])
                      for j in range(2) for g in range(len(plan)) for k in range(len(plan[g])))
    except Exception:  # noqa: BLE001 - any failure means "do not auto-select this device"
        ok = False
    _SELFCHECK[device] = ok
    return ok
