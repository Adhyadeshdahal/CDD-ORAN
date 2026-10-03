"""xmethod adapter ``cmi_knn``: kNN conditional mutual information (tigramite ``CMIknn``, authors' package).

Frenzel-Pompe / KSG-type CMI estimator with tigramite's standard local-permutation (nearest-neighbour restricted)
shuffle null. tigramite defaults (5.2.10.1): knn = 0.2 (fraction of n), shuffle_neighbors = 5,
significance = "shuffle_test", transform = "ranks", permute = "Y", workers = -1; the seed is derived from the
citests tag (tigramite default 42).

F6 deviation (R-9): sig_samples raised from native 500 to sequential 9999 (Besag-Clifford, h = 20) for pooled-BY
resolution. ``SequentialCMIknn.get_shuffle_significance`` is tigramite's restricted-shuffle loop verbatim (same RNG
consumption per surrogate) with a stopping rule; with ``native_config()`` (500, no stopping) it returns exactly
tigramite's p-value. Native rule: tigramite tests one hypothesis at a time (PCMCI's pc_alpha); recorded as raw
p <= .05 in ``declared_native``.

Backends (config ``backend``): "torch" = PRODUCTION default (F2-GPU passed: Kaggle Tesla T4, torch 2.11, 6 worlds,
18/18 tests bit-identical statistic and p, neighbour counts identical at n 500-4000, |Z| up to 41; local RTX 2050
and the audit's 24 checks the same), "cpu" = tigramite's own cKDTree search (``native_config()``, the fidelity
gates). The torch backend is the SAME estimator and null: only ``_get_nearest_neighbors``' search is replaced;
tigramite's tie-breaking noise (same ``random_state`` draws) and rank transform run verbatim on the CPU, so every
coordinate is a distinct integer rank; with integer coordinates, tigramite's "query_ball_point(r = 0.999999999 *
eps)" count is exactly #{j : max-norm distance d_ij < d_(knn+1)(i)} (self included), computed in int32 (exact),
caching the Z / XZ distance matrices while their ranks are unchanged (``TorchKnnCounts``). ``device`` None = cuda
when available, else torch on the CPU (``default_device``); recorded in notes["device"].
"""
from __future__ import annotations

import time
from importlib.metadata import version as _pkg_version
from typing import Any

import numpy as np

from cdd_oran.xmethod.methods._citests_common import (
    B_MAX,
    BC_H,
    CITestBase,
    Prepared,
    SequentialP,
    method_seed,
    progress,
)

TIGRAMITE_VERSION = _pkg_version("tigramite")


def default_device() -> str:
    """Production device of the torch backend: "cuda" when available, else torch on the CPU (also exact)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class TorchKnnCounts:
    """(k_xz, k_yz, k_z) of tigramite's ``_get_nearest_neighbors`` for an integer-valued [T, dim] rank array: per point
    i, eps_i = the (knn+1)-th smallest max-norm distance in the full space (self included); each count is
    #{j : subspace max-norm distance < eps_i}. Exact integer arithmetic, rows in chunks of ``chunk_rows``.

    Cache (exact): in the Y-permutation shuffle null x and z are re-noised and re-ranked on every call; columns
    without ties keep the same ranks, tied columns (e.g. setpoints) do not. The max-norm distance over the STATIC
    x / z columns (ranks identical to the cached ones, checked on every call) is kept on the device as [T, T]
    matrices; only the volatile columns (ranks changed since the previous call) and y are recomputed. A static column
    whose ranks change turns volatile and the cache is rebuilt, so every call equals the uncached computation.
    Used only if the two [T, T] matrices fit in ``cache_bytes`` (int16 when T < 32768)."""

    def __init__(self, device: str = "cuda", chunk_rows: int = 1024, cache_bytes: float = 4e9):
        self.device, self.chunk_rows, self.cache_bytes = device, int(chunk_rows), float(cache_bytes)
        self.sig: tuple | None = None         # (T, x_idx, z_idx) the cache belongs to
        self.prev: np.ndarray | None = None   # x / z ranks of the previous call
        self.volatile: set[int] = set()       # column positions (in the x + z column list) recomputed every call
        self.cached_ranks: np.ndarray | None = None
        self.built_volatile: set[int] | None = None  # the volatile set the cache was built with
        self.m_z = self.m_xz = None           # max over the static z columns; and with x (if x static)
        self.hits = self.misses = 0

    @staticmethod
    def _dist(b, r0, r1):
        d = (b[r0:r1, 0, None] - b[None, :, 0]).abs_()
        for c in range(1, b.shape[1]):
            d = torch_max(d, (b[r0:r1, c, None] - b[None, :, c]).abs_())
        return d

    def _build(self, a, xz_cols, nx, T):
        import torch

        dt = torch.int16 if T < 32768 else torch.int32
        st_z = [c for k, c in enumerate(xz_cols) if k >= nx and k not in self.volatile]
        x_static = all(k not in self.volatile for k in range(nx))
        self.m_z = None if not st_z else torch.empty((T, T), dtype=dt, device=self.device)
        self.m_xz = None if not x_static else torch.empty((T, T), dtype=dt, device=self.device)
        for r0 in range(0, T, self.chunk_rows):
            r1 = min(T, r0 + self.chunk_rows)
            mz = self._dist(a[:, st_z], r0, r1) if st_z else None
            if self.m_z is not None:
                self.m_z[r0:r1] = mz
            if self.m_xz is not None:
                dx = self._dist(a[:, xz_cols[:nx]], r0, r1)
                self.m_xz[r0:r1] = dx if mz is None else torch.maximum(mz, dx)
        self.misses += 1

    def __call__(self, ranks: np.ndarray, x_idx, y_idx, z_idx, knn: int):
        import torch

        T = ranks.shape[0]
        if not np.array_equal(ranks, np.round(ranks)) or ranks.max() >= 2**31 - 1:
            raise ValueError("TorchKnnCounts: needs integer-valued ranks")
        r32 = ranks.astype(np.int32)
        a = torch.as_tensor(r32, device=self.device)
        nx = len(x_idx)
        xz_cols = list(x_idx) + list(z_idx)
        cur = r32[:, xz_cols]
        sig = (T, tuple(x_idx), tuple(z_idx))
        use_cache = len(z_idx) > 0 and 2 * T * T * (2 if T < 32768 else 4) <= self.cache_bytes
        if use_cache:
            if sig != self.sig:
                self.sig, self.prev, self.volatile, self.cached_ranks = sig, None, set(), None
                self.built_volatile = None
            if self.prev is not None:
                self.volatile |= {k for k in range(len(xz_cols)) if not np.array_equal(cur[:, k], self.prev[:, k])}
            static = [k for k in range(len(xz_cols)) if k not in self.volatile]
            if (self.cached_ranks is None or self.built_volatile != self.volatile
                    or not np.array_equal(cur[:, static], self.cached_ranks[:, static])):
                self._build(a, xz_cols, nx, T)
                self.cached_ranks, self.built_volatile = cur.copy(), set(self.volatile)
            else:
                self.hits += 1
            self.prev = cur
            vol_z = [xz_cols[k] for k in sorted(self.volatile) if k >= nx]
            x_vol = any(k in self.volatile for k in range(nx))
        out = [np.empty(T, dtype=np.int64) for _ in range(3)]
        ax, ay = a[:, list(x_idx)], a[:, list(y_idx)]
        az = a[:, list(z_idx)] if len(z_idx) else None
        for r0 in range(0, T, self.chunk_rows):
            r1 = min(T, r0 + self.chunk_rows)
            dy = self._dist(ay, r0, r1)
            if use_cache:
                dz = None if self.m_z is None else self.m_z[r0:r1].to(torch.int32)
                if vol_z:
                    v = self._dist(a[:, vol_z], r0, r1)
                    dz = v if dz is None else torch.maximum(dz, v)
                if x_vol or self.m_xz is None:
                    dxz = torch.maximum(dz, self._dist(ax, r0, r1))
                else:
                    dxz = self.m_xz[r0:r1].to(torch.int32)
                    if vol_z:
                        dxz = torch.maximum(dxz, v)
            else:
                dz = self._dist(az, r0, r1) if az is not None else None
                dxz = self._dist(ax, r0, r1) if dz is None else torch.maximum(dz, self._dist(ax, r0, r1))
            dyz = dy if dz is None else torch.maximum(dz, dy)
            eps = torch.maximum(dxz, dy).kthvalue(knn + 1, dim=1).values[:, None]
            out[0][r0:r1] = (dxz < eps).sum(1).cpu().numpy()
            out[1][r0:r1] = (dyz < eps).sum(1).cpu().numpy()
            out[2][r0:r1] = (dz < eps).sum(1).cpu().numpy() if dz is not None else T
        return out[0], out[1], out[2]


def torch_max(a, b):
    import torch

    return torch.maximum(a, b, out=a)


def _sequential_cls():
    from scipy import spatial
    from tigramite.independence_tests.cmiknn import CMIknn

    class SequentialCMIknn(CMIknn):
        """CMIknn whose nearest-neighbour shuffle test stops by Besag-Clifford (``bc_h``; None = fixed B)."""

        bc_h: int | None = None
        last_draws: int = 0
        backend: str = "cpu"
        device: str | None = None
        chunk_rows: int = 1024
        knn_counts: TorchKnnCounts | None = None
        deadline: float | None = None      # wall-clock stop for cost probes (never set in a real run)
        last_capped: bool = False

        def _get_nearest_neighbors(self, array, xyz, knn):
            if self.backend == "cpu":
                return super()._get_nearest_neighbors(array, xyz, knn)
            if self.transform != "ranks":
                raise ValueError("cmi_knn torch backend: transform='ranks' only")
            # tigramite 5.2.10.1 CMIknn._get_nearest_neighbors, verbatim up to the neighbour search
            array = array.astype(np.float64)
            xyz = xyz.astype(np.int32)
            dim, T = array.shape
            array += (1E-6 * array.std(axis=1).reshape(dim, 1)
                      * self.random_state.random((array.shape[0], array.shape[1])))
            array = array.argsort(axis=1).argsort(axis=1).astype(np.float64)
            array = array.T
            if self.knn_counts is None:
                self.knn_counts = TorchKnnCounts(self.device or default_device(), self.chunk_rows)
            return self.knn_counts(array, np.where(xyz == 0)[0], np.where(xyz == 1)[0], np.where(xyz == 2)[0], knn)

        def get_shuffle_significance(self, array, xyz, value, return_null_dist=False, data_type=None):
            dim, T = array.shape
            x_indices = np.where(xyz == 0)[0]
            y_indices = np.where(xyz == 1)[0]
            z_indices = np.where(xyz == 2)[0]
            if not (len(z_indices) > 0 and self.shuffle_neighbors < T) or self.null_fit is not None \
                    or self.transform not in ("ranks",):
                raise ValueError("SequentialCMIknn: only the restricted-shuffle, transform='ranks', empirical-p path "
                                 "is supported")
            z_array = array[z_indices, :].copy()
            z_array = z_array.argsort(axis=1).argsort(axis=1).astype(np.float64)
            z_array = z_array.T
            tree_xyz = spatial.cKDTree(z_array)
            neighbors = tree_xyz.query(z_array, k=self.shuffle_neighbors, p=np.inf, eps=0.)[1].astype(np.int32)
            seq = SequentialP(value, self.sig_samples, self.bc_h)
            null_dist = []
            self.last_capped = False
            while not seq.done:
                if self.deadline is not None and time.time() > self.deadline:   # cost probes only (p invalid)
                    self.last_capped = True
                    break
                order = self.random_state.permutation(T).astype(np.int32)
                for i in range(len(neighbors)):
                    self.random_state.shuffle(neighbors[i])
                restricted_permutation = self.get_restricted_permutation(
                    T=T, shuffle_neighbors=self.shuffle_neighbors, neighbors=neighbors, order=order)
                array_shuffled = np.copy(array)
                if self.permute == 'X':
                    for i in x_indices:
                        array_shuffled[i] = array[i, restricted_permutation]
                else:
                    for i in y_indices:
                        array_shuffled[i] = array[i, restricted_permutation]
                null_dist.append(self.get_dependence_measure(array_shuffled, xyz))
                seq.add(null_dist[-1])
            self.last_draws = seq.k
            if return_null_dist:
                return seq.p, np.sort(np.asarray(null_dist))
            return seq.p

    return SequentialCMIknn


class CMIKnnMethod(CITestBase):
    name = "cmi_knn"
    version = f"xm-cmiknn-tigramite-{TIGRAMITE_VERSION}-bc"
    method_key = 5

    def default_config(self) -> dict[str, Any]:
        return {"knn": 0.2, "shuffle_neighbors": 5, "sig_samples": B_MAX, "bc_h": BC_H, "transform": "ranks",
                "workers": -1, "backend": "torch", "device": None, "chunk_rows": 1024}

    def native_config(self) -> dict[str, Any]:
        return {**self.default_config(), "sig_samples": 500, "bc_h": None, "arm": "native", "backend": "cpu"}

    def make_test(self, cfg: dict[str, Any], seed: int):
        test = _sequential_cls()(knn=cfg["knn"], shuffle_neighbors=cfg["shuffle_neighbors"],
                                 significance="shuffle_test", transform=cfg["transform"], workers=cfg["workers"],
                                 sig_samples=cfg["sig_samples"], seed=seed)
        test.bc_h = cfg["bc_h"]
        test.backend = cfg.get("backend", "torch")
        test.device = (cfg.get("device") or default_device()) if test.backend == "torch" else None
        test.chunk_rows = int(cfg.get("chunk_rows", 1024))
        return test

    def _test(self, prep: Prepared, data, cfg: dict[str, Any]) -> dict[str, Any]:
        test = self.make_test(cfg, method_seed(data.seed, self.method_key))
        m = len(prep.pairs)
        val, p, k_used = np.full(m, np.nan), np.full(m, np.nan), np.zeros(m, dtype=int)
        for e, (i, j) in enumerate(prep.pairs):
            z = prep.z_of(prep.S, i)
            v, pv = test.run_test_raw(prep.S[:, [i]], prep.Y[:, [j]], z)
            val[e], p[e], k_used[e] = v, pv, test.last_draws
            progress(cfg, e + 1, m)
        return {"score": val, "p": p, "declared_native": p <= 0.05,
                "notes": {"tigramite": TIGRAMITE_VERSION, "mc_draws": k_used.tolist(),
                          "backend": test.backend, "device": test.device,
                          "knn_cache_hits": None if test.knn_counts is None else test.knn_counts.hits,
                          "knn_cache_misses": None if test.knn_counts is None else test.knn_counts.misses,
                          "native_rule": "raw p <= .05 per test (pc_alpha)"}}
