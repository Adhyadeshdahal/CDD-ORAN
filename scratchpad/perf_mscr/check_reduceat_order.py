"""Is np.add.reduceat(axis=1) a strict left-to-right sum within each segment? (decides what a GPU/alt
kernel must replicate for bit-identity)."""
import numpy as np

rng = np.random.default_rng(0)
a = rng.normal(size=(64, 700)) * np.exp(rng.normal(size=(64, 700)) * 6)  # wide dynamic range -> order matters
off = np.array([0, 90, 180, 270, 360, 450, 540, 630])
r = np.add.reduceat(a, off, axis=1)
seq = np.zeros_like(r)
for k, (lo, hi) in enumerate(zip(off, list(off[1:]) + [a.shape[1]], strict=True)):
    acc = np.zeros(a.shape[0])
    for c in range(lo, hi):
        acc = acc + a[:, c]
    seq[:, k] = acc
pw = np.stack([a[:, lo:hi].sum(axis=1) for lo, hi in zip(off, list(off[1:]) + [a.shape[1]], strict=True)], 1)
print("reduceat == sequential:", np.array_equal(r, seq), " reduceat == pairwise np.sum:", np.array_equal(r, pw))
# row-chunk invariance of reduceat and of the (B, nb) row sums
r2 = np.concatenate([np.add.reduceat(a[:10], off, axis=1), np.add.reduceat(a[10:], off, axis=1)])
print("reduceat row-chunk invariant:", np.array_equal(r, r2))
s = np.sum(r ** 2 / 3.0, axis=1)
s2 = np.concatenate([np.sum(r[:7] ** 2 / 3.0, axis=1), np.sum(r[7:] ** 2 / 3.0, axis=1)])
print("row-sum chunk invariant:", np.array_equal(s, s2))
# RNG stream chunk invariance
u = np.random.default_rng(5).random((300, 50))
g = np.random.default_rng(5)
u2 = np.concatenate([g.random((128, 50)), g.random((128, 50)), g.random((44, 50))])
print("rng.random row-chunk invariant:", np.array_equal(u, u2))
