"""Pin down np.add.reduceat's within-segment summation order (what an alternative kernel must replicate)."""
import numpy as np


def pairwise(a):  # numpy's pairwise_sum (loops_utils.h.src) for a 1-D contiguous run
    n = len(a)
    if n < 8:
        res = 0.0
        for v in a:
            res = res + v
        return res
    if n <= 128:
        r = [a[i] for i in range(8)]
        for i in range(8, n - n % 8, 8):
            for j in range(8):
                r[j] = r[j] + a[i + j]
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        for i in range(n - n % 8, n):
            res = res + a[i]
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return pairwise(a[:n2]) + pairwise(a[n2:])


def main():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(64, 700)) * np.exp(rng.normal(size=(64, 700)) * 6)
    off = [0, 90, 180, 270, 360, 450, 540, 630]
    ends = off[1:] + [700]
    r = np.add.reduceat(a, off, axis=1)
    cands = {
        "first+pairwise(rest)": lambda row, lo, hi: row[lo] + pairwise(row[lo + 1:hi]),
        "pairwise(all)": lambda row, lo, hi: pairwise(row[lo:hi]),
    }
    for name, f in cands.items():
        m = np.array([[f(a[b], lo, hi) for lo, hi in zip(off, ends, strict=True)] for b in range(64)])
        print(name, np.array_equal(m, r), float(np.mean(m == r)))
    seq = np.stack([np.cumsum(a[:, lo:hi], axis=1)[:, -1] for lo, hi in zip(off, ends, strict=True)], 1)
    print("sequential", np.array_equal(seq, r), float(np.mean(seq == r)))


if __name__ == "__main__":
    main()
