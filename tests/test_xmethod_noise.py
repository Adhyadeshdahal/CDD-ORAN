"""R-24 observation noise in every world (cdd_oran/xmethod/worlds/generate.py, kappa) and the runner's required kappa."""
from __future__ import annotations

import json

import numpy as np
import pytest

from cdd_oran.xmethod import runner
from cdd_oran.xmethod.covariates import design_covariates
from cdd_oran.xmethod.worlds import (
    KAPPA_DEFAULT,
    NOISE_SIGMA,
    REGIMES_OF,
    dataset_hash,
    generate_dataset,
)

SEED = 3_000_000
CELLS = [(w, r) for w, rs in REGIMES_OF.items() for r in rs]


@pytest.mark.parametrize("world,regime", CELLS)
def test_default_is_world_default_and_explicit_kappa_is_uniform(world, regime):
    none, _ = generate_dataset(world, regime, 300, SEED, lam=1.5)
    expl, _ = generate_dataset(world, regime, 300, SEED, lam=1.5, kappa=KAPPA_DEFAULT[world])
    assert dataset_hash(none) == dataset_hash(expl)
    zero, t0 = generate_dataset(world, regime, 300, SEED, lam=1.5, kappa=0.0)
    noisy, t1 = generate_dataset(world, regime, 300, SEED, lam=1.5, kappa=0.2)
    assert "obs_noise" not in zero.meta and noisy.meta["obs_noise"] == {"kappa": 0.2, "sigma": list(NOISE_SIGMA[world])}
    assert ("obs_noise" in none.meta) == (KAPPA_DEFAULT[world] > 0)
    # only the KPI series change; actions, designs, context and truth are untouched
    assert np.array_equal(zero.X_action, noisy.X_action) and t0 == t1
    assert all(np.array_equal(a.random_part if a.random_part is not None else 0,
                              b.random_part if b.random_part is not None else 0)
               for a, b in zip(zero.designs, noisy.designs, strict=True))
    assert (zero.context is None) or np.array_equal(zero.context, noisy.context)
    assert not np.array_equal(zero.Y, noisy.Y)
    assert np.array_equal(noisy.Y[:-1], noisy.X_kpi_lag[1:])            # one observed series: lag == target


@pytest.mark.parametrize("world", list(REGIMES_OF))
def test_realised_noise_sd_and_common_random_numbers(world):
    base, _ = generate_dataset(world, "R1", 4000, SEED, kappa=0.0)
    e = {}
    for k in (0.1, 0.5):
        d, _ = generate_dataset(world, "R1", 4000, SEED, kappa=k)
        e[k] = np.concatenate([d.X_kpi_lag, d.Y[-1:]]) - np.concatenate([base.X_kpi_lag, base.Y[-1:]])
        ratio = e[k].std(0) / (k * np.array(NOISE_SIGMA[world]))
        assert np.all(np.abs(ratio - 1) < 0.05), ratio
        assert np.all(np.abs(e[k].mean(0)) < 4 * k * np.array(NOISE_SIGMA[world]) / np.sqrt(4001))
    np.testing.assert_allclose(e[0.5] / 0.5, e[0.1] / 0.1, rtol=1e-6, atol=1e-9 * max(NOISE_SIGMA[world]))


def test_e5_has_one_noise_layer():
    d3, _ = generate_dataset("E5", "R1", 2000, SEED)                     # default kappa .3
    d0, _ = generate_dataset("E5", "R1", 2000, SEED, kappa=0.0)
    assert d3.meta["obs_noise"]["kappa"] == 0.3
    resid = d0.Y[:, 2] - d0.X_action[:, 0]                              # K_mid = P0 exactly when noiseless
    assert np.max(np.abs(resid)) == 0.0
    r3 = d3.Y[:, 2] - d3.X_action[:, 0]
    assert abs(r3.std() / (0.3 * NOISE_SIGMA["E5"][2]) - 1) < 0.05


def _share(ds):
    _, Z, mask = design_covariates(ds)
    X = np.column_stack([np.ones(ds.n), ds.X_action, Z])[mask]
    Y = ds.Y[mask]
    R = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
    return R.var(0) / Y.var(0)


@pytest.mark.parametrize("world,regime", [("E1", "R1"), ("E3", "R1"), ("E4", "R3"), ("E1", "R2"), ("E3", "R2")])
def test_exact_fits_broken_by_noise(world, regime):
    """Noiseless E1 / E3 / E4-R3 are exact linear fits under Z_eq (R-22); with kappa > 0 they are not. In R1 the
    unpredictable share is kappa^2 / (1 + kappa^2) (the R-24 calibration map)."""
    s0 = _share(generate_dataset(world, regime, 3000, SEED, lam=1.5, kappa=0.0)[0])
    assert np.all(s0 < 1e-20)
    for k in (0.1, 0.3):
        s = _share(generate_dataset(world, regime, 3000, SEED, lam=1.5, kappa=k)[0])
        assert np.all(s > 0.005)
        if regime == "R1":
            assert np.all(np.abs(s / (k ** 2 / (1 + k ** 2)) - 1) < 0.2), s


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_invalid_kappa(bad):
    with pytest.raises(ValueError):
        generate_dataset("E1", "R1", 50, SEED, kappa=bad)


# ---------------------------------------------------------------------------------------------- runner
BASE = {"methods": ["dummy"], "worlds": ["E1", "E5"], "regimes": ["R1"], "ns": [100], "seeds": [SEED, SEED + 1]}


def test_runner_requires_kappas():
    with pytest.raises(ValueError, match="kappas"):
        runner.dataset_groups(BASE)
    for bad in ([], [-1], [None], "0.3", [float("nan")]):
        with pytest.raises(ValueError):
            runner.dataset_groups({**BASE, "kappas": bad})


def test_runner_kappa_in_keys_configs_and_records(tmp_path):
    spec = {**BASE, "kappas": [0, 0.3],
            "configs": {"dummy": {"default": {"q": 0.05}, "E1|R1|n100": {"q": 0.1}, "E1|R1|k0.3|n100": {"q": 0.2}}}}
    keys = runner.all_keys(spec)
    assert len(keys) == 2 * 2 * 2 and "dummy|E1|R1|k0|n100|s3000000" in keys and "dummy|E5|R1|k0.3|n100|s3000001" in keys
    assert runner.method_config(spec, "dummy", "E1", "R1", None, 100, 0.0) == {"q": 0.1}
    assert runner.method_config(spec, "dummy", "E1", "R1", None, 100, 0.3) == {"q": 0.2}
    assert runner.cell_key("E1", "R1", None, 100) == "E1|R1|n100"                       # kappa-free key unchanged
    out = str(tmp_path / "r.jsonl")
    assert runner.run_part(spec, 0, 1, out, log=lambda s: None) == 8
    summ = runner.merge(spec, [out], str(tmp_path / "m.jsonl"))
    assert not summ["missing"] and not summ["errors"]
    assert set(summ["per_cell"]) == {f"dummy|{w}|R1|k{k}|n100" for w in ("E1", "E5") for k in ("0", "0.3")}
    recs = {json.loads(x)["key"]: json.loads(x) for x in open(out)}
    r = recs["dummy|E5|R1|k0|n100|s3000000"]
    assert r["job"]["kappa"] == 0.0 and r["config"] == {"q": 0.05}
    ds, _ = generate_dataset("E5", "R1", 100, SEED, kappa=0.0)
    assert r["dataset_sha256"] == dataset_hash(ds)                     # E5 at an explicit kappa 0: noiseless
    assert recs["dummy|E1|R1|k0.3|n100|s3000001"]["job"]["kappa"] == 0.3
