"""Verification tests (docs/token-sieve-plan.md step 11). Run: pytest tests/"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from primenet.baselines import residue_rule_predict
from primenet.data import PrimeOracle
from primenet.dissect import detection_by_smallest_factor
from primenet.features import SMALL_PRIMES, make_feature_fn, token_features
from primenet.metrics import average_precision, precision_at_recall
from primenet.model import DeepSet
from primenet.nt import factor_stats


def test_sieve_gives_pi_of_1e6():
    assert len(PrimeOracle(10**6).primes) == 78_498


def test_token_bits_roundtrip():
    primes = np.array([2, 5, 97])
    toks = token_features(np.array([13, 100]), primes, rb=12, pb=12)
    assert toks.shape == (2, 3, 24)
    ns = [13, 100]
    for row in range(2):
        for col in range(3):
            t = toks[row, col]
            res = sum(int(t[k]) << k for k in range(12))
            pv = sum(int(t[12 + k]) << k for k in range(12))
            assert res == ns[row] % primes[col]
            assert pv == primes[col]
    # "all residue bits zero" is one uniform conjunction: same slot for every prime
    zero_res = (toks[:, :, :12].sum(axis=2) == 0)
    for row in range(2):
        assert zero_res[row].argmax() == int(np.argmin(ns[row] % primes))


def test_residue_rule_recall_above_max():
    primes = np.array([2, 3, 5, 7])
    assert (residue_rule_predict(np.array([11, 13, 97]), primes) == 1).all()
    # composite with both factors above max(primes): the known precision ceiling
    assert residue_rule_predict(np.array([11 * 13]), primes)[0] == 1
    # with the full default set, primes above 97 are all caught
    assert (residue_rule_predict(np.array([101, 103, 997])) == 1).all()


def test_precision_at_recall_toy():
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    assert precision_at_recall(y, s, 0.5) == pytest.approx(0.75)
    assert precision_at_recall(y, s, 0.999) == pytest.approx(4 / 7)
    assert precision_at_recall(y, s, 1.0) == pytest.approx(4 / 7)
    assert precision_at_recall(y, s, 1.1) is None


def test_average_precision_toy():
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    # .25*1 + .25*(2/3) + .25*(3/4) + .25*(4/7)
    assert average_precision(y, s) == pytest.approx(0.25 * (1 + 2 / 3 + 3 / 4 + 4 / 7))


def test_average_precision_handles_ties():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.7, 0.7, 0.7, 0.7])  # one tie group: threshold covers all
    # recall jumps 0->1 at the group end, precision there = 2/4
    assert average_precision(y, s) == pytest.approx(0.5)


def test_deepset_permutation_invariance():
    torch.manual_seed(0)
    model = DeepSet(24).eval()
    x = torch.randn(3, 7, 24)
    perm = torch.randperm(7)
    with torch.no_grad():
        out = model(x)
        out_permuted = model(x[:, perm])
    assert torch.allclose(out, out_permuted, atol=1e-6)


def test_deepset_token_count_invariance():
    """Adding a prime adds a token, not a weight: max-pool is monotone in tokens,
    so appending any token can only raise or hold each logit pre-rho activation."""
    torch.manual_seed(1)
    model = DeepSet(24).eval()
    x = torch.randn(1, 5, 24)
    extra = torch.zeros(1, 1, 24)  # an all-zero token: residues all zero -> should FIRE
    with torch.no_grad():
        pooled_x = model.phi(x).amax(dim=1)
        pooled_more = model.phi(torch.cat([x, extra], dim=1)).amax(dim=1)
    assert (pooled_more >= pooled_x - 1e-6).all()
    assert model(x).shape == (1,) and model(torch.cat([x, extra], dim=1)).shape == (1,)


def test_detection_by_smallest_factor():
    oracle = PrimeOracle(10**6)
    n = np.array([15, 21, 33, 91, 1001, 143])  # smallest: 3,3,3,7,7,11(above max)
    y = oracle.is_prime(n).astype(int)
    pred = np.array([1, 0, 1, 0, 1, 1])  # 15 missed, 21 caught, 33 missed, 91 caught, 1001 missed, 143 missed
    d = detection_by_smallest_factor(n, y, pred, np.array([3, 7]))
    assert d["counts"] == {3: 3, 7: 2, "above_max": 1}
    assert d["by_prime"][3] == pytest.approx(1 / 3)
    assert d["by_prime"][7] == pytest.approx(0.5)
    assert d["above_max"] == 0.0


def test_factor_stats_matches_sympy():
    pytest.importorskip("sympy")
    from sympy import factorint

    rng = np.random.default_rng(0)
    nums = rng.integers(2, 10**6, size=1000)
    oracle = PrimeOracle(1000)  # primes <= 1000 >= sqrt(1e6)
    smallest, omega, _ = factor_stats(nums, oracle.primes)
    for i, v in enumerate(nums):
        factors = factorint(int(v))
        assert omega[i] == len(factors), f"omega({v})"
        assert smallest[i] == min(factors), f"smallest factor of {v}"


def test_make_feature_fn_contract():
    fn = make_feature_fn("tokens", primes=SMALL_PRIMES[:5])
    assert fn.names == ["tokens"] and fn.dim == 24 and fn.primes is not None
    assert fn.chunk > 0
    with pytest.raises(ValueError):
        make_feature_fn("tokens,residues", primes=SMALL_PRIMES[:5])
    with pytest.raises(ValueError):
        make_feature_fn("tokens")  # primes required
    legacy = make_feature_fn("all")
    assert legacy.names == ["binary", "residues", "fourier"] and legacy.dim == 99
    assert legacy.primes is None
