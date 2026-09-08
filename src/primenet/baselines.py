"""Explicit baselines the network must beat or explain."""

from __future__ import annotations

import numpy as np

from .features import SMALL_PRIMES


def residue_rule_predict(n: np.ndarray, primes: np.ndarray = SMALL_PRIMES) -> np.ndarray:
    """The sieve as a rule: composite iff divisible by any of `primes`, else prime.

    Parameterizable so token-sieve comparisons stay honest at every depth:
    model and baseline must see the same prime set. Recall of true primes is
    ~100% for n above max(primes); precision of "prime" predictions is limited
    by numbers with no small factor but a large composite structure.
    """
    n = np.asarray(n, dtype=np.int64)
    primes = np.asarray(primes, dtype=np.int64)
    composite = (n[:, None] % primes[None, :] == 0).any(axis=1)
    return (~composite).astype(np.float32)


def all_composite_predict(n: np.ndarray) -> np.ndarray:
    """Degenerate baseline: never predict prime (the class-imbalance solution)."""
    return np.zeros(len(n), dtype=np.float32)
