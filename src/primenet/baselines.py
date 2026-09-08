"""Explicit baselines the network must beat or explain."""

from __future__ import annotations

import numpy as np

from .features import SMALL_PRIMES


def residue_rule_predict(n: np.ndarray) -> np.ndarray:
    """The sieve as a rule: composite iff divisible by any prime < 100, else prime.

    Recall of true primes is ~100% (for n > 97); precision of "prime" predictions
    is limited by numbers with no small factor but a large composite structure.
    """
    n = np.asarray(n, dtype=np.int64)
    composite = (n[:, None] % SMALL_PRIMES[None, :] == 0).any(axis=1)
    return (~composite).astype(np.float32)


def all_composite_predict(n: np.ndarray) -> np.ndarray:
    """Degenerate baseline: never predict prime (the class-imbalance solution)."""
    return np.zeros(len(n), dtype=np.float32)
