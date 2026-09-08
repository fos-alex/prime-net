"""Ground-truth oracle (sieve of Eratosthenes) and on-the-fly batch generation."""

from __future__ import annotations

from typing import Callable

import numpy as np


def build_sieve(n_max: int) -> np.ndarray:
    """Boolean sieve: return[i] is True iff i is prime, for 0 <= i <= n_max."""
    sieve = np.ones(n_max + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n_max**0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = False
    return sieve


class PrimeOracle:
    """Lazy sieve with prime lookups. ~1 byte per integer (10 MB for N=10^7)."""

    def __init__(self, n_max: int):
        self.n_max = n_max
        self.sieve = build_sieve(n_max)
        self.primes = np.flatnonzero(self.sieve)

    def is_prime(self, n: np.ndarray) -> np.ndarray:
        return self.sieve[n]

    def primes_up_to(self, limit: int) -> np.ndarray:
        return self.primes[self.primes <= limit]


def sample_batch(
    rng: np.random.Generator,
    oracle: PrimeOracle,
    n_min: int,
    n_max: int,
    batch_size: int,
    feature_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample n ~ U[n_min, n_max], label via sieve, compute features.

    Returns (n, x, y) with x float32 [B, D] and y float32 [B] (1 = prime).
    """
    n = rng.integers(n_min, n_max + 1, size=batch_size, dtype=np.int64)
    y = oracle.is_prime(n).astype(np.float32)
    x = feature_fn(n)
    return n, x, y
