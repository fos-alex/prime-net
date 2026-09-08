"""Vectorized trial-division factorization, used to dissect false positives."""

from __future__ import annotations

import numpy as np


def factor_stats(nums: np.ndarray, primes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor small integers by trial division over `primes`.

    `primes` must contain every prime <= sqrt(max(nums)); anything remaining
    after dividing all of them out is necessarily a single large prime factor.

    Returns:
        smallest: smallest prime factor (n itself if n is prime)
        omega:    number of distinct prime factors
        remaining: n with all primes <= sqrt(max) divided out (1 or one large prime)
    """
    n = np.asarray(nums, dtype=np.int64).copy()
    if len(n) == 0:
        e = np.zeros(0, dtype=np.int64)
        return e.copy(), e.copy(), e.copy()

    smallest = np.ones(len(n), dtype=np.int64)
    omega = np.zeros(len(n), dtype=np.int64)
    limit = int(np.sqrt(int(n.max())))

    for p in primes:
        if p > limit:
            break
        divides = n % p == 0
        if not divides.any():
            continue
        smallest[divides & (smallest == 1)] = p
        omega[divides] += 1
        while True:
            m = n % p == 0
            if not m.any():
                break
            n[m] //= p

    big = n > 1
    omega[big] += 1
    smallest[big & (smallest == 1)] = n[big & (smallest == 1)]
    return smallest, omega, n
