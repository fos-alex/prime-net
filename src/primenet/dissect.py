"""Dissect model errors by smallest prime factor: the staircase view.

For composites grouped by their smallest prime factor, what fraction did the
model correctly flag? A network that has learned divisibility by p only up to
some depth shows a staircase here; held-out primes measure transfer of the
shared detector.
"""

from __future__ import annotations

import numpy as np

from .nt import factor_stats


def detection_by_smallest_factor(
    n: np.ndarray,
    y: np.ndarray,
    pred_prime: np.ndarray,
    primes: np.ndarray,
) -> dict:
    """Per-prime detection rate for composites whose smallest factor is p.

    Detection = predicted composite. Composites whose smallest factor exceeds
    every prime in `primes` land in the "above_max" bucket (the semiprime
    floor of that depth).

    Returns:
        by_prime: {p: rate | None} for p in primes
        above_max: rate | None
        counts: {p: count, ..., "above_max": count}
    """
    n = np.asarray(n, dtype=np.int64)
    y = np.asarray(y)
    pred = np.asarray(pred_prime)
    primes = np.asarray(primes, dtype=np.int64)

    comp = y == 0
    smallest, _, _ = factor_stats(n[comp], primes)
    detected = pred[comp] == 0

    by_prime: dict[int, float | None] = {}
    counts: dict[int | str, int] = {}
    for p in primes:
        m = smallest == p
        counts[int(p)] = int(m.sum())
        by_prime[int(p)] = float(detected[m].mean()) if m.any() else None
    above = smallest > primes.max()
    counts["above_max"] = int(above.sum())
    above_rate = float(detected[above].mean()) if above.any() else None

    return {"by_prime": by_prime, "above_max": above_rate, "counts": counts}


def min_detection_above(d: dict, min_examples: int = 20, primes=None) -> tuple[float | None, list[int]]:
    """Minimum detection rate over primes with >= min_examples composites.

    With `primes` (the training set), only held-out primes are considered.
    Returns (min_rate, primes_at_that_rate).
    """
    by_prime, counts = d["by_prime"], d["counts"]
    if primes is not None:
        seen = {int(p) for p in np.asarray(primes)}
        by_prime = {p: v for p, v in by_prime.items() if p not in seen}
    rates = [
        (v, p)
        for p, v in by_prime.items()
        if v is not None and counts.get(p, 0) >= min_examples
    ]
    if not rates:
        return None, []
    lo = min(v for v, _ in rates)
    return lo, [p for v, p in rates if v == lo]
