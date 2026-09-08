"""Swappable feature encodings for integers.

The literature is unambiguous: the encoding is the single most important design
choice (arXiv 2304.01333). Raw integers fail; binary, residue and Fourier
encodings make modular structure learnable.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# First 25 primes: the explicit sieve uses all of these; the network sees them
# through residues/Fourier features.
SMALL_PRIMES = np.array(
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97],
    dtype=np.int64,
)

BINARY_BITS = 24  # 2^24 = 16.7M > 10^7


def binary_features(n: np.ndarray, bits: int = BINARY_BITS) -> np.ndarray:
    """Fixed-width binary representation, least significant bit first."""
    shifts = np.arange(bits, dtype=np.int64)
    return ((n[:, None] >> shifts) & 1).astype(np.float32)


def residue_features(n: np.ndarray, primes: np.ndarray = SMALL_PRIMES) -> np.ndarray:
    """n mod p normalized by p, for the first 25 primes.

    This is exactly the information an explicit sieve uses, in [0, 1) form.
    """
    p = primes[None, :]
    return (n[:, None] % p).astype(np.float32) / p.astype(np.float32)


def fourier_features(n: np.ndarray, primes: np.ndarray = SMALL_PRIMES) -> np.ndarray:
    """sin/cos(2*pi*n/p) for the first 25 primes (Fourier basis of each residue class).

    Motivated by the Fourier-circuits literature: networks that solve modular
    tasks internally build exactly these features.
    """
    angle = 2.0 * np.pi * n[:, None].astype(np.float64) / primes[None, :].astype(np.float64)
    return np.concatenate([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)


_FEATURE_FNS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "binary": binary_features,
    "residues": residue_features,
    "fourier": fourier_features,
}


def make_feature_fn(spec: str) -> Callable[[np.ndarray], np.ndarray]:
    """Build a combined feature function from a comma-separated spec.

    spec: "binary", "residues", "fourier", "all", or e.g. "binary,fourier".
    The returned function has attributes .names and .dim.
    """
    if spec == "all":
        names = list(_FEATURE_FNS)
    else:
        names = [s.strip() for s in spec.split(",")]
        unknown = [k for k in names if k not in _FEATURE_FNS]
        if unknown:
            raise ValueError(f"unknown features {unknown}; choose from {list(_FEATURE_FNS)} or 'all'")

    fns = [_FEATURE_FNS[k] for k in names]

    def fn(n: np.ndarray) -> np.ndarray:
        parts = [f(n) for f in fns]
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

    fn.names = names  # type: ignore[attr-defined]
    fn.dim = int(fn(np.array([0, 1, 2], dtype=np.int64)).shape[1])  # type: ignore[attr-defined]
    return fn
