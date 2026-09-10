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

TOKEN_BITS = (12, 0)   # (residue bits, prime bits): residues up to 4093 = sqrt(2^24).
# Prime bits default to 0: with them, the shared detector's confidence drifted with the
# prime's high bits and lost the max-pool to other tokens at far-OOD full depth (five primes
# in 1847..2039 undetected). Residue-only tokens make the detector prime-independent by
# construction. Pass --token-bits 12,12 to reproduce the original variant.
TOKEN_CHUNK = 5000     # predict_range enumeration chunk: rank-3 features are memory-heavy


def token_features(n: np.ndarray, primes: np.ndarray, rb: int = 12, pb: int = 0) -> np.ndarray:
    """One token per prime p: [ bits(n mod p, rb) | bits(p, pb) ], shape [B, T, rb+pb].

    "All residue bits zero" is the same easy conjunction for every prime, so a
    shared detector transfers: adding a prime at inference adds a token, not a
    weight (docs/token-sieve-plan.md). pb=0 (default) omits the prime bits, which
    makes the detector prime-independent by construction.
    """
    n = np.asarray(n, dtype=np.int64)
    primes = np.asarray(primes, dtype=np.int64)
    res = n[:, None] % primes[None, :]                                   # [B, T]
    res_bits = ((res[..., None] >> np.arange(rb)) & 1).astype(np.float32)  # [B, T, rb]
    if pb == 0:
        return res_bits
    p_bits = ((primes[:, None] >> np.arange(pb)) & 1).astype(np.float32)   # [T, pb]
    p_bits = np.broadcast_to(p_bits[None, :, :], (len(n), len(primes), pb)).copy()
    return np.concatenate([res_bits, p_bits], axis=2)


def token_bits_from_config(cfg: dict) -> tuple[int, int]:
    """Token bit widths for a checkpoint. Old checkpoints without the key were 12+12."""
    bits = cfg.get("token_bits")
    if bits is not None:
        return int(bits[0]), int(bits[1])
    return (12, 12) if cfg.get("in_dim") == 24 else TOKEN_BITS


def normalize_spec(spec: str) -> str:
    """Accept the singular typo 'token' as 'tokens' everywhere."""
    return "tokens" if spec == "token" else spec


def make_feature_fn(spec: str, primes: np.ndarray | None = None, bits: tuple[int, int] = TOKEN_BITS):
    """Build a feature function from a spec.

    - spec "tokens" requires primes= and produces rank-3 output [B, T, rb+pb];
      it cannot be combined with other specs in v1.
    - other specs are comma-separated subsets of {binary, residues, fourier}
      or "all"; rank-2 [B, D].

    The returned function carries .names, .dim, and for tokens also .primes
    and .chunk (a predict_range memory hint).
    """
    spec = normalize_spec(spec)
    if spec == "tokens":
        if primes is None:
            raise ValueError("spec 'tokens' requires primes=")
        primes = np.asarray(primes, dtype=np.int64)
        rb, pb = bits

        def fn(n: np.ndarray) -> np.ndarray:
            return token_features(n, primes, rb, pb)

        fn.names = ["tokens"]  # type: ignore[attr-defined]
        fn.dim = rb + pb  # type: ignore[attr-defined]
        fn.primes = primes  # type: ignore[attr-defined]
        fn.chunk = TOKEN_CHUNK  # type: ignore[attr-defined]
        return fn

    if spec == "all":
        names = list(_FEATURE_FNS)
    else:
        names = [s.strip() for s in spec.split(",")]
    if "tokens" in names:
        raise ValueError("'tokens' is rank-3 and cannot be combined with other features in v1")
    unknown = [k for k in names if k not in _FEATURE_FNS]
    if unknown:
        raise ValueError(f"unknown features {unknown}; choose from {list(_FEATURE_FNS)} or 'all'")

    fns = [_FEATURE_FNS[k] for k in names]

    def fn(n: np.ndarray) -> np.ndarray:
        parts = [f(n) for f in fns]
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

    fn.names = names  # type: ignore[attr-defined]
    fn.dim = int(fn(np.array([0, 1, 2], dtype=np.int64)).shape[1])  # type: ignore[attr-defined]
    fn.primes = None  # type: ignore[attr-defined]
    return fn
