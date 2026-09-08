"""Range inference shared by training validation and evaluation."""

from __future__ import annotations

import numpy as np
import torch


def predict_range(model, oracle, feature_fn, n_min: int, n_max: int, chunk: int | None = None):
    """Enumerate [n_min, n_max] inclusive; return (y_true, p_prime).

    chunk defaults to feature_fn.chunk when present (token features are
    rank-3 and need a smaller enumeration chunk), else 50_000.
    """
    if chunk is None:
        chunk = int(getattr(feature_fn, "chunk", 0) or 50_000)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for lo in range(n_min, n_max + 1, chunk):
            hi = min(lo + chunk - 1, n_max)
            n = np.arange(lo, hi + 1, dtype=np.int64)
            x = torch.from_numpy(feature_fn(n))
            ps.append(torch.sigmoid(model(x)).numpy())
            ys.append(oracle.is_prime(n).astype(np.float32))
    return np.concatenate(ys), np.concatenate(ps)
