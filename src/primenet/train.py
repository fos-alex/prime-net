"""Train a tiny network to classify primes. CPU-only, on-the-fly data, minutes per run.

Example:
    python -m primenet.train --epochs 3
    python -m primenet.train --smoke            # fast plumbing check
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .data import PrimeOracle, sample_batch
from .evaluate import evaluate_checkpoint
from .features import BINARY_BITS, TOKEN_BITS, make_feature_fn, normalize_spec
from .metrics import format_metrics, prf
from .model import build_model
from .predict import predict_range
from .track import append_record, make_record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default="all", help="binary,residues,fourier,'all', or 'tokens'")
    p.add_argument("--model", default="mlp", choices=["mlp", "cnn", "deepset"])
    p.add_argument("--hidden", default="128,128,64")
    p.add_argument("--train-primes-count", type=int, default=25, help="tokens: random primes beyond the always set")
    p.add_argument("--train-primes-max", type=int, default=4093, help="tokens: pool = primes <= this")
    p.add_argument("--train-primes-always", default="2,3,5,7", help="tokens: primes always included")
    p.add_argument("--n-max", type=int, default=10_000_000, help="sieve bound (labels)")
    p.add_argument("--train-min", type=int, default=2)
    p.add_argument("--train-max", type=int, default=700_000)
    p.add_argument("--val-min", type=int, default=700_000)
    p.add_argument("--val-max", type=int, default=750_000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps-per-epoch", type=int, default=1250, help="1250 x 4096 ~ 5.1M samples")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="free-text label for the progress board (e.g. 'residues-only')")
    p.add_argument("--no-auto-eval", action="store_true", help="skip automatic evaluation after training")
    p.add_argument("--eval-in-dist-start", type=int, default=750_000, help="held-out interval inside the training span")
    p.add_argument("--eval-in-dist-end", type=int, default=800_000)
    p.add_argument("--eval-near-ood-start", type=int, default=1_000_000, help="just beyond the training range")
    p.add_argument("--eval-near-ood-end", type=int, default=1_200_000)
    p.add_argument("--eval-far-ood-start", type=int, default=5_000_000, help="far beyond the training range")
    p.add_argument("--eval-far-ood-end", type=int, default=5_200_000)
    p.add_argument("--compile", action="store_true", help="torch.compile (slow warmup, faster steps)")
    p.add_argument("--out", default=None, help="output dir (default runs/<timestamp>)")
    p.add_argument("--smoke", action="store_true", help="tiny config, end-to-end in ~1 min")
    args = p.parse_args()
    args.features = normalize_spec(args.features)

    if args.smoke:
        args.n_max, args.train_max = 1_200_000, 400_000
        args.val_min, args.val_max = 400_000, 450_000
        args.epochs, args.batch_size = 1, 4096
        # tokens need ~300 steps for held-out transfer (prototype scale); explicit
        # CLI values below the cap are respected
        args.steps_per_epoch = min(args.steps_per_epoch, 300 if args.features == "tokens" else 100)

    torch.manual_seed(args.seed)
    device = "cpu"
    out = Path(args.out) if args.out else Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | out {out}")
    oracle = PrimeOracle(args.n_max)
    rng = np.random.default_rng(args.seed)  # reseeded here so the prime draw is reproducible
    if args.features == "tokens":
        pool = oracle.primes_up_to(args.train_primes_max)
        if len(pool) == 0:
            raise SystemExit(f"--train-primes-max {args.train_primes_max} exceeds the sieve")
        always = np.array([int(v) for v in args.train_primes_always.split(",")], dtype=np.int64)
        n_random = max(args.train_primes_count - len(always), 0)
        picked = rng.choice(pool[~np.isin(pool, always)], size=n_random, replace=False)
        train_primes = np.sort(np.concatenate([always, picked]))
        feature_fn = make_feature_fn("tokens", primes=train_primes)
        args.train_primes = [int(p) for p in train_primes]
        args.token_bits = list(TOKEN_BITS)
        print(f"train primes ({len(train_primes)}): {train_primes.tolist()}")
    else:
        feature_fn = make_feature_fn(args.features)
    if feature_fn.primes is not None and args.model != "deepset":
        raise SystemExit("--features tokens requires --model deepset (rank-3 token input)")
    if "binary" in feature_fn.names and args.n_max >= 2**BINARY_BITS:
        raise SystemExit(f"--n-max {args.n_max:,} needs > {BINARY_BITS} binary bits")
    print(f"features {feature_fn.names} dim={feature_fn.dim} | sieve to {args.n_max:,}")
    (out / "config.json").write_text(json.dumps({**vars(args), "host": socket.gethostname()}, indent=2))

    model = build_model(args.model, feature_fn.dim, tuple(int(h) for h in args.hidden.split(",")))
    raw_model = model
    if args.compile:
        model = torch.compile(model)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"model {args.model} params={n_params:,}")

    pos_weight = torch.tensor(float(np.log(args.n_max)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses: list[float] = []
    val_history: list[dict] = []
    total_samples, total_time = 0, 0.0
    for epoch in range(args.epochs):
        model.train()
        t0, seen = time.time(), 0
        bar = tqdm(range(args.steps_per_epoch), desc=f"epoch {epoch}", unit="step")
        for _ in bar:
            _, x, y = sample_batch(rng, oracle, args.train_min, args.train_max, args.batch_size, feature_fn)
            x_t, y_t = torch.from_numpy(x), torch.from_numpy(y)
            opt.zero_grad()
            loss = loss_fn(model(x_t), y_t)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            seen += len(x)
            bar.set_postfix(loss=f"{loss:.4f}", sps=f"{seen / (time.time() - t0):,.0f}/s")

        yv, pv = predict_range(raw_model, oracle, feature_fn, args.val_min, args.val_max)
        vm = prf(yv, pv)
        val_history.append(vm)
        total_samples += seen
        total_time += time.time() - t0
        print(f"[val {args.val_min:,}-{args.val_max:,}] {format_metrics('model', vm)}")

    torch.save(
        {
            "state_dict": raw_model.state_dict(),
            "config": vars(args),
            "feature_names": feature_fn.names,
            "in_dim": feature_fn.dim,
        },
        out / "model.pt",
    )
    (out / "metrics.json").write_text(json.dumps({"val_history": val_history}, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, lw=0.4, alpha=0.4)
    k = 51
    if len(losses) > k:
        smooth = np.convolve(losses, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(k - 1, k - 1 + len(smooth)), smooth, lw=2)
    ax.set(xlabel="step", ylabel="BCE loss", title="Training loss")
    fig.tight_layout()
    fig.savefig(out / "loss_curve.png", dpi=120)

    sps = total_samples / total_time if total_time else None
    append_record(
        Path("runs") / "registry.jsonl",
        make_record(
            out.name, vars(args), n_params, total_samples, sps,
            losses[-1] if losses else None, val_history[-1] if val_history else None, tag=args.tag,
        ),
    )
    print(f"saved checkpoint + metrics + loss_curve.png to {out}")
    print(f"run tracked in runs/registry.jsonl (sps {sps:,.0f})" if sps else "run tracked in runs/registry.jsonl")

    if not args.no_auto_eval:
        print("auto-evaluating...")
        evaluate_checkpoint(  # also refreshes the dashboard
            out / "model.pt",
            (args.eval_in_dist_start, args.eval_in_dist_end),
            (args.eval_near_ood_start, args.eval_near_ood_end),
            (args.eval_far_ood_start, args.eval_far_ood_end),
        )
    else:
        from .dashboard import refresh_default_dashboard

        print(f"dashboard refreshed: {refresh_default_dashboard(Path('runs'))}")


if __name__ == "__main__":
    main()
