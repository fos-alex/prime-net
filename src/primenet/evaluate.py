"""Dissect a trained checkpoint: in-dist + OOD metrics, baselines, false-positive anatomy.

Example:
    python -m primenet.evaluate runs/20260907-101112/model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .baselines import all_composite_predict, residue_rule_predict
from .data import PrimeOracle
from .features import make_feature_fn
from .metrics import format_metrics, prf
from .model import build_model
from .nt import factor_stats
from .train import predict_range


def load_model(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    hidden = cfg.get("hidden", "128,128,64")
    if isinstance(hidden, str):
        hidden = tuple(int(h) for h in hidden.split(","))
    model = build_model(cfg["model"], ckpt["in_dim"], hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def eval_all(model, oracle, feature_fn, lo, hi) -> dict:
    n = np.arange(lo, hi + 1, dtype=np.int64)
    _, p = predict_range(model, oracle, feature_fn, lo, hi)
    y = oracle.is_prime(n).astype(np.float32)
    residue_pred = residue_rule_predict(n)
    return {
        "n": n,
        "y": y,
        "p": p,
        "residue_pred": residue_pred,
        "model": prf(y, p),
        "residue_rule": prf(y, residue_pred),
        "all_composite": prf(y, all_composite_predict(n)),
    }


def fp_anatomy(n, y, pred_prime, oracle) -> dict:
    fp = n[(pred_prime == 1) & (y == 0)]
    smallest, omega, _ = factor_stats(fp, oracle.primes)
    return {"count": int(len(fp)), "smallest": smallest, "omega": omega}


def write_report(out: Path, cfg: dict, results: dict, fp: dict) -> None:
    lines = [
        "# prime-net evaluation report",
        "",
        f"- features: `{','.join(cfg['features']) if isinstance(cfg['features'], list) else cfg['features']}`",
        f"- model: `{cfg['model']}` | trained on [{cfg['train_min']:,}, {cfg['train_max']:,}]"
        f" | epochs {cfg['epochs']}",
        "",
    ]
    for label, key in (("In-distribution", "in_dist"), ("Out-of-distribution", "ood")):
        r = results[key]
        lines += [f"## {label} [{r['n'][0]:,}, {r['n'][-1]:,}]", "", "```"]
        lines.append(format_metrics("model", r["model"]))
        lines.append(format_metrics("residue rule", r["residue_rule"]))
        lines.append(format_metrics("all-composite", r["all_composite"]))
        lines += ["```", ""]

    lines += ["## False positives (composite predicted prime)", ""]
    for label, key in (("model in-dist", "in_dist"), ("model OOD", "ood"), ("residue rule in-dist", "residue_in_dist")):
        a = fp[key]
        lines.append(
            f"- **{label}**: {a['count']:,} FPs | median smallest factor "
            f"{int(np.median(a['smallest'])) if a['count'] else 0:,} | "
            f"mean distinct prime factors {float(np.mean(a['omega'])) if a['count'] else 0:.2f}"
        )
    lines += ["", "Plots: `confusion.png`, `fp_anatomy.png`", ""]
    (out / "report.md").write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--in-dist-start", type=int, default=1_000_000)
    p.add_argument("--in-dist-end", type=int, default=1_200_000)
    p.add_argument("--ood-start", type=int, default=5_000_000)
    p.add_argument("--ood-end", type=int, default=5_200_000)
    args = p.parse_args()

    out = args.checkpoint.parent
    model, cfg = load_model(args.checkpoint)
    oracle = PrimeOracle(cfg["n_max"])
    feature_fn = make_feature_fn(cfg["features"])
    print(f"loaded {args.checkpoint} (features={feature_fn.names}, model={cfg['model']})")

    results = {
        "in_dist": eval_all(model, oracle, feature_fn, args.in_dist_start, args.in_dist_end),
        "ood": eval_all(model, oracle, feature_fn, args.ood_start, args.ood_end),
    }
    for label, r in results.items():
        print(f"\n== {label} [{r['n'][0]:,}, {r['n'][-1]:,}] ==")
        print(format_metrics("model", r["model"]))
        print(format_metrics("residue rule", r["residue_rule"]))
        print(format_metrics("all-composite", r["all_composite"]))

    pred_prime = (results["in_dist"]["p"] >= 0.5).astype(np.int8)
    pred_prime_ood = (results["ood"]["p"] >= 0.5).astype(np.int8)
    fp = {
        "in_dist": fp_anatomy(results["in_dist"]["n"], results["in_dist"]["y"], pred_prime, oracle),
        "ood": fp_anatomy(results["ood"]["n"], results["ood"]["y"], pred_prime_ood, oracle),
        "residue_in_dist": fp_anatomy(
            results["in_dist"]["n"], results["in_dist"]["y"], results["in_dist"]["residue_pred"], oracle
        ),
    }

    # confusion matrices, model only
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (label, r) in zip(axes, results.items()):
        m = r["model"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center")
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["comp", "prime"], yticklabels=["comp", "prime"],
            xlabel="predicted", ylabel="true", title=f"{'in-dist' if label == 'in_dist' else 'OOD'}",
        )
    fig.suptitle("Model confusion matrices")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=120)

    # false-positive smallest-factor anatomy: model vs explicit residue sieve
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(2, np.log10(cfg["n_max"]) + 0.2, 30)
    for ax, (label, key) in zip(axes, (("in-dist", "in_dist"), ("OOD", "ood"))):
        a, b = fp[key], fp["residue_in_dist"] if key == "in_dist" else None
        if a["count"]:
            ax.hist(np.log10(a["smallest"]), bins=bins, alpha=0.6, label="model FPs")
        if b is not None and b["count"]:
            ax.hist(np.log10(b["smallest"]), bins=bins, alpha=0.6, label="residue-rule FPs")
        ax.set(xlabel="log10(smallest prime factor)", ylabel="count", title=label)
        ax.legend()
    fig.suptitle("Anatomy of false positives: composites predicted prime")
    fig.tight_layout()
    fig.savefig(out / "fp_anatomy.png", dpi=120)

    metrics_out = {
        k: {b: v2 for b, v2 in v.items() if b not in ("n", "y", "p", "residue_pred")}
        for k, v in results.items()
    }
    (out / "eval_metrics.json").write_text(json.dumps(metrics_out, indent=2))
    write_report(out, cfg, results, fp)
    print(f"\nreport + plots written to {out}")


if __name__ == "__main__":
    main()
