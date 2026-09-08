"""Dissect a trained checkpoint: in-dist + near/far-OOD metrics, baselines, FP anatomy.

Callable directly (CLI) or from train.py (automatic post-training evaluation).
Builds its own sieve covering the evaluation ranges, independent of the
training sieve bound.

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
from .predict import predict_range
from .track import append_record, make_record, update_record

EVAL_KEYS = ("in_dist", "near_ood", "far_ood")
RANGE_LABELS = {"in_dist": "In-distribution", "near_ood": "Near-OOD", "far_ood": "Far-OOD"}


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
        f"- features: `{cfg['features']}`",
        f"- model: `{cfg['model']}` | trained on [{cfg['train_min']:,}, {cfg['train_max']:,}]"
        f" | epochs {cfg['epochs']}",
        "",
    ]
    for key in EVAL_KEYS:
        r = results[key]
        lines += [f"## {RANGE_LABELS[key]} [{r['n'][0]:,}, {r['n'][-1]:,}]", "", "```"]
        lines.append(format_metrics("model", r["model"]))
        lines.append(format_metrics("residue rule", r["residue_rule"]))
        lines.append(format_metrics("all-composite", r["all_composite"]))
        lines += ["```", ""]

    lines += ["## False positives (composite predicted prime)", ""]
    for label, key in (
        ("model in-dist", "in_dist"),
        ("model near-OOD", "near_ood"),
        ("model far-OOD", "far_ood"),
        ("residue rule in-dist", "residue_in_dist"),
    ):
        a = fp[key]
        lines.append(
            f"- **{label}**: {a['count']:,} FPs | median smallest factor "
            f"{int(np.median(a['smallest'])) if a['count'] else 0:,} | "
            f"mean distinct prime factors {float(np.mean(a['omega'])) if a['count'] else 0:.2f}"
        )
    lines += ["", "Plots: `confusion.png`, `fp_anatomy.png`", ""]
    (out / "report.md").write_text("\n".join(lines))


def evaluate_checkpoint(
    checkpoint: Path,
    in_dist: tuple[int, int] = (750_000, 800_000),
    near_ood: tuple[int, int] = (1_000_000, 1_200_000),
    far_ood: tuple[int, int] = (5_000_000, 5_200_000),
) -> dict:
    """Evaluate a checkpoint on the three ranges; write artifacts + registry KPIs.

    Ranges are validated and the label sieve is built to cover them, so eval is
    independent of the training-time --n-max.
    """
    out = checkpoint.parent
    ranges = {"in_dist": in_dist, "near_ood": near_ood, "far_ood": far_ood}
    for name, (lo, hi) in ranges.items():
        if not (2 <= lo < hi):
            raise SystemExit(f"invalid {name} range ({lo}, {hi}): need 2 <= start < end")

    model, cfg = load_model(checkpoint)
    feature_fn = make_feature_fn(cfg["features"])
    eval_n_max = max(hi for _, hi in ranges.values())
    oracle = PrimeOracle(eval_n_max)
    print(f"loaded {checkpoint} (features={feature_fn.names}, model={cfg['model']}) | eval sieve to {eval_n_max:,}")

    results = {key: eval_all(model, oracle, feature_fn, *ranges[key]) for key in EVAL_KEYS}
    for key in EVAL_KEYS:
        r = results[key]
        print(f"\n== {RANGE_LABELS[key]} [{r['n'][0]:,}, {r['n'][-1]:,}] ==")
        print(format_metrics("model", r["model"]))
        print(format_metrics("residue rule", r["residue_rule"]))
        print(format_metrics("all-composite", r["all_composite"]))

    fp = {
        "residue_in_dist": fp_anatomy(
            results["in_dist"]["n"], results["in_dist"]["y"], results["in_dist"]["residue_pred"], oracle
        ),
    }
    for key in EVAL_KEYS:
        r = results[key]
        pred = (r["p"] >= 0.5).astype(np.int8)
        fp[key] = fp_anatomy(r["n"], r["y"], pred, oracle)

    # confusion matrices, model only
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    for ax, key in zip(axes, EVAL_KEYS):
        m = results[key]["model"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center")
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["comp", "prime"], yticklabels=["comp", "prime"],
            xlabel="predicted", ylabel="true", title=RANGE_LABELS[key],
        )
    fig.suptitle("Model confusion matrices")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=120)

    # false-positive smallest-factor anatomy: model vs explicit residue sieve.
    # Bins start at log10(2) so FPs with small factors are NOT silently dropped.
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    bins = np.linspace(np.log10(2), np.log10(eval_n_max) + 0.2, 40)
    for ax, key in zip(axes, EVAL_KEYS):
        a = fp[key]
        if a["count"]:
            ax.hist(np.log10(a["smallest"]), bins=bins, alpha=0.7, label="model FPs")
        if key == "in_dist" and fp["residue_in_dist"]["count"]:
            ax.hist(np.log10(fp["residue_in_dist"]["smallest"]), bins=bins, alpha=0.5, label="residue-rule FPs")
        ax.set(xlabel="log10(smallest prime factor)", ylabel="count", title=RANGE_LABELS[key])
        ax.legend(fontsize=8)
    fig.suptitle("Anatomy of false positives: composites predicted prime")
    fig.tight_layout()
    fig.savefig(out / "fp_anatomy.png", dpi=120)

    metrics_out = {
        key: {b: v2 for b, v2 in v.items() if b not in ("n", "y", "p", "residue_pred")}
        for key, v in results.items()
    }
    fp_summary = {
        k: {
            "count": v["count"],
            "median_smallest": int(np.median(v["smallest"])) if v["count"] else None,
        }
        for k, v in fp.items()
    }
    metrics_out["fp"] = fp_summary
    (out / "eval_metrics.json").write_text(json.dumps(metrics_out, indent=2))
    write_report(out, cfg, results, fp)

    eval_record = {
        "in_dist": results["in_dist"]["model"],
        "near_ood": results["near_ood"]["model"],
        "far_ood": results["far_ood"]["model"],
        "residue_rule": results["in_dist"]["residue_rule"],
        "fp": fp_summary,
    }
    registry = Path("runs") / "registry.jsonl"
    if not update_record(registry, out.name, {"eval": eval_record}):
        stub = make_record(out.name, cfg, sum(p_.numel() for p_ in model.parameters()), None, None, None, None)
        stub["eval"] = eval_record
        append_record(registry, stub)
    print(f"report + plots written to {out}")
    print(f"eval KPIs attached to run {out.name} in {registry}")
    return {"results": results, "fp": fp}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--in-dist-start", type=int, default=750_000)
    p.add_argument("--in-dist-end", type=int, default=800_000)
    p.add_argument("--near-ood-start", type=int, default=1_000_000)
    p.add_argument("--near-ood-end", type=int, default=1_200_000)
    p.add_argument("--far-ood-start", type=int, default=5_000_000)
    p.add_argument("--far-ood-end", type=int, default=5_200_000)
    args = p.parse_args()
    evaluate_checkpoint(
        args.checkpoint,
        (args.in_dist_start, args.in_dist_end),
        (args.near_ood_start, args.near_ood_end),
        (args.far_ood_start, args.far_ood_end),
    )


if __name__ == "__main__":
    main()
