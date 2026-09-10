"""Dissect a trained checkpoint: metrics across ranges, baselines, FP anatomy.

Two paths:
- legacy features (binary/residues/fourier/all): single pass per range.
- tokens: sweeps inference-time prime depth (--eval-primes-max, e.g.
  "97,199,499,full"); per depth: model prf, P@R.999, AP, residue rule at the
  same depth, per-prime detection split into seen vs held-out primes.
  Registry comparability: `in_dist` is the FIRST depth; `in_dist_full` and
  `p_at_r999_full` come from the last ("full" by default).

Callable directly (CLI) or from train.py (automatic post-training evaluation).
Builds its own sieve covering the evaluation ranges, independent of the
training sieve bound.

Example:
    python -m primenet.evaluate runs/20260907-101112/model.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .baselines import all_composite_predict, residue_rule_predict
from .data import PrimeOracle
from .dissect import detection_by_smallest_factor
from .features import make_feature_fn, normalize_spec, token_bits_from_config
from .metrics import average_precision, format_metrics, precision_at_recall, prf
from .model import build_model
from .nt import factor_stats
from .predict import predict_range
from .track import append_record, make_record, update_record

EVAL_KEYS = ("in_dist", "near_ood", "far_ood")
RANGE_LABELS = {"in_dist": "In-distribution", "near_ood": "Near-OOD", "far_ood": "Far-OOD"}
TOKEN_DEPTHS_DEFAULT: list[int | str] = [97, 199, 499, "full"]


def load_model(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    hidden = cfg.get("hidden", "128,128,64")
    if isinstance(hidden, str):
        hidden = tuple(int(h) for h in hidden.split(","))
    model = build_model(cfg["model"], ckpt["in_dim"], hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    cfg["features"] = normalize_spec(cfg.get("features", "all"))
    cfg["in_dim"] = int(ckpt["in_dim"])
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


# --------------------------------------------------------------------------
# token-sieve evaluation (depth sweep)
# --------------------------------------------------------------------------


def primes_for_depth(oracle: PrimeOracle, depth: int | str, range_end: int) -> np.ndarray:
    """full -> all primes <= sqrt(range_end); else all primes <= depth."""
    limit = math.isqrt(range_end) if depth == "full" else int(depth)
    primes = oracle.primes_up_to(limit)
    if len(primes) == 0:
        raise SystemExit(f"depth {depth}: no primes <= {limit}")
    return primes


def _split_detection(det: dict, train_primes: set[int]) -> dict:
    by_prime, counts = det["by_prime"], det["counts"]
    seen = {str(p): v for p, v in by_prime.items() if p in train_primes}
    held = {str(p): v for p, v in by_prime.items() if p not in train_primes}
    seen_c = {str(p): c for p, c in counts.items() if isinstance(p, int) and p in train_primes}
    held_c = {str(p): c for p, c in counts.items() if isinstance(p, int) and p not in train_primes}
    return {
        "seen": seen,
        "held_out": held,
        "seen_counts": seen_c,
        "held_out_counts": held_c,
        "above_max": det["above_max"],
        "above_max_count": det["counts"].get("above_max", 0),
    }


def write_report_tokens(out: Path, cfg: dict, by_depth: dict, depths: list, fp_summary: dict) -> None:
    last = str(depths[-1])
    lines = [
        "# prime-net token-sieve evaluation report",
        "",
        f"- features: `tokens` | model: `{cfg['model']}` | trained on "
        f"[{cfg['train_min']:,}, {cfg['train_max']:,}] | epochs {cfg['epochs']}",
        f"- trained on {len(cfg.get('train_primes', []))} primes: {cfg.get('train_primes', [])}",
        f"- inference depths swept: {depths}",
        "",
        "Framing: the token sieve beats the residue rule at any depth above the trained "
        "baseline by construction (it gets more primes). The finding to watch is held-out "
        "prime transfer and the depth/precision ceiling, not the rule comparison.",
        "",
    ]
    for rkey, per_depth in by_depth.items():
        lo, hi = per_depth[last]["range"]
        lines += [f"## {RANGE_LABELS.get(rkey, rkey)} [{lo:,}, {hi:,}]", "", "| depth | primes | P@0.5 | P@R.999 | AP | rule P | rule R | FP@0.5 |",
                  "|---|---|---|---|---|---|---|---|"]
        for d in map(str, depths):
            e = per_depth[d]
            par = e["p_at_r999"]
            lines.append(
                f"| {d} | {e['n_primes']} | {e['model']['precision_prime']:.4f} | "
                f"{'–' if par is None else f'{par:.4f}'} | {e['ap']:.4f} | "
                f"{e['residue_rule']['precision_prime']:.4f} | {e['residue_rule']['recall_prime']:.4f} | "
                f"{e['model']['fp']:,} |"
            )
        lines.append("")

    # held-out-prime table: in-dist at full depth
    det = by_depth["in_dist"][last]["detection"]
    all_primes = sorted(
        int(p) for p in list(det["seen"]) + list(det["held_out"])
    )
    lines += [
        f"## Held-out prime transfer — in-dist, depth {last}",
        "",
        "Detection = fraction of that prime's composites correctly flagged. "
        "Held-out primes were not in the training set.",
        "",
        "| prime | status | composites | detection |",
        "|---|---|---|---|",
    ]
    for p in all_primes:
        key = str(p)
        rate = det["seen"].get(key) if key in det["seen"] else det["held_out"].get(key)
        status = "seen" if key in det["seen"] else "**held-out**"
        cnt = det["seen_counts"].get(key, det["held_out_counts"].get(key, 0))
        lines.append(f"| {p} | {status} | {cnt} | {'–' if rate is None else f'{rate:.4f}'} |")
    above = det["above_max"]
    lines += [
        f"| > max prime | above_max | {det['above_max_count']} | "
        f"{'–' if above is None else f'{above:.4f}'} |",
        "",
        "Plots: `confusion.png`, `fp_anatomy.png`",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))


def _evaluate_tokens(model, cfg: dict, out: Path, ranges: dict, depths: list) -> dict:
    train_primes = {int(p) for p in cfg.get("train_primes", [])}
    token_bits = token_bits_from_config(cfg)
    eval_n_max = max(hi for _, hi in ranges.values())
    oracle = PrimeOracle(eval_n_max)
    print(
        f"loaded {out / 'model.pt'} (tokens, trained on {len(train_primes)} primes) "
        f"| eval sieve to {eval_n_max:,} | depths {depths}"
    )

    by_depth: dict[str, dict] = {key: {} for key in ranges}
    full_pass: dict[str, tuple] = {}
    for rkey, (lo, hi) in ranges.items():
        n = np.arange(lo, hi + 1, dtype=np.int64)
        y = oracle.is_prime(n).astype(np.float32)
        for depth in depths:
            primes = primes_for_depth(oracle, depth, hi)
            ffn = make_feature_fn("tokens", primes=primes, bits=token_bits)
            _, p = predict_range(model, oracle, ffn, lo, hi)
            m = prf(y, p)
            rule = prf(y, residue_rule_predict(n, primes))
            par = precision_at_recall(y, p, 0.999)
            ap = average_precision(y, p)
            pred = (p >= 0.5).astype(np.int8)
            det = _split_detection(detection_by_smallest_factor(n, y, pred, primes), train_primes)
            by_depth[rkey][str(depth)] = {
                "range": [int(lo), int(hi)],
                "n_primes": int(len(primes)),
                "model": m,
                "residue_rule": rule,
                "p_at_r999": par,
                "ap": ap,
                "detection": det,
            }
            par_s = "  –  " if par is None else f"{par:.4f}"
            print(
                f"[{rkey} depth {depth} T={len(primes)}] {format_metrics('model', m)} | "
                f"P@R.999={par_s} AP={ap:.4f} | rule P={rule['precision_prime']:.4f}"
            )
            if depth == depths[-1]:
                full_pass[rkey] = (n, y, pred, primes)

    # verification gate: held-out prime transfer on in-dist at the final depth
    det = by_depth["in_dist"][str(depths[-1])]["detection"]
    rates = [
        (v, int(p))
        for p, v in det["held_out"].items()
        if v is not None and det["held_out_counts"].get(p, 0) >= 20
    ]
    if rates:
        lo_rate = min(v for v, _ in rates)
        worst = [p for v, p in rates if v == lo_rate]
        status = "PASS" if lo_rate >= 0.99 else "FAIL"
        print(
            f"held-out prime transfer (in-dist, depth {depths[-1]}, >=20 examples): "
            f"min={lo_rate:.4f} at {worst} -> {status}"
        )
    else:
        print("held-out prime transfer: no held-out prime has >=20 examples in this range")

    fp = {
        rkey: fp_anatomy(n, y, pred, oracle) for rkey, (n, y, pred, _) in full_pass.items()
    }
    n_in, y_in, _, primes_in = full_pass["in_dist"]
    fp["residue_in_dist"] = fp_anatomy(n_in, y_in, residue_rule_predict(n_in, primes_in), oracle)
    fp_summary = {
        k: {
            "count": v["count"],
            "median_smallest": int(np.median(v["smallest"])) if v["count"] else None,
        }
        for k, v in fp.items()
    }

    first, last = str(depths[0]), str(depths[-1])
    eval_record = {
        "in_dist": by_depth["in_dist"][first]["model"],
        "near_ood": by_depth["near_ood"][first]["model"],
        "far_ood": by_depth["far_ood"][first]["model"],
        "residue_rule": by_depth["in_dist"][first]["residue_rule"],
        "in_dist_full": by_depth["in_dist"][last]["model"],
        "p_at_r999_full": by_depth["in_dist"][last]["p_at_r999"],
        "n_primes_first": by_depth["in_dist"][first]["n_primes"],
        "n_primes_full": by_depth["in_dist"][last]["n_primes"],
        "fp": fp_summary,
    }

    metrics_out = {**eval_record, "by_depth": by_depth}
    (out / "eval_metrics.json").write_text(json.dumps(metrics_out, indent=2))
    write_report_tokens(out, cfg, by_depth, depths, fp_summary)

    # confusion matrices at the last depth
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    for ax, key in zip(axes, EVAL_KEYS):
        m = by_depth[key][last]["model"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center")
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["comp", "prime"], yticklabels=["comp", "prime"],
            xlabel="predicted", ylabel="true", title=f"{RANGE_LABELS[key]} (depth {last})",
        )
    fig.suptitle("Model confusion matrices")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=120)

    registry = Path("runs") / "registry.jsonl"
    if not update_record(registry, out.name, {"eval": eval_record}):
        stub = make_record(out.name, cfg, sum(p_.numel() for p_ in model.parameters()), None, None, None, None)
        stub["eval"] = eval_record
        append_record(registry, stub)
    from .dashboard import refresh_default_dashboard

    print(f"report + plots written to {out}")
    print(f"eval KPIs attached to run {out.name} in {registry}")
    print(f"dashboard refreshed: {refresh_default_dashboard(registry.parent)}")
    return {"results": by_depth, "fp": fp_summary}


def evaluate_checkpoint(
    checkpoint: Path,
    in_dist: tuple[int, int] = (750_000, 800_000),
    near_ood: tuple[int, int] = (1_000_000, 1_200_000),
    far_ood: tuple[int, int] = (5_000_000, 5_200_000),
    token_depths: list | None = None,
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
    if cfg.get("features") == "tokens":
        return _evaluate_tokens(model, cfg, out, ranges, token_depths or TOKEN_DEPTHS_DEFAULT)

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
    from .dashboard import refresh_default_dashboard

    print(f"report + plots written to {out}")
    print(f"eval KPIs attached to run {out.name} in {registry}")
    print(f"dashboard refreshed: {refresh_default_dashboard(registry.parent)}")
    return {"results": results, "fp": fp_summary}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--in-dist-start", type=int, default=750_000)
    p.add_argument("--in-dist-end", type=int, default=800_000)
    p.add_argument("--near-ood-start", type=int, default=1_000_000)
    p.add_argument("--near-ood-end", type=int, default=1_200_000)
    p.add_argument("--far-ood-start", type=int, default=5_000_000)
    p.add_argument("--far-ood-end", type=int, default=5_200_000)
    p.add_argument("--eval-primes-max", default="97,199,499,full",
                   help="token models: comma-separated depths ('full' = primes <= sqrt(range end))")
    args = p.parse_args()
    depths: list[int | str] = ["full" if d.strip() == "full" else int(d) for d in args.eval_primes_max.split(",")]
    evaluate_checkpoint(
        args.checkpoint,
        (args.in_dist_start, args.in_dist_end),
        (args.near_ood_start, args.near_ood_end),
        (args.far_ood_start, args.far_ood_end),
        token_depths=depths,
    )


if __name__ == "__main__":
    main()
