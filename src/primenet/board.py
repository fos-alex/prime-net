"""Progress board: comparison table + charts from runs/registry.jsonl.

Usage:
    python -m primenet.board --backfill   # one-time: import pre-registry runs
    python -m primenet.board              # table + runs/progress/progress.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .track import load_registry, make_record, save_registry


def run_label(r: dict) -> str:
    c = r.get("config", {})
    feats = c.get("features", "?")
    cfg = f"{c.get('model', '?')}/{feats}/e{c.get('epochs', '?')}"
    return r["tag"] or cfg


def backfill(registry: Path) -> None:
    """Import pre-registry runs (config.json/metrics.json/eval_metrics.json) into the registry."""
    records = {r["run_id"]: r for r in load_registry(registry)}
    runs_root = Path("runs")
    if not runs_root.exists():
        print("no runs/ directory")
        return
    added = 0
    for d in sorted(runs_root.iterdir()):
        cfg_file = d / "config.json"
        if not d.is_dir() or d.name == "progress" or not cfg_file.exists() or d.name in records:
            continue
        cfg = json.loads(cfg_file.read_text())
        val = None
        metrics_file = d / "metrics.json"
        if metrics_file.exists():
            hist = json.loads(metrics_file.read_text()).get("val_history") or []
            val = hist[-1] if hist else None
        eval_rec = None
        eval_file = d / "eval_metrics.json"
        if eval_file.exists():
            em = json.loads(eval_file.read_text())
            eval_rec = {
                "in_dist": em.get("in_dist", {}).get("model"),
                "ood": em.get("ood", {}).get("model"),
                "residue_rule": em.get("in_dist", {}).get("residue_rule"),
                "fp": em.get("fp"),
            }
        rec = make_record(d.name, cfg, None, None, None, None, val, tag=cfg.get("tag", ""))
        rec["eval"] = eval_rec
        records[d.name] = rec
        added += 1
    save_registry(registry, sorted(records.values(), key=lambda r: r["run_id"]))
    print(f"backfilled {added} run(s) into {registry}")


def print_table(records: list[dict]) -> list[dict]:
    evaluated = [r for r in records if r.get("eval") and r["eval"].get("in_dist")]
    print(f"\n{len(records)} run(s) tracked, {len(evaluated)} evaluated\n")
    header = (
        f"{'#':>2}  {'run':<15} {'label':<14} {'Msamp':>6} {'P_in':>6} {'dP':>7} "
        f"{'P_ood':>6} {'R_in':>6} {'compR':>6} {'FPmed':>6} {'sps':>7}  host"
    )
    print(header)
    print("-" * len(header))
    prev_p = None
    for i, r in enumerate(evaluated):
        e, m = r["eval"], r["eval"]["in_dist"]
        o = e.get("ood") or {}
        fp = (e.get("fp") or {}).get("in_dist") or {}
        p = m["precision_prime"]
        dp = "-" if prev_p is None else f"{p - prev_p:+.3f}"
        prev_p = p
        msamp = f"{r['samples'] / 1e6:.2f}" if r.get("samples") else "-"
        sps = f"{r['sps'] / 1e3:.0f}k" if r.get("sps") else "-"
        fpmed = f"{fp['median_smallest']}" if fp.get("median_smallest") else "-"
        print(
            f"{i:>2}  {r['run_id']:<15} {run_label(r)[:14]:<14} {msamp:>6} {p:>6.3f} {dp:>7} "
            f"{o.get('precision_prime', float('nan')):>6.3f} {m['recall_prime']:>6.3f} "
            f"{m['recall_composite']:>6.3f} {fpmed:>6} {sps:>7}  {r.get('host', '?')}"
        )
    if not evaluated:
        print("(no evaluated runs yet — run primenet.evaluate on a checkpoint)")
    return evaluated


def make_charts(evaluated: list[dict], out_dir: Path) -> None:
    x = np.arange(len(evaluated))
    labels = [f"{run_label(r)}\n{r['run_id'][-6:]}" for r in evaluated]

    def series(key: str, scope: str) -> np.ndarray:
        vals = [((r["eval"].get(scope) or {}).get(key)) for r in evaluated]
        return np.array([np.nan if v is None else v for v in vals], dtype=float)

    def fp_series(scope: str) -> np.ndarray:
        vals = [((r["eval"].get("fp") or {}).get(scope) or {}).get("median_smallest") for r in evaluated]
        return np.array([np.nan if v is None else v for v in vals], dtype=float)

    pin = series("precision_prime", "in_dist")
    pood = series("precision_prime", "ood")
    rin = series("recall_prime", "in_dist")
    r_ood = series("recall_prime", "ood")
    comp = series("recall_composite", "in_dist")
    comp_ood = series("recall_composite", "ood")
    res_p = series("precision_prime", "residue_rule")
    res_c = series("recall_composite", "residue_rule")
    fp_med = fp_series("in_dist")
    fp_res = fp_series("residue_in_dist")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    ax.plot(x, pin, "o-", label="model in-dist")
    if not np.isnan(pood).all():
        ax.plot(x, pood, "s--", label="model OOD")
    if not np.isnan(res_p).all():
        ax.plot(x, res_p, "k:^", lw=1, label="residue rule (in-dist)")
    for xi, v in zip(x, pin):
        if not np.isnan(v):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_title("Prime precision (higher = fewer false alarms)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x, rin, "o-", label="in-dist")
    if not np.isnan(series("recall_prime", "ood")).all():
        ax.plot(x, series("recall_prime", "ood"), "s--", label="OOD")
    ax.set_title("Prime recall (catching true primes)")
    ax.set_ylim(0.8, 1.01)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, comp, "o-", label="in-dist")
    if not np.isnan(comp_ood).all():
        ax.plot(x, comp_ood, "s--", label="OOD")
    if not np.isnan(res_c).all():
        ax.plot(x, res_c, "k:^", lw=1, label="residue rule (in-dist)")
    ax.set_title("Composite recall (1 - false positive rate)")
    ax.set_ylim(0.5, 1.01)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for arr, style, label in ((fp_med, "o-", "model FPs (in-dist)"), (fp_res, "k:^", "residue rule FPs")):
        mask = ~np.isnan(arr)
        if mask.any():
            ax.plot(x[mask], arr[mask], style, label=label)
    ax.set_yscale("log")
    ax.set_title("FP anatomy: median smallest factor (higher = closer to the semiprime floor)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)

    fig.suptitle("prime-net progress across runs")
    fig.tight_layout()
    out = out_dir / "progress.png"
    fig.savefig(out, dpi=120)
    print(f"\nchart written to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backfill", action="store_true", help="one-time import of existing runs/")
    p.add_argument("--include-smoke", action="store_true")
    args = p.parse_args()

    registry = Path("runs") / "registry.jsonl"
    if args.backfill:
        backfill(registry)
    records = load_registry(registry)
    if not records:
        print("registry empty — train a run first, or use --backfill")
        return
    shown = records if args.include_smoke else [r for r in records if not r.get("smoke")]
    evaluated = print_table(shown)
    if evaluated:
        out_dir = Path("runs") / "progress"
        out_dir.mkdir(parents=True, exist_ok=True)
        make_charts(evaluated, out_dir)


if __name__ == "__main__":
    main()
