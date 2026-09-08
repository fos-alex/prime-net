"""Progress board: comparison table + HTML dashboard from runs/registry.jsonl.

Usage:
    python -m primenet.board --backfill   # one-time: import pre-registry runs
    python -m primenet.board              # table + runs/progress/index.html
    python -m primenet.board --open       # ...and open it in a browser
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dashboard import generate_dashboard
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


def print_table(records: list[dict]) -> None:
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
    unevaluated = [r for r in records if not (r.get("eval") and r["eval"].get("in_dist"))]
    if unevaluated:
        names = ", ".join(r["run_id"] for r in unevaluated)
        print(f"\nnot evaluated yet: {names}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backfill", action="store_true", help="one-time import of existing runs/")
    p.add_argument("--include-smoke", action="store_true")
    p.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = p.parse_args()

    registry = Path("runs") / "registry.jsonl"
    if args.backfill:
        backfill(registry)
    records = load_registry(registry)
    if not records:
        print("registry empty — train a run first, or use --backfill")
        return
    shown = records if args.include_smoke else [r for r in records if not r.get("smoke")]
    print_table(shown)
    out = generate_dashboard(Path("runs"), shown)
    print(f"dashboard written to {out}")
    if args.open:
        import webbrowser

        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
