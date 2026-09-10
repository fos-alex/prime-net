"""Aggregate token-sieve seed runs by tag prefix: mean and range per range x depth.

Usage: python scripts/summarize_seeds.py tier1-residue-only
"""
import json, sys, glob
from pathlib import Path
prefix = sys.argv[1] if len(sys.argv) > 1 else "tier1-residue-only"
runs = []
for cfgp in sorted(glob.glob("runs/*/config.json")):
    cfg = json.load(open(cfgp))
    if not str(cfg.get("tag", "")).startswith(prefix): continue
    ep = Path(cfgp).parent / "eval_metrics.json"
    if not ep.exists(): continue
    runs.append((Path(cfgp).parent.name, cfg, json.load(open(ep))))
print(f"{len(runs)} runs with tag prefix '{prefix}': {[r[0] for r in runs]}")
if not runs: sys.exit(0)
ranges = ["in_dist", "near_ood", "far_ood"]; depths = ["97", "199", "499", "full"]
def stat(vals): return f"{sum(vals)/len(vals):.4f} [{min(vals):.4f}, {max(vals):.4f}]"
print("\n| range | depth | P@0.5 mean [min, max] | P@R.999 mean [min, max] | FP@0.5 per seed | rule P |")
print("|---|---|---|---|---|---|")
for r in ranges:
    for d in depths:
        e = [run[2]["by_depth"][r][d] for run in runs]
        p05 = [x["model"]["precision_prime"] for x in e]; pr = [x["p_at_r999"] for x in e]
        fps = [x["model"]["fp"] for x in e]; rule = e[0]["residue_rule"]["precision_prime"]
        print(f"| {r} | {d} | {stat(p05)} | {stat(pr)} | {fps} | {rule:.4f} |")
print("\nheld-out transfer (in-dist, full depth): min detection over held-out primes with >= 20 examples, per seed:")
for name, cfg, e in runs:
    det = e["by_depth"]["in_dist"]["full"]["detection"]
    held = det["held_out"]; cnt = det.get("held_out_counts", {})
    vals = [v for p, v in held.items() if cnt.get(p, 0) >= 20]
    print(f"  {name}: min={min(vals):.4f} over {len(vals)} primes")
print("\nrecall (in-dist, full depth) per seed:", [f"{e['by_depth']['in_dist']['full']['model']['recall_prime']:.4f}" for _,_,e in runs])
print("val_history (full depth) per seed:")
for name, cfg, _ in runs:
    vh = json.load(open(f"runs/{name}/metrics.json"))
    print(f"  {name}: " + ", ".join(f"P={v['precision_prime']:.4f}/R={v['recall_prime']:.4f}" for v in vh["val_history"]), "| depth:", vh.get("val_depth"))
