"""Minimal run registry: one JSON line per training run, in runs/registry.jsonl.

Every train run appends a record; every evaluate run attaches its eval KPIs to
the matching record. primenet.board reads the file and renders the progress
table + charts. The file is plain JSONL: rsync-able, git-diffable, mergeable.
"""

from __future__ import annotations

import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# config keys kept in the registry (the ones that define a run)
_CONFIG_KEYS = (
    "features",
    "model",
    "epochs",
    "steps_per_epoch",
    "batch_size",
    "lr",
    "n_max",
    "train_max",
)


def git_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def load_registry(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"warning: skipping malformed registry line: {line[:80]}...")
    return records


def save_registry(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def append_record(path: Path, record: dict) -> None:
    records = [r for r in load_registry(path) if r["run_id"] != record["run_id"]]
    records.append(record)
    save_registry(path, records)


def update_record(path: Path, run_id: str, updates: dict) -> bool:
    records = load_registry(path)
    for r in records:
        if r["run_id"] == run_id:
            r.update(updates)
            save_registry(path, records)
            return True
    return False


def make_record(
    run_id: str,
    config: dict,
    params: int | None,
    samples: int | None,
    sps: float | None,
    final_loss: float | None,
    val_metrics: dict | None,
    tag: str = "",
) -> dict:
    """Build a registry record. config is the train-time vars(args) dict."""
    return {
        "run_id": run_id,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "git": git_commit(),
        "tag": tag,
        "smoke": bool(config.get("smoke")),
        "config": {k: config[k] for k in _CONFIG_KEYS if k in config},
        "params": params,
        "samples": samples,
        "sps": sps,
        "final_loss": final_loss,
        "val": val_metrics,
        "eval": None,
    }
