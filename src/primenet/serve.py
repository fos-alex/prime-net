"""Interactive single-number probe: type integers, see what the model says and why.

The point is not the prediction, it is the disagreement. Every answer is shown
next to the sieve ground truth, the residue-rule baseline, and — when the model
is wrong — the factorization that explains the miss.

Usage:
    python -m primenet.serve                      # newest run in runs/
    python -m primenet.serve runs/<ts>/model.pt
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

from .baselines import residue_rule_predict
from .data import PrimeOracle
from .evaluate import load_model
from .features import BINARY_BITS, SMALL_PRIMES, make_feature_fn, token_bits_from_config
from .nt import factor_stats

MAX_INPUTS = 64

_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>prime-net probe</title>
<style>
:root { --bg:#f6f7f9; --card:#fff; --ink:#1a1f2e; --muted:#6b7280; --line:#e5e7eb; --blue:#2563eb; }
* { box-sizing:border-box; }
body { margin:0; font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--ink); }
main { max-width:920px; margin:0 auto; padding:24px 20px 60px; }
h1 { font-size:22px; margin:0 0 2px; }
.sub { color:var(--muted); font-size:12.5px; margin:0 0 18px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:16px; }
input[type=text] { width:100%; font:15px ui-monospace,SFMono-Regular,Menlo,monospace; padding:10px 12px;
  border:1px solid var(--line); border-radius:8px; background:#fff; color:var(--ink); }
.row { display:flex; gap:10px; margin-top:10px; flex-wrap:wrap; }
button { font:13px system-ui; padding:8px 14px; border-radius:8px; border:1px solid var(--line);
  background:#fff; color:var(--ink); cursor:pointer; }
button.primary { background:var(--blue); color:#fff; border-color:var(--blue); font-weight:600; }
button:hover { border-color:var(--blue); }
table { border-collapse:collapse; width:100%; font-size:13px; }
th,td { padding:8px 10px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }
th { color:var(--muted); font-weight:600; font-size:12px; }
th:first-child, td:first-child { text-align:left; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
tr:last-child td { border-bottom:none; }
.pill { display:inline-block; border-radius:6px; padding:2px 8px; font-size:12px; font-weight:600; }
.ok { background:#dcfce7; color:#14532d; }
.fp { background:#fee2e2; color:#7f1d1d; }
.fn { background:#fef3c7; color:#78350f; }
.bar { display:inline-block; height:8px; border-radius:4px; background:var(--blue); vertical-align:middle; }
.muted { color:var(--muted); }
.why { font-size:12.5px; color:var(--muted); margin-top:14px; line-height:1.6; }
.hhead { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.hhead b { font-size:14px; }
.hhead .muted { font-size:12.5px; }
.hhead button { margin-left:auto; }
#hist a { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.err { color:#7f1d1d; font-size:13px; }
code { background:#f1f3f7; border-radius:4px; padding:1px 5px; font-size:12.5px; }
</style>
</head>
<body>
<main>
<h1>prime-net probe</h1>
<p class="sub" id="sub"></p>

<div class="card">
  <input type="text" id="q" placeholder="1000003, 1018081, 999983" autocomplete="off" spellcheck="false">
  <div class="row">
    <button class="primary" id="go">Ask the model</button>
    <button data-ex="1000003, 1000033, 1000037">large primes</button>
    <button data-ex="1018081, 1022117, 1032247">semiprimes p·q</button>
    <button data-ex="1141247, 1124521, 1081447">beaten by the rule</button>
    <button id="rnd">random trap →</button>
  </div>
</div>

<div class="card" id="out"><span class="muted">Enter integers separated by commas or spaces.</span></div>

<div class="card" id="histcard" hidden>
  <div class="hhead"><b>This session</b><span class="muted" id="hcount"></span>
    <button id="clear">clear</button></div>
  <div id="hist"></div>
</div>

<div class="why">
<b>How to read this.</b> <code>p(prime)</code> is the model's confidence; it predicts prime above 0.5.
<code>truth</code> comes from the sieve. <code>residue rule</code> is the explicit baseline
("composite if divisible by any prime under 100") the model has to beat.
<code>smallest factor</code> is why a miss happened. Two kinds of miss matter, and the
preset buttons show both. <b>Semiprimes p·q</b> with two large factors fool the model
<i>and</i> the rule — they look locally identical to primes, and that is the irreducible
floor. <b>Beaten by the rule</b> is the damning case: a small factor the trivial baseline
catches instantly while the network still votes prime. The gap between those two
categories is how much sieve the model has actually learned.
</div>
</main>
<script>
const qs = s => document.querySelector(s);
const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

/* History is scoped to the checkpoint: a stored p(prime) belongs to one model,
   so a different run gets its own key rather than mixing verdicts. sessionStorage
   (not localStorage) means it survives reload but dies with the tab. */
const HMAX = 200;
let HKEY = null, hist = [];
const loadHist = () => { try { return JSON.parse(sessionStorage.getItem(HKEY) || "[]"); } catch (e) { return []; } };
const saveHist = () => { try { if (HKEY) sessionStorage.setItem(HKEY, JSON.stringify(hist)); } catch (e) {} };

fetch("/api/meta").then(r => r.json()).then(m => {
  qs("#sub").textContent = m.run_id + " · " + m.model + "/" + m.features + " · trained on [2, " +
    m.train_max.toLocaleString() + "] · accepts n ≤ " + m.max_n.toLocaleString() +
    " · residue rule: " + m.primes_depth + " primes" +
    (m.primes_max ? " (≤ " + m.primes_max.toLocaleString() + ")" : " (< 100)");
  HKEY = "primenet-history:" + m.run_id;
  hist = loadHist();
  renderHistory();
});

function verdictPill(r) {
  if (r.correct) return '<span class="pill ok">correct</span>';
  return r.predicted_prime
    ? '<span class="pill fp">false positive</span>'
    : '<span class="pill fn">false negative</span>';
}

function render(rows) {
  const bad = rows.filter(r => r.ok && !r.correct).length;
  let h = "<table><tr><th>n</th><th>p(prime)</th><th></th><th>model</th><th>truth</th>" +
          "<th>residue rule</th><th>smallest factor</th><th>ω(n)</th><th></th></tr>";
  rows.forEach(r => {
    if (!r.ok) {
      h += "<tr><td>" + esc(r.input) + '</td><td colspan="8" class="err">' + esc(r.error) + "</td></tr>";
      return;
    }
    h += "<tr><td>" + r.n.toLocaleString() + "</td>" +
      "<td>" + r.p.toFixed(4) + "</td>" +
      '<td style="text-align:left"><span class="bar" style="width:' + Math.round(r.p * 60) + 'px"></span></td>' +
      "<td>" + (r.predicted_prime ? "prime" : "composite") + "</td>" +
      "<td>" + (r.is_prime ? "prime" : "composite") + "</td>" +
      "<td>" + (r.residue_prime ? "prime" : "composite") + "</td>" +
      "<td>" + (r.is_prime ? '<span class="muted">–</span>' : r.smallest_factor.toLocaleString()) + "</td>" +
      "<td>" + (r.is_prime ? '<span class="muted">–</span>' : r.omega) + "</td>" +
      "<td>" + verdictPill(r) + "</td></tr>";
  });
  h += "</table>";
  if (bad) h += '<p class="sub" style="margin:12px 0 0">' + bad + " of " + rows.length + " wrong.</p>";
  qs("#out").innerHTML = h;
  pushHistory(rows);
}

function pushHistory(rows) {
  rows.filter(r => r.ok).forEach(r => {
    hist = hist.filter(h => h.n !== r.n);   // re-asking a number moves it to the top
    hist.unshift(r);
  });
  hist = hist.slice(0, HMAX);
  saveHist();
  renderHistory();
}

function renderHistory() {
  const card = qs("#histcard");
  if (!hist.length) { card.hidden = true; return; }
  card.hidden = false;
  const wrong = hist.filter(r => !r.correct).length;
  qs("#hcount").textContent = hist.length + " asked · " + wrong + " wrong · " +
    (100 * (1 - wrong / hist.length)).toFixed(0) + "% correct";
  let h = "<table><tr><th>n</th><th>p(prime)</th><th>model</th><th>truth</th>" +
          "<th>residue rule</th><th>smallest factor</th><th></th></tr>";
  hist.forEach(r => {
    h += "<tr><td><a href=\"#\" data-n=\"" + r.n + "\">" + r.n.toLocaleString() + "</a></td>" +
      "<td>" + r.p.toFixed(4) + "</td>" +
      "<td>" + (r.predicted_prime ? "prime" : "composite") + "</td>" +
      "<td>" + (r.is_prime ? "prime" : "composite") + "</td>" +
      "<td>" + (r.residue_prime ? "prime" : "composite") + "</td>" +
      "<td>" + (r.is_prime ? '<span class="muted">–</span>' : r.smallest_factor.toLocaleString()) + "</td>" +
      "<td>" + verdictPill(r) + "</td></tr>";
  });
  qs("#hist").innerHTML = h + "</table>";
}

function ask(text) {
  qs("#q").value = text;
  qs("#out").innerHTML = '<span class="muted">thinking…</span>';
  fetch("/api/predict?n=" + encodeURIComponent(text))
    .then(r => r.json())
    .then(d => d.error ? qs("#out").innerHTML = '<span class="err">' + esc(d.error) + "</span>" : render(d.results))
    .catch(e => qs("#out").innerHTML = '<span class="err">' + esc(e) + "</span>");
}

qs("#go").onclick = () => ask(qs("#q").value);
qs("#q").addEventListener("keydown", e => { if (e.key === "Enter") ask(qs("#q").value); });
document.querySelectorAll("button[data-ex]").forEach(b => b.onclick = () => ask(b.dataset.ex));
qs("#rnd").onclick = () => fetch("/api/trap").then(r => r.json()).then(d => ask(d.n.join(", ")));
qs("#clear").onclick = () => { hist = []; saveHist(); renderHistory(); };
qs("#hist").addEventListener("click", e => {
  const a = e.target.closest("a[data-n]");
  if (a) { e.preventDefault(); ask(a.dataset.n); }
});
</script>
</body>
</html>
"""


class Probe:
    """Holds the model, the sieve and the feature encoder for the server's lifetime."""

    def __init__(self, checkpoint: Path, max_n: int | None = None, primes_max: int | None = None):
        self.model, self.cfg = load_model(checkpoint)
        if self.cfg["features"] == "tokens":
            self.feature_fn = make_feature_fn(
                "tokens", primes=self.cfg["train_primes"], bits=token_bits_from_config(self.cfg)
            )
        else:
            self.feature_fn = make_feature_fn(self.cfg["features"])
        self.run_id = checkpoint.parent.name
        self.max_n = int(max_n or self.cfg["n_max"])
        if "binary" in self.feature_fn.names and self.max_n >= 2**BINARY_BITS:
            raise SystemExit(
                f"max-n {self.max_n:,} exceeds the {BINARY_BITS}-bit binary encoding "
                f"(limit {2**BINARY_BITS - 1:,}); retrain with more bits or lower --max-n"
            )
        self.oracle = PrimeOracle(self.max_n)
        # the residue-rule baseline shares the model's prime depth so the
        # comparison stays honest at every sieve depth
        if primes_max is None and self.cfg["features"] == "tokens":
            primes_max = 97
        self.rule_primes = (
            np.asarray(SMALL_PRIMES, dtype=np.int64)
            if primes_max is None
            else PrimeOracle(int(primes_max)).primes_up_to(int(primes_max))
        )
        self.primes_max = primes_max

    def predict(self, nums: list[int]) -> list[dict]:
        n = np.array(nums, dtype=np.int64)
        with torch.no_grad():
            p = torch.sigmoid(self.model(torch.from_numpy(self.feature_fn(n)))).numpy()
        truth = self.oracle.is_prime(n)
        residue = residue_rule_predict(n, self.rule_primes) == 1.0
        smallest, omega, _ = factor_stats(n, self.oracle.primes)
        out = []
        for i, v in enumerate(nums):
            predicted = bool(p[i] >= 0.5)
            out.append({
                "ok": True,
                "n": v,
                "p": float(p[i]),
                "predicted_prime": predicted,
                "is_prime": bool(truth[i]),
                "residue_prime": bool(residue[i]),
                "smallest_factor": int(smallest[i]),
                "omega": int(omega[i]),
                "correct": predicted == bool(truth[i]),
            })
        return out

    def trap(self, rng: np.random.Generator, count: int = 3) -> list[int]:
        """Semiprimes p*q with both factors large: the model's expected blind spot."""
        lo = int(np.sqrt(self.max_n // 4))
        candidates = self.oracle.primes[self.oracle.primes >= lo]
        picks = []
        while len(picks) < count and len(candidates) > 1:
            a, b = rng.choice(candidates, size=2, replace=True)
            if 2 <= a * b <= self.max_n:
                picks.append(int(a) * int(b))
        return picks


def parse_numbers(raw: str, max_n: int) -> list[dict]:
    """Split user input into ints, keeping per-token errors instead of failing the batch."""
    tokens = [t for t in raw.replace(",", " ").split() if t]
    parsed = []
    for t in tokens[:MAX_INPUTS]:
        try:
            v = int(t)
        except ValueError:
            parsed.append({"ok": False, "input": t, "error": "not an integer"})
            continue
        if v < 2:
            parsed.append({"ok": False, "input": t, "error": "primality is defined for n ≥ 2"})
        elif v > max_n:
            parsed.append({"ok": False, "input": t, "error": f"beyond the sieve (max {max_n:,})"})
        else:
            parsed.append({"ok": True, "value": v})
    return parsed


def make_handler(probe: Probe, rng: np.random.Generator):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict) -> None:
            self._send(json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path == "/":
                self._send(_PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/meta":
                self._json({
                    "run_id": probe.run_id,
                    "model": probe.cfg["model"],
                    "features": probe.cfg["features"],
                    "train_max": probe.cfg["train_max"],
                    "max_n": probe.max_n,
                    "primes_depth": len(probe.rule_primes),
                    "primes_max": probe.primes_max,
                })
            elif url.path == "/api/trap":
                self._json({"n": probe.trap(rng)})
            elif url.path == "/api/predict":
                raw = parse_qs(url.query).get("n", [""])[0]
                parsed = parse_numbers(raw, probe.max_n)
                if not parsed:
                    self._json({"error": "no numbers given"})
                    return
                valid = [p["value"] for p in parsed if p["ok"]]
                scored = iter(probe.predict(valid) if valid else [])
                self._json({"results": [next(scored) if p["ok"] else p for p in parsed]})
            else:
                self.send_error(404)

        def log_message(self, fmt: str, *a) -> None:
            pass  # keep the console clean; this is a single-user local probe

    return Handler


def latest_checkpoint(runs_dir: Path) -> Path:
    ckpts = sorted(runs_dir.glob("*/model.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {runs_dir}/ — train a run first")
    return ckpts[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path, nargs="?", help="default: newest runs/*/model.pt")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--max-n", type=int, default=None, help="largest queryable n (default: training sieve bound)")
    p.add_argument("--primes-max", type=int, default=None,
                   help="residue-rule prime depth (all primes <= this); default: 97 for token models, "
                        "the classic primes<100 rule otherwise")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ckpt = args.checkpoint or latest_checkpoint(Path("runs"))
    probe = Probe(ckpt, args.max_n, args.primes_max)
    print(f"loaded {ckpt} (features={probe.feature_fn.names}, model={probe.cfg['model']})")
    print(f"sieve to {probe.max_n:,} | residue rule on {len(probe.rule_primes)} primes "
          f"| serving http://{args.host}:{args.port}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(probe, np.random.default_rng(args.seed)))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
