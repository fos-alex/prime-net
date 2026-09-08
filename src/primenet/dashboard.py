"""Self-contained HTML dashboard from the run registry.

Generates runs/progress/index.html: overview charts (canvas + inline JS, hover
tooltips), a run table, metric tiles per run, and a detail section per run.
No dependencies, works offline over file://.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .track import load_registry

_ARTIFACTS = [
    "loss_curve.png",
    "confusion.png",
    "fp_anatomy.png",
    "report.md",
    "eval_metrics.json",
    "model.pt",
]

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>prime-net dashboard</title>
<style>
:root { --bg:#f6f7f9; --card:#fff; --ink:#1a1f2e; --muted:#6b7280; --line:#e5e7eb; --blue:#2563eb; --orange:#ea580c; }
* { box-sizing: border-box; }
body { margin:0; font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--ink); }
nav { position:sticky; top:0; background:#fff; border-bottom:1px solid var(--line); padding:10px 20px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; z-index:5;}
nav a { color:var(--muted); text-decoration:none; font-size:13px;}
nav a:hover { color:var(--blue);}
nav .brand { font-weight:700; color:var(--ink); margin-right:8px;}
main { max-width:1200px; margin:0 auto; padding:20px; }
h1 { font-size:22px; margin:4px 0 2px;} h2 { font-size:17px; margin:26px 0 10px;} h3 { font-size:15px; margin:0 0 8px;}
.sub { color:var(--muted); font-size:12.5px; margin:0 0 18px;}
.charts { display:grid; grid-template-columns:1fr 1fr; gap:16px;}
.chartbox { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px;}
.chartbox canvas { width:100%; height:270px; display:block;}
table { border-collapse:collapse; width:100%; background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; font-size:13px;}
th,td { padding:7px 10px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap;}
th { background:#fafbfc; color:var(--muted); font-weight:600;}
th:first-child, td:first-child { text-align:left;}
tr:last-child td { border-bottom:none;}
a { color:var(--blue); text-decoration:none;}
.tiles { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px;}
.tile { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px;}
.tile .runid { font-weight:600;}
.tile .meta { color:var(--muted); font-size:12px; margin:2px 0 8px;}
.chips { display:flex; flex-wrap:wrap; gap:6px;}
.chip { background:#f1f3f7; border-radius:6px; padding:3px 8px; font-size:12px;}
.chip b { font-weight:650;}
.chip.good { background:#dcfce7; color:#14532d;}
.chip.mid { background:#fef3c7; color:#78350f;}
.chip.bad { background:#fee2e2; color:#7f1d1d;}
.tagbadge { display:inline-block; background:#e0e7ff; color:#3730a3; border-radius:6px; padding:1px 7px; font-size:11.5px; margin-left:6px;}
.detail { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:18px;}
.detail .art { margin-top:10px; display:flex; gap:12px; flex-wrap:wrap; align-items:flex-start;}
.detail .art img { max-width:340px; width:100%; border:1px solid var(--line); border-radius:8px;}
.detail .links { margin-top:8px; font-size:12.5px; display:flex; gap:12px; flex-wrap:wrap;}
.muted { color:var(--muted);}
.badge-none { color:var(--muted); font-style:italic;}
#tooltip { position:fixed; pointer-events:none; background:#111827; color:#fff; padding:6px 9px; border-radius:6px; font-size:12px; display:none; z-index:10; box-shadow:0 2px 8px rgba(0,0,0,.25);}
.cols2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:start;}
@media (max-width:820px){ .charts,.cols2{grid-template-columns:1fr;} }
</style>
</head>
<body>
<nav id="nav"><span class="brand">prime-net</span></nav>
<main>
<h1>Progress dashboard</h1>
<p class="sub" id="subhead"></p>
<section id="overview">
<h2>Overview — KPIs across evaluated runs</h2>
<div class="charts">
  <div class="chartbox"><canvas id="c-precision"></canvas></div>
  <div class="chartbox"><canvas id="c-recall"></canvas></div>
  <div class="chartbox"><canvas id="c-composite"></canvas></div>
  <div class="chartbox"><canvas id="c-fp"></canvas></div>
</div>
<h2>Run table</h2>
<div id="tablewrap"></div>
</section>
<section id="runs">
<h2>Runs</h2>
<div class="tiles" id="tiles"></div>
</section>
<section id="details">
<h2>Run details</h2>
<div id="details"></div>
</section>
</main>
<div id="tooltip"></div>
<script>
const DATA = __DATA__;

const qs = s => document.querySelector(s);
const fmt = (v, d) => (v == null || !isFinite(v)) ? "–" : (+v).toFixed(d == null ? 3 : d);
const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const cfgLabel = r => { const c = r.config || {}; return (c.model||"?") + "/" + (c.features||"?") + "/e" + (c.epochs=="?"?"?":c.epochs); };
const runs = DATA.runs;
const evaluated = runs.filter(r => r.eval && r.eval.in_dist);

/* ---------- nav + header ---------- */
{
  qs("#nav").innerHTML += ' <a href="#overview">Overview</a>' +
    runs.slice().reverse().map(r =>
      '<a href="#run-' + r.run_id + '" title="' + esc(r.run_id) + '">' +
      r.run_id.slice(-6) + (r.tag ? " · " + esc(r.tag) : "") + "</a>").join("");
  qs("#subhead").textContent =
    runs.length + " run(s) · " + evaluated.length + " evaluated · generated " + DATA.generated;
}

/* ---------- charts ---------- */
function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= raw) return m * p;
  return 10 * p;
}

const tip = document.getElementById("tooltip");

function drawChart(canvas, cfg) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 520, H = canvas.clientHeight || 270;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr); ctx.clearRect(0, 0, W, H);
  const pad = {l: 56, r: 10, t: 30, b: 44};
  const tf = v => cfg.logY ? Math.log10(v) : v;
  const ys = [];
  cfg.series.forEach(s => s.values.forEach(v => { if (v != null && isFinite(v)) ys.push(tf(v)); }));
  if (!ys.length) {
    ctx.fillStyle = "#9ca3af"; ctx.font = "12px system-ui";
    ctx.fillText("no data yet", pad.l, H / 2); return;
  }
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (cfg.logY) {
    ymin = Math.floor(ymin + 1e-9); ymax = Math.ceil(ymax - 1e-9);
    if (ymax - ymin < 1) ymax = ymin + 1;
  } else {
    if (ymax === ymin) { ymin -= 0.1; ymax += 0.1; }
    const m = (ymax - ymin) * 0.08; ymin -= m; ymax += m;
  }
  const n = cfg.labels.length;
  const X = i => n === 1 ? (pad.l + W - pad.r) / 2 : pad.l + i * (W - pad.l - pad.r) / (n - 1);
  const Y = v => pad.t + (1 - (tf(v) - ymin) / (ymax - ymin)) * (H - pad.t - pad.b);
  ctx.font = "11px system-ui"; ctx.lineWidth = 1;
  ctx.strokeStyle = "#eef0f3"; ctx.fillStyle = "#6b7280";
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  const ticks = [];
  if (cfg.logY) { for (let p = ymin; p <= ymax; p++) ticks.push(Math.pow(10, p)); }
  else { const st = niceStep((ymax - ymin) / 4); for (let v = Math.ceil(ymin / st) * st; v <= ymax + 1e-12; v += st) ticks.push(v); }
  ticks.forEach(v => {
    const y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillText(cfg.logY ? v.toExponential(0) : String(+v.toFixed(3)), pad.l - 6, y);
  });
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const every = Math.max(1, Math.ceil(n / 8));
  cfg.labels.forEach((lb, i) => {
    if (i % every === 0 || i === n - 1) ctx.fillText(String(lb), X(i), H - pad.b + 6);
  });
  ctx.strokeStyle = "#d1d5db";
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke();
  cfg.series.forEach(s => {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.dash ? 1.6 : 2;
    ctx.setLineDash(s.dash ? [5, 4] : []);
    ctx.beginPath(); let started = false;
    s.values.forEach((v, i) => {
      if (v == null || !isFinite(v)) { started = false; return; }
      const x = X(i), y = Y(v);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.stroke(); ctx.setLineDash([]);
    if (!s.noDots) s.values.forEach((v, i) => {
      if (v == null || !isFinite(v)) return;
      ctx.beginPath(); ctx.arc(X(i), Y(v), 3, 0, 7); ctx.fillStyle = s.color; ctx.fill();
    });
  });
  ctx.fillStyle = "#1a1f2e"; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  ctx.font = "600 12.5px system-ui"; ctx.fillText(cfg.title, 12, 16);
  ctx.font = "11px system-ui";
  let lx = W - pad.r;
  for (let i = cfg.series.length - 1; i >= 0; i--) {
    const s = cfg.series[i];
    const w = ctx.measureText(s.name).width + 24; lx -= w;
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.setLineDash(s.dash ? [5, 4] : []);
    ctx.beginPath(); ctx.moveTo(lx, 12); ctx.lineTo(lx + 12, 12); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#374151"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(s.name, lx + 16, 12); lx -= 10;
  }
  canvas.onmousemove = ev => {
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    let best = 0, bd = 1e9;
    for (let i = 0; i < n; i++) { const d = Math.abs(X(i) - mx); if (d < bd) { bd = d; best = i; } }
    showTooltip(ev.clientX, ev.clientY, cfg, best);
  };
  canvas.onmouseleave = () => { tip.style.display = "none"; };
}

function showTooltip(px, py, cfg, i) {
  const rows = cfg.series.map(s => {
    const v = s.values[i];
    const txt = (v == null || !isFinite(v)) ? "–" : (cfg.logY ? Math.round(v).toLocaleString() : (+v).toFixed(4));
    return '<div><span style="color:' + s.color + '">■</span> ' + esc(s.name) + ": <b>" + txt + "</b></div>";
  }).join("");
  tip.innerHTML = '<div style="opacity:.7;font-size:10.5px;margin-bottom:2px">' + esc(cfg.labels[i] || "") + "</div>" + rows;
  tip.style.display = "block";
  tip.style.left = Math.min(px + 14, window.innerWidth - 210) + "px";
  tip.style.top = (py + 14) + "px";
}

function renderCharts() {
  if (!evaluated.length) { qs("#overview .charts").style.opacity = .4; return; }
  const labels = evaluated.map(r => (r.tag ? r.tag + " · " : "") + r.run_id.slice(-6));
  const sv = (key, scope) => evaluated.map(r => { const m = ((r.eval[scope] || (scope === "near_ood" ? r.eval.ood : null)) || {})[key]; return m == null ? null : m; });
  const fpv = scope => evaluated.map(r => { const f = ((r.eval.fp || {})[scope] || {}); return f.median_smallest == null ? null : f.median_smallest; });
  const RES = "#111827", NEAR = "#ea580c", FAR = "#dc2626";
  drawChart(qs("#c-precision"), { title: "Prime precision (higher = fewer false alarms)", labels, series: [
    { name: "model in-dist", color: "#2563eb", values: sv("precision_prime", "in_dist") },
    { name: "model near-OOD", color: NEAR, values: sv("precision_prime", "near_ood") },
    { name: "model far-OOD", color: FAR, values: sv("precision_prime", "far_ood") },
    { name: "residue rule (in-dist)", color: RES, dash: true, noDots: true, values: sv("precision_prime", "residue_rule") },
  ]});
  drawChart(qs("#c-recall"), { title: "Prime recall (catching true primes)", labels, series: [
    { name: "in-dist", color: "#2563eb", values: sv("recall_prime", "in_dist") },
    { name: "near-OOD", color: NEAR, values: sv("recall_prime", "near_ood") },
    { name: "far-OOD", color: FAR, values: sv("recall_prime", "far_ood") },
  ]});
  drawChart(qs("#c-composite"), { title: "Composite recall (1 − false positive rate)", labels, series: [
    { name: "in-dist", color: "#2563eb", values: sv("recall_composite", "in_dist") },
    { name: "near-OOD", color: NEAR, values: sv("recall_composite", "near_ood") },
    { name: "far-OOD", color: FAR, values: sv("recall_composite", "far_ood") },
    { name: "residue rule (in-dist)", color: RES, dash: true, noDots: true, values: sv("recall_composite", "residue_rule") },
  ]});
  drawChart(qs("#c-fp"), { title: "FP anatomy: median smallest factor (log scale)", labels, logY: true, series: [
    { name: "FPs in-dist", color: "#2563eb", values: fpv("in_dist") },
    { name: "FPs near-OOD", color: NEAR, values: fpv("near_ood") },
    { name: "FPs far-OOD", color: FAR, values: fpv("far_ood") },
    { name: "residue rule FPs", color: RES, dash: true, noDots: true, values: fpv("residue_in_dist") },
  ]});
}

/* ---------- run table ---------- */
function renderTable() {
  let html = '<table><tr><th>run</th><th>tag</th><th>cfg</th><th>Msamp</th><th>P_in</th><th>P_near</th><th>P_far</th><th>R_in</th><th>compR</th><th>FPmed</th><th>sps</th><th>host</th></tr>';
  runs.slice().reverse().forEach(r => {
    const e = r.eval || {}, m = e.in_dist || {};
    const near = e.near_ood || e.ood || {}, far = e.far_ood || {};
    const fp = ((e.fp || {}).in_dist) || {};
    html += "<tr><td><a href='#run-" + r.run_id + "'>" + r.run_id + "</a></td>" +
      "<td>" + (r.tag ? esc(r.tag) : '<span class="muted">–</span>') + "</td>" +
      "<td>" + esc(cfgLabel(r)) + "</td>" +
      "<td>" + (r.samples ? (r.samples / 1e6).toFixed(2) : "–") + "</td>" +
      "<td>" + fmt(m.precision_prime) + "</td><td>" + fmt(near.precision_prime) + "</td>" +
      "<td>" + fmt(far.precision_prime) + "</td>" +
      "<td>" + fmt(m.recall_prime) + "</td><td>" + fmt(m.recall_composite) + "</td>" +
      "<td>" + (fp.median_smallest == null ? "–" : fp.median_smallest) + "</td>" +
      "<td>" + (r.sps ? Math.round(r.sps / 1e3) + "k" : "–") + "</td>" +
      "<td>" + esc(r.host || "") + "</td></tr>";
  });
  qs("#tablewrap").innerHTML = html + "</table>";
}

/* ---------- tiles ---------- */
function chip(name, value, cls) {
  return '<span class="chip ' + (cls || "") + '">' + name + " <b>" + value + "</b></span>";
}
function renderTiles() {
  let html = "";
  runs.slice().reverse().forEach(r => {
    const e = r.eval, m = (e && e.in_dist) || {};
    const near = (e && (e.near_ood || e.ood)) || {}, far = (e && e.far_ood) || {};
    const fp = (e && e.fp && e.fp.in_dist) || {}, rr = (e && e.residue_rule) || {};
    const p = m.precision_prime;
    let cls = "";
    if (p != null && rr.precision_prime)
      cls = p >= 0.95 * rr.precision_prime ? "good" : (p >= 0.6 * rr.precision_prime ? "mid" : "bad");
    html += '<div class="tile"><div class="runid">' + r.run_id +
      (r.tag ? '<span class="tagbadge">' + esc(r.tag) + "</span>" : "") + "</div>" +
      '<div class="meta">' + esc((r.time || "").replace("T", " ")) + " · " + esc(r.host || "?") +
      (r.git ? " · " + esc(r.git) : "") + " · " + esc(cfgLabel(r)) + "</div>" +
      '<div class="chips">' +
      chip("P_in", fmt(p), cls) + chip("P_near", fmt(near.precision_prime), cls) +
      chip("P_far", fmt(far.precision_prime), cls) +
      chip("R_in", fmt(m.recall_prime), m.recall_prime != null && m.recall_prime >= 0.99 ? "good" : "") +
      chip("compR", fmt(m.recall_composite)) +
      chip("FPmed", fp.median_smallest == null ? "–" : fp.median_smallest) +
      chip("Msamp", r.samples ? (r.samples / 1e6).toFixed(1) : "–") +
      chip("sps", r.sps ? Math.round(r.sps / 1e3) + "k" : "–") +
      "</div>" +
      '<div style="margin-top:8px"><a href="#run-' + r.run_id + '">details →</a></div></div>';
  });
  qs("#tiles").innerHTML = html;
}

/* ---------- details ---------- */
function metricsTable(name, m) {
  if (!m) return "";
  return "<table><tr><th>" + esc(name) + "</th><th>P prime</th><th>R prime</th><th>F1 prime</th><th>R composite</th><th>acc</th><th>FP</th></tr>" +
    "<tr><td>values</td><td>" + fmt(m.precision_prime) + "</td><td>" + fmt(m.recall_prime) + "</td><td>" +
    fmt(m.f1_prime) + "</td><td>" + fmt(m.recall_composite) + "</td><td>" + fmt(m.accuracy) + "</td><td>" +
    (m.fp == null ? "–" : m.fp.toLocaleString()) + "</td></tr></table>";
}
function renderDetails() {
  let html = "";
  runs.slice().reverse().forEach(r => {
    const e = r.eval;
    const c = r.config || {};
    const art = r.artifacts || [];
    const imgs = art.filter(a => a.endsWith(".png"));
    const links = art.filter(a => !a.endsWith(".png"));
    html += '<div class="detail" id="run-' + r.run_id + '"><h3>' + r.run_id +
      (r.tag ? '<span class="tagbadge">' + esc(r.tag) + "</span>" : "") +
      (r.smoke ? ' <span class="muted">(smoke)</span>' : "") + "</h3>" +
      '<div class="meta muted" style="font-size:12.5px;margin-bottom:10px">' +
      esc(r.time || "") + " · host " + esc(r.host || "?") +
      " · git " + esc(r.git || "?") + " · " + (r.params == null ? "?" : r.params.toLocaleString()) + " params" +
      " · features " + esc(c.features || "?") + " · " + esc(c.model || "?") +
      " · lr " + esc(c.lr) + " · bs " + esc(c.batch_size) + " · sieve " + esc((c.n_max == null ? "?" : c.n_max.toLocaleString())) + "</div>";
    if (e && e.in_dist) {
      const fp = e.fp || {};
      const fchip = (k, lbl) => chip(lbl, (fp[k] && fp[k].median_smallest != null) ? fp[k].median_smallest : "–");
      html += '<div class="cols2"><div>' + metricsTable("model in-dist", e.in_dist) +
        metricsTable("model near-OOD", e.near_ood || e.ood) + metricsTable("model far-OOD", e.far_ood) +
        "</div><div>" + metricsTable("residue rule in-dist", e.residue_rule) +
        '<div class="chips" style="margin-top:10px">' +
        fchip("in_dist", "FP in-dist (median smallest factor)") +
        fchip("near_ood", "FP near-OOD") + fchip("far_ood", "FP far-OOD") +
        fchip("residue_in_dist", "FP residue rule") +
        "</div></div></div>";
    } else {
      html += '<span class="badge-none">not evaluated yet</span>';
    }
    if (imgs.length || links.length) {
      html += '<div class="art">' + imgs.map(a =>
        '<a href="../' + r.run_id + "/" + a + '" target="_blank"><img src="../' + r.run_id + "/" + a + '" alt="' + a + '"></a>').join("") + "</div>" +
        '<div class="links">' + links.map(a => '<a href="../' + r.run_id + "/" + a + '">' + a + "</a>").join("") + "</div>";
    }
    html += "</div>";
  });
  qs("#details").innerHTML = html;
}

renderCharts();
renderTable();
renderTiles();
renderDetails();
window.addEventListener("resize", () => renderCharts());
</script>
</body>
</html>
"""


def _with_artifacts(runs_dir: Path, records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        r = dict(r)
        run_dir = runs_dir / r["run_id"]
        r["artifacts"] = [a for a in _ARTIFACTS if (run_dir / a).exists()]
        out.append(r)
    return out


def generate_dashboard(runs_dir: Path, records: list[dict]) -> Path:
    data = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": _with_artifacts(runs_dir, records),
    }
    blob = json.dumps(data).replace("</", "<\\/")
    out_dir = runs_dir / "progress"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "index.html"
    out.write_text(_TEMPLATE.replace("__DATA__", blob, 1))
    return out


def refresh_default_dashboard(runs_dir: Path, include_smoke: bool = False) -> Path:
    """Regenerate the dashboard from the current registry (board's default view)."""
    records = load_registry(runs_dir / "registry.jsonl")
    if not include_smoke:
        records = [r for r in records if not r.get("smoke")]
    return generate_dashboard(runs_dir, records)
