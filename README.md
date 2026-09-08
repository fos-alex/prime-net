# prime-net

**Can a neural network learn primality?** A small, reproducible PyTorch experiment that
trains tiny models to classify integers as prime or composite, and then dissects exactly
*what* they learn — and why it can never replace Miller–Rabin.

## The short answer (from the literature)

| Study | Approach | Result |
|---|---|---|
| Kolpakov & Rocke 2024 ([PLoS ONE](https://doi.org/10.1371/journal.pone.0301240)) | Info-theory + XGBoost, 18–24-bit ints | Prime indicator sequence is near "algorithmically random" for inductive learning; TPR ~2.2% on raw representations |
| [arXiv 2402.03363](https://arxiv.org/abs/2402.03363) (2024) | ResNet + Transformer, sparse encoding, ~10⁶ ints | 99% prime recall, but ~20% of composites become false positives (prime precision ≈ 0.27). FPs are *consistently* semiprimes p·q with large factors |
| [arXiv 2304.01333](https://arxiv.org/abs/2304.01333) | Divisibility (mod p) classification, CNN/RNN/BERT/AutoML | Networks learn "divisible by small p" *only* with engineered features (binary/Fourier); raw integers fail everywhere, GPT-4 included |
| Charton, [arXiv 2308.15594](https://arxiv.org/abs/2308.15594) | Transformers on GCD | Models learn a **sieve**: base divisibility first, then "grok" small primes one at a time |
| [arXiv 2502.10335](https://arxiv.org/abs/2502.10335) | Transformers on Möbius / squarefree μ(n) | Nontrivial predictive power, with a theoretical account of what is learned |

**Consensus mechanism:** a network trained on primality converges to a *soft, learned
approximation of the Sieve of Eratosthenes*. It picks up divisibility by small primes
fast (high recall, early convergence) and then plateaus. Its ceiling is numbers like
`p·q` with both factors large: those are locally indistinguishable from primes, and no
amount of commodity-hardware training fixes that.

**This repo's contribution is the dissection:** train on a disjoint range, test in- and
out-of-distribution, then factorize every false positive and compare against an explicit
residue-sieve baseline to see precisely what the network did and didn't learn.

## Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │                 data.py                      │
                 │  Sieve of Eratosthenes (bool array, ~10 MB   │
                 │  for N=10⁷) = ground-truth oracle            │
                 │  batches sampled on the fly: n ~ U[2, N_max] │
                 └──────────────┬───────────────────────────────┘
                                │  (n, y)
                 ┌──────────────▼───────────────────────────────┐
                 │               features.py                    │
                 │  binary   : 24-bit vector            (24)    │
                 │  residues : n mod p, p ∈ first 25 primes, /p│
                 │  fourier  : sin/cos(2πn/p)           (50)    │
                 │  "all"    : concatenated              (99)   │
                 └──────────────┬───────────────────────────────┘
                                │  x ∈ R^{B×99}
                 ┌──────────────▼───────────────────────────────┐
                 │                model.py                      │
                 │  MLP  99→128→128→64→1      (37,633 params)   │
                 │  or 1D-CNN over the feature axis             │
                 └──────────────┬───────────────────────────────┘
                                │  logits
                 ┌──────────────▼───────────────────────────────┐
                 │             train.py / evaluate.py           │
                 │  BCE(pos_weight=ln N_max), Adam, bs=4096     │
                 │  val each epoch; test in-dist + far-OOD;     │
                 │  factorize every false positive              │
                 └──────────────────────────────────────────────┘
```

Everything runs on CPU. No dataset is ever stored: labels come from a sieve lookup,
features are vectorized NumPy (C speed), so the data pipeline is never the bottleneck.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate

# CPU-only torch: much smaller download, all this needs
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .

# 1. plumbing check (~1 min)
python -m primenet.train --smoke

# 2. real run: 3 epochs ≈ 5M samples/epoch, a few minutes per epoch on 8 threads
python -m primenet.train --epochs 3

# 3. dissect the trained model (in-distribution + out-of-distribution + FP factorization)
python -m primenet.evaluate runs/<timestamp>/model.pt
```

## Protocol

- **Train** on `n ~ U[2, 7·10⁵]`
- **Validate** each epoch on `[7·10⁵, 7.5·10⁵]` — held out, inside the training span
- **Test in-distribution** on `[7.5·10⁵, 8·10⁵]` — a second held-out interval inside the
  trained range; every integer enumerated. (Testing *inside* the trained span matters:
  with binary features, ranges above 2²⁰ ≈ 1.05M light up bits the model has literally
  never seen during training, which would confound "OOD degradation" with untrained
  weights.)
- **Near OOD**: `[10⁶, 1.2·10⁶]` — just beyond training, where unseen bits activate
- **Far OOD**: `[5·10⁶, 5.2·10⁶]` — ~7× the training range; evaluation builds its own
  sieve to cover whatever ranges are requested, independent of the training `--n-max`
- **Baselines**: (1) the explicit residue rule "composite iff n mod p == 0 for p < 100",
  (2) predict-all-composite. If the network can't beat the residue rule, that *is* the
  result: it learned a sieve and nothing more.
- **False-positive analysis**: for every composite the model calls prime, compute the
  smallest prime factor and ω(n) (distinct prime factors). Expected: reproduces the
  published finding that FPs are dominated by numbers with no small prime factors.

## What to expect

- Convergence is fast: the literature reports peak recall before one epoch is done.
- Prime recall will be high; prime precision will be the weak number — semiprimes of
  large primes are the irreducible error floor.
- OOD degradation on the far range tells you whether the model learned *arithmetic*
  (scale-invariant modular structure) or *statistics of the training range*.

## Token sieve (DeepSet over per-prime tokens)

The residue/Fourier encodings give every prime its own input slot and therefore its own
weights — divisibility by each prime is learned separately, from the rare examples where
that prime is the smallest factor. That is the staircase, and it stalls (17 after 3 epochs
in the legacy runs). The token sieve instead shares one detector across primes:

```
n  ->  one token per prime p:  [ bits(n mod p) | bits(p) ]     (12 + 12 bits)
       phi   : shared MLP, applied to every token    (24 -> 64 -> 64)
       pool  : element-wise max across tokens        ("any token fired")
       rho   : MLP on the pooled vector              (64 -> 64 -> 1, ~10k params)
```

Adding a prime at inference adds a token, not a weight: the prime set is a runtime knob
and the precision ceiling moves with it. Train with
`python -m primenet.train --features tokens --model deepset --tag "token-sieve"`;
evaluation sweeps inference prime depth (`97, 199, 499, full`), reports P@R.999 and AP
per depth, and splits per-prime detection into seen vs held-out primes (the transfer
test — `python -m primenet.serve --primes-max 499` probes it interactively).

What this does *not* do: discover primes from the bits of n — residues are computed
outside the network. And beating the residue rule at any depth above the trained
baseline is a *construction consequence* of giving the model more primes, not a
discovery about networks; the results to watch are held-out transfer and the ceiling.

**Measured (3 seeds × 1,500 steps, batch 4096, mean precision at threshold 0.5):**

| range | depth 97 | depth 499 | full | P@R.999 (full) |
|---|---|---|---|---|
| in-dist [750k, 800k] | 0.613 (= rule, identical FPs) | 0.865 (= rule) | **0.993** | 1.000 |
| near-OOD [1M, 1.2M] | – | – | **0.989** | 0.996 |
| far-OOD [5M, 5.2M] | – | – | **0.968** | 0.978 |

Held-out prime transfer: detection **1.000** on every held-out prime with ≥ 20 examples,
all 3 seeds. At every depth the model converges to the *exact* sieve for that depth —
identical confusion counts to the explicit rule — and extrapolates the shared detector
to 338 unseen primes at far-OOD. Targets, gates and next steps (bit-width
extrapolation, ablations, stage two: divisibility from the bits of n):
`docs/token-sieve-plan.md`.

## Tracking progress across runs (mini ML Ops)

Every training run appends a line to `runs/registry.jsonl` (config, throughput, final
validation KPIs), then **automatically evaluates itself** on the in-dist + near/far-OOD
ranges (`--no-auto-eval` to disable), attaches the full KPIs to its record, and
**regenerates the dashboard**. The file is plain JSONL — rsync-able, git-diffable,
one line per run.

```bash
python -m primenet.train --tag "my-tweak"    # train → auto-eval → track → dashboard
python -m primenet.board --open              # regenerate + open the dashboard any time
python -m primenet.board --backfill          # one-time import of pre-registry runs
```

The dashboard is a single self-contained HTML file (no dependencies, works offline):
overview charts across runs (prime precision vs the residue rule, recalls, FP anatomy
with hover tooltips), a full run table, metric tiles per run, and a detail section per
run with its confusion numbers, FP anatomy and training artifacts. Manual evaluation of
old checkpoints still works: `python -m primenet.evaluate runs/<run>/model.pt`.

Runs from other machines merge cleanly: `scripts/sync-cloud-runs.sh` rsyncs a VPS's
`runs/` (excluding its registry and generated dashboard), backfills the local registry
preserving each run's originating host, and refreshes the dashboard. Wired up as a
systemd user timer (`prime-net-sync.timer`, every 5 min) the local dashboard stays
current with cloud runs automatically. If you later outgrow this,
`pip install mlflow && mlflow ui` is the standard upgrade path.

## Roadmap

- [ ] Encoding ablation: binary vs residues vs Fourier vs all
- [ ] Focal / Asymmetric loss sweep (Focal collapsed for sparse prime families in prior work)
- [ ] Curriculum: grow N_max during training
- [ ] Interpret first-layer weights as learned divisibility detectors (Fourier-circuit lens)
- [ ] Export trained weights to Candle/ONNX for a Rust inference demo (train Python, serve Rust)
- [ ] Optional: Rust/PyO3 data generator — only if profiling ever shows the pipeline starving

## License

MIT
