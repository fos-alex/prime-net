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
                 │  MLP  99→128→128→64→1      (~25k params)     │
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

- **Train** on `n ~ U[2, 8·10⁵]`
- **Validate** each epoch on `[8·10⁵, 10⁶]`
- **Test in-distribution** on `[10⁶, 1.2·10⁶]` — every integer, enumerated
- **Test out-of-distribution** on `[5·10⁶, 5.2·10⁶]` — 6× beyond the training range
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

## Tracking progress across runs (mini ML Ops)

Every training run appends a line to `runs/registry.jsonl` (config, throughput, final
validation KPIs); every evaluation attaches its full KPIs to the matching record. The
file is plain JSONL — rsync-able, git-diffable, one line per run.

```bash
python -m primenet.train  --tag "my-tweak" ...   # tag each experiment for the board
python -m primenet.evaluate runs/<run>/model.pt  # attaches eval KPIs to the record
python -m primenet.board                         # table + runs/progress/progress.png
python -m primenet.board --backfill              # one-time import of pre-registry runs
```

The board prints a table (prime precision with Δ vs previous run, OOD precision, prime
recall, composite recall, FP median smallest factor, throughput) and renders four charts
against the residue-rule reference: prime precision (in-dist + OOD), prime recall,
composite recall, and FP anatomy on a log scale.

Smoke runs are tracked but hidden from the board by default (`--include-smoke` to show
them). Runs from other machines (e.g. the VPS) merge cleanly: rsync their `runs/` over,
delete duplicate `run_id` lines, and re-run the board. If you later outgrow this,
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
