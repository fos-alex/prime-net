# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research experiment, not a product. The question is *what* a small network learns when
trained on primality, and the expected answer (from the literature summarized in README.md)
is "a soft approximation of the Sieve of Eratosthenes." Low prime precision, false positives
dominated by semiprimes with large factors, and failure to beat the explicit residue-rule
baseline are **results, not bugs**. Do not tune them away or reframe them as defects; the
comparison against `baselines.py` is the point of the repo.

## Commands

```bash
source .venv/bin/activate                    # CPU-only torch already installed here

python -m primenet.train --smoke             # ~1 min end-to-end plumbing check
python -m primenet.train --epochs 3          # real run, minutes per epoch on 8 threads
python -m primenet.evaluate runs/<ts>/model.pt   # train.py also runs this automatically
python -m primenet.board                        # table + runs/progress/index.html dashboard
python -m primenet.serve                        # interactive probe on :8000 (newest checkpoint)
python -m pytest -q tests/                      # unit tests (needs: pip install -e '.[dev]')

# encoding ablation (the main open experiment)
python -m primenet.train --features residues --epochs 3
python -m primenet.train --features binary,fourier --model cnn
```

Console scripts `primenet-train` / `primenet-eval` / `primenet-serve` are equivalent entry points.

`tests/test_primenet.py` covers the sieve, the encodings, factorization (against sympy)
and the feature-function contract; there is no linter configured. Tests do not exercise
training, so `--smoke` is still the end-to-end check: it shrinks `n_max`, ranges, epochs
and steps so the whole pipeline runs in about a minute. Run both after touching anything
in the data → features → model → train chain.

## Architecture

The spine is `sieve → features → model → dissection`, and each stage is swappable without
touching the others.

- **`data.py` — the oracle.** `PrimeOracle` holds one boolean sieve array (~1 byte/integer)
  that serves as ground truth for *every* label in the project, and `sample_batch` draws
  `n ~ U[n_min, n_max]` and labels it by lookup. No dataset is ever stored on disk; batches
  are generated per step. Feature computation is vectorized NumPy so the pipeline never
  starves the model.
- **`features.py` — the ablation axis.** `make_feature_fn(spec)` builds a callable from a
  comma-separated spec (`binary`, `residues`, `fourier`, `all`) and stamps `.names` and
  `.dim` on it, which is how the rest of the code learns the input width. Encoding is the
  single most consequential knob here — raw integers do not work, per arXiv 2304.01333.
- **`model.py`** — `build_model(name, in_dim, hidden)` returns an MLP or a 1D-CNN over the
  feature axis. Both are deliberately tiny (<200k params) and CPU-only.
- **`train.py`** — argparse config → run dir `runs/<timestamp>/` containing
  `config.json`, `model.pt`, `metrics.json`, `loss_curve.png`. Class imbalance is handled
  by `BCEWithLogitsLoss(pos_weight=ln(n_max))`, not by resampling. Evaluation runs
  automatically at the end of training.
- **`predict.py`** — `predict_range` enumerates a full integer interval in chunks; shared
  by training validation, evaluation and the probe so inference is identical everywhere.
- **`evaluate.py`** — the dissection. It reconstructs the model *from the checkpoint's own
  config*, then scores in-distribution and far-OOD ranges against two baselines
  (`baselines.py`) and factorizes every false positive (`nt.py`) into smallest prime factor
  and ω(n), writing `report.md`, `eval_metrics.json`, `confusion.png`, `fp_anatomy.png`
  next to the checkpoint.
- **`metrics.py`** — `prf()` is the one metric function; both training and evaluation and
  all baselines go through it so the numbers are directly comparable.
- **`track.py` / `board.py` / `dashboard.py`** — the run registry. `runs/registry.jsonl`
  holds **one record per run**, keyed by `run_id` (the run directory name); train appends
  it, evaluate attaches eval KPIs to it, and `append_record` de-dupes on `run_id`. Only
  `val_history[-1]` reaches the registry, so the board shows the *final* epoch, not the
  best one. `board.py` prints the table and calls `dashboard.py`, which writes a
  dependency-free `runs/progress/index.html`.
- **`serve.py`** — the interactive probe. Loads one checkpoint, holds the sieve for the
  server's lifetime, and answers `/api/predict?n=...` with the model probability next to
  sieve truth, the residue-rule verdict, and the factorization of every miss. Hand-rolled
  HTML with no JS dependencies, matching `dashboard.py`.

Range discipline is the experimental protocol: train `[2, 7e5]`, validate `[7e5, 7.5e5]`,
then test in-distribution `[7.5e5, 8e5]` (inside the training range), near-OOD
`[1e6, 1.2e6]` and far-OOD `[5e6, 5.2e6]`. Keep these deliberate when changing defaults —
the in-dist → far-OOD gap is what distinguishes "learned modular arithmetic" from
"learned the statistics of the training range."

For token models, **prime depth is part of that discipline**: tokens are residue bits
only (`--token-bits 12,0`; the prime bits of the first version caused max-pool dilution
at far-OOD and were dropped), per-epoch validation runs at full depth, training primes
(`2,3,5,7` + K random ≤ 4093) are a runtime knob, evaluation sweeps inference depth
(`--eval-primes-max 97,199,499,full`), and per-prime detection on held-out primes is
the transfer test. Registry `in_dist` is the first depth (97) for legacy comparability;
`in_dist_full` / `p_at_r999_full` carry the full-depth numbers. Note that the
"results not bugs" bar applies to *stage two* (divisibility from the bits of n), not
to the token sieve: beating the residue rule at any depth above the trained baseline
is a consequence of giving the model more primes, and is reported as such.

`runs/` is gitignored. Treat its contents as disposable output, never as inputs to code.

## Constraints that bite

- **The sieve bounds what you can ask about.** `evaluate.py` sizes its own oracle to the
  ranges it was asked for, independent of the training bound, but `serve.py` can only
  answer for `n <= max_n` because ground truth is a sieve lookup. There is no primality
  test in this repo other than the sieve.
- **Binary encoding is 24 bits**, so it only addresses n < 2^24 ≈ 16.7M. `train.py` guards
  this with a substring check on `--features`, which does not fire for `--features all`.
  Raising `--n-max` past 2^24 requires raising `BINARY_BITS` too.
- **`factor_stats` assumes** its `primes` argument covers every prime ≤ √max(nums); it is
  called with `oracle.primes`, which holds only because the sieve bound exceeds the
  evaluated range. Anything left after trial division is taken to be a single large prime.

## In flight

Legacy encodings (binary / residues / fourier): the MLP reaches prime precision ~0.42
against the residue rule's ~0.61 and has **not** beaten the baseline, which is a result
worth preserving, not a bug to fix. It learns divisibility one prime at a time and stalls
around 17 after 3 epochs.

Token sieve: Tier 1 of `docs/token-sieve-plan.md` is met (2026-09-10, 3 seeds, residue-only
tokens): precision 1.000 at full depth on in-dist, near-OOD and far-OOD, identical FP
counts to the explicit rule at every depth, held-out prime detection 1.000. Precision is
finished as a goal for this model. The open experiment is **stage two**: `[bits(n) |
bits(p)]` tokens where the shared network must compute divisibility itself (plan
section 6, item 4). `scripts/summarize_seeds.py <tag-prefix>` aggregates seed runs.
