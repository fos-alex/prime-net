# Token sieve: implementation plan, targets, roadmap

Status: proposal, 2026-09-08. Prototype validated as a scratch script (not committed):
DeepSet over per-prime tokens, trained on 25 primes for 300 steps, reached the sieve-depth
ceiling at every inference-time prime set up to 997 (precision 1.000 at recall 1.0 on the
in-dist range, 9 FPs at threshold 0.5, 9,729 params). Held-out primes were detected at 1.000.

## 1. What changes and why

The current encodings give every prime its own input slot and therefore its own weights.
Divisibility by each prime is learned separately, from the rare examples where that prime
is the smallest factor. That is the staircase, and it stalls at 17 after 3 epochs.

The token sieve makes one detector shared across primes:

```
n  ->  one token per prime p:  [ bits(n mod p) | bits(p) ]        (RB + PB dims)
       phi   : shared MLP, applied to every token            token  -> h in R^H
       pool  : element-wise max across tokens                [T, H] -> H
       rho   : MLP on the pooled vector                      H -> logit
```

Adding a prime at inference adds a token, not a weight. The prime set becomes a runtime
knob, and the precision ceiling moves with it (see sieve-depth table in section 5).

What this does *not* do: it does not discover primes from the bits of n. The residue is
still computed outside the network. Section 6 (stage two) is where that question lives.

## 2. Locked design decisions

| decision | choice | reason |
|---|---|---|
| token contents | `bits(n mod p)` then `bits(p)` | "all residue bits zero" is one easy conjunction, same for every p |
| bit widths | RB = PB = 12 | primes up to 4093 = sqrt(2^24), matches BINARY_BITS = 24 |
| training prime set | 2, 3, 5, 7 always + K random primes <= 4093, seeded | small primes give abundant signal; random large primes exercise high residue bits |
| pooling | max | "any token fired" semantics, invariant to token count |
| phi / rho sizes | 20 -> 64 -> 64 ; 64 -> 64 -> 1 | matches prototype; ~10k params |
| loss | keep `BCEWithLogitsLoss(pos_weight=ln n_max)` | consistency with existing runs; report threshold-free metrics alongside |
| feature spec | `--features tokens` (cannot be combined with 2-D specs in v1) | tokens are rank-3; mixing needs a different contract |
| baseline | residue rule parameterized by the same prime set | the comparison must stay honest at every depth |
| eval protocol | sweep inference prime depth: 97, 199, 499, full | "full" = all primes <= sqrt(range end) |
| registry KPIs | `in_dist` at depth 97 (comparable to legacy runs) + `in_dist_full` | old rows stay comparable |

## 3. Implementation steps (in order, each verifiable alone)

1. **metrics.py**: add `precision_at_recall(y, p, r)`, `average_precision(y, p)`.
   Verify on a 10-element toy array by hand.
2. **nt.py / new `dissect.py`**: `detection_by_smallest_factor(n, y, pred, primes)` returns
   per-prime detection rate for composites whose smallest factor is p, plus the `> max p`
   bucket. This is the held-out-prime transfer test and the staircase plot.
3. **baselines.py**: `residue_rule_predict(n, primes=SMALL_PRIMES)`. Verify recall is
   exactly 1.0 for primes above max(primes).
4. **features.py**: `token_features(n, primes, rb=12, pb=12) -> [B, T, rb+pb] float32`.
   `make_feature_fn("tokens", primes=...)` stamps `.names=["tokens"]`, `.dim=rb+pb`,
   `.primes`, `.chunk=5000`. Reject "tokens" mixed with other specs.
5. **model.py**: `DeepSet(in_dim, phi_hidden=(64,64), rho_hidden=(64,), pool="max")`;
   `build_model("deepset", ...)`. Unit check: permuting tokens leaves the output unchanged.
6. **predict.py**: honor `getattr(feature_fn, "chunk", 50_000)`. At 168 tokens the old
   50k chunk is ~670 MB of float32; 5k is ~67 MB.
7. **train.py**: flags `--train-primes-count` (default 25), `--train-primes-max`
   (default 4093), `--train-primes-always` (default `2,3,5,7`). Save `train_primes`,
   `token_bits` in the checkpoint config. `--smoke` must work with `--features tokens
   --model deepset`.
8. **evaluate.py**: when features are tokens, loop over `--eval-primes-max` (default
   `97,199,499,full`). Per depth: model `prf`, `precision_at_recall(0.999)`, AP, residue
   rule at the same depth, per-prime detection split into seen / held-out primes.
   `report.md` gains a depth table and a held-out-prime table. `eval_metrics.json` gains
   `by_depth`. Registry: `in_dist` = depth 97, plus `in_dist_full`, `p_at_r999_full`.
9. **dashboard.py**: one extra column, `P@R.999 (full)`. Legacy rows show a dash.
10. **serve.py**: `--primes-max` flag; the residue-rule column uses the same primes; the
    page states the prime depth in use.
11. **tests/** (pytest, dev-only dependency): sieve gives pi(10^6) = 78,498; token bit
    round-trip; residue-rule recall; `precision_at_recall` on toy data; DeepSet
    permutation invariance; `factor_stats` vs sympy on 1,000 random ints.
12. **Docs**: README section "Token sieve" with the depth table; CLAUDE.md protocol
    update (prime depth is now part of the range discipline); note that the
    "results not bugs" rule applies to *stage two*, not to the token sieve.

Estimated effort: steps 1-7 one session, 8-10 one session, 11-12 half a session.

## 4. Verification gates

- After step 7: `python -m primenet.train --features tokens --model deepset --smoke`
  runs end to end in about a minute.
- After step 8: on the smoke checkpoint, per-prime detection on held-out primes with at
  least 20 examples is >= 0.99. If not, the shared detector did not transfer and the
  plan stops here.
- After step 11: `pytest` green.

## 5. Targets

Ceilings measured on the current protocol ranges (precision at recall 1.0 for an explicit
sieve of the given depth):

| depth | in-dist [750k, 800k] | near-OOD [1M, 1.2M] | far-OOD [5M, 5.2M] |
|---|---|---|---|
| 97 | 0.613 | 0.598 | 0.536 |
| 199 | 0.711 | 0.690 | 0.616 |
| 499 | 0.865 | 0.833 | 0.721 |
| 997 | 1.000 | 0.985 | 0.826 |
| full (895 / 1096 / 2281) | 1.000 | 1.000 | 1.000 |

All targets are precision at recall >= 0.999, mean over 3 seeds, 1,500 training steps
at batch 4096 unless stated.

**Tier 0, reproduce the prototype inside the repo.**
in-dist >= 0.60 at depth 97 and >= 0.99 at full depth. Every held-out prime with >= 20
examples detected at >= 0.99.

**Tier 1, all three ranges at full depth.**
in-dist >= 0.99, near-OOD >= 0.98, far-OOD >= 0.95. Far-OOD needs primes to 2,281 and
is the bit-width extrapolation test; the gap to 1.0 is budget for that.

**Tier 2, stage two (section 6).**
No precision target. The deliverable is the per-prime detection table for training and
held-out primes. Success criterion for continuing: detection >= 0.9 on training primes
up to 31 from bits of n alone.

## 6. Roadmap after Tier 0 passes

Ordered by information per hour.

1. **Bit-width extrapolation.** Train with `--train-primes-max 1021`, evaluate far-OOD
   at full depth. Then repeat with 4093. Reports whether untrained high bits hurt, and
   settles the far-OOD target.
2. **Ablations on the token.** (a) drop `bits(p)`: if precision is unchanged, phi only
   needs the residue and the feature is simply "residue in binary". (b) pooling: max vs
   logsumexp vs mean. (c) phi width 16 / 32 / 64. (d) training prime count 8 / 25 / 60.
3. **Interpret phi.** It is a 24-input function. Plot its pooled activation against the
   residue value for a few primes. Expected: a single "all bits zero" unit. This replaces
   the first-layer weight-inspection roadmap item with something that is actually legible.
4. **Stage two: divisibility from bits.** New spec `--features ntokens`: token =
   `[bits(n) | bits(p)]` (24 + 12 dims). Phi must compute n mod p == 0 itself. Try the
   MLP phi first; expect failure above small p. Then try a phi that processes bit pairs
   sequentially (a tiny GRU or 2-layer transformer over positions), which can in
   principle implement shift-subtract division. Curriculum: train primes <= 7, then
   grow. This is the only path to a network that generalizes to primes it computes
   nothing for outside itself, and the token harness is what makes success measurable
   on held-out primes.
5. **Threshold-free reporting everywhere.** PR curves in the dashboard; registry rows
   carry AP. Drop pos_weight once the token path is the default, since it is no longer
   needed for recall.
6. **Rust / Candle export.** The DeepSet is ~10k params and pure matmul + max; this is the
   natural candidate for the existing "train Python, serve Rust" roadmap item.
7. **Speed.** Token features are one `[B, T]` modulo and a bit unpack; at T = 338 (far-OOD
   full depth) and B = 4096 that is 1.4M ints per batch. Profile before optimizing.

## 7. Risks

- **Max-pool gradient sparsity.** Only the winning token per hidden unit gets gradient.
  The prototype converged in 300 steps, so this is unlikely to bite, but logsumexp is the
  fallback.
- **Memory in predict_range.** Handled by the chunk hint (step 6).
- **Registry schema.** Add keys, never rename; the dashboard must tolerate missing keys.
- **Scientific framing.** The token sieve *will* beat the residue rule at any depth above
  97, by construction. That is not a discovery about networks; it is a consequence of
  giving the model more primes. README and CLAUDE.md must say so plainly, and keep the
  r/p and Fourier results as the "what does it learn unaided" finding.
