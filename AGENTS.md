# AGENTS.md

Guidance for coding agents working in this repository.

The full working guide lives in **[CLAUDE.md](./CLAUDE.md)** — read it first. It covers the
commands, the sieve → features → model → dissection architecture, the experimental range
protocol, and the constraints that are easy to trip over (sieve bounds, the 24-bit binary
encoding, trial-division assumptions).

Two points worth repeating up front:

- This is a research experiment. Weak prime precision, semiprime-dominated false positives,
  and losing to the explicit residue-rule baseline are findings to report, not defects to
  tune away.
- Verify changes with `python -m pytest -q tests/` (unit tests; needs `pip install -e '.[dev]'`)
  and `python -m primenet.train --smoke` (~1 min), since the tests do not exercise training.
