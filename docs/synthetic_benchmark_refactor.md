# Gaussian benchmark structure

The notebook runs Gaussian configuration, population inspection, unconditional
DFM fitting and evaluation, then H training and conditional comparisons.
Shared implementations are in configuration.py, unconditional.py, conditional.py
and reporting.py under synthetic_benchmark/. Mixture-specific population code,
notebook stages and tests are preserved on codex/mixture-simulation.

Conditional summaries, W1/KS/sliced-W1 tables and distribution figures include
both raw and event-filtered H-guided draws. Empty subsets have unavailable
metrics and labeled empty plot entries. Training and unconditional diagnostics
precede guided sampling. Tests execute the entire notebook with reduced budgets.

The main guided run uses H-weighted conditional initial states with one global
proposal cap and no fallback on exhaustion. Sampling artifacts record its
initialization method, target, cap, and counts. Training split seeds are checked
before sampling to prevent validation or test trajectories from repeating the
training stream. Regression tests exercise the benchmark with exact H, including
conditional-law recovery, unfiltered terminal violations, and budget exhaustion.
