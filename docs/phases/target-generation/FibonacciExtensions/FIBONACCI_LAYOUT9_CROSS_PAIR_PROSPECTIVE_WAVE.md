Purpose:
  Test whether Fibonacci contribution is symbol-dependent among
  Pocket/reversion-friendly pairs.

Already evaluated:
  cadchfrmp
  audcadrmp

Prospective contexts:
  audchfrmp
  audnzdrmp
  eurcadrmp
  nzdcadrmp
  nzdchfrmp

Per context:
  control = Fibonacci present
  treatment = exact Fibonacci feature mask ablated
  H = 4
  target epochs = 20
  seed = 1001
  threshold = 0.0008
  core_lr_mult = 120
  head_lr_mult = 25
  checkpoint_interval = 20
  objective = legacy_first_hit_weighted_ce_v1
  width = 103
  semantic layout = 9
  warmup = full_history_warmup
  calendar snapshot = 1
  train = 2010-01-01 .. 2025-01-01
  infer = 2025-01-01 .. 2026-01-01

Future materialization:
  5 matched pairs / 10 experiments
  initial state paused/train
  do not silently substitute current Layout 10
  use preserved Layout 9 semantic workers
