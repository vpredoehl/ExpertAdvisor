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

## Operator workflow

`SupportedSymbols::TrainingSymbols()` remains the frozen sweep and historical
research universe.  Individual historical cross-symbol materialization instead
uses the production canonical FX pip/Fibonacci capability boundary; it does not
expand the sweep list.

For each reviewed source/target arm, first run:

```sh
LSTM_Release --preview-cross-symbol-historical-experiment=682 \
  --cross-symbol-historical-target=audchfrmp
```

The preview is repeatable-read and reports the complete persisted configured
identity, exact mask, duplicate state, and exact preserved Layout-9 train and
inference worker selections. It never inserts, queues, starts, or publishes a
worker. It fails closed unless the source is Layout 9/width 103 and both
historical workers resolve.

After explicit review, an operator may create one paused/train record with:

```sh
LSTM_Release --materialize-cross-symbol-historical-experiment=682 \
  --cross-symbol-historical-target=audchfrmp
```

The command serializes against experiment writers, rejects an existing or
ambiguous equivalent target, copies the full source configuration, and changes
only the canonical symbol. Release/resume remains a separate explicit scheduler
operation; materialization never dispatches work. Use source 682 for the
Fibonacci-present arm and 683 for the exact frozen-mask arm after reviewing
each preview.
