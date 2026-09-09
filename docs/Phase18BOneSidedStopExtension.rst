Phase 18B One-Sided Probability-Conditioned Stop Extension
===========================================================

Locked version-1 hypothesis
---------------------------

Phase 18B adds ``probability_conditioned_stop_extension_v1`` with mapping
identity ``directional_probability_one_sided_stop_extension_v1``.  It uses the
unchanged Phase 18A directional probability and normalization::

  directional_probability = P(entered direction)
  normalized_directional_confidence = clamp((p - 1/3) / (2/3), 0, 1)

The predeclared configuration is base logarithmic stop distance ``0.001``,
activation confidence ``0.50``, minimum multiplier ``1.00``, and maximum
multiplier ``1.25``::

  extension_fraction = clamp((confidence - 0.50) / 0.50, 0, 1)
  stop_multiplier = 1.00 + 0.25 * extension_fraction
  effective_stop_log_distance = 0.001 * stop_multiplier

The multiplier is exactly one through confidence ``0.50`` and is monotonic and
continuous above it.  The minimum multiplier is fixed at one by validation, so
the strategy cannot tighten below the fixed-stop attribution control.  Setting
the maximum multiplier to one exactly reproduces fixed-stop executions and
returns.  Phase 18A's family, identity configuration, mapping, evaluator, and
artifact serialization are unchanged.

Controlled comparison
---------------------

``EvaluateControlledOneSidedStopExtensionExperiment`` evaluates exactly these
ordered output identities on one immutable ``AuthoritativeMarketPath``:

* ``baseline_terminal_v1``
* ``fixed_stop_loss_v1``
* ``probability_conditioned_stop_loss_v1`` (unchanged Phase 18A diagnostic)
* ``probability_conditioned_stop_extension_v1``

The primary pairwise record is extension versus fixed stop.  The artifact also
contains extension versus baseline, extension versus Phase 18A, and retained
Phase 18A versus fixed-stop records.  It retains the Phase 18A confidence
buckets and reports per-strategy metrics, deterministic identities and result
hashes, per-observation evidence, floor/extension counts and rates, the average
multiplier among extended observations, and within-bucket stop-hit count/rate
and aggregate-return differences versus fixed stop.

The evaluator fails closed if the four strategies do not share the same path
and entry population, if multiplier accounting does not reconcile, or if a
confidence-at-or-below-activation Phase 18B execution differs from fixed stop.
It has no database or scheduler dependency and performs no persistence.

Read-only CLI
-------------

The standalone interface is::

  --infer --model=<model_id> --symbol=<symbol_table> \
    --probability-conditioned-stop-extension-evaluation=<artifact_path> \
    <fromDate> <toDate>

It rejects ``--infer-all``, training mode, missing explicit model identity,
scheduler experiment/checkpoint/worker context, and frozen-outcome context.
The runtime opens the database transaction with ``SET TRANSACTION READ ONLY``
before model or evaluation work, writes only the requested filesystem artifact,
commits the read-only transaction, and skips inference-result persistence and
model save.  Both the artifact and terminal summary state
``production_rows_modified=false``.

Locked production cohort commands
---------------------------------

After a Release build at
``DerivedData/Release/Phase18B/Build/Products/Release/LSTM_Release``, run the
unchanged Phase 18A cohort over ``2025-01-01`` through ``2026-01-01``::

  BIN=./DerivedData/Release/Phase18B/Build/Products/Release/LSTM_Release
  mkdir -p /tmp/lstm-phase18b
  "$BIN" --infer --model=1745 --symbol=audchfrmp --probability-conditioned-stop-extension-evaluation=/tmp/lstm-phase18b/experiment610-model1745-audchf-h4.csv 2025-01-01 2026-01-01
  "$BIN" --infer --model=1743 --symbol=eurchfrmp --probability-conditioned-stop-extension-evaluation=/tmp/lstm-phase18b/experiment609-model1743-eurchf-h12.csv 2025-01-01 2026-01-01
  "$BIN" --infer --model=1729 --symbol=usdcadrmp --probability-conditioned-stop-extension-evaluation=/tmp/lstm-phase18b/experiment608-model1729-usdcad-h6.csv 2025-01-01 2026-01-01
  "$BIN" --infer --model=1735 --symbol=gbpusdrmp --probability-conditioned-stop-extension-evaluation=/tmp/lstm-phase18b/experiment606-model1735-gbpusd-h6.csv 2025-01-01 2026-01-01
  "$BIN" --infer --model=1662 --symbol=gbpjpyrmp --probability-conditioned-stop-extension-evaluation=/tmp/lstm-phase18b/experiment596-model1662-gbpjpy-h8.csv 2025-01-01 2026-01-01
  "$BIN" --infer --model=1692 --symbol=audcadrmp --probability-conditioned-stop-extension-evaluation=/tmp/lstm-phase18b/experiment565-model1692-audcad-h14.csv 2025-01-01 2026-01-01
  "$BIN" --infer --model=1805 --symbol=eurusdrmp --probability-conditioned-stop-extension-evaluation=/tmp/lstm-phase18b/experiment624-model1805-eurusd-h4.csv 2025-01-01 2026-01-01

These commands are the empirical phase, not implementation tests.  Do not
change the cohort after observing Phase 18B outcomes.  The conclusion must be
based primarily on extension versus fixed stop, report profitability together
with drawdown, and remain inconclusive for sparse confidence buckets.
