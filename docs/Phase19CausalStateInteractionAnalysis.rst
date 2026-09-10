Phase 19 Causal State Interaction Analysis
==========================================

Scientific status
-----------------

Phase 19 is exploratory historical mechanism analysis.  It is neither an
independent validation phase nor a production strategy promotion phase.  An
observed interaction cannot be converted into an independently validated rule
by this analysis.

The primary comparison remains ``probability_conditioned_stop_extension_v1``
minus ``fixed_stop_loss_v1``.  Phase 18B is invoked unchanged with base stop
logarithmic distance ``0.001``, activation confidence ``0.50``, minimum
multiplier ``1.00``, and maximum multiplier ``1.25``.  Phase 19 exposes no CLI
parameters that can alter those values.

Controlled invocation
---------------------

::

  LSTM_Release \
    --infer \
    --model=<predeclared_model_id> \
    --symbol=<persisted_symbol_when_required> \
    --probability-stop-extension-state-analysis=<artifact_path> \
    <fromDate> <toDate>

Only the six primary models and the separate diagnostic model declared for
Phase 19 are accepted.  Only ``2025-01-01`` through ``2026-01-01`` and
``2026-01-01`` through ``2026-09-01`` are accepted, and the artifact labels
them ``discovery_history_2025`` and
``temporal_validation_history_2026`` respectively.  ``--infer-all``, scheduler
context, frozen-outcome context, training, and conflicting controlled
evaluation modes are rejected.

The PostgreSQL transaction is set read-only before model loading or evaluation.
After the artifact is written, the transaction is committed and normal
inference persistence and model save are skipped.  The artifact and CLI report
``production_rows_modified=false``.

Entry-state contract
--------------------

Phase 19 extracts these exact authoritative Tensor features from the completed
decision row used by inference:

* ``volatility_regime`` (column 38)
* ``rolling_range_expansion`` (column 46)
* ``directional_range`` (column 39)
* ``directional_efficiency`` (column 41)
* ``return_sign_persistence`` (column 42)
* ``return_direction_imbalance`` (column 43)
* ``historical_level_proximity`` (column 47)

The artifact records the exact runtime identities and causal timing semantics.
No replacement feature is synthesized.  State is carried in a dedicated type
that cannot represent strategy outcomes.  Directional return, stop/terminal
exit, holding duration, maximum adverse excursion, maximum favorable
excursion, and drawdown are recorded only as outcomes and are explicitly
labeled as prohibited gating inputs.

Stratification and reporting
----------------------------

All current predeclared features are continuous.  Quartiles are computed
separately for each model/window from activated entry-state values before any
strategy outcome is inspected.  Boundaries use the deterministic inverse
empirical CDF at floor-index quartiles.  A value equal to a boundary is always
assigned to the upper stratum, so tied values are never split by observation
order.  Exact boundaries are persisted.

Artifacts report all actionable trades, activated trades, the ``[0.50,0.75)``
confidence population, and the ``[0.75,1.00]`` population.  State interactions
are reported only within activated trades.  Sparse, limited, moderate, and
substantial labels use the predeclared 30/100/300 cutoffs and are descriptive
only.

Cross-model aggregation
-----------------------

After all fourteen model/window artifacts exist, run::

  Scripts/aggregate_phase19_state_analysis.py \
    --artifact-dir=/tmp/lstm-phase19-state-analysis \
    --output-prefix=/tmp/lstm-phase19-state-analysis/cross-model-summary

The aggregator fails closed unless the complete seven-model by two-window grid
is present.  Primary-model counts, positive/negative/zero model directions,
combined deltas, total support, acceptance-qualified secondary summaries, and
window sign agreement are reported separately.  The diagnostic model is
validated as present but is never pooled into the primary six-model result.
